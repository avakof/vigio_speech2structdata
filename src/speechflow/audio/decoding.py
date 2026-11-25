"""Decode a bounded local file into a temporary 16 kHz mono float32 spool."""

from collections import deque
from hashlib import file_digest
from pathlib import Path
import re
import shutil
import subprocess
from threading import Thread

from ..contracts import AudioSource, stable_id
from .hojo import SAMPLE_RATE

SAMPLE_BYTES = 4  # Little-endian float32 PCM.
BLOCK_BYTES = 64 * 1024
FILE_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a"}
FIRST_AUDIO = re.compile(r"^\s*Stream #0:\d+.*?: Audio: .+?, (\d+) Hz, ([^,]+)")
FFMPEG_VERSION = re.compile(r"^ffmpeg version (\S+)")
NAMED_LAYOUTS = {
    "mono": 1,
    "stereo": 2,
    "downmix": 2,
    "quad": 4,
    "quad(side)": 4,
    "hexagonal": 6,
    "octagonal": 8,
    "cube": 8,
    "hexadecagonal": 16,
}


class AudioDecodeError(RuntimeError):
    """The selected local audio stream could not be safely decoded."""


def channel_count(layout: str) -> int:
    if layout in NAMED_LAYOUTS:
        return NAMED_LAYOUTS[layout]
    if match := re.fullmatch(r"(\d+) channels?(?: \(.*\))?", layout):
        return int(match[1])
    if match := re.fullmatch(r"(\d+)\.(\d+)(?:\.(\d+))?(?:\(.*\))?", layout):
        return sum(int(part) for part in match.groups() if part is not None)
    raise AudioDecodeError(f"Cannot identify the channel count of FFmpeg layout {layout!r}")


def ffmpeg_executable() -> str:
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ModuleNotFoundError as error:
        raise ImportError("Install the pinned imageio-ffmpeg package with uv sync --frozen") from error
    try:
        return get_ffmpeg_exe()
    except RuntimeError as error:
        raise AudioDecodeError(
            "Set IMAGEIO_FFMPEG_EXE to an installed local FFmpeg binary"
        ) from error


def _identity(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def decode_file(
    path: Path, pcm: Path, max_spool_bytes: int, *, metrics: dict | None = None
) -> AudioSource:
    if path.suffix.lower() not in FILE_SUFFIXES:
        raise AudioDecodeError(f"Unsupported input {path.name!r}. Use WAV, FLAC, MP3, or M4A.")
    if not path.is_file():
        raise AudioDecodeError(f"Audio file not found or not a regular file at {path}")
    identity = _identity(path)
    if identity[2] == 0:
        raise AudioDecodeError(f"Audio file {path.name!r} is empty")
    with path.open("rb") as source:
        digest = file_digest(source, "sha256").hexdigest()

    configured_executable = ffmpeg_executable()
    resolved_executable = shutil.which(configured_executable)
    if not resolved_executable:
        raise AudioDecodeError(f"FFmpeg executable not found at {configured_executable!r}")
    executable = Path(resolved_executable).resolve(strict=True)
    executable_identity = _identity(executable)
    with executable.open("rb") as binary:
        executable_digest = file_digest(binary, "sha256").hexdigest()

    command = [
        str(executable), "-nostdin", "-nostats", "-loglevel", "info",
        "-xerror", "-protocol_whitelist", "file,pipe", "-format_whitelist", "wav,flac,mp3,mov",
        "-i", str(path.resolve()),
        "-map", "0:a:0", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1",
    ]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as error:
        raise AudioDecodeError(f"Could not start the local FFmpeg executable: {error}") from error

    diagnostics = deque(maxlen=20)
    native: list[str] = []
    decoder_version = None

    def drain_stderr():
        nonlocal decoder_version
        for raw in process.stderr:
            line = raw.decode(errors="replace").strip("\r\n")
            if decoder_version is None and (match := FFMPEG_VERSION.match(line)):
                decoder_version = match[1]
            if not native and (match := FIRST_AUDIO.match(line)):
                native.extend(match.groups())
            diagnostics.append(line[:1000])

    # Drain stderr while copying stdout so neither pipe can deadlock.
    drain = Thread(target=drain_stderr, name="ffmpeg-stderr")
    drain.start()
    written = 0
    try:
        with pcm.open("wb") as output:
            while chunk := process.stdout.read(BLOCK_BYTES):
                written += len(chunk)
                if written > max_spool_bytes:
                    raise AudioDecodeError(
                        f"Normalized audio exceeds max_spool_bytes ({max_spool_bytes}); "
                        "raise the limit if disk space permits, or split the source file"
                    )
                output.write(chunk)
        exit_code = process.wait()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        drain.join()
        process.stdout.close()
        process.stderr.close()

    if exit_code:
        detail = "\n".join(diagnostics)
        raise AudioDecodeError(f"FFmpeg could not decode {path.name!r}:\n{detail}")
    if not native or written % SAMPLE_BYTES:
        raise AudioDecodeError(f"FFmpeg did not produce a complete audio stream for {path.name!r}")
    if decoder_version is None:
        raise AudioDecodeError(f"Could not identify the version reported by FFmpeg at {executable}")
    if _identity(executable) != executable_identity:
        raise AudioDecodeError(f"FFmpeg executable at {executable} changed during decoding")
    if _identity(path) != identity:
        raise AudioDecodeError(f"Audio file {path.name!r} changed during processing")

    rate, layout = native
    channels = channel_count(layout)
    if metrics is not None:
        metrics["decoder"] = {"sha256": executable_digest, "version": decoder_version}
    return AudioSource(
        source_id=stable_id("file", digest),
        kind="file",
        digest=digest,
        name=path.name,
        native_rate=int(rate),
        native_channels=channels,
        sample_count=written // SAMPLE_BYTES,
        channel_mix="mono" if channels == 1 else f"ffmpeg_downmix_to_mono({layout})",
    )
