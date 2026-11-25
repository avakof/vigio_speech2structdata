"""Coordinate one local ASR session and one ordered extraction pass."""

import hashlib
import importlib.metadata as md
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal

from . import __version__
from .assets import DIGEST, HOJO, extractor_digest, local_request, silero_hash
from .config import Settings
from .contracts import AudioResult, ExtractionResult, RunManifest, SCHEMA_VERSION, stable_id


def _audio(path, mode, settings, out):
    with tempfile.TemporaryDirectory(prefix="speechflow-ipc-") as directory:
        work = Path(directory)
        request, response = work / "settings.json", work / "result.json"
        request.write_text(settings.model_dump_json())
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"}
        cmd = [sys.executable, "-m", "speechflow.asr_process", str(request),
               str(response), mode, str(out if mode == "mic" else path)]
        # The child receives the first interrupt and can finalize its capture.
        with subprocess.Popen(cmd, env=env, start_new_session=True) as child:
            try:
                code = child.wait()
            except KeyboardInterrupt:
                child.send_signal(signal.SIGINT)
                try:
                    code = child.wait()
                except KeyboardInterrupt:
                    child.kill()
                    raise
        if code or not response.exists():
            error_path = response.with_suffix(".error")
            if error_path.exists():
                message = error_path.read_text()
                if code == 3:
                    raise ImportError(message)
                if code == 4:
                    from .audio.decoding import AudioDecodeError

                    raise AudioDecodeError(message)
            raise RuntimeError(f"Audio process exited with code {code}. See its diagnostic above.")
        return AudioResult.model_validate_json(response.read_bytes())


def _validate_timeline(audio, settings):
    ids = set()
    previous_end = 0
    for ordinal, segment in enumerate(audio.segments):
        if (segment.segment_id in ids or segment.source_id != audio.source.source_id
                or segment.ordinal != ordinal or segment.core_start < previous_end
                or segment.input_end > audio.source.sample_count
                or segment.input_end - segment.input_start > int(16000 * settings.max_chunk_seconds)):
            raise ValueError(f"Inconsistent audio segment at ordinal {ordinal}")
        ids.add(segment.segment_id)
        previous_end = segment.core_end
    if audio.status == "no_speech" and audio.segments:
        raise ValueError("Silence result contains speech segments")
    if audio.status == "ok" and (not audio.segments or any(s.status == "error" for s in audio.segments)):
        raise ValueError("Successful audio result has missing or failed segments")


def _provenance(audio, settings, mode):
    root = Path(__file__).parent
    code_hash = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        code_hash.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    excluded = {"hojo_dir", "ollama_url", "database", "offline"}
    if mode != "mic":
        excluded.update({"mic_device", "mic_seconds", "mic_queue_seconds"})
    packages = ("torch", "torchaudio", "transformers", "hojo-asr", "silero-vad",
                "numpy", "soundfile", "imageio-ffmpeg", "pydantic")
    installed = {}
    for name in packages:
        try:
            installed[name] = md.version(name)
        except md.PackageNotFoundError:
            installed[name] = None
    try:
        vad_digest = silero_hash()
    except (OSError, RuntimeError, md.PackageNotFoundError):
        if audio.status != "error":
            raise
        vad_digest = None
    return {"speechflow_version": __version__, "implementation_sha256": code_hash.hexdigest(),
            "schema_version": SCHEMA_VERSION, "config": settings.model_dump(mode="json", exclude=excluded),
            "silero_sha256": vad_digest, "hojo_revision": HOJO["revision"],
            "expected_extractor_digest": DIGEST if mode != "transcribe" else None,
            "backend": audio.backend, "dtype": audio.dtype, "packages": installed}


def process(path: Path | None, *, mode: Literal["run", "transcribe", "mic"],
            settings: Settings, out: Path) -> RunManifest:
    start = time.perf_counter()
    if mode != "mic" and (path is None or not path.is_file()):
        raise ValueError(f"Audio file does not exist: {path}")
    out.mkdir(parents=True, exist_ok=True)
    audio = _audio(path, mode, settings, out)
    _validate_timeline(audio, settings)
    provenance = _provenance(audio, settings, mode)
    if mode != "transcribe" and audio.status in ("ok", "needs_review"):
        try:
            provenance["ollama_version"] = local_request(settings, "/api/version")["version"]
        except (OSError, ValueError, KeyError):
            provenance["ollama_version"] = None
    version = stable_id("processing", provenance)
    run_id = stable_id("run", audio.source.source_id, version, mode)
    extraction = ExtractionResult(observations=[], status="no_speech")
    stage = "not_run"
    status = audio.status
    if mode != "transcribe" and audio.status == "no_speech":
        stage = "no_speech"
    elif mode != "transcribe" and audio.status in ("ok", "needs_review"):
        from .extraction import extract

        try:
            digest = extractor_digest(settings)
            provenance["extractor_digest"] = digest
            if digest != DIGEST:
                raise RuntimeError(f"Pinned local extractor is unavailable. Run speechflow models prepare. Found {digest}")
            extraction = extract(audio.segments, run_id=run_id, settings=settings)
        except (RuntimeError, OSError, ValueError, KeyError) as exc:
            extraction = ExtractionResult(observations=[], status="error", errors=[str(exc)])
        stage = extraction.status
        status = "needs_review" if audio.status == "needs_review" and stage == "ok" else stage
    elapsed = time.perf_counter() - start
    duration = audio.source.sample_count / 16000
    return RunManifest(run_id=run_id, processing_version=version, mode=mode, status=status,
                       transcription_status=audio.status, extraction_status=stage, source=audio.source,
                       segments=audio.segments, observations=extraction.observations, provenance=provenance,
                       metrics={"audio": audio.metrics, "extraction": extraction.metrics,
                                "duration_s": duration, "pipeline_seconds": elapsed,
                                "pipeline_rtf": elapsed / duration if duration else None},
                       errors=audio.errors + extraction.errors, attempts=extraction.attempts)
