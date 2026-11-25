"""Private file-based IPC for a single ASR lifetime, including its upstream patches."""

import sys
from pathlib import Path

from .config import Settings


def main(request: Path, result: Path, mode: str, source: str):
    settings = Settings.model_validate_json(request.read_bytes())
    if mode == "mic":
        from .audio import capture_microphone

        audio = capture_microphone(settings, Path(source))
    else:
        from .audio import transcribe_file

        audio = transcribe_file(Path(source), settings)
    result.write_text(audio.model_dump_json(), encoding="utf-8")


if __name__ == "__main__":
    try:
        main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4])
    except Exception as exc:
        if isinstance(exc, ImportError):
            code = 3
        elif type(exc).__module__.startswith("speechflow.audio"):
            code = 4
        else:
            raise
        Path(sys.argv[2]).with_suffix(".error").write_text(str(exc), encoding="utf-8")
        sys.exit(code)
