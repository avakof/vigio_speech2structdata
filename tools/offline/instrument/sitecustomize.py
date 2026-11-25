"""Evidence-only Python startup hook for the ASR child in tools/offline/verify.py."""

import atexit
import json
import os
from pathlib import Path
import sys

if "speechflow.asr_process" in sys.orig_argv and "SPEECHFLOW_TEST_CONTROL" in os.environ:
    from probe import check

    path, external_ip, loopback = json.loads(os.environ["SPEECHFLOW_TEST_CONTROL"])
    record = check(external_ip, loopback)
    if "SPEECHFLOW_TEST_INPUT_FILE" in os.environ:
        try:
            from fixture_input import install

            fixture = install(Path(os.environ["SPEECHFLOW_TEST_INPUT_FILE"]))
            if fixture is not sys.modules.get("sounddevice") or not getattr(fixture, "__speechflow_fixture_input__", False):
                raise RuntimeError("Synthetic transport missing its identity marker")
            record["input_transport"] = "fixture callbacks, no physical PortAudio"

            def record_fixture_exit():
                record["fixture"] = fixture.fixture_metadata
                record["native_portaudio_module_loaded"] = "_sounddevice" in sys.modules
                Path(path).write_text(json.dumps(record) + "\n")

            atexit.register(record_fixture_exit)
        except Exception as error:
            print(f"Fixture setup failed: {error}", file=sys.stderr, flush=True)
            os._exit(86)
    Path(path).write_text(json.dumps(record) + "\n")
