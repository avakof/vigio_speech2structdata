"""Evidence-only Python startup hook for the ASR child in tools/offline/verify.py."""

import json
import os
from pathlib import Path
import sys

if "speechflow.asr_process" in sys.orig_argv and "SPEECHFLOW_TEST_CONTROL" in os.environ:
    from probe import check

    path, external_ip, loopback = json.loads(os.environ["SPEECHFLOW_TEST_CONTROL"])
    record = check(external_ip, loopback)
    Path(path).write_text(json.dumps(record) + "\n")
