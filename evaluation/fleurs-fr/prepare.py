"""Restore the two licensed evaluation WAVs, stopping at the second selected member."""

import hashlib
import json
from pathlib import Path
import tarfile
from urllib.request import urlopen

root = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
needed = {r["audio"]: r["sha256"] for r in rows}
for name, digest in list(needed.items()):
    local = root / name
    if local.is_file() and hashlib.sha256(local.read_bytes()).hexdigest() == digest:
        del needed[name]
if needed:
    url = ("https://huggingface.co/datasets/google/fleurs/resolve/"
           "70bb2e84b976b7e960aa89f1c648e09c59f894dd/data/fr_fr/audio/dev.tar.gz")
    with urlopen(url, timeout=60) as source, tarfile.open(fileobj=source, mode="r|gz") as archive:
        for member in archive:
            name = Path(member.name).name
            if member.isfile() and name in needed:
                audio = archive.extractfile(member).read()
                if hashlib.sha256(audio).hexdigest() != needed.pop(name):
                    raise ValueError(f"Wrong digest for {name}")
                (root / name).write_bytes(audio)
                if not needed:
                    break
if needed:
    raise SystemExit(f"Missing archive members: {sorted(needed)}")
print("Both FLEURS French validation recordings match the manifest.")
