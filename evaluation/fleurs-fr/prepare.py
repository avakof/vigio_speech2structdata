"""Restore the two licensed evaluation WAVs, stopping at the second selected member."""

import hashlib
import io
import json
from pathlib import Path
import tarfile
from urllib.request import Request, urlopen

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
    # Both selected archive members end in this verified prefix of the 142 MB archive.
    with urlopen(Request(url, headers={"Range": "bytes=0-1499999"}), timeout=90) as source:
        prefix = source.read(1500000)
    if hashlib.sha256(prefix).hexdigest() != "92dcf5f1e96b18735fe2342c6491d0fb589b1a76b13a752112a67fed8617ee40":
        raise ValueError("Incomplete or unexpected FLEURS archive prefix")
    with tarfile.open(fileobj=io.BytesIO(prefix), mode="r|gz") as archive:
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
