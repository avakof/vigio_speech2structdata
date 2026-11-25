"""Reviewer check (no model): which committed implementation produced each runtime evidence manifest?
Recomputes pipeline._provenance's implementation_sha256 over the current src/speechflow/**/*.py and
compares it with provenance.implementation_sha256 stored in each evidence manifest / run.json / DB."""
import hashlib, json, re, sqlite3, sys
from pathlib import Path
root = Path("src/speechflow")
h = hashlib.sha256()
for p in sorted(root.rglob("*.py")):
    h.update(p.relative_to(root).as_posix().encode() + b"\0" + p.read_bytes())
current = h.hexdigest()
print("current implementation_sha256:", current)
found = {}
def walk(obj, where):
    if isinstance(obj, dict):
        v = obj.get("implementation_sha256")
        if isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v):
            found.setdefault(v, set()).add(where)
        for x in obj.values(): walk(x, where)
    elif isinstance(obj, list):
        for x in obj: walk(x, where)
for p in sorted(Path("reports").rglob("*.json")) + sorted(Path("reports").rglob("*.jsonl")):
    if "review" in p.parts: continue
    try:
        text = p.read_text()
        docs = [json.loads(l) for l in text.splitlines() if l.strip()] if p.suffix == ".jsonl" else [json.loads(text)]
    except Exception:
        continue
    for d in docs: walk(d, str(p))
for v, where in sorted(found.items(), key=lambda kv: -len(kv[1])):
    tag = "CURRENT" if v == current else "older"
    print(f"{v[:12]} {tag:7} {len(where):3} files: " + ", ".join(sorted(where))[:900])
