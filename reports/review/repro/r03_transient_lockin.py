"""Reviewer reproducer: a transient extractor failure in ONE window yields status needs_review;
the successful retry under the same processing version can then never be stored (ConflictError).
Uses production persistence.save_run only; manifests are hand-built (no model)."""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, "tests/persistence_own")
from own_factory import make_manifest
from speechflow import persistence

tmp = Path(tempfile.mkdtemp(prefix="review-r03-"))
db, out = tmp / "h.sqlite3", tmp / "out"
texts = ["Il reste deux virgule cinq kilos de tomates", "Douze bouteilles de lait"]
first = make_manifest(texts=texts, status="needs_review")
# first attempt: window 1 timed out -> only the first observation, run status needs_review
first = first.model_copy(update={"observations": first.observations[:1],
    "errors": ["window 1 (segments [...]) failed (attempts used: 1): ['backend timeout: timeout: no reply within 120s']"]})
retry = make_manifest(texts=texts, status="ok")
assert first.run_id == retry.run_id and first.processing_version == retry.processing_version
r1 = persistence.save_run(first, db, out)
print("first save:", r1.disposition, r1.manifest.status, len(r1.manifest.observations), "observations")
try:
    r2 = persistence.save_run(retry, db, out)
    print("retry save:", r2.disposition)
except persistence.PersistenceError as exc:
    print("retry save FAILED:", type(exc).__name__, str(exc)[:300])
print("stored:", persistence.list_runs(db)[0]["status"], "recovery files:", sorted(p.name[:20] for p in (out / ".recovery").glob("*.json")))
