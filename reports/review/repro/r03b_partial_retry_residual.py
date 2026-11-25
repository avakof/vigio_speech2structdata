"""Reviewer reproducer (after integrator fix): a complete retry whose result is legitimately
needs_review (one observation lacks a unit) still cannot replace a stored partial needs_review run
caused by a transient window timeout. Production persistence.save_run; hand-built manifests."""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, "tests/persistence_own")
from own_factory import make_manifest
from speechflow import persistence

tmp = Path(tempfile.mkdtemp(prefix="review-r03b-"))
db, out = tmp / "h.sqlite3", tmp / "out"
texts = ["Il reste deux virgule cinq kilos de tomates", "Douze bouteilles de lait"]
complete = make_manifest(texts=texts, status="needs_review")
obs = list(complete.observations)
obs[1] = obs[1].model_copy(update={"status": "needs_review", "unit": None, "unit_spoken": None, "errors": ["unit_missing"]})
complete = complete.model_copy(update={"observations": obs})
partial = complete.model_copy(update={"observations": obs[:1],
    "errors": ["window 1 failed: ['backend timeout: timeout: no reply within 120s']"]})
r1 = persistence.save_run(partial, db, out)
print("partial save:", r1.disposition, r1.manifest.status, len(r1.manifest.observations), "observations")
try:
    r2 = persistence.save_run(complete, db, out)
    print("complete retry:", r2.disposition, r2.manifest.status, len(r2.manifest.observations))
except persistence.PersistenceError as exc:
    print("complete retry FAILED:", type(exc).__name__, str(exc)[:160])
print("stored:", [(r["status"], r["observations"]) for r in persistence.list_runs(db)])
