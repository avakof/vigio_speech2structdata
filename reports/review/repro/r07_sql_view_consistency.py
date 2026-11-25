"""Reviewer check (no model): open each committed SQLite DB read-only and compare the current save's
manifest_json, normalized rows, SQL views, run.json and observations.csv byte-for-byte/field-for-field."""
import csv, io, json, sqlite3, sys
from pathlib import Path
from speechflow.contracts import RunManifest
from speechflow import persistence

def check(report):
    d = json.loads(Path(report).read_text())
    cmd = d["app_command"]
    out = Path(cmd[cmd.index("--out") + 1]); db = Path(cmd[cmd.index("--database") + 1])
    print(f"== {report}: db={db.exists()} out={out.exists()} result={d['result']}")
    if not db.exists():
        return False
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True); con.row_factory = sqlite3.Row
    ok = True
    runs = con.execute("select * from current_runs").fetchall()
    print("  runs:", [(r["run_id"][:16], r["status"], r["generation"]) for r in runs], "integrity:", con.execute("pragma integrity_check").fetchone()[0])
    for r in runs:
        m = RunManifest.model_validate_json(con.execute("select manifest_json from run_saves where save_id=?", (r["save_id"],)).fetchone()[0])
        ok &= r["content_digest"] == persistence.content_digest(m)
        segs = con.execute("select segment_id, normalized_text, raw_json, status from current_segments where run_id=? order by ordinal", (r["run_id"],)).fetchall()
        ok &= [(s.segment_id, s.normalized_text, s.status) for s in m.segments] == [(s[0], s[1], s[3]) for s in segs]
        ok &= all(json.loads(s[2]) == ms.raw for s, ms in zip(segs, m.segments))
        obs = con.execute("select observation_id, quantity, unit, status, correction_of, is_final from current_observations where run_id=? order by ordinal", (r["run_id"],)).fetchall()
        ok &= [(o.observation_id, o.quantity, o.unit, o.status, o.correction_of) for o in m.observations] == [tuple(x)[:5] for x in obs]
        ev = con.execute("select e.segment_id, e.span_start, e.span_end, e.quote, g.normalized_text from observation_evidence e join segments g on g.save_id=e.save_id and g.segment_id=e.segment_id where e.save_id=?", (r["save_id"],)).fetchall()
        ok &= all(t[4][t[1]:t[2]] == t[3] for t in ev)
        runjson = out / "run.json"
        if runjson.exists():
            ok &= RunManifest.model_validate_json(runjson.read_text()) == m
            rows = list(csv.DictReader(io.StringIO((out / "observations.csv").read_text())))
            ok &= [(x["observation_id"], x["quantity"], x["unit"], x["status"], x["is_final"]) for x in rows] == \
                  [(o[0], o[1] or "", o[2] or "", o[3], "true" if o[5] else "false") for o in obs]
        print("  observations:", [(o["quantity"], o["unit"], o["status"]) for o in obs], "evidence rows:", len(ev),
              "raw keys:", [list(json.loads(s[2]).keys()) for s in segs])
    print("  CONSISTENT" if ok else "  MISMATCH")
    return ok

results = [check(p) for p in sys.argv[1:]]
sys.exit(0 if all(results) else 1)
