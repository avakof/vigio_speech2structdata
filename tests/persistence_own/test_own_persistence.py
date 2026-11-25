"""SQLite history: replay, identity, immutability, transactions, exports, concurrency."""

from __future__ import annotations

import csv
import io
import json
import multiprocessing
import os
import sqlite3
import subprocess
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest
from own_factory import (
    FLOUR, export_in_worker, make_manifest, make_obs, packaged, save_in_worker,
)

from speechflow import persistence as P
from speechflow.contracts import RunManifest


@pytest.fixture
def db(tmp_path):
    return tmp_path / "history.sqlite3"


def count(db, table, where=""):
    with sqlite3.connect(db) as conn:
        return conn.execute(f"SELECT count(*) FROM {table} {where}").fetchone()[0]


# --------------------------------------------------------------- migrations

def test_migrations_idempotent_checksummed_and_versioned(db):
    P.connect(db).close()
    conn = P.connect(db)
    assert conn.execute("SELECT version FROM schema_migrations").fetchall()[0][0] == 1
    conn.close()
    assert oct(db.stat().st_mode & 0o777) == "0o600"
    with sqlite3.connect(db) as raw:
        raw.execute("UPDATE schema_migrations SET checksum='tampered'")
    with pytest.raises(P.PersistenceError, match="modified after it was applied"):
        P.connect(db)
    newer = db.with_name("newer.sqlite3")
    P.connect(newer).close()
    with sqlite3.connect(newer) as raw:
        raw.execute("INSERT INTO schema_migrations VALUES (99, 'future', 'x', 'now')")
    with pytest.raises(P.PersistenceError, match="newer than this speechflow"):
        P.connect(newer)


def test_concurrent_first_open_migrates_once(db):
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(6, mp_context=ctx) as pool:
        list(pool.map(export_missing, [str(db)] * 6))
    assert count(db, "schema_migrations") == 1


def export_missing(path):
    from speechflow import persistence
    persistence.connect(Path(path)).close()


# ------------------------------------------------------------ save and read

@pytest.mark.parametrize("status", ["ok", "no_speech", "needs_review", "error"])
def test_every_review_status_is_saved_and_round_trips(db, tmp_path, status):
    m = make_manifest(status=status, texts=[] if status == "no_speech" else None,
                      observations=[] if status in ("no_speech", "error") else "default",
                      errors=["boom"] if status == "error" else [])
    result = P.save_run(m, db, tmp_path / "out")
    assert result.disposition == "inserted"
    stored = P.load_manifest(db, m.run_id)
    assert stored == m
    assert [r["status"] for r in P.list_runs(db)] == [status]
    assert P.verify(db) == []
    assert (tmp_path / "out/run.json").exists()


def test_frozen_save_signature_commits_and_exports(db, tmp_path):
    m = make_manifest()
    assert P.save(m, db, tmp_path / "o") is None
    assert {p.name for p in (tmp_path / "o").iterdir()} == {
        "run.json", "observations.csv", "segments.csv", "observations.txt", "segments.txt"}


def test_replay_after_restart_is_idempotent(db, tmp_path):
    m = make_manifest(texts=["un", "deux"])
    first = P.save_run(m, db, tmp_path / "o1")
    snapshot = [p.read_bytes() for p in first.files]
    code = textwrap.dedent("""
        import sys
        from pathlib import Path
        from speechflow import persistence as P
        from speechflow.contracts import RunManifest
        m = RunManifest.model_validate_json(Path(sys.argv[1]).read_text())
        r = P.save_run(m, Path(sys.argv[2]), Path(sys.argv[3]))
        print(r.disposition)
    """)
    src = tmp_path / "m.json"
    src.write_text(m.model_dump_json())
    out = subprocess.run([sys.executable, "-c", code, str(src), str(db), str(tmp_path / "o2")],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "duplicate"
    assert [p.read_bytes() for p in sorted((tmp_path / "o2").glob("*")) if p.is_file()] == [
        (tmp_path / "o1" / p.name).read_bytes() for p in sorted((tmp_path / "o2").glob("*"))
        if p.is_file()]
    assert count(db, "runs") == count(db, "run_saves") == 1
    assert count(db, "segments") == 2 and count(db, "observations") == 2
    assert count(db, "save_events") == 2  # inserted + duplicate are both journaled
    assert snapshot == [p.read_bytes() for p in first.files]


def test_retry_with_other_timings_and_raw_is_a_duplicate(db, tmp_path):
    first = make_manifest(metrics={"asr_seconds": 1.0})
    again = make_manifest(metrics={"asr_seconds": 9.0})
    again.segments[0].raw["latency"] = 123
    P.save_run(first, db, tmp_path / "o")
    result = P.save_run(again, db, tmp_path / "o")
    assert result.disposition == "duplicate"
    assert result.manifest.metrics == {"asr_seconds": 1.0}  # the stored result is authoritative


def test_distinct_identical_utterances_stay_distinct(db, tmp_path):
    m = make_manifest(texts=["Deux kilos de farine", "Deux kilos de farine"],
                      observations=[])
    obs = [make_obs(s, "x", quote="Deux", product="farine", quantity="2") for s in m.segments]
    m = m.model_copy(update={"observations": obs})
    assert obs[0].observation_id != obs[1].observation_id
    P.save_run(m, db, tmp_path / "o")
    assert count(db, "segments") == 2 and count(db, "observations") == 2
    assert count(db, "segments", "WHERE normalized_text='Deux kilos de farine'") == 2


def test_duplicate_ids_inside_one_manifest_are_rejected_and_recoverable(db, tmp_path):
    m = make_manifest(texts=["un deux"])
    twin = m.observations[0].model_copy()
    bad = m.model_copy(update={"observations": [m.observations[0], twin]})
    with pytest.raises(P.IntegrityError, match="duplicate observation_id"):
        P.save_run(bad, db, tmp_path / "o")
    assert not db.exists() or count(db, "runs") == 0
    assert len(list((tmp_path / "o" / ".recovery").glob("*.json"))) == 1


def test_same_source_two_modes_are_separate_runs(db, tmp_path):
    t = make_manifest(mode="transcribe")
    r = make_manifest(mode="run")
    P.save_run(t, db, tmp_path / "t")
    P.save_run(r, db, tmp_path / "r")
    assert t.source == r.source and t.run_id != r.run_id
    assert count(db, "sources") == 1 and count(db, "runs") == 2
    assert count(db, "segments") == 2 and count(db, "observations") == 1
    assert P.load_manifest(db, t.run_id).observations == []
    assert P.load_manifest(db, t.run_id[:12]) == t


def test_changed_processing_version_is_a_new_run_and_keeps_history(db, tmp_path):
    P.save_run(make_manifest(pv="pv-1"), db, tmp_path / "o")
    P.save_run(make_manifest(pv="pv-2"), db, tmp_path / "o")
    assert count(db, "runs") == 2 and count(db, "run_saves") == 2
    assert count(db, "observation_history") == 2


def test_same_run_different_result_is_refused_unless_previous_failed(db, tmp_path):
    ok = make_manifest()
    P.save_run(ok, db, tmp_path / "o")
    other = ok.model_copy(update={"observations": []})
    with pytest.raises(P.ConflictError, match="new processing version"):
        P.save_run(other, db, tmp_path / "o")
    assert P.load_manifest(db, ok.run_id) == ok
    assert len(list((tmp_path / "o/.recovery").glob("*.json"))) == 1

    failed = make_manifest(seed="b", status="error", observations=[], errors=["ollama down"])
    P.save_run(failed, db, tmp_path / "p")
    fixed = make_manifest(seed="b")
    result = P.save_run(fixed, db, tmp_path / "p")
    assert (result.disposition, result.generation) == ("replaced", 2)
    assert P.load_manifest(db, fixed.run_id).status == "ok"
    assert count(db, "run_saves", f"WHERE run_id='{fixed.run_id}'") == 2


def test_inconsistent_identity_is_a_conflict(db, tmp_path):
    m = make_manifest()
    P.save_run(m, db, tmp_path / "o")
    renamed = m.model_copy(update={"run_id": "run_other"})
    with pytest.raises(P.ConflictError, match="already belongs to run"):
        P.save_run(renamed, db, tmp_path / "o")
    tampered = m.model_copy(update={"source": m.source.model_copy(update={"digest": "0" * 64})})
    with pytest.raises(P.ConflictError, match="different kind or content digest"):
        P.save_run(tampered, db, tmp_path / "o")


# -------------------------------------------------------------- corrections

def correction_manifest(*, retry_target=False):
    m = make_manifest(texts=[FLOUR, "Non en fait deux kilos de farine"], observations=[])
    s0, s1 = m.segments
    first = make_obs(s0, "first", quote="Trois", product="farine", quantity="3")
    fix = make_obs(s1, "fix", quote="deux", product="farine", quantity="2",
                   operation="correction", correction_of=first.observation_id, revision=2)
    return m.model_copy(update={"observations": [first, fix]}), first, fix


def test_correction_history_is_preserved_and_applied_once(db, tmp_path):
    m, first, fix = correction_manifest()
    P.save_run(m, db, tmp_path / "o")
    P.save_run(m, db, tmp_path / "o")  # retry
    assert count(db, "observations") == 2
    rows = {r["observation_id"]: r for r in P.observation_rows(m)}
    assert rows[first.observation_id]["is_final"] == "false"
    assert rows[first.observation_id]["corrected_by"] == fix.observation_id
    assert rows[fix.observation_id]["is_final"] == "true"
    assert rows[first.observation_id]["quantity"] == "3"  # superseded value kept
    with sqlite3.connect(db) as conn:
        sql = {r[0]: (r[1], r[2]) for r in conn.execute(
            "SELECT observation_id, is_final, corrected_by FROM current_observations")}
    assert sql == {k: (int(v["is_final"] == "true"), v["corrected_by"]) for k, v in rows.items()}


def test_unresolved_correction_must_be_reviewable(db, tmp_path):
    m, first, fix = correction_manifest()
    dangling = fix.model_copy(update={"correction_of": "observation_missing"})
    with pytest.raises(P.IntegrityError, match="target is missing but status is ok"):
        P.save_run(m.model_copy(update={"observations": [first, dangling]}), db, tmp_path / "o")
    review = dangling.model_copy(update={"status": "needs_review",
                                         "errors": ["ambiguous correction target"]})
    ok = m.model_copy(update={"observations": [first, review]})
    P.save_run(ok, db, tmp_path / "o")
    rows = P.observation_rows(ok)
    assert rows[0]["is_final"] == "true" and rows[1]["errors"] == "ambiguous correction target"
    cyc_a = first.model_copy(update={"correction_of": fix.observation_id, "operation": "correction"})
    with pytest.raises(P.IntegrityError, match="cyclic"):
        P.check_manifest(m.model_copy(update={"observations": [cyc_a, fix]}))


def test_evidence_must_quote_the_segment(db, tmp_path):
    m = make_manifest()
    bad = m.observations[0].model_copy(deep=True)
    bad.evidence[0] = bad.evidence[0].model_copy(update={"quote": "tomates"})
    with pytest.raises(P.IntegrityError, match="evidence quote does not match"):
        P.check_manifest(m.model_copy(update={"observations": [bad]}))
    worse = m.observations[0].model_copy(update={"quantity": "2,5"})
    with pytest.raises(P.IntegrityError, match="not a base-ten decimal"):
        P.check_manifest(m.model_copy(update={"observations": [worse]}))
    nr = m.model_copy(update={"extraction_status": "not_run"})
    with pytest.raises(P.IntegrityError, match="not_run"):
        P.check_manifest(nr)


def test_packaging_total_and_unknown_quantity_survive(db, tmp_path):
    m = make_manifest(texts=["Deux cartons de douze bouteilles"], observations=[])
    m = m.model_copy(update={"observations": [packaged(m.segments[0])]})
    P.save_run(m, db, tmp_path / "o")
    row = P.observation_rows(P.load_manifest(db, m.run_id))[0]
    assert (row["quantity"], row["qualifier"]) == (None, "unknown")  # unknown is not zero
    assert (row["packaging_count"], row["per_package_quantity"], row["packaging_total"]) == (
        "2", "12", "24")
    with sqlite3.connect(db) as conn:
        pack = json.loads(conn.execute("SELECT packaging_json FROM observations").fetchone()[0])
    assert pack["total"] == "24"


# ------------------------------------------------------------- immutability

def test_history_cannot_be_updated_or_deleted(db, tmp_path):
    m, first, _ = correction_manifest()
    other = make_manifest(seed="z")
    P.save_run(m, db, tmp_path / "o")
    P.save_run(other, db, tmp_path / "o")
    conn = P.connect(db)
    for sql in (
        "UPDATE observations SET quantity='9'", "DELETE FROM observations",
        "UPDATE segments SET normalized_text='x'", "DELETE FROM segments",
        "UPDATE run_saves SET status='ok'", "DELETE FROM run_saves",
        "UPDATE sources SET name='x'", "DELETE FROM sources", "DELETE FROM runs",
        "UPDATE observation_evidence SET quote='x'", "DELETE FROM save_events",
        "UPDATE runs SET mode='mic'", "UPDATE runs SET current_save_id=NULL",
    ):
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(sql)
    foreign = conn.execute("SELECT current_save_id FROM runs WHERE run_id=?",
                           (other.run_id,)).fetchone()[0]
    with pytest.raises(sqlite3.IntegrityError, match="same run"):
        conn.execute("UPDATE runs SET current_save_id=? WHERE run_id=?", (foreign, m.run_id))
    conn.close()
    assert P.verify(db) == []


def test_uniqueness_constraints(db, tmp_path):
    P.save_run(make_manifest(), db, tmp_path / "o")
    conn = P.connect(db)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO runs (run_id, source_id, mode, processing_version, "
                     "schema_version, created_at) SELECT 'dup', source_id, mode, "
                     "processing_version, schema_version, created_at FROM runs")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO observations SELECT * FROM observations")
    conn.close()


# ------------------------------------------------- failure and recoverability

def test_failed_transaction_rolls_back_completely_and_is_recoverable(db, tmp_path, monkeypatch):
    m = make_manifest(texts=["un", "deux"])
    real = P._insert_children

    def partial(conn, save_id, manifest):
        conn.execute("INSERT INTO segments VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                     (save_id, "partial", manifest.source.source_id, 0, 0, 1, 2, 3, "{}", "t",
                      "ok", "[]"))
        raise RuntimeError("disk full")

    monkeypatch.setattr(P, "_insert_children", partial)
    with pytest.raises(RuntimeError, match="disk full"):
        P.save_run(m, db, tmp_path / "o")
    for table in ("sources", "runs", "run_saves", "segments", "observations", "save_events"):
        assert count(db, table) == 0, table
    assert not (tmp_path / "o" / "run.json").exists()
    assert len(list((tmp_path / "o/.recovery").glob("*.json"))) == 1

    monkeypatch.setattr(P, "_insert_children", real)
    results = P.recover(db, tmp_path / "o")
    assert [r[1] for r in results] == ["inserted"]
    assert not list((tmp_path / "o/.recovery").glob("*.json"))
    assert P.load_manifest(db, m.run_id) == m and (tmp_path / "o/run.json").exists()
    assert P.recover(db, tmp_path / "o") == []


def test_killed_process_mid_transaction_leaves_database_intact(db, tmp_path):
    m = make_manifest(texts=["un", "deux"])
    src = tmp_path / "m.json"
    src.write_text(m.model_dump_json())
    code = textwrap.dedent("""
        import os, sys
        from pathlib import Path
        from speechflow import persistence as P
        from speechflow.contracts import RunManifest
        real = P._insert_children
        def die(conn, save_id, manifest):
            real(conn, save_id, manifest)   # everything written, not yet committed
            os._exit(9)
        P._insert_children = die
        P.save_run(RunManifest.model_validate_json(Path(sys.argv[1]).read_text()),
                   Path(sys.argv[2]), Path(sys.argv[3]))
    """)
    proc = subprocess.run([sys.executable, "-c", code, str(src), str(db), str(tmp_path / "o")])
    assert proc.returncode == 9
    assert count(db, "runs") == 0 and count(db, "observations") == 0
    assert P.verify(db) == []
    results = P.recover(db, tmp_path / "o")
    assert [r[1] for r in results] == ["inserted"]
    assert P.load_manifest(db, m.run_id) == m and P.verify(db) == []


def test_export_failure_after_commit_is_recoverable(db, tmp_path, monkeypatch):
    m = make_manifest(texts=["un", "deux"])
    out = tmp_path / "o"
    calls = {"n": 0}
    real = os.replace

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("no space left")
        return real(a, b)

    monkeypatch.setattr(P.os, "replace", flaky)
    # the recovery write itself uses os.replace once; the third call is mid-export
    with pytest.raises(P.PersistenceError, match="run is committed"):
        P.save_run(m, db, out)
    assert count(db, "runs") == 1
    assert len(list((out / ".recovery").glob("*.json"))) == 1
    assert not any(p.name.startswith(".export-") for p in out.iterdir())
    monkeypatch.setattr(P.os, "replace", real)

    files = P.export_run(db, m.run_id[:10], "csv", out=out) + P.export_run(
        db, m.run_id, "table", out=out) + P.export_run(db, m.run_id, "json", out=out)
    assert {f.name for f in files} == {"observations.csv", "segments.csv", "observations.txt",
                                       "segments.txt", "run.json"}
    assert json.loads((out / "run.json").read_text())["run_id"] == m.run_id
    assert [r[1] for r in P.recover(db, out)] == ["duplicate"]


def test_unwritable_out_fails_before_touching_the_database(db, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(P.PersistenceError, match="recovery"):
        P.save_run(make_manifest(), db, blocker / "sub")
    assert not db.exists()


def test_locked_database_reports_retry(db, tmp_path, monkeypatch):
    P.connect(db).close()
    monkeypatch.setattr(P, "BUSY_TIMEOUT_MS", 100)
    holder = sqlite3.connect(db, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(P.PersistenceError, match="locked"):
            P.save_run(make_manifest(), db, tmp_path / "o")
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert P.save_run(make_manifest(), db, tmp_path / "o").disposition == "inserted"


# ---------------------------------------------------------------- concurrency

def test_concurrent_identical_saves_and_exports_are_consistent(db, tmp_path):
    m = make_manifest(texts=["un", "deux", "trois"])
    ctx = multiprocessing.get_context("spawn")
    jobs = [(m.model_dump_json(), str(db), str(tmp_path / f"out{i}")) for i in range(8)]
    P.connect(db).close()
    with ProcessPoolExecutor(8, mp_context=ctx) as pool:
        saves = list(pool.map(save_in_worker, jobs))
    assert sorted(d for d, _ in saves).count("inserted") == 1
    assert sorted(d for d, _ in saves).count("duplicate") == 7
    assert all(files == saves[0][1] for _, files in saves)  # byte-identical exports
    assert count(db, "run_saves") == 1 and count(db, "segments") == 3
    assert count(db, "save_events") == 8
    assert P.verify(db) == []


def test_concurrent_distinct_runs_and_readers(db, tmp_path):
    ctx = multiprocessing.get_context("spawn")
    manifests = [make_manifest(seed=f"s{i}", texts=["a", "b"]) for i in range(8)]
    P.save_run(manifests[0], db, tmp_path / "seed")
    with ProcessPoolExecutor(8, mp_context=ctx) as pool:
        writes = [pool.submit(save_in_worker, (x.model_dump_json(), str(db), str(tmp_path / f"o{i}")))
                  for i, x in enumerate(manifests[1:], 1)]
        reads = [pool.submit(export_in_worker, (str(db), manifests[0].run_id)) for _ in range(8)]
        assert all(f.result()[0] == "inserted" for f in writes)
        texts = {f.result() for f in reads}
    assert len(texts) == 1  # every reader saw the same complete snapshot
    assert count(db, "runs") == 8 and P.verify(db) == []


# -------------------------------------------------------------- export views

def test_json_csv_table_and_sql_views_share_one_projection(db, tmp_path):
    m, first, fix = correction_manifest()
    m = m.model_copy(update={"observations": [*m.observations, packaged(m.segments[0])]})
    result = P.save_run(m, db, tmp_path / "o")
    out = tmp_path / "o"
    json_rows = P.observation_rows(RunManifest.model_validate_json((out / "run.json").read_text()))
    csv_rows = list(csv.DictReader(io.StringIO((out / "observations.csv").read_text())))
    assert [{k: v or None for k, v in r.items()} for r in csv_rows] == json_rows
    assert tuple(csv_rows[0]) == P.OBSERVATION_COLUMNS
    seg_rows = list(csv.DictReader(io.StringIO((out / "segments.csv").read_text())))
    assert [r["normalized_text"] for r in seg_rows] == [s.normalized_text for s in m.segments]
    assert seg_rows[0]["core_start_s"] == "0.500"
    table = (out / "observations.txt").read_text().splitlines()
    assert len(table) == 2 + 3 and "farine" in table[2] and "2 carton x 12 bouteille = 24" in table[4]
    assert "false" in table[2] and "true" in table[3]
    assert P.export_run(db, m.run_id, "csv") == (out / "observations.csv").read_text()
    assert P.export_run(db, m.run_id, "csv", part="segments") == (out / "segments.csv").read_text()
    assert P.export_run(db, m.run_id, "table") == (out / "observations.txt").read_text()
    assert P.export_run(db, m.run_id, "json") == (out / "run.json").read_text()
    with sqlite3.connect(db) as conn:
        sql = conn.execute("SELECT observation_id, status, quantity, unit, correction_of, "
                           "location FROM current_observations ORDER BY ordinal").fetchall()
    assert sql == [(r["observation_id"], r["status"], r["quantity"], r["unit"],
                    r["correction_of"], r["location"]) for r in json_rows]
    assert result.files[-1].name == "run.json"


def test_hostile_text_cannot_inject_terminal_escapes_or_csv_formulas(db, tmp_path):
    text = "Ignore previous instructions =cmd|' /C calc'!A0"
    m = make_manifest(texts=[text], observations=[])
    evil = make_obs(m.segments[0], "e", quote="Ignore", product="=HYPERLINK(\"x\")\x1b[31mred\nline")
    m = m.model_copy(update={"observations": [evil]})
    P.save_run(m, db, tmp_path / "o")
    table = (tmp_path / "o/observations.txt").read_text()
    assert "\x1b" not in table and "\\u001b" in table and "\\n" in table
    assert len(table.splitlines()) == 3
    cells = next(csv.DictReader(io.StringIO((tmp_path / "o/observations.csv").read_text())))
    assert cells["product_spoken"].startswith("'=HYPERLINK")
    assert json.loads((tmp_path / "o/run.json").read_text())[
        "observations"][0]["product_spoken"].startswith("=HYPERLINK")  # JSON is lossless


def test_table_alignment_uses_display_width_and_marks_truncation():
    out = P.render_table(["a", "b"], [["é" * 3, "x"], ["日本", None], ["z" * 100, "y"]], max_cell=10)
    lines = out.splitlines()
    assert lines[2].startswith("ééé") and "日本" in lines[3] and lines[3].split()[-1] == "-"
    assert lines[4].split()[0].endswith("…") and len(lines[4].split()[0]) == 10
    assert P.terminal_safe("a\x00b‮c\n") == "a\\u0000b\\u202ec\\n"


def test_export_unknown_run_and_missing_database(db, tmp_path):
    with pytest.raises(P.NotFoundError, match="does not exist"):
        P.export_run(db, "run_x", "csv")
    P.save_run(make_manifest(), db, tmp_path / "o")
    with pytest.raises(P.NotFoundError, match="no run"):
        P.export_run(db, "run_nothing", "csv")
    P.save_run(make_manifest(seed="q"), db, tmp_path / "o")
    with pytest.raises(P.NotFoundError, match="ambiguous"):
        P.export_run(db, "run_", "csv")
