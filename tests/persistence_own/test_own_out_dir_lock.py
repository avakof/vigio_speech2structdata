"""Exports must stay coherent when saves and exports target the same output directory.

Each scenario interleaves workers deterministically with events (no sleeps decide
the outcome). Pausing points sit between the SQLite commit and the view swap, or
between two file swaps, which is exactly where an unserialized writer can be
overtaken. Joins that only ever time out when the directory lock works are bounded at
WAIT seconds; without the lock the overtaking worker finishes inside that window.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from own_factory import make_manifest

from speechflow import persistence as P

WAIT = 0.5
SWAP_ORDER = ["observations.csv", "observations.txt", "segments.csv", "segments.txt", "run.json"]


def failed_then_fixed():
    """Same run identity; the first attempt failed, the retry produced a transcript."""
    failed = make_manifest(seed="same", texts=["un"], status="error", observations=[],
                           errors=["extractor down"])
    fixed = make_manifest(seed="same", texts=["un", "deux"])
    assert failed.run_id == fixed.run_id
    return failed, fixed


def assert_dir_matches_db(out: Path, db: Path, run_id: str, *, keep=()) -> None:
    expected = P.render_views(P.load_manifest(db, run_id))
    assert {n: (out / n).read_text() for n in expected} == expected
    assert not (out / P.RECOVERY_DIR).exists(), "completed saves leave no pending recovery copy"
    assert not [p for p in out.iterdir()
                if p.name.startswith(".export-") and p.name not in keep], "own stage left behind"


class Worker(threading.Thread):
    def __init__(self, name, fn):
        super().__init__(name=name, daemon=True)
        self.fn, self.result, self.error = fn, None, None

    def run(self):
        try:
            self.result = self.fn()
        except BaseException as exc:  # surfaced by finish()
            self.error = exc

    def finish(self):
        self.join(10)
        assert not self.is_alive(), f"worker {self.name} hung"
        if self.error:
            raise self.error
        return self.result


def entering_clock():
    """A clock that reports when a save reaches its database transaction."""
    entered = threading.Event()

    def clock():
        entered.set()
        return P._now()

    return entered, clock


def pause_first_view_write(monkeypatch, thread_name):
    real = P.write_views
    reached, release = threading.Event(), threading.Event()

    def wrapper(*args, **kwargs):
        if threading.current_thread().name == thread_name and not reached.is_set():
            reached.set()
            assert release.wait(10)
        return real(*args, **kwargs)

    monkeypatch.setattr(P, "write_views", wrapper)
    return reached, release


def test_save_overtaken_between_commit_and_views_cannot_leave_stale_exports(tmp_path, monkeypatch):
    db, out = tmp_path / "h.sqlite3", tmp_path / "out"
    failed, fixed = failed_then_fixed()
    reached, release = pause_first_view_write(monkeypatch, "A")
    entered, clock = entering_clock()

    a = Worker("A", lambda: P.save_run(failed, db, out))
    a.start()
    assert reached.wait(10)                    # A committed generation 1 and holds the views
    b = Worker("B", lambda: P.save_run(fixed, db, out, clock=clock))
    b.start()
    b.join(WAIT)                               # unserialized, B overtakes A and finishes here
    b_entered_while_a_paused = entered.is_set()
    release.set()
    a.finish()
    b_result = b.finish()

    assert b_result.disposition == "replaced" and b_result.generation == 2
    assert_dir_matches_db(out, db, fixed.run_id)
    assert P.load_manifest(db, fixed.run_id).status == "ok"
    assert not b_entered_while_a_paused, "B's commit must wait until A's views are written"


def test_interleaved_file_swaps_cannot_mix_generations(tmp_path, monkeypatch):
    db, out = tmp_path / "h.sqlite3", tmp_path / "out"
    failed, fixed = failed_then_fixed()
    counts = {"A": 0, "B": 0}
    a_after_first, b_after_third, a_done = (threading.Event() for _ in range(3))
    real_replace = P.os.replace

    def replace(src, dst):
        real_replace(src, dst)
        who = threading.current_thread().name
        if who in counts and Path(dst).name in SWAP_ORDER:
            counts[who] += 1
            if who == "A" and counts["A"] == 1:      # A: one file, then let B run ahead
                a_after_first.set()
                b_after_third.wait(WAIT)
            if who == "B" and counts["B"] == 3:      # B: three files, then let A finish
                b_after_third.set()
                a_done.wait(WAIT)

    monkeypatch.setattr(P.os, "replace", replace)
    a = Worker("A", lambda: P.save_run(failed, db, out))
    a.start()
    assert a_after_first.wait(10)
    b = Worker("B", lambda: P.save_run(fixed, db, out))
    b.start()
    a.finish()
    a_done.set()
    b.finish()
    assert_dir_matches_db(out, db, fixed.run_id)


def test_export_started_before_a_newer_save_cannot_overwrite_it(tmp_path, monkeypatch):
    db, out = tmp_path / "h.sqlite3", tmp_path / "out"
    failed, fixed = failed_then_fixed()
    P.save_run(failed, db, out)
    reached, release = pause_first_view_write(monkeypatch, "X")
    entered, clock = entering_clock()

    x = Worker("X", lambda: P.export_run(db, failed.run_id, "csv", out=out))
    x.start()
    assert reached.wait(10)                    # X read generation 1 and is about to write it
    a = Worker("A", lambda: P.save_run(fixed, db, out, clock=clock))
    a.start()
    a.join(WAIT)
    save_entered_while_export_paused = entered.is_set()
    release.set()
    x.finish()
    a.finish()

    assert_dir_matches_db(out, db, fixed.run_id)
    assert not save_entered_while_export_paused


def test_save_after_pending_explicit_export_of_same_directory_is_serialized(tmp_path, monkeypatch):
    """The reverse order: a save holds the directory, an explicit export must wait for
    it and then render the newest generation, not the one it would have read first."""
    db, out = tmp_path / "h.sqlite3", tmp_path / "out"
    failed, fixed = failed_then_fixed()
    P.save_run(failed, db, out)
    reached, release = pause_first_view_write(monkeypatch, "A")
    loaded = threading.Event()
    real_load = P.load_manifest

    def load(*args, **kwargs):
        if threading.current_thread().name == "X":
            loaded.set()
        return real_load(*args, **kwargs)

    monkeypatch.setattr(P, "load_manifest", load)
    a = Worker("A", lambda: P.save_run(fixed, db, out))
    a.start()
    assert reached.wait(10)
    x = Worker("X", lambda: P.export_run(db, failed.run_id, "table", out=out))
    x.start()
    x.join(WAIT)
    export_read_database_while_save_held_directory = loaded.is_set()
    release.set()
    a.finish()
    x.finish()
    assert_dir_matches_db(out, db, fixed.run_id)
    assert not export_read_database_while_save_held_directory


def test_different_output_directories_stay_independent(tmp_path, monkeypatch):
    db = tmp_path / "h.sqlite3"
    one, other = make_manifest(seed="one"), make_manifest(seed="other")
    reached, release = pause_first_view_write(monkeypatch, "A")
    a = Worker("A", lambda: P.save_run(one, db, tmp_path / "out-a"))
    a.start()
    assert reached.wait(10)                    # A holds out-a's lock and is paused
    b = Worker("B", lambda: (P.save_run(other, db, tmp_path / "out-b"),
                             P.export_run(db, other.run_id, "json", out=tmp_path / "out-b")))
    b.start()
    b.join(5)
    finished_while_a_held_its_lock = not b.is_alive()
    release.set()
    a.finish()
    b.finish()
    assert finished_while_a_held_its_lock, "a lock on one directory must not block another"
    assert_dir_matches_db(tmp_path / "out-a", db, one.run_id)
    assert_dir_matches_db(tmp_path / "out-b", db, other.run_id)


def test_many_threads_saving_and_exporting_one_directory_end_coherent(tmp_path):
    db, out = tmp_path / "h.sqlite3", tmp_path / "out"
    failed, fixed = failed_then_fixed()
    P.save_run(failed, db, out)
    with ThreadPoolExecutor(10) as pool:
        jobs = [pool.submit(P.save_run, fixed, db, out) for _ in range(5)]
        jobs += [pool.submit(P.export_run, db, fixed.run_id, fmt, out=out)
                 for fmt in ("csv", "table", "json", "csv", "table")]
        for job in jobs:
            job.result(20)
    assert_dir_matches_db(out, db, fixed.run_id)
    assert P.verify(db) == []


def test_lock_dies_with_its_holder_and_a_timeout_keeps_the_manifest(tmp_path, monkeypatch):
    db, out = tmp_path / "h.sqlite3", tmp_path / "out"
    out.mkdir()
    (out / ".export-crashed").mkdir()          # leftover stage of a killed writer
    (out / ".export-crashed" / "run.json").write_text("{}")
    holder = subprocess.Popen([sys.executable, "-c", textwrap.dedent("""
        import sys, time
        from pathlib import Path
        from speechflow import persistence as P
        with P._out_lock(Path(sys.argv[1])):
            Path(sys.argv[2]).write_text("held")
            time.sleep(60)
    """), str(out), str(tmp_path / "held")])
    try:
        for _ in range(500):
            if (tmp_path / "held").exists():
                break
            time.sleep(0.02)
        assert (tmp_path / "held").exists()
        monkeypatch.setattr(P, "OUT_LOCK_TIMEOUT_S", 0.3, raising=False)
        manifest = make_manifest()
        with pytest.raises(P.PersistenceError, match="locked by another speechflow process"):
            P.save_run(manifest, db, out)
        assert not db.exists() or P.list_runs(db) == []       # nothing committed unlocked
        assert len(list((out / P.RECOVERY_DIR).glob("*.json"))) == 1   # but nothing was lost
    finally:
        holder.kill()                                          # abrupt death releases the lock
        holder.wait(10)
    monkeypatch.setattr(P, "OUT_LOCK_TIMEOUT_S", 5, raising=False)
    assert [r[1] for r in P.recover(db, out)] == ["inserted"]
    assert (out / ".export-crashed" / "run.json").read_text() == "{}"   # never swept
    assert_dir_matches_db(out, db, manifest.run_id, keep={".export-crashed"})


def test_unrelated_dot_export_entries_survive_save_export_and_recovery(tmp_path):
    db, out = tmp_path / "h.sqlite3", tmp_path / "out"
    out.mkdir()
    sentinels = {".export-family-photos/img.txt": "precious", ".export-notes": "mine",
                 ".export-/inner": "empty-suffix"}
    for rel, body in sentinels.items():
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    failed, fixed = failed_then_fixed()
    P.save_run(failed, db, out)
    P.export_run(db, failed.run_id, "csv", out=out)
    P.write_views(P.load_manifest(db, failed.run_id), out)
    pending = P._write_recovery(out, fixed, P.content_digest(fixed))   # a crashed save's copy
    assert [r[1] for r in P.recover(db, out)] == ["replaced"]
    assert not pending.exists()
    P.save_run(fixed, db, out)
    for rel, body in sentinels.items():
        assert (out / rel).read_text() == body, rel
    assert_dir_matches_db(out, db, fixed.run_id, keep={".export-family-photos", ".export-notes",
                                                       ".export-"})


def test_forked_child_neither_reenters_nor_prolongs_the_parents_lock(tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setattr(P, "OUT_LOCK_TIMEOUT_S", 0.3, raising=False)
    result_r, result_w = os.pipe()
    exit_r, exit_w = os.pipe()
    with P._out_lock(out):
        pid = os.fork()
        if pid == 0:                                   # child: inherited the open lock FD
            code = 1
            try:
                try:
                    with P._out_lock(out):
                        os.write(result_w, b"reentered")
                except P.PersistenceError:
                    os.write(result_w, b"blocked")
                os.read(exit_r, 1)                     # stay alive, still holding the FD copy
                code = 0
            finally:
                os._exit(code)
        outcome = os.read(result_r, 16)
    started = time.monotonic()
    monkeypatch.setattr(P, "OUT_LOCK_TIMEOUT_S", 2.0, raising=False)
    try:
        with P._out_lock(out):                         # child is alive; parent already left
            waited = time.monotonic() - started
    finally:
        os.write(exit_w, b"x")
        _, status = os.waitpid(pid, 0)
    assert outcome == b"blocked", "a forked child must not inherit ownership of the lock"
    assert waited < 1.0, "the parent's exit must release the lock even if a child holds the FD"
    assert os.waitstatus_to_exitcode(status) == 0
