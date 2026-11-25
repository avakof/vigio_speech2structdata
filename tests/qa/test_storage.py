"""Exercise restart, rollback and exports against SQLite without audio or model stubs."""

import csv
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from speechflow import persistence
from speechflow.contracts import (AudioSource, EvidenceSpan, RunManifest, TranscriptSegment,
                                  ValidatedObservation, stable_id)


def inventory(version="v1"):
    digest = hashlib.sha256(b"independent QA source, no external recording").hexdigest()
    source_id = stable_id("file", digest)
    source = AudioSource(source_id=source_id, digest=digest, kind="file", name="qa.wav",
                         native_rate=16000, native_channels=1, sample_count=48000,
                         channel_mix="mono")
    run_id = stable_id("run", source_id, version, "run")
    segments, observations = [], []
    for i, text in enumerate(("Trois kilos de farine", "Trois kilos de farine", "Non, deux kilos")):
        start, end = i * 16000, (i + 1) * 16000
        sid = stable_id("segment", source_id, start, end)
        segments.append(TranscriptSegment(segment_id=sid, source_id=source_id, ordinal=i,
                                          core_start=start, core_end=end, input_start=start,
                                          input_end=end, raw={"text": text + " "}, normalized_text=text))
        oid = stable_id("observation", run_id, sid, 0, len(text))
        is_correction = i == 2
        observations.append(ValidatedObservation(
            observation_id=oid, product_spoken="farine", quantity_text="deux" if is_correction else "Trois",
            quantity="2" if is_correction else "3", unit="kg", qualifier="exact",
            operation="correction" if is_correction else "count", revision=2 if is_correction else 1,
            evidence=[EvidenceSpan(segment_id=sid, start=0, end=len(text), quote=text)],
            correction_of=observations[1].observation_id if is_correction else None,
        ))
    return RunManifest(run_id=run_id, processing_version=version, source=source, segments=segments,
                       observations=observations, mode="run", status="ok", transcription_status="ok",
                       extraction_status="ok", provenance={}, metrics={})


def cli(*args):
    executable = Path(sys.executable).parent / "speechflow"
    if executable.is_file():
        command = [str(executable)]
    else:
        command = [sys.executable, "-m", "speechflow.cli"]
    return subprocess.run([*command, *map(str, args)], capture_output=True, text=True, timeout=15)


def test_rollback_then_recover_in_new_process_preserves_history(tmp_path):
    db, out = tmp_path / "run.db", tmp_path / "result"
    manifest = inventory()
    persistence.connect(db).close()
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TRIGGER qa_abort BEFORE INSERT ON observations WHEN NEW.revision = 2 "
                     "BEGIN SELECT RAISE(ABORT, 'QA forced abort after earlier row writes'); END")
    with pytest.raises(persistence.PersistenceError, match="QA forced abort"):
        persistence.save(manifest, db, out)
    with sqlite3.connect(db) as conn:
        for table in ("sources", "runs", "run_saves", "segments", "observations", "observation_evidence"):
            assert conn.execute(f"SELECT count(*) FROM {table}").fetchone() == (0,)
        conn.execute("DROP TRIGGER qa_abort")
    assert list((out / ".recovery").glob("*.json"))
    recovered = cli("recover", "--database", db, "--out", out, "--json")
    assert recovered.returncode == 0, recovered.stderr
    assert not list((out / ".recovery").glob("*.json"))

    retry_out = tmp_path / "retry"
    persistence.save(manifest, db, retry_out)
    alternate = inventory("v2")
    persistence.save(alternate, db, tmp_path / "new-processing-version")
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM sources").fetchone() == (1,)
        assert conn.execute("SELECT count(*) FROM runs").fetchone() == (2,)
        assert conn.execute("SELECT count(*) FROM run_saves").fetchone() == (2,)
        rows = conn.execute("SELECT ordinal, quantity, is_final, correction_of "
                            "FROM current_observations WHERE run_id=? ORDER BY ordinal",
                            (manifest.run_id,)).fetchall()
        assert rows == [(0, "3", 1, None), (1, "3", 0, None),
                        (2, "2", 1, manifest.observations[1].observation_id)]
    assert persistence.verify(db) == []
    written = json.loads((out / "run.json").read_text())
    assert written["observations"][2]["correction_of"] == rows[2][3]
    exported = cli("export", manifest.run_id, "--database", db, "--format", "csv")
    assert exported.returncode == 0, exported.stderr
    assert exported.stdout == (retry_out / "observations.csv").read_text()
    csv_rows = list(csv.DictReader(io.StringIO(exported.stdout)))
    assert [o["is_final"] for o in csv_rows] == ["true", "false", "true"]
    assert [o["quantity"] for o in csv_rows] == ["3", "3", "2"]
    assert "farine" in (out / "observations.txt").read_text()


def test_out_of_source_audio_is_rejected(tmp_path):
    bad = inventory()
    bad.segments[2].input_end += 1
    with pytest.raises(persistence.IntegrityError):
        persistence.save(bad, tmp_path / "run.db", tmp_path / "bad-timeline")


def test_unevidenced_observation_is_rejected(tmp_path):
    bad = inventory()
    bad.observations[0].evidence = []
    with pytest.raises(persistence.IntegrityError):
        persistence.save(bad, tmp_path / "run.db", tmp_path / "no-evidence")


def test_recovery_rejects_bad_evidence_quote(tmp_path):
    bad = inventory()
    bad.observations[0].evidence[0].quote = "Quatre kilos de farine"
    db, out = tmp_path / "run.db", tmp_path / "result"
    with pytest.raises(persistence.IntegrityError, match="evidence"):
        persistence.save(bad, db, out)
    assert not db.exists() or not persistence.list_runs(db)
