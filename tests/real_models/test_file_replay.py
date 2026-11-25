"""Independent installed-CLI acceptance for one pinned French recording and its replay."""

import csv
from contextlib import closing
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import shlex
import signal
import sqlite3
import subprocess
import sys
import time
from urllib.request import ProxyHandler, build_opener
import wave

import pytest

from speechflow.assets import DIGEST, HOJO, silero_hash

pytestmark = pytest.mark.real_models
JACQUES = Path.home() / ".cache/speechflow/hojo-jacques.wav"
JACQUES_SHA256 = "d92c0759b7d4d3e03ae22eb3c47aa97ef7122789eac187c7cf42e8e6035d1330"
SAMPLES = 57_142
GOLD = [("lait", "12", "bouteille"), ("tomates", "2.5", "kg")]
VIEWS = ("run.json", "observations.csv", "segments.csv", "observations.txt", "segments.txt")


def _stable_id(prefix, *parts):
    canonical = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return prefix + "_" + hashlib.sha256(canonical.encode()).hexdigest()


def _check_fixture(path):
    assert path.is_file(), f"BLOCKED: missing pinned Jacques WAV at {path}"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == JACQUES_SHA256
    with wave.open(str(path), "rb") as audio:
        assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()) == (
            1, 2, 16_000, SAMPLES,
        )


def _run_cli(work, label, args, *, ollama_url=None):
    cli = Path(sys.executable).parent / "speechflow"
    assert cli.is_file(), "BLOCKED: installed speechflow executable is missing"
    cmd = [str(cli), *map(str, args)]
    stdout, stderr = work / f"{label}.stdout", work / f"{label}.stderr"
    snapshots = []
    opener = build_opener(ProxyHandler({}))
    started = time.monotonic()

    def loaded_models():
        with opener.open(ollama_url + "/api/ps", timeout=3) as response:
            return [(model["name"], model["digest"].removeprefix("sha256:"))
                    for model in json.load(response)["models"]]

    before = loaded_models() if ollama_url else None
    if ollama_url:
        assert before == [], "The serial test requires an idle Ollama"
    with stdout.open("wb") as out, stderr.open("wb") as err:
        with subprocess.Popen(cmd, stdout=out, stderr=err) as process:
            print(f"$ {shlex.join(cmd)}", flush=True)
            try:
                while ollama_url and process.poll() is None and time.monotonic() - started < 2100:
                    try:
                        sample = loaded_models()
                        if not snapshots or snapshots[-1]["models"] != sample:
                            snapshots.append({"seconds": time.monotonic() - started, "models": sample})
                    except (OSError, ValueError, KeyError) as exc:
                        snapshots.append({"seconds": time.monotonic() - started, "poll_error": str(exc),
                                          "models": None})
                    time.sleep(0.1)
                process.wait(timeout=max(0, 2100 - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
            finally:
                receipt = {"command": cmd, "pid": process.pid, "returncode": process.returncode,
                           "seconds": time.monotonic() - started, "ollama_snapshots": snapshots,
                           "ollama_models_before": before,
                           "stdout": stdout.name, "stderr": stderr.name}
                (work / f"{label}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if ollama_url:
        receipt["ollama_models_after"] = loaded_models()
        (work / f"{label}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    assert receipt["returncode"] == 0, (
        f"{shlex.join(cmd)} exited {receipt['returncode']}\n"
        f"stdout: {stdout.read_text()}\nstderr: {stderr.read_text()}"
    )
    if ollama_url:
        assert any(("qwen3:8b", DIGEST) in (entry["models"] or []) for entry in snapshots)
        assert receipt["ollama_models_after"] == [], "Ollama must release the pinned extractor after each CLI run"
    return receipt


def _snapshot(db, path):
    # Query the committed rows directly rather than through persistence or export helpers.
    with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        queries = {
            "source": "SELECT * FROM sources",
            "run": "SELECT * FROM current_runs",
            "saves": "SELECT * FROM run_saves ORDER BY generation",
            "events": "SELECT * FROM save_events ORDER BY event_id",
            "observations": "SELECT * FROM current_observations ORDER BY ordinal",
            "segments": "SELECT * FROM current_segments ORDER BY ordinal",
            "evidence": "SELECT e.* FROM observation_evidence e "
                        "JOIN current_observations o USING (save_id, observation_id) "
                        "ORDER BY o.ordinal, e.ordinal",
        }
        rows = {name: [dict(row) for row in conn.execute(sql)] for name, sql in queries.items()}
        rows["physical_counts"] = {table: conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                                   for table in ("runs", "segments", "observations", "observation_evidence")}
        rows["foreign_key_violations"] = [tuple(row) for row in conn.execute("PRAGMA foreign_key_check")]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    return rows


def _check_run(manifest, summary, settings):
    assert summary["exit_code"] == 0 and summary["observations"] == 2
    for field in ("run_id", "processing_version", "mode", "status", "transcription_status",
                  "extraction_status"):
        assert summary[field] == manifest[field]
    assert manifest["mode"] == "run" and all(manifest[field] == "ok" for field in (
        "status", "transcription_status", "extraction_status"))
    source = manifest["source"]
    assert (source["kind"], source["digest"], source["sample_count"], source["sample_rate"],
            source["native_rate"], source["native_channels"]) == (
        "file", JACQUES_SHA256, SAMPLES, 16_000, 16_000, 1)
    assert summary["source"] == {key: source[key] for key in ("source_id", "kind", "name", "digest")}
    assert source["source_id"] == _stable_id("file", JACQUES_SHA256)
    assert manifest["run_id"] == _stable_id("run", source["source_id"], manifest["processing_version"], "run")
    provenance = manifest["provenance"]
    assert (provenance["backend"], provenance["hojo_revision"], provenance["extractor_digest"],
            provenance["silero_sha256"]) == (settings.backend, HOJO["revision"], DIGEST, silero_hash())
    assert provenance["dtype"] and provenance["dtype"] != "not_loaded"
    audio = manifest["metrics"]["audio"]
    assert audio["asr_seconds"] > 0 and audio["model_load_seconds"] > 0
    assert manifest["metrics"]["pipeline_seconds"] > 0
    segments = manifest["segments"]
    assert len(segments) == summary["segments"] > 0
    assert len({s["segment_id"] for s in segments}) == len(segments)
    sizes = []
    previous_end = 0
    for ordinal, seg in enumerate(segments):
        input_start, core_start, core_end, input_end = (
            seg[key] for key in ("input_start", "core_start", "core_end", "input_end"))
        assert 0 <= input_start <= core_start < core_end <= input_end <= SAMPLES
        assert core_start >= previous_end
        size = input_end - input_start
        assert size <= int(settings.max_chunk_seconds * 16_000)
        assert seg["ordinal"] == ordinal and seg["source_id"] == source["source_id"]
        assert seg["segment_id"] == _stable_id(
            "segment", source["source_id"], core_start, core_end, input_start, input_end)
        assert seg["raw"]["key"] == seg["segment_id"] and seg["raw"]["text"].strip()
        sizes.append(size)
        previous_end = core_end
    assert sizes == audio["asr_chunk_sample_counts"]
    assert manifest["attempts"]
    assert {key for attempt in manifest["attempts"] for key in attempt["segment_ids"]} == {
        seg["segment_id"] for seg in segments}
    for attempt in manifest["attempts"]:
        assert attempt["extractor"]["digest"].removeprefix("sha256:") == DIGEST
        response = attempt["response_raw"]
        assert response["model"] == "qwen3:8b" and response["done"] is True
        assert response["eval_count"] > 0 and response["total_duration"] > 0
    assert manifest["metrics"]["extraction"]["unload"]["idle_confirmed"] is True
    observations = manifest["observations"]
    assert sorted((o["product_spoken"], o["quantity"], o["unit"]) for o in observations) == GOLD
    for o in observations:
        assert o["status"] == "ok" and o["operation"] == "count" and o["qualifier"] == "exact"
        assert o["revision"] == 1 and o["correction_of"] is None and o["evidence"]
        assert o["packaging"] is None or o["packaging"]["per_package_quantity"] is None


def _read_table(path):
    lines = path.read_text().splitlines()
    columns = re.split(r" {2,}", lines[0].strip())
    return [dict(zip(columns, re.split(r" {2,}", line.strip()), strict=True)) for line in lines[2:]]


def _check_committed_views(snapshot, out):
    save, = snapshot["saves"]
    current, = snapshot["run"]
    source, = snapshot["source"]
    assert snapshot["foreign_key_violations"] == []
    assert current["save_id"] == save["save_id"] and current["generation"] == save["generation"] == 1
    assert hashlib.sha256(save["manifest_json"].encode()).hexdigest() == save["manifest_digest"]
    manifest = json.loads(save["manifest_json"])
    assert (out / "run.json").read_bytes() == (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    for key, value in manifest["source"].items():
        assert source[key] == value
    for key in ("run_id", "processing_version", "schema_version", "mode", "status", "transcription_status",
                "extraction_status"):
        assert current[key] == manifest[key]
    assert current["source_id"] == source["source_id"]
    assert snapshot["physical_counts"] == {
        "runs": 1, "segments": len(manifest["segments"]), "observations": 2,
        "observation_evidence": sum(len(o["evidence"]) for o in manifest["observations"]),
    }
    with (out / "segments.csv").open(newline="") as file:
        segment_csv = list(csv.DictReader(file))
    with (out / "observations.csv").open(newline="") as file:
        observation_csv = list(csv.DictReader(file))
    segment_table = _read_table(out / "segments.txt")
    observation_table = _read_table(out / "observations.txt")
    texts = {}
    for row, seg, csv_row, table_row in zip(snapshot["segments"], manifest["segments"], segment_csv,
                                            segment_table, strict=True):
        for key, value in seg.items():
            assert (json.loads(row[key + "_json"]) if key in {"raw", "errors"} else row[key]) == value
        for key in ("ordinal", "segment_id", "status", "input_start", "core_start", "core_end", "input_end",
                    "normalized_text"):
            assert csv_row[key] == str(row[key])
        assert table_row["segment"] == row["segment_id"].replace("segment_", "seg_")[:14]
        assert table_row["status"] == row["status"]
        for part in ("start", "end"):
            seconds = f"{Decimal(row[f'core_{part}']) / 16_000:.3f}"
            assert table_row[f"{part}_s"] == csv_row[f"core_{part}_s"] == seconds
        if table_row["text"].endswith("…"):
            assert row["normalized_text"].startswith(table_row["text"][:-1])
        else:
            assert table_row["text"] == row["normalized_text"]
        texts[row["segment_id"]] = row["normalized_text"]

    for row, obs, csv_row, table_row in zip(snapshot["observations"], manifest["observations"],
                                            observation_csv, observation_table, strict=True):
        for key, value in obs.items():
            if key in {"packaging", "errors"}:
                raw = row[key + "_json"]
                assert (json.loads(raw) if raw is not None else None) == value
            elif key != "evidence":
                assert row[key] == value
                assert csv_row[key] == (str(value) if value is not None else "")
        evidence = [e for e in snapshot["evidence"] if e["observation_id"] == row["observation_id"]]
        for e in evidence:
            assert texts[e["segment_id"]][e["span_start"]:e["span_end"]] == e["quote"]
        assert [{"segment_id": e["segment_id"], "start": e["span_start"], "end": e["span_end"],
                 "quote": e["quote"]} for e in evidence] == obs["evidence"]
        assert csv_row["evidence"] == ";".join(f"{e['segment_id']}:{e['span_start']}-{e['span_end']}"
                                               for e in evidence)
        assert csv_row["evidence_quotes"] == " | ".join(e["quote"] for e in evidence)
        assert row["is_final"] == 1 and row["corrected_by"] is None
        assert table_row["final"] == csv_row["is_final"] == "true"
        assert (table_row["product"], table_row["qty"], table_row["unit"]) == (
            row["product_spoken"], row["quantity"], row["unit"])
        assert table_row["obs"] == row["observation_id"][:14]
        assert table_row["status"] == row["status"]
    return manifest


def test_real_file_installed_cli_replay(tmp_path, pinned_hojo, pinned_ollama, model_settings):
    _check_fixture(JACQUES)
    db = tmp_path / "inventory.sqlite3"
    assert not db.exists()
    flags = ("--offline", "--backend", model_settings.backend, "--database", db)
    first, replay, exports = (tmp_path / name for name in ("first", "replay", "exports"))
    first_receipt = _run_cli(tmp_path, "first", ("run", JACQUES, "--out", first, "--json", *flags),
                             ollama_url=model_settings.ollama_url)
    first_summary = json.loads((tmp_path / "first.stdout").read_bytes())
    assert first_summary["disposition"] == "inserted" and first_summary["generation"] == 1
    before = _snapshot(db, tmp_path / "first-sql.json")
    manifest = _check_committed_views(before, first)
    _check_run(manifest, first_summary, model_settings)

    replay_receipt = _run_cli(tmp_path, "replay", ("run", JACQUES, "--out", replay, "--json", *flags),
                              ollama_url=model_settings.ollama_url)
    assert replay_receipt["pid"] != first_receipt["pid"]
    replay_summary = json.loads((tmp_path / "replay.stdout").read_bytes())
    assert replay_summary["disposition"] == "duplicate" and replay_summary["generation"] == 1
    after = _snapshot(db, tmp_path / "replay-sql.json")
    _check_run(_check_committed_views(after, replay), replay_summary, model_settings)
    assert {k: v for k, v in before.items() if k != "events"} == {
        k: v for k, v in after.items() if k != "events"}
    inserted, duplicate = after["events"]
    assert before["events"] == [inserted]
    assert [event["disposition"] for event in (inserted, duplicate)] == ["inserted", "duplicate"]
    assert inserted["save_id"] == duplicate["save_id"] == after["run"][0]["save_id"]
    assert inserted["run_id"] == duplicate["run_id"] == manifest["run_id"]
    assert inserted["manifest_digest"] == after["saves"][0]["manifest_digest"]
    assert duplicate["manifest_digest"] != inserted["manifest_digest"], "Expected a fresh inference receipt"

    for fmt in ("json", "csv", "table"):
        _run_cli(tmp_path, f"export-{fmt}",
                 ("export", manifest["run_id"], "--format", fmt, "--out", exports, *flags))
    for name in VIEWS:
        assert (first / name).read_bytes() == (replay / name).read_bytes() == (exports / name).read_bytes()
    assert _snapshot(db, tmp_path / "export-sql.json") == after
    _check_fixture(JACQUES)
