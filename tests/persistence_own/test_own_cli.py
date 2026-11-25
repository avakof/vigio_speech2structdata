"""Command line: parsing, configuration, exit codes, offline/missing-dependency behaviour."""

from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import pytest
from own_factory import make_manifest

from speechflow import cli
from speechflow import persistence as P

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "in.m4a"
    path.write_bytes(b"not decoded by the stub pipeline")
    return path


@pytest.fixture
def stub(monkeypatch):
    """Replace speechflow.pipeline with a recorder returning a chosen manifest."""
    state = types.SimpleNamespace(manifest=make_manifest(), calls=[], raises=None)

    def process(path, *, mode, settings, out):
        state.calls.append(types.SimpleNamespace(path=path, mode=mode, settings=settings, out=out))
        if state.raises:
            raise state.raises
        return state.manifest.model_copy(update={"mode": mode}) if mode != state.manifest.mode \
            else state.manifest

    module = types.ModuleType("speechflow.pipeline")
    module.process = process
    monkeypatch.setitem(sys.modules, "speechflow.pipeline", module)
    monkeypatch.setattr("speechflow.pipeline", module, raising=False)
    return state


def run(args, capsys):
    code = cli.main([str(a) for a in args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ------------------------------------------------------------ help / config

def test_example_settings_validate_and_cover_every_setting():
    from speechflow.config import Settings, load_settings
    path = REPO / "examples/settings.toml"
    keys = set(tomllib.loads(path.read_text()))
    assert load_settings(path) is not None
    commented = {line.lstrip("# ").split(" =")[0] for line in path.read_text().splitlines()
                 if line.startswith("# mic_")}
    assert keys | commented == set(Settings.model_fields)


def test_help_lists_every_required_command(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for word in ("doctor", "models", "transcribe", "run", "mic", "evaluate", "export"):
        assert word in out
    for sub in (["models", "prepare"], ["mic"], ["export"], ["run"]):
        with pytest.raises(SystemExit) as exc:
            cli.main([*sub, "--help"])
        assert exc.value.code == 0
    assert "--seconds" in capsys.readouterr().out


def test_cli_module_imports_only_stdlib():
    code = ("import sys; import speechflow.cli; "
            "bad = [m for m in ('pydantic', 'torch', 'numpy', 'transformers') if m in sys.modules];"
            "sys.exit(len(bad))")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_usage_and_config_errors_exit_2(tmp_path, audio, stub, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["export"])
    assert exc.value.code == 2
    capsys.readouterr()
    code, _, err = run(["run", audio, "--backend", "cuda"], capsys)
    assert code == 2 and "backend" in err and not stub.calls
    code, out, _ = run(["doctor", "--config", tmp_path / "missing.toml"], capsys)
    assert code == 2 and "[FAIL] configuration: invalid configuration" in out
    bad = tmp_path / "bad.toml"
    bad.write_text("surprise = 1\n")
    code, _, err = run(["run", audio, "--config", bad], capsys)
    assert code == 2 and "surprise" in err
    bad.write_text("not toml [")
    code, _, err = run(["run", audio, "--config", bad], capsys)
    assert code == 2 and "invalid configuration" in err


def test_options_override_settings_file_in_either_position(tmp_path, audio, stub, capsys):
    cfg = tmp_path / "s.toml"
    cfg.write_text('backend = "cpu"\noffline = false\n')
    db = tmp_path / "x.sqlite3"
    code, *_ = run(["--config", cfg, "run", audio, "--backend", "mps", "--offline",
                    "--database", db, "--out", tmp_path / "o", "-q"], capsys)
    settings = stub.calls[-1].settings
    assert code == 0 and (settings.backend, settings.offline, settings.database) == (
        "mps", True, db.resolve())
    run(["run", audio, "--config", cfg, "--database", db, "--out", tmp_path / "o", "-q"], capsys)
    assert stub.calls[-1].settings.backend == "cpu" and not stub.calls[-1].settings.offline
    run(["--offline", "run", audio, "--database", db, "--out", tmp_path / "o", "-q"], capsys)
    assert stub.calls[-1].settings.offline


def test_mic_options_reach_settings_and_no_path(tmp_path, stub, capsys):
    stub.manifest = make_manifest(mode="mic", kind="mic")
    code, *_ = run(["mic", "--seconds", "3.5", "--device", "2", "--database", tmp_path / "d",
                    "--out", tmp_path / "live", "-q"], capsys)
    call = stub.calls[0]
    assert code == 0 and call.path is None and call.mode == "mic"
    assert (call.settings.mic_seconds, call.settings.mic_device) == (3.5, 2)
    run(["mic", "--device", "USB Mic", "--database", tmp_path / "d", "--out", tmp_path / "live",
         "-q"], capsys)
    assert stub.calls[1].settings.mic_device == "USB Mic"
    code, _, err = run(["mic", "--seconds", "0", "--database", tmp_path / "d"], capsys)
    assert code == 2 and "mic_seconds" in err


# --------------------------------------------------------------- exit codes

@pytest.mark.parametrize("status, expected", [("ok", 0), ("no_speech", 10),
                                              ("needs_review", 11), ("error", 12)])
def test_exit_status_is_deterministic_and_run_is_saved(tmp_path, audio, stub, capsys, status,
                                                       expected):
    stub.manifest = make_manifest(status=status, texts=[] if status == "no_speech" else None,
                                  observations=[] if status in ("no_speech", "error") else "default")
    args = ["run", audio, "--database", tmp_path / "d.sqlite3", "--out", tmp_path / "o"]
    code, out, _ = run(args, capsys)
    assert code == expected and f"status: {status}" in out
    code2, *_ = run(args, capsys)  # retry: same status, no duplicate rows
    assert code2 == expected
    assert [r["status"] for r in P.list_runs(tmp_path / "d.sqlite3")] == [status]


def test_transcribe_only_success_is_distinct_from_extraction(tmp_path, audio, stub, capsys):
    stub.manifest = make_manifest(mode="transcribe")
    code, out, _ = run(["transcribe", audio, "--database", tmp_path / "d", "--out", tmp_path / "o"],
                       capsys)
    assert code == 0 and "extraction: not_run" in out and "normalized" not in out
    assert stub.calls[0].mode == "transcribe" and stub.calls[0].path == audio
    stored = P.list_runs(tmp_path / "d")[0]
    assert (stored["transcription_status"], stored["extraction_status"]) == ("ok", "not_run")


def test_run_json_summary_is_one_valid_document(tmp_path, audio, stub, capsys):
    code, out, _ = run(["run", audio, "--json", "--database", tmp_path / "d",
                        "--out", tmp_path / "o"], capsys)
    data = json.loads(out)
    assert code == 0 and data["disposition"] == "inserted" and data["observations"] == 1
    assert data["exit_code"] == 0 and data["files"][-1].endswith("run.json")


def test_input_checks_come_before_the_pipeline(tmp_path, stub, capsys):
    (tmp_path / "empty.wav").write_bytes(b"")
    for path in (tmp_path / "nope.wav", tmp_path, tmp_path / "empty.wav"):
        code, _, err = run(["run", path, "--database", tmp_path / "d"], capsys)
        assert code == 4 and "error:" in err
    assert not stub.calls


def make_error(module, name="Boom", base=Exception):
    return type(name, (base,), {"__module__": module})("something actionable")


@pytest.mark.parametrize("exc, expected", [
    (lambda: make_error("speechflow.audio.decoding"), 4),
    (lambda: make_error("speechflow.assets", base=RuntimeError), 3),
    (lambda: ModuleNotFoundError("No module named 'torch'"), 3),
    (lambda: make_error("speechflow.persistence"), 5),
    (lambda: ValueError("surprise"), 1),
])
def test_fatal_errors_map_to_exit_codes_without_saving(tmp_path, audio, stub, capsys, exc, expected):
    stub.raises = exc()
    code, out, err = run(["run", audio, "--database", tmp_path / "d", "--out", tmp_path / "o"],
                         capsys)
    assert code == expected and "error:" in err and not (tmp_path / "d").exists()


def test_missing_pipeline_module_reports_install_hint(tmp_path, audio, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "speechflow.pipeline", None)  # import raises ImportError
    code, _, err = run(["run", audio, "--database", tmp_path / "d"], capsys)
    assert code == 3 and "uv sync --frozen" in err and "models prepare" in err


def test_interrupt_exits_130_and_mentions_recovery(tmp_path, audio, stub, capsys):
    stub.raises = KeyboardInterrupt()
    code, _, err = run(["run", audio, "--database", tmp_path / "d"], capsys)
    assert code == 130 and "recover" in err


def test_persistence_failure_exits_5_and_recover_completes_it(tmp_path, audio, stub, capsys,
                                                              monkeypatch):
    real = P._insert_children
    monkeypatch.setattr(P, "_insert_children",
                        lambda *a: (_ for _ in ()).throw(sqlite3_error()))
    db, out = tmp_path / "d", tmp_path / "o"
    code, _, err = run(["run", audio, "--database", db, "--out", out], capsys)
    assert code == 5 and "disk I/O error" in err and "recover" in err
    assert list((out / ".recovery").glob("*.json"))
    monkeypatch.setattr(P, "_insert_children", real)
    code, text, _ = run(["recover", "--database", db, "--out", out], capsys)
    assert code == 0 and "inserted" in text
    code, text, _ = run(["recover", "--database", db, "--out", out], capsys)
    assert code == 0 and "nothing to recover" in text


def sqlite3_error():
    import sqlite3
    return sqlite3.OperationalError("disk I/O error")


# ----------------------------------------------------------- stored history

def test_export_and_runs_commands(tmp_path, audio, stub, capsys):
    db = tmp_path / "d.sqlite3"
    stub.manifest = make_manifest(texts=["un", "deux"])
    run(["run", audio, "--database", db, "--out", tmp_path / "o", "-q"], capsys)
    run_id = stub.manifest.run_id
    code, out, _ = run(["export", run_id, "--format", "csv", "--database", db], capsys)
    assert code == 0 and out == (tmp_path / "o/observations.csv").read_text()
    assert list(csv.reader(io.StringIO(out)))[0][1] == "observation_id"
    code, out, _ = run(["export", run_id[:10], "--format", "csv", "--part", "segments",
                        "--database", db], capsys)
    assert out == (tmp_path / "o/segments.csv").read_text()
    code, out, _ = run(["export", run_id, "--format", "table", "--database", db], capsys)
    assert out == (tmp_path / "o/observations.txt").read_text()
    code, out, _ = run(["export", run_id, "--format", "json", "--database", db], capsys)
    assert json.loads(out)["run_id"] == run_id
    code, out, _ = run(["export", run_id, "--database", db, "--out", tmp_path / "e"], capsys)
    assert code == 0 and (tmp_path / "e/observations.csv").exists()
    code, out, _ = run(["runs", "--database", db], capsys)
    assert code == 0 and run_id in out
    code, out, _ = run(["runs", "--json", "--database", db], capsys)
    assert json.loads(out)[0]["observations"] == 2
    code, _, err = run(["export", "run_unknown", "--database", db], capsys)
    assert code == 5 and "no run" in err
    code, _, err = run(["export", run_id, "--database", tmp_path / "none"], capsys)
    assert code == 5 and "does not exist" in err


def test_evaluate_delegates_and_maps_status(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("{}\n")
    seen = {}

    def evaluate(path, out, settings):
        seen.update(path=path, out=out, offline=settings.offline)
        return {"status": "needs_review", "wer": "NOT MEASURED"}

    module = types.ModuleType("speechflow.evaluation")
    module.evaluate = evaluate
    monkeypatch.setitem(sys.modules, "speechflow.evaluation", module)
    monkeypatch.setattr("speechflow.evaluation", module, raising=False)
    code, out, _ = run(["evaluate", manifest, "--out", tmp_path / "ev", "--offline"], capsys)
    assert code == 11 and json.loads(out)["wer"] == "NOT MEASURED"
    assert seen == {"path": manifest, "out": tmp_path / "ev", "offline": True}
    code, _, err = run(["evaluate", tmp_path / "absent.jsonl"], capsys)
    assert code == 4


def test_models_prepare_delegates_and_reports_missing_assets(tmp_path, monkeypatch, capsys):
    module = types.ModuleType("speechflow.assets")
    calls = []

    def prepare(settings):
        calls.append(settings.offline)
        if settings.offline:
            raise RuntimeError("Missing weights\nMissing tokenizer")
        return {"hojo_files_verified": 3}

    module.prepare = prepare
    monkeypatch.setitem(sys.modules, "speechflow.assets", module)
    monkeypatch.setattr("speechflow.assets", module, raising=False)
    code, out, _ = run(["models", "prepare", "--database", tmp_path / "d"], capsys)
    assert code == 0 and json.loads(out) == {"hojo_files_verified": 3}
    code, out, err = run(["models", "prepare", "--offline", "--database", tmp_path / "d"], capsys)
    assert code == 3 and "Missing weights" in err and "never downloads" in err
    assert calls == [False, True]


# -------------------------------------------------- installed console script

def _entry():
    found = shutil.which("speechflow", path=str(Path(sys.executable).parent))
    if not found:
        pytest.skip("speechflow console script is not installed in this interpreter's venv")
    return found


def test_installed_script_help_and_doctor_without_models(tmp_path):
    exe = _entry()
    helped = subprocess.run([exe, "--help"], capture_output=True, text=True)
    assert helped.returncode == 0 and "doctor" in helped.stdout
    done = subprocess.run([exe, "doctor", "--json", "--database", str(tmp_path / "d")],
                          capture_output=True, text=True,
                          env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"})
    report = json.loads(done.stdout)
    assert done.returncode == (0 if report["ready"] else 3)
    assert {"checks", "assets", "ready"} <= set(report)
    assert any(c["name"] == "python" and c["ok"] for c in report["checks"])
    assert not (tmp_path / "d").exists()  # doctor never creates the database
    text = subprocess.run([exe, "doctor", "--database", str(tmp_path / "d")], capture_output=True,
                          text=True, env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"})
    assert text.stdout.rstrip().splitlines()[-1] in ("result: READY", "result: NOT READY")
    offline = subprocess.run([exe, "models", "prepare", "--offline", "--database",
                              str(tmp_path / "d")], capture_output=True, text=True,
                             env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"})
    assert offline.returncode == 3 and not (tmp_path / ".cache/speechflow/assets/hojo").exists()


def test_doctor_reports_missing_dependencies_and_pins(monkeypatch, capsys, tmp_path):
    real = cli.importlib.util.find_spec
    monkeypatch.setattr(cli.importlib.util, "find_spec",
                        lambda name, *a: None if name == "torch" else real(name, *a))
    code, out, _ = run(["doctor", "--json", "--database", tmp_path / "d"], capsys)
    report = json.loads(out)
    torch = next(c for c in report["checks"] if c["name"] == "dependency torch")
    assert code == 3 and not report["ready"]
    assert not torch["ok"] and torch["fix"] == "uv sync --frozen"
