"""speechflow command line.

Only the standard library is imported at module level, so ``--help`` and
``doctor`` work before ``uv sync`` has installed the heavy dependencies and
before ``models prepare`` has fetched any asset. Pipeline, assets and
evaluation modules are imported when a command needs them.

Exit codes (stable; also documented in the README):
  0 ok | 1 unexpected error | 2 usage or configuration error
  3 missing dependency or asset | 4 audio input or device error
  5 database or export error | 10 no_speech | 11 needs_review
  12 run finished with status error | 130 interrupted
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

EXIT_OK, EXIT_INTERNAL, EXIT_USAGE, EXIT_DEPENDENCY = 0, 1, 2, 3
EXIT_INPUT, EXIT_PERSISTENCE = 4, 5
STATUS_EXIT = {"ok": 0, "no_speech": 10, "needs_review": 11, "error": 12}
EXIT_INTERRUPTED = 130

# Import name -> distribution name, for `doctor`.
REQUIRED_MODULES = {
    "hojo_asr": "hojo-asr", "torch": "torch", "torchaudio": "torchaudio",
    "transformers": "transformers", "huggingface_hub": "huggingface-hub", "numpy": "numpy",
    "silero_vad": "silero-vad", "sounddevice": "sounddevice",
    "imageio_ffmpeg": "imageio-ffmpeg", "pydantic": "pydantic", "psutil": "psutil",
}
INSTALL_HINT = "uv sync --frozen"
PREPARE_HINT = "uv run speechflow models prepare"


class CliError(Exception):
    def __init__(self, message: str, code: int = EXIT_INTERNAL):
        super().__init__(message)
        self.code = code


def _safe(text: object) -> str:
    """Escape control characters so untrusted text cannot drive the terminal."""
    return "".join(
        ch if unicodedata.category(ch)[0] != "C" or ch == " " else f"\\u{ord(ch):04x}"
        for ch in str(text))


def _emit(text: str = "", *, err: bool = False) -> None:
    print(text, file=sys.stderr if err else sys.stdout)


# ------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    S = argparse.SUPPRESS
    common = argparse.ArgumentParser(add_help=False)
    group = common.add_argument_group("common options")
    group.add_argument("--config", type=Path, default=S, metavar="FILE",
                       help="TOML settings file (default: $SPEECHFLOW_CONFIG, else built-in)")
    group.add_argument("--offline", action="store_true", default=S,
                       help="never download; report missing assets instead")
    group.add_argument("--backend", default=S, metavar="{cpu,mps}",
                       help="ASR backend (overrides the settings file)")
    group.add_argument("--database", type=Path, default=S, metavar="FILE",
                       help="SQLite history database (overrides the settings file)")
    group.add_argument("--json", action="store_true", default=S,
                       help="print one machine-readable JSON document on stdout")
    group.add_argument("-q", "--quiet", action="store_true", default=S,
                       help="print only the one-line result")
    group.add_argument("--debug", action="store_true", default=S,
                       help="show tracebacks for unexpected errors")

    parser = argparse.ArgumentParser(
        prog="speechflow", parents=[common],
        description="Local French speech -> structured inventory data.",
        epilog="Exit codes: 0 ok, 1 internal error, 2 usage/config, 3 missing dependency or "
               "asset, 4 audio input/device, 5 database/export, 10 no_speech, "
               "11 needs_review, 12 run status error, 130 interrupted.")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    sub.add_parser("doctor", parents=[common], help="check machine, dependencies and assets "
                   "(downloads nothing)")
    models = sub.add_parser("models", parents=[common], help="model asset management")
    msub = models.add_subparsers(dest="models_command", metavar="ACTION", required=True)
    msub.add_parser("prepare", parents=[common],
                    help="download and verify pinned model assets (needs network unless "
                         "--offline)")

    for name, summary in (("transcribe", "audio file -> transcript only"),
                          ("run", "audio file -> transcript -> validated observations")):
        p = sub.add_parser(name, parents=[common], help=summary)
        p.add_argument("path", type=Path, help="audio file (WAV, FLAC, MP3, M4A)")
        p.add_argument("--out", type=Path, default=Path("outputs/run"),
                       help="directory for JSON/CSV/table views (default: %(default)s)")
    mic = sub.add_parser("mic", parents=[common],
                         help="segmented microphone transcription and extraction")
    mic.add_argument("--out", type=Path, default=Path("outputs/live"),
                     help="directory for views and capture spool (default: %(default)s)")
    mic.add_argument("--seconds", type=float, default=None, metavar="N",
                     help="stop capturing after N seconds (default: until interrupted)")
    mic.add_argument("--device", default=None, help="input device name or index")

    ev = sub.add_parser("evaluate", parents=[common], help="score a JSONL evaluation manifest")
    ev.add_argument("manifest", type=Path, help="evaluation manifest (.jsonl)")
    ev.add_argument("--out", type=Path, default=Path("outputs/evaluation"),
                    help="directory for evaluation results (default: %(default)s)")

    ex = sub.add_parser("export", parents=[common], help="re-render a stored run")
    ex.add_argument("run_id", help="run id or unique prefix (see `speechflow runs`)")
    ex.add_argument("--format", choices=("json", "csv", "table"), default="csv")
    ex.add_argument("--part", choices=("observations", "segments"), default="observations",
                    help="what to print to stdout for csv/table (default: %(default)s)")
    ex.add_argument("--out", type=Path, default=None,
                    help="write the format's files into this directory instead of stdout")

    sub.add_parser("runs", parents=[common], help="list stored runs")
    rec = sub.add_parser("recover", parents=[common],
                         help="commit recovery copies left by an interrupted save")
    rec.add_argument("--out", type=Path, default=Path("outputs/run"),
                     help="output directory that holds .recovery/ (default: %(default)s)")
    return parser


# --------------------------------------------------------------- settings

def _config_error(exc: Exception) -> CliError:
    errors = getattr(exc, "errors", None)
    if callable(errors):
        detail = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'settings'}: {e['msg']}"
                           for e in errors())
    else:
        detail = str(exc)
    return CliError(f"invalid configuration: {_safe(detail)}", EXIT_USAGE)


def _settings(args: argparse.Namespace):
    try:
        from .config import load_settings
    except ImportError as exc:
        raise CliError(f"cannot load configuration: {exc}. Install dependencies first: "
                       f"{INSTALL_HINT}", EXIT_DEPENDENCY) from exc
    overrides: dict[str, Any] = {
        "offline": True if getattr(args, "offline", False) else None,
        "backend": getattr(args, "backend", None),
        "database": getattr(args, "database", None),
    }
    if args.command == "mic":
        device = args.device
        if device is not None and re.fullmatch(r"\d+", device):
            device = int(device)
        overrides.update(mic_seconds=args.seconds, mic_device=device)
    try:
        return load_settings(getattr(args, "config", None), **overrides)
    except (OSError, ValueError) as exc:  # missing file, bad TOML, pydantic ValidationError
        raise _config_error(exc) from exc


# ----------------------------------------------------------------- doctor

def _pins() -> dict[str, str]:
    try:
        requires = importlib.metadata.requires("speechflow") or []
    except importlib.metadata.PackageNotFoundError:
        return {}
    pins = {}
    for req in requires:
        match = re.match(r"([A-Za-z0-9_.-]+)==([^\s;]+)", req)
        if match:
            pins[match[1].lower().replace("_", "-")] = match[2]
    return pins


def _host_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, *, required: bool = True, fix: str = "") -> None:
        checks.append({"name": name, "ok": ok, "required": required, "detail": detail,
                       "fix": fix})

    v = sys.version_info
    add("python", (3, 12) <= v[:2] < (3, 13), f"{v.major}.{v.minor}.{v.micro}",
        fix="uv python install 3.12 (the project pins 3.12)")
    mac = platform.mac_ver()[0]
    add("platform", sys.platform == "darwin" and platform.machine() == "arm64",
        f"{platform.system()} {mac or platform.release()} {platform.machine()}",
        fix="the pinned environment targets native Apple Silicon macOS")
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        add("memory", True, f"{ram / 2**30:.1f} GiB physical", required=False)
    except (ValueError, OSError, AttributeError):
        add("memory", True, "unknown", required=False)
    if sys.platform == "darwin":
        try:
            chip = subprocess.run(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=5).stdout.strip()
            add("chip", True, chip or "unknown", required=False)
        except (OSError, subprocess.SubprocessError):
            pass
    pins = _pins()
    for module, dist in REQUIRED_MODULES.items():
        try:
            present = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            present = False
        if not present:
            add(f"dependency {dist}", False, "not installed", fix=INSTALL_HINT)
            continue
        try:
            installed = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            installed = "unknown"
        pin = pins.get(dist)
        if pin and installed != pin:
            add(f"dependency {dist}", False, f"{installed} installed, {pin} pinned",
                fix=INSTALL_HINT)
        else:
            add(f"dependency {dist}", True, installed + (" (pinned)" if pin else ""))
    return checks


def _settings_checks(args: argparse.Namespace, checks: list[dict[str, Any]]):
    def add(name: str, ok: bool, detail: str, *, required: bool = True, fix: str = "") -> None:
        checks.append({"name": name, "ok": ok, "required": required, "detail": detail,
                       "fix": fix})

    try:
        settings = _settings(args)
    except CliError as exc:
        add("configuration", False, str(exc),
            fix="fix the --config file" if exc.code == EXIT_USAGE else INSTALL_HINT)
        checks[-1]["code"] = exc.code
        return None
    add("configuration", True, f"offline={settings.offline} backend={settings.backend}",
        required=False)
    probe = settings.database
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    add("database location", os.access(probe, os.W_OK),
        f"{settings.database} (writable ancestor {probe})", fix="choose --database elsewhere")
    free = shutil.disk_usage(probe).free / 2**30
    add("disk free", True, f"{free:.1f} GiB near the database", required=False)
    if settings.database.exists():
        try:
            from . import persistence
            problems = persistence.verify(settings.database)
            add("database", not problems, "consistent" if not problems else "; ".join(problems),
                fix="restore from backup; see README 'Database'")
        except Exception as exc:  # report, never crash the diagnostic
            add("database", False, _safe(exc), fix="see README 'Database'")
    else:
        add("database", True, "not created yet (created on first save)", required=False)
    return settings


def _assets_section(settings) -> tuple[dict[str, Any], bool]:
    try:
        from . import assets
    except ImportError as exc:
        return ({"available": False,
                 "reason": f"assets module cannot be imported: {_safe(exc)}",
                 "fix": INSTALL_HINT}, False)
    try:
        report = assets.doctor(settings)
    except Exception as exc:
        return ({"available": False, "reason": f"assets.doctor failed: {_safe(exc)}"}, False)
    # assets.doctor returns a flat dict without a verdict. A non-empty *_issues,
    # *_error(s) or missing entry means something is absent or unreachable.
    ok = isinstance(report, dict) and report.get("ok", True) is not False and not any(
        value and key.endswith(("_issues", "_error", "_errors", "missing", "issues", "errors"))
        for key, value in report.items())
    return {"available": True, "report": report}, ok


def cmd_doctor(args: argparse.Namespace) -> int:
    checks = _host_checks()
    settings = _settings_checks(args, checks)
    if settings is not None:
        assets_report, assets_ok = _assets_section(settings)
    else:
        assets_report, assets_ok = {"available": False, "reason": "configuration could not be loaded"}, False
    failed = [c for c in checks if c["required"] and not c["ok"]]
    ready = not failed and assets_ok
    if getattr(args, "json", False):
        _emit(json.dumps({"ready": ready, "checks": checks, "assets": assets_report},
                         indent=2, ensure_ascii=False, default=str))
    else:
        for c in checks:
            mark = "ok  " if c["ok"] else ("FAIL" if c["required"] else "warn")
            _emit(f"[{mark}] {c['name']}: {_safe(c['detail'])}")
            if not c["ok"] and c["fix"]:
                _emit(f"       fix: {c['fix']}")
        if assets_report["available"]:
            _emit(f"[{'ok  ' if assets_ok else 'FAIL'}] assets and services:")
            for key, value in assets_report["report"].items():
                _emit(f"       {key}: " + _safe(json.dumps(value, ensure_ascii=False,
                                                            default=str)))
            if not assets_ok:
                _emit(f"       fix: {PREPARE_HINT} (also pulls the pinned extractor through the "
                      "local Ollama service, which must be running)")
        else:
            _emit(f"[FAIL] assets and services: not checked, {_safe(assets_report['reason'])}")
        _emit("result: " + ("READY" if ready else "NOT READY"))
    if settings is None and any(c.get("code") == EXIT_USAGE for c in checks):
        return EXIT_USAGE  # an invalid settings file is a usage error, not a missing asset
    return EXIT_OK if ready else EXIT_DEPENDENCY


def cmd_models_prepare(args: argparse.Namespace) -> int:
    settings = _settings(args)
    try:
        from . import assets
    except ImportError as exc:
        raise CliError(f"cannot prepare models: {exc}. Install dependencies first "
                       f"({INSTALL_HINT}), then run `{PREPARE_HINT}`.", EXIT_DEPENDENCY) from exc
    try:
        report = assets.prepare(settings)
    except Exception as exc:  # prepare raises with one line per missing or unverifiable asset
        suffix = " (offline mode never downloads)" if settings.offline else ""
        raise CliError(f"models are not ready{suffix}:\n  " + "\n  ".join(_safe(line) for line in str(exc).splitlines())
                       + "\nInstall dependencies with `uv sync --frozen`; this command "
                       "needs network access for the first download, and the local Ollama "
                       "service for the extractor model.", EXIT_DEPENDENCY) from exc
    _emit(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return EXIT_OK


# ------------------------------------------------------------------ runs

def _check_input(path: Path) -> Path:
    if not path.exists():
        raise CliError(f"audio file not found: {_safe(path)}", EXIT_INPUT)
    if not path.is_file():
        raise CliError(f"not a regular file: {_safe(path)}", EXIT_INPUT)
    if not os.access(path, os.R_OK):
        raise CliError(f"audio file is not readable: {_safe(path)}", EXIT_INPUT)
    if path.stat().st_size == 0:
        raise CliError(f"audio file is empty: {_safe(path)}", EXIT_INPUT)
    return path


def _process_and_save(args: argparse.Namespace, mode: str, path: Path | None) -> int:
    settings = _settings(args)
    out = Path(args.out)
    try:
        from . import persistence
        pipeline = importlib.import_module(".pipeline", __package__)
    except ImportError as exc:
        raise CliError(f"cannot start the pipeline: {_safe(exc)}. Install dependencies "
                       f"({INSTALL_HINT}); prepare models with `{PREPARE_HINT}`.",
                       EXIT_DEPENDENCY) from exc
    manifest = pipeline.process(path, mode=mode, settings=settings, out=out)
    result = persistence.save_run(manifest, settings.database, out)
    return _report_run(args, result, persistence)


def _report_run(args: argparse.Namespace, result, persistence) -> int:
    m = result.manifest
    code = STATUS_EXIT[m.status]
    review = sum(o.status == "needs_review" for o in m.observations)
    summary = {
        "run_id": m.run_id, "mode": m.mode, "status": m.status,
        "transcription_status": m.transcription_status,
        "extraction_status": m.extraction_status, "processing_version": m.processing_version,
        "source": {"source_id": m.source.source_id, "kind": m.source.kind, "name": m.source.name,
                   "digest": m.source.digest},
        "segments": len(m.segments), "observations": len(m.observations),
        "observations_needing_review": review, "errors": m.errors,
        "database": str(result.database), "disposition": result.disposition,
        "generation": result.generation, "files": [str(f) for f in result.files],
        "exit_code": code,
    }
    if getattr(args, "json", False):
        _emit(json.dumps(summary, indent=2, ensure_ascii=False))
        return code
    _emit(f"run_id: {m.run_id}")
    _emit(f"status: {m.status} (transcription: {m.transcription_status}, "
          f"extraction: {m.extraction_status})")
    if getattr(args, "quiet", False):
        return code
    _emit(f"source: {_safe(m.source.name)} [{m.source.kind}] sha256 {m.source.digest[:16]}")
    _emit(f"processing_version: {m.processing_version}")
    _emit(f"segments: {len(m.segments)}  observations: {len(m.observations)} "
          f"({review} need review)")
    for error in m.errors:
        _emit(f"error: {_safe(error)}")
    _emit(f"database: {result.database} ({result.disposition}, generation {result.generation})")
    for f in result.files:
        _emit(f"wrote: {f}")
    _emit()
    _emit((persistence.segments_table(m) if m.mode == "transcribe"
           else persistence.observations_table(m)).rstrip("\n"))
    return code


def cmd_file(args: argparse.Namespace) -> int:
    return _process_and_save(args, args.command, _check_input(args.path))


def cmd_mic(args: argparse.Namespace) -> int:
    return _process_and_save(args, "mic", None)


def cmd_evaluate(args: argparse.Namespace) -> int:
    settings = _settings(args)
    if not args.manifest.is_file():
        raise CliError(f"evaluation manifest not found: {_safe(args.manifest)}", EXIT_INPUT)
    try:
        from . import evaluation
    except ImportError as exc:
        raise CliError(f"cannot evaluate: {_safe(exc)}. Install dependencies ({INSTALL_HINT}).",
                       EXIT_DEPENDENCY) from exc
    report = evaluation.evaluate(args.manifest, args.out, settings)
    _emit(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    status = report.get("status") if isinstance(report, dict) else None
    return STATUS_EXIT.get(status, EXIT_OK) if isinstance(status, str) else EXIT_OK


# -------------------------------------------------------- stored history

def _database(args: argparse.Namespace) -> Path:
    return _settings(args).database


def cmd_export(args: argparse.Namespace) -> int:
    from . import persistence
    result = persistence.export_run(_database(args), args.run_id, args.format,
                                    part=args.part, out=args.out)
    if isinstance(result, str):
        sys.stdout.write(result)
    else:
        for f in result:
            _emit(f"wrote: {f}")
    return EXIT_OK


def cmd_runs(args: argparse.Namespace) -> int:
    from . import persistence
    runs = persistence.list_runs(_database(args))
    if getattr(args, "json", False):
        _emit(json.dumps(runs, indent=2, ensure_ascii=False))
    elif not runs:
        _emit("(no runs)")
    else:
        cols = ("run_id", "mode", "status", "source_name", "segments", "observations",
                "generation", "saved_at")
        _emit(persistence.render_table(
            cols, [[str(r[c]) for c in cols] for r in runs], max_cell=None).rstrip("\n"))
    return EXIT_OK


def cmd_recover(args: argparse.Namespace) -> int:
    from . import persistence
    settings = _settings(args)
    results = persistence.recover(settings.database, args.out)
    for path, outcome, detail in results:
        _emit(f"{outcome}: {path.name}: {_safe(detail)}")
    if not results:
        _emit("nothing to recover")
    return EXIT_PERSISTENCE if any(r[1] == "failed" for r in results) else EXIT_OK


HANDLERS = {
    "doctor": cmd_doctor, "transcribe": cmd_file, "run": cmd_file, "mic": cmd_mic,
    "evaluate": cmd_evaluate, "export": cmd_export, "runs": cmd_runs, "recover": cmd_recover,
}


def _exit_code_for(exc: BaseException) -> int:
    module = type(exc).__module__
    if module.startswith("speechflow.persistence"):
        return EXIT_PERSISTENCE
    if isinstance(exc, ImportError) or module.startswith("speechflow.assets"):
        return EXIT_DEPENDENCY
    if module.startswith("speechflow.audio"):
        return EXIT_INPUT
    return EXIT_INTERNAL


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = (cmd_models_prepare if args.command == "models" else HANDLERS[args.command])
    try:
        return handler(args)
    except KeyboardInterrupt:
        _emit("interrupted; nothing was saved by this command. Any recovery copy under "
              "OUT/.recovery can be committed with `speechflow recover`.", err=True)
        return EXIT_INTERRUPTED
    except CliError as exc:
        _emit(f"error: {exc}", err=True)
        return exc.code
    except Exception as exc:
        if getattr(args, "debug", False):
            traceback.print_exc()
        _emit(f"error: {_safe(exc)}", err=True)
        if isinstance(exc, ImportError):
            _emit(f"install dependencies with `{INSTALL_HINT}`", err=True)
        if _exit_code_for(exc) == EXIT_PERSISTENCE:
            _emit("the run was not lost if a copy exists under OUT/.recovery; "
                  "commit it with `speechflow recover --out OUT`", err=True)
        return _exit_code_for(exc)


if __name__ == "__main__":
    sys.exit(main())
