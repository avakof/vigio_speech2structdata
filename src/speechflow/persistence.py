"""SQLite history, recoverable exports and consistent JSON/CSV/table views.

Every view (``run.json``, CSV, terminal table, SQL views) derives from one
validated ``RunManifest``. ``save`` commits it in one transaction and then
renders the views from the manifest *stored* in the database, so an export of
a retried run is byte-identical to the first one.

Public entry points: ``save`` (frozen contract signature), ``save_run``,
``load_manifest``, ``export_run``, ``list_runs``, ``recover``, ``verify``.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import tempfile
import threading
import time
import unicodedata
import uuid
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from importlib import resources
from pathlib import Path
from typing import Any

from .contracts import RunManifest

MIN_SQLITE = (3, 38, 0)
BUSY_TIMEOUT_MS = 30_000
RECOVERY_DIR = ".recovery"
OUT_LOCK_NAME = ".speechflow.lock"
OUT_LOCK_TIMEOUT_S = 30.0
VIEW_FILES = {
    "csv": ("observations.csv", "segments.csv"),
    "table": ("observations.txt", "segments.txt"),
    "json": ("run.json",),
}
OBSERVATION_COLUMNS = (
    "ordinal", "observation_id", "revision", "status", "operation", "product_spoken",
    "catalog_match", "quantity_text", "quantity", "quantity_max", "qualifier",
    "unit_spoken", "unit", "packaging_count", "packaging_unit", "per_package_quantity",
    "per_package_unit", "packaging_total", "packaging_total_unit", "location",
    "correction_of", "corrected_by", "is_final", "evidence", "evidence_quotes", "errors",
)
SEGMENT_COLUMNS = (
    "ordinal", "segment_id", "status", "input_start", "core_start", "core_end",
    "input_end", "core_start_s", "core_end_s", "normalized_text", "errors",
)
# Cells of these columns are validated numbers/ids; every other CSV cell is text.
_NUMERIC = {
    "ordinal", "revision", "quantity", "quantity_max", "packaging_count",
    "per_package_quantity", "packaging_total", "input_start", "core_start", "core_end",
    "input_end", "core_start_s", "core_end_s",
}
_DECIMAL = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")


class PersistenceError(Exception):
    """A database or export operation failed. The message says what to do."""


class IntegrityError(PersistenceError):
    """The manifest violates a stored-history invariant. Nothing was written."""


class ConflictError(PersistenceError):
    """The database already holds a different result for this identity."""


class NotFoundError(PersistenceError):
    """No such database or run."""


@dataclass(frozen=True)
class SaveResult:
    run_id: str
    disposition: str  # inserted | duplicate | replaced
    save_id: int
    generation: int
    content_digest: str
    database: Path
    manifest: RunManifest
    files: list[Path] = field(default_factory=list)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


# --------------------------------------------------------------------- digests

def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_digest(manifest: RunManifest) -> str:
    """Digest of the deterministic result. Timings, provenance, raw model
    output and raw extractor replies are deliberately excluded so that a retry
    of the same processing version with identical transcript and observations
    is recognised as the same result."""
    source = manifest.source.model_dump(mode="json")
    source.pop("name")  # the same bytes may be re-imported under another name
    body = {
        "run_id": manifest.run_id,
        "processing_version": manifest.processing_version,
        "schema_version": manifest.schema_version,
        "mode": manifest.mode,
        "status": manifest.status,
        "transcription_status": manifest.transcription_status,
        "extraction_status": manifest.extraction_status,
        "source": source,
        "segments": [
            {k: v for k, v in s.model_dump(mode="json").items() if k != "raw"}
            for s in manifest.segments
        ],
        "observations": [o.model_dump(mode="json") for o in manifest.observations],
        "errors": manifest.errors,
    }
    return hashlib.sha256(_canonical(body).encode()).hexdigest()


def _manifest_text(manifest: RunManifest) -> str:
    return manifest.model_dump_json()


# ---------------------------------------------------------------- integrity

def check_manifest(manifest: RunManifest) -> None:
    """Raise IntegrityError listing every violated invariant."""
    problems: list[str] = []
    src = manifest.source
    if manifest.mode == "mic" and src.kind != "mic":
        problems.append("mode 'mic' requires a mic source")
    if manifest.mode != "mic" and src.kind != "file":
        problems.append(f"mode {manifest.mode!r} requires a file source")
    if manifest.extraction_status == "not_run" and manifest.observations:
        problems.append("extraction_status is not_run but observations are present")
    if manifest.mode == "transcribe" and (
        manifest.extraction_status != "not_run" or manifest.observations
    ):
        problems.append("a transcribe run must not carry extraction results")

    segments: dict[str, Any] = {}
    ordinals: set[int] = set()
    for seg in manifest.segments:
        if seg.segment_id in segments:
            problems.append(f"duplicate segment_id {seg.segment_id}")
        if seg.ordinal in ordinals:
            problems.append(f"duplicate segment ordinal {seg.ordinal}")
        if seg.source_id != src.source_id:
            problems.append(f"segment {seg.segment_id} belongs to another source")
        if not 0 <= seg.input_start <= seg.core_start < seg.core_end <= seg.input_end <= src.sample_count:
            problems.append(f"segment {seg.segment_id} exceeds source bounds or has invalid offsets")
        segments[seg.segment_id] = seg
        ordinals.add(seg.ordinal)

    observations: dict[str, Any] = {}
    for obs in manifest.observations:
        if obs.observation_id in observations:
            problems.append(f"duplicate observation_id {obs.observation_id}")
        observations[obs.observation_id] = obs
    for obs in manifest.observations:
        oid = obs.observation_id
        if obs.status == "ok" and not obs.evidence:
            problems.append(f"{oid}: accepted observation requires supporting transcript evidence")
        for field_name in ("quantity", "quantity_max"):
            value = getattr(obs, field_name)
            if value is not None and not _is_decimal(value):
                problems.append(f"{oid}: {field_name} {value!r} is not a base-ten decimal")
        pack = obs.packaging
        if pack is not None:
            for field_name in ("count", "per_package_quantity", "total"):
                value = getattr(pack, field_name)
                if value is not None and not _is_decimal(value):
                    problems.append(f"{oid}: packaging.{field_name} {value!r} is not a decimal")
        for span in obs.evidence:
            seg = segments.get(span.segment_id)
            if seg is None:
                problems.append(f"{oid}: evidence names unknown segment {span.segment_id}")
            elif span.end > len(seg.normalized_text) or (
                seg.normalized_text[span.start:span.end] != span.quote
            ):
                problems.append(f"{oid}: evidence quote does not match segment text at "
                                f"[{span.start}, {span.end})")
        if obs.correction_of is not None:
            if obs.correction_of == oid:
                problems.append(f"{oid}: corrects itself")
            elif obs.correction_of not in observations and obs.status == "ok":
                problems.append(f"{oid}: correction target is missing but status is ok")
        elif obs.operation == "correction" and obs.status == "ok":
            problems.append(f"{oid}: correction has no target but status is ok")
    for obs in manifest.observations:  # cycles
        seen, cur = set(), obs
        while cur is not None and cur.correction_of is not None:
            if cur.observation_id in seen:
                problems.append(f"{obs.observation_id}: correction chain is cyclic")
                break
            seen.add(cur.observation_id)
            cur = observations.get(cur.correction_of)
    if problems:
        raise IntegrityError("manifest violates stored-history invariants:\n  - "
                             + "\n  - ".join(problems))


def _is_decimal(value: str) -> bool:
    return bool(_DECIMAL.fullmatch(value))


# ------------------------------------------------------------------ database

def _open(database: Path, *, create: bool) -> sqlite3.Connection:
    if sqlite3.sqlite_version_info < MIN_SQLITE:
        raise PersistenceError(
            f"SQLite {sqlite3.sqlite_version} is too old; 3.38 or newer is required")
    database = Path(database)
    if not database.exists():
        if not create:
            raise NotFoundError(f"database {database} does not exist")
        database.parent.mkdir(parents=True, exist_ok=True)
        os.close(os.open(database, os.O_CREAT | os.O_RDWR, 0o600))
    try:
        conn = sqlite3.connect(database, timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None)
        conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA foreign_keys=ON")
        _retry_locked(lambda: conn.execute("PRAGMA journal_mode=WAL"))
        conn.execute("PRAGMA synchronous=FULL")
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        raise _wrap(exc, database) from exc
    return conn


def _retry_locked(action: Callable[[], Any]) -> Any:
    """Switching to WAL needs an exclusive lock and SQLite may not invoke the busy
    handler for it, so concurrent first opens retry until the timeout."""
    deadline = time.monotonic() + BUSY_TIMEOUT_MS / 1000
    while True:
        try:
            return action()
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc) or time.monotonic() > deadline:
                raise
            time.sleep(0.02)


def connect(database: Path, *, create: bool = True) -> sqlite3.Connection:
    """Open the database and apply pending migrations."""
    conn = _open(database, create=create)
    try:
        migrate(conn)
    except BaseException:
        conn.close()
        raise
    return conn


def _wrap(exc: sqlite3.Error, database: Path | None = None) -> PersistenceError:
    where = f" ({database})" if database else ""
    if "locked" in str(exc) or "busy" in str(exc):
        return PersistenceError(
            f"database is locked by another process{where}; retry the command. "
            "No partial data was written.")
    return PersistenceError(f"SQLite error{where}: {exc}")


def _migrations() -> list[tuple[int, str, str, str]]:
    found = []
    for entry in resources.files("speechflow.migrations").iterdir():
        match = re.fullmatch(r"(\d{4})_([a-z0-9_]+)\.sql", entry.name)
        if match:
            sql = entry.read_text(encoding="utf-8")
            found.append((int(match[1]), match[2], sql, hashlib.sha256(sql.encode()).hexdigest()))
    found.sort()
    if [m[0] for m in found] != list(range(1, len(found) + 1)):
        raise PersistenceError("migration files are not numbered contiguously from 0001")
    return found


def _statements(sql: str) -> Iterable[str]:
    buffer = ""
    for line in sql.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            yield buffer
            buffer = ""
    if buffer.strip() and not all(x.strip().startswith("--") or not x.strip()
                                  for x in buffer.splitlines()):
        raise PersistenceError("migration ends with an unterminated statement")


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations, each in its own transaction. Returns the version."""
    known = _migrations()
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations ("
                     "version INTEGER PRIMARY KEY, name TEXT NOT NULL, "
                     "checksum TEXT NOT NULL, applied_at TEXT NOT NULL) STRICT")
        for version, name, sql, checksum in known:
            if _applied(conn).get(version) == checksum:
                continue
            conn.execute("BEGIN IMMEDIATE")
            try:
                applied = _applied(conn)  # another process may have applied it
                if version not in applied:
                    for statement in _statements(sql):
                        conn.execute(statement)
                    conn.execute("INSERT INTO schema_migrations VALUES (?, ?, ?, ?)",
                                 (version, name, checksum, _now()))
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
        applied = _applied(conn)
    except sqlite3.Error as exc:
        raise _wrap(exc) from exc
    by_version = {v: c for v, _, _, c in known}
    for version, checksum in applied.items():
        if version not in by_version:
            raise PersistenceError(
                f"database schema version {version} is newer than this speechflow "
                f"({len(known)}); upgrade the application")
        if by_version[version] != checksum:
            raise PersistenceError(
                f"migration {version:04d} was modified after it was applied; restore it")
    return max(applied, default=0)


def _applied(conn: sqlite3.Connection) -> dict[int, str]:
    return {r["version"]: r["checksum"] for r in conn.execute(
        "SELECT version, checksum FROM schema_migrations")}


# --------------------------------------------------------------------- write

def _write_run(conn: sqlite3.Connection, manifest: RunManifest, digest: str,
               now: str) -> tuple[str, int, int, str]:
    """Insert inside an open transaction. Returns (disposition, save_id, generation, json)."""
    src = manifest.source
    row = conn.execute("SELECT kind, digest FROM sources WHERE source_id=?",
                       (src.source_id,)).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?,?)",
            (src.source_id, src.kind, src.digest, src.name, src.native_rate,
             src.native_channels, src.sample_rate, src.sample_count, src.channel_mix, now))
    elif (row["kind"], row["digest"]) != (src.kind, src.digest):
        raise ConflictError(f"source_id {src.source_id} is already stored with a different "
                            "kind or content digest")

    triple = conn.execute(
        "SELECT run_id FROM runs WHERE source_id=? AND mode=? AND processing_version=?",
        (src.source_id, manifest.mode, manifest.processing_version)).fetchone()
    if triple is not None and triple["run_id"] != manifest.run_id:
        raise ConflictError(
            f"source/mode/processing version already belongs to run {triple['run_id']}, "
            f"not {manifest.run_id}; run_id must be stable for a given identity")
    run = conn.execute("SELECT * FROM runs WHERE run_id=?", (manifest.run_id,)).fetchone()
    text = _manifest_text(manifest)
    manifest_digest = hashlib.sha256(text.encode()).hexdigest()

    if run is None:
        conn.execute("INSERT INTO runs (run_id, source_id, mode, processing_version, "
                     "schema_version, created_at) VALUES (?,?,?,?,?,?)",
                     (manifest.run_id, src.source_id, manifest.mode,
                      manifest.processing_version, manifest.schema_version, now))
        generation, disposition = 1, "inserted"
    else:
        if (run["source_id"], run["mode"], run["processing_version"]) != (
                src.source_id, manifest.mode, manifest.processing_version):
            raise ConflictError(f"run {manifest.run_id} exists with a different identity")
        current = conn.execute(
            "SELECT save_id, generation, content_digest, status, manifest_json "
            "FROM run_saves WHERE save_id=?", (run["current_save_id"],)).fetchone()
        if current["content_digest"] == digest:
            conn.execute("INSERT INTO save_events (run_id, save_id, disposition, "
                         "manifest_digest, event_at) VALUES (?,?,?,?,?)",
                         (manifest.run_id, current["save_id"], "duplicate", manifest_digest, now))
            return "duplicate", current["save_id"], current["generation"], current["manifest_json"]
        can_retry = current["status"] == "error"
        if current["status"] == "needs_review" and manifest.status in {"ok", "needs_review"}:
            previous = RunManifest.model_validate_json(current["manifest_json"])
            new_events = {o.observation_id: o for o in manifest.observations}
            improved = (manifest.status == "ok" or len(new_events) > len(previous.observations)
                        or set(manifest.errors) < set(previous.errors))
            can_retry = (improved and previous.segments == manifest.segments and all(
                o.observation_id in new_events and (o.status == "needs_review" or new_events[o.observation_id] == o)
                for o in previous.observations
            ))
        if not can_retry:
            raise ConflictError(
                f"run {manifest.run_id} is already stored with status {current['status']!r} "
                f"and different content (stored {current['content_digest'][:12]}, new "
                f"{digest[:12]}). Same processing version must reproduce the same result; "
                "change the model/schema/configuration to create a new processing version. "
                "The new result is kept in the recovery directory.")
        generation, disposition = current["generation"] + 1, "replaced"

    cursor = conn.execute(
        "INSERT INTO run_saves (run_id, generation, content_digest, manifest_digest, status, "
        "transcription_status, extraction_status, manifest_json, saved_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (manifest.run_id, generation, digest, manifest_digest, manifest.status,
         manifest.transcription_status, manifest.extraction_status, text, now))
    save_id = cursor.lastrowid
    _insert_children(conn, save_id, manifest)
    conn.execute("UPDATE runs SET current_save_id=? WHERE run_id=?", (save_id, manifest.run_id))
    conn.execute("INSERT INTO save_events (run_id, save_id, disposition, manifest_digest, "
                 "event_at) VALUES (?,?,?,?,?)",
                 (manifest.run_id, save_id, disposition, manifest_digest, now))
    return disposition, save_id, generation, text


def _insert_children(conn: sqlite3.Connection, save_id: int, manifest: RunManifest) -> None:
    conn.executemany(
        "INSERT INTO segments VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [(save_id, s.segment_id, s.source_id, s.ordinal, s.input_start, s.core_start,
          s.core_end, s.input_end, _canonical(s.raw), s.normalized_text, s.status,
          json.dumps(s.errors, ensure_ascii=False)) for s in manifest.segments])
    conn.executemany(
        "INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(save_id, o.observation_id, i, o.revision, o.status, o.operation, o.product_spoken,
          o.catalog_match, o.quantity_text, o.quantity, o.quantity_max, o.qualifier,
          o.unit_spoken, o.unit,
          None if o.packaging is None else _canonical(o.packaging.model_dump(mode="json")),
          o.location, o.correction_of, json.dumps(o.errors, ensure_ascii=False))
         for i, o in enumerate(manifest.observations)])
    conn.executemany(
        "INSERT INTO observation_evidence VALUES (?,?,?,?,?,?,?)",
        [(save_id, o.observation_id, j, e.segment_id, e.start, e.end, e.quote)
         for o in manifest.observations for j, e in enumerate(o.evidence)])


def _fsync_write(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


# ------------------------------------------------------------ output-directory lock

_held = threading.local()


@contextmanager
def _out_lock(out: Path) -> Iterator[None]:
    """Exclusive, re-entrant lock scoped to one output directory.

    Every save, explicit export and recovery that targets ``out`` runs under it,
    from the database transaction through the last file swap, so the order in
    which writers finish equals the order in which they committed and a
    directory never mixes generations. It is an ``flock`` on a file inside
    ``out``: the OS releases it when its holder dies, different directories use
    different locks, and nothing global is held. Lock order is always
    directory lock, then the SQLite write lock.

    Ownership is recorded per (pid, path) in thread-local state, so a forked
    child never mistakes the parent's critical section for its own. The
    acquirer unlocks explicitly before closing, because a child that inherited
    the descriptor would otherwise keep the (shared) lock alive after the
    parent has left.
    """
    out = Path(out)
    try:
        out.mkdir(parents=True, exist_ok=True)
        fd = os.open(out / OUT_LOCK_NAME, os.O_CREAT | os.O_RDWR, 0o600)
    except OSError as exc:
        raise PersistenceError(f"cannot lock output directory {out}: {exc}") from exc
    pid = os.getpid()
    key = (pid, os.path.realpath(out / OUT_LOCK_NAME))
    held: dict[tuple[int, str], int] = _held.__dict__.setdefault("paths", {})
    try:
        if key in held:  # this thread of this process owns it (recover -> save_run)
            held[key] += 1
            try:
                yield
            finally:
                held[key] -= 1
            return
        deadline = time.monotonic() + OUT_LOCK_TIMEOUT_S
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise PersistenceError(
                        f"output directory {out} is locked by another speechflow process; "
                        "retry once it finishes. Nothing was committed, and a copy of the "
                        f"result is in {out / RECOVERY_DIR} if it was written.") from None
                time.sleep(0.01)
            except OSError as exc:
                raise PersistenceError(f"cannot lock output directory {out}: {exc}") from exc
        held[key] = 1
        try:
            yield
        finally:
            held.pop(key, None)
            if os.getpid() == pid:  # a forked copy of this frame must not unlock the parent
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
    finally:
        os.close(fd)


def _write_recovery(out: Path, manifest: RunManifest, digest: str) -> Path:
    directory = Path(out) / RECOVERY_DIR
    for attempt in range(5):
        try:
            directory.mkdir(parents=True, exist_ok=True)
            final = directory / f"{manifest.run_id}.{digest[:16]}.{uuid.uuid4().hex[:8]}.json"
            tmp = final.with_suffix(".tmp")
            _fsync_write(tmp, manifest.model_dump_json(indent=2) + "\n")
            os.replace(tmp, final)
            return final
        except FileNotFoundError:
            if attempt == 4:  # another save removed the empty directory under us; retry
                raise PersistenceError(f"cannot write recovery copy under {directory}") from None
        except OSError as exc:
            raise PersistenceError(f"cannot write recovery copy under {directory}: {exc}") from exc
    raise AssertionError("unreachable")


def save_run(manifest: RunManifest, database: Path, out: Path, *,
             clock: Callable[[], str] = _now, recovery: Path | None = None) -> SaveResult:
    """Commit ``manifest`` to ``database`` and write the views into ``out``.

    Order: (1) recovery copy in ``out/.recovery``; (2) the output-directory lock,
    held through steps 3-5; (3) one IMMEDIATE transaction; (4) views rendered from
    the stored manifest and swapped in file by file; (5) recovery copy removed. A failure at any step leaves the recovery copy,
    and ``recover`` or ``export_run`` completes the work later.
    """
    manifest = RunManifest.model_validate_json(manifest.model_dump_json())
    out, database = Path(out), Path(database)
    digest = content_digest(manifest)
    # The recovery copy is written before waiting for the directory lock so that
    # a lock timeout or a crash anywhere below still leaves the result on disk.
    recovery = recovery or _write_recovery(out, manifest, digest)
    check_manifest(manifest)

    with _out_lock(out):  # one writer per directory, from commit to the last view
        conn = connect(database)
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                disposition, save_id, generation, text = _write_run(
                    conn, manifest, digest, clock())
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
        except sqlite3.Error as exc:
            raise _wrap(exc, database) from exc
        finally:
            conn.close()

        stored = RunManifest.model_validate_json(text)
        files = write_views(stored, out)
        recovery.unlink(missing_ok=True)
        try:
            recovery.parent.rmdir()  # only succeeds when no other copy is pending
        except OSError:
            pass
    return SaveResult(manifest.run_id, disposition, save_id, generation, digest,
                      database, stored, files)


def save(manifest: RunManifest, database: Path, out: Path) -> None:
    """Frozen contract signature: commit history and write JSON/CSV/table views."""
    save_run(manifest, database, out)


# ---------------------------------------------------------------------- read

def resolve_run_id(conn: sqlite3.Connection, run_id: str) -> str:
    exact = conn.execute("SELECT run_id FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if exact:
        return exact["run_id"]
    if len(run_id) >= 4:
        rows = conn.execute("SELECT run_id FROM runs WHERE substr(run_id, 1, ?) = ? "
                            "ORDER BY run_id LIMIT 5", (len(run_id), run_id)).fetchall()
        if len(rows) == 1:
            return rows[0]["run_id"]
        if rows:
            raise NotFoundError(f"run id prefix {run_id!r} is ambiguous: "
                                + ", ".join(r["run_id"][:24] for r in rows))
    raise NotFoundError(f"no run {run_id!r} in this database; list runs with `speechflow runs`")


def load_manifest(database: Path, run_id: str) -> RunManifest:
    """The stored manifest of a run's current save (one consistent snapshot)."""
    conn = connect(database, create=False)
    try:
        run_id = resolve_run_id(conn, run_id)
        row = conn.execute("SELECT s.manifest_json FROM runs r JOIN run_saves s "
                           "ON s.save_id = r.current_save_id WHERE r.run_id=?",
                           (run_id,)).fetchone()
        return RunManifest.model_validate_json(row["manifest_json"])
    except sqlite3.Error as exc:
        raise _wrap(exc, Path(database)) from exc
    finally:
        conn.close()


def list_runs(database: Path) -> list[dict[str, Any]]:
    conn = connect(database, create=False)
    try:
        rows = conn.execute(
            "SELECT c.run_id, c.mode, c.status, c.transcription_status, c.extraction_status, "
            "c.processing_version, c.generation, c.saved_at, s.name AS source_name, "
            "(SELECT count(*) FROM segments WHERE save_id=c.save_id) AS segments, "
            "(SELECT count(*) FROM observations WHERE save_id=c.save_id) AS observations "
            "FROM current_runs c JOIN sources s ON s.source_id=c.source_id "
            "ORDER BY c.saved_at, c.run_id").fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error as exc:
        raise _wrap(exc, Path(database)) from exc
    finally:
        conn.close()


def verify(database: Path) -> list[str]:
    """Return a list of problems (empty when the database is consistent)."""
    problems: list[str] = []
    conn = connect(database, create=False)
    try:
        problems += [f"integrity_check: {r[0]}" for r in conn.execute("PRAGMA integrity_check")
                     if r[0] != "ok"]
        problems += [f"foreign_key_check: {tuple(r)}" for r in conn.execute(
            "PRAGMA foreign_key_check")]
        for row in conn.execute("SELECT save_id, run_id, content_digest, manifest_digest, "
                                "manifest_json FROM run_saves"):
            manifest = RunManifest.model_validate_json(row["manifest_json"])
            if content_digest(manifest) != row["content_digest"]:
                problems.append(f"save {row['save_id']}: content digest mismatch")
            n_seg, n_obs = conn.execute(
                "SELECT (SELECT count(*) FROM segments WHERE save_id=?), "
                "(SELECT count(*) FROM observations WHERE save_id=?)",
                (row["save_id"], row["save_id"])).fetchone()
            if (n_seg, n_obs) != (len(manifest.segments), len(manifest.observations)):
                problems.append(f"save {row['save_id']}: rows do not match manifest")
        for row in conn.execute("SELECT run_id FROM runs WHERE current_save_id IS NULL"):
            problems.append(f"run {row['run_id']} has no current save")
    except sqlite3.Error as exc:
        raise _wrap(exc, Path(database)) from exc
    finally:
        conn.close()
    return problems


# --------------------------------------------------------------------- views

def _s(value: Any) -> str | None:
    return None if value is None else str(value)


def observation_rows(manifest: RunManifest) -> list[dict[str, str | None]]:
    corrected_by: dict[str, list[Any]] = {}
    for obs in manifest.observations:
        if obs.correction_of is not None:
            corrected_by.setdefault(obs.correction_of, []).append(obs)
    rows = []
    for i, o in enumerate(manifest.observations):
        pack = o.packaging
        later = corrected_by.get(o.observation_id, [])
        rows.append({
            "ordinal": str(i), "observation_id": o.observation_id, "revision": str(o.revision),
            "status": o.status, "operation": o.operation, "product_spoken": o.product_spoken,
            "catalog_match": o.catalog_match, "quantity_text": o.quantity_text,
            "quantity": o.quantity, "quantity_max": o.quantity_max, "qualifier": o.qualifier,
            "unit_spoken": o.unit_spoken, "unit": o.unit,
            "packaging_count": pack.count if pack else None,
            "packaging_unit": pack.unit if pack else None,
            "per_package_quantity": pack.per_package_quantity if pack else None,
            "per_package_unit": pack.per_package_unit if pack else None,
            "packaging_total": pack.total if pack else None,
            "packaging_total_unit": pack.total_unit if pack else None,
            "location": o.location, "correction_of": o.correction_of,
            "corrected_by": ";".join(k.observation_id for k in later) or None,
            "is_final": "false" if any(k.status == "ok" for k in later) else "true",
            "evidence": ";".join(f"{e.segment_id}:{e.start}-{e.end}" for e in o.evidence) or None,
            "evidence_quotes": " | ".join(e.quote for e in o.evidence) or None,
            "errors": " | ".join(o.errors) or None,
        })
    return rows


def segment_rows(manifest: RunManifest) -> list[dict[str, str | None]]:
    def seconds(samples: int) -> str:
        return f"{Decimal(samples) / Decimal(16000):.3f}"

    return [{
        "ordinal": str(s.ordinal), "segment_id": s.segment_id, "status": s.status,
        "input_start": str(s.input_start), "core_start": str(s.core_start),
        "core_end": str(s.core_end), "input_end": str(s.input_end),
        "core_start_s": seconds(s.core_start), "core_end_s": seconds(s.core_end),
        "normalized_text": s.normalized_text, "errors": " | ".join(s.errors) or None,
    } for s in manifest.segments]


def _csv_cell(column: str, value: str | None) -> str:
    if value is None:
        return ""
    if column not in _NUMERIC and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value  # neutralize spreadsheet formula injection in text cells
    return value


def render_csv(columns: Sequence[str], rows: Sequence[dict[str, str | None]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_csv_cell(c, row[c]) for c in columns])
    return buffer.getvalue()


def terminal_safe(text: str) -> str:
    """Escape control and format characters so transcript text cannot drive a terminal."""
    out = []
    for ch in text:
        if ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif unicodedata.category(ch) in ("Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"):
            out.append(f"\\u{ord(ch):04x}" if ord(ch) <= 0xFFFF else f"\\U{ord(ch):08x}")
        else:
            out.append(ch)
    return "".join(out)


def _width(text: str) -> int:
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in "WF"
               else 1 for c in text)


def _short(identifier: str | None) -> str | None:
    if identifier is None or "_" not in identifier:
        return identifier
    kind, _, rest = identifier.partition("_")
    return f"{kind[:3]}_{rest[:10]}"


def _pack(row: dict[str, str | None]) -> str | None:
    parts = []
    if row["packaging_count"] or row["packaging_unit"]:
        parts.append(" ".join(x for x in (row["packaging_count"], row["packaging_unit"]) if x))
    if row["per_package_quantity"] or row["per_package_unit"]:
        parts.append("x " + " ".join(x for x in (row["per_package_quantity"],
                                                 row["per_package_unit"]) if x))
    if row["packaging_total"] or row["packaging_total_unit"]:
        parts.append("= " + " ".join(x for x in (row["packaging_total"],
                                                 row["packaging_total_unit"]) if x))
    return " ".join(parts) or None


def render_table(headers: Sequence[str], rows: Sequence[Sequence[str | None]], *,
                 max_cell: int | None = 48) -> str:
    """Fixed-width table. Null is shown as ``-``; long cells end with ``…``."""
    def cell(value: str | None) -> str:
        text = "-" if value is None else terminal_safe(value)
        if max_cell and _width(text) > max_cell:
            while _width(text) > max_cell - 1:
                text = text[:-1]
            text += "…"
        return text

    body = [[cell(v) for v in row] for row in rows]
    widths = [max([_width(h)] + [_width(r[i]) for r in body]) for i, h in enumerate(headers)]

    def line(cells: Sequence[str]) -> str:
        return "  ".join(c + " " * (w - _width(c)) for c, w in zip(cells, widths)).rstrip()

    return "\n".join([line(list(headers)), line(["-" * w for w in widths]),
                      *[line(r) for r in body]]) + "\n"


def observations_table(manifest: RunManifest, *, max_cell: int | None = 48) -> str:
    rows = observation_rows(manifest)
    if not rows:
        return "(no observations)\n"
    headers = ("#", "obs", "status", "op", "product", "qty", "max", "qualifier", "unit",
               "packaging", "location", "final", "corrects", "errors")
    table = [(r["ordinal"], _short(r["observation_id"]), r["status"], r["operation"],
              r["product_spoken"], r["quantity"], r["quantity_max"], r["qualifier"], r["unit"],
              _pack(r), r["location"], r["is_final"], _short(r["correction_of"]), r["errors"])
             for r in rows]
    return render_table(headers, table, max_cell=max_cell)


def segments_table(manifest: RunManifest, *, max_cell: int | None = 48) -> str:
    rows = segment_rows(manifest)
    if not rows:
        return "(no segments)\n"
    headers = ("#", "segment", "status", "start_s", "end_s", "text", "errors")
    table = [(r["ordinal"], _short(r["segment_id"]), r["status"], r["core_start_s"],
              r["core_end_s"], r["normalized_text"], r["errors"]) for r in rows]
    return render_table(headers, table, max_cell=max_cell)


def render_views(manifest: RunManifest, formats: Sequence[str] = ("csv", "table", "json"),
                 ) -> dict[str, str]:
    """File name -> text for every requested format, all from ``manifest``."""
    views: dict[str, str] = {}
    for fmt in formats:
        if fmt == "csv":
            views["observations.csv"] = render_csv(OBSERVATION_COLUMNS, observation_rows(manifest))
            views["segments.csv"] = render_csv(SEGMENT_COLUMNS, segment_rows(manifest))
        elif fmt == "table":
            views["observations.txt"] = observations_table(manifest)
            views["segments.txt"] = segments_table(manifest)
        elif fmt == "json":
            views["run.json"] = manifest.model_dump_json(indent=2) + "\n"
        else:
            raise ValueError(f"unknown export format {fmt!r}")
    return views


def write_views(manifest: RunManifest, out: Path,
                formats: Sequence[str] = ("csv", "table", "json")) -> list[Path]:
    """Stage every file, then swap each in atomically. ``run.json`` goes last.

    Serialized per output directory with every other writer of that directory.
    """
    views = render_views(manifest, formats)
    names = sorted(views, key=lambda n: (n == "run.json", n))
    out = Path(out)
    with _out_lock(out):
        try:
            stage = Path(tempfile.mkdtemp(prefix=".export-", dir=out))
        except OSError as exc:
            raise PersistenceError(f"cannot write exports to {out}: {exc}") from exc
        try:
            for name in names:
                _fsync_write(stage / name, views[name])
            for name in names:
                os.replace(stage / name, out / name)
        except OSError as exc:
            raise PersistenceError(
                f"export to {out} failed ({exc}); the run is committed. Re-run "
                f"`speechflow export {manifest.run_id} --out {out}` to complete it") from exc
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    return [out / n for n in names]


def export_run(database: Path, run_id: str, fmt: str, *, part: str = "observations",
               out: Path | None = None) -> str | list[Path]:
    """Render a stored run. Returns text when ``out`` is None, else the files written."""
    if out is not None:
        # Read the run *inside* the directory lock: a save that committed while
        # we waited must not be overwritten by the older generation we would
        # otherwise have read first.
        with _out_lock(Path(out)):
            return write_views(load_manifest(database, run_id), Path(out), (fmt,))
    manifest = load_manifest(database, run_id)
    views = render_views(manifest, (fmt,))
    if fmt == "json":
        return views["run.json"]
    return views[f"{part}.{'csv' if fmt == 'csv' else 'txt'}"]


def recover(database: Path, out: Path) -> list[tuple[Path, str, str]]:
    """Replay recovery copies left in ``out`` by an interrupted save.

    Returns (file, outcome, detail); outcome is saved, duplicate, replaced or
    failed. A copy is removed only when it was committed or already stored.
    """
    results: list[tuple[Path, str, str]] = []
    directory = Path(out) / RECOVERY_DIR
    if not directory.is_dir():
        return results
    with _out_lock(Path(out)):  # one recoverer per directory; re-entered by save_run
        for path in sorted(directory.glob("*.json")):
            try:
                manifest = RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
                result = save_run(manifest, database, out, recovery=path)
            except (PersistenceError, ValueError, OSError) as exc:
                results.append((path, "failed", str(exc)))
                continue
            results.append((path, result.disposition, result.run_id))
    return results
