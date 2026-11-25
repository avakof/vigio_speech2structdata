# Output-directory lock for saves and exports

## Defect

`save_run` committed its SQLite transaction, released it, and only then called
`write_views`, which swaps five files into the output directory one at a time.
Nothing serialized two writers of the same directory, and `export_run(out=...)`
read the run *before* writing, so it could write an older generation over a newer
one. Interleavings reproduced deterministically (events, no sleeps deciding the
outcome; `tests/persistence_own/test_own_out_dir_lock.py`):

| scenario | observed before the fix |
|---|---|
| A commits generation 1 (`error`) and pauses before its views; B commits generation 2 (`ok`) and writes views; A resumes | `run.json` says `error`, database current is `ok`, A's recovery copy is gone: stale, no pending recovery |
| A swaps 1 file and yields; B swaps 3; A swaps the rest; B finishes | directory holds B's `observations.csv`, A's `observations.txt`/`segments.csv`, B's `segments.txt`/`run.json`: a mixture |
| `export --out` reads generation 1, a newer save commits and writes, the export then writes | CSV files of generation 1 next to JSON/table of generation 2 |
| a save holds the directory; an explicit export of the same directory | the export read the database without waiting |

## Fix (`src/speechflow/persistence.py`)

* `_out_lock(out)`: exclusive, re-entrant `flock` on `OUT_LOCK_NAME`
  (`.speechflow.lock`) inside the output directory. Polls with a deadline
  (`OUT_LOCK_TIMEOUT_S`, 30 s) and then fails with a `PersistenceError`. The OS
  drops the lock when its holder dies, so a crash never wedges the directory.
  Different directories use different locks; no global lock exists. Lock order is
  always directory lock, then the SQLite write lock, so there is no cycle.
* `save_run`: recovery copy first (so a lock timeout or crash still leaves the
  result on disk), then lock, then `BEGIN IMMEDIATE` ... commit, view swap and
  recovery cleanup all under the lock. Views are rendered from the manifest
  stored by that transaction, so the last writer to finish is the last to commit.
* `export_run(..., out=...)`: loads the run inside the lock.
* `write_views` and `recover` take the same (re-entrant) lock.
* No bulk cleanup of any kind: the lock never deletes entries in the output
  directory. `write_views` removes only the stage directory it created itself
  (random `mkdtemp` name, `finally`). A stage left behind by a killed writer is
  private garbage that does not affect the transaction or later saves, and
  unrelated user entries such as `.export-family-photos` are never touched.
  (An earlier revision swept `.export-*` on entry; that was removed.)
* Lock ownership is thread-local and keyed by `(pid, path)`, so a forked child
  cannot re-enter its parent's critical section. The acquirer calls
  `flock(LOCK_UN)` before closing, only when `getpid()` still equals the acquiring
  pid, so a child holding an inherited copy of the descriptor does not prolong the
  lock after the parent leaves, and a forked copy of the frame cannot unlock the
  parent's lock.
* `_write_recovery` retries if a concurrent save removes the empty `.recovery`
  directory underneath it.

Public signatures, the migration, idempotent retry, parameterized SQL and crash
recovery are unchanged. New files in an output directory: `.speechflow.lock`
(persistent by design, never unlinked, so no unlink race).

## Evidence

Commands (isolated venv with pydantic/pytest/psutil, or the shared `.venv`):

```
pytest tests/persistence_own/test_own_out_dir_lock.py        # before the fix, x3
pytest tests/persistence_own                                  # after, x5
pytest -q                                                      # default suite, shared .venv
```

* Before: 5 of 7 new tests fail on all three runs (`raw/before-regression.txt`);
  the two that pass are the different-directory and many-threads cases (the latter
  by luck of scheduling).
* After (first revision): `tests/persistence_own` 67 passed on five consecutive runs;
  the new file, run six times in parallel
  to load the machine, passed every time (`raw/after-parallel-regression.txt`).
* Process level: a subprocess holds the lock and is killed; a save meanwhile times
  out without committing and leaves its recovery copy; after the kill the lock is
  free and `recover` commits. A `.export-crashed` leftover next to it is untouched.
* Follow-up (sentinels and fork), 3 of 9 tests failed before
  (`raw/before-followup.txt`): `.export-family-photos/`, `.export-notes` and
  `.export-/` survive save, explicit export, `write_views` and `recover`; a forked
  child blocks (does not re-enter) while the parent holds the lock, and the parent
  re-acquires immediately after leaving even though the live child still holds the
  inherited descriptor. After: `tests/persistence_own` 69 passed on three runs
  (`raw/after-persistence-suite.txt`), default suite in `raw/after-default-suite.txt`.

## Remaining issues

* Readers are not locked: a process reading several files of a directory while a
  writer swaps them can still see old and new files. Readers wanting a consistent
  set should take the same `flock` (shared) or use `export_run(out=None)`/`run.json`.
* `flock` is advisory and assumes a local filesystem; network filesystems may not
  honour it.
* `README.md` (not in this task's scope) should list `.speechflow.lock` among the
  files in an output directory and the 30 s lock timeout.
* Writers to one directory are serialized for the whole commit-plus-export, so a
  slow export delays other saves to that directory (not other directories).
