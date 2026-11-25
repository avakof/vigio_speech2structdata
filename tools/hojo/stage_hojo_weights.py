#!/usr/bin/env python3
"""Resume the pinned Hojo weights from an APFS clone of a partial Xet file."""

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import ctypes
import errno
import fcntl
from hashlib import sha256
from http.client import HTTPException
import json
import os
from pathlib import Path
import re
import socket
import time
from urllib.error import HTTPError

from download_torch_wheel import digest, read_range_prefix

ROOT = Path(__file__).resolve().parents[2]
MODEL = json.loads((ROOT / "src/speechflow/model_manifest.json").read_text())["hojo"]
NAME = "merged_full_model.safetensors"
SIZE, _, HASH = MODEL["files"][NAME]
URL = f"https://huggingface.co/{MODEL['repository']}/resolve/{MODEL['revision']}/{NAME}"
BUNDLE = Path("~/.cache/speechflow/assets/hojo").expanduser()
STAGE_DIR = Path("~/.cache/speechflow/hojo-weight-range-stage").expanduser()
PART_BYTES = 1024 * 1024
MAX_ATTEMPTS = 4


def emit(event, **data):
    print(json.dumps({"event": event, **data}), flush=True)


def source_signature(path):
    st = path.stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_blocks


def clone_snapshot(source, target):
    if not source.is_file():
        raise OSError(f"Expected a regular sparse source at {source}")
    before = source_signature(source)
    if before[2] > SIZE:
        raise OSError("The source is larger than the pinned model")
    clonefile = ctypes.CDLL(None, use_errno=True).clonefile
    clonefile.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32)
    clonefile.restype = ctypes.c_int
    if clonefile(os.fsencode(source), os.fsencode(target), 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))
    try:
        unchanged = source_signature(source) == before
    except OSError:
        unchanged = False
    if not unchanged:
        target.unlink()
        raise OSError("Xet changed the source during the clone. Retry on a stable source.")
    return before


def part_bounds(index):
    start = index * PART_BYTES
    return start, min(start + PART_BYTES, SIZE)


def sparse_plan(path):
    reused = set()
    extent_count = 0
    extent_bytes = 0
    total = min(path.stat().st_size, SIZE)
    guard = os.statvfs(path).f_frsize
    with path.open("rb") as source:
        fd = source.fileno()
        offset = 0
        while offset < total:
            try:
                start = os.lseek(fd, offset, os.SEEK_DATA)
            except OSError as error:
                if error.errno == errno.ENXIO:
                    break
                raise
            end = min(os.lseek(fd, start, os.SEEK_HOLE), total)
            if not offset <= start < end:
                raise OSError("Unexpected APFS extent boundary")
            extent_count += 1
            extent_bytes += end - start
            first = (start + guard + PART_BYTES - 1) // PART_BYTES
            last = (end - guard) // PART_BYTES
            reused.update(range(first, last))
            offset = end
    return reused, {
        "extent_count": extent_count,
        "extent_bytes": extent_bytes,
        "guard_bytes_each_edge": guard,
        "reused_parts": len(reused),
        "reused_bytes": len(reused) * PART_BYTES,
        "required_download_bytes": SIZE - len(reused) * PART_BYTES,
    }


def save_state(path, state, stage_file):
    os.fsync(stage_file.fileno())
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(state, output, separators=(",", ":"), sort_keys=True)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def verify_state(state, file):
    stat = os.fstat(file.fileno())
    if (state["asset_sha256"], state["part_bytes"], state["stage_id"], stat.st_size) != (
        HASH, PART_BYTES, [stat.st_dev, stat.st_ino], SIZE
    ):
        raise OSError("Staged file identity or pinned transfer metadata changed")
    downloaded = {}
    for index_text, expected in state["downloaded"].items():
        index = int(index_text)
        start, end = part_bounds(index)
        actual = os.pread(file.fileno(), end - start, start)
        if len(actual) == end - start and sha256(actual).hexdigest() == expected:
            downloaded[index_text] = expected
        else:
            emit("invalid_checkpoint_part", index=index)
    state["downloaded"] = downloaded


def transfer(stage_file, state_path, state, pending, workers, request_bytes):
    attempts = {}
    prefixes = {}
    inflight = {}
    fatal = None
    hold_until = 0.0
    last_decrease = 0.0
    checkpoint_at = time.perf_counter()
    checkpoint_bytes = 0
    checkpoint_parts = 0
    fd = stage_file.fileno()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while inflight or (pending and fatal is None):
            now = time.perf_counter()
            while pending and len(inflight) < workers and not fatal and now >= hold_until:
                index = pending.popleft()
                base, end = part_bounds(index)
                start = base + len(prefixes.get(index, b""))
                stop = min(start + request_bytes, end)
                job = pool.submit(read_range_prefix, URL, SIZE, start, stop - 1)
                inflight[job] = index, stop - start
            if not inflight:
                time.sleep(max(0, min(1, hold_until - now)))
                continue

            done, _ = wait(inflight, timeout=2, return_when=FIRST_COMPLETED)
            for job in done:
                index, requested = inflight.pop(job)
                status = None
                retry_delay = 4
                try:
                    block, timed_out = job.result()
                except (OSError, HTTPException) as error:
                    block = b""
                    attempts[index] = attempts.get(index, 0) + 1
                    status = error.code if isinstance(error, HTTPError) else None
                    timed_out = isinstance(error, TimeoutError) or isinstance(
                        getattr(error, "reason", None), socket.timeout
                    )
                    if status == 429:
                        retry_after = error.headers.get("Retry-After", "")
                        if retry_after.isdecimal():
                            retry_delay = max(retry_delay, int(retry_after))
                    detail = re.sub(r"https?://\S+", "<url redacted>", str(error))
                    emit("range_retry", index=index, attempt=attempts[index], http=status, detail=detail)
                    if attempts[index] >= MAX_ATTEMPTS:
                        fatal = f"Stopped after {MAX_ATTEMPTS} attempts without new bytes for part {index}"
                now = time.perf_counter()
                if status == 429 or timed_out:
                    hold_until = max(hold_until, now + retry_delay)
                    if now - last_decrease >= 10:
                        workers = max(4, workers // 2)
                        last_decrease = now
                        emit("reduced_concurrency", workers=workers, retry_delay_seconds=retry_delay)
                if block and len(block) < requested:
                    request_bytes = max(256 * 1024, request_bytes // 2)
                    emit("range_prefix", index=index, received=len(block), requested=requested,
                         next_request_kib=request_bytes // 1024)
                if block:
                    prefixes.setdefault(index, bytearray()).extend(block)
                    attempts.pop(index, None)
                start, end = part_bounds(index)
                if len(prefixes.get(index, b"")) != end - start:
                    if fatal is None:
                        pending.appendleft(index)
                    continue
                part = prefixes.pop(index)
                view = memoryview(part)
                while view:
                    written = os.pwrite(fd, view, start)
                    if written <= 0:
                        raise OSError(f"Short write for model part {index}")
                    start += written
                    view = view[written:]
                state["downloaded"][str(index)] = sha256(part).hexdigest()
                checkpoint_bytes += len(part)
                checkpoint_parts += 1
            seconds = time.perf_counter() - checkpoint_at
            if checkpoint_parts and (checkpoint_parts >= 8 or seconds >= 5 or fatal or not (pending or inflight)):
                save_state(state_path, state, stage_file)
                emit("checkpoint", parts=checkpoint_parts, received=checkpoint_bytes, wall_seconds=seconds,
                     bytes_per_second=checkpoint_bytes / seconds, workers=workers,
                     pending=len(pending), in_flight=len(inflight),
                     downloaded_parts=len(state["downloaded"]))
                checkpoint_at = time.perf_counter()
                checkpoint_bytes = checkpoint_parts = 0
    if fatal:
        raise RuntimeError(fatal)


def verify_file(path):
    if path.stat().st_size != SIZE or digest(path) != HASH:
        raise OSError(f"Staged weight verification failed. Stage and original Xet cache retained at {path}.")


def run(args):
    source = args.source.expanduser().resolve()
    directory = args.directory.expanduser().resolve()
    if args.action == "plan":
        _, plan = sparse_plan(source)
        emit("plan", source=str(source), **plan)
        return
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "transfer.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(args, source, directory)


def run_locked(args, source, directory):
    staging = directory / f"{NAME}.staging"
    verified = directory / f"{NAME}.verified"
    state_path = directory / "state.json"
    if not verified.exists():
        if not state_path.exists():
            if args.action == "verify-state":
                raise OSError("No staged checkpoint exists")
            if staging.exists():
                raise OSError(f"Existing stage lacks its checkpoint metadata at {staging}")
            original = clone_snapshot(source, staging)
            reused, plan = sparse_plan(staging)
            with staging.open("r+b") as file:
                file.truncate(SIZE)
                stat = os.fstat(file.fileno())
                state = {"asset_sha256": HASH, "part_bytes": PART_BYTES, "source": str(source),
                         "source_signature": original, "stage_id": [stat.st_dev, stat.st_ino],
                         "reused": sorted(reused), "downloaded": {}}
                save_state(state_path, state, file)
            emit("clone", source=str(source), stage=str(staging), **plan)

        state = json.loads(state_path.read_text())
        with staging.open("r+b") as file:
            verify_state(state, file)
            reused = set(state["reused"])
            if args.action == "verify-state":
                emit("verified_checkpoint", reused_parts=len(reused),
                     downloaded_parts=len(state["downloaded"]), source_signature=state["source_signature"])
                return
            if args.refetch_interiors and reused:
                state["reused"] = []
                reused.clear()
                save_state(state_path, state, file)
            missing = [
                index for index, _ in enumerate(range(0, SIZE, PART_BYTES))
                if index not in reused and str(index) not in state["downloaded"]
            ]
            limit = len(missing) if not args.limit_mib else args.limit_mib * 1024 * 1024 // PART_BYTES
            transfer(
                file, state_path, state, deque(missing[:limit]), args.workers,
                args.request_kib * 1024,
            )
            if len(missing) > limit:
                return
        verify_file(staging)
        staging.replace(verified)
        emit("verified_weight", sha256=HASH, bytes=SIZE, path=str(verified))
    else:
        verify_file(verified)
    if args.publish:
        destination = BUNDLE / NAME
        # An atomic hard link publishes complete bytes and fails if another file already exists.
        os.link(verified, destination)
        emit("published", sha256=HASH, bytes=SIZE, path=str(destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original sparse Xet file")
    parser.add_argument("--directory", type=Path, default=STAGE_DIR)
    parser.add_argument("--action", choices=("plan", "transfer", "verify-state"), default="plan")
    parser.add_argument("--workers", type=int, choices=range(1, 49), default=24, metavar="1-48")
    parser.add_argument("--request-kib", type=int, choices=(256, 512, 1024), default=512)
    parser.add_argument("--limit-mib", type=int, choices=range(1, 257), metavar="1-256")
    parser.add_argument("--refetch-interiors", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if args.publish and (args.action != "transfer" or args.limit_mib):
        parser.error("--publish requires an unlimited --action transfer")
    run(args)


if __name__ == "__main__":
    main()
