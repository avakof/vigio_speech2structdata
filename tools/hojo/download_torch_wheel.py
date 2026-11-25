#!/usr/bin/env python3
"""Resume the locked native macOS PyTorch wheel via bounded HTTP ranges."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
from hashlib import file_digest
from http.client import HTTPException
from pathlib import Path
import shutil
import time
import tomllib
from urllib.request import Request, urlopen

WHEEL = "torch-2.5.1-cp312-none-macosx_11_0_arm64.whl"
PART_BYTES = 256 * 1024
MAX_ATTEMPTS = 4
ROOT = Path(__file__).resolve().parents[2]


def locked_wheel():
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    entry = next(
        wheel
        for package in lock["package"]
        if package["name"] == "torch" and package["version"] == "2.5.1"
        for wheel in package["wheels"]
        if wheel["url"].endswith("/" + WHEEL)
    )
    return entry["url"], entry["size"], entry["hash"].removeprefix("sha256:")


def read_range(url, size, start, end):
    """Retrieve one inclusive byte range, validated before it is saved."""
    expected_range = f"bytes {start}-{end}/{size}"
    expected_size = end - start + 1
    request = Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
    with urlopen(request, timeout=45) as response:
        if response.status != 206 or response.headers.get("Content-Range") != expected_range:
            raise OSError(f"Expected HTTP 206 {expected_range}, received "
                          f"{response.status} {response.headers.get('Content-Range')!r}")
        data = response.read(expected_size + 1)
    if len(data) != expected_size:
        raise OSError(f"Received {len(data)} bytes for {expected_range}, expected {expected_size}")
    return data


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def download(destination, workers):
    url, size, expected_hash = locked_wheel()
    final = destination / WHEEL
    if final.is_file():
        if final.stat().st_size != size or digest(final) != expected_hash:
            raise OSError(f"Existing wheel has the wrong digest or length at {final}")
        print(f"Verified existing wheel {final}", flush=True)
        return

    parts = destination / (WHEEL + ".parts")
    parts.mkdir(exist_ok=True)
    ranges = [(start, min(start + PART_BYTES, size) - 1) for start in range(0, size, PART_BYTES)]

    def fetch(item):
        index, (start, end) = item
        part = parts / f"{index:04d}.part"
        if part.is_file() and part.stat().st_size == end - start + 1:
            return part
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                data = read_range(url, size, start, end)
                break
            except (OSError, HTTPException) as error:
                if attempt == MAX_ATTEMPTS:
                    raise
                print(f"Retry {attempt} for bytes {start}-{end}: {error}", flush=True)
                time.sleep(attempt)
        temporary = part.with_suffix(".incomplete")
        temporary.write_bytes(data)
        temporary.replace(part)
        return part

    print(f"Downloading {WHEEL} in {len(ranges)} ranges, {size} bytes", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        resolved_parts = []
        for completed, part in enumerate(pool.map(fetch, enumerate(ranges)), start=1):
            resolved_parts.append(part)
            if completed % 25 == 0 or completed == len(ranges):
                print(f"{completed}/{len(ranges)} ordered ranges available", flush=True)

    incomplete = final.with_suffix(".unverified")
    with incomplete.open("wb") as output:
        for part in resolved_parts:
            with part.open("rb") as stream:
                shutil.copyfileobj(stream, output)
    if incomplete.stat().st_size != size or digest(incomplete) != expected_hash:
        raise OSError(f"Assembled wheel did not match the frozen lock SHA-256 {expected_hash}")
    incomplete.replace(final)
    shutil.rmtree(parts)
    print(f"Verified SHA-256 {expected_hash} for {final}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--workers", type=int, default=24, choices=range(1, 33), metavar="1-32")
    parser.add_argument("--directory", type=Path, default=Path("~/.cache/speechflow/hojo-wheels"))
    args = parser.parse_args()
    if args.probe:
        url, size, _ = locked_wheel()
        part = read_range(url, size, 0, PART_BYTES - 1)
        print(f"HTTP 206, Content-Range bytes 0-{PART_BYTES - 1}/{size}, received {len(part)} bytes")
        return

    destination = args.directory.expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / (WHEEL + ".lock")).open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit(f"Another downloader is writing {WHEEL} in {destination}") from error
        download(destination, args.workers)


if __name__ == "__main__":
    main()
