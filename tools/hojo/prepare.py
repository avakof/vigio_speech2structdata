#!/usr/bin/env python3
"""Prepare the complete pinned Hugging Face snapshot and verify runtime assets."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from speechflow.assets import HOJO
from speechflow.audio.hojo import verify_local_bundle
from speechflow.config import Settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Settings().hojo_dir)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    settings = Settings(hojo_dir=args.bundle, offline=args.verify_only)
    if not args.verify_only:
        from huggingface_hub import snapshot_download

        snapshot_download(
            HOJO["repository"],
            revision=HOJO["revision"],
            local_dir=settings.hojo_dir,
            token=False,
            max_workers=4,
        )
    verify_local_bundle(settings)
    print(f"Verified {len(HOJO['files'])} runtime assets for {HOJO['repository']} at {HOJO['revision']}")


if __name__ == "__main__":
    main()
