#!/usr/bin/env python3
"""Inspect pinned Hojo output IDs on one authorized CPU or MPS child."""

import argparse
from dataclasses import asdict
from functools import wraps
from hashlib import file_digest
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speechflow.audio.hojo import (
    generation_cap_warning, generation_loop_warning, infer_wav, load_local_model,
    verify_local_bundle,
)
from speechflow.config import Settings

JACQUES_SHA256 = "d92c0759b7d4d3e03ae22eb3c47aa97ef7122789eac187c7cf42e8e6035d1330"


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, ensure_ascii=False, default=str), flush=True)


def checked_input(path, expected):
    import soundfile

    with path.open("rb") as wav:
        digest = file_digest(wav, "sha256").hexdigest()
    if digest != expected:
        raise ValueError(f"Unexpected input digest for {path}: {digest}")
    meta = soundfile.info(path)
    return {"file": path, "sha256": digest, "frames": meta.frames,
            "rate_hz": meta.samplerate, "duration_s": meta.duration}


def run(backend, long_center, cpu_eager):
    if any(os.environ.get(var) != val for var, val in (
        ("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"), ("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    )):
        raise RuntimeError("Set HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0")
    import torch

    jacques = checked_input(Path.home() / ".cache/speechflow/hojo-jacques.wav", JACQUES_SHA256)
    center = None
    if long_center:
        saved = json.loads((ROOT / "reports/qa/real-full-long-reproducer.json").read_text())
        center = checked_input(ROOT / saved["isolated_hojo_input"], saved["isolated_wav_sha256"])
    settings = Settings(backend=backend, offline=True)
    verify_local_bundle(settings)
    start = time.perf_counter()
    model = load_local_model(settings.hojo_dir, backend)
    if backend == "mps":
        torch.mps.synchronize()
    load_seconds = time.perf_counter() - start
    decoder = model.decoder_model
    placements = sorted({(name.split(".")[0], str(p.device), str(p.dtype))
                         for name, p in model.named_parameters()})
    if any(torch.device(device).type != backend for _, device, _ in placements):
        raise RuntimeError(f"Unexpected placement for {backend}: {placements}")
    emit("model", backend=backend, load_seconds=load_seconds, fallback=False, placements=placements,
         generate=dict(model.config.generate))
    generate = decoder.generate
    selected = []

    @wraps(generate)
    def observe_first_tokens(*args, **kwargs):
        ids = generate(*args, **kwargs)
        first = ids[0, :3].tolist()
        selected.append(first)
        emit("before_decode", first_token_ids=first,
             first_token_spellings=model.tokenizer.convert_ids_to_tokens(first))
        return ids

    decoder.generate = observe_first_tokens
    try:
        clips = [("jacques", jacques)]
        if center:
            clips.append(("center_25s", center))
        if cpu_eager:
            clips.append(("jacques_cpu_eager_contrast", jacques))
        for label, meta in clips:
            if label == "jacques_cpu_eager_contrast":
                decoder.set_attn_implementation("eager")
            implementations = sorted({
                layer.self_attn.config._attn_implementation for layer in decoder.model.layers
            })
            emit("input", backend=backend, label=label, decoder_attention=implementations, **meta)
            selected.clear()
            start = time.perf_counter()
            raw = infer_wav(model, meta["file"])
            if backend == "mps":
                torch.mps.synchronize()
            seconds = time.perf_counter() - start
            observation = model.last_generation
            warnings = [warning for check in (generation_cap_warning, generation_loop_warning)
                        if (warning := check(model, raw["text"]))]
            if (decoder.generate is not observe_first_tokens or len(selected) != 1
                    or observation is None or observation.output_tokens is None):
                raise RuntimeError(f"Did not observe exactly one complete decoder output for {label}")
            emit("reply", backend=backend, label=label, seconds=seconds, raw=raw,
                 observation=asdict(observation), warnings=warnings,
                 retokenized_text_tokens=len(model.tokenizer(raw["text"], add_special_tokens=False)["input_ids"]))
    finally:
        del decoder.generate
    emit("finished", backend=backend, calls=len(clips))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("cpu", "mps"))
    parser.add_argument("--long-center", action="store_true")
    parser.add_argument("--cpu-eager", action="store_true")
    args = parser.parse_args()
    if args.backend != "cpu" and (args.long_center or args.cpu_eager):
        parser.error("The authorized MPS probe is the Jacques recording only")
    run(args.backend, args.long_center, args.cpu_eager)
