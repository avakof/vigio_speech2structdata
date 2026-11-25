#!/usr/bin/env python3
"""Probe the pinned Hojo frontend or benchmark one already-normalized WAV."""

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import inspect
import json
import logging
import os
from pathlib import Path
import platform
import resource
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from speechflow.audio.hojo import (
    BUNDLE_LIMIT_SECONDS,
    DEFAULT_CHUNK_SECONDS,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_RATE,
    infer_wav,
    load_local_model,
    validate_frame_count,
    verify_local_bundle,
)
from speechflow.config import Settings

DEFAULT_BUNDLE = Path("~/.cache/speechflow/assets/hojo").expanduser()
FEATURE_PATH = Path("whisper-large-v3/preprocessor_config.json")


def emit(event, **data):
    print(json.dumps({"event": event, **data}, ensure_ascii=False, default=str), flush=True)


def memory(torch, psutil, backend):
    result = {
        "process_rss_bytes": psutil.Process().memory_info().rss,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "system_available_bytes": psutil.virtual_memory().available,
    }
    if backend == "mps" and torch.backends.mps.is_available():
        result["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
        result["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    return result


def probe_frontend(bundle):
    import numpy as np
    import torch
    from transformers import WhisperFeatureExtractor

    configuration = json.loads((bundle / FEATURE_PATH).read_text())
    emit("frontend_environment", python=platform.python_version(), machine=platform.machine(),
         torch=torch.__version__, transformers=importlib.metadata.version("transformers"))
    for options in ({}, {"chunk_length": 40}):
        extractor = WhisperFeatureExtractor.from_pretrained(
            str(bundle / FEATURE_PATH.parent), local_files_only=True, **options
        )
        for seconds in (25, 30, 31, 40, 41):
            # Count frames using the same arguments as the wheel's data pipe.
            samples = np.zeros(seconds * SAMPLE_RATE, dtype=np.float32)
            features = extractor(samples, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=False)
            frames = int(features.input_features.shape[-1])
            expected = min(seconds * SAMPLE_RATE, extractor.n_samples) // extractor.hop_length
            if frames != expected:
                raise RuntimeError(f"Expected {expected} frontend frames, got {frames}")
            emit(
                "frontend",
                bundle_chunk_seconds=configuration["chunk_length"],
                runtime_chunk_seconds=extractor.chunk_length,
                runtime_n_samples=extractor.n_samples,
                input_seconds=seconds,
                output_mel_frames=frames,
                truncated=frames < seconds * SAMPLE_RATE // extractor.hop_length,
            )


def probe_primitives(backend):
    """Check small representative operations on the requested backend without weights."""
    import torch
    from torch.nn import functional as ops

    emit(
        "primitive_environment",
        backend=backend,
        torch=torch.__version__,
        machine=platform.machine(),
        mps_built=torch.backends.mps.is_built(),
        mps_available=torch.backends.mps.is_available(),
        mps_fallback=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
    )
    device = torch.device(backend)

    def check(name, tensor, dtype):
        if tensor.device.type != backend or tensor.dtype != dtype:
            raise RuntimeError(f"{name} ran on {tensor.device} with {tensor.dtype}")
        if not torch.isfinite(tensor).all().item():
            raise RuntimeError(f"{name} produced non-finite values")
        emit("primitive", name=name, device=tensor.device, dtype=tensor.dtype, shape=list(tensor.shape))

    samples = torch.arange(40, device=device, dtype=torch.float32).view(1, 1, 5, 8) / 40
    kernel = torch.ones(1, 1, 3, 3, device=device)
    audio = ops.gelu(ops.conv2d(samples, kernel, padding=1))
    check("audio_convolution", audio, torch.float32)
    packed = audio[0, 0, torch.tensor([True, True, True, True, False], device=device)]
    check("audio_boolean_index", packed, torch.float32)
    embeddings = audio.to(dtype=torch.bfloat16)
    projection = torch.matmul(embeddings, embeddings.transpose(-2, -1))
    check("decoder_matmul", projection, torch.bfloat16)
    mask = torch.ones(5, 5, dtype=torch.bool, device=device).tril()
    attention = ops.scaled_dot_product_attention(embeddings, embeddings, embeddings, attn_mask=mask)
    check("decoder_masked_sdpa", attention, torch.bfloat16)
    beam_index = torch.tensor([3, 1, 0, 2], device=device)
    check("decoder_beam_reorder", attention.repeat(4, 1, 1, 1).index_select(0, beam_index), torch.bfloat16)


def benchmark(args, torch, psutil, soundfile):
    info = soundfile.info(args.audio)
    if info.format != "WAV" or info.samplerate != SAMPLE_RATE or info.channels != 1:
        raise ValueError("Supply a normalized mono 16 kHz WAV")
    validate_frame_count(info.frames, args.max_seconds)
    with args.audio.open("rb") as audio_file:
        digest = hashlib.file_digest(audio_file, "sha256").hexdigest()
    emit(
        "input",
        file=args.audio,
        sha256=digest,
        sample_rate=info.samplerate,
        frames=info.frames,
        duration_seconds=info.duration,
        reference=args.reference,
        provenance=args.provenance,
        chunk_limit_seconds=args.max_seconds,
    )
    verification_start = time.perf_counter()
    verify_local_bundle(Settings(hojo_dir=args.bundle, backend=args.backend, offline=True))
    emit("assets_verified", seconds=time.perf_counter() - verification_start)
    emit("before_load", memory=memory(torch, psutil, args.backend))
    load_start = time.perf_counter()
    model = load_local_model(args.bundle, args.backend)
    if args.backend == "mps":
        torch.mps.synchronize()
    load_seconds = time.perf_counter() - load_start
    placements = defaultdict(set)
    for name, parameter in model.named_parameters():
        placements[name.split(".")[0]].add(f"{parameter.device}:{parameter.dtype}")
    emit(
        "after_load",
        seconds=load_seconds,
        parameter_placements={name: sorted(values) for name, values in placements.items()},
        encoder_attention=model.speech_encoder.config._attn_implementation,
        decoder_attention=model.decoder_model.config._attn_implementation,
        feature_limit_seconds=model.feat_extractor.chunk_length,
        compatibility_subclass=type(model).__name__,
        memory=memory(torch, psutil, args.backend),
    )
    for index in range(args.iterations):
        if index == 0:
            logging.info("Inference uses batch size 1 and one model instance")
        before = (model.autocast_calls, model.encode_calls)
        start = time.perf_counter()
        record = infer_wav(model, args.audio, max_seconds=args.max_seconds)
        if args.backend == "mps":
            torch.mps.synchronize()
        seconds = time.perf_counter() - start
        hooks = (model.autocast_calls - before[0], model.encode_calls - before[1])
        emit(
            "inference",
            iteration=index + 1,
            seconds=seconds,
            realtime_factor=seconds / info.duration,
            result=record,
            adapter_hooks={"autocast_context": hooks[0], "encode_speech": hooks[1]},
            memory=memory(torch, psutil, args.backend),
        )
        if not record["text"].strip():
            raise RuntimeError("The non-silent feasibility fixture returned no transcript")
        if min(hooks) == 0:
            raise RuntimeError("Hojo inference did not execute both local compatibility hooks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, nargs="?")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--backend", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    parser.add_argument("--reference")
    parser.add_argument("--provenance")
    probe = parser.add_mutually_exclusive_group()
    probe.add_argument("--frontend-only", action="store_true")
    probe.add_argument("--primitives-only", action="store_true")
    args = parser.parse_args()
    if not (args.frontend_only or args.primitives_only) and (args.audio is None or args.iterations < 1):
        parser.error("Inference requires a WAV and a positive iteration count")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.frontend_only:
        probe_frontend(args.bundle)
        return
    if args.primitives_only:
        probe_primitives(args.backend)
        return

    import psutil
    import soundfile
    import torch
    from hojo_asr import HOJO_ASR, __version__ as hojo_module_version

    emit(
        "environment",
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        bundle=args.bundle,
        backend=args.backend,
        machine=platform.machine(),
        host=platform.mac_ver()[0],
        python=platform.python_version(),
        cpu_threads=torch.get_num_threads(),
        mps_built=torch.backends.mps.is_built(),
        mps_available=torch.backends.mps.is_available(),
        mps_fallback=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
        packages={
            name: importlib.metadata.version(name)
            for name in ("hojo-asr", "torch", "torchaudio", "transformers", "huggingface-hub",
                         "accelerate", "safetensors", "numpy", "soundfile", "psutil")
        },
        hojo_module_version=hojo_module_version,
        load_model_signature=str(inspect.signature(HOJO_ASR.load_model)),
        run_infer_signature=str(inspect.signature(HOJO_ASR.run_infer)),
        bundle_limit_seconds=BUNDLE_LIMIT_SECONDS,
    )
    benchmark(args, torch, psutil, soundfile)


if __name__ == "__main__":
    main()
