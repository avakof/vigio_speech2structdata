#!/usr/bin/env python3
"""Prepare or run the fixed three-crop Hojo diagnostic, one backend per process."""

import argparse
from collections import Counter
from hashlib import file_digest, sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
from struct import pack
import sys
from tempfile import TemporaryDirectory
import time

import soundfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bisect_long_cpu import emit, match_gold
from speechflow.audio.decoding import SAMPLE_BYTES, decode_file
from speechflow.audio.hojo import MODEL_ID, MODEL_REVISION, SAMPLE_RATE, infer_wav, load_local_model, verify_local_bundle
from speechflow.config import Settings


ARCHIVE = ROOT / "reports/qa/real-full-long-reproducer.json"
GOLD = ROOT / "reports/qa/real-full-long-hojo.json"
CROPS = (
    ("A", 486400, 511200, "f9204aa6ad118a97f6a0e55ec9288e984aa8cd2458687590066316c40ca7b52f"),
    ("B", 645760, 672320, "046bf0096c8745388b46886cb14e374dfa9e50fb8e6e95e98e7f7d3adbc848bb"),
    ("C", 806880, 834880, "da86c4601d3930ce6a66f5de77233c1ccd8b08570701cdb9c65dc008c0347e93"),
)
WAV_SHA256 = {
    "A": "afd8921b1d6a297bd1016886f29bd2bc23848ecf1be8d7685e77aaae59223b9a",
    "B": "102f4f68b26daa24e39049437d48e3758ed8b5ed4c2d35a0f40b2efab2f4f9f9",
    "C": "048c3b3189d5faee7bf2bd01fe179d20832b9d115a19ece6ff4380031b14cfe3",
}
DEFAULT_CROP_DIR = Path.home() / ".cache/speechflow/hojo-exact-crops"


def sha256_file(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def verify_input(directory, crop):
    label, start, end, pcm_sha = crop
    path = directory / f"{label}.wav"
    info = soundfile.info(path)
    samples, rate = soundfile.read(path, dtype="float32")
    if (info.format, info.subtype, info.channels, info.frames, rate) != (
        "WAV", "FLOAT", 1, end - start, SAMPLE_RATE
    ):
        raise ValueError(f"Unexpected WAV format or length for {path}")
    wav_sha = sha256_file(path)
    if wav_sha != WAV_SHA256[label] or sha256(samples.astype("<f4").tobytes()).hexdigest() != pcm_sha:
        raise ValueError(f"Checksum mismatch for {path}")
    emit("input", id=label, path=path, source_start_sample=start, source_end_sample=end,
         padding_samples=0, samples=info.frames, sample_rate=rate, duration_s=info.duration,
         pcm_sha256=pcm_sha, wav_sha256=wav_sha)
    return path, info.duration


def prepare(directory):
    source = json.loads(ARCHIVE.read_text())
    original = ROOT / source["original_wav"]
    archived_pcm = ROOT / source["normalized_pcm"]
    settings = Settings(offline=True, backend="cpu")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with TemporaryDirectory(prefix="hojo-exact-crops-") as temp:
        decoded = Path(temp) / "decoded.f32le"
        metrics = {}
        meta = decode_file(original, decoded, settings.max_spool_bytes, metrics=metrics)
        normalized = decoded.read_bytes()
    if meta.digest != source["original_wav_sha256"] or normalized != archived_pcm.read_bytes():
        raise ValueError("Production normalization differs from the original long fixture")
    emit("source", path=original, original_wav_sha256=meta.digest, source_samples=meta.sample_count,
         normalized_pcm_sha256=sha256(normalized).hexdigest(), decoder=metrics["decoder"],
         archived_normalized_pcm=archived_pcm)
    for crop in CROPS:
        label, start, end, expected = crop
        data = normalized[start * SAMPLE_BYTES:end * SAMPLE_BYTES]
        if sha256(data).hexdigest() != expected:
            raise ValueError(f"Unexpected source sample hash for {label}")
        # Minimal IEEE float WAV. libsndfile's PEAK chunk includes a wall-clock timestamp.
        header = pack("<4sIHHIIHH4sII4sI", b"fmt ", 16, 3, 1, SAMPLE_RATE,
                      SAMPLE_RATE * SAMPLE_BYTES, SAMPLE_BYTES, 32,
                      b"fact", 4, end - start, b"data", len(data))
        riff = pack("<4sI4s", b"RIFF", 4 + len(header) + len(data), b"WAVE")
        (directory / f"{label}.wav").write_bytes(riff + header + data)
        verify_input(directory, crop)
    emit("prepared", inference_calls=0)


def _placement(model, backend):
    decoder = model.decoder_model
    layers = decoder.model.layers
    if len(layers) != 36:
        raise RuntimeError(f"Expected 36 decoder layers from the pinned bundle, found {len(layers)}")
    expected_attention = "eager" if backend == "mps" else "sdpa"
    groups = Counter()
    for layer in layers:
        attention = layer.self_attn
        config = attention.config
        head_dim = attention.head_dim
        q, k, v, o = (getattr(attention, f"{name}_proj").weight for name in "qkvo")
        if ((config.num_attention_heads, config.num_key_value_heads, head_dim) != (32, 8, 128)
                or config._attn_implementation != expected_attention
                or q.shape[0] != config.num_attention_heads * head_dim
                or k.shape[0] != config.num_key_value_heads * head_dim
                or v.shape != k.shape or o.shape[1] != q.shape[0]):
            raise RuntimeError("Unexpected decoder attention setting or head shape")
        key = (config._attn_implementation, config.num_attention_heads, config.num_key_value_heads,
               head_dim, tuple((str(w.device), str(w.dtype)) for w in (q, k, v, o)))
        groups[key] += 1
    all_parameters = Counter()
    for name, parameter in model.named_parameters():
        if parameter.device.type != backend:
            raise RuntimeError(f"{name} is on {parameter.device}, expected {backend}")
        all_parameters[(name.split(".", 1)[0], str(parameter.device), str(parameter.dtype))] += 1
    return layers, {
        "actual_parameter_placements": [{"module": m, "device": d, "dtype": t, "tensors": n}
                                        for (m, d, t), n in all_parameters.items()],
        "decoder_attention_groups": [
            {"attention": impl, "query_heads": qh, "kv_heads": kh, "head_dim": hd,
             "qkvo_device_dtype": loc, "layers": count}
            for (impl, qh, kh, hd, loc), count in groups.items()
        ],
        "encoder_attention": model.speech_encoder.config._attn_implementation,
    }


def run(directory, backend):
    import torch

    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("Start this process with PYTORCH_ENABLE_MPS_FALLBACK=0")
    if any(os.environ.get(key) != "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise RuntimeError("Start this process with HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1")
    inputs = [verify_input(directory, crop) for crop in CROPS]
    settings = Settings(offline=True, backend=backend)
    start = time.perf_counter()
    verify_local_bundle(settings)
    emit("assets_verified", seconds=time.perf_counter() - start, backend=backend,
         repository=MODEL_ID, revision=MODEL_REVISION)
    start = time.perf_counter()
    model = load_local_model(settings.hojo_dir, backend)
    if backend == "mps":
        torch.mps.synchronize()
    load_seconds = time.perf_counter() - start
    layers, placement = _placement(model, backend)
    emit("model", backend=backend, load_seconds=load_seconds, batch_size=1, mps_fallback=False,
         generation_settings=dict(model.config.generate), layers=len(layers), **placement,
         packages={name: version(name) for name in ("hojo-asr", "torch", "torchaudio", "transformers")})

    activity = {}

    def record_attention(index):
        def check_input(_module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None or hidden.device.type != backend:
                raise RuntimeError(f"Decoder attention layer {index} did not receive {backend} hidden states")
            summary = activity.setdefault(index, {"calls": 0, "device": str(hidden.device),
                                                   "dtype": str(hidden.dtype)})
            summary["calls"] += 1
        return check_input

    indices = (0, len(layers) - 1)
    hooks = [layers[i].self_attn.register_forward_pre_hook(record_attention(i), with_kwargs=True)
             for i in indices]
    try:
        for wav, duration in inputs:
            activity.clear()
            before = (model.autocast_calls, model.encode_calls)
            start = time.perf_counter()
            raw = infer_wav(model, wav)
            if backend == "mps":
                torch.mps.synchronize()
            seconds = time.perf_counter() - start
            emit("inference", backend=backend, id=wav.stem, raw=raw, duration_s=duration, seconds=seconds,
                 retokenized_text_tokens=len(model.tokenizer(raw["text"], add_special_tokens=False)["input_ids"]),
                 observed_attention=activity, adapter_hooks={
                     "autocast_context": model.autocast_calls - before[0],
                     "encode_speech": model.encode_calls - before[1],
                 })
            if set(activity) != set(indices) or model.encode_calls == before[1]:
                raise RuntimeError("Hojo did not exercise the expected attention and encoder hooks")
    finally:
        for hook in hooks:
            hook.remove()
    emit("finished", backend=backend, inference_calls=len(inputs))


def score(logs):
    archive = json.loads(GOLD.read_text())
    anchors = {(a["sample_start"], a["sample_end"]): a for a in archive["anchors"]}
    gold = {label: anchors[(start, end)] for label, start, end, _ in CROPS}
    all_scores = []
    for backend, path in zip(("cpu", "mps"), logs, strict=True):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        finished = [r for r in rows if r["event"] == "finished"]
        replies = [r for r in rows if r["event"] == "inference"]
        if (finished != [{"event": "finished", "backend": backend, "inference_calls": 3}]
                or [(r["backend"], r["id"], r["raw"]["key"]) for r in replies]
                != [(backend, label, label) for label in gold]):
            raise ValueError(f"{path} does not contain exactly the three completed {backend} calls")
        for row in replies:
            target = gold[row["id"]]
            all_scores.append({"backend": backend, "id": row["id"], "gold": target,
                               "checks": match_gold(row["raw"]["text"], target)})
    emit("score", results=all_scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare").add_argument("--crops", type=Path, default=DEFAULT_CROP_DIR)
    run_parser = sub.add_parser("run", help="Load Hojo only with explicit serial slot authorization")
    run_parser.add_argument("backend", choices=("cpu", "mps"))
    run_parser.add_argument("--crops", type=Path, default=DEFAULT_CROP_DIR)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("cpu_log", type=Path)
    score_parser.add_argument("mps_log", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.crops)
    elif args.command == "run":
        run(args.crops, args.backend)
    else:
        score((args.cpu_log, args.mps_log))


if __name__ == "__main__":
    main()
