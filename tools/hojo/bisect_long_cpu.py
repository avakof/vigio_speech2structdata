#!/usr/bin/env python3
"""Reproduce the original long failure's PCM and test two disjoint CPU Hojo cores."""

import argparse
from hashlib import file_digest, sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
import time
import unicodedata

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speechflow.audio import _model_placement
from speechflow.audio.decoding import SAMPLE_BYTES, decode_file
from speechflow.audio.hojo import MODEL_ID, MODEL_REVISION, SAMPLE_RATE, infer_wav, load_local_model, verify_local_bundle
from speechflow.audio.segmentation import make_segment
from speechflow.config import Settings

REPRODUCER = ROOT / "reports/qa/real-full-long-reproducer.json"
CENTER = (394_112, 787_712)
MIDPOINT = 590_912
LOW_RMS_CUT = 588_000
PADDING = 3_200


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False, default=str), flush=True)


def read_span(pcm, start, end):
    with pcm.open("rb") as file:
        file.seek(start * SAMPLE_BYTES)
        payload = file.read((end - start) * SAMPLE_BYTES)
    if len(payload) != (end - start) * SAMPLE_BYTES:
        raise OSError(f"PCM did not contain the complete half-open span {start}:{end}")
    return payload


def prepare_pair(pcm, source, destination, anchors, cut):
    import numpy as np
    import soundfile

    pair = []
    for ordinal, (left, right) in enumerate(zip((CENTER[0], cut), (cut, CENTER[1]), strict=True)):
        segment = make_segment(source.source_id, ordinal, left, right, source.sample_count, PADDING)
        expected = [
            anchor for anchor in anchors
            if left <= anchor["sample_start"] < anchor["sample_end"] <= right
        ]
        if len(expected) != 1:
            raise ValueError(f"Expected exactly one original anchor in core {left}:{right}")
        payload = read_span(pcm, segment.input_start, segment.input_end)
        wav = destination / f"{segment.segment_id}.wav"
        soundfile.write(wav, np.frombuffer(payload, dtype="<f4"), SAMPLE_RATE, subtype="FLOAT")
        with wav.open("rb") as stream:
            wav_digest = file_digest(stream, "sha256").hexdigest()
        emit("input", cut_sample=cut, segment=segment.model_dump(), wav=wav, wav_sha256=wav_digest,
             pcm_sha256=sha256(payload).hexdigest(), padding_samples=PADDING,
             duration_s=(segment.input_end - segment.input_start) / SAMPLE_RATE,
             original_source_gold=expected[0])
        pair.append((wav, expected[0]))
    return pair


def match_gold(text, anchor):
    normalized = " ".join(unicodedata.normalize("NFC", text).casefold().split())
    quantity_word = re.escape(anchor["phrase"].split()[0])
    quantity = re.escape(anchor["quantity"])
    product = re.escape(anchor["product"])
    pattern = fr"\b(?:{quantity_word}|{quantity})\s+(?:kilos?|kg)\s+d(?:e\s+|['’]){product}\b"
    return {"literal_phrase": anchor["phrase"] in normalized,
            "quantity_unit_product": bool(re.search(pattern, normalized))}


def token_count(model, text):
    return len(model.tokenizer(text, add_special_tokens=False)["input_ids"])


def run_pair(model, pair, cut):
    recovered = True
    for wav, anchor in pair:
        started = time.perf_counter()
        record = infer_wav(model, wav)
        elapsed = time.perf_counter() - started
        checks = match_gold(record["text"], anchor)
        recovered = recovered and checks["quantity_unit_product"]
        emit("inference", cut_sample=cut, wav=wav, raw=record, seconds=elapsed,
             retokenized_text_tokens=token_count(model, record["text"]),
             original_source_gold=anchor, checks=checks)
    emit("comparison", cut_sample=cut, restored_both_original_anchors=recovered)
    return recovered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("~/.cache/speechflow/hojo-long-bisection"))
    parser.add_argument("--run-model", action="store_true", help="Use only with the granted serial Hojo slot")
    parser.add_argument("--low-rms-if-failed", action="store_true")
    args = parser.parse_args()
    fixture = json.loads(REPRODUCER.read_text())
    destination = args.out.expanduser().resolve()
    destination.mkdir(parents=True, mode=0o700, exist_ok=True)
    pcm = destination / "original-normalized.f32le"
    settings = Settings(offline=True, backend="cpu")
    metrics = {}
    started = time.perf_counter()
    original = ROOT / fixture["original_wav"]
    source = decode_file(original, pcm, settings.max_spool_bytes, metrics=metrics)
    if source.digest != fixture["original_wav_sha256"]:
        raise ValueError(f"Original long fixture digest changed: {source.digest}")
    central = read_span(pcm, *fixture["input_samples"])
    central_digest = sha256(central).hexdigest()
    if central_digest != fixture["isolated_f32_pcm_sha256"]:
        raise ValueError(f"Production normalization differs from the known failing PCM: {central_digest}")

    emit("source", path=original, source=source.model_dump(), decoder=metrics["decoder"],
         original_center_input=fixture["input_samples"], original_center_core=list(CENTER),
         original_center_pcm_sha256=central_digest, decode_verification_seconds=time.perf_counter() - started,
         original_gold=fixture["expected_anchors"])
    pair = prepare_pair(pcm, source, destination, fixture["expected_anchors"], MIDPOINT)
    if not args.run_model:
        emit("prepared", model_loaded=False)
        return 0

    start = time.perf_counter()
    verify_local_bundle(settings)
    emit("assets_verified", seconds=time.perf_counter() - start, repository=MODEL_ID, revision=MODEL_REVISION)
    start = time.perf_counter()
    model = load_local_model(settings.hojo_dir, "cpu")
    backend, dtype = _model_placement(model)
    emit("model", seconds=time.perf_counter() - start, backend=backend, dtype=dtype, batch_size=1,
         decoder_attention=model.decoder_model.config._attn_implementation,
         generation_settings=dict(model.config.generate),
         packages={name: version(name) for name in ("hojo-asr", "torch", "torchaudio", "transformers")})
    known = fixture["unchanged_hojo_result"]
    emit("known_25_second_failure", raw=known, retokenized_text_tokens=token_count(model, known["text"]),
         checks={a["product"]: match_gold(known["text"], a) for a in fixture["expected_anchors"]})
    recovered = run_pair(model, pair, MIDPOINT)
    if not recovered and args.low_rms_if_failed:
        alternate = prepare_pair(pcm, source, destination, fixture["expected_anchors"], LOW_RMS_CUT)
        recovered = run_pair(model, alternate, LOW_RMS_CUT)
    emit("experiment", restored_both_original_anchors=recovered, models_loaded=1)
    return 0 if recovered else 1


if __name__ == "__main__":
    raise SystemExit(main())
