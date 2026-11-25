#!/usr/bin/env python3
"""Run one native generation prefill on each device using identical CPU embeddings."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
import time

from probe_generation_cap import JACQUES_SHA256, checked_input, emit
from speechflow.audio.hojo import infer_wav, load_local_model, verify_local_bundle
from speechflow.config import Settings


class PrefillCaptured(Exception):
    """Stop the diagnostic before generation consumes the first logits."""


def main(backend, snapshot):
    if any(os.environ.get(k) != v for k, v in (
        ("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"), ("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    )):
        raise RuntimeError("Require offline loading and PYTORCH_ENABLE_MPS_FALLBACK=0")
    import torch
    from hojo_asr.hojo_asr_model import StopOnTokenSequences
    from transformers import StoppingCriteriaList

    def tensor_provenance(tensor):
        if tensor is None:
            return None
        contiguous = tensor.detach().cpu().contiguous()
        digest = sha256(contiguous.view(torch.uint8).numpy().tobytes()).hexdigest()
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "device": str(tensor.device), "sha256": digest}

    clip = checked_input(Path.home() / ".cache/speechflow/hojo-jacques.wav", JACQUES_SHA256)
    settings = Settings(backend=backend, offline=True)
    verify_local_bundle(settings)
    start = time.perf_counter()
    model = load_local_model(settings.hojo_dir, backend)
    if backend == "mps":
        torch.mps.synchronize()
    decoder = model.decoder_model
    decoder.set_attn_implementation("eager")
    placements = {(str(p.device), str(p.dtype)) for p in decoder.parameters()}
    expected = {(str(decoder.device), "torch.bfloat16")}
    attention = {layer.self_attn.config._attn_implementation for layer in decoder.model.layers}
    if placements != expected or attention != {"eager"} or decoder.device.type != backend:
        raise RuntimeError(f"Unexpected decoder path: placements={placements}, attention={attention}")
    emit("input", backend=backend, **clip, decoder_placements=sorted(placements),
         decoder_attention=sorted(attention), load_seconds=time.perf_counter() - start)

    generate = decoder.generate
    before = None
    calls = 0
    captured = {}
    prepared_names = ("inputs_embeds", "attention_mask", "position_ids", "cache_position", "input_ids")

    def record_inputs(_module, args, kwargs):
        nonlocal before, calls
        calls += 1
        embeddings = kwargs.get("inputs_embeds")
        if args or calls != 1 or embeddings is None or embeddings.shape[0] != 4:
            raise RuntimeError("Did not enter the expected four-beam embedding prefill")
        captured.update({key: tensor_provenance(kwargs.get(key)) for key in prepared_names})
        before = time.perf_counter()

    def capture_logits(_module, _args, _kwargs, output):
        logits = output.logits[:, -1, :].detach().cpu()
        prefill_seconds = time.perf_counter() - before
        if logits.ndim != 2 or logits.shape[0] != 4 or not bool(torch.isfinite(logits).all()):
            raise RuntimeError("Expected one finite vocabulary logit vector per beam")
        scores, ids = torch.topk(logits.float(), k=8, dim=-1)
        emit("prefill", backend=backend, prepared=captured, prefill_seconds=prefill_seconds,
             logits=tensor_provenance(logits), top_ids=ids.tolist(), raw_logits=scores.tolist(),
             top_tokens=[model.tokenizer.convert_ids_to_tokens(row) for row in ids.tolist()],
             rank_1_minus_rank_2=(scores[:, 0] - scores[:, 1]).tolist())
        raise PrefillCaptured

    def save_cpu_kwargs(**kwargs):
        criteria = kwargs.pop("stopping_criteria")
        if len(criteria) != 1 or not isinstance(criteria[0], StopOnTokenSequences):
            raise RuntimeError("Unexpected pinned Hojo stop criterion")
        if kwargs.get("num_beams") != 4 or not all(
            isinstance(v, (torch.Tensor, int, float, bool, type(None))) for v in kwargs.values()
        ):
            raise RuntimeError("Unexpected generation argument or beam layout")
        snapshot.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        torch.save({"kwargs": kwargs, "stop_token_seqs": criteria[0].stop_token_seqs,
                    "jacques_sha256": clip["sha256"]}, snapshot)
        return generate(**kwargs, stopping_criteria=criteria)

    pre = decoder.register_forward_pre_hook(record_inputs, with_kwargs=True)
    post = decoder.register_forward_hook(capture_logits, with_kwargs=True)
    try:
        if backend == "cpu":
            decoder.generate = save_cpu_kwargs
            infer_wav(model, clip["file"])
        else:
            saved = torch.load(snapshot, map_location="cpu", weights_only=True)
            if saved["jacques_sha256"] != clip["sha256"]:
                raise RuntimeError("Unexpected snapshot fixture")
            kwargs = {k: v.to(decoder.device) if isinstance(v, torch.Tensor) else v
                      for k, v in saved["kwargs"].items()}
            stop_tokens = [seq.to(decoder.device) for seq in saved["stop_token_seqs"]]
            criteria = StoppingCriteriaList([StopOnTokenSequences(stop_token_seqs=stop_tokens)])
            generate(**kwargs, stopping_criteria=criteria)
    except PrefillCaptured:
        emit("finished", backend=backend, decoder_forwards=calls,
             snapshot_sha256=sha256(snapshot.read_bytes()).hexdigest())
    else:
        raise RuntimeError("Generation returned before the diagnostic captured the first decoder forward")
    finally:
        pre.remove()
        post.remove()
        if backend == "cpu":
            del decoder.generate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("cpu", "mps"))
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    main(args.backend, args.snapshot)
