#!/usr/bin/env python3
"""Reproduce the pinned Hojo decoder's BF16 grouped-query MPS attention shape."""

import argparse
import json
from types import SimpleNamespace

import torch
from torch.nn.functional import scaled_dot_product_attention
from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("grouped-sdpa", "repeated-kv-eager"))
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(777)
    query_cpu = torch.randn(4, 32, 1, 128, generator=generator, dtype=torch.bfloat16)
    key_cpu = torch.randn(4, 8, 47, 128, generator=generator, dtype=torch.bfloat16)
    value_cpu = torch.randn(4, 8, 47, 128, generator=generator, dtype=torch.bfloat16)
    query, key, value = (tensor.to("mps") for tensor in (query_cpu, key_cpu, value_cpu))
    print(json.dumps({
        "event": "input", "mode": args.mode, "torch": torch.__version__, "device": str(query.device),
        "dtype": str(query.dtype), "query": list(query.shape), "key": list(key.shape),
    }), flush=True)
    if args.mode == "grouped-sdpa":
        result = scaled_dot_product_attention(query, key, value, enable_gqa=True)
        torch.mps.synchronize()
        print(json.dumps({"event": "output", "shape": list(result.shape)}))
        return

    head_groups = SimpleNamespace(num_key_value_groups=4, training=False)
    result, _ = eager_attention_forward(head_groups, query, key, value, None, 128 ** -0.5)
    reference, _ = eager_attention_forward(
        head_groups, query_cpu, key_cpu, value_cpu, None, 128 ** -0.5
    )
    torch.mps.synchronize()
    assert list(result.shape) == [4, 1, 32, 128] and result.device.type == "mps"
    difference = (result.float().cpu() - reference.float()).abs().max().item()
    assert torch.allclose(result.float().cpu(), reference.float(), atol=0.025, rtol=0.025)
    print(json.dumps({
        "event": "output", "shape": list(result.shape), "device": str(result.device),
        "dtype": str(result.dtype), "max_abs_difference_cpu": difference,
    }))


if __name__ == "__main__":
    main()
