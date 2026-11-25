"""Reviewer reproducer (no model): the near-cap Hojo guard only flags exact long loops at the
fixed 200-token cap. (a) a non-repetitive reply that fills the cap (possible truncation) is clear;
(b) a loop truncated at the wheel's lower feature-based cap min(200, 2*feat_len+10) for a short
segment is clear. Wheel: hojo_asr_model.py:255-258. Guard: speechflow/audio/hojo.py:21-45."""
from types import SimpleNamespace
from speechflow.audio.hojo import generation_loop_warning

class Tok:
    def __init__(self, ids): self.ids = ids
    def __call__(self, text, add_special_tokens=False): return {"input_ids": self.ids}

def model(ids):
    return SimpleNamespace(config=SimpleNamespace(generate={"max_new_tokens": 200}), tokenizer=Tok(ids))

cases = {
    "200 distinct tokens (cap filled, no loop)": list(range(200)),
    "16-token block x12 = 192 +8 (loop at 200 cap)": (list(range(16)) * 13)[:200],
    "4 s segment: feat_len~50 -> wheel cap 110; 16-token block x7 truncated at 110": (list(range(16)) * 7)[:110],
    "short loop period 4 filling 200": (list(range(4)) * 50),
}
for name, ids in cases.items():
    print(f"{name}: tokens={len(ids)} warning={generation_loop_warning(model(ids), 'x')!r}")
print("wheel effective cap for N samples (12.5 Hz embeddings ~ N/1280):",
      {s: min(200, int(2 * (s * 16000 // 1280)) + 10) for s in (1, 2, 4, 6, 8, 25)})
