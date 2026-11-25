"""Reviewer check (tokenizer file only, no model weights): retokenize archived real Hojo replies with the
pinned Qwen tokenizer.json and apply the CURRENT production guard. Shows whether real replies sat at the
generation cap while the guard stayed silent (R06 observed on real output)."""
import json, sys
from pathlib import Path
from types import SimpleNamespace
from tokenizers import Tokenizer
from speechflow.audio.hojo import generation_loop_warning
tok = Tokenizer.from_file(str(Path.home() / ".cache/speechflow/assets/hojo/Qwen3-4B-Instruct-2507/tokenizer.json"))
class T:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": tok.encode(text, add_special_tokens=False).ids}
model = SimpleNamespace(config=SimpleNamespace(generate={"max_new_tokens": 200}), tokenizer=T())
for f in sys.argv[1:]:
    d = json.load(open(f)); r = d.get("report", d)
    for case in r.get("cases", []):
        for s in (case.get("asr") or {}).get("segments", []):
            n = len(tok.encode(s["raw"]["text"], add_special_tokens=False).ids)
            secs = (s["input_end"] - s["input_start"]) / 16000
            cap = min(200, int(2 * ((s["input_end"] - s["input_start"]) // 1280)) + 10)
            w = generation_loop_warning(model, s["raw"]["text"])
            print(f"{Path(f).name} seg{s['ordinal']} input={secs:.2f}s tokens={n} approx_wheel_cap={cap} stored_status={s['status']} current_guard={'FLAG' if w else 'clear'}")
