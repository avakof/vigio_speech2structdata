"""Reviewer reproducer (no weights): drive the production entry speechflow.audio._infer_segment ->
infer_wav -> _observe_wav_generation -> generation_cap_warning/generation_loop_warning with a fake
Hojo-shaped model whose run_infer mimics hojo_asr 0.1.3 (hojo_asr_model.py:236-266): encode_speech sets
the feature length, the decoder cap is max(10, min(200, 2*feat_len+10)), and decoder_model.generate
returns a (1, L) id tensor for the inputs_embeds-only path. Independent of the engineer's tests."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
from speechflow.audio import _infer_segment
from speechflow.audio.hojo import _observe_wav_generation
from speechflow.contracts import AudioSegment, stable_id

EOS = 151645

class Decoder:
    def __init__(self, ids): self.ids = ids; self.calls = 0
    def generate(self, *args, **kw):
        self.calls += 1
        return torch.tensor([self.ids[: kw["max_new_tokens"]]])

class FakeHojo:
    def __init__(self, feat_frames, ids, text, generate_calls=1):
        self.feat_frames, self.text, self.generate_calls = feat_frames, text, generate_calls
        self.decoder_model = Decoder(ids)
        self.device = SimpleNamespace(type="cpu")
        self.config = SimpleNamespace(generate={"max_new_tokens": 200})
        self.tokenizer = lambda t, add_special_tokens=False: {"input_ids": [int(w[1:]) for w in t.split()]}
    def _run_observed_wav(self, file): return _observe_wav_generation(self, file)
    def run_infer(self, files, batch_size=1, cuda_enabled=False):
        self._speech_feature_frames = self.feat_frames          # LocalHojo.encode_speech
        cap = max(10, min(200, 2 * self.feat_frames + 10))        # wheel lines 255-258
        for _ in range(self.generate_calls):
            self.decoder_model.generate(inputs_embeds=object(), max_new_tokens=cap, eos_token_id=EOS)
        return [{"key": Path(files[0]).stem, "text": self.text}]

def words(ids): return " ".join(f"w{i}" for i in ids if i != EOS)

def run(name, seconds, ids, generate_calls=1):
    n = int(seconds * 16000)
    with tempfile.TemporaryDirectory() as d:
        work = Path(d); pcm = work / "src.f32le"
        pcm.write_bytes(np.zeros(n, dtype="<f4").tobytes())
        seg = AudioSegment(segment_id=stable_id("segment", name), source_id="file_x", ordinal=0,
                           input_start=0, core_start=0, core_end=n, input_end=n)
        feat = n // 1280
        m = FakeHojo(feat, ids, words(ids), generate_calls)
        with pcm.open("rb") as f:
            t = _infer_segment(m, f, seg, work, 25)
        g = m.last_generation
        print(f"{name}: wheel_cap={max(10, min(200, 2*feat+10))} observed={g.max_new_tokens}/{g.output_tokens} eos={g.terminal_eos} "
              f"-> status={t.status} errors={t.errors}")

distinct = list(range(1000, 1300))
run("A 4 s, 110 of 110 non-repeating, no EOS", 4, distinct[:110])
run("B 4 s, 110 of 110 ending in EOS", 4, distinct[:109] + [EOS])
run("C 4 s, early EOS at 60", 4, distinct[:59] + [EOS])
run("D 4 s, short period-4 loop filling 110", 4, (list(range(1000, 1004)) * 40)[:110])
run("E 4 s, 16-token block loop filling 110", 4, (list(range(1000, 1016)) * 10)[:110])
run("F 25 s, 200 of 200 non-repeating", 25, distinct[:200])
run("G 25 s, 140 tokens with EOS (normal dense speech)", 25, distinct[:139] + [EOS])
run("H 1 s, cap 34, 34 tokens", 1, distinct[:34])
run("I 4 s, two decoder calls (observation unavailable) 110 tokens", 4, distinct[:110], generate_calls=2)
run("J 4 s, two decoder calls, 60 tokens", 4, distinct[:59] + [EOS], generate_calls=2)
