"""Pinned tokenizer and recorded or scripted ASR replies, with no model inference."""

import json
from pathlib import Path
from types import SimpleNamespace
import unicodedata

import pytest
import torch
from transformers import AutoTokenizer
import yaml

from speechflow import audio
from speechflow.audio import microphone
from speechflow.audio.hojo import (
    PINNED_GENERATION_CAP, GenerationObservation, _observe_wav_generation, generation_cap_warning,
    generation_loop_warning, infer_wav,
)
from speechflow.config import Settings
from speechflow.contracts import AudioSegment, AudioSource
from speechflow.extraction import extract
from speechflow.extraction.backend import BackendReply


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "reports/qa/real-full-long-reproducer.json"
BISECTION = ROOT / "reports/hojo/long-cpu-bisection.jsonl"
COUNT = "cinq kilos de poissons"


@pytest.fixture(scope="module")
def local_model_metadata():
    bundle = Settings().hojo_dir
    tokenizer_dir = bundle / "Qwen3-4B-Instruct-2507"
    config = yaml.safe_load((bundle / "config.yaml").read_text())
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    assert config["generate"]["max_new_tokens"] == PINNED_GENERATION_CAP
    return SimpleNamespace(tokenizer=tokenizer, config=SimpleNamespace(**config))


def _tokens(model, text):
    return model.tokenizer(text, add_special_tokens=False)["input_ids"]


def _near_cap_text(model, tail):
    text = f"Il reste {COUNT}. {tail}"
    tokens = _tokens(model, text)
    assert len(tokens) > PINNED_GENERATION_CAP
    return model.tokenizer.decode(tokens[:PINNED_GENERATION_CAP], clean_up_tokenization_spaces=False)


def _raw_cases(model, case):
    if case == "archived":
        return json.loads(ARCHIVE.read_text())["unchanged_hojo_result"], COUNT, 4
    if case == "short_natural_repeat":
        text = f"Il reste {COUNT} dans la réserve et {COUNT} dans la chambre froide."
        return {"key": "natural-repeat", "text": text}, COUNT, 2
    if case == "unicode":
        text = "Il reste cinq kilos de cafe\u0301.  J’ai rangé le registre près de l’entrée."
        return {"key": "unicode", "text": text}, "cinq kilos de café", 1
    if case == "near_cap_short_loop":
        return {"key": "short-loop", "text": _near_cap_text(model, "encore " * 210)}, COUNT, 1
    tail = (
        "Ensuite je ferme la porte et je prépare le chariot pour la réception du matin. "
        "Le camion arrive par le quai derrière le bâtiment. Le conducteur attend auprès du portail "
        "pendant que la gardienne vérifie le badge et ouvre les portes. Je retrouve le planning "
        "sur le bureau près de l'imprimante et je signe la feuille de présence avec un stylo bleu. "
        "Dans le couloir la lumière clignote et mon collègue cherche le numéro du technicien. "
        "Nous rangeons les cartons vides sous l'étagère et nous laissons le passage dégagé. "
        "Avant la pause je retourne à l'accueil, où le téléphone sonne. La responsable répond, "
        "puis elle conduit les visiteurs à la salle de réunion du premier étage. "
        "Dehors les nuages arrivent de l'ouest et le vent pousse les feuilles vers le parking. "
        "Une employée ferme calmement les fenêtres, tandis que je rédige le rapport du soir "
        "pour transmettre le journal et les messages au responsable de la prochaine équipe."
    )
    return {"key": "nonrepeating", "text": _near_cap_text(model, tail)}, COUNT, 1


class ScriptedExtraction:
    """Only chat is replaced. The production extraction validator sees genuine spans."""

    def __init__(self, quote, occurrences):
        self.quote = quote
        self.occurrences = occurrences

    def identify(self):
        return {"backend": "scripted-loop-qa"}

    def chat(self, *_args, **_kwargs):
        items = [
            dict(segment="S1", quote=self.quote, operation="count", ref=f"o{i + 1}",
                 product_spoken=self.quote.split()[-1], quantity_text="cinq", quantity="5",
                 qualifier="exact", unit_spoken="kilos", package=None, location=None,
                 correction_of=None)
            for i in range(self.occurrences)
        ]
        return BackendReply(json.dumps({"language": "fr", "observations": items}), "stop")


def _segment(raw, path):
    return AudioSegment(
        segment_id=raw["key"], source_id=path.stem, ordinal=0,
        input_start=0, core_start=0, core_end=512, input_end=512,
    )


def _source(path, source_id):
    return AudioSource(source_id=source_id, kind="file", digest="scripted", name=path.name,
                       native_rate=16000, native_channels=1, sample_count=512,
                       channel_mix="mono")


@pytest.mark.parametrize(
    "case,expected",
    [
        ("archived", "needs_review"),
        ("short_natural_repeat", "ok"),
        ("unicode", "ok"),
        ("near_cap_short_loop", "needs_review"),
        ("near_cap_nonrepetition", "needs_review"),
    ],
)
def test_file_review_and_extraction(monkeypatch, tmp_path, local_model_metadata, case, expected):
    model = local_model_metadata
    raw, quote, occurrences = _raw_cases(model, case)
    path = tmp_path / "scripted.f32le"
    path.write_bytes(b"\x00" * 512 * 4)
    seg = _segment(raw, path)
    monkeypatch.setattr(audio, "segment_pcm", lambda *_: [seg])
    monkeypatch.setattr(audio, "silero_provenance", lambda: {"source": "scripted"})
    monkeypatch.setattr(audio, "verify_local_bundle", lambda *_: None)
    monkeypatch.setattr(audio, "load_local_model", lambda *_: model)
    monkeypatch.setattr(audio, "_model_placement", lambda *_: ("cpu", "scripted"))
    monkeypatch.setattr(audio, "infer_wav", lambda *_args, **_kwargs: raw)

    result = audio._transcribe_pcm(path, _source(path, seg.source_id), Settings(), {})
    [record] = result.segments
    count = len(_tokens(model, raw["text"]))
    if case == "archived" or case.startswith("near_cap_"):
        assert count == PINNED_GENERATION_CAP
    else:
        assert count < PINNED_GENERATION_CAP - 4
    assert result.status == record.status == expected, result.errors
    assert record.raw == raw
    assert record.normalized_text == " ".join(unicodedata.normalize("NFC", raw["text"]).split())
    assert all(getattr(record, k) == v for k, v in seg.model_dump().items())
    if expected == "needs_review":
        assert record.errors[0].startswith("hojo_possible_generation_cap: 200 retokenized tokens")
        assert record.errors[1:] == ([
            "hojo_possible_generation_loop: 200 retokenized tokens near the "
            "200-token cap and at least 3 consecutive copies of a 48-token block"
        ] if case == "archived" else [])
    else:
        assert record.errors == []

    extracted = extract([record], run_id=f"loop-{case}", settings=Settings(),
                        backend=ScriptedExtraction(quote, occurrences))
    assert extracted.status == expected, extracted.errors
    assert len(extracted.observations) == occurrences
    spans = set()
    for observation in extracted.observations:
        assert observation.status == expected
        assert (observation.product_spoken, observation.quantity, observation.unit) == (
            quote.split()[-1], "5", "kg",
        )
        [span] = observation.evidence
        assert record.normalized_text[span.start:span.end] == span.quote == quote
        spans.add((span.start, span.end))
        if expected == "needs_review":
            assert any("asr_needs_review:" in e and "hojo_possible_generation_cap:" in e
                       for e in observation.errors)
    assert len(spans) == occurrences


def test_mic_queue_uses_same_review_and_keeps_provisional(monkeypatch, tmp_path, local_model_metadata):
    from io import StringIO

    raw = json.loads(ARCHIVE.read_text())["unchanged_hojo_result"]
    pcm = tmp_path / "capture.f32le"
    pcm.write_bytes(b"\x00" * 512 * 4)
    segment = _segment(raw, pcm)
    monkeypatch.setattr(microphone, "verify_local_bundle", lambda *_: None)
    monkeypatch.setattr(microphone, "load_local_model", lambda *_: local_model_metadata)
    monkeypatch.setattr(microphone, "_model_placement", lambda *_: ("cpu", "scripted"))
    monkeypatch.setattr(audio, "infer_wav", lambda *_args, **_kwargs: raw)
    asr = microphone._QueuedASR(Settings(), pcm, SimpleNamespace(origin_s=None))
    asr.submit(segment, "eof")
    asr.queue.put_nowait(None)
    provisional = StringIO()
    with pcm.open("rb") as samples:
        asr._transcribe(samples, provisional)
    assert asr.records[0].status == "needs_review"
    row = json.loads(provisional.getvalue())
    assert row["state"] == "provisional"
    assert row["segment"]["raw"] == raw
    assert row["segment"]["status"] == "needs_review"


def test_archived_shorter_bisections_and_unknown_cap(local_model_metadata):
    model = local_model_metadata
    rows = [json.loads(line) for line in BISECTION.read_text().splitlines()]
    shorter = [row for row in rows if row["event"] == "inference"]
    assert [len(_tokens(model, row["raw"]["text"])) for row in shorter] == [61, 55, 63, 53]
    assert all(generation_loop_warning(model, row["raw"]["text"]) is None for row in shorter)
    text = json.loads(ARCHIVE.read_text())["unchanged_hojo_result"]["text"]
    changed = SimpleNamespace(tokenizer=model.tokenizer,
                              config=SimpleNamespace(generate={"max_new_tokens": 220}))
    assert generation_loop_warning(changed, text) is None


class ScriptedDecoder:
    def __init__(self):
        self.called = []

    def generate(self, **kwargs):
        self.called.append(kwargs)
        return self.tokens


class ScriptedHojo:
    """Use real CPU tensor shapes and no model to exercise the per-call hook."""

    device = SimpleNamespace(type="cpu")
    config = SimpleNamespace(generate={"max_new_tokens": PINNED_GENERATION_CAP})

    def __init__(self, feature_frames, cap, ids, *, eos_id=999, decoded_ids=None):
        self.feature_frames = feature_frames
        self.cap = cap
        self.eos_id = eos_id
        self.decoder_model = ScriptedDecoder()
        self.decoder_model.tokens = torch.tensor([ids]) if ids is not None else object()
        self.decoded_ids = ids if decoded_ids is None else decoded_ids

    def tokenizer(self, text, add_special_tokens):
        assert text == "unchanged text" and not add_special_tokens
        return {"input_ids": self.decoded_ids}

    def run_infer(self, paths, batch_size, cuda_enabled):
        assert batch_size == 1 and not cuda_enabled
        self._speech_feature_frames = self.feature_frames
        kwargs = {"inputs_embeds": "the original embedding argument", "eos_token_id": self.eos_id}
        if self.cap is not None:
            kwargs["max_new_tokens"] = self.cap
        received = self.decoder_model.generate(**kwargs)
        assert received is self.decoder_model.tokens
        self.record = {"key": Path(paths[0]).stem, "text": "unchanged text"}
        return [self.record]

    def _run_observed_wav(self, file):
        return _observe_wav_generation(self, file)


@pytest.mark.parametrize(
    "name,frames,cap,ids,terminal_eos,review,loop",
    [
        ("distinct at 200", 200, 200, list(range(200)), False, True, False),
        ("period 4 at 200", 200, 200, list(range(4)) * 50, False, True, False),
        ("long cycle at 200", 200, 200, (list(range(16)) * 13)[:200], False, True, True),
        ("dynamic 110", 50, 110, (list(range(16)) * 7)[:110], False, True, True),
        ("short natural EOS", 8, 26, [15, 17, 21, 999], True, False, False),
        ("EOS one before cap", 50, 110, [1] * 108 + [999], True, False, False),
        ("selected below cap", 50, 110, [1] * 109, False, False, False),
        ("EOS at cap", 50, 110, [1] * 109 + [999], True, True, False),
        ("actual argument wins", 50, 105, list(range(105)), False, True, False),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_observed_decoder_outputs(
    tmp_path, name, frames, cap, ids, terminal_eos, review, loop,
):
    import soundfile

    path = tmp_path / "known.wav"
    soundfile.write(path, [0.0] * 512, 16000, subtype="FLOAT")
    model = ScriptedHojo(frames, cap, ids)
    decoder = model.decoder_model
    prior = decoder.generate
    first_override = lambda **kwargs: prior(**kwargs)
    decoder.generate = first_override

    raw = infer_wav(model, path)
    assert decoder.generate is first_override
    assert raw is model.record
    assert raw == {"key": "known", "text": "unchanged text"}
    assert decoder.called == [
        {"inputs_embeds": "the original embedding argument", "eos_token_id": 999,
         "max_new_tokens": cap}
    ]
    assert model.last_generation == GenerationObservation(frames, cap, len(ids), terminal_eos)
    warning = generation_cap_warning(model, raw["text"])
    assert bool(warning) is review
    if warning:
        assert f"{cap}/{cap}" in warning
        assert ("terminal EOS present" in warning) is terminal_eos
    assert bool(generation_loop_warning(model, raw["text"])) is loop


def test_missing_ids_fall_back_to_exact_feature_limit_and_reset_after_failure(tmp_path):
    path = tmp_path / "fallback.wav"
    model = ScriptedHojo(50, None, None, decoded_ids=list(range(108)))
    decoder = model.decoder_model
    raw = _observe_wav_generation(model, path)[0]
    assert "generate" not in vars(decoder)
    assert model.last_generation == GenerationObservation(50, 110, None, None)
    assert generation_cap_warning(model, raw["text"]).startswith(
        "hojo_possible_generation_cap: 108 retokenized tokens near the 110-token"
    )

    def fail(**_kwargs):
        raise RuntimeError("decoder failed")

    decoder.generate = fail
    with pytest.raises(RuntimeError, match="decoder failed"):
        _observe_wav_generation(model, path)
    assert decoder.generate is fail
    assert model.last_generation is None
