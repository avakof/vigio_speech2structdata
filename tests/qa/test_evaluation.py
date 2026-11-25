"""Counter and reporting tests use fabricated inference solely as evaluator inputs."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from speechflow import evaluation
from speechflow.config import Settings
from speechflow.contracts import EvidenceSpan, ExtractionResult, ValidatedObservation


def case_file(tmp_path, *cases):
    manifest = tmp_path / "cases.jsonl"
    manifest.write_text("\n".join(json.dumps(case) for case in cases))
    return manifest


def gold(product, quantity, **extra):
    return dict(product_spoken=product, quantity=quantity, unit="kg", **extra)


def test_score_uses_all_cases_when_predictions_reject_or_misrecognize(tmp_path, monkeypatch):
    expected = [gold("pommes", "6"), gold("café", "2"), gold("sucre", None, operation="negation")]
    manifest = case_file(tmp_path, {"reference": "six kilos de pommes et deux kilos de café",
                                    "expected_observations": expected})
    monkeypatch.setattr(evaluation.assets, "extractor_digest", lambda settings: evaluation.assets.DIGEST)
    predictions = [
        ValidatedObservation(observation_id="x", product_spoken="poires", quantity="6", unit="kg"),
        ValidatedObservation(observation_id="y", product_spoken="café", quantity="2", unit="kg",
                             status="needs_review"),
        ValidatedObservation(observation_id="z", product_spoken="sucre", operation="count"),
    ]
    monkeypatch.setattr(evaluation.extraction, "extract",
                        lambda *args, **kwargs: ExtractionResult(status="needs_review",
                                                                 observations=predictions))
    report = evaluation.evaluate(manifest, tmp_path, Settings())
    ref = report["extraction"]["reference"]
    assert json.loads((tmp_path / "evaluation.json").read_text()) == report
    assert ref["gold_coverage"] == pytest.approx(1 / 3)
    assert ref["abstention_rate"] == pytest.approx(1 / 3)
    assert ref["fields"]["quantity"] == {
        "tp": 0, "fp": 1, "fn": 2, "predicted": 1, "expected": 2, "precision": 0.0, "recall": 0.0}
    assert ref["fields"]["negation"]["fn"] == 1
    assert ref["fields"]["unit"]["fn"] == 3
    assert report["extraction"]["asr"]["cases"] == 0


def test_wer_eligible_conditions_and_separate_asr_extraction(tmp_path, monkeypatch):
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"Mock audio, evaluator test only")
    digest = hashlib.sha256(audio.read_bytes()).hexdigest()
    annotated = {"audio": audio.name, "sha256": digest, "reference": "Deux kilos. L’été !",
                 "expected_observations": [gold("café", "2")], "duration_s": 1,
                 "source": "mock evaluator test", "revision": "1", "human_verified": True}
    manifest = case_file(tmp_path, {**annotated, "id": "verified", "condition": "clean"},
                         {**annotated, "id": "synthetic", "condition": "synthetic",
                          "synthetic": True, "human_verified": False})
    monkeypatch.setattr(evaluation.assets, "hojo_issues", lambda settings: [])
    monkeypatch.setattr(evaluation.assets, "extractor_digest", lambda settings: evaluation.assets.DIGEST)
    monkeypatch.setattr(evaluation.extraction, "extract", lambda *args, **kwargs: ExtractionResult(
        status="ok", observations=[ValidatedObservation(observation_id="ref",
            product_spoken="café", quantity="2.00", unit="kg")]))
    calls = []

    def fake_process(path, *, mode, settings, out):
        calls.append((path, mode, out))
        return SimpleNamespace(source=SimpleNamespace(sample_count=16000, digest=digest),
                               segments=evaluation._reference_segments(
                                   evaluation.Case(reference="Deux kilos, l'éte.", audio=audio.name), "asr"),
                               transcription_status="ok", extraction_status="ok", status="ok",
                               errors=[], attempts=[], observations=[], provenance={}, metrics={})

    monkeypatch.setattr(evaluation.pipeline, "process", fake_process)
    report = evaluation.evaluate(manifest, tmp_path / "report", Settings())
    assert [call[1] for call in calls] == ["run", "run"]
    assert report["wer"]["eligible_clips"] == 1
    assert report["wer"]["reference_words"] == 3
    assert report["wer"]["edits"] == 1
    assert report["wer"]["by_condition"]["clean"]["wer"] == pytest.approx(1 / 3)
    assert "synthetic" not in report["wer"]["by_condition"]
    assert report["extraction"]["reference"]["fields"]["quantity"]["recall"] == 1
    assert report["extraction"]["asr"]["fields"]["quantity"]["fn"] == 2
    assert report["extraction"]["asr"]["gold_coverage"] == 0


def test_wrong_correction_target_is_fp_and_fn():
    expected = [
        evaluation.GoldObservation(**gold("riz", "3")),
        evaluation.GoldObservation(**gold("riz", "4")),
        evaluation.GoldObservation(**gold("riz", "5", operation="correction", correction_of=0)),
    ]
    predicted = [
        ValidatedObservation(observation_id=f"p{i}", product_spoken="riz",
                             quantity=str(i + 3), unit="kg", correction_of="p1" if i == 2 else None,
                             operation="correction" if i == 2 else "count")
        for i in range(3)
    ]
    score = evaluation._score(expected, predicted)
    assert score["fields"]["quantity"]["tp"] == 3
    assert score["fields"]["correction"]["tp"] == 0
    assert score["fields"]["correction"]["fp"] == score["fields"]["correction"]["fn"] == 1


def test_repeated_product_alignment_uses_evidence_chronology_not_reply_order():
    segments = evaluation._reference_segments(
        evaluation.Case(reference_segments=["Trois kilos de riz", "Cinq kilos de riz"],
                        audio="sample.wav"), "rice")
    expected = [evaluation.GoldObservation(**gold("riz", quantity)) for quantity in ("3", "5")]
    predictions = [
        ValidatedObservation(observation_id=segment.segment_id, product_spoken="riz",
                             quantity=quantity, unit="kg",
                             evidence=[EvidenceSpan(segment_id=segment.segment_id, start=0,
                                                    end=len(segment.normalized_text),
                                                    quote=segment.normalized_text)])
        for quantity, segment in zip(("3", "5"), segments, strict=True)
    ]
    score = evaluation._score(expected, predictions[::-1], segments)
    assert score["fields"]["quantity"]["tp"] == 2


def test_pinned_corpus_reference_cannot_be_replaced_without_losing_wer_eligibility():
    root = Path(__file__).resolve().parents[2]
    first = (root / "evaluation/fleurs-fr/manifest.jsonl").read_text().splitlines()[0]
    row = evaluation.Case.model_validate_json(first)
    assert evaluation._wer_basis(row)
    row.reference = "Ce texte n'est pas la référence du corpus."
    assert evaluation._wer_basis(row) is None


def test_blocked_models_do_not_turn_absent_predictions_into_measurement(tmp_path, monkeypatch):
    manifest = case_file(tmp_path, {"reference": "Deux kilos de café.",
                                   "expected_observations": [gold("café", "2")]})
    monkeypatch.setattr(evaluation.assets, "extractor_digest", lambda settings: None)
    report = evaluation.evaluate(manifest, tmp_path, Settings())
    assert report["status"] == "error"
    ref = report["extraction"]["reference"]
    assert ref["blocked_cases"] == 1
    assert ref["measurement_status"] == "NOT MEASURED"
    assert ref["fields"]["quantity"]["expected"] == 1
    assert ref["fields"]["quantity"]["fn"] == 1


def test_invalid_manifest_names_line_and_duplicate_key(tmp_path):
    case = {"id": "duplicate", "reference": "Un kilo de café.", "expected_observations": []}
    manifest = case_file(tmp_path, case, case)
    with pytest.raises(ValueError, match=r"cases\.jsonl:2: duplicate case ID"):
        evaluation.evaluate(manifest, tmp_path, Settings())
