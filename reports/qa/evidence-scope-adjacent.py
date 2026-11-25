"""Probe clause validation against scripted replies in the frozen environment."""

import hashlib
import json
from pathlib import Path

from speechflow.extraction import validate
from test_semantics import item, reply, run, transcript


def count(product, quantity, qualifier="exact", maximum=None):
    return ("count", product, quantity, maximum, "kg", qualifier)


def probe(name, texts, candidate, expected, *, review_allowed=False, gap_samples=0):
    segments = transcript(*texts)
    if gap_samples:
        second = segments[1]
        segments[1] = second.model_copy(update={
            bound: getattr(second, bound) + gap_samples
            for bound in ("input_start", "core_start", "core_end", "input_end")
        })
    scripted = reply(candidate)
    result, calls = run(segments, scripted, scripted)
    assert 1 <= len(calls) <= 2 and result.attempts[0]["content"] == scripted.content
    for observation in result.observations:
        for span in observation.evidence:
            source = next(s.normalized_text for s in segments if s.segment_id == span.segment_id)
            assert source[span.start:span.end] == span.quote
    accepted = [(o.operation, o.product_spoken, o.quantity, o.quantity_max, o.unit, o.qualifier)
                for o in result.observations if o.status == "ok"]
    passed = accepted == expected or (review_allowed and not accepted)
    print(json.dumps({
        "case": name, "passed": passed, "gap_samples": gap_samples, "texts": texts,
        "scripted_reply": json.loads(scripted.content), "expected_ok": expected,
        "review_allowed": review_allowed, "result_status": result.status,
        "result_errors": result.errors, "accepted": accepted,
        "observations": [o.model_dump(mode="json") for o in result.observations],
    }, ensure_ascii=False))
    return passed


def main():
    validator = Path(validate.__file__)
    print(json.dumps({"validator": str(validator),
                      "sha256": hashlib.sha256(validator.read_bytes()).hexdigest()}))
    approximate = [count("farine", "3", "approximate")]
    split = ("Il reste environ", "trois kilos de farine.")
    tests = [
        probe("omitted prefix in previous segment", split,
              item("trois kilos de farine", "farine", "trois", "3", "kilos", segment="S2"),
              approximate, review_allowed=True),
        probe("prefix in evidence but omitted from quantity_text", split,
              item("environ", "farine", "trois", "3", "kilos",
                   more_evidence=[{"segment": "S2", "quote": "trois kilos de farine"}]),
              approximate, review_allowed=True),
        probe("complete prefix and quantity across two segments", split,
              item("environ", "farine", "environ trois", "3", "kilos", qualifier="approximate",
                   more_evidence=[{"segment": "S2", "quote": "trois kilos de farine"}]),
              approximate),
        probe("omitted lower range endpoint in previous segment",
              ("Il reste entre deux et", "cinq kilos de farine."),
              item("cinq kilos de farine", "farine", "cinq", "5", "kilos", segment="S2"),
              [count("farine", "2", "range", "5")], review_allowed=True),
        probe("omitted trailing qualifier in the following segment",
              ("Il reste trois kilos de farine", "environ."),
              item("trois kilos de farine", "farine", "trois", "3", "kilos"),
              approximate, review_allowed=True),
        probe("trailing qualifier included in quote and omitted from quantity_text",
              ("Il reste trois kilos de farine environ.",),
              item("trois kilos de farine environ", "farine", "trois", "3", "kilos"),
              approximate, review_allowed=True),
        probe("et is part of a product name", ("Il reste deux kilos de noix et raisins.",),
              item("deux kilos de noix et raisins", "noix et raisins", "deux", "2", "kilos"),
              [count("noix et raisins", "2")]),
        probe("independent positive after punctuated negative segment",
              ("Il ne reste pas de sucre.", "Deux kilos de farine."),
              item("Deux kilos de farine", "farine", "Deux", "2", "kilos", segment="S2"),
              [count("farine", "2")]),
        probe("independent positive with its own predicate",
              ("Il ne reste pas de sucre", "Il reste deux kilos de farine."),
              item("deux kilos de farine", "farine", "deux", "2", "kilos", segment="S2"),
              [count("farine", "2")]),
        probe("independent positive after five-second gap without punctuation",
              ("Il ne reste pas de sucre", "Deux kilos de farine."),
              item("Deux kilos de farine", "farine", "Deux", "2", "kilos", segment="S2"),
              [count("farine", "2")], gap_samples=80_000),
        probe("negative predicate in previous segment",
              ("Il ne reste pas", "Deux kilos de sucre."),
              item("Deux kilos de sucre", "sucre", "Deux", "2", "kilos", segment="S2"),
              []),
        probe("quantity and product from different unpunctuated segments",
              ("Trois kilos de farine", "Cinq kilos de sucre."),
              item("Trois kilos de farine", "farine", "Cinq", "5", "kilos",
                   more_evidence=[{"segment": "S2", "quote": "Cinq kilos de sucre"}]),
              []),
    ]
    print(json.dumps({"passed": sum(tests), "failed": len(tests) - sum(tests)}))
    return int(not all(tests))


if __name__ == "__main__":
    raise SystemExit(main())
