"""Check the predeclared long-file assertions on an installed CLI evaluation."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

from speechflow.contracts import TranscriptSegment


def score(evaluation_path: Path, anchors_path: Path) -> dict:
    report = json.loads(evaluation_path.read_text())
    row, = report["cases"]
    anchors = json.loads(anchors_path.read_text())
    manifest = Path(report["manifest"])
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != report["manifest_sha256"]:
        raise ValueError("Evaluation manifest has changed")
    source = manifest.parent / row["audio"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError("Evaluation input has changed")
    runtime_test = Path(__file__).resolve().parents[2] / "tests/real_models/test_runtime.py"
    checks = runpy.run_path(str(runtime_test))
    asr = row["asr"]
    segments = [TranscriptSegment.model_validate(s) for s in asr["segments"]]
    samples = asr["metrics"]["audio"]["normalized_spool_bytes"] // 4
    audio = SimpleNamespace(segments=segments, source=SimpleNamespace(sample_count=samples))
    content_error = None
    try:
        checks["test_real_hojo_long_audio_boundary_content_and_tail"]((audio, anchors))
    except AssertionError as error:
        content_error = str(error) or "A frozen boundary assertion failed"
    matches = []
    for anchor in anchors:
        text = " ".join(s.normalized_text for s in segments
                        if s.core_start < anchor["sample_end"] and anchor["sample_start"] < s.core_end)
        matches.append({"phrase": anchor["phrase"], "present": checks["_has_anchor"](text, anchor)})
    extraction = row["asr_extraction"]
    checks["assert_real_ollama_attempts"](extraction["attempts"])
    actual = Counter((o["product_spoken"], o["quantity"], o["unit"]) for o in extraction["observations"])
    expected = Counter((a["product"], a["quantity"], "kg") for a in anchors)
    chunk_sizes = asr["metrics"]["audio"]["asr_chunk_sample_counts"]
    bounded = chunk_sizes == [s.input_end - s.input_start for s in segments] and all(
        0 < n <= 400_000 for n in chunk_sizes)
    return {"input_sha256": row["sha256"], "anchors_sha256": hashlib.sha256(anchors_path.read_bytes()).hexdigest(),
            "backend": asr["provenance"]["backend"], "source_samples": samples, "input_samples": chunk_sizes,
            "all_inputs_bounded": bounded, "samples_after_final_core": samples - segments[-1].core_end,
            "anchors": matches, "exact_anchors": sum(a["present"] for a in matches),
            "asr_assertion": "PASS" if bounded and content_error is None else "FAIL",
            "first_content_failure": content_error, "extraction_status": extraction["status"],
            "inventory_assertion": "PASS" if extraction["status"] == "ok" and expected == actual else "FAIL"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("anchors", type=Path)
    args = parser.parse_args()
    result = score(args.evaluation, args.anchors)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["asr_assertion"] == result["inventory_assertion"] == "PASS" else 1)
