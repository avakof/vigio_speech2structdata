# Review of an incomplete negated French quantity

The [real CPU evaluation](../evaluation/semantic-audio-real.json) recorded the
unchanged Hojo text `il ne reste pas deux ou trois` in one keyed segment. The
predeclared script is `Il ne reste pas de sucre.` The archived Qwen response was
`{"language":"fr","observations":[]}` with `done_reason=stop`. Production
`extract` returned `ok` with no observations, errors or repair when replaying
that exact segment and response. The archived ASR-derived score is zero of one
gold negations recovered. The transcript supplies neither a product nor a unit
and cannot identify sugar or ground a reduction, zero or numeric inventory
count. This probe does not establish why Hojo produced the fragment.

| Source and scripted response | Status | Calls | Validated result |
| --- | --- | ---: | --- |
| Archived incomplete fragment, archived empty reply | `ok` | 1 | No rows or review |
| `Le technicien ne reste pas dans le bâtiment.`, empty reply | `ok` | 1 | No rows or review |
| `Il ne reste pas deux ou trois jours avant la réunion.`, empty reply | `ok` | 1 | No rows or review |
| `Il ne reste pas de temps avant la fermeture.`, empty reply twice | `needs_review` | 2 | No rows. Existing absence heuristic misclassifies `temps` as inventory |
| Complete `Il ne reste pas de sucre.`, empty reply twice | `needs_review` | 2 | No rows. Both attempts flag the grounded omission |
| Complete `Il ne reste pas de sucre.`, empty reply then archived repaired reply | `ok` | 2 | One sugar negation, quantity and unit null, evidence at `[0,24)` |
| Original fragment plus a separate 12-bottle milk statement and grounded milk reply | `ok` | 1 | Accepts the 12 milk bottles and leaves the original fragment unmarked |

I recommend a narrow **segment-level incomplete quantity review** in the
inventory workflow for an inventory-style predicate ending in an unresolved
numeric phrase at the end of its clause or available transcript. Keep the
raw ASR and empty reply, record the exact reason as missing unit or product
context, and emit zero inventory rows for the fragment. Preserve independently
grounded rows in the same window. Keep fully expressed factual negation and
time expressions clean. Avoid widening the existing lexical absence detector
to all negation cues since the `pas de temps` control already shows an
unrelated false review. A review flag would improve triage of this fixture.
The ASR-derived gold negation would remain a miss.

## Reproduce without inference

Run at repo root with the installed virtual environment. The only injected
component is the existing scripted backend. The original segment and both
archived reply payloads are loaded from the evaluation report. The generated
[replay receipt](negation-empty-replay.json) contains every segment, attempt,
status, error and evidence span.

```sh
.venv/bin/python - > reports/qa/negation-empty-replay.json <<'PY'
import json
import runpy
from pathlib import Path
from speechflow.config import Settings
from speechflow.contracts import TranscriptSegment
from speechflow.extraction import extract
from speechflow.extraction.backend import BackendReply

helpers = runpy.run_path("tests/qa/test_semantics.py")
source = Path("reports/evaluation/semantic-audio-real.json")
case = next(c for c in json.loads(source.read_text())["cases"]
            if c["id"] == "synthetic-jacques-negation")

def archived(attempt):
    return BackendReply(content=attempt["content"], done_reason=attempt["done_reason"],
                        prompt_tokens=attempt["prompt_tokens"], output_tokens=attempt["output_tokens"],
                        timings_ns=attempt["timings_ns"], wall_s=attempt["wall_s"], raw=attempt["response_raw"])

empty = archived(case["asr_extraction"]["attempts"][0])
repair = archived(case["reference_extraction"]["attempts"][-1])
assert json.loads(empty.content) == {"language": "fr", "observations": []}
original = [TranscriptSegment.model_validate(s) for s in case["asr"]["segments"]]
segments = helpers["transcript"]
neighbor = segments("Il reste douze bouteilles de lait.")[0].model_copy(update={
    "segment_id": "qa-neighbor-seg-1", "source_id": original[0].source_id, "ordinal": 1,
    "input_start": 32000, "core_start": 32000, "core_end": 48000, "input_end": 48000,
})
neighbor_reply = helpers["reply"](helpers["item"](
    "douze bouteilles de lait", "lait", "douze", "12", "bouteilles", segment="S2"))
checks = [
    ("fragment_with_independently_grounded_neighbor", [*original, neighbor], [neighbor_reply, neighbor_reply]),
    ("archived_asr_fragment", original, [empty, empty]),
    ("ordinary_non_inventory_negation", segments("Le technicien ne reste pas dans le bâtiment."), [empty, empty]),
    ("complete_non_inventory_time_count", segments("Il ne reste pas deux ou trois jours avant la réunion."), [empty, empty]),
    ("non_inventory_absence_same_cue", segments("Il ne reste pas de temps avant la fermeture."), [empty, empty]),
    ("grounded_inventory_empty_twice", segments(*case["reference"]), [empty, empty]),
    ("grounded_inventory_archived_repair", segments(*case["reference"]), [empty, repair]),
]
records = {}
for name, transcript, replies in checks:
    backend = helpers["ScriptedChat"](*replies)
    result = extract(transcript, run_id=f"qa-{name}", settings=Settings(offline=True), backend=backend)
    records[name] = {
        "source_segments": [s.model_dump(mode="json") for s in transcript],
        "backend_calls": len(backend.calls),
        "status": result.status,
        "errors": result.errors,
        "observations": [o.model_dump(mode="json") for o in result.observations],
        "attempts": [{k: a[k] for k in ("kind", "content", "user_message", "errors")} for a in result.attempts],
    }
assert records["archived_asr_fragment"]["status"] == "ok"
assert records["archived_asr_fragment"]["observations"] == []
assert records["archived_asr_fragment"]["backend_calls"] == 1
grounded, = records["grounded_inventory_archived_repair"]["observations"]
assert (grounded["status"], grounded["operation"], grounded["product_spoken"], grounded["quantity"], grounded["unit"]) == (
    "ok", "negation", "sucre", None, None)
print(json.dumps({
    "archive": str(source),
    "case": case["id"],
    "mode": "Production extract and validator with caller-owned scripted backend. No inference.",
    "archived_empty_reply": empty.raw,
    "archived_asr_scores": {
        "gold_mentions": case["asr_extraction"]["scores"]["gold_mentions"],
        "covered_gold_mentions": case["asr_extraction"]["scores"]["covered_gold_mentions"],
        "negation": case["asr_extraction"]["scores"]["fields"]["negation"],
    },
    "replays": records,
}, ensure_ascii=False, indent=2))
PY
```
