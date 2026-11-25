"""Reviewer helpers: build transcript segments and run the production validator (no model)."""
from speechflow.contracts import TranscriptSegment, stable_id
from speechflow.extraction.schema import LLMOutput
from speechflow.extraction.validate import check_reply

SRC = stable_id("file", "review")

def segs(*texts, gap=16000, pad=3200):
    out, t = [], 0
    for i, text in enumerate(texts):
        cs, ce = t + pad, t + pad + 16000
        out.append(TranscriptSegment(segment_id=stable_id("segment", SRC, i), source_id=SRC, ordinal=i,
                   input_start=cs - pad, core_start=cs, core_end=ce, input_end=ce + pad,
                   raw={"key": "k", "text": text}, normalized_text=text))
        t = ce + gap
    return out

def obs(seg, quote, **kw):
    base = dict(segment=seg, quote=quote, more_evidence=[], operation="count", product_spoken=None,
                quantity_text=None, quantity=None, qualifier="exact", unit_spoken=None, package=None,
                location=None, ref="o1", correction_of=None)
    base.update(kw)
    return base

def run(segments, observations, language="fr", known=None, already=None, context=()):
    labels = {f"S{i+1}": s for i, s in enumerate(segments)}
    reply = LLMOutput.model_validate({"language": language, "observations": observations})
    return check_reply(reply, run_id="run_review", labels=labels, context_labels=set(context),
                       known=known or {}, already=already or [])

def show(title, check):
    print(f"== {title}")
    print("  repairable:", check.repairable, " reply_errors:", check.reply_errors)
    for o in check.observations:
        pk = o.packaging.model_dump(exclude_none=True) if o.packaging else None
        print(f"  {o.observation_id[:14]} rev={o.revision} {o.status:12} op={o.operation:10} product={o.product_spoken!r} "
              f"q={o.quantity} max={o.quantity_max} {o.qualifier} unit={o.unit} pack={pk} corr={o.correction_of and o.correction_of[:14]} errors={o.errors}")
