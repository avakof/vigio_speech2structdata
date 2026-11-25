"""Reviewer check (no model): raw MPS long-file segments (reports/evaluation/synthetic-mps.json, older
implementation f4744f) contain mixed-script/English hallucinations, yet the CURRENT lexical language
veto returns 'fr' and the audio stage has no script check, so accepted observations citing them stay ok."""
import json, unicodedata
from speechflow.extraction.language import assess
d = json.load(open("reports/evaluation/synthetic-mps.json")); r = d.get("report", d)
for case in r["cases"]:
    for s in case["asr"]["segments"]:
        t = s["normalized_text"]
        scripts = sorted({unicodedata.name(c, "?").split()[0] for c in t if c.isalpha()})
        print(f"segment {s['ordinal']} status={s['status']} assess={assess(t)!r} scripts={scripts} head={t[:70]!r}")
    acc = [o for o in case.get("asr_extraction", {}).get("observations", []) if o["status"] == "ok"]
    print("accepted ok observations:", [(o["product_spoken"], o["quantity"], o["unit"], o["evidence"][0]["segment_id"][-8:]) for o in acc])
