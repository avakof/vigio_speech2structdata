"""Reviewer check (archived outputs only): compare CPU vs MPS replies for the same segment inputs.
Every MPS reply differs from CPU in its leading words while the remainders largely agree."""
import json
pairs = [("reports/evaluation/synthetic-real.json", "reports/evaluation/synthetic-mps.json")]
for cpu_f, mps_f in pairs:
    c = json.load(open(cpu_f)); m = json.load(open(mps_f)); c = c.get("report", c); m = m.get("report", m)
    for cs, ms in zip(c["cases"][0]["asr"]["segments"], m["cases"][0]["asr"]["segments"]):
        same_input = (cs["input_start"], cs["input_end"]) == (ms["input_start"], ms["input_end"])
        cw, mw = cs["normalized_text"].split(), ms["normalized_text"].split()
        k = 0
        while k < min(len(cw), len(mw)) and cw[-1 - k] == mw[-1 - k]: k += 1
        print(f"same_input={same_input} CPU first words={' '.join(cw[:4])!r} | MPS first words={' '.join(mw[:4])!r} | common suffix words={k}/{len(cw)}")
for f in ("reports/hojo/cpu.jsonl", "reports/hojo/mps.jsonl"):
    for line in open(f):
        r = json.loads(line)
        if r.get("event") == "inference":
            print(f, "iter", r["iteration"], repr(r["result"]["text"]))
