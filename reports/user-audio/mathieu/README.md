# French Canadian inventory MP3 test

**The file and table exports completed. Inventory extraction needs review.** On November 25, 2026, the installed CLI transcribed 13 segments and returned 27 candidate rows on the run that covered both extraction windows. The validator held all 27. This table is a review queue for the recording. A blank quantity means unknown.

[Download the rows as CSV](observations.csv) · [Original CLI observation table](observations.txt) · [Verification receipt](verification.json)

## Comparison with the supplied script

**Hojo’s word error rate against the supplied script was 18.18%** on this 127.399-second synthetic recording. The two saved runs had identical transcripts. Across 275 reference words and 279 hypothesis words, one minimum-edit alignment counted 40 substitutions, 3 deletions and 7 insertions. The comparison removed 15 bracketed performance cues and retained the standalone “18”. The project’s word normalization preserves accents and number spellings, so a change from a spoken number to digits can contribute an edit. The project’s human-recorded French benchmark retains its original score. [Score and method](script-comparison.json)

The reference contains **29 final product-and-location entries**. A compound amount, such as packages plus loose units, belongs to one entry. A spoken correction changes the final amount for that product at that site. Only rows accepted by the validator can match a final reference entry.

| Saved extraction | Rows accepted | Rows held for review | Correct final entries | Precision of accepted entries | Recall |
|---|---:|---:|---:|---:|---:|
| Project defaults | 1 | 10 of 11 (90.91%) | 0 of 29 | 0% | 0% |
| Larger bounded budgets | 0 | 27 of 27 (100%) | 0 of 29 | Undefined (zero accepted) | 0% |

The default run’s single accepted bottle count was assigned to the wrong product. Every row from the larger-budget run omitted the location. Its rows cited only segments 1–9 of the 13 when numbered from one. The transcript contained the last site and the closing correction, but the exported rows did not cite either passage. The private comparison preserves the script, word alignment and expected counts for each site.

## Results on the same recording

| CPU run | Context / output token limits | Extraction windows returning candidates | Rows | Accepted by validator | Held for review | Pipeline time |
|---|---:|---:|---:|---:|---:|---:|
| Project defaults | 8,192 / 2,048 | 1 of 2 | 11 | 1 | 10 | 305.11 s |
| Larger bounded budgets | 32,768 / 6,144 | 2 of 2 | 27 | 0 | 27 | 963.54 s |

The default run filled its output budget on the first window. The prior reply and proposed repair exceeded the available context. Its only accepted row assigned one bottle to *vodka zéro*. The supplied script assigns the bottle to another product and gives vodka a final count of zero at that site. [Default rows](default/observations.csv) · [Default receipt](default/verification.json)

The second run used the same model revisions, CPU backend and audio. All 13 raw Hojo replies matched the first run exactly. Qwen returned complete responses for both windows and made one repair per window. The corrections still lacked a reliable prior observation, and several proposed evidence spans could not be matched to the cited segment. The two runs therefore represent separate extraction attempts. Their rows cannot be added together. [Rerun settings](settings.toml)

## Exported observations

The row numbers below start at 1. The CLI and CSV ordinals start at 0. Each row retains its operation and review status. A correction without a target remains unresolved.

| Row | Product as extracted | Quantity | Unit | Packaging | Location | Operation | Corrects row | Status |
|---:|---|---:|---|---|---|---|---:|---|
| 1 | marais chambre froide | — | — | — | — | count | — | needs_review |
| 2 | premier comptage | — | — | — | — | count | — | needs_review |
| 3 | tomates cerises | 2 | caisse | 2 caisse × 5 kg | — | count | — | needs_review |
| 4 | crème fraîche | — | l | — | — | count | — | needs_review |
| 5 | crème fraîche | 4 | sauts | — | — | correction | — | needs_review |
| 6 | pavé de saumon | 0 | kg | — | — | negation | — | needs_review |
| 7 | oeufs | 31 | plateau | 31 plateau | — | count | — | needs_review |
| 8 | beurre | 7 | plaquette | 7 plaquette × 250 g | — | count | — | needs_review |
| 9 | citron | 17 | plaquettes | — | — | count | — | needs_review |
| 10 | basilic | — | cagette | — | — | count | — | needs_review |
| 11 | farine | 55 | sac | 55 sac × 25 kg | — | count | — | needs_review |
| 12 | huile d'olive | — | l | — | — | count | — | needs_review |
| 13 | pommes de terre bintje | 4 | sac | 4 sac × 10 kg | — | count | — | needs_review |
| 14 | vin rouge côte du rhône | — | bouteille | — | — | count | — | needs_review |
| 15 | bouteille | 15 | bouteille | — | — | count | — | needs_review |
| 16 | sirop de menthe | 0 | bouteille | — | — | negation | — | needs_review |
| 17 | vodka | 2 | bouteille | 2 bouteille × 70 cl | — | count | — | needs_review |
| 18 | farine | — | sac | — | — | count | — | needs_review |
| 19 | huile d'olive | 4 | bidon | 4 bidon | — | count | — | needs_review |
| 20 | tomates cerises | 3 | caisse | 3 caisse | — | count | — | needs_review |
| 21 | oeufs | 6 | plateau | 6 plateau | — | count | — | needs_review |
| 22 | beurre | 12 | plaquettes | — | — | count | — | needs_review |
| 23 | crème fraîche | — | seau | — | — | count | — | needs_review |
| 24 | citron | — | cagette | — | — | count | — | needs_review |
| 25 | basilic | 22 | bottes | — | — | count | — | needs_review |
| 26 | sirop d'orange | — | — | — | — | count | — | needs_review |
| 27 | vodka | 0 | bouteille | — | — | count | — | needs_review |

## Test and verification

The source was a user-supplied MP3 labeled ElevenLabs Mathieu, a French Canadian narrator. It decoded to 127.399 seconds from 44.1 kHz mono. Speechflow ran on an Apple M3 Max with the CPU backend, Silero VAD, HojoAI/Hojo-ASR-Multi-V1 and loopback Ollama qwen3:8b. The offline flag was enabled for both runs. Process-level egress blocking was tested separately in the [release verification](../../integration/release-verification.json).

Both runs used the supplied audio as input. The user supplied the script after both runs. The verification receipts describe the evidence available at inference time. The comparison above uses the saved output and the subsequently supplied reference. The inventory table preserves the original model output and review status. The recording, verbatim transcript, quoted evidence and database are held locally. The public CSV contains the observation fields, including unit, packaging, location, correction references, evidence coordinates and validation errors. The public CSV omits the verbatim evidence-quote column and source excerpts embedded in diagnostic messages.

The rerun spent 490.35 s on ASR and 457.60 s on extraction. All ASR inputs stayed within 400,000 samples (25 s at 16 kHz), including padding and overlap. Both CLI invocations returned exit 11 for a saved result requiring review. SQLite integrity and foreign-key checks passed. For each run, a fresh CLI export read the saved run from SQLite and reproduced the original table and full local CSV byte for byte. Each ordered observation matched its database fields, status and correction link. The local manifest matched the database copy. [Validation hashes and timings](verification.json)

This is a follow-up to the [signed release assessment](../../review/architect-final.md). That assessment predates receipt of this file. The follow-up verifies file processing and export. The subsequent script comparison measures recognition and final inventory agreement for this single synthetic recording.

## Reproduce with a local copy

Set `audio_path` to the original MP3. With the installed environment and local Ollama running, the following commands reproduce the larger-budget configuration. The run ID is available in the saved JSON even when the CLI returns 11.

```sh
audio_path="/path/to/recording.mp3"
out="outputs/mathieu-reproduction"
db="$out/inventory.sqlite3"
mkdir -p "$out"
.venv/bin/speechflow --offline --backend cpu --database "$db" \
  --config reports/user-audio/mathieu/settings.toml run "$audio_path" --out "$out/run"
run_id=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$out/run/run.json")
.venv/bin/speechflow --offline --database "$db" export "$run_id" --format table
.venv/bin/speechflow --offline --database "$db" export "$run_id" --format csv
```
