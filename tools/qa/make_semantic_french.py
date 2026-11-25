"""Render the frozen negation and correction fixtures to WAV without playback."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import wave

GOLD = Path(__file__).resolve().parents[2] / "evaluation/fixtures/semantic-audio-fr.jsonl"
RATE = 16_000
VOICE = "Jacques"
WORDS_PER_MINUTE = 160
LICENSE = "Local QA only. No redistribution license asserted for the macOS system voice output."


def generate(out: Path) -> Path:
    gold_bytes = GOLD.read_bytes()
    gold_digest = hashlib.sha256(gold_bytes).hexdigest()
    version = subprocess.check_output(["/usr/bin/sw_vers", "-productVersion"], text=True).strip()
    build = subprocess.check_output(["/usr/bin/sw_vers", "-buildVersion"], text=True).strip()
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    with TemporaryDirectory(prefix=".semantic-fr-", dir=out) as temporary:
        staged = Path(temporary)
        rows = []
        for line in gold_bytes.splitlines():
            gold = json.loads(line)
            audio = staged / f"{gold['id']}.wav"
            subprocess.run(
                ["/usr/bin/say", "-v", VOICE, "-r", str(WORDS_PER_MINUTE), "-o", str(audio),
                 "--file-format=WAVE", f"--data-format=LEI16@{RATE}", gold["reference"]],
                check=True, capture_output=True, timeout=30,
            )
            with wave.open(str(audio), "rb") as handle:
                channels, sample_width, rate, frames = (
                    handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getnframes()
                )
                if (channels, sample_width, rate) != (1, 2, RATE) or frames <= 0:
                    raise ValueError(f"Expected nonempty 16 kHz mono PCM-16 in {audio.name}")
                samples = handle.readframes(frames)
                if len(samples) != frames * sample_width or not any(samples):
                    raise ValueError(f"Incomplete or silent PCM in {audio.name}")
            rows.append({
                **gold, "audio": audio.name, "condition": "synthetic-macos-say",
                "source": f"macOS /usr/bin/say, {VOICE} (fr_FR), {WORDS_PER_MINUTE} words/min",
                "revision": f"macOS {version} ({build}), gold SHA-256 {gold_digest}",
                "license": LICENSE, "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "duration_s": frames / rate, "sample_rate": rate, "frames": frames,
                "channels": channels, "bits": sample_width * 8,
            })
        receipt = staged / manifest.name
        receipt.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                           encoding="utf-8")
        for row in rows:
            (staged / row["audio"]).replace(out / row["audio"])
        receipt.replace(manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="directory for the two WAV files and manifest.jsonl")
    print(generate(parser.parse_args().out))
