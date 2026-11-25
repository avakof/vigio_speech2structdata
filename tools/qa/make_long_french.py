"""Build a labeled synthetic French inventory clip with a sample timeline."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import wave
from array import array

RATE = 16000
VOICE_RATE = 160
FILLER = (
    "Pour continuer la vérification je retourne vers les étagères de la réserve "
    "pendant que mon collègue termine tranquillement la vérification du petit réfrigérateur"
)
# Each anchor is spoken exactly once and independently annotated.
INVENTORY = (
    ("huit kilos de tomates", "tomates", "8"),
    ("quatre kilos de carottes", "carottes", "4"),
    ("six kilos de courgettes", "courgettes", "6"),
    ("neuf kilos de pommes", "pommes", "9"),
    ("cinq kilos de poires", "poires", "5"),
    ("trois kilos de kiwis", "kiwis", "3"),
    ("sept kilos de citrons", "citrons", "7"),
    ("deux kilos de gingembre", "gingembre", "2"),
)


def _speak(text: str, work: Path) -> array:
    output = work / "utterance.wav"
    output.unlink(missing_ok=True)
    subprocess.run(["/usr/bin/say", "-v", "Jacques", "-r", str(VOICE_RATE), "-o", str(output),
                    "--file-format=WAVE", "--data-format=LEI16@16000", text],
                   check=True, capture_output=True, timeout=30)
    speech = array("h")
    with wave.open(str(output)) as audio:
        assert audio.getframerate() == RATE and audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        speech.frombytes(audio.readframes(audio.getnframes()))
    if sys.byteorder != "little":
        speech.byteswap()
    # Keep short within-utterance pauses, limit longer quiet stretches, and
    # concatenate each phrase without an artificial endpointing gap.
    frame = RATE // 100
    active = [i // frame for i in range(0, len(speech), frame)
              if max(map(abs, speech[i:i + frame])) >= 131]
    if not active:
        raise RuntimeError(f"Speech synthesis returned silence for {text!r}")
    joined = array("h")
    for i, j in zip(active[:-1], active[1:], strict=True):
        joined.extend(speech[i * frame:min(i + 4, j) * frame])
    joined.extend(speech[active[-1] * frame:(active[-1] + 2) * frame])
    return joined


def generate(audio: Path) -> tuple[Path, list[dict]]:
    audio.parent.mkdir(parents=True, exist_ok=True)
    pieces, anchors, transcript, gold = [], [], [], []
    cursor = 0
    with TemporaryDirectory(prefix="speechflow-qa-say-") as directory:
        work = Path(directory)
        filler = _speak(FILLER, work)
        for i, (phrase, product, quantity) in enumerate(INVENTORY):
            utterance = f"Je compte {phrase}."
            samples = _speak(utterance, work)
            anchors.append({"phrase": phrase, "product": product, "quantity": quantity,
                            "sample_start": cursor, "sample_end": cursor + len(samples)})
            pieces.append(samples)
            transcript.append(utterance)
            gold.append({"product_spoken": product, "quantity": quantity, "unit": "kg"})
            cursor += len(samples)
            if i < len(INVENTORY) - 1:
                pieces.append(filler)
                transcript.append(FILLER + ".")
                cursor += len(filler)
    with wave.open(str(audio), "wb") as output:
        output.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        for samples in pieces:
            if sys.byteorder != "little":
                samples = array("h", samples)
                samples.byteswap()
            output.writeframesraw(samples.tobytes())
    row = {"id": "long-jacques-inventory", "audio": audio.name, "synthetic": True,
           "condition": "synthetic-joined-speech", "language": "fr",
           "source": f"macOS say, Jacques, {VOICE_RATE} words/min, trimmed quiet intervals",
           "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(), "duration_s": cursor / RATE,
           "reference": " ".join(transcript), "expected_observations": gold}
    manifest = audio.with_suffix(".manifest.jsonl")
    manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    audio.with_suffix(".anchors.json").write_text(json.dumps(anchors, indent=2) + "\n")
    return manifest, anchors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    args = parser.parse_args()
    print(generate(args.audio)[0])
