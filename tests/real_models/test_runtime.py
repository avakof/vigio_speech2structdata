"""Run local pins and fresh outputs; synthetic speech is functional evidence only."""

import errno
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unicodedata
import wave

import pytest

from speechflow.config import Settings

pytestmark = pytest.mark.real_models
ROOT = Path(__file__).resolve().parents[2]
JACQUES = Path.home() / ".cache/speechflow/hojo-jacques.wav"


def checked(command, *, timeout=30):
    process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    assert process.returncode == 0, (
        f"command {command!r} exited {process.returncode}\n"
        f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
    )
    return process.stdout


@pytest.fixture(scope="module")
def long_clip(tmp_path_factory):
    directory = tmp_path_factory.mktemp("synthetic-jacques")
    audio = directory / "long.wav"
    checked([sys.executable, str(ROOT / "tools/qa/make_long_french.py"), str(audio)])
    anchors = json.loads(audio.with_suffix(".anchors.json").read_text())
    with wave.open(str(audio)) as handle:
        assert handle.getnframes() > 60 * 16000
    assert anchors[-1]["sample_start"] > 60 * 16000
    return audio, anchors


@pytest.mark.parametrize("suffix", ["wav", "flac", "mp3", "m4a"])
def test_codec_decodes_right_channel_and_preserves_source(tmp_path, suffix, codecs):
    import math
    import struct

    from speechflow.audio.decoding import decode_file, ffmpeg_executable

    source_wav = tmp_path / "right-channel.wav"
    sample_rate = 44100
    with wave.open(str(source_wav), "wb") as handle:
        handle.setparams((2, 2, sample_rate, 0, "NONE", "not compressed"))
        handle.writeframes(b"".join(struct.pack("<hh", 0, int(12000 * math.sin(i / 14)))
                                   for i in range(sample_rate // 2)))
    encoded = source_wav
    if suffix != "wav":
        encoded = tmp_path / f"right-channel.{suffix}"
        checked([ffmpeg_executable(), "-v", "error", "-nostdin", "-i", str(source_wav), str(encoded)])
    original = encoded.read_bytes()
    spool = tmp_path / "spool"
    source = decode_file(encoded, spool, 160000)
    assert source.native_channels == 2 and source.native_rate == sample_rate
    assert source.channel_mix != "mono"
    assert source.sample_count == pytest.approx(8000, abs=800)
    assert max(abs(frame[0]) for frame in struct.iter_unpack("<f", spool.read_bytes())) > 0.1
    assert encoded.read_bytes() == original
    assert source.digest == hashlib.sha256(original).hexdigest()


def test_codec_corrupt_wav_and_spool_limit(tmp_path, codecs):
    from speechflow.audio.decoding import AudioDecodeError, decode_file

    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"RIFF\x00\x01WAVEfmtdisturbed")
    with pytest.raises(AudioDecodeError, match="FFmpeg could not decode"):
        decode_file(corrupt, tmp_path / "corrupt.f32", 100000)
    wav = tmp_path / "above-budget.wav"
    with wave.open(str(wav), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(b"\x00\x00" * 400)
    spool = tmp_path / "bounded.f32"
    with pytest.raises(AudioDecodeError, match="max_spool_bytes"):
        decode_file(wav, spool, 4)
    assert spool.stat().st_size <= 4


def test_real_silero_silence_and_speech_active_at_eof(tmp_path, silero, long_clip):
    from speechflow.assets import silero_hash
    from speechflow.audio import transcribe_file
    from speechflow.audio.decoding import decode_file
    from speechflow.audio.segmentation import segment_pcm, silero_provenance

    silence = tmp_path / "silence.wav"
    with wave.open(str(silence), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(b"\x00\x00" * 16039)
    result = transcribe_file(silence, Settings(offline=True, hojo_dir=tmp_path / "missing"))
    assert result.status == "no_speech", result.errors
    assert result.segments == [] and result.backend == "not_loaded"
    assert result.metrics["silero"]["sha256"] == silero_hash()

    # End in the middle of the final utterance to remove trailing silence,
    # including a deliberately partial final 512-sample VAD frame.
    wav, anchors = long_clip
    last = anchors[-1]
    tail_frames = (last["sample_end"] - last["sample_start"]) - 1600 + 37
    shortened = tmp_path / "active-eof.wav"
    with wave.open(str(wav)) as original, wave.open(str(shortened), "wb") as output:
        original.setpos(last["sample_start"])
        output.setparams(original.getparams())
        output.writeframes(original.readframes(tail_frames))
    pcm = tmp_path / "eof.f32"
    source = decode_file(shortened, pcm, 1000000)
    bounded = segment_pcm(pcm, source, 25, 200)
    assert silero_provenance()["version"] == "6.2.0"
    assert source.sample_count == tail_frames and source.sample_count % 512
    assert bounded and bounded[-1].core_end == bounded[-1].input_end == tail_frames


def test_real_silero_long_fixture_exercises_forced_chunk_boundaries(tmp_path, silero, long_clip):
    from speechflow.audio.decoding import decode_file
    from speechflow.audio.segmentation import segment_pcm

    wav, _ = long_clip
    pcm = tmp_path / "long.f32"
    source = decode_file(wav, pcm, 10_000_000)
    limits = segment_pcm(pcm, source, 25, 200)
    assert len(limits) >= 3
    assert sum(a.core_end == b.core_start and a.input_end > b.input_start
               for a, b in zip(limits, limits[1:])) >= 2
    assert all(s.input_end - s.input_start <= 400_000 for s in limits)
    assert source.sample_count - limits[-1].core_end < 1600


@pytest.fixture(scope="module")
def real_long_hojo(pinned_hojo, silero, long_clip):
    from hojo_asr import HOJO_ASR
    import soundfile
    import speechflow.audio as audio_module

    wav, anchors = long_clip
    seen = []
    original = audio_module.infer_wav

    def inspect_actual_input(model, chunk, *, max_seconds):
        info = soundfile.info(chunk)
        assert isinstance(model, HOJO_ASR)
        assert info.samplerate == 16000 and info.channels == 1 and info.format == "WAV"
        assert 0 < info.frames <= max_seconds * info.samplerate <= 400000
        assert info.frames <= model.feat_extractor.n_samples
        result = original(model, chunk, max_seconds=max_seconds)
        seen.append((id(model), chunk.stem, info.frames, result))
        return result

    # Wrap the production call with assertions and invoke the underlying model
    # unchanged. Patch context ends before any other real-model test runs.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(audio_module, "infer_wav", inspect_actual_input)
        result = audio_module.transcribe_file(wav, pinned_hojo)
    assert result.status in {"ok", "needs_review"}, result.errors
    assert result.backend == pinned_hojo.backend and result.dtype != "not_loaded"
    assert result.metrics.get("model_load_seconds", 0) > 0
    assert result.metrics.get("asr_seconds", 0) > 0
    assert len(seen) == len(result.segments) >= 3
    assert len({key for key, _, _, _ in seen}) == 1
    for segment, (_, chunk_id, frames, raw) in zip(result.segments, seen, strict=True):
        assert chunk_id == raw["key"] == segment.segment_id
        assert frames == segment.input_end - segment.input_start
        assert raw == segment.raw and raw["text"].strip()
    return result, anchors


def _has_anchor(text, anchor):
    text = unicodedata.normalize("NFC", text).casefold()
    quantity_word = anchor["phrase"].split()[0]
    return bool(re.search(fr"\b(?:{quantity_word}|{anchor['quantity']})\s+(?:kilos?|kg)"
                          fr"\s+d(?:e\s+|['’]){anchor['product']}\b", text))


def test_real_hojo_long_audio_boundary_content_and_tail(real_long_hojo):
    result, anchors = real_long_hojo
    segments = result.segments
    forced = [s.core_end for s, following in zip(segments, segments[1:])
              if s.core_end == following.core_start and s.input_end > following.input_start]
    assert len(forced) >= 2, "the fixture did not exercise at least two forced padded boundaries"
    for edge in forced:
        before = [a for a in anchors if a["sample_end"] <= edge]
        after = [a for a in anchors if a["sample_start"] >= edge]
        assert before and after, f"forced boundary {edge} lacks independently timed anchors"
        for anchor in (before[-1], after[0]):
            assert min(abs(edge - anchor["sample_start"]), abs(edge - anchor["sample_end"])) < 12 * 16000
    assert any(a["sample_start"] > 30 * 16000 for a in anchors)
    for anchor in anchors:
        intersecting = [s for s in segments if s.core_start < anchor["sample_end"]
                        and anchor["sample_start"] < s.core_end]
        assert _has_anchor(" ".join(s.normalized_text for s in intersecting), anchor), anchor["phrase"]
    assert _has_anchor(segments[-1].normalized_text, anchors[-1])
    assert result.source.sample_count - segments[-1].core_end < 1600


def test_real_extractor_long_hojo_outputs_have_no_duplicate_observations(pinned_ollama, real_long_hojo):
    from speechflow.extraction import extract

    audio, anchors = real_long_hojo
    result = extract(audio.segments, settings=pinned_ollama, run_id="qa-real-long-clip")
    assert result.status == "ok", result.errors
    assert_real_ollama_attempts(result.attempts)
    actual = sorted((obs.product_spoken, obs.quantity, obs.unit) for obs in result.observations)
    assert actual == sorted((a["product"], a["quantity"], "kg") for a in anchors)


def assert_real_ollama_attempts(attempts):
    from speechflow.assets import DIGEST

    assert attempts
    for attempt in attempts:
        assert attempt["extractor"]["digest"].removeprefix("sha256:") == DIGEST
        raw = attempt["response_raw"]
        assert raw["eval_count"] > 0 and raw["total_duration"] > 0
        assert raw["model"] == "qwen3:8b" and raw["done"] is True


def test_real_extractor_installed_cli_exact_transcript(tmp_path, pinned_ollama):
    cli = Path(sys.executable).parent / "speechflow"
    assert cli.is_file(), "BLOCKED: missing installed speechflow CLI"
    report = json.loads(checked([str(cli), "evaluate", "--offline", str(
        ROOT / "evaluation/fixtures/reference-fr.jsonl"), "--out", str(tmp_path)], timeout=2100))
    assert report["extraction"]["reference"]["measurement_status"] == "MEASURED"
    tests = {case["id"]: case["reference_extraction"] for case in report["cases"]}
    for item in tests.values():
        assert_real_ollama_attempts(item["attempts"])
        assert item["review_decision_correct"] is True
        assert all(field["fp"] == field["fn"] == 0 for field in item["scores"]["fields"].values())


def test_real_file_installed_cli_offline_both_processes(tmp_path, pinned_hojo, pinned_ollama):
    assert JACQUES.is_file(), "BLOCKED: missing Jacques functional fixture"
    root = tmp_path / "offline"
    receipt = json.loads(checked([sys.executable, str(ROOT / "tools/offline/verify.py"), "file",
                                  "--audio", str(JACQUES), "--out", str(root),
                                  "--backend", pinned_hojo.backend], timeout=2100))
    raw = json.loads(Path(receipt["report"]).read_text())
    assert raw["result"] == "PASS" and raw["mode"] == "file"
    assert raw["unsandboxed_tcp_443"] is True
    for actor in ("server_control", "app_control"):
        assert raw[actor]["denied"]["errno"] in (errno.EPERM, errno.EACCES)
        assert raw[actor]["allowed"]["status"] == 200
    manifest = json.loads((Path(receipt["report"]).parent / "output/run.json").read_text())
    from speechflow.assets import DIGEST, HOJO, silero_hash
    assert manifest["source"]["digest"] == hashlib.sha256(JACQUES.read_bytes()).hexdigest()
    assert manifest["provenance"]["hojo_revision"] == HOJO["revision"]
    assert manifest["provenance"]["silero_sha256"] == silero_hash()
    assert manifest["provenance"]["extractor_digest"] == DIGEST
    assert manifest["metrics"]["audio"]["asr_seconds"] > 0
    assert_real_ollama_attempts(manifest["attempts"])
    assert sorted((o["product_spoken"], o["quantity"], o["unit"]) for o in manifest["observations"]
                  if o["status"] == "ok") == [("lait", "12", "bouteille"), ("tomates", "2.5", "kg")]


@pytest.mark.microphone
def test_real_microphone_offline_after_file_gate(tmp_path, pinned_hojo, pinned_ollama):
    receipt = os.environ.get("SPEECHFLOW_QA_FILE_EVIDENCE")
    if not receipt:
        pytest.fail("BLOCKED: supply passing real offline file evidence before microphone capture")
    proof = json.loads(Path(receipt).read_text())
    assert proof["result"] == "PASS" and proof["mode"] == "file"
    for actor in ("server_control", "app_control"):
        assert proof[actor]["denied"]["errno"] in (errno.EPERM, errno.EACCES)
        assert proof[actor]["allowed"]["status"] == 200
    root = tmp_path / "offline-mic"
    result = json.loads(checked([sys.executable, str(ROOT / "tools/offline/verify.py"), "mic",
                                 "--out", str(root), "--backend", pinned_hojo.backend,
                                 "--seconds", "12"], timeout=2100))
    report = json.loads(Path(result["report"]).read_text())
    assert report["result"] == "PASS" and report["mode"] == "mic"
    for actor in ("server_control", "app_control"):
        assert report[actor]["denied"]["errno"] in (errno.EPERM, errno.EACCES)
        assert report[actor]["allowed"]["status"] == 200
    manifest = json.loads((Path(result["report"]).parent / "output/run.json").read_text())
    assert manifest["source"]["kind"] == "mic"
    assert manifest["source"]["sample_count"] > 0
    assert_real_ollama_attempts(manifest["attempts"])
