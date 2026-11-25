"""Local Hojo ASR with a conservative input limit and explicit device handling."""

from pathlib import Path

MODEL_ID = "HojoAI/Hojo-ASR-Multi-V1"
MODEL_REVISION = "aa1d0dfc369deb3b3235572fa151acd7b498481f"
PACKAGE_VERSION = "0.1.3"
SAMPLE_RATE = 16_000
BUNDLE_LIMIT_SECONDS = 30
DEFAULT_CHUNK_SECONDS = 25.0


class HojoResultError(RuntimeError):
    """The checkpoint did not return the expected record for an input chunk."""


def verify_local_bundle(settings) -> None:
    from ..assets import hojo_issues

    issues = hojo_issues(settings, verify=True)
    if issues:
        details = "\n".join(issues)
        raise RuntimeError(
            f"Cannot load pinned Hojo assets from {settings.hojo_dir}. "
            f"Prepare the local models first.\n{details}"
        )


def validate_chunk_limit(max_seconds: float) -> None:
    if not 0 < max_seconds <= BUNDLE_LIMIT_SECONDS:
        raise ValueError(f"Hojo chunk limit must be within (0, {BUNDLE_LIMIT_SECONDS}] seconds")


def validate_frame_count(frames: int, max_seconds: float) -> None:
    validate_chunk_limit(max_seconds)
    if not 0 < frames <= int(SAMPLE_RATE * max_seconds):
        raise ValueError(
            f"Hojo input has {frames} samples. Supply 1 to "
            f"{int(SAMPLE_RATE * max_seconds)} samples, including padding and overlap."
        )


def load_local_model(bundle: Path, backend: str):
    """Load one instance, preserving the checkpoint's encoder and decoder dtypes."""
    from contextlib import nullcontext
    from importlib.metadata import version

    import torch
    from hojo_asr import HOJO_ASR

    if version("hojo-asr") != PACKAGE_VERSION:
        raise RuntimeError(f"The Hojo adapter requires hojo-asr=={PACKAGE_VERSION}")
    if backend not in {"cpu", "mps"}:
        raise ValueError(f"Unsupported Hojo backend {backend!r}. Choose cpu or mps.")
    if backend == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch reports that MPS is unavailable")
    if not bundle.is_dir():
        raise FileNotFoundError(f"Local Hojo bundle not found at {bundle}")
    overrides = ("autocast_context", "encode_speech")
    for name in overrides:
        if not callable(getattr(HOJO_ASR, name, None)):
            raise RuntimeError(f"The installed Hojo wheel has no callable {name} hook")

    class LocalHojo(HOJO_ASR):
        autocast_calls = 0
        encode_calls = 0

        def autocast_context(self, dtype=torch.float16):
            # The wheel selects CUDA autocast on MPS. The audio path uses the
            # checkpoint's float32 weights with float32 Whisper features.
            self.autocast_calls += 1
            return nullcontext()

        def encode_speech(self, spectrogram, spectrogram_lengths):
            self.encode_calls += 1
            embeddings, attention = super().encode_speech(spectrogram, spectrogram_lengths)
            embedding_dtype = self.decoder_model.get_input_embeddings().weight.dtype
            return embeddings.to(dtype=embedding_dtype), attention

    model = LocalHojo.load_model(str(bundle.resolve()), device=backend, local_files_only=True)
    if not isinstance(model, LocalHojo):
        raise RuntimeError("The Hojo wheel did not construct the local compatibility subclass")
    for name in overrides:
        bound = getattr(model, name, None)
        if getattr(bound, "__func__", None) is not getattr(LocalHojo, name):
            raise RuntimeError(f"The loaded Hojo instance did not bind the local {name} hook")
    if model.device.type != backend:
        raise RuntimeError(f"Hojo requested {backend} but loaded on {model.device}")
    if backend == "mps":
        # Torch 2.5.1 aborts on Hojo's grouped-query SDPA with 32 query and
        # eight KV heads. Qwen3's eager path explicitly repeats KV on device.
        decoder = model.decoder_model
        decoder.set_attn_implementation("eager")
        implementations = {
            layer.self_attn.config._attn_implementation for layer in decoder.model.layers
        }
        if implementations != {"eager"}:
            raise RuntimeError(f"Hojo decoder did not switch all MPS layers to eager: {implementations}")
    if model.feat_extractor.sampling_rate != SAMPLE_RATE:
        raise RuntimeError("Hojo's runtime sample rate disagrees with the application")
    if model.feat_extractor.n_samples < BUNDLE_LIMIT_SECONDS * SAMPLE_RATE:
        raise RuntimeError("Hojo's runtime feature limit is shorter than the verified bundle limit")
    return model


def infer_wav(model, file: Path, *, max_seconds: float = DEFAULT_CHUNK_SECONDS) -> dict[str, str]:
    """Submit a single normalized WAV and verify the actual Hojo response key."""
    import soundfile

    info = soundfile.info(file)
    if info.format != "WAV" or info.channels != 1 or info.samplerate != SAMPLE_RATE:
        raise ValueError("Hojo requires a mono 16 kHz WAV")
    validate_frame_count(info.frames, max_seconds)
    results = model.run_infer([str(file)], batch_size=1, cuda_enabled=model.device.type == "mps")
    if len(results) != 1:
        raise HojoResultError(f"Hojo returned {len(results)} records for chunk {file.stem!r}")
    record = results[0]
    if not isinstance(record, dict) or record.get("key") != file.stem:
        raise HojoResultError(f"Hojo returned an invalid record for chunk {file.stem!r}: {record!r}")
    if not isinstance(record.get("text"), str):
        raise HojoResultError(f"Hojo returned a non-text result for chunk {file.stem!r}: {record!r}")
    return record
