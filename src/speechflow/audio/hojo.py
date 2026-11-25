"""Local Hojo ASR with a conservative input limit and explicit device handling."""

from dataclasses import dataclass
from functools import wraps
import logging
from pathlib import Path

MODEL_ID = "HojoAI/Hojo-ASR-Multi-V1"
MODEL_REVISION = "aa1d0dfc369deb3b3235572fa151acd7b498481f"
PACKAGE_VERSION = "0.1.3"
SAMPLE_RATE = 16_000
BUNDLE_LIMIT_SECONDS = 30
DEFAULT_CHUNK_SECONDS = 25.0
PINNED_GENERATION_CAP = 200
CAP_TOKEN_TOLERANCE = 4
MIN_LOOP_BLOCK_TOKENS = 16
MIN_LOOP_COPIES = 3
log = logging.getLogger(__name__)


class HojoResultError(RuntimeError):
    """The checkpoint did not return the expected record for an input chunk."""


@dataclass(frozen=True)
class GenerationObservation:
    feature_frames: int | None
    max_new_tokens: int | None
    output_tokens: int | None
    terminal_eos: bool | None


def _observe_wav_generation(model, file: Path):
    """Pass through the pinned decoder call and observe its unmodified result."""
    decoder = model.decoder_model
    generate = decoder.generate
    previous_override = vars(decoder).get("generate")
    calls = []
    model.last_generation = None
    model._speech_feature_frames = None

    @wraps(generate)
    def observe(*args, **kwargs):
        output = generate(*args, **kwargs)
        embeds_only = not args and "inputs_embeds" in kwargs and "input_ids" not in kwargs
        calls.append((kwargs.get("max_new_tokens"), kwargs.get("eos_token_id"),
                      output if embeds_only else None))
        return output

    decoder.generate = observe
    try:
        records = model.run_infer([str(file)], batch_size=1, cuda_enabled=model.device.type == "mps")
    finally:
        if previous_override is None:
            del decoder.generate
        else:
            decoder.generate = previous_override

    try:
        model.last_generation = _generation_observation(model, calls)
    except Exception:
        log.warning("Could not observe Hojo decoder output", exc_info=True)
    return records


def _generation_observation(model, calls):
    feature_frames = model._speech_feature_frames
    dynamic_cap = None
    if feature_frames is not None:
        # hojo-asr 0.1.3, HOJO_ASR.infer uses the length of these same embeddings.
        configured = model.config.generate.get("max_new_tokens", PINNED_GENERATION_CAP)
        dynamic_cap = max(10, min(configured, 2 * feature_frames + 10))
    count = terminal_eos = None
    if len(calls) == 1:
        cap, eos_id, ids = calls[0]
        if type(cap) is int and cap > 0:
            dynamic_cap = cap
        # Transformers 4.57.3 starts the pinned decoder's inputs_embeds-only
        # path with zero token IDs and returns the single beam's cropped sequence.
        shape = getattr(ids, "shape", ())
        if len(shape) == 2 and shape[0] == 1 and 0 < shape[1] <= (dynamic_cap or 0):
            count = shape[1]
            if isinstance(eos_id, (int, list, tuple)):
                eos_ids = (eos_id,) if isinstance(eos_id, int) else eos_id
                try:
                    terminal_eos = ids[0, -1].item() in eos_ids
                except (AttributeError, RuntimeError, TypeError):
                    pass
    return GenerationObservation(feature_frames, dynamic_cap, count, terminal_eos)


def _generation_cap(model, observation):
    if observation is not None and observation.max_new_tokens is not None:
        return observation.max_new_tokens
    generation = getattr(getattr(model, "config", None), "generate", {})
    return PINNED_GENERATION_CAP if generation.get("max_new_tokens") == PINNED_GENERATION_CAP else None


def generation_cap_warning(model, text: str) -> str | None:
    """Identify a filled token budget, with a text-length proxy when IDs are unavailable."""
    observation = getattr(model, "last_generation", None)
    cap = _generation_cap(model, observation)
    if cap is None:
        return None
    if observation is not None and observation.output_tokens is not None:
        if observation.output_tokens == cap:
            eos = {True: "present", False: "absent", None: "unknown"}[observation.terminal_eos]
            return (f"hojo_generation_cap_reached: decoder returned {cap}/{cap} output tokens "
                    f"(terminal EOS {eos})")
        return None
    count = len(model.tokenizer(text, add_special_tokens=False)["input_ids"])
    if abs(count - cap) <= CAP_TOKEN_TOLERANCE:
        return (f"hojo_possible_generation_cap: {count} retokenized tokens near the "
                f"{cap}-token generation limit")
    return None


def generation_loop_warning(model, text: str) -> str | None:
    """Recognize long exact token cycles near the observed output limit."""
    observation = getattr(model, "last_generation", None)
    cap = _generation_cap(model, observation)
    if cap is None or (observation is not None and observation.output_tokens not in (None, cap)):
        return None
    tokens = model.tokenizer(text, add_special_tokens=False)["input_ids"]
    count = len(tokens)
    if abs(count - cap) > CAP_TOKEN_TOLERANCE:
        return None

    for width in range(MIN_LOOP_BLOCK_TOKENS, count // MIN_LOOP_COPIES + 1):
        for start in range(count - MIN_LOOP_COPIES * width + 1):
            block = tokens[start:start + width]
            if tokens[start:start + MIN_LOOP_COPIES * width] != block * MIN_LOOP_COPIES:
                continue
            # A short periodic phrase can also form larger repeated blocks.
            if any(width % period == 0 and block == block[:period] * (width // period)
                   for period in range(1, MIN_LOOP_BLOCK_TOKENS)):
                continue
            return (
                f"hojo_possible_generation_loop: {count} retokenized tokens near the "
                f"{cap}-token cap and at least {MIN_LOOP_COPIES} consecutive "
                f"copies of a {width}-token block"
            )
    return None


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
            self._speech_feature_frames = embeddings.shape[1]
            embedding_dtype = self.decoder_model.get_input_embeddings().weight.dtype
            return embeddings.to(dtype=embedding_dtype), attention

        def _run_observed_wav(self, file):
            return _observe_wav_generation(self, file)

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
    observe = getattr(model, "_run_observed_wav", None)
    results = (
        observe(file) if observe is not None
        else model.run_infer([str(file)], batch_size=1, cuda_enabled=model.device.type == "mps")
    )
    if len(results) != 1:
        raise HojoResultError(f"Hojo returned {len(results)} records for chunk {file.stem!r}")
    record = results[0]
    if not isinstance(record, dict) or record.get("key") != file.stem:
        raise HojoResultError(f"Hojo returned an invalid record for chunk {file.stem!r}: {record!r}")
    if not isinstance(record.get("text"), str):
        raise HojoResultError(f"Hojo returned a non-text result for chunk {file.stem!r}: {record!r}")
    return record
