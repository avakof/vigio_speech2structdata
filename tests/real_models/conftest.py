"""Select this directory or its marker explicitly to run model acceptance tests."""

import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent


def require_package(package, expected):
    try:
        installed = version(package)
    except PackageNotFoundError:
        pytest.fail(f"BLOCKED: install {package}=={expected} with uv sync --frozen")
    assert installed == expected, f"BLOCKED: {package}=={expected} required, found {installed}"


def pytest_collection_modifyitems(config, items):
    explicit = any(ROOT == Path(arg.split("::")[0]).resolve() or
                   ROOT in Path(arg.split("::")[0]).resolve().parents
                   for arg in config.args)
    expression = config.option.markexpr
    microphone = "microphone" in expression or any(
        "::test_real_microphone_offline_after_file_gate" in arg for arg in config.args
    )
    deselected = [item for item in items if ROOT in Path(item.path).parents and (
        (not explicit and "real_models" not in expression and not microphone)
        or (item.get_closest_marker("microphone") and not microphone))]
    config.hook.pytest_deselected(items=deselected)
    items[:] = [item for item in items if item not in deselected]


@pytest.fixture(scope="session")
def model_slot():
    if os.environ.get("SPEECHFLOW_QA_MODEL_SLOT") != "1":
        pytest.fail("BLOCKED: obtain the local heavy-model slot and set SPEECHFLOW_QA_MODEL_SLOT=1")


@pytest.fixture(scope="session")
def model_settings():
    from speechflow.config import load_settings
    return load_settings(offline=True)


@pytest.fixture(scope="session")
def codecs():
    require_package("imageio-ffmpeg", "0.6.0")


@pytest.fixture(scope="session")
def silero(codecs):
    require_package("silero-vad", "6.2.0")
    require_package("torch", "2.5.1")
    require_package("torchaudio", "2.5.1")


@pytest.fixture(scope="session")
def pinned_hojo(model_slot, model_settings):
    from speechflow.assets import hojo_issues
    issues = hojo_issues(model_settings)
    if issues:
        pytest.fail(f"BLOCKED: {'; '.join(issues)}")
    require_package("hojo-asr", "0.1.3")
    return model_settings


@pytest.fixture(scope="session")
def pinned_ollama(model_slot, model_settings):
    from speechflow.assets import DIGEST, extractor_digest
    try:
        digest = extractor_digest(model_settings)
    except (OSError, ValueError, KeyError) as exc:
        pytest.fail(f"BLOCKED: Ollama unavailable: {exc}")
    assert digest == DIGEST, f"BLOCKED: pinned Ollama artifact absent (found {digest})"
    return model_settings
