"""Local storage formats observed in the pinned Ollama integration."""

import hashlib
import json

import pytest

from speechflow import assets


@pytest.mark.parametrize("wrapped", [False, True])
def test_verify_publisher_manifest_and_local_layers(tmp_path, monkeypatch, wrapped):
    blobs = tmp_path / "blobs"
    blobs.mkdir()

    def blob(raw, media_type):
        digest = hashlib.sha256(raw).hexdigest()
        (blobs / f"sha256-{digest}").write_bytes(raw)
        return {"mediaType": media_type, "digest": f"sha256:{digest}", "size": len(raw)}

    config = blob(b'{"architecture":"example"}', "config")
    weight = blob(b"test weights", "model")
    registry = {"schemaVersion": 2, "mediaType": "manifest", "config": config, "layers": [weight]}
    raw = json.dumps(registry).encode()
    reference = blob(raw, "manifest")
    monkeypatch.setattr(assets, "DIGEST", reference["digest"].removeprefix("sha256:"))
    tag = tmp_path / "manifests/registry.ollama.ai/library/qwen3/8b"
    tag.parent.mkdir(parents=True)
    local = {**registry, "layers": [weight, reference]} if wrapped else registry
    tag.write_text(json.dumps(local))
    result = assets.verify_ollama_files(tmp_path)
    assert result["manifest_sha256"] == assets.DIGEST
    assert result["local_descriptor_sha256"] == hashlib.sha256(tag.read_bytes()).hexdigest()

    tag.write_text(json.dumps({**local, "layers": [config, *local["layers"]]}))
    with pytest.raises(ValueError, match="descriptor"):
        assets.verify_ollama_files(tmp_path)
    tag.write_text(json.dumps(local))
    (blobs / weight["digest"].replace(":", "-")).write_bytes(b"bad weights!")
    with pytest.raises(ValueError, match="blob"):
        assets.verify_ollama_files(tmp_path)
