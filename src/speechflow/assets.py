"""Provisioning is explicit. The inference path uses verified local artifacts."""

import hashlib
import importlib.metadata as md
import json
import os
import platform
from pathlib import Path
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .config import Settings

MANIFEST = json.loads(Path(__file__).with_name("model_manifest.json").read_text())
HOJO = MANIFEST["hojo"]
DIGEST = MANIFEST["ollama"]["manifest_sha256"]


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Local Ollama returned a redirect")


def local_request(settings, route, body=None, timeout=5):
    return loopback_request(settings.ollama_url, route, body, timeout)


def loopback_request(url, route, body=None, timeout=5):
    # Both adapters share redirect and proxy policy, including when instantiated directly.
    url = Settings.loopback(url)
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    request = Request(url + route,
                      data=json.dumps(body).encode() if body is not None else None,
                      headers={"Content-Type": "application/json"})
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def file_digest(path, algorithm="sha256"):
    with path.open("rb") as stream:
        if algorithm == "sha256":
            return hashlib.file_digest(stream, "sha256").hexdigest()
        digest = hashlib.sha1(f"blob {os.fstat(stream.fileno()).st_size}\0".encode())
        while block := stream.read(1024 * 1024):
            digest.update(block)
        return digest.hexdigest()


def hojo_issues(settings, *, verify=False):
    issues = []
    for name, (size, algorithm, digest) in HOJO["files"].items():
        path = settings.hojo_dir / name
        if not path.is_file() or path.stat().st_size != size:
            issues.append(f"Missing or incomplete {name} (expected {size} bytes)")
        elif verify and file_digest(path, algorithm) != digest:
            issues.append(f"Digest mismatch for {name}")
    return issues


def silero_hash():
    dist = md.distribution("silero-vad")
    path = Path(dist.locate_file(MANIFEST["vad"]["path"]))
    if dist.version != "6.2.0" or not path.is_file():
        raise RuntimeError("Restore the pinned Silero VAD package with uv sync --frozen")
    return file_digest(path)


def extractor_digest(settings):
    tags = local_request(settings, "/api/tags")["models"]
    return next((r["digest"].removeprefix("sha256:") for r in tags if r["name"] == settings.ollama_model), None)


def doctor(settings: Settings) -> dict:
    data = {"machine": platform.machine(), "macos": platform.mac_ver()[0],
            "offline": settings.offline, "hojo_revision": HOJO["revision"],
            "hojo_dir": str(settings.hojo_dir), "hojo_issues": hojo_issues(settings)}
    try:
        data["silero_sha256"] = silero_hash()
    except (OSError, RuntimeError, md.PackageNotFoundError) as exc:
        data["silero_error"] = str(exc)
    try:
        data["ollama_version"] = local_request(settings, "/api/version")["version"]
        data["extractor_digest"] = extractor_digest(settings)
        if data["extractor_digest"] != DIGEST:
            data["extractor_error"] = "Pinned qwen3:8b digest is unavailable"
    except (ValueError, OSError, KeyError) as exc:
        data["extractor_error"] = str(exc)
    return data


def prepare(settings: Settings) -> dict:
    issues = hojo_issues(settings, verify=True)
    if issues and not settings.offline:
        from huggingface_hub import snapshot_download

        snapshot_download(HOJO["repository"], revision=HOJO["revision"], token=False,
                          local_dir=settings.hojo_dir, allow_patterns=list(HOJO["files"]))
        issues = hojo_issues(settings, verify=True)
    digest = None
    try:
        digest = extractor_digest(settings)
        if digest != DIGEST and not settings.offline:
            result = local_request(settings, "/api/pull", {"model": settings.ollama_model, "stream": False},
                                   timeout=24 * 3600)
            if result.get("status") != "success":
                issues.append("Ollama pull failed: " + str(result))
            digest = extractor_digest(settings)
    except (OSError, ValueError, KeyError) as exc:
        issues.append(f"Ollama at {settings.ollama_url}: {exc}")
    if digest != DIGEST:
        issues.append(f"Missing pinned {settings.ollama_model} (expected {DIGEST}, found {digest})")
    try:
        vad = silero_hash()
    except (OSError, RuntimeError, md.PackageNotFoundError) as exc:
        issues.append(str(exc))
    if issues:
        raise RuntimeError("\n".join(issues))
    return {"hojo_revision": HOJO["revision"], "hojo_files_verified": len(HOJO["files"]),
            "extractor_digest": digest, "silero_sha256": vad}
