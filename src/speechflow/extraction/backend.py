"""Replaceable chat backend. The default talks to a loopback Ollama via the stdlib only."""

from __future__ import annotations

import json
import time
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..assets import loopback_request
from ..config import Settings

KEEP_ALIVE = "5m"


class BackendError(RuntimeError):
    """Transport/service failure (unreachable, HTTP error, timeout, bad body)."""

    def __init__(self, kind: str, message: str):
        super().__init__(f"{kind}: {message}")
        self.kind = kind


@dataclass
class BackendReply:
    content: str
    done_reason: str | None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    timings_ns: dict[str, int | None] = field(default_factory=dict)
    wall_s: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)  # unchanged service response


class ChatBackend(Protocol):
    name: str

    def chat(self, messages: list[dict], schema: dict, *, options: dict, timeout: float) -> BackendReply: ...

    def identify(self) -> dict[str, Any]: ...


class OllamaBackend:
    name = "ollama"

    def __init__(self, url: str, model: str, keep_alive: str | int = KEEP_ALIVE):
        try:
            self.url = Settings.loopback(url)  # numeric loopback only, also when constructed directly
        except ValueError as exc:
            raise BackendError("not_loopback", str(exc)) from exc
        self.model, self.keep_alive = model, keep_alive

    def _call(self, path: str, body: dict | None, timeout: float) -> dict:
        try:
            return loopback_request(self.url, path, body, timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise BackendError("http_error", f"{exc.code} {detail}") from exc
        except TimeoutError as exc:
            raise BackendError("timeout", f"no reply within {timeout}s") from exc
        except urllib.error.URLError as exc:
            kind = "timeout" if isinstance(exc.reason, TimeoutError) else "unreachable"
            raise BackendError(kind, str(exc.reason)) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise BackendError("bad_response", str(exc)) from exc
        except ValueError as exc:  # redirect refused / non-loopback URL (shared local_request policy)
            raise BackendError("refused", str(exc)) from exc

    def chat(self, messages, schema, *, options, timeout):
        body = {"model": self.model, "messages": messages, "stream": False, "think": False,
                "format": schema, "options": options, "keep_alive": self.keep_alive}
        t0 = time.perf_counter()
        resp = self._call("/api/chat", body, timeout)
        wall = time.perf_counter() - t0
        if "error" in resp:
            raise BackendError("service_error", str(resp["error"])[:500])
        return BackendReply(
            content=(resp.get("message") or {}).get("content", ""),
            done_reason=resp.get("done_reason"),
            prompt_tokens=resp.get("prompt_eval_count"), output_tokens=resp.get("eval_count"),
            timings_ns={k: resp.get(k) for k in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration")},
            wall_s=wall, raw=resp)

    def identify(self) -> dict[str, Any]:
        """Artifact identity from the local registry listing (manifest digest) and details."""
        tags = self._call("/api/tags", None, 10)
        for m in tags.get("models", []):
            if m.get("name") == self.model or m.get("model") == self.model:
                return {"backend": "ollama", "model": self.model, "digest": m.get("digest"),
                        "details": m.get("details"), "size": m.get("size")}
        raise BackendError("model_missing", f"{self.model} is not installed locally; run `speechflow models prepare`")

    def unload(self) -> None:
        self._call("/api/generate", {"model": self.model, "keep_alive": 0}, 30)
