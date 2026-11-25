"""Validated configuration for local inference."""

from __future__ import annotations

import ipaddress
import os
import tomllib
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from .contracts import Contract


class Settings(Contract):
    offline: bool = False
    hojo_dir: Path = Path.home() / ".cache/speechflow/assets/hojo"
    backend: Literal["cpu", "mps"] = "cpu"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: Literal["qwen3:8b"] = "qwen3:8b"
    max_chunk_seconds: float = Field(default=25, ge=1/16000, le=25)
    padding_ms: int = Field(default=200, ge=0, le=2000)
    max_spool_bytes: int = Field(default=2**30, gt=0)
    mic_queue_seconds: float = Field(default=15, gt=0, le=120)
    mic_seconds: float | None = Field(default=None, gt=0)
    mic_device: str | int | None = None
    extraction_timeout: float = Field(default=120, gt=0, le=600)
    context_tokens: int = Field(default=8192, ge=2048, le=32768)
    output_tokens: int = Field(default=2048, ge=256, le=8192)
    database: Path = Path.home() / ".local/share/speechflow/inventory.sqlite3"

    @model_validator(mode="after")
    def check_generation_budget(self):
        if self.output_tokens >= self.context_tokens:
            raise ValueError("context_tokens must leave room for the prompt and the output budget")
        return self

    @field_validator("ollama_url")
    @classmethod
    def loopback(cls, value: str) -> str:
        url = urlsplit(value)
        if (url.scheme != "http" or url.username or url.password or url.query or
                url.fragment or url.path not in ("", "/") or not url.hostname or
                not ipaddress.ip_address(url.hostname).is_loopback):
            raise ValueError("Use a numeric HTTP loopback address for Ollama")
        _ = url.port
        return value.rstrip("/")

    @field_validator("hojo_dir", "database")
    @classmethod
    def absolute_path(cls, path: Path) -> Path:
        return path.expanduser().resolve()


def load_settings(path: Path | None = None, **overrides) -> Settings:
    path = path or (Path(os.environ["SPEECHFLOW_CONFIG"]) if os.getenv("SPEECHFLOW_CONFIG") else None)
    values = tomllib.loads(path.read_text()) if path else {}
    values.update({k: v for k, v in overrides.items() if v is not None})
    return Settings.model_validate(values)
