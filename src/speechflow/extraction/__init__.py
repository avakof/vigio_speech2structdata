"""Local structured extraction (inventory.v1). Public entry point: `extract`."""

from .backend import BackendError, BackendReply, ChatBackend, OllamaBackend
from .core import extract, generation_options, plan_windows
from .schema import GENERATION_SCHEMA_VERSION, generation_json_schema

__all__ = ["extract", "generation_options", "plan_windows", "generation_json_schema", "GENERATION_SCHEMA_VERSION",
           "BackendError", "BackendReply", "ChatBackend", "OllamaBackend"]
