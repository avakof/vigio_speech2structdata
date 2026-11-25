"""Generation-time schema: what the local LLM is asked to emit (not the public contract).

Deliberately uses only a conservative JSON-Schema subset (object/array/string/boolean,
enum, nullable anyOf, additionalProperties=false). Every other rule (patterns, ranges,
cross-field consistency, evidence, numbers, units, corrections) is enforced independently
in `validate.py`; a grammar-constrained reply is schema-shaped, not necessarily correct.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

GENERATION_SCHEMA_VERSION = "inventory.v1.gen4"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LLMPackage(_Strict):
    per_package_text: str | None  # explicit content per package, e.g. "douze" in "deux cartons de douze bouteilles"
    per_package_unit_spoken: str | None


class LLMSpan(_Strict):
    segment: str  # label of a segment in the prompt, e.g. "S2"
    quote: str  # exact contiguous words copied from that segment


class LLMObservation(_Strict):
    # Evidence first: grammar-constrained decoding follows property order.
    segment: str  # label of the FIRST evidence segment in the prompt, e.g. "S2"
    quote: str  # exact contiguous words copied from that segment
    # Further evidence in later segments when one statement is split across segments
    # (e.g. "Deux cartons de" | "douze bouteilles"). Ascending segment order.
    more_evidence: list[LLMSpan] = []
    operation: Literal["count", "correction", "negation", "unknown"]
    product_spoken: str | None
    quantity_text: str | None
    quantity: str | None = Field(pattern=r"^[0-9]+(?:\.[0-9]+)?$")
    qualifier: Literal["exact", "approximate", "range", "unknown"]
    unit_spoken: str | None
    package: LLMPackage | None
    location: str | None
    ref: str  # label for this observation: "o1", "o2", ...
    correction_of: str | None  # label ("o#" or "k#") of the observation this one corrects


class LLMOutput(_Strict):
    language: Literal["fr", "en", "other"]
    observations: list[LLMObservation]


def generation_json_schema() -> dict:
    schema = LLMOutput.model_json_schema()
    observation = schema["$defs"]["LLMObservation"]
    # New generation explicitly chooses [] or continuation spans. The reader
    # still accepts saved gen2 replies which omitted the default empty array.
    observation["required"].append("more_evidence")
    return schema
