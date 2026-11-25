"""PRIVATE feasibility schema + validators for the extraction experiment (Gate A).

NOT the frozen public contract. Exists only to exercise Ollama schema-constrained
generation and independent Pydantic validation. Replace with the Gate B contracts.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DECIMAL_RE = r"^-?[0-9]+(\.[0-9]+)?$"

Unit = Literal["kg", "g", "l", "ml", "cl", "bouteille", "carton", "boite", "sac", "paquet", "piece"]
Operation = Literal["stock_count", "add", "remove", "unknown"]


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    obs_id: str = Field(pattern=r"^o[0-9]+$")
    product_as_spoken: str = Field(min_length=1)
    quantity_text: str | None
    quantity: str | None = Field(pattern=DECIMAL_RE)  # decimal string; null = unknown (never 0)
    unit_text: str | None
    unit: Unit | None
    package_contents_quantity: str | None = Field(pattern=DECIMAL_RE)
    package_contents_unit: Unit | None
    location: str | None
    operation: Operation
    corrects: str | None = Field(pattern=r"^o[0-9]+$")  # obs_id this observation corrects
    superseded: bool  # True when a later observation corrects this one
    segment_ids: list[str] = Field(min_length=1)
    evidence_text: str = Field(min_length=1)

    @field_validator("quantity", "package_contents_quantity")
    @classmethod
    def _decimal(cls, v: str | None) -> str | None:
        if v is None:
            return v
        try:
            Decimal(v)
        except InvalidOperation as e:  # pragma: no cover
            raise ValueError(f"not a decimal: {v!r}") from e
        return v


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observations: list[Observation]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("…", "...")).strip().lower()


def independent_checks(ext: Extraction, segments: dict[str, str]) -> list[str]:
    """Checks that do NOT rely on the generation grammar. Returns error strings."""
    errs: list[str] = []
    ids = [o.obs_id for o in ext.observations]
    if len(set(ids)) != len(ids):
        errs.append(f"duplicate obs_id: {ids}")
    idset = set(ids)
    for o in ext.observations:
        for sid in o.segment_ids:
            if sid not in segments:
                errs.append(f"{o.obs_id}: unknown segment id {sid}")
        joined = norm(" ".join(segments.get(s, "") for s in o.segment_ids))
        if norm(o.evidence_text) not in joined:
            errs.append(f"{o.obs_id}: evidence_text not a substring of cited segments: {o.evidence_text!r}")
        if o.quantity is not None and o.quantity_text is None:
            errs.append(f"{o.obs_id}: quantity without quantity_text")
        if o.unit is not None and o.unit_text is None:
            errs.append(f"{o.obs_id}: unit without unit_text")
        if o.corrects is not None:
            if o.corrects not in idset:
                errs.append(f"{o.obs_id}: corrects unknown {o.corrects}")
            elif o.corrects == o.obs_id:
                errs.append(f"{o.obs_id}: corrects itself")
        if o.quantity is not None and Decimal(o.quantity) < 0:
            errs.append(f"{o.obs_id}: negative quantity {o.quantity}")
    # superseded flag must agree with correction references
    corrected = {o.corrects for o in ext.observations if o.corrects}
    for o in ext.observations:
        if o.superseded != (o.obs_id in corrected):
            errs.append(f"{o.obs_id}: superseded={o.superseded} inconsistent with corrects graph")
    return errs


def derived_total(o: Observation) -> Decimal | None:
    """Deterministic total only from explicit statement (count * per-package)."""
    if o.quantity is None or o.package_contents_quantity is None:
        return None
    return Decimal(o.quantity) * Decimal(o.package_contents_quantity)


def final_view(ext: Extraction) -> list[Observation]:
    return [o for o in ext.observations if not o.superseded]


# Fixture semantic expectations (checked separately from schema validity).
# Each returns list of failure strings; empty = meets expectation.
def _find(ext: Extraction, needle: str, final_only=True):
    obs = final_view(ext) if final_only else ext.observations
    return [o for o in obs if needle in norm(o.product_as_spoken)]


def expect_inventory(ext: Extraction) -> list[str]:
    f = []
    t = _find(ext, "tomate")
    if len(t) != 1 or t[0].quantity is None or Decimal(t[0].quantity) != Decimal("2.5") or t[0].unit != "kg":
        f.append(f"tomatoes != 2.5 kg: {[(o.quantity, o.unit) for o in t]}")
    m = _find(ext, "lait")
    if len(m) != 1 or m[0].quantity is None or Decimal(m[0].quantity) != 12 or m[0].unit != "bouteille":
        f.append(f"milk != 12 bouteille: {[(o.quantity, o.unit) for o in m]}")
    elif m[0].package_contents_quantity is not None or m[0].package_contents_unit in ("l", "ml", "cl"):
        f.append(f"milk bottle volume invented: {m[0].package_contents_quantity} {m[0].package_contents_unit}")
    return f


def expect_correction(ext: Extraction) -> list[str]:
    f = []
    fl = _find(ext, "farine")
    if len(fl) != 1 or fl[0].quantity is None or Decimal(fl[0].quantity) != 2 or fl[0].unit != "kg":
        f.append(f"final flour != 2 kg: {[(o.quantity, o.unit) for o in fl]}")
    else:
        old = [o for o in _find(ext, "farine", final_only=False) if o.superseded]
        if not any(o.quantity is not None and Decimal(o.quantity) == 3 for o in old):
            f.append("superseded value 3 kg not preserved")
        if fl[0].corrects is None:
            f.append("final flour does not reference corrected observation")
    return f


def expect_cartons_unknown(ext: Extraction) -> list[str]:
    f = []
    m = _find(ext, "lait")
    if len(m) != 1 or m[0].quantity is None or Decimal(m[0].quantity) != 2 or m[0].unit != "carton":
        f.append(f"expected 2 carton: {[(o.quantity, o.unit) for o in m]}")
    elif m[0].package_contents_quantity is not None:
        f.append(f"invented carton contents {m[0].package_contents_quantity}")
    return f


def expect_cartons_explicit(ext: Extraction) -> list[str]:
    f = []
    obs = final_view(ext)
    if len(obs) != 1:
        return [f"expected 1 final observation, got {len(obs)}"]
    o = obs[0]
    if o.unit != "carton" or o.quantity is None or Decimal(o.quantity) != 2:
        f.append(f"expected 2 carton, got {o.quantity} {o.unit}")
    if o.package_contents_quantity is None or Decimal(o.package_contents_quantity) != 12 or o.package_contents_unit != "bouteille":
        f.append(f"expected contents 12 bouteille, got {o.package_contents_quantity} {o.package_contents_unit}")
    tot = derived_total(o)
    if tot != 24:
        f.append(f"derived total != 24: {tot}")
    return f
