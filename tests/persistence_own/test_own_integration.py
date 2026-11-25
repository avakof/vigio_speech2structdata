"""Persistence against observations produced by the real extraction validator.

The chat backend here is a SCRIPTED stand-in for the model (no Ollama call). It
exercises the real extraction code and the real persistence code together; it
is not evidence of model quality.
"""

from __future__ import annotations

import json

import pytest
from own_factory import make_manifest

from speechflow import persistence as P
from speechflow.config import Settings

extraction = pytest.importorskip("speechflow.extraction")
from speechflow.extraction.backend import BackendReply  # noqa: E402


class Scripted:
    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)

    def identify(self):
        return {"model": "scripted", "digest": "none"}

    def chat(self, messages, schema, *, options, timeout):
        return BackendReply(content=json.dumps(self.replies.pop(0)), done_reason="stop")


def obs(**kw):
    base = dict(segment="S1", operation="count", product_spoken=None, quantity_text=None,
                quantity=None, qualifier="exact", unit_spoken=None, package=None, location=None,
                correction_of=None)
    return {**base, **kw}


def test_real_validator_output_persists_and_exports_consistently(tmp_path):
    text = "Trois kilos de farine non deux kilos de farine et deux cartons de douze bouteilles"
    manifest = make_manifest(texts=[text], observations=[])
    reply = {"language": "fr", "observations": [
        obs(ref="o1", quote="Trois kilos de farine", product_spoken="farine",
            quantity_text="Trois", quantity="3", unit_spoken="kilos"),
        obs(ref="o2", quote="deux kilos de farine", operation="correction", product_spoken="farine",
            quantity_text="deux", quantity="2", unit_spoken="kilos", correction_of="o1"),
        obs(ref="o3", quote="deux cartons de douze bouteilles", product_spoken="bouteilles",
            quantity_text="deux", quantity="2", unit_spoken="cartons",
            package={"per_package_text": "douze", "per_package_unit_spoken": "bouteilles"}),
    ]}
    result = extraction.extract(manifest.segments, run_id=manifest.run_id, settings=Settings(),
                                backend=Scripted([reply]))
    assert result.observations, result.errors
    final = manifest.model_copy(update={
        "observations": result.observations, "attempts": result.attempts,
        "status": "ok" if result.status == "ok" else result.status,
        "extraction_status": result.status, "errors": result.errors})
    saved = P.save_run(final, tmp_path / "d.sqlite3", tmp_path / "o")
    assert saved.disposition == "inserted"
    rows = P.observation_rows(P.load_manifest(tmp_path / "d.sqlite3", final.run_id))
    flour = [r for r in rows if r["product_spoken"] == "farine"]
    assert [r["is_final"] for r in flour] == ["false", "true"]
    assert [r["quantity"] for r in flour] == ["3", "2"]
    assert P.save_run(final, tmp_path / "d.sqlite3", tmp_path / "o2").disposition == "duplicate"
    assert P.verify(tmp_path / "d.sqlite3") == []


def test_pending_correction_does_not_hide_the_original(tmp_path):
    """The validator keeps a correction without product evidence at needs_review;
    only a confirmed (status ok) revision makes the earlier value non-final."""
    text = "Trois kilos de farine non deux kilos"
    manifest = make_manifest(texts=[text], observations=[])
    reply = {"language": "fr", "observations": [
        obs(ref="o1", quote="Trois kilos de farine", product_spoken="farine",
            quantity_text="Trois", quantity="3", unit_spoken="kilos"),
        obs(ref="o2", quote="deux kilos", operation="correction", product_spoken="farine",
            quantity_text="deux", quantity="2", unit_spoken="kilos", correction_of="o1")]}
    result = extraction.extract(manifest.segments, run_id=manifest.run_id, settings=Settings(),
                                backend=Scripted([reply]))
    final = manifest.model_copy(update={
        "observations": result.observations, "status": result.status,
        "extraction_status": result.status, "errors": result.errors})
    P.save_run(final, tmp_path / "d.sqlite3", tmp_path / "o")
    first, second = P.observation_rows(P.load_manifest(tmp_path / "d.sqlite3", final.run_id))
    assert (second["status"], second["is_final"]) == ("needs_review", "true")
    assert (first["is_final"], first["corrected_by"]) == ("true", second["observation_id"])
    assert second["errors"] and result.status == "needs_review"
