"""Unit tests for the Verifier.

The Verifier is pure — no API calls, no I/O. Tests run fast and do not
require ANTHROPIC_API_KEY.
"""

from __future__ import annotations

from typing import Any

from pgx_digest.bundle import (
    Bundle,
    DrugRec,
    PGxFinding,
    PrivacyTier,
    Variant,
)
from pgx_digest.drafter import Draft, DraftedCard
from pgx_digest.verifier import Verifier


def _make_bundle() -> Bundle[PGxFinding]:
    """Minimal Bundle with one CYP2C19 finding for clopidogrel."""
    variant = Variant(
        rsid="rs4244285",
        chromosome="10",
        position=96541616,
        genotype="AG",
        reference_strand="+",
    )
    drug_rec = DrugRec(
        drug="clopidogrel",
        recommendation=(
            "Reduced clopidogrel activation; consider alternative "
            "antiplatelet therapy."
        ),
        cpic_guideline_id="CPIC-GUI-1",
        pmids=(28198005, 30093137),
        evidence_level="A",
    )
    finding = PGxFinding(
        gene="CYP2C19",
        diplotype="*1/*2",
        source_variants=(variant,),
        phenotype="Intermediate Metabolizer",
        phenotype_source="CPIC CYP2C19 Allele Functionality Table",
        affected_drugs=(drug_rec,),
        confidence="high",
    )
    return Bundle(
        items=(finding,),
        privacy_tier=PrivacyTier.PUBLIC,
        source="test-fixture",
    )


def _make_draft(**overrides: Any) -> Draft:
    base: dict[str, Any] = {
        "gene": "CYP2C19",
        "diplotype": "*1/*2",
        "phenotype": "Intermediate Metabolizer",
        "drug": "clopidogrel",
        "recommendation": "Reduced metabolism; discuss alternatives.",
        "cited_pmids": (28198005,),
    }
    base.update(overrides)
    return Draft(cards=(DraftedCard(**base),), raw_text="")


def test_verifier_accepts_valid_draft() -> None:
    result = Verifier().verify(_make_draft(), _make_bundle())
    assert result.passed
    assert not result.failures


def test_verifier_rejects_unknown_gene() -> None:
    result = Verifier().verify(_make_draft(gene="BRCA1"), _make_bundle())
    assert not result.passed
    assert any(f.field == "gene" for f in result.failures)


def test_verifier_rejects_unknown_drug() -> None:
    result = Verifier().verify(_make_draft(drug="aspirin"), _make_bundle())
    assert not result.passed
    assert any(f.field == "drug" for f in result.failures)


def test_verifier_rejects_diplotype_mismatch() -> None:
    result = Verifier().verify(_make_draft(diplotype="*1/*1"), _make_bundle())
    assert not result.passed
    assert any(f.field == "diplotype" for f in result.failures)


def test_verifier_rejects_fabricated_pmid() -> None:
    result = Verifier().verify(
        _make_draft(cited_pmids=(99999999,)),
        _make_bundle(),
    )
    assert not result.passed
    assert any(f.field == "cited_pmids" for f in result.failures)


def test_verifier_rejects_phenotype_mismatch() -> None:
    result = Verifier().verify(
        _make_draft(phenotype="Poor Metabolizer"),
        _make_bundle(),
    )
    assert not result.passed
    assert any(f.field == "phenotype" for f in result.failures)
