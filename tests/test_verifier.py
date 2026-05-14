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


# ---------------------------------------------------------------------------
# Cross-gene reference check on recommendation prose
# ---------------------------------------------------------------------------


def _two_gene_bundle() -> Bundle[PGxFinding]:
    """Bundle with two findings: CYP2C19 and CYP2D6.

    Both real-world genes; both Bundle-resident — gives the cross-gene
    rule something to detect when a card for one mentions the other.
    """
    cyp2c19 = PGxFinding(
        gene="CYP2C19",
        diplotype="*1/*2",
        source_variants=(),
        phenotype="Intermediate Metabolizer",
        phenotype_source="CPIC",
        affected_drugs=(
            DrugRec(
                drug="clopidogrel",
                recommendation="...",
                cpic_guideline_id="CPIC-1",
                pmids=(1,),
                evidence_level="A",
            ),
        ),
        confidence="high",
    )
    cyp2d6 = PGxFinding(
        gene="CYP2D6",
        diplotype="*1/*3",
        source_variants=(),
        phenotype="Intermediate Metabolizer",
        phenotype_source="CPIC",
        affected_drugs=(
            DrugRec(
                drug="amitriptyline",
                recommendation="...",
                cpic_guideline_id="CPIC-2",
                pmids=(2,),
                evidence_level="B",
            ),
        ),
        confidence="medium",
    )
    return Bundle(
        items=(cyp2c19, cyp2d6),
        privacy_tier=PrivacyTier.PUBLIC,
        source="test",
    )


def _card(gene: str, drug: str, recommendation: str) -> DraftedCard:
    diplotypes = {"CYP2C19": "*1/*2", "CYP2D6": "*1/*3"}
    return DraftedCard(
        gene=gene,
        diplotype=diplotypes[gene],
        phenotype="Intermediate Metabolizer",
        drug=drug,
        recommendation=recommendation,
        cited_pmids=(1,) if gene == "CYP2C19" else (2,),
    )


def test_verifier_rejects_cross_gene_reference_in_prose() -> None:
    """CYP2D6 card mentioning CYP2C19 should fail verification."""
    bad_card = _card(
        "CYP2D6",
        "amitriptyline",
        "Consider an alternative not metabolized by CYP2C19.",
    )
    draft = Draft(cards=(bad_card,), raw_text="")
    result = Verifier().verify(draft, _two_gene_bundle())
    assert not result.passed
    cross_gene = [
        f for f in result.failures if f.reason.startswith("cross-gene")
    ]
    assert len(cross_gene) == 1
    assert cross_gene[0].value == "CYP2C19"
    assert cross_gene[0].field == "recommendation"


def test_verifier_accepts_own_gene_mention_in_prose() -> None:
    """A card's own gene name in the recommendation is fine."""
    good_card = _card(
        "CYP2D6",
        "amitriptyline",
        "As a CYP2D6 Intermediate Metabolizer, consider an alternative.",
    )
    draft = Draft(cards=(good_card,), raw_text="")
    assert Verifier().verify(draft, _two_gene_bundle()).passed


def test_verifier_uses_word_boundary_no_false_positives() -> None:
    """`CYP2C19_v2` should not trigger on the `CYP2C19` match."""
    card = _card(
        "CYP2D6",
        "amitriptyline",
        "Reference: CYP2C19_v2 internal note — see CYP2D6 guidance.",
    )
    draft = Draft(cards=(card,), raw_text="")
    result = Verifier().verify(draft, _two_gene_bundle())
    assert result.passed, (
        f"unexpected failures: {[f.reason for f in result.failures]}"
    )


def test_verifier_flags_multiple_cross_gene_references() -> None:
    """Two other Bundle genes named in one card -> two failures."""
    # Add a third gene to the bundle so we have two "others" to leak.
    bundle = _two_gene_bundle()
    cyp2c9 = PGxFinding(
        gene="CYP2C9",
        diplotype="*1/*1",
        source_variants=(),
        phenotype="Normal Metabolizer",
        phenotype_source="CPIC",
        affected_drugs=(),
        confidence="high",
    )
    bundle = Bundle(
        items=bundle.items + (cyp2c9,),
        privacy_tier=bundle.privacy_tier,
        source=bundle.source,
    )
    card = _card(
        "CYP2D6",
        "amitriptyline",
        "Avoid drugs metabolized by CYP2C19 or CYP2C9.",
    )
    draft = Draft(cards=(card,), raw_text="")
    result = Verifier().verify(draft, bundle)
    cross_gene = [
        f for f in result.failures if f.reason.startswith("cross-gene")
    ]
    leaked = {f.value for f in cross_gene}
    assert leaked == {"CYP2C19", "CYP2C9"}
