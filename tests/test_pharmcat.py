"""Tests for the PharmCAT JSON parser.

Uses tests/fixtures/pharmcat_cyp2c19_minimal.json — a subset of the
official pharmcat.example2.report.json (CYP2C19 *2/*2 case with
clopidogrel + voriconazole recommendations).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pgx_digest.bundle import PrivacyTier
from pgx_digest.pharmcat import parse_pharmcat_json

FIXTURE = Path(__file__).parent / "fixtures" / "pharmcat_cyp2c19_minimal.json"


def test_fixture_exists() -> None:
    assert FIXTURE.exists(), f"missing fixture: {FIXTURE}"


def test_parse_returns_bundle_with_one_finding() -> None:
    bundle = parse_pharmcat_json(FIXTURE)
    assert len(bundle) == 1
    assert bundle.privacy_tier == PrivacyTier.PUBLIC


def test_finding_has_correct_cyp2c19_call() -> None:
    bundle = parse_pharmcat_json(FIXTURE)
    finding = bundle.items[0]
    assert finding.gene == "CYP2C19"
    assert finding.diplotype == "*2/*2"
    assert finding.phenotype == "Poor Metabolizer"


def test_finding_has_clopidogrel_drug_rec() -> None:
    bundle = parse_pharmcat_json(FIXTURE)
    finding = bundle.items[0]
    drug_names = {d.drug for d in finding.affected_drugs}
    assert "clopidogrel" in drug_names


def test_finding_has_voriconazole_drug_rec() -> None:
    bundle = parse_pharmcat_json(FIXTURE)
    finding = bundle.items[0]
    drug_names = {d.drug for d in finding.affected_drugs}
    assert "voriconazole" in drug_names


def test_drug_rec_carries_real_pmids() -> None:
    bundle = parse_pharmcat_json(FIXTURE)
    finding = bundle.items[0]
    by_name = {d.drug: d for d in finding.affected_drugs}
    clop = by_name["clopidogrel"]
    # Clopidogrel CPIC guideline cites 3 papers
    assert len(clop.pmids) == 3
    assert all(isinstance(p, int) for p in clop.pmids)
    # Known PMIDs for CPIC clopidogrel guideline
    assert 21716271 in clop.pmids


def test_drug_rec_has_strongest_classification() -> None:
    """Clopidogrel example has 3 annotations: Strong, Moderate, Moderate.

    Parser must pick the Strong one (mapped to evidence_level 'A').
    """
    bundle = parse_pharmcat_json(FIXTURE)
    finding = bundle.items[0]
    clop = next(d for d in finding.affected_drugs if d.drug == "clopidogrel")
    assert clop.evidence_level == "A"


def test_drug_rec_recommendation_is_nonempty() -> None:
    bundle = parse_pharmcat_json(FIXTURE)
    finding = bundle.items[0]
    for drug_rec in finding.affected_drugs:
        assert drug_rec.recommendation
        assert len(drug_rec.recommendation) > 20


def test_metadata_carries_pharmcat_version() -> None:
    bundle = parse_pharmcat_json(FIXTURE)
    assert "pharmcat_version" in bundle.metadata
    assert bundle.metadata["pharmcat_version"].startswith("v")


def test_confidence_reflects_phasing() -> None:
    bundle = parse_pharmcat_json(FIXTURE)
    finding = bundle.items[0]
    # The example2 CYP2C19 call is effectivelyPhased=true, matchScore=6
    assert finding.confidence == "high"


def test_source_filename_in_bundle() -> None:
    bundle = parse_pharmcat_json(FIXTURE)
    assert bundle.source == "pharmcat_cyp2c19_minimal.json"
