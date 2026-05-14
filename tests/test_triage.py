"""Tests for the Triage layer + TemplateDrafter + TriagingDrafter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pgx_digest.bundle import (
    Bundle,
    DrugRec,
    PGxFinding,
    PrivacyTier,
)
from pgx_digest.drafter import (
    AnthropicProvider,
    Drafter,
    Draft,
    DraftedCard,
    LLMDrafter,
    PrivacyViolation,
    TriagingDrafter,
)
from pgx_digest.pharmcat import parse_pharmcat_json
from pgx_digest.triage import (
    SAFETY_FOOTER,
    TemplateDrafter,
    Triage,
    TriageDecision,
)
from pgx_digest.verifier import Verifier

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MULTIGENE_FIXTURE = FIXTURES_DIR / "pharmcat_multigene.json"


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _finding(
    *,
    gene: str = "TPMT",
    phenotype: str = "Normal Metabolizer",
    drug_name: str = "mercaptopurine",
    recommendation: str = "Start with normal starting dose.",
    evidence_level: str = "A",
) -> PGxFinding:
    return PGxFinding(
        gene=gene,
        diplotype="*1/*1",
        source_variants=(),
        phenotype=phenotype,
        phenotype_source="test",
        affected_drugs=(
            DrugRec(
                drug=drug_name,
                recommendation=recommendation,
                cpic_guideline_id="CPIC-T",
                pmids=(1,),
                evidence_level=evidence_level,  # type: ignore[arg-type]
            ),
        ),
        confidence="high",
    )


# ---------------------------------------------------------------------------
# Triage.classify rule coverage
# ---------------------------------------------------------------------------


def test_triage_skips_evidence_level_d() -> None:
    f = _finding(evidence_level="D")
    d = f.affected_drugs[0]
    assert Triage().classify(f, d).route == "skip"


def test_triage_skips_unknown_phenotype() -> None:
    f = _finding(phenotype="Unknown / No Result")
    d = f.affected_drugs[0]
    decision = Triage().classify(f, d)
    assert decision.route == "skip"
    assert "not actionable" in decision.reason


def test_triage_skips_indeterminate_phenotype() -> None:
    f = _finding(phenotype="Indeterminate Metabolizer")
    d = f.affected_drugs[0]
    assert Triage().classify(f, d).route == "skip"


def test_triage_templates_normal_with_short_unconditional_rec() -> None:
    f = _finding(
        phenotype="Normal Metabolizer",
        recommendation="No dose adjustment recommended.",
    )
    d = f.affected_drugs[0]
    decision = Triage().classify(f, d)
    assert decision.route == "template"
    assert "Normal" in decision.reason


def test_triage_routes_normal_with_conditional_rec_to_llm() -> None:
    """Conditional language ('consider', 'avoid'…) defeats templating."""
    f = _finding(
        phenotype="Normal Metabolizer",
        recommendation="Consider an alternative agent if symptoms persist.",
    )
    d = f.affected_drugs[0]
    assert Triage().classify(f, d).route == "llm"


def test_triage_routes_normal_with_long_rec_to_llm() -> None:
    f = _finding(
        phenotype="Normal Metabolizer",
        recommendation="x" * 250,
    )
    d = f.affected_drugs[0]
    assert Triage().classify(f, d).route == "llm"


def test_triage_routes_poor_metabolizer_to_llm() -> None:
    f = _finding(
        phenotype="Poor Metabolizer",
        recommendation="No dose adjustment recommended.",
    )
    d = f.affected_drugs[0]
    assert Triage().classify(f, d).route == "llm"


def test_triage_max_template_length_is_tunable() -> None:
    f = _finding(
        phenotype="Normal Metabolizer",
        recommendation="x" * 150,
    )
    d = f.affected_drugs[0]
    # Default: would template
    assert Triage().classify(f, d).route == "template"
    # With a shorter threshold: routes to LLM
    assert Triage(max_template_length=100).classify(f, d).route == "llm"


# ---------------------------------------------------------------------------
# TemplateDrafter
# ---------------------------------------------------------------------------


def test_template_drafter_emits_well_formed_card() -> None:
    f = _finding(
        gene="TPMT",
        phenotype="Normal Metabolizer",
        drug_name="mercaptopurine",
        recommendation="Start with normal starting dose.",
    )
    card = TemplateDrafter().draft_card(f, f.affected_drugs[0])
    assert card.gene == "TPMT"
    assert card.drug == "mercaptopurine"
    assert card.phenotype == "Normal Metabolizer"
    assert card.cited_pmids == (1,)
    assert SAFETY_FOOTER in card.recommendation
    assert "TPMT" in card.recommendation
    assert "mercaptopurine" in card.recommendation


def test_template_drafter_card_passes_verifier() -> None:
    f = _finding()
    bundle = Bundle(items=(f,), privacy_tier=PrivacyTier.PUBLIC, source="t")
    card = TemplateDrafter().draft_card(f, f.affected_drugs[0])
    assert Verifier().verify(Draft(cards=(card,), raw_text=""), bundle).passed


# ---------------------------------------------------------------------------
# TriagingDrafter integration
# ---------------------------------------------------------------------------


class _CountingDrafter(Drafter):
    """Fake LLMDrafter that records what bundle it was given."""

    def __init__(self, response_cards: tuple[DraftedCard, ...] = ()) -> None:
        self.calls: list[Bundle[PGxFinding]] = []
        self.response_cards = response_cards
        self.last_response = None

    def draft(self, bundle: Bundle[PGxFinding]) -> Draft:
        self.calls.append(bundle)
        return Draft(cards=self.response_cards, raw_text="<fake>")


def _two_finding_bundle() -> Bundle[PGxFinding]:
    """One CYP2C19 PM (needs LLM) + one TPMT Normal (templatable)."""
    cyp = PGxFinding(
        gene="CYP2C19",
        diplotype="*2/*2",
        source_variants=(),
        phenotype="Poor Metabolizer",
        phenotype_source="test",
        affected_drugs=(
            DrugRec(
                drug="clopidogrel",
                recommendation=(
                    "Avoid use. Consider an alternative antiplatelet agent."
                ),
                cpic_guideline_id="CPIC-C",
                pmids=(1,),
                evidence_level="A",
            ),
        ),
        confidence="high",
    )
    tpmt = PGxFinding(
        gene="TPMT",
        diplotype="*1/*1",
        source_variants=(),
        phenotype="Normal Metabolizer",
        phenotype_source="test",
        affected_drugs=(
            DrugRec(
                drug="mercaptopurine",
                recommendation="Start with normal starting dose.",
                cpic_guideline_id="CPIC-T",
                pmids=(2,),
                evidence_level="A",
            ),
        ),
        confidence="high",
    )
    return Bundle(
        items=(cyp, tpmt),
        privacy_tier=PrivacyTier.PUBLIC,
        source="test",
    )


def test_triaging_drafter_routes_mixed_bundle() -> None:
    """One LLM card + one template card → 1 API call, 2 cards out."""
    bundle = _two_finding_bundle()

    # Fake LLM returns the card it was supposed to draft.
    fake_llm_card = DraftedCard(
        gene="CYP2C19",
        diplotype="*2/*2",
        phenotype="Poor Metabolizer",
        drug="clopidogrel",
        recommendation="LLM-generated text.",
        cited_pmids=(1,),
    )
    fake_llm = _CountingDrafter(response_cards=(fake_llm_card,))
    drafter = TriagingDrafter(llm_drafter=fake_llm)
    draft = drafter.draft(bundle)

    # One API call.
    assert len(fake_llm.calls) == 1
    # Trimmed bundle: only the CYP2C19 finding (TPMT was templated out).
    trimmed = fake_llm.calls[0]
    assert {f.gene for f in trimmed.items} == {"CYP2C19"}
    # Two cards out: the LLM one + the template one.
    assert len(draft.cards) == 2
    genes = {c.gene for c in draft.cards}
    assert genes == {"CYP2C19", "TPMT"}
    # Audit trail captures both decisions.
    routes = {(g, d): r for (g, d, r) in drafter.last_decisions}
    assert routes[("CYP2C19", "clopidogrel")] == "llm"
    assert routes[("TPMT", "mercaptopurine")] == "template"


def test_triaging_drafter_skips_llm_call_when_all_templatable() -> None:
    """Fully-Normal bundle → zero API calls."""
    bundle = Bundle(
        items=(
            _finding(
                gene="TPMT",
                phenotype="Normal Metabolizer",
                drug_name="mercaptopurine",
                recommendation="Start with normal starting dose.",
            ),
            _finding(
                gene="DPYD",
                phenotype="Normal Metabolizer",
                drug_name="fluorouracil",
                recommendation="Use label-recommended dose.",
            ),
        ),
        privacy_tier=PrivacyTier.PUBLIC,
        source="test",
    )
    fake_llm = _CountingDrafter()
    drafter = TriagingDrafter(llm_drafter=fake_llm)
    draft = drafter.draft(bundle)

    assert fake_llm.calls == []  # zero API calls
    assert len(draft.cards) == 2
    assert drafter.last_response is None
    # All decisions are template.
    assert all(r == "template" for (_, _, r) in drafter.last_decisions)


def test_triaging_drafter_skips_evidence_level_d_entirely() -> None:
    """Level D pairs produce no card at all."""
    bundle = Bundle(
        items=(
            _finding(
                gene="TPMT",
                phenotype="Normal Metabolizer",
                drug_name="thioguanine",
                recommendation="Use label dose.",
                evidence_level="D",
            ),
        ),
        privacy_tier=PrivacyTier.PUBLIC,
        source="test",
    )
    fake_llm = _CountingDrafter()
    drafter = TriagingDrafter(llm_drafter=fake_llm)
    draft = drafter.draft(bundle)
    assert fake_llm.calls == []
    assert draft.cards == ()


def test_triaging_drafter_inherits_llm_privacy_refusal() -> None:
    """If any case needs LLM and bundle is LOCAL_ONLY, refusal stands."""
    real_llm = LLMDrafter(provider=AnthropicProvider(model="claude-haiku-4-5"))
    bundle = Bundle(
        items=(
            _finding(
                gene="CYP2C19",
                phenotype="Poor Metabolizer",
                drug_name="clopidogrel",
                recommendation="Avoid use. Consider alternative.",
            ),
        ),
        privacy_tier=PrivacyTier.LOCAL_ONLY,
        source="test",
    )
    drafter = TriagingDrafter(llm_drafter=real_llm)
    with pytest.raises(PrivacyViolation):
        drafter.draft(bundle)


def test_triaging_drafter_on_multigene_fixture_reduces_api_calls() -> None:
    """On the real multigene fixture, DPYD Normal drugs are short and
    unconditional ("Based on genotype, there is no indication to change
    dose..."), so they template out. TPMT's Normal recommendations are
    longer with conditional language ("adjust doses based on..."), so
    they correctly stay on the LLM path even though the phenotype is
    Normal. This documents the *real* CPIC-source-text behavior.
    """
    bundle = parse_pharmcat_json(MULTIGENE_FIXTURE)
    fake_llm = _CountingDrafter()
    drafter = TriagingDrafter(llm_drafter=fake_llm)
    drafter.draft(bundle)
    assert len(fake_llm.calls) == 1
    trimmed = fake_llm.calls[0]
    genes_sent_to_llm = {f.gene for f in trimmed.items}
    # DPYD Normal drugs template out (short + unconditional).
    assert "DPYD" not in genes_sent_to_llm
    # TPMT Normal drugs do NOT template — recommendations contain
    # conditional dosing language even at the Normal phenotype.
    assert "TPMT" in genes_sent_to_llm
    # PM/IM phenotypes always go to LLM.
    assert "CYP2C19" in genes_sent_to_llm
    assert "CYP2D6" in genes_sent_to_llm
    # Audit: at least the two DPYD cards templated.
    template_routes = [
        (g, d) for (g, d, r) in drafter.last_decisions if r == "template"
    ]
    assert len(template_routes) >= 2
    assert all(g == "DPYD" for g, _ in template_routes)
