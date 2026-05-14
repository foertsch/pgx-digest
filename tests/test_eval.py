"""Tests for the eval harness.

No network calls — the Judge and Drafter are mocked. These tests
exercise: rule-based assertions, the adversarial drafter, the Verifier
detection rate, the runner end-to-end on a real fixture, and the
report writers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from pgx_digest.bundle import (
    Bundle,
    DrugRec,
    PGxFinding,
    PrivacyTier,
    Variant,
)
from pgx_digest.drafter import Draft, DraftedCard
from pgx_digest.eval import (
    AdversarialDrafter,
    AdversarialMode,
    EvalCase,
    Judge,
    check_rules,
    load_cases,
    make_faithful_draft,
    run_case,
    write_ablation_markdown,
    write_results_jsonl,
)
from pgx_digest.eval.cases import SAFETY_FOOTER
from pgx_digest.eval.report import summarize
from pgx_digest.pharmcat import parse_pharmcat_json

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MULTIGENE_FIXTURE = FIXTURES_DIR / "pharmcat_multigene.json"
SINGLE_FIXTURE = FIXTURES_DIR / "pharmcat_cyp2c19_minimal.json"


# ---------------------------------------------------------------------------
# Small bundle + draft factory for fast pure-Python tests.
# ---------------------------------------------------------------------------


def _bundle() -> Bundle[PGxFinding]:
    finding = PGxFinding(
        gene="CYP2C19",
        diplotype="*1/*2",
        source_variants=(
            Variant("rs4244285", "10", 96541616, "AG", "+"),
        ),
        phenotype="Intermediate Metabolizer",
        phenotype_source="test",
        affected_drugs=(
            DrugRec(
                drug="clopidogrel",
                recommendation="Consider alternative.",
                cpic_guideline_id="CPIC-1",
                pmids=(1, 2),
                evidence_level="A",
            ),
        ),
        confidence="high",
    )
    return Bundle(
        items=(finding,),
        privacy_tier=PrivacyTier.PUBLIC,
        source="test",
    )


def _draft(
    *,
    gene: str = "CYP2C19",
    diplotype: str = "*1/*2",
    phenotype: str = "Intermediate Metabolizer",
    drug: str = "clopidogrel",
    recommendation: str = f"Consider alternative. {SAFETY_FOOTER}",
    cited_pmids: tuple[int, ...] = (1,),
) -> Draft:
    return Draft(
        cards=(
            DraftedCard(
                gene=gene,
                diplotype=diplotype,
                phenotype=phenotype,
                drug=drug,
                recommendation=recommendation,
                cited_pmids=cited_pmids,
            ),
        ),
        raw_text="<test>",
    )


# ---------------------------------------------------------------------------
# Multi-gene fixture sanity
# ---------------------------------------------------------------------------


def test_multigene_fixture_parses_four_findings() -> None:
    bundle = parse_pharmcat_json(MULTIGENE_FIXTURE)
    assert len(bundle) == 4
    genes = {f.gene for f in bundle.items}
    assert genes == {"CYP2C19", "CYP2D6", "DPYD", "TPMT"}


def test_multigene_fixture_has_actionable_cyp2c19() -> None:
    bundle = parse_pharmcat_json(MULTIGENE_FIXTURE)
    cyp = next(f for f in bundle.items if f.gene == "CYP2C19")
    assert cyp.diplotype == "*2/*2"
    assert cyp.phenotype == "Poor Metabolizer"
    drugs = {d.drug for d in cyp.affected_drugs}
    assert {"clopidogrel", "voriconazole"} <= drugs


def test_multigene_fixture_has_cyp2d6_im() -> None:
    bundle = parse_pharmcat_json(MULTIGENE_FIXTURE)
    cyp2d6 = next(f for f in bundle.items if f.gene == "CYP2D6")
    assert cyp2d6.diplotype == "*1/*3"
    assert cyp2d6.phenotype == "Intermediate Metabolizer"


# ---------------------------------------------------------------------------
# Eval case loader + rule checks
# ---------------------------------------------------------------------------


def test_load_cases_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id": "x", "fixture": "f.json", "expected_genes": ["CYP2C19"]}\n'
    )
    cases = load_cases(path)
    assert len(cases) == 1
    assert cases[0].id == "x"
    assert cases[0].expected_genes == ("CYP2C19",)


def test_load_cases_skips_blank_and_comment_lines(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '# header comment\n'
        '\n'
        '{"id": "x", "fixture": "f.json"}\n'
        '\n'
    )
    assert len(load_cases(path)) == 1


def _case(**overrides: Any) -> EvalCase:
    defaults: dict[str, Any] = {
        "id": "t",
        "fixture": "test.json",
        "expected_genes": ("CYP2C19",),
        "expected_drugs": ("clopidogrel",),
        "expected_phenotypes": {"CYP2C19": "Intermediate Metabolizer"},
        "must_include_footer": True,
    }
    defaults.update(overrides)
    return EvalCase(**defaults)


def test_rules_pass_on_matching_draft() -> None:
    failures = check_rules(_case(), _draft())
    assert failures == ()


def test_rules_fail_when_expected_gene_missing() -> None:
    failures = check_rules(_case(expected_genes=("BRCA1",)), _draft())
    assert any(f.rule == "expected_gene_missing" for f in failures)


def test_rules_fail_when_expected_drug_missing() -> None:
    failures = check_rules(_case(expected_drugs=("warfarin",)), _draft())
    assert any(f.rule == "expected_drug_missing" for f in failures)


def test_rules_fail_when_unexpected_drug_present() -> None:
    failures = check_rules(
        _case(unexpected_drugs=("clopidogrel",)),
        _draft(),
    )
    assert any(f.rule == "unexpected_drug_present" for f in failures)


def test_rules_fail_when_phenotype_wrong() -> None:
    failures = check_rules(
        _case(expected_phenotypes={"CYP2C19": "Poor Metabolizer"}),
        _draft(),
    )
    assert any(f.rule == "phenotype_mismatch" for f in failures)


def test_rules_fail_when_safety_footer_missing() -> None:
    failures = check_rules(
        _case(),
        _draft(recommendation="Consider alternative."),
    )
    assert any(f.rule == "missing_safety_footer" for f in failures)


def test_rules_skip_footer_when_disabled() -> None:
    failures = check_rules(
        _case(must_include_footer=False),
        _draft(recommendation="Consider alternative."),
    )
    assert failures == ()


# ---------------------------------------------------------------------------
# Adversarial drafter + Verifier detection rate
# ---------------------------------------------------------------------------


def test_faithful_drafter_passes_verifier() -> None:
    from pgx_digest.verifier import Verifier

    bundle = parse_pharmcat_json(SINGLE_FIXTURE)
    draft = AdversarialDrafter(mode=AdversarialMode.FAITHFUL).draft(bundle)
    assert Verifier().verify(draft, bundle).passed


def test_make_faithful_draft_produces_one_card_per_drug() -> None:
    bundle = parse_pharmcat_json(MULTIGENE_FIXTURE)
    draft = make_faithful_draft(bundle)
    expected_n = sum(len(f.affected_drugs) for f in bundle.items)
    assert len(draft.cards) == expected_n


@pytest.mark.parametrize(
    "mode",
    [
        AdversarialMode.SWAP_PMID,
        AdversarialMode.SWAP_DRUG,
        AdversarialMode.SWAP_PHENOTYPE,
        AdversarialMode.SWAP_DIPLOTYPE,
    ],
)
def test_adversarial_drafter_caught_by_verifier(
    mode: AdversarialMode,
) -> None:
    from pgx_digest.verifier import Verifier

    bundle = parse_pharmcat_json(SINGLE_FIXTURE)
    draft = AdversarialDrafter(mode=mode, rate=1.0, seed=42).draft(bundle)
    result = Verifier().verify(draft, bundle)
    assert not result.passed, f"Verifier failed to catch {mode}"
    assert len(result.failures) >= 1


def test_adversarial_drafter_is_seed_deterministic() -> None:
    bundle = parse_pharmcat_json(MULTIGENE_FIXTURE)
    d1 = AdversarialDrafter(
        mode=AdversarialMode.SWAP_PMID, rate=0.5, seed=7
    ).draft(bundle)
    d2 = AdversarialDrafter(
        mode=AdversarialMode.SWAP_PMID, rate=0.5, seed=7
    ).draft(bundle)
    assert d1.cards == d2.cards


# ---------------------------------------------------------------------------
# Mocked Judge
# ---------------------------------------------------------------------------


@dataclass
class _FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeUsage:
    input_tokens: int
    output_tokens: int


class _FakeMessage:
    def __init__(self, text: str, in_tok: int = 10, out_tok: int = 20):
        self.content = [_FakeTextBlock(text=text)]
        self.usage = _FakeUsage(input_tokens=in_tok, output_tokens=out_tok)


class _FakeMessagesEndpoint:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeMessage:
        self.calls.append(kwargs)
        return _FakeMessage(self.response_text)


class _FakeAnthropic:
    def __init__(self, response_text: str):
        self.messages = _FakeMessagesEndpoint(response_text)


def test_judge_parses_structured_output() -> None:
    payload = json.dumps(
        {
            "patient_clarity": 5,
            "clinical_accuracy": 4,
            "actionability": 5,
            "safety_framing": 5,
            "conciseness": 4,
            "comments": "Solid.",
        }
    )
    fake = _FakeAnthropic(payload)
    judge = Judge(client=fake)
    result = judge.judge(_bundle(), _draft())

    assert result.scores.patient_clarity == 5
    assert result.scores.clinical_accuracy == 4
    assert result.scores.mean == pytest.approx((5 + 4 + 5 + 5 + 4) / 5)
    assert result.comments == "Solid."
    # And one API call was made with the expected model.
    assert len(fake.messages.calls) == 1
    assert fake.messages.calls[0]["model"] == "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# Runner end-to-end with adversarial drafter + mocked judge
# ---------------------------------------------------------------------------


def test_run_case_with_faithful_drafter_passes(tmp_path: Path) -> None:
    case = EvalCase(
        id="t",
        fixture="pharmcat_cyp2c19_minimal.json",
        expected_genes=("CYP2C19",),
        expected_drugs=("clopidogrel",),
        expected_phenotypes={"CYP2C19": "Poor Metabolizer"},
    )
    drafter = AdversarialDrafter(mode=AdversarialMode.FAITHFUL)
    result = run_case(
        case,
        fixtures_dir=FIXTURES_DIR,
        drafter=drafter,
        ranker=None,
        verifier_on=True,
        judge=None,
    )
    assert result.verification.passed
    assert result.rule_passed


def test_run_case_with_adversarial_drafter_caught_by_verifier() -> None:
    case = EvalCase(
        id="t",
        fixture="pharmcat_cyp2c19_minimal.json",
        expected_genes=("CYP2C19",),
    )
    drafter = AdversarialDrafter(
        mode=AdversarialMode.SWAP_DRUG, rate=1.0, seed=0
    )
    result = run_case(
        case,
        fixtures_dir=FIXTURES_DIR,
        drafter=drafter,
        verifier_on=True,
    )
    assert not result.verification.passed


def test_run_case_with_verifier_off_does_not_reject() -> None:
    case = EvalCase(id="t", fixture="pharmcat_cyp2c19_minimal.json")
    drafter = AdversarialDrafter(
        mode=AdversarialMode.SWAP_DRUG, rate=1.0, seed=0
    )
    result = run_case(
        case,
        fixtures_dir=FIXTURES_DIR,
        drafter=drafter,
        verifier_on=False,
    )
    assert result.verification.passed  # vacuous pass


def test_run_eval_dedupes_drafts_for_shared_fixture() -> None:
    """Five of six default cases use the multi-gene fixture; the
    Drafter should be invoked exactly once per unique fixture, not
    once per case.
    """
    from pgx_digest.eval import run_eval as _run_eval

    class _CountingDrafter:
        def __init__(self) -> None:
            self.call_count = 0
            self.last_response = None

        def draft(self, bundle):
            self.call_count += 1
            return make_faithful_draft(bundle)

    fixtures = (
        EvalCase(
            id="a", fixture="pharmcat_multigene.json",
        ),
        EvalCase(
            id="b", fixture="pharmcat_multigene.json",
        ),
        EvalCase(
            id="c", fixture="pharmcat_multigene.json",
        ),
        EvalCase(
            id="d", fixture="pharmcat_cyp2c19_minimal.json",
        ),
        EvalCase(
            id="e", fixture="pharmcat_cyp2c19_minimal.json",
        ),
    )
    counter = _CountingDrafter()
    results = _run_eval(
        fixtures,
        fixtures_dir=FIXTURES_DIR,
        drafter=counter,
        ranker=None,
    )
    assert len(results) == 5
    # Two unique fixtures -> two draft calls total.
    assert counter.call_count == 2
    # First case to use each fixture absorbs the latency; cache hits
    # report zero so the summary doesn't double-count.
    multigene = [r for r in results if r.case_id in ("a", "b", "c")]
    single = [r for r in results if r.case_id in ("d", "e")]
    assert sum(r.drafter_latency_s > 0 for r in multigene) == 1
    assert sum(r.drafter_latency_s > 0 for r in single) == 1


def test_run_case_invokes_judge_when_provided() -> None:
    payload = json.dumps(
        {
            "patient_clarity": 4,
            "clinical_accuracy": 4,
            "actionability": 4,
            "safety_framing": 4,
            "conciseness": 4,
            "comments": "fine",
        }
    )
    judge = Judge(client=_FakeAnthropic(payload))
    case = EvalCase(id="t", fixture="pharmcat_cyp2c19_minimal.json")
    result = run_case(
        case,
        fixtures_dir=FIXTURES_DIR,
        drafter=AdversarialDrafter(mode=AdversarialMode.FAITHFUL),
        judge=judge,
    )
    assert result.judge is not None
    assert result.judge.scores.mean == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def test_write_results_jsonl_round_trips(tmp_path: Path) -> None:
    case = EvalCase(id="t", fixture="pharmcat_cyp2c19_minimal.json")
    result = run_case(
        case,
        fixtures_dir=FIXTURES_DIR,
        drafter=AdversarialDrafter(mode=AdversarialMode.FAITHFUL),
    )
    out = write_results_jsonl((result,), tmp_path / "r.jsonl")
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["case_id"] == "t"
    assert row["rule_passed"] is True
    assert row["verifier_passed"] is True


def test_write_ablation_markdown_includes_all_rows(tmp_path: Path) -> None:
    case = EvalCase(id="t", fixture="pharmcat_cyp2c19_minimal.json")
    faithful = run_case(
        case,
        fixtures_dir=FIXTURES_DIR,
        drafter=AdversarialDrafter(mode=AdversarialMode.FAITHFUL),
    )
    swap = run_case(
        case,
        fixtures_dir=FIXTURES_DIR,
        drafter=AdversarialDrafter(
            mode=AdversarialMode.SWAP_DRUG, rate=1.0, seed=0
        ),
    )
    rows = (
        summarize("faithful", (faithful,)),
        summarize("swap_drug", (swap,)),
    )
    out = write_ablation_markdown("test", rows, tmp_path / "a.md")
    txt = out.read_text()
    assert "faithful" in txt
    assert "swap_drug" in txt
    assert "| Variant |" in txt


# ---------------------------------------------------------------------------
# Committed eval_cases.jsonl is loadable + every fixture exists
# ---------------------------------------------------------------------------


def test_committed_eval_cases_jsonl_loads() -> None:
    path = Path(__file__).parent / "eval_cases.jsonl"
    cases = load_cases(path)
    assert len(cases) >= 1
    for case in cases:
        assert (FIXTURES_DIR / case.fixture).exists(), (
            f"missing fixture for case {case.id}: {case.fixture}"
        )
