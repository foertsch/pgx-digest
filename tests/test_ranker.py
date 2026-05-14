"""Unit tests for rank() and LLMRanker.

Deterministic `rank` is pure — tested directly. `LLMRanker` is exercised
via a fake Anthropic client that returns a canned `order` payload, so
no network access is required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from pgx_digest.bundle import (
    Bundle,
    DrugRec,
    PGxFinding,
    PrivacyTier,
)
from pgx_digest.drafter import PrivacyViolation
from pgx_digest.ranker import LLMRanker, rank


def _finding(gene: str, level: str) -> PGxFinding:
    return PGxFinding(
        gene=gene,
        diplotype="*1/*1",
        source_variants=(),
        phenotype="Normal Metabolizer",
        phenotype_source="test",
        affected_drugs=(
            DrugRec(
                drug=f"{gene}-drug",
                recommendation="...",
                cpic_guideline_id="g",
                pmids=(1,),
                evidence_level=level,  # type: ignore[arg-type]
            ),
        ),
        confidence="high",
    )


def _bundle(*findings: PGxFinding, tier: PrivacyTier = PrivacyTier.PUBLIC) -> Bundle[PGxFinding]:
    return Bundle(items=tuple(findings), privacy_tier=tier, source="t")


# ---------------------------------------------------------------------------
# Deterministic ranker
# ---------------------------------------------------------------------------


def test_rank_puts_level_a_first() -> None:
    bundle = _bundle(
        _finding("ZZZ", "B"),
        _finding("AAA", "A"),
        _finding("MMM", "D"),
    )
    out = rank(bundle)
    assert [f.gene for f in out.items] == ["AAA", "ZZZ", "MMM"]


def test_rank_breaks_ties_by_gene_name() -> None:
    bundle = _bundle(
        _finding("CYP2C19", "A"),
        _finding("CYP2C9", "A"),
    )
    out = rank(bundle)
    assert [f.gene for f in out.items] == ["CYP2C19", "CYP2C9"]


def test_rank_preserves_privacy_tier_and_metadata() -> None:
    bundle = Bundle(
        items=(_finding("X", "A"),),
        privacy_tier=PrivacyTier.PSEUDONYMIZED,
        source="src",
        metadata={"k": "v"},
    )
    out = rank(bundle)
    assert out.privacy_tier == PrivacyTier.PSEUDONYMIZED
    assert out.metadata == {"k": "v"}
    assert out.source == "src"


# ---------------------------------------------------------------------------
# LLMRanker — privacy refusal + happy path + fallback
# ---------------------------------------------------------------------------


@dataclass
class _Block:
    text: str
    type: str = "text"


class _Msg:
    def __init__(self, text: str) -> None:
        self.content = [_Block(text=text)]


class _Endpoint:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _Msg:
        self.calls.append(kwargs)
        return _Msg(self.text)


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.messages = _Endpoint(text)


def _ranker_with_response(text: str) -> LLMRanker:
    r = LLMRanker.__new__(LLMRanker)
    r.model = "claude-haiku-4-5"
    r.max_tokens = 512
    r._client = _FakeClient(text)  # type: ignore[attr-defined]
    return r


def test_llm_ranker_refuses_local_only_bundle() -> None:
    bundle = _bundle(_finding("X", "A"), tier=PrivacyTier.LOCAL_ONLY)
    with pytest.raises(PrivacyViolation):
        _ranker_with_response("{}").rank(bundle)


def test_llm_ranker_returns_input_when_one_finding() -> None:
    bundle = _bundle(_finding("X", "A"))
    # No API call should be needed — the ranker short-circuits.
    out = _ranker_with_response("{}").rank(bundle)
    assert out.items == bundle.items


def test_llm_ranker_applies_returned_order() -> None:
    bundle = _bundle(
        _finding("AAA", "B"),
        _finding("BBB", "A"),
        _finding("CCC", "C"),
    )
    payload = json.dumps({"order": ["CCC", "AAA", "BBB"]})
    out = _ranker_with_response(payload).rank(bundle)
    assert [f.gene for f in out.items] == ["CCC", "AAA", "BBB"]


def test_llm_ranker_falls_back_to_deterministic_on_invalid_order() -> None:
    bundle = _bundle(
        _finding("AAA", "B"),
        _finding("BBB", "A"),
    )
    # LLM drops a gene -> invalid permutation -> fallback to rank().
    payload = json.dumps({"order": ["AAA"]})
    out = _ranker_with_response(payload).rank(bundle)
    # rank() puts Level A first -> BBB before AAA.
    assert [f.gene for f in out.items] == ["BBB", "AAA"]
