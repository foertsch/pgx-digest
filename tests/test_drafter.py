"""Tests for the Drafter abstraction and its providers.

The real LLM clients are never called — both AnthropicProvider and
GeminiProvider accept an injected `client` so tests run offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from pgx_digest.bundle import (
    Bundle,
    DrugRec,
    PGxFinding,
    PrivacyTier,
    Variant,
)
from pgx_digest.drafter import (
    AnthropicProvider,
    DraftedCard,
    GeminiProvider,
    LLMDrafter,
    OllamaDrafter,
    PrivacyViolation,
    ProviderResponse,
    select_drafter,
)


# ---------------------------------------------------------------------------
# Bundle factory
# ---------------------------------------------------------------------------


def _bundle(
    tier: PrivacyTier = PrivacyTier.PUBLIC,
) -> Bundle[PGxFinding]:
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
    return Bundle(items=(finding,), privacy_tier=tier, source="test")


# Canonical valid Drafter output for a single-card draft.
_VALID_CARDS_JSON = json.dumps(
    {
        "cards": [
            {
                "gene": "CYP2C19",
                "diplotype": "*1/*2",
                "phenotype": "Intermediate Metabolizer",
                "drug": "clopidogrel",
                "recommendation": "Consider alternative.",
                "cited_pmids": [1, 2],
            }
        ]
    }
)


# ---------------------------------------------------------------------------
# Fake Anthropic client (matches the SDK surface our provider uses)
# ---------------------------------------------------------------------------


@dataclass
class _AnthropicBlock:
    text: str
    type: str = "text"


@dataclass
class _AnthropicUsage:
    input_tokens: int = 12
    output_tokens: int = 34
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class _AnthropicMessage:
    content: list[_AnthropicBlock]
    usage: _AnthropicUsage


class _AnthropicMessagesEndpoint:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _AnthropicMessage:
        self.calls.append(kwargs)
        return _AnthropicMessage(
            content=[_AnthropicBlock(text=self.text)],
            usage=_AnthropicUsage(
                input_tokens=12,
                output_tokens=34,
                cache_creation_input_tokens=5,
                cache_read_input_tokens=7,
            ),
        )


class _FakeAnthropicClient:
    def __init__(self, text: str) -> None:
        self.messages = _AnthropicMessagesEndpoint(text)


# ---------------------------------------------------------------------------
# Fake Gemini client (matches the google-genai surface our provider uses)
# ---------------------------------------------------------------------------


@dataclass
class _GeminiUsage:
    prompt_token_count: int = 15
    candidates_token_count: int = 42
    cached_content_token_count: int = 0


@dataclass
class _GeminiResponse:
    text: str
    usage_metadata: _GeminiUsage = field(default_factory=_GeminiUsage)


class _GeminiModelsEndpoint:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> _GeminiResponse:
        self.calls.append(kwargs)
        return _GeminiResponse(text=self.text)


class _FakeGeminiClient:
    def __init__(self, text: str) -> None:
        self.models = _GeminiModelsEndpoint(text)


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------


def test_anthropic_provider_calls_with_cache_control() -> None:
    client = _FakeAnthropicClient(_VALID_CARDS_JSON)
    provider = AnthropicProvider(model="claude-haiku-4-5", client=client)
    resp = provider.generate(
        system="SYS", user="USR", schema={"type": "object"}, max_tokens=64
    )

    assert isinstance(resp, ProviderResponse)
    assert resp.text == _VALID_CARDS_JSON
    assert resp.input_tokens == 12
    assert resp.output_tokens == 34
    assert resp.cache_creation_tokens == 5
    assert resp.cache_read_tokens == 7

    # Verify the call shape matches what Anthropic expects.
    [call] = client.messages.calls
    assert call["model"] == "claude-haiku-4-5"
    assert call["max_tokens"] == 64
    # System carries cache_control marker (a project invariant).
    sys_block = call["system"][0]
    assert sys_block["text"] == "SYS"
    assert sys_block["cache_control"] == {"type": "ephemeral"}
    # Output config requests JSON-schema constrained output.
    assert call["output_config"]["format"]["type"] == "json_schema"


# ---------------------------------------------------------------------------
# GeminiProvider
# ---------------------------------------------------------------------------


def test_gemini_provider_passes_system_instruction_and_schema() -> None:
    client = _FakeGeminiClient(_VALID_CARDS_JSON)
    provider = GeminiProvider(model="gemini-2.5-flash", client=client)
    resp = provider.generate(
        system="SYS", user="USR", schema={"type": "object"}, max_tokens=64
    )

    assert isinstance(resp, ProviderResponse)
    assert resp.text == _VALID_CARDS_JSON
    assert resp.input_tokens == 15
    assert resp.output_tokens == 42

    [call] = client.models.calls
    assert call["model"] == "gemini-2.5-flash"
    assert call["contents"] == "USR"
    config = call["config"]
    # google-genai's GenerateContentConfig should carry system + schema.
    assert config.system_instruction == "SYS"
    assert config.response_mime_type == "application/json"
    assert config.response_schema == {"type": "object"}
    assert config.max_output_tokens == 64


def test_gemini_provider_sets_thinking_budget_zero_on_2_5_models() -> None:
    """2.5-flash has reasoning on by default; we must disable it."""
    client = _FakeGeminiClient(_VALID_CARDS_JSON)
    provider = GeminiProvider(model="gemini-2.5-flash", client=client)
    provider.generate(
        system="S", user="U", schema={"type": "object"}, max_tokens=64
    )
    [call] = client.models.calls
    cfg = call["config"]
    assert cfg.thinking_config is not None
    assert cfg.thinking_config.thinking_budget == 0


def test_gemini_provider_omits_thinking_config_on_2_0_models() -> None:
    """2.0-flash rejects `thinking_config`; the provider must omit it."""
    client = _FakeGeminiClient(_VALID_CARDS_JSON)
    provider = GeminiProvider(model="gemini-2.0-flash", client=client)
    provider.generate(
        system="S", user="U", schema={"type": "object"}, max_tokens=64
    )
    [call] = client.models.calls
    cfg = call["config"]
    assert getattr(cfg, "thinking_config", None) is None


def test_gemini_provider_strips_additional_properties_from_schema() -> None:
    """Gemini's response_schema rejects `additionalProperties`."""
    client = _FakeGeminiClient(_VALID_CARDS_JSON)
    provider = GeminiProvider(client=client)
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "additionalProperties": False,
                },
            }
        },
        "additionalProperties": False,
    }
    provider.generate(
        system="SYS", user="USR", schema=schema, max_tokens=64
    )
    [call] = client.models.calls
    sent = call["config"].response_schema

    def _has_additional_properties(node: object) -> bool:
        if isinstance(node, dict):
            if "additionalProperties" in node:
                return True
            return any(_has_additional_properties(v) for v in node.values())
        if isinstance(node, list):
            return any(_has_additional_properties(v) for v in node)
        return False

    assert not _has_additional_properties(sent)


# ---------------------------------------------------------------------------
# LLMDrafter
# ---------------------------------------------------------------------------


def test_llm_drafter_refuses_local_only_anthropic() -> None:
    drafter = LLMDrafter(
        provider=AnthropicProvider(
            client=_FakeAnthropicClient(_VALID_CARDS_JSON)
        )
    )
    with pytest.raises(PrivacyViolation, match="anthropic"):
        drafter.draft(_bundle(tier=PrivacyTier.LOCAL_ONLY))


def test_llm_drafter_refuses_local_only_gemini() -> None:
    drafter = LLMDrafter(
        provider=GeminiProvider(
            client=_FakeGeminiClient(_VALID_CARDS_JSON)
        )
    )
    with pytest.raises(PrivacyViolation, match="gemini"):
        drafter.draft(_bundle(tier=PrivacyTier.LOCAL_ONLY))


def test_llm_drafter_emits_typed_cards_via_anthropic() -> None:
    drafter = LLMDrafter(
        provider=AnthropicProvider(
            client=_FakeAnthropicClient(_VALID_CARDS_JSON)
        )
    )
    draft = drafter.draft(_bundle())
    assert len(draft.cards) == 1
    card = draft.cards[0]
    assert card.gene == "CYP2C19"
    assert card.drug == "clopidogrel"
    assert card.cited_pmids == (1, 2)


def test_llm_drafter_emits_typed_cards_via_gemini() -> None:
    drafter = LLMDrafter(
        provider=GeminiProvider(
            client=_FakeGeminiClient(_VALID_CARDS_JSON)
        )
    )
    draft = drafter.draft(_bundle())
    assert len(draft.cards) == 1
    assert draft.cards[0].gene == "CYP2C19"


def test_llm_drafter_exposes_last_response() -> None:
    drafter = LLMDrafter(
        provider=AnthropicProvider(
            client=_FakeAnthropicClient(_VALID_CARDS_JSON)
        )
    )
    drafter.draft(_bundle())
    assert isinstance(drafter.last_response, ProviderResponse)
    assert drafter.last_response.input_tokens == 12


def test_llm_drafter_redacts_source_variants_by_default() -> None:
    """The default redact_provenance=True strips raw variant calls from
    the JSON payload sent to the provider. The LLM doesn't need rsids
    or chromosomal positions to write prose; only diplotype names.
    """
    client = _FakeAnthropicClient(_VALID_CARDS_JSON)
    drafter = LLMDrafter(provider=AnthropicProvider(client=client))
    bundle = _bundle()
    # Sanity: the bundle has variants we'd want redacted.
    assert bundle.items[0].source_variants[0].rsid == "rs4244285"

    drafter.draft(bundle)

    [call] = client.messages.calls
    user_payload = call["messages"][0]["content"]
    assert "rs4244285" not in user_payload
    assert "96541616" not in user_payload  # variant position
    # But the diplotype name and gene must remain.
    assert "*1/*2" in user_payload
    assert "CYP2C19" in user_payload


def test_llm_drafter_includes_variants_when_redaction_disabled() -> None:
    """`redact_provenance=False` ships the full bundle (debug mode)."""
    client = _FakeAnthropicClient(_VALID_CARDS_JSON)
    drafter = LLMDrafter(
        provider=AnthropicProvider(client=client),
        redact_provenance=False,
    )
    drafter.draft(_bundle())

    [call] = client.messages.calls
    user_payload = call["messages"][0]["content"]
    assert "rs4244285" in user_payload
    assert "96541616" in user_payload


def test_llm_drafter_redaction_does_not_affect_verifier() -> None:
    """Redaction is only applied to the LLM payload. The Verifier still
    sees the full Bundle object and can check provenance fields.
    """
    from pgx_digest.verifier import Verifier

    client = _FakeAnthropicClient(_VALID_CARDS_JSON)
    drafter = LLMDrafter(provider=AnthropicProvider(client=client))
    bundle = _bundle()
    draft = drafter.draft(bundle)
    # The standard Verifier still passes on the resulting draft —
    # redaction only changed what the LLM saw, not the Bundle object.
    assert Verifier().verify(draft, bundle).passed


def test_llm_drafter_model_property_reads_through_provider() -> None:
    drafter = LLMDrafter(
        provider=AnthropicProvider(
            model="claude-sonnet-4-6",
            client=_FakeAnthropicClient(_VALID_CARDS_JSON),
        )
    )
    assert drafter.model == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Default Drafter selection
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Per-card mode
# ---------------------------------------------------------------------------


def _multi_pair_bundle() -> Bundle[PGxFinding]:
    """Two findings, total of 4 (finding, drug) pairs — per_card mode
    should produce 4 distinct provider calls and 4 cards.
    """
    cyp = PGxFinding(
        gene="CYP2C19",
        diplotype="*1/*2",
        source_variants=(),
        phenotype="Intermediate Metabolizer",
        phenotype_source="test",
        affected_drugs=(
            DrugRec(
                drug="clopidogrel",
                recommendation="Avoid use.",
                cpic_guideline_id="CPIC-C",
                pmids=(1,),
                evidence_level="A",
            ),
            DrugRec(
                drug="voriconazole",
                recommendation="Choose alternative.",
                cpic_guideline_id="CPIC-V",
                pmids=(2,),
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
        phenotype_source="test",
        affected_drugs=(
            DrugRec(
                drug="amitriptyline",
                recommendation="Reduce dose.",
                cpic_guideline_id="CPIC-A",
                pmids=(3,),
                evidence_level="B",
            ),
            DrugRec(
                drug="codeine",
                recommendation="Use alternative.",
                cpic_guideline_id="CPIC-CO",
                pmids=(4,),
                evidence_level="A",
            ),
        ),
        confidence="high",
    )
    return Bundle(
        items=(cyp, cyp2d6),
        privacy_tier=PrivacyTier.PUBLIC,
        source="test",
    )


def _per_card_response_for(card: DraftedCard) -> str:
    """Build a one-card JSON response that the fake provider returns."""
    return json.dumps(
        {
            "cards": [
                {
                    "gene": card.gene,
                    "diplotype": card.diplotype,
                    "phenotype": card.phenotype,
                    "drug": card.drug,
                    "recommendation": card.recommendation,
                    "cited_pmids": list(card.cited_pmids),
                }
            ]
        }
    )


class _RoutingAnthropicClient:
    """Fake Anthropic client that returns one card per pair, routed
    by inspecting the user message content for the drug name.
    """

    def __init__(self, bundle: Bundle[PGxFinding]) -> None:
        self._by_drug = {
            d.drug: DraftedCard(
                gene=f.gene,
                diplotype=f.diplotype,
                phenotype=f.phenotype,
                drug=d.drug,
                recommendation=d.recommendation,
                cited_pmids=d.pmids,
            )
            for f in bundle.items
            for d in f.affected_drugs
        }
        self.messages = _RoutingMessagesEndpoint(self._by_drug)


class _RoutingMessagesEndpoint:
    def __init__(self, by_drug: dict[str, DraftedCard]) -> None:
        self._by_drug = by_drug
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _AnthropicMessage:
        self.calls.append(kwargs)
        # Find which drug this call is about (one-pair mini-bundle).
        user_msg = kwargs["messages"][0]["content"]
        matched = next(
            (d for d in self._by_drug if f'"{d}"' in user_msg), None
        )
        if matched is None:
            raise AssertionError(
                f"per-card call payload didn't reference any known drug: "
                f"{user_msg[:200]}"
            )
        card = self._by_drug[matched]
        return _AnthropicMessage(
            content=[_AnthropicBlock(text=_per_card_response_for(card))],
            usage=_AnthropicUsage(input_tokens=11, output_tokens=22),
        )


def test_per_card_makes_one_call_per_pair() -> None:
    bundle = _multi_pair_bundle()
    client = _RoutingAnthropicClient(bundle)
    drafter = LLMDrafter(
        provider=AnthropicProvider(client=client),
        mode="per_card",
        max_workers=4,
    )
    draft = drafter.draft(bundle)
    # 4 pairs total -> 4 API calls + 4 cards out.
    assert len(client.messages.calls) == 4
    assert len(draft.cards) == 4
    drugs = {c.drug for c in draft.cards}
    assert drugs == {"clopidogrel", "voriconazole", "amitriptyline", "codeine"}


def test_per_card_each_call_sees_only_one_finding() -> None:
    """A per-card call must not have visibility into other genes —
    that's the privacy + cross-gene-contamination guarantee.
    """
    bundle = _multi_pair_bundle()
    client = _RoutingAnthropicClient(bundle)
    drafter = LLMDrafter(
        provider=AnthropicProvider(client=client), mode="per_card"
    )
    drafter.draft(bundle)
    for call in client.messages.calls:
        user_msg = call["messages"][0]["content"]
        gene_mentions = sum(
            user_msg.count(f'"{g}"') for g in ("CYP2C19", "CYP2D6")
        )
        # At least one (this call's gene), but never both.
        assert 1 <= gene_mentions
        if "CYP2C19" in user_msg and "CYP2D6" in user_msg:
            raise AssertionError(
                "per-card payload mentions multiple genes — leakage!"
            )


def test_per_card_aggregates_usage() -> None:
    bundle = _multi_pair_bundle()
    client = _RoutingAnthropicClient(bundle)
    drafter = LLMDrafter(
        provider=AnthropicProvider(client=client), mode="per_card"
    )
    drafter.draft(bundle)
    assert drafter.last_response is not None
    # 4 calls × (11 in + 22 out) each.
    assert drafter.last_response.input_tokens == 4 * 11
    assert drafter.last_response.output_tokens == 4 * 22


def test_per_card_refuses_local_only() -> None:
    bundle = _multi_pair_bundle()
    bundle = Bundle(
        items=bundle.items,
        privacy_tier=PrivacyTier.LOCAL_ONLY,
        source=bundle.source,
    )
    drafter = LLMDrafter(
        provider=AnthropicProvider(client=_FakeAnthropicClient("{}")),
        mode="per_card",
    )
    with pytest.raises(PrivacyViolation):
        drafter.draft(bundle)


def test_per_card_handles_empty_bundle() -> None:
    bundle = Bundle(items=(), privacy_tier=PrivacyTier.PUBLIC, source="t")
    drafter = LLMDrafter(
        provider=AnthropicProvider(client=_FakeAnthropicClient("{}")),
        mode="per_card",
    )
    draft = drafter.draft(bundle)
    assert draft.cards == ()
    assert drafter.last_response is None


def test_per_card_filters_pairs_with_empty_cards_response() -> None:
    """The LLM may legitimately return `{"cards": []}` for some pairs
    (e.g. Unknown/Uncertain phenotype with no actionable rec). Per-card
    mode must filter those out, not crash.
    """
    bundle = _multi_pair_bundle()  # 4 pairs

    class _MixedResponses:
        """Fake client: emit a valid card for the first 3 pairs,
        an empty-cards response for the 4th (codeine).
        """

        def __init__(self) -> None:
            self.messages = self  # client.messages.create(...) pattern
            self.calls: list[dict[str, Any]] = []

        def create(self, **kwargs: Any) -> _AnthropicMessage:
            self.calls.append(kwargs)
            user_msg = kwargs["messages"][0]["content"]
            if '"codeine"' in user_msg:
                payload = json.dumps({"cards": []})
            else:
                drug = next(
                    d
                    for d in ("clopidogrel", "voriconazole", "amitriptyline")
                    if f'"{d}"' in user_msg
                )
                payload = json.dumps(
                    {
                        "cards": [
                            {
                                "gene": "CYP2C19" if drug in ("clopidogrel", "voriconazole") else "CYP2D6",
                                "diplotype": "*1/*2" if drug in ("clopidogrel", "voriconazole") else "*1/*3",
                                "phenotype": "Intermediate Metabolizer",
                                "drug": drug,
                                "recommendation": "ok",
                                "cited_pmids": [1],
                            }
                        ]
                    }
                )
            return _AnthropicMessage(
                content=[_AnthropicBlock(text=payload)],
                usage=_AnthropicUsage(input_tokens=10, output_tokens=10),
            )

    client = _MixedResponses()
    drafter = LLMDrafter(
        provider=AnthropicProvider(client=client), mode="per_card"
    )
    draft = drafter.draft(bundle)
    # 4 calls were made; 3 cards out, the codeine empty-response was filtered.
    assert len(client.calls) == 4
    assert len(draft.cards) == 3
    assert "codeine" not in {c.drug for c in draft.cards}
    # Usage still aggregated from all 4 calls.
    assert drafter.last_response is not None
    assert drafter.last_response.input_tokens == 4 * 10


def test_per_card_default_mode_is_batch() -> None:
    """Default mode must remain `batch` to preserve backward compatibility."""
    drafter = LLMDrafter(
        provider=AnthropicProvider(client=_FakeAnthropicClient("{}"))
    )
    assert drafter.mode == "batch"


# ---------------------------------------------------------------------------
# CPIC retriever integration — the Drafter injects retrieved context
# into the user message when a retriever is configured.
# ---------------------------------------------------------------------------


class _StubCPICRetriever:
    """Fake Retriever that returns a fixed snippet for every query."""

    def __init__(self, snippet_text: str = "fake CPIC snippet") -> None:
        self.snippet_text = snippet_text
        self.queries: list[str] = []

    def retrieve(self, query: str, *, k: int = 5):
        from pgx_digest.retriever import RetrievedItem

        self.queries.append(query)
        return [
            RetrievedItem(
                text=self.snippet_text,
                score=0.42,
                metadata={
                    "drug": "clopidogrel",
                    "phenotypes": {"CYP2C19": "Intermediate Metabolizer"},
                },
            )
        ]


def test_llm_drafter_injects_cpic_context_when_retriever_set() -> None:
    client = _FakeAnthropicClient(_VALID_CARDS_JSON)
    retr = _StubCPICRetriever(snippet_text="canonical CPIC text here")
    drafter = LLMDrafter(
        provider=AnthropicProvider(client=client),
        retriever=retr,
    )
    drafter.draft(_bundle())

    [call] = client.messages.calls
    user_msg = call["messages"][0]["content"]
    # CPIC retrieved snippet shows up in the prompt.
    assert "canonical CPIC text here" in user_msg
    # And the bundle JSON is still there.
    assert "Bundle (JSON):" in user_msg
    # Retriever was queried (at least once for the one CYP2C19/clopidogrel pair).
    assert retr.queries


def test_llm_drafter_omits_cpic_block_when_no_retriever() -> None:
    client = _FakeAnthropicClient(_VALID_CARDS_JSON)
    drafter = LLMDrafter(provider=AnthropicProvider(client=client))
    drafter.draft(_bundle())
    [call] = client.messages.calls
    user_msg = call["messages"][0]["content"]
    assert "Authoritative CPIC reference" not in user_msg


def test_select_drafter_picks_ollama_for_local_only() -> None:
    drafter = select_drafter(_bundle(tier=PrivacyTier.LOCAL_ONLY))
    assert isinstance(drafter, OllamaDrafter)


def test_select_drafter_picks_llm_for_public() -> None:
    drafter = select_drafter(_bundle(tier=PrivacyTier.PUBLIC))
    assert isinstance(drafter, LLMDrafter)
