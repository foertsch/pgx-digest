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


def test_select_drafter_picks_ollama_for_local_only() -> None:
    drafter = select_drafter(_bundle(tier=PrivacyTier.LOCAL_ONLY))
    assert isinstance(drafter, OllamaDrafter)


def test_select_drafter_picks_llm_for_public() -> None:
    drafter = select_drafter(_bundle(tier=PrivacyTier.PUBLIC))
    assert isinstance(drafter, LLMDrafter)
