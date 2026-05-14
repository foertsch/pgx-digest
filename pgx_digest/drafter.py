"""Drafter — turns a typed Bundle into a structured narrative draft.

Privacy-tier aware: the cloud-backed ClaudeDrafter refuses to run on
LOCAL_ONLY bundles. Local Ollama support is stubbed for future use.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import anthropic

from pgx_digest.bundle import Bundle, PGxFinding, PrivacyTier


class PrivacyViolation(RuntimeError):
    """Raised when a cloud Drafter is invoked on LOCAL_ONLY data."""


@dataclass(frozen=True)
class DraftedCard:
    """One claim card produced by the Drafter. Verifier checks each field."""

    gene: str
    diplotype: str
    phenotype: str
    drug: str
    recommendation: str
    cited_pmids: tuple[int, ...]


@dataclass(frozen=True)
class Draft:
    cards: tuple[DraftedCard, ...]
    raw_text: str


SYSTEM_PROMPT = """\
You write patient-facing pharmacogenomic summaries from a typed evidence
bundle.

Rules (enforced by a downstream verifier — violations cause the draft to
be rejected):

1. Mention only genes, star alleles (diplotypes), phenotypes, drugs, and
   PMIDs that appear in the input Bundle.
2. Match every claim to a field on the source PGxFinding or DrugRec. Do
   not extrapolate beyond the CPIC guideline text.
3. Write in plain, direct English. End each recommendation with: "Discuss
   with your physician before any medication change."

Output: JSON conforming to the provided schema. One card per
(gene, drug) finding.
"""


OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "cards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "gene": {"type": "string"},
                    "diplotype": {"type": "string"},
                    "phenotype": {"type": "string"},
                    "drug": {"type": "string"},
                    "recommendation": {"type": "string"},
                    "cited_pmids": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                },
                "required": [
                    "gene",
                    "diplotype",
                    "phenotype",
                    "drug",
                    "recommendation",
                    "cited_pmids",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["cards"],
    "additionalProperties": False,
}


class Drafter(ABC):
    """Abstract Drafter — picks a model and runs structured generation."""

    @abstractmethod
    def draft(self, bundle: Bundle[PGxFinding]) -> Draft: ...


class ClaudeDrafter(Drafter):
    """Cloud Drafter, defaulting to Claude Haiku 4.5.

    Refuses to run on LOCAL_ONLY bundles. The system prompt carries a
    cache_control marker so it will be cached automatically once it
    grows past Haiku's ~4K-token minimum cacheable prefix.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        max_tokens: int = 2048,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def draft(self, bundle: Bundle[PGxFinding]) -> Draft:
        if bundle.privacy_tier == PrivacyTier.LOCAL_ONLY:
            raise PrivacyViolation(
                f"ClaudeDrafter refuses to run on LOCAL_ONLY bundle "
                f"(source={bundle.source!r}). Use OllamaDrafter for "
                f"personal genome data."
            )

        bundle_json = json.dumps(
            [asdict(finding) for finding in bundle.items],
            sort_keys=True,
            default=str,
        )

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            output_config={
                "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}
            },
            messages=[
                {
                    "role": "user",
                    "content": f"Bundle (JSON):\n{bundle_json}",
                }
            ],
        )

        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)

        cards = tuple(
            DraftedCard(
                gene=c["gene"],
                diplotype=c["diplotype"],
                phenotype=c["phenotype"],
                drug=c["drug"],
                recommendation=c["recommendation"],
                cited_pmids=tuple(c["cited_pmids"]),
            )
            for c in data["cards"]
        )
        return Draft(cards=cards, raw_text=text)


class OllamaDrafter(Drafter):
    """Local Drafter backed by Ollama. Not yet implemented."""

    def __init__(self, model: str = "qwen2.5:7b-instruct") -> None:
        self.model = model

    def draft(self, bundle: Bundle[PGxFinding]) -> Draft:
        raise NotImplementedError(
            "OllamaDrafter is pending implementation. Use ClaudeDrafter "
            "for PUBLIC bundles."
        )


def select_drafter(bundle: Bundle[PGxFinding]) -> Drafter:
    """Default privacy-tier-aware Drafter selection."""
    if bundle.privacy_tier == PrivacyTier.LOCAL_ONLY:
        return OllamaDrafter()
    return ClaudeDrafter()
