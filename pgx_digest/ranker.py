"""Rankers — produce an ordered Bundle from an unordered one.

Two variants:

- `rank()` — deterministic, no LLM. CPIC Level A first, ties broken by
  gene name. This is the default; most ranking decisions in this domain
  do not need an LLM.
- `LLMRanker` — calls Claude to reorder findings by clinical importance.
  Used in the ablation harness as a baseline for "would an LLM ranker
  beat the deterministic one?" The expected answer is "no, and that's
  the point."
"""

from __future__ import annotations

import json
from dataclasses import asdict

import anthropic

from pgx_digest.bundle import Bundle, PGxFinding, PrivacyTier
from pgx_digest.drafter import PrivacyViolation

_LEVEL_PRIORITY = {"A": 0, "B": 1, "C": 2, "D": 3}


def rank(bundle: Bundle[PGxFinding]) -> Bundle[PGxFinding]:
    """Return a new Bundle with items reordered by CPIC evidence level."""

    def sort_key(finding: PGxFinding) -> tuple[int, str]:
        best_level = min(
            (
                _LEVEL_PRIORITY[d.evidence_level]
                for d in finding.affected_drugs
            ),
            default=99,
        )
        return (best_level, finding.gene)

    return Bundle(
        items=tuple(sorted(bundle.items, key=sort_key)),
        privacy_tier=bundle.privacy_tier,
        source=bundle.source,
        schema_version=bundle.schema_version,
        metadata=bundle.metadata,
    )


_RANKER_SYSTEM_PROMPT = """\
You order pharmacogenomic findings by clinical importance for a
patient-facing summary. Higher-strength CPIC evidence and stronger
phenotypes (e.g. Poor Metabolizer) should come first; Normal Metabolizer
findings with no actionable recommendations should come last.

Return ONLY a JSON object with one key "order": a list of gene symbols
in the desired display order. Every gene in the input must appear
exactly once.
"""


_RANKER_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "order": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["order"],
    "additionalProperties": False,
}


class LLMRanker:
    """LLM-driven ranker. Ablation baseline only — prefer `rank()`.

    Refuses LOCAL_ONLY bundles for the same reason ClaudeDrafter does.
    Falls back to deterministic order if the LLM returns an invalid
    permutation (drops or duplicates a gene).
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        max_tokens: int = 512,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def rank(self, bundle: Bundle[PGxFinding]) -> Bundle[PGxFinding]:
        if bundle.privacy_tier == PrivacyTier.LOCAL_ONLY:
            raise PrivacyViolation(
                f"LLMRanker refuses to run on LOCAL_ONLY bundle "
                f"(source={bundle.source!r})."
            )

        if len(bundle.items) <= 1:
            return bundle

        findings_payload = json.dumps(
            [asdict(f) for f in bundle.items],
            sort_keys=True,
            default=str,
        )

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": _RANKER_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": _RANKER_OUTPUT_SCHEMA,
                }
            },
            messages=[
                {
                    "role": "user",
                    "content": f"Findings (JSON):\n{findings_payload}",
                }
            ],
        )

        text = next(b.text for b in response.content if b.type == "text")
        order = json.loads(text).get("order") or []
        by_gene = {f.gene: f for f in bundle.items}

        if sorted(order) != sorted(by_gene):
            return rank(bundle)

        return Bundle(
            items=tuple(by_gene[g] for g in order),
            privacy_tier=bundle.privacy_tier,
            source=bundle.source,
            schema_version=bundle.schema_version,
            metadata=bundle.metadata,
        )
