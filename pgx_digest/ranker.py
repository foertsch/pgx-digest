"""Ranker — deterministic ordering of PGxFinding rows.

No LLM. CPIC Level A first (lowest priority value sorts first), then
ties broken by gene name. Most ranking logic should never need an LLM.
"""

from __future__ import annotations

from pgx_digest.bundle import Bundle, PGxFinding

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
