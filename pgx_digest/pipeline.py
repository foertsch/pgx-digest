"""End-to-end pipeline orchestration.

Rank -> Draft -> Verify, with retry-on-verification-failure.
"""

from __future__ import annotations

from dataclasses import dataclass

from pgx_digest.bundle import Bundle, PGxFinding
from pgx_digest.drafter import Draft, select_drafter
from pgx_digest.ranker import rank
from pgx_digest.verifier import VerificationResult, Verifier


@dataclass(frozen=True)
class PipelineResult:
    bundle: Bundle[PGxFinding]
    draft: Draft
    verification: VerificationResult


def run(
    bundle: Bundle[PGxFinding],
    *,
    max_retries: int = 1,
) -> PipelineResult:
    """Run the full Rank -> Draft -> Verify pipeline.

    On verification failure, retries up to `max_retries` times. Retry
    feedback (passing failure reasons back into the Drafter) is stubbed
    — for now retries re-draft from scratch.
    """
    ranked = rank(bundle)
    drafter = select_drafter(ranked)
    verifier = Verifier()

    last_draft: Draft | None = None
    last_result: VerificationResult | None = None
    for _ in range(max_retries + 1):
        draft = drafter.draft(ranked)
        result = verifier.verify(draft, ranked)
        last_draft = draft
        last_result = result
        if result:
            break

    assert last_draft is not None
    assert last_result is not None
    return PipelineResult(
        bundle=ranked,
        draft=last_draft,
        verification=last_result,
    )
