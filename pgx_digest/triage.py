"""Triage — deterministic routing of (finding, drug) pairs.

The Drafter is expensive (LLM API call per Bundle) and most cards on a
real patient report don't need narrative synthesis: a Normal Metabolizer
of TPMT with a one-line "no adjustment needed" recommendation can be
rendered by a template. Triage classifies each pair as:

- `llm`      — needs a Drafter call (genuine narrative needed)
- `template` — render deterministically; no API call
- `skip`     — no card emitted (no recommendation / not actionable)

This extends the project's deterministic core (Bundle → Ranker → ...)
one layer further. The LLM is the most expensive component; using it
only when deterministic rules genuinely fall short is the cleanest
scaling answer.

Conservative by default: when in doubt, route to LLM. False-negative
triage (templating a case that should have gone to LLM) is the only
patient-safety risk; false-positive triage just costs an API call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pgx_digest.bundle import DrugRec, PGxFinding
from pgx_digest.drafter import DraftedCard

# Re-exported here so callers don't have to reach into eval/.
SAFETY_FOOTER = "Discuss with your physician before any medication change."


Route = Literal["llm", "template", "skip"]


@dataclass(frozen=True)
class TriageDecision:
    """One routing decision plus the rule that produced it (for audit)."""

    route: Route
    reason: str


# Phrases that mark a recommendation as conditional/narrative-requiring.
# A "Normal Metabolizer" recommendation with any of these stays on the
# LLM path even if it's short — the conditional logic deserves real
# prose handling.
_CONDITIONAL_KEYWORDS: tuple[str, ...] = (
    "if ",
    "consider",
    "alternative",
    "avoid",
    "unless",
    "switch",
    "monitor",
    "caution",
)


# Phrases that signal a "not actionable" phenotype call from PharmCAT.
_NON_ACTIONABLE_PHENOTYPES: tuple[str, ...] = (
    "unknown",
    "no result",
    "indeterminate",
)


class Triage:
    """Rule-based router. Pure function; deterministic and unit-testable."""

    def __init__(self, *, max_template_length: int = 200) -> None:
        """
        max_template_length: any source recommendation longer than this
        is assumed too nuanced to template, even for Normal phenotypes.
        Default of 200 chars chosen to fit "Initiate therapy with
        recommended starting dose. No CPIC dose adjustment recommended."
        and similar short formulas, while excluding multi-clause
        recommendations.
        """
        self.max_template_length = max_template_length

    def classify(
        self, finding: PGxFinding, drug: DrugRec
    ) -> TriageDecision:
        """Decide whether (finding, drug) needs an LLM call."""
        # Rule 1: Level D = "No Recommendation" — nothing to write.
        if drug.evidence_level == "D":
            return TriageDecision("skip", "evidence level D — no CPIC recommendation")

        # Rule 2: Phenotype not actionable — skip.
        phen_lower = finding.phenotype.lower()
        for marker in _NON_ACTIONABLE_PHENOTYPES:
            if marker in phen_lower:
                return TriageDecision(
                    "skip",
                    f"phenotype {finding.phenotype!r} is not actionable",
                )

        # Rule 3: Normal Metabolizer + short, unconditional source rec → template.
        rec = drug.recommendation
        rec_lower = rec.lower()
        is_normal = "normal" in phen_lower
        is_short = len(rec) <= self.max_template_length
        has_conditionals = any(k in rec_lower for k in _CONDITIONAL_KEYWORDS)
        if is_normal and is_short and not has_conditionals:
            return TriageDecision(
                "template",
                "Normal phenotype with short, unconditional CPIC text",
            )

        # Default: LLM.
        return TriageDecision(
            "llm", "non-standard recommendation requires narrative synthesis"
        )


class TemplateDrafter:
    """Deterministic card synthesis. No API calls.

    The output mirrors the LLM Drafter's structure (same DraftedCard
    shape, same safety footer) so downstream stages (Verifier, judge,
    rule checks) can't tell the difference. The recommendation prose
    is a structured paraphrase of the source CPIC text.
    """

    def draft_card(
        self, finding: PGxFinding, drug: DrugRec
    ) -> DraftedCard:
        return DraftedCard(
            gene=finding.gene,
            diplotype=finding.diplotype,
            phenotype=finding.phenotype,
            drug=drug.drug,
            recommendation=(
                f"You are a {finding.phenotype} of {finding.gene}. "
                f"For {drug.drug}, the CPIC recommendation is: "
                f"{drug.recommendation} {SAFETY_FOOTER}"
            ),
            cited_pmids=drug.pmids,
        )
