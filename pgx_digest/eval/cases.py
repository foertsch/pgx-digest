"""Eval cases: typed assertions against drafted cards.

Each case lives as one JSON object per line in `tests/eval_cases.jsonl`.
Cases are intentionally substring-style (in the spirit of promptsmith's
`must_contain` / `must_not_contain`) — but the haystack is the typed
fields of `DraftedCard`, not free-form text.

Schema (one line of `tests/eval_cases.jsonl`):

    {
      "id": "multigene_01_cyp2c19_pm",
      "fixture": "pharmcat_multigene.json",
      "expected_genes": ["CYP2C19", "CYP2D6"],
      "expected_drugs": ["clopidogrel", "voriconazole"],
      "expected_phenotypes": {"CYP2C19": "Poor Metabolizer"},
      "unexpected_drugs": [],
      "must_include_footer": true,
      "notes": "free text"
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pgx_digest.drafter import Draft

# The Drafter system prompt asks every recommendation to end with this
# phrase verbatim. We assert on it as part of the rules tier.
SAFETY_FOOTER = "Discuss with your physician before any medication change."


@dataclass(frozen=True)
class EvalCase:
    """One eval case: a fixture + the assertions to run against its draft."""

    id: str
    fixture: str
    expected_genes: tuple[str, ...] = ()
    expected_drugs: tuple[str, ...] = ()
    expected_phenotypes: dict[str, str] = field(default_factory=dict)
    unexpected_drugs: tuple[str, ...] = ()
    must_include_footer: bool = True
    notes: str = ""


@dataclass(frozen=True)
class RuleFailure:
    rule: str
    detail: str


def load_cases(path: Path) -> tuple[EvalCase, ...]:
    """Load eval cases from a JSONL file. Skips blank lines."""
    cases: list[EvalCase] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        raw: dict[str, Any] = json.loads(line)
        cases.append(
            EvalCase(
                id=raw["id"],
                fixture=raw["fixture"],
                expected_genes=tuple(raw.get("expected_genes") or ()),
                expected_drugs=tuple(raw.get("expected_drugs") or ()),
                expected_phenotypes=dict(raw.get("expected_phenotypes") or {}),
                unexpected_drugs=tuple(raw.get("unexpected_drugs") or ()),
                must_include_footer=bool(raw.get("must_include_footer", True)),
                notes=str(raw.get("notes", "")),
            )
        )
    return tuple(cases)


def check_rules(case: EvalCase, draft: Draft) -> tuple[RuleFailure, ...]:
    """Apply rule-based assertions to a Draft.

    These are deliberately structural — gene/drug presence, expected
    phenotype, safety footer, and absence of named-bad-actors. The
    Verifier handles token-level containment; this layer captures the
    case-author's intent (this case is *about* CYP2C19 / clopidogrel).
    """
    failures: list[RuleFailure] = []

    cards_by_gene: dict[str, list[str]] = {}
    for c in draft.cards:
        cards_by_gene.setdefault(c.gene, []).append(c.drug)

    for gene in case.expected_genes:
        if gene not in cards_by_gene:
            failures.append(
                RuleFailure(
                    rule="expected_gene_missing",
                    detail=(
                        f"gene {gene!r} expected but not present in any card"
                    ),
                )
            )

    drugs_in_draft = {c.drug for c in draft.cards}
    for drug in case.expected_drugs:
        if drug not in drugs_in_draft:
            failures.append(
                RuleFailure(
                    rule="expected_drug_missing",
                    detail=f"drug {drug!r} expected but not covered by any card",
                )
            )

    for drug in case.unexpected_drugs:
        if drug in drugs_in_draft:
            failures.append(
                RuleFailure(
                    rule="unexpected_drug_present",
                    detail=f"drug {drug!r} should not appear, but did",
                )
            )

    for gene, expected_phen in case.expected_phenotypes.items():
        # Use the first card for this gene; all cards for one gene share
        # phenotype since they come from the same PGxFinding.
        gene_cards = [c for c in draft.cards if c.gene == gene]
        if not gene_cards:
            continue  # expected_gene_missing already covers this
        actual = gene_cards[0].phenotype
        if actual != expected_phen:
            failures.append(
                RuleFailure(
                    rule="phenotype_mismatch",
                    detail=(
                        f"{gene}: expected phenotype {expected_phen!r}, "
                        f"got {actual!r}"
                    ),
                )
            )

    if case.must_include_footer:
        for i, c in enumerate(draft.cards):
            if SAFETY_FOOTER not in c.recommendation:
                failures.append(
                    RuleFailure(
                        rule="missing_safety_footer",
                        detail=(
                            f"card {i} ({c.gene}/{c.drug}) recommendation "
                            f"does not end with the safety footer"
                        ),
                    )
                )

    return tuple(failures)
