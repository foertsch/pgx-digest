"""Verifier — typed containment check on a drafted output.

The Verifier enforces that every claim in the LLM output traces back
to a field in the Bundle. If a card mentions a gene, drug, phenotype,
or PMID that isn't in the source data, the draft is rejected.

The check is intentionally simple — pure dict lookups. Most of the
"verifier" complexity in token-level verification literature comes
from extracting claims from free-form prose. Here we punt that to the
Drafter by requiring structured JSON output, which makes verification
deterministic.

There is one prose-level rule: a card's `recommendation` text may
mention only its own gene. Cross-gene references in the prose are
rejected. This catches the highest-stakes prose error we observed in
eval (CYP2D6 cards naming CYP2C19 alternatives) without paying the
cost of full claim extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pgx_digest.bundle import Bundle, PGxFinding
from pgx_digest.drafter import Draft


@dataclass(frozen=True)
class VerificationFailure:
    card_index: int
    field: str
    value: str
    reason: str


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    failures: tuple[VerificationFailure, ...]

    def __bool__(self) -> bool:
        return self.passed


class Verifier:
    """Typed containment verifier."""

    def verify(
        self,
        draft: Draft,
        bundle: Bundle[PGxFinding],
    ) -> VerificationResult:
        by_gene: dict[str, PGxFinding] = {f.gene: f for f in bundle.items}
        all_bundle_genes = set(by_gene)
        failures: list[VerificationFailure] = []

        for i, card in enumerate(draft.cards):
            finding = by_gene.get(card.gene)

            if finding is None:
                failures.append(
                    VerificationFailure(
                        card_index=i,
                        field="gene",
                        value=card.gene,
                        reason="gene not present in Bundle",
                    )
                )
                continue

            for other in all_bundle_genes - {card.gene}:
                # Word-boundary match — gene symbols always contain
                # digits, so the boundary is unambiguous and we avoid
                # accidental hits on partial substrings.
                if re.search(
                    rf"\b{re.escape(other)}\b", card.recommendation
                ):
                    failures.append(
                        VerificationFailure(
                            card_index=i,
                            field="recommendation",
                            value=other,
                            reason=(
                                f"cross-gene reference: recommendation "
                                f"for {card.gene} mentions other Bundle "
                                f"gene {other}"
                            ),
                        )
                    )

            if card.diplotype != finding.diplotype:
                failures.append(
                    VerificationFailure(
                        card_index=i,
                        field="diplotype",
                        value=card.diplotype,
                        reason=(
                            f"diplotype mismatch (Bundle has "
                            f"{finding.diplotype!r})"
                        ),
                    )
                )

            if card.phenotype != finding.phenotype:
                failures.append(
                    VerificationFailure(
                        card_index=i,
                        field="phenotype",
                        value=card.phenotype,
                        reason=(
                            f"phenotype mismatch (Bundle has "
                            f"{finding.phenotype!r})"
                        ),
                    )
                )

            drug_lookup = {d.drug: d for d in finding.affected_drugs}
            drug_rec = drug_lookup.get(card.drug)
            if drug_rec is None:
                failures.append(
                    VerificationFailure(
                        card_index=i,
                        field="drug",
                        value=card.drug,
                        reason=(
                            f"drug not in affected_drugs for gene "
                            f"{card.gene}"
                        ),
                    )
                )
                continue

            bundle_pmids = set(drug_rec.pmids)
            for pmid in card.cited_pmids:
                if pmid not in bundle_pmids:
                    failures.append(
                        VerificationFailure(
                            card_index=i,
                            field="cited_pmids",
                            value=str(pmid),
                            reason=(
                                f"PMID {pmid} not in DrugRec.pmids for "
                                f"{card.gene}/{card.drug}"
                            ),
                        )
                    )

        return VerificationResult(
            passed=len(failures) == 0,
            failures=tuple(failures),
        )
