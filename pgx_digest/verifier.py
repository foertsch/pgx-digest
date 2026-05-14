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
mention another Bundle gene ONLY IF that gene also appears in the
source CPIC recommendation for the same (gene, drug) pair. Legitimate
CPIC cross-references (e.g. CACNA1S anesthetic recommendations
literally say "Based on RYR1 status...") pass through; LLM-invented
cross-gene references (e.g. a CYP2D6 card's prose introducing a
gene the source text never mentioned) are rejected. This is the
narrowest definition of a "cross-gene hallucination" — the source
text is the ground truth for which other genes may appear.

Optional citation grounding: when constructed with a `retriever`
(typically a `PubMedRetriever` indexed over the Bundle's cited PMIDs),
each card's `cited_pmids` is additionally checked at semantic level
— is the card's recommendation similar enough to the abstract? A
similarity below `citation_grounding_threshold` is flagged. This
turns PMID citations from "the integer is in the Bundle" into "the
prose is plausibly supported by what the cited paper actually says."
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

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
    """Typed containment verifier with optional citation grounding."""

    def __init__(
        self,
        retriever: Any | None = None,
        citation_grounding_threshold: float = 0.35,
    ) -> None:
        """
        retriever: when provided, each card's cited PMIDs are
            additionally checked at semantic-similarity level against
            their abstract (intended use: a PubMedRetriever indexed
            over the Bundle's PMIDs).
        citation_grounding_threshold: cosine similarity below which a
            (card, cited PMID) pair is flagged as ungrounded. The
            BGE-small embeddings used by `LocalEmbedder` typically score
            unrelated medical text around 0.2-0.3 and related text
            around 0.4-0.7; 0.35 is a conservative middle.
        """
        self.retriever = retriever
        self.citation_grounding_threshold = citation_grounding_threshold

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

            # Look up the source CPIC text for this (gene, drug) so we
            # can distinguish "LLM invented a cross-gene reference"
            # from "CPIC source text legitimately mentions another gene"
            # (e.g. CACNA1S anesthetic recs literally say "Based on
            # RYR1 status..."). If the drug isn't in the finding (a
            # separate failure handled below), we use an empty source.
            source_text = ""
            for d in finding.affected_drugs:
                if d.drug == card.drug:
                    source_text = d.recommendation
                    break

            for other in all_bundle_genes - {card.gene}:
                # Word-boundary match — gene symbols always contain
                # digits, so the boundary is unambiguous and we avoid
                # accidental hits on partial substrings.
                pattern = rf"\b{re.escape(other)}\b"
                in_prose = re.search(pattern, card.recommendation)
                in_source = re.search(pattern, source_text)
                if in_prose and not in_source:
                    failures.append(
                        VerificationFailure(
                            card_index=i,
                            field="recommendation",
                            value=other,
                            reason=(
                                f"cross-gene hallucination: "
                                f"recommendation for {card.gene} "
                                f"mentions {other} but the source CPIC "
                                f"text does not"
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
                    continue
                # Optional semantic-grounding check against the abstract.
                if self.retriever is None:
                    continue
                abstract_item = self.retriever.get_by_metadata("pmid", pmid)
                if abstract_item is None:
                    # PMID not in the retriever's index (e.g. PubMed
                    # eutils returned no abstract for a retracted /
                    # missing record). Don't fail — record this as a
                    # soft signal via a no-op so the rest of the
                    # verification proceeds.
                    continue
                sim = self.retriever.similarity(
                    card.recommendation, abstract_item
                )
                if sim < self.citation_grounding_threshold:
                    failures.append(
                        VerificationFailure(
                            card_index=i,
                            field="cited_pmids",
                            value=str(pmid),
                            reason=(
                                f"citation grounding similarity "
                                f"{sim:.3f} < threshold "
                                f"{self.citation_grounding_threshold} "
                                f"for PMID {pmid}"
                            ),
                        )
                    )

        return VerificationResult(
            passed=len(failures) == 0,
            failures=tuple(failures),
        )
