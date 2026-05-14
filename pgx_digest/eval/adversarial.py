"""Adversarial drafter for the verifier-on/off ablation.

The real Drafter (structured JSON + Haiku 4.5) rarely hallucinates — the
schema rails it in. That makes a naive "Verifier on vs off" run boring:
zero failures detected, zero failures present. The honest version of the
ablation asks: *if* the Drafter did invent something, would the Verifier
catch it? `AdversarialDrafter` answers that with a concrete number by
seeding known corruptions.

It is **not** a real Drafter. It builds a faithful card-per-drug Draft
deterministically from the Bundle, then perturbs one field per card at
a configured rate. This costs zero tokens and is fully reproducible
(seeded RNG).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from enum import Enum

from pgx_digest.bundle import Bundle, PGxFinding
from pgx_digest.drafter import Draft, Drafter, DraftedCard
from pgx_digest.eval.cases import SAFETY_FOOTER


class AdversarialMode(str, Enum):
    """What field the adversary mutates."""

    FAITHFUL = "faithful"  # no mutation; baseline
    SWAP_PMID = "swap_pmid"
    SWAP_DRUG = "swap_drug"
    SWAP_PHENOTYPE = "swap_phenotype"
    SWAP_DIPLOTYPE = "swap_diplotype"


# Visible-junk values used when we want to inject a clearly-invalid mutation.
_FAKE_PMID = 99999999
_FAKE_DRUG = "unicornium"
_FAKE_PHENOTYPE = "Ultrarapid Metabolizer of unicornium"
_FAKE_DIPLOTYPE = "*999/*999"

def make_faithful_draft(bundle: Bundle[PGxFinding]) -> Draft:
    """Build a Draft that mirrors the Bundle exactly. Zero LLM calls.

    Used as both the baseline for adversarial perturbation and as a
    deterministic Drafter in unit tests.
    """
    cards: list[DraftedCard] = []
    for finding in bundle.items:
        for drug in finding.affected_drugs:
            cards.append(
                DraftedCard(
                    gene=finding.gene,
                    diplotype=finding.diplotype,
                    phenotype=finding.phenotype,
                    drug=drug.drug,
                    recommendation=(
                        f"For {drug.drug}, the CPIC recommendation is: "
                        f"{drug.recommendation} {SAFETY_FOOTER}"
                    ),
                    cited_pmids=drug.pmids,
                )
            )
    return Draft(cards=tuple(cards), raw_text="<faithful-stub>")


def _perturb(
    card: DraftedCard,
    mode: AdversarialMode,
) -> DraftedCard:
    """Return a corrupted copy of `card` according to `mode`."""
    if mode == AdversarialMode.SWAP_PMID:
        # Inject a known-fake PMID so Verifier has something to catch.
        return replace(card, cited_pmids=card.cited_pmids + (_FAKE_PMID,))
    if mode == AdversarialMode.SWAP_DRUG:
        return replace(card, drug=_FAKE_DRUG)
    if mode == AdversarialMode.SWAP_PHENOTYPE:
        return replace(card, phenotype=_FAKE_PHENOTYPE)
    if mode == AdversarialMode.SWAP_DIPLOTYPE:
        return replace(card, diplotype=_FAKE_DIPLOTYPE)
    return card


@dataclass(frozen=True)
class AdversarialDrafter(Drafter):
    """Deterministic synthetic Drafter that injects controlled hallucinations.

    Builds a faithful Draft from the Bundle, then perturbs each card with
    probability `rate` according to `mode`. Reproducible via `seed`.
    """

    mode: AdversarialMode = AdversarialMode.FAITHFUL
    rate: float = 1.0
    seed: int = 0

    def draft(self, bundle: Bundle[PGxFinding]) -> Draft:
        base = make_faithful_draft(bundle)
        if self.mode == AdversarialMode.FAITHFUL or self.rate <= 0:
            return base

        rng = random.Random(self.seed)
        new_cards: list[DraftedCard] = []
        for c in base.cards:
            if rng.random() < self.rate:
                new_cards.append(_perturb(c, self.mode))
            else:
                new_cards.append(c)
        return Draft(cards=tuple(new_cards), raw_text=base.raw_text)
