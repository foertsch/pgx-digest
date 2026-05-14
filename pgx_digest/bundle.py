"""Typed bundles with mandatory provenance.

A Bundle is the typed evidence container that flows from a deterministic
source (PharmCAT) through the Ranker into the Drafter. Every row carries
identifiers that the Verifier can use for token-level containment checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, Literal, TypeVar

T = TypeVar("T")


class PrivacyTier(str, Enum):
    """Sensitivity tier of the data inside a Bundle.

    Drives Drafter selection: cloud-backed Drafters refuse to run on
    LOCAL_ONLY bundles.
    """

    PUBLIC = "public"
    PSEUDONYMIZED = "pseudonymized"
    LOCAL_ONLY = "local_only"


EvidenceLevel = Literal["A", "B", "C", "D"]


@dataclass(frozen=True)
class Variant:
    """One genotype call at one position. Provenance: rsid."""

    rsid: str
    chromosome: str
    position: int
    genotype: str
    reference_strand: Literal["+", "-"]


@dataclass(frozen=True)
class DrugRec:
    """One CPIC-guideline recommendation for one drug-gene pair."""

    drug: str
    recommendation: str
    cpic_guideline_id: str
    pmids: tuple[int, ...]
    evidence_level: EvidenceLevel


@dataclass(frozen=True)
class PGxFinding:
    """One actionable pharmacogenomic finding for one gene."""

    gene: str
    diplotype: str
    source_variants: tuple[Variant, ...]
    phenotype: str
    phenotype_source: str
    affected_drugs: tuple[DrugRec, ...]
    confidence: Literal["high", "medium", "low"]


@dataclass(frozen=True)
class Bundle(Generic[T]):
    """Typed evidence container with privacy and provenance metadata."""

    items: tuple[T, ...]
    privacy_tier: PrivacyTier
    source: str
    schema_version: str = "0.1.0"
    metadata: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.items)
