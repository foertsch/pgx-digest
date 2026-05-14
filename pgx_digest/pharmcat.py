"""PharmCAT JSON parser.

PharmCAT (https://pharmcat.clinpgx.org/) emits a JSON Reporter report with
diplotype calls and CPIC guideline matches. This module converts that
JSON into a typed Bundle[PGxFinding].

Tested against PharmCAT v3.x Reporter JSON. Top-level shape:

    {
      "title": str,
      "timestamp": str,
      "pharmcatVersion": str,
      "dataVersion": str,
      "genes": { gene_symbol: GeneCall, ... },
      "drugs": { "CPIC Guideline Annotation": { drug: DrugAnnotation }, ... },
      ...
    }

MVP scope: CPIC annotations only (DPWG / FDA Label / FDA PGx skipped).
For each (drug, gene), we keep the single strongest-classification
annotation — PharmCAT may emit multiple annotations per drug-gene pair
for different clinical populations; we collapse them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pgx_digest.bundle import (
    Bundle,
    DrugRec,
    EvidenceLevel,
    PGxFinding,
    PrivacyTier,
    Variant,
)


# PharmCAT recommendation strength -> CPIC-style evidence level
_CLASSIFICATION_TO_LEVEL: dict[str, EvidenceLevel] = {
    "Strong": "A",
    "Moderate": "B",
    "Optional": "C",
    "No Recommendation": "D",
}

# Used for deterministic strongest-pick when multiple annotations apply
_STRENGTH_PRIORITY = {"Strong": 0, "Moderate": 1, "Optional": 2}


def parse_pharmcat_json(
    path: Path,
    *,
    privacy_tier: PrivacyTier = PrivacyTier.PUBLIC,
) -> Bundle[PGxFinding]:
    """Load a PharmCAT JSON report into a typed Bundle."""
    data = json.loads(Path(path).read_text())

    findings: list[PGxFinding] = []
    for gene_symbol, gene_data in data.get("genes", {}).items():
        finding = _gene_to_finding(gene_symbol, gene_data, data)
        if finding is not None:
            findings.append(finding)

    return Bundle(
        items=tuple(findings),
        privacy_tier=privacy_tier,
        source=str(Path(path).name),
        metadata={
            "pharmcat_version": data.get("pharmcatVersion", "unknown"),
            "data_version": data.get("dataVersion", "unknown"),
            "title": data.get("title", ""),
        },
    )


def _gene_to_finding(
    gene_symbol: str,
    gene_data: dict[str, Any],
    full_report: dict[str, Any],
) -> PGxFinding | None:
    """Build one PGxFinding from a gene entry. Returns None if uncalled."""
    rec_diplotypes = gene_data.get("recommendationDiplotypes") or []
    if not rec_diplotypes:
        return None

    primary = rec_diplotypes[0]
    diplotype = primary.get("label", "")
    phenotypes = primary.get("phenotypes") or []
    phenotype = phenotypes[0] if phenotypes else "Unknown"

    effectively_phased = bool(gene_data.get("effectivelyPhased", False))
    match_score = int(primary.get("matchScore", 0) or 0)
    if effectively_phased and match_score >= 5:
        confidence: str = "high"
    elif effectively_phased or match_score >= 3:
        confidence = "medium"
    else:
        confidence = "low"

    variants = tuple(
        Variant(
            rsid=v.get("dbSnpId") or f"{v.get('chromosome')}:{v.get('position')}",
            chromosome=str(v.get("chromosome", "")),
            position=int(v.get("position", 0)),
            genotype=str(v.get("call", "./.")).replace("|", "/"),
            reference_strand="+",
        )
        for v in (gene_data.get("variants") or [])
    )

    drugs = tuple(_drugs_for_gene(gene_symbol, full_report))

    return PGxFinding(
        gene=gene_symbol,
        diplotype=diplotype,
        source_variants=variants,
        phenotype=phenotype,
        phenotype_source=(
            f"PharmCAT {full_report.get('pharmcatVersion', 'unknown')}"
        ),
        affected_drugs=drugs,
        confidence=confidence,  # type: ignore[arg-type]
    )


def _drugs_for_gene(
    gene_symbol: str,
    full_report: dict[str, Any],
) -> list[DrugRec]:
    """Extract one DrugRec per drug whose annotations target the gene.

    When multiple annotations apply (different clinical populations),
    keeps the one with the strongest CPIC classification.
    """
    cpic = (full_report.get("drugs") or {}).get(
        "CPIC Guideline Annotation", {}
    )

    drug_recs: list[DrugRec] = []
    for drug_name, drug in cpic.items():
        pmids = tuple(
            int(c["pmid"])
            for c in (drug.get("citations") or [])
            if c.get("pmid")
        )

        best_annotation: dict[str, Any] | None = None
        best_strength = 99
        best_guideline_id = ""

        for guideline in drug.get("guidelines") or []:
            gid = str(guideline.get("id", ""))
            for ann in guideline.get("annotations") or []:
                if gene_symbol not in (ann.get("phenotypes") or {}):
                    continue
                strength = _STRENGTH_PRIORITY.get(
                    ann.get("classification", ""), 99
                )
                if strength < best_strength:
                    best_annotation = ann
                    best_strength = strength
                    best_guideline_id = gid

        if best_annotation is None:
            continue

        recommendation = (best_annotation.get("drugRecommendation") or "").strip()
        if not recommendation:
            continue

        classification = best_annotation.get("classification", "")
        evidence_level: EvidenceLevel = _CLASSIFICATION_TO_LEVEL.get(
            classification, "D"
        )

        drug_recs.append(
            DrugRec(
                drug=drug_name,
                recommendation=recommendation,
                cpic_guideline_id=best_guideline_id,
                pmids=pmids,
                evidence_level=evidence_level,
            )
        )

    return drug_recs
