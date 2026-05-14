# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "anthropic>=0.40",
#   "python-dotenv>=1.0",
# ]
# ///
"""End-to-end demo: PharmCAT JSON -> Verified narrative.

Loads the CYP2C19 minimal fixture, runs the full pipeline (Ranker ->
Drafter -> Verifier), prints the generated cards and verification
status.

Costs approximately $0.001 in Claude Haiku 4.5 API usage per run.

Run from the repo root:
    uv run examples/run_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).parent.parent
load_dotenv(REPO_ROOT / ".env")

sys.path.insert(0, str(REPO_ROOT))

from pgx_digest import parse_pharmcat_json, run

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "pharmcat_cyp2c19_minimal.json"


def main() -> int:
    bundle = parse_pharmcat_json(FIXTURE)

    print(f"Parsed Bundle from {FIXTURE.name}:")
    for finding in bundle.items:
        print(
            f"  Gene: {finding.gene} | "
            f"Diplotype: {finding.diplotype} | "
            f"Phenotype: {finding.phenotype}"
        )
        print(f"  Drugs: {[d.drug for d in finding.affected_drugs]}")
    print()

    print("Running pipeline (Ranker -> Drafter -> Verifier)...")
    result = run(bundle)
    print()

    if result.verification.passed:
        print("Verification: PASSED")
    else:
        print("Verification: FAILED")
        for failure in result.verification.failures:
            print(
                f"  - card {failure.card_index} "
                f"{failure.field}: {failure.reason}"
            )

    print()
    print("=" * 70)
    print("Drafted cards")
    print("=" * 70)
    for i, card in enumerate(result.draft.cards):
        print(f"\n[Card {i}] {card.gene} {card.diplotype} -> {card.phenotype}")
        print(f"  Drug: {card.drug}")
        print(f"  Recommendation: {card.recommendation}")
        print(f"  Cited PMIDs: {list(card.cited_pmids)}")

    return 0 if result.verification.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
