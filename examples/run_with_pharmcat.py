# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "anthropic>=0.40",
#   "python-dotenv>=1.0",
# ]
# ///
"""End-to-end demo from a VCF, via PharmCAT, to a verified narrative.

Pipeline:
    VCF -> docker pgkb/pharmcat -> report.json -> Bundle -> Drafter -> Verifier

Requires Docker running. First invocation pulls the ``pgkb/pharmcat``
image (~1 GB). Subsequent runs reuse the image.

Usage:
    uv run examples/run_with_pharmcat.py path/to/my.vcf
    uv run examples/run_with_pharmcat.py path/to/my.vcf --output ./pcat_out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).parent.parent
load_dotenv(REPO_ROOT / ".env")

sys.path.insert(0, str(REPO_ROOT))

from pgx_digest import (
    DockerUnavailable,
    PharmCATRunError,
    docker_available,
    run as run_pipeline,
    vcf_to_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vcf", type=Path, help="Path to the VCF (or .vcf.gz)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for PharmCAT intermediate outputs. "
        "Defaults to the VCF's parent directory.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="PharmCAT subprocess timeout in seconds (default: 1800).",
    )
    args = parser.parse_args()

    if not docker_available():
        print(
            "Docker is not available. Start Docker Desktop and try again.",
            file=sys.stderr,
        )
        return 2

    print(f"Running PharmCAT on {args.vcf} ...")
    print("(first run pulls ~1 GB image — be patient)")
    try:
        bundle = vcf_to_bundle(
            args.vcf, output_dir=args.output, timeout=args.timeout
        )
    except DockerUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except PharmCATRunError as exc:
        print(f"PharmCAT failed:\n{exc}", file=sys.stderr)
        return 3

    print(f"\nParsed {len(bundle)} gene finding(s) from PharmCAT report.")
    for finding in bundle.items:
        print(
            f"  {finding.gene} | {finding.diplotype} | "
            f"{finding.phenotype} | confidence={finding.confidence}"
        )

    print("\nRunning narrative pipeline (Ranker -> Drafter -> Verifier)...")
    result = run_pipeline(bundle)

    print()
    print("Verification:", "PASSED" if result.verification.passed else "FAILED")
    if not result.verification.passed:
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
