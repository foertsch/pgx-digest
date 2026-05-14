# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "anthropic>=0.40",
#   "fastembed>=0.4",
#   "google-genai>=1.0",
#   "joblib>=1.3",
#   "python-dotenv>=1.0",
#   "scikit-learn>=1.5",
# ]
# ///
"""Bootstrap a Triage training dataset from the rule-based Triage on all
fixtures.

Walks every PharmCAT report JSON under `tests/fixtures/`, runs the
existing rule-based `Triage` on every `(PGxFinding, DrugRec)` pair, and
writes the resulting labeled examples to `triage_data/training.jsonl`.

This is the "bootstrap" labeling step — labels come from the rules we
already trust. The next phase (out of scope here) is reviewing the
JSONL and overriding obviously-wrong labels by hand, which is how
production rule-bootstrapped datasets typically improve.

Run from the repo root:
    uv run examples/build_triage_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pgx_digest import Triage, parse_pharmcat_json  # noqa: E402
from pgx_digest.triage_ml import (  # noqa: E402
    TrainingExample,
    example_from_pair,
    save_dataset,
)


def main() -> int:
    fixtures_dir = REPO_ROOT / "tests" / "fixtures"
    out_path = REPO_ROOT / "triage_data" / "training.jsonl"

    triage = Triage()
    examples: list[TrainingExample] = []
    seen: set[tuple[str, str, str]] = set()

    fixture_paths = sorted(fixtures_dir.glob("*.json")) + sorted(
        fixtures_dir.glob("*.report.json")
    )
    # `*.json` already matches `*.report.json`; dedupe.
    fixture_paths = sorted(set(fixture_paths))

    n_fixtures = 0
    n_pairs = 0
    n_unique = 0
    for fixture_path in fixture_paths:
        bundle = parse_pharmcat_json(fixture_path)
        n_fixtures += 1
        for finding in bundle.items:
            for drug in finding.affected_drugs:
                n_pairs += 1
                key = (finding.gene, drug.drug, drug.recommendation)
                if key in seen:
                    continue
                seen.add(key)
                n_unique += 1
                decision = triage.classify(finding, drug)
                examples.append(
                    example_from_pair(
                        finding,
                        drug,
                        decision.route,
                        notes=f"bootstrap:{decision.reason}",
                    )
                )

    written = save_dataset(tuple(examples), out_path)

    by_label: dict[str, int] = {}
    for ex in examples:
        by_label[ex.label] = by_label.get(ex.label, 0) + 1

    print(f"Fixtures processed:      {n_fixtures}")
    print(f"(finding, drug) pairs:   {n_pairs}")
    print(f"Unique examples kept:    {n_unique}")
    print(f"Labels:                  {dict(sorted(by_label.items()))}")
    print(f"Written to:              {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
