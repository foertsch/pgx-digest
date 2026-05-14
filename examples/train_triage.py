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
"""Fit a LearnedTriage classifier from the bootstrap dataset.

Reads `triage_data/training.jsonl`, fits an embedding-based classifier
(`fastembed` BGE-small + sklearn LogisticRegression), and persists the
fitted model to `triage_data/classifier.joblib`.

Also writes a small `triage_data/eval_report.md` summarizing held-out
performance — useful for the PR description and the next ablation.

Run from the repo root:
    uv run examples/train_triage.py
    uv run examples/train_triage.py --test-fraction 0.25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from pgx_digest.embeddings import LocalEmbedder  # noqa: E402
from pgx_digest.triage_ml import (  # noqa: E402
    LearnedTriage,
    TrainingExample,
    _example_to_drug,
    _example_to_finding,
    _featurize,
    load_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "triage_data" / "training.jsonl",
    )
    parser.add_argument(
        "--out-model",
        type=Path,
        default=REPO_ROOT / "triage_data" / "classifier.joblib",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT / "triage_data" / "eval_report.md",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of examples held out for evaluation.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()

    examples = load_dataset(args.dataset)
    print(f"Loaded {len(examples)} examples from {args.dataset.name}")

    rng = np.random.default_rng(args.random_state)
    idx = rng.permutation(len(examples))
    n_test = max(1, int(round(args.test_fraction * len(examples))))
    test_idx = set(idx[:n_test].tolist())
    train_examples = tuple(
        ex for i, ex in enumerate(examples) if i not in test_idx
    )
    test_examples = tuple(
        ex for i, ex in enumerate(examples) if i in test_idx
    )
    print(
        f"Train: {len(train_examples)}  |  Test: {len(test_examples)}"
    )

    triage = LearnedTriage(embedder=LocalEmbedder())
    summary = triage.fit(train_examples, C=args.C, random_state=args.random_state)
    print(f"Fit summary: {summary}")

    # Held-out evaluation.
    test_acc, per_class = _evaluate(triage, test_examples)
    print(f"Held-out accuracy: {test_acc:.3f}")
    for cls, stats in per_class.items():
        print(f"  {cls:8s}  n={stats['n']:3d}  acc={stats['acc']:.3f}")

    model_path = triage.save(args.out_model)
    print(f"Saved classifier to {model_path}")

    _write_report(
        args.out_report,
        n_train=len(train_examples),
        n_test=len(test_examples),
        fit_summary=summary,
        test_acc=test_acc,
        per_class=per_class,
    )
    print(f"Wrote report to {args.out_report}")
    return 0


def _evaluate(
    triage: LearnedTriage, examples: tuple[TrainingExample, ...]
) -> tuple[float, dict[str, dict[str, float]]]:
    if not examples:
        return 0.0, {}
    correct = 0
    per_class: dict[str, dict[str, int]] = {}
    for ex in examples:
        # Use the public classify() path, but bypass the rule-based
        # fallback so we measure the classifier on its own.
        text = _featurize(_example_to_finding(ex), _example_to_drug(ex))
        X = triage.embedder.embed([text])
        proba = triage._classifier.predict_proba(X)[0]  # type: ignore[union-attr]
        pred = triage._classifier.classes_[int(proba.argmax())]  # type: ignore[union-attr]
        per_class.setdefault(ex.label, {"n": 0, "correct": 0})
        per_class[ex.label]["n"] += 1
        if pred == ex.label:
            correct += 1
            per_class[ex.label]["correct"] += 1
    out = {
        label: {
            "n": int(d["n"]),
            "acc": d["correct"] / max(1, d["n"]),
        }
        for label, d in per_class.items()
    }
    return correct / len(examples), out


def _write_report(
    path: Path,
    *,
    n_train: int,
    n_test: int,
    fit_summary: dict,
    test_acc: float,
    per_class: dict[str, dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LearnedTriage — training report",
        "",
        "Embedding-based classifier (fastembed BGE-small + sklearn",
        "LogisticRegression) for routing (PGxFinding, DrugRec) pairs.",
        "",
        f"- Train examples: **{n_train}**",
        f"- Held-out examples: **{n_test}**",
        f"- Embedding dim: **{fit_summary.get('embedding_dim')}**",
        f"- Train accuracy: **{fit_summary.get('train_accuracy')}**",
        f"- Class distribution (train): `{fit_summary.get('class_counts')}`",
        "",
        f"## Held-out accuracy: **{test_acc:.3f}**",
        "",
        "| Class | n | Accuracy |",
        "|---|---|---|",
    ]
    for cls in ("llm", "template", "skip"):
        stats = per_class.get(cls)
        if stats is None:
            lines.append(f"| {cls} | 0 | — |")
        else:
            lines.append(
                f"| {cls} | {stats['n']} | {stats['acc']:.3f} |"
            )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
