"""Learned Triage classifier — embedding-based alternative to rules.

The rule-based `Triage` (`pgx_digest/triage.py`) hand-codes a few
heuristics over phenotype/recommendation-text features. It works well
at the current scale, but the rules can miss patterns that a learned
model would catch (e.g. CPIC recommendations that *look* short and
unconditional but actually require nuance).

This module adds `LearnedTriage`: same `(PGxFinding, DrugRec) ->
TriageDecision` contract, but the decision comes from a small
classifier on top of sentence embeddings of the source CPIC text.

Architecture:

    (finding, drug)
        │
        ▼
    _featurize(finding, drug)  -> str
        │
        ▼
    Embedder.embed([...])      -> (1, dim) float32
        │
        ▼
    sklearn LogisticRegression -> proba over {llm, template, skip}
        │
        ▼
    TriageDecision

`fit()` trains the classifier from labeled examples; `save()` /
`load()` persist the fitted classifier (joblib). The embedder itself
is stateless at the model-weights level — no re-training across runs.

A confidence threshold (`fallback_threshold`) routes low-confidence
predictions back through the rule-based `Triage` so we never make a
worse routing decision than the deterministic baseline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from pgx_digest.bundle import DrugRec, PGxFinding
from pgx_digest.embeddings import Embedder, LocalEmbedder
from pgx_digest.triage import Route, Triage, TriageDecision


_LABELS: tuple[Route, ...] = ("llm", "template", "skip")


@dataclass(frozen=True)
class TrainingExample:
    """One labeled (finding, drug) pair for training."""

    gene: str
    phenotype: str
    drug: str
    evidence_level: str
    recommendation: str
    label: Route
    notes: str = ""


def _featurize(finding: PGxFinding, drug: DrugRec) -> str:
    """Compose a single text representation of (finding, drug) for the
    embedder.

    Format chosen for embedding alignment with CPIC-style language —
    the same phrasing tends to cluster across genes when the
    recommendation pattern is similar (e.g. all "Normal Metabolizer /
    no adjustment" pairs cluster together regardless of gene).
    """
    return (
        f"Gene: {finding.gene}. "
        f"Phenotype: {finding.phenotype}. "
        f"Drug: {drug.drug}. "
        f"Evidence Level: {drug.evidence_level}. "
        f"CPIC Recommendation: {drug.recommendation}"
    )


def example_from_pair(
    finding: PGxFinding, drug: DrugRec, label: Route, notes: str = ""
) -> TrainingExample:
    """Build a `TrainingExample` from a live (finding, drug) pair."""
    return TrainingExample(
        gene=finding.gene,
        phenotype=finding.phenotype,
        drug=drug.drug,
        evidence_level=str(drug.evidence_level),
        recommendation=drug.recommendation,
        label=label,
        notes=notes,
    )


def load_dataset(path: Path) -> tuple[TrainingExample, ...]:
    """Load training examples from a JSONL file."""
    examples: list[TrainingExample] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        raw = json.loads(line)
        examples.append(
            TrainingExample(
                gene=raw["gene"],
                phenotype=raw["phenotype"],
                drug=raw["drug"],
                evidence_level=raw["evidence_level"],
                recommendation=raw["recommendation"],
                label=raw["label"],
                notes=raw.get("notes", ""),
            )
        )
    return tuple(examples)


def save_dataset(examples: tuple[TrainingExample, ...], path: Path) -> Path:
    """Persist examples to JSONL. Sorted by (gene, drug) for stable diffs."""
    rows = sorted(
        (asdict(ex) for ex in examples), key=lambda r: (r["gene"], r["drug"])
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return path


class LearnedTriage(Triage):
    """Embedding-based Triage classifier.

    Subclasses the rule-based `Triage` and overrides `classify()`.
    `fit()` is called once (typically offline) to train the classifier
    on a labeled dataset; the resulting model can be persisted to disk
    via `save()` and reloaded via `load_classifier()`.

    Low-confidence predictions (max class probability below
    `fallback_threshold`) defer to the rule-based parent — the learned
    model is a *refinement*, not a replacement.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        fallback_threshold: float = 0.55,
        max_template_length: int = 200,
    ) -> None:
        super().__init__(max_template_length=max_template_length)
        self.embedder: Embedder = embedder or LocalEmbedder()
        self.fallback_threshold = fallback_threshold
        self._classifier: Any | None = None
        self._fit_summary: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        examples: tuple[TrainingExample, ...],
        *,
        C: float = 1.0,
        random_state: int = 0,
    ) -> dict[str, Any]:
        """Fit the classifier on labeled examples. Returns a summary dict."""
        if not examples:
            raise ValueError("LearnedTriage.fit requires at least one example")

        from sklearn.linear_model import LogisticRegression

        texts = [
            _featurize(
                _example_to_finding(ex), _example_to_drug(ex)
            )
            for ex in examples
        ]
        X = self.embedder.embed(texts)
        y = np.array([ex.label for ex in examples])

        # sklearn 1.5+ removed the `multi_class` kwarg; the default
        # solver (lbfgs) is multinomial when n_classes > 2 automatically.
        clf = LogisticRegression(
            C=C,
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )
        clf.fit(X, y)
        self._classifier = clf

        train_acc = float(clf.score(X, y))
        class_counts = {label: int((y == label).sum()) for label in _LABELS}
        self._fit_summary = {
            "n_examples": len(examples),
            "embedding_dim": int(X.shape[1]),
            "train_accuracy": round(train_acc, 4),
            "class_counts": class_counts,
            "classes": list(clf.classes_),
        }
        return dict(self._fit_summary)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> Path:
        """Persist the fitted classifier (joblib). Embedder is not saved."""
        if self._classifier is None:
            raise RuntimeError("LearnedTriage is not fitted; cannot save")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "classifier": self._classifier,
                "fit_summary": self._fit_summary,
            },
            path,
        )
        return path

    def load_classifier(self, path: Path) -> dict[str, Any]:
        """Load a previously-fitted classifier."""
        payload = joblib.load(path)
        self._classifier = payload["classifier"]
        self._fit_summary = dict(payload.get("fit_summary", {}))
        return dict(self._fit_summary)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def classify(
        self, finding: PGxFinding, drug: DrugRec
    ) -> TriageDecision:
        """Route this (finding, drug) — learned model with rule-based fallback."""
        if self._classifier is None:
            # Untrained: degrade gracefully to rule-based.
            return super().classify(finding, drug)

        text = _featurize(finding, drug)
        X = self.embedder.embed([text])
        proba = self._classifier.predict_proba(X)[0]
        idx = int(proba.argmax())
        confidence = float(proba[idx])
        pred: Route = self._classifier.classes_[idx]

        if confidence < self.fallback_threshold:
            base = super().classify(finding, drug)
            return TriageDecision(
                route=base.route,
                reason=(
                    f"low confidence (max proba {confidence:.2f} < "
                    f"{self.fallback_threshold}); fell back to rule "
                    f"({base.reason})"
                ),
            )

        return TriageDecision(
            route=pred,
            reason=f"learned (proba={confidence:.2f})",
        )


# ---------------------------------------------------------------------------
# Helpers — reconstruct lightweight PGxFinding / DrugRec from a TrainingExample
# so the same `_featurize` works for both training and inference.
# ---------------------------------------------------------------------------


def _example_to_finding(ex: TrainingExample) -> PGxFinding:
    return PGxFinding(
        gene=ex.gene,
        diplotype="-",
        source_variants=(),
        phenotype=ex.phenotype,
        phenotype_source="training",
        affected_drugs=(),
        confidence="high",
    )


def _example_to_drug(ex: TrainingExample) -> DrugRec:
    return DrugRec(
        drug=ex.drug,
        recommendation=ex.recommendation,
        cpic_guideline_id="-",
        pmids=(),
        evidence_level=ex.evidence_level,  # type: ignore[arg-type]
    )
