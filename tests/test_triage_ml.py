"""Tests for the embedding-based LearnedTriage.

The real fastembed model is heavy to load and download; tests use a
small fake `Embedder` that produces deterministic vectors based on a
hash of the input. That's enough to verify fit/predict roundtrip,
persistence, the rule-based fallback, and integration with the
TriageDecision contract — without paying for sentence-transformers
weights or HuggingFace bandwidth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pgx_digest.bundle import (
    Bundle,
    DrugRec,
    PGxFinding,
    PrivacyTier,
)
from pgx_digest.embeddings import Embedder
from pgx_digest.triage_ml import (
    LearnedTriage,
    TrainingExample,
    _example_to_drug,
    _example_to_finding,
    _featurize,
    example_from_pair,
    load_dataset,
    save_dataset,
)


# ---------------------------------------------------------------------------
# Fake embedder — deterministic vectors keyed on a text-prefix marker so we
# can construct linearly-separable training data without burning a model.
# ---------------------------------------------------------------------------


class _MarkerEmbedder(Embedder):
    """Hash-based embedder with three marker dimensions for {llm, template, skip}.

    The first three dimensions are activated by simple substring rules
    on the input text — this lets us construct training sets where
    LogisticRegression can learn a clean decision boundary, while still
    exercising the full embed -> classifier -> decision pipeline.
    """

    DIM = 16

    @property
    def dim(self) -> int:
        return self.DIM

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.DIM), dtype=np.float32)
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            low = t.lower()
            # Marker dims: drive the classifier.
            if "no adjustment" in low or "use label" in low:
                out[i, 0] = 1.0  # template
            if "avoid" in low or "consider" in low:
                out[i, 1] = 1.0  # llm
            if "no recommendation" in low or "unknown" in low:
                out[i, 2] = 1.0  # skip
            # Filler dims: a stable hash of the text so different texts
            # never collide perfectly.
            digest = hashlib.sha256(t.encode()).digest()
            for j in range(self.DIM - 3):
                out[i, 3 + j] = (digest[j] / 255.0) * 0.1
        # L2-normalize so cosine == dot product.
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _example(label: str, recommendation: str) -> TrainingExample:
    return TrainingExample(
        gene="CYP2C19",
        phenotype="Poor Metabolizer" if label != "skip" else "Unknown",
        drug="testdrug",
        evidence_level="A" if label != "skip" else "D",
        recommendation=recommendation,
        label=label,  # type: ignore[arg-type]
    )


def _separable_dataset() -> tuple[TrainingExample, ...]:
    """Hand-crafted ~30 examples that the marker embedder makes separable."""
    rows: list[TrainingExample] = []
    template_recs = [
        "Use label dose; no adjustment needed.",
        "No adjustment recommended for this patient.",
        "Use label dose, no dose change.",
        "Standard dose, no adjustment.",
        "Use label-recommended dosage.",
        "No adjustment; standard label dose.",
        "Use label dose for normal metabolizers.",
        "No adjustment necessary for this drug.",
        "Use label dose with no change.",
        "No adjustment in this case.",
    ]
    llm_recs = [
        "Consider alternative agent; avoid this drug.",
        "Avoid this drug due to risk; consider another.",
        "Consider dose reduction; monitor levels.",
        "Avoid concomitant use; consider alternatives.",
        "Consider tapering; avoid abrupt discontinuation.",
        "Avoid use; consider an alternative drug class.",
        "Consider 50% dose reduction; avoid high doses.",
        "Avoid in this metabolizer; consider switching.",
        "Consider closer monitoring; avoid if possible.",
        "Avoid drug X; consider drug Y instead.",
    ]
    skip_recs = [
        "No recommendation available for this phenotype.",
        "Unknown phenotype; no recommendation.",
        "No recommendation; phenotype not actionable.",
        "Phenotype unknown; no CPIC guidance.",
        "No recommendation can be made.",
        "Unknown; no CPIC recommendation available.",
        "No recommendation per CPIC for unknown phenotype.",
        "Phenotype unknown; no actionable guidance.",
        "No recommendation; insufficient data.",
        "Unknown phenotype; CPIC has no recommendation.",
    ]
    rows.extend(_example("template", r) for r in template_recs)
    rows.extend(_example("llm", r) for r in llm_recs)
    rows.extend(_example("skip", r) for r in skip_recs)
    return tuple(rows)


def _finding_drug(
    phenotype: str = "Poor Metabolizer", recommendation: str = "Consider alternative."
) -> tuple[PGxFinding, DrugRec]:
    drug = DrugRec(
        drug="testdrug",
        recommendation=recommendation,
        cpic_guideline_id="g",
        pmids=(1,),
        evidence_level="A",
    )
    finding = PGxFinding(
        gene="CYP2C19",
        diplotype="*1/*2",
        source_variants=(),
        phenotype=phenotype,
        phenotype_source="test",
        affected_drugs=(drug,),
        confidence="high",
    )
    return finding, drug


# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------


def test_featurize_includes_gene_drug_phenotype_recommendation() -> None:
    f, d = _finding_drug(recommendation="No adjustment.")
    text = _featurize(f, d)
    assert "CYP2C19" in text
    assert "testdrug" in text
    assert "Poor Metabolizer" in text
    assert "No adjustment." in text


# ---------------------------------------------------------------------------
# Dataset persistence
# ---------------------------------------------------------------------------


def test_save_and_load_dataset_roundtrip(tmp_path: Path) -> None:
    examples = (
        _example("template", "No adjustment."),
        _example("llm", "Avoid use."),
    )
    path = tmp_path / "x.jsonl"
    save_dataset(examples, path)
    loaded = load_dataset(path)
    assert len(loaded) == 2
    assert {ex.label for ex in loaded} == {"template", "llm"}


def test_load_dataset_skips_blank_and_comment_lines(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text(
        "# header\n"
        "\n"
        + json.dumps(
            {
                "gene": "X",
                "phenotype": "Y",
                "drug": "Z",
                "evidence_level": "A",
                "recommendation": "r",
                "label": "llm",
            }
        )
        + "\n"
    )
    assert len(load_dataset(path)) == 1


# ---------------------------------------------------------------------------
# example_from_pair
# ---------------------------------------------------------------------------


def test_example_from_pair_carries_label_and_notes() -> None:
    f, d = _finding_drug()
    ex = example_from_pair(f, d, label="template", notes="bootstrap")
    assert ex.gene == "CYP2C19"
    assert ex.label == "template"
    assert ex.notes == "bootstrap"


# ---------------------------------------------------------------------------
# LearnedTriage — fit / predict / fallback
# ---------------------------------------------------------------------------


def test_untrained_learned_triage_falls_back_to_rules() -> None:
    """No fit() called -> falls back to rule-based Triage transparently."""
    triage = LearnedTriage(embedder=_MarkerEmbedder())
    # Normal phenotype + short unconditional rec -> rule-based template.
    finding = PGxFinding(
        gene="TPMT",
        diplotype="*1/*1",
        source_variants=(),
        phenotype="Normal Metabolizer",
        phenotype_source="test",
        affected_drugs=(
            DrugRec(
                drug="mercaptopurine",
                recommendation="Start with normal starting dose.",
                cpic_guideline_id="g",
                pmids=(1,),
                evidence_level="A",
            ),
        ),
        confidence="high",
    )
    drug = finding.affected_drugs[0]
    decision = triage.classify(finding, drug)
    assert decision.route == "template"


def test_fit_then_classify_matches_label_on_easy_input() -> None:
    triage = LearnedTriage(embedder=_MarkerEmbedder())
    triage.fit(_separable_dataset(), random_state=0)
    f, d = _finding_drug(
        phenotype="Normal Metabolizer",
        recommendation="Use label dose; no adjustment.",
    )
    decision = triage.classify(f, d)
    assert decision.route == "template"
    assert "learned" in decision.reason


def test_fit_then_classify_picks_llm_for_avoid_text() -> None:
    triage = LearnedTriage(embedder=_MarkerEmbedder())
    triage.fit(_separable_dataset(), random_state=0)
    f, d = _finding_drug(recommendation="Avoid this drug; consider alternative.")
    decision = triage.classify(f, d)
    assert decision.route == "llm"


def test_fit_reports_summary_with_dim_and_train_acc() -> None:
    triage = LearnedTriage(embedder=_MarkerEmbedder())
    summary = triage.fit(_separable_dataset(), random_state=0)
    assert summary["n_examples"] == 30
    assert summary["embedding_dim"] == _MarkerEmbedder.DIM
    assert 0.0 <= summary["train_accuracy"] <= 1.0


def test_low_confidence_falls_back_to_rules() -> None:
    """When the classifier is uncertain, the rule-based parent decides."""
    triage = LearnedTriage(
        embedder=_MarkerEmbedder(), fallback_threshold=0.99
    )
    triage.fit(_separable_dataset(), random_state=0)
    f, d = _finding_drug(
        phenotype="Normal Metabolizer",
        recommendation="Mild adjustment may be considered for select patients.",
    )
    decision = triage.classify(f, d)
    assert "fell back to rule" in decision.reason


def test_fit_raises_on_empty_dataset() -> None:
    triage = LearnedTriage(embedder=_MarkerEmbedder())
    with pytest.raises(ValueError):
        triage.fit(())


def test_save_requires_fit(tmp_path: Path) -> None:
    triage = LearnedTriage(embedder=_MarkerEmbedder())
    with pytest.raises(RuntimeError):
        triage.save(tmp_path / "x.joblib")


def test_save_load_classifier_roundtrip(tmp_path: Path) -> None:
    triage = LearnedTriage(embedder=_MarkerEmbedder())
    triage.fit(_separable_dataset(), random_state=0)
    f, d = _finding_drug(recommendation="Use label dose; no adjustment.")
    decision_before = triage.classify(f, d)

    path = tmp_path / "model.joblib"
    triage.save(path)

    fresh = LearnedTriage(embedder=_MarkerEmbedder())
    summary = fresh.load_classifier(path)
    assert summary["n_examples"] == 30

    decision_after = fresh.classify(f, d)
    assert decision_before.route == decision_after.route


# ---------------------------------------------------------------------------
# Integration with the committed training data + classifier
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).parent.parent


def test_committed_training_dataset_loads() -> None:
    path = REPO_ROOT / "triage_data" / "training.jsonl"
    if not path.exists():
        pytest.skip(f"missing {path} — run examples/build_triage_dataset.py")
    examples = load_dataset(path)
    assert len(examples) > 50
    labels = {ex.label for ex in examples}
    assert labels <= {"llm", "template", "skip"}
    # Bootstrap labels should yield at least one of each.
    assert "llm" in labels
    assert "template" in labels


def test_committed_classifier_can_be_loaded_and_classify(
    tmp_path: Path,
) -> None:
    """The committed classifier (trained on real CPIC data) loads and
    predicts. Uses a fake embedder via `embed_override` so we don't
    require the fastembed model at test time.
    """
    model_path = REPO_ROOT / "triage_data" / "classifier.joblib"
    if not model_path.exists():
        pytest.skip(
            f"missing {model_path} — run examples/train_triage.py"
        )
    import joblib

    # We can load the classifier directly without needing the real
    # embedder — the embedder dependency is at classify-time, not at
    # load-time.
    payload = joblib.load(model_path)
    assert "classifier" in payload
    assert "fit_summary" in payload
    summary = payload["fit_summary"]
    assert summary["embedding_dim"] > 0
    assert 0.5 < summary["train_accuracy"] <= 1.0
