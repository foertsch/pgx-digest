"""NLI-based claim-citation grounding for the Verifier.

The default Verifier checks whether a card's narrative claim is grounded in
its cited PMIDs using cosine similarity over fastembed embeddings (threshold
~0.35). That's brittle — abstracts can be topically similar without actually
supporting the specific claim, and adversarial pairs can have high cosine
similarity by accident.

This module provides an alternative: a learned natural-language-inference
(NLI) classifier that scores `entailment / neutral / contradiction` on
(premise=abstract, hypothesis=claim) pairs. The Verifier accepts a claim
as grounded only if at least one cited abstract is judged to `entail` it
above a configurable threshold.

We use a pre-trained biomedical NLI checkpoint (PubMedBERT fine-tuned on
MNLI+MedNLI) rather than training from scratch — 65 fixtures is far too
few for that. The model runs on CPU in ~150-250 ms per pair; lazy-loaded
so importing this module is cheap if you never instantiate the class.

Install: ``uv sync --extra nli`` (pulls torch + transformers, ~1.5 GB).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover — types only
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

log = logging.getLogger(__name__)

# Default checkpoint — biomedical PubMedBERT fine-tuned on MNLI then MedNLI.
# About 440 MB; downloads on first use into ~/.cache/huggingface/.
DEFAULT_MODEL = "pritamdeka/PubMedBERT-MNLI-MedNLI"

# Label index → name. The PubMedBERT-MNLI-MedNLI checkpoint uses the
# transformers default MNLI label order: contradiction=0, neutral=1,
# entailment=2. We confirm at load time by reading model.config.id2label.
NLILabel = Literal["entailment", "neutral", "contradiction"]


@dataclass(frozen=True)
class NLIScore:
    """One NLI inference result.

    `score` is the probability of `label` — the argmax of the softmax over
    the three classes. `probs` keeps the full distribution if a caller
    wants to threshold differently (e.g. require entailment > 0.7 AND
    neutral < 0.2).
    """
    label: NLILabel
    score: float
    probs: dict[NLILabel, float]


class NLIGrounder:
    """Score whether a PubMed abstract entails a clinical claim.

    Lazy-loads the model on first call to `score()` so importing this
    module is cheap. The model lives on CPU; GPU is overkill for the
    handful of pairs we run per Verifier invocation.

    Example:
        grounder = NLIGrounder()
        result = grounder.score(
            premise="CYP2C19 poor metabolizers showed reduced clopidogrel "
                    "efficacy in a meta-analysis of 12 studies (n=8743).",
            hypothesis="Clopidogrel efficacy is reduced in CYP2C19 PMs.",
        )
        # result.label == "entailment", result.score ~= 0.95
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        entail_threshold: float = 0.5,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.entail_threshold = entail_threshold
        self.max_length = max_length
        # Loaded lazily on first .score() call.
        self._tokenizer: "AutoTokenizer | None" = None
        self._model: "AutoModelForSequenceClassification | None" = None
        self._id2label: dict[int, NLILabel] | None = None

    def _ensure_loaded(self) -> None:
        """Load the tokenizer + model on first inference."""
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401  — proves the optional extra is installed
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as e:
            raise ImportError(
                "NLIGrounder requires the 'nli' extra. Install with "
                "`uv sync --extra nli` (pulls torch + transformers)."
            ) from e

        log.info("Loading %s (first call; may take ~30s for ~440 MB download)",
                 self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        self._model.eval()

        # Normalize the label map. MNLI convention is contradiction=0,
        # neutral=1, entailment=2 but checkpoints sometimes flip this; trust
        # the checkpoint's own id2label.
        raw_id2label = self._model.config.id2label
        self._id2label = {
            int(k): self._normalize_label(v) for k, v in raw_id2label.items()
        }
        log.info("NLI label map: %s", self._id2label)

    @staticmethod
    def _normalize_label(raw: str) -> NLILabel:
        """Coerce a raw label string to one of the three NLI labels."""
        s = raw.strip().lower()
        if s.startswith("entail"):
            return "entailment"
        if s.startswith("contradict"):
            return "contradiction"
        if s.startswith("neutral"):
            return "neutral"
        raise ValueError(f"Unrecognized NLI label from checkpoint: {raw!r}")

    def score(self, premise: str, hypothesis: str) -> NLIScore:
        """Score one (premise, hypothesis) pair.

        `premise` is usually the abstract (longer, more authoritative).
        `hypothesis` is the claim being verified (shorter, what the LLM
        Drafter wrote). The NLI convention: does the premise entail the
        hypothesis?
        """
        import torch

        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._id2label is not None

        inputs = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).tolist()

        # Map idx → label, build full prob dist, find argmax.
        probs_by_label: dict[NLILabel, float] = {
            self._id2label[i]: float(p) for i, p in enumerate(probs)
        }
        best_label = max(probs_by_label.items(), key=lambda kv: kv[1])
        return NLIScore(
            label=best_label[0],
            score=best_label[1],
            probs=probs_by_label,
        )

    def is_grounded(self, claim: str, abstract: str) -> bool:
        """`True` iff the abstract entails the claim above `entail_threshold`."""
        result = self.score(premise=abstract, hypothesis=claim)
        return result.label == "entailment" and result.score >= self.entail_threshold


__all__ = ["NLIGrounder", "NLIScore", "NLILabel", "DEFAULT_MODEL"]
