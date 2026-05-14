"""Embedder abstraction for the learned-Triage and RAG layers.

Concrete embedders are responsible for one thing: take strings, return
fixed-dimension vectors. The interface deliberately mirrors `Provider`
in `drafter.py` — one method, swap-in alternatives, no per-backend
leak into callers.

Default `LocalEmbedder` uses `fastembed` (ONNX Runtime + BGE-small-EN).
~33 MB on first download, then cached locally. No PyTorch dep, no
CUDA, no API calls — privacy-first by construction. The first call
downloads the model from HuggingFace; subsequent calls reuse the
on-disk cache.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder(ABC):
    """Embed strings into dense vectors. Concrete subclasses pick a model."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension."""
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (N, dim) float32 array of L2-normalized embeddings."""
        ...


class LocalEmbedder(Embedder):
    """fastembed-backed local embedder. No API calls, no PyTorch.

    Lazy: the underlying model is downloaded + loaded on first `embed`
    call, not at `__init__`. Tests that pass an injected `_model`
    avoid the download.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        _model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = _model
        self._dim: int | None = None

    def _ensure_model(self) -> None:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)

    @property
    def dim(self) -> int:
        if self._dim is None:
            # Probe with a tiny input. Cheap on a loaded model.
            arr = self.embed(["probe"])
            self._dim = int(arr.shape[1])
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            # `dim` may not be set yet — return an empty (0, ?) array
            # by probing the model once.
            self._ensure_model()
            probe = list(self._model.embed(["probe"]))  # type: ignore[union-attr]
            return np.zeros((0, len(probe[0])), dtype=np.float32)

        self._ensure_model()
        # fastembed returns a generator of numpy arrays.
        vectors = list(self._model.embed(texts))  # type: ignore[union-attr]
        return np.stack([v.astype(np.float32) for v in vectors])
