"""Retriever abstraction for RAG layers (PubMed grounding, CPIC enrichment).

Two concrete subclasses (`PubMedRetriever`, `CPICRetriever`) live in
`pgx_digest.corpora`. They share the same embedding-index plumbing
implemented here:

    items (text + metadata)
        │ Embedder.embed
        ▼
    (N, dim) float32 cached on disk
        │
        ▼  retrieve(query, k)
    query → embedder → top-k cosine matches → list[RetrievedItem]

Index serialization is a single npz + JSON pair so the cache survives
across runs (PubMed eutils + the CPIC API are external; we want to
hit them at most once per unique key).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pgx_digest.embeddings import Embedder


@dataclass(frozen=True)
class RetrievedItem:
    """One retrieved snippet, with a score and free-form metadata."""

    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class Retriever(ABC):
    """A keyword-or-semantic search over a fixed corpus."""

    @abstractmethod
    def retrieve(self, query: str, *, k: int = 5) -> list[RetrievedItem]:
        """Return the top-k items most relevant to `query`."""
        ...


class EmbeddingRetriever(Retriever):
    """Generic embedding-backed retriever.

    Subclasses are responsible for *what* goes in the index (a list
    of `RetrievedItem` instances with text + metadata). This base
    class owns the embedding + cosine-similarity + persistence
    machinery so concrete corpus retrievers stay small.

    Embeddings are L2-normalized by the underlying `Embedder` already
    (per the `Embedder.embed` contract), so cosine similarity is just
    matrix multiplication. No explicit normalization here.
    """

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._index_embeddings: np.ndarray | None = None
        self._index_items: list[RetrievedItem] = []

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    @property
    def n_items(self) -> int:
        return len(self._index_items)

    def build_index(self, items: list[RetrievedItem]) -> None:
        """Replace the index with `items` (text -> embedding)."""
        if not items:
            self._index_embeddings = None
            self._index_items = []
            return
        texts = [item.text for item in items]
        self._index_embeddings = self.embedder.embed(texts)
        self._index_items = list(items)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def retrieve(self, query: str, *, k: int = 5) -> list[RetrievedItem]:
        if self._index_embeddings is None or self.n_items == 0:
            return []
        q = self.embedder.embed([query])  # (1, dim), L2-normalized
        # cosine similarity = dot product on normalized vectors
        scores = (q @ self._index_embeddings.T)[0]  # (N,)
        k = min(k, self.n_items)
        top_idx = np.argpartition(-scores, k - 1)[:k]
        # Sort the k candidates descending by score.
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [
            RetrievedItem(
                text=self._index_items[i].text,
                score=float(scores[i]),
                metadata=dict(self._index_items[i].metadata),
            )
            for i in top_idx
        ]

    def get_by_metadata(
        self, key: str, value: Any
    ) -> RetrievedItem | None:
        """Find the first indexed item whose metadata[key] == value.

        Used by the Verifier's PubMed-grounding check: look up an
        abstract by its PMID without going through similarity search.
        """
        for item in self._index_items:
            if item.metadata.get(key) == value:
                return item
        return None

    def similarity(self, query: str, item: RetrievedItem) -> float:
        """Cosine similarity between `query` and `item`'s text.

        If the item is in the index, we reuse its cached embedding —
        otherwise we embed both fresh.
        """
        if not item.text:
            return 0.0
        q = self.embedder.embed([query])[0]  # L2-normalized
        # Try to find the item in the index by exact text match.
        if self._index_embeddings is not None:
            for i, indexed in enumerate(self._index_items):
                if indexed.text == item.text:
                    return float(q @ self._index_embeddings[i])
        # Not in index — embed the item fresh.
        item_emb = self.embedder.embed([item.text])[0]
        return float(q @ item_emb)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_index(self, path: Path) -> Path:
        """Persist embeddings + items to `<path>.npz` and `<path>.json`."""
        if self._index_embeddings is None:
            raise RuntimeError("EmbeddingRetriever has no index to save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path.with_suffix(".npz"),
            embeddings=self._index_embeddings,
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                [
                    {"text": item.text, "metadata": item.metadata}
                    for item in self._index_items
                ],
                sort_keys=True,
            )
        )
        return path

    def load_index(self, path: Path) -> None:
        """Load embeddings + items from the pair persisted by save_index."""
        path = Path(path)
        npz_path = path.with_suffix(".npz")
        json_path = path.with_suffix(".json")
        if not (npz_path.exists() and json_path.exists()):
            raise FileNotFoundError(
                f"missing index files for {path} (need .npz and .json)"
            )
        with np.load(npz_path) as data:
            self._index_embeddings = data["embeddings"]
        raw = json.loads(json_path.read_text())
        self._index_items = [
            RetrievedItem(text=r["text"], score=0.0, metadata=r.get("metadata") or {})
            for r in raw
        ]


# ---------------------------------------------------------------------------
# A few tiny convenience helpers used by both PubMed and CPIC retrievers.
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for two 1-D vectors (assumes they may not be normalized)."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
