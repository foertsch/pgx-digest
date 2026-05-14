"""Tests for the Retriever abstraction.

A fake embedder produces deterministic vectors from a hash of the
input string, so we can test the retrieval / persistence / similarity
plumbing without burning a real embedder model.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from pgx_digest.embeddings import Embedder
from pgx_digest.retriever import EmbeddingRetriever, RetrievedItem


class _HashEmbedder(Embedder):
    """Deterministic embedder for tests. Produces L2-normalized vectors."""

    DIM = 8

    @property
    def dim(self) -> int:
        return self.DIM

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.DIM), dtype=np.float32)
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            digest = hashlib.sha256(t.encode()).digest()
            for j in range(self.DIM):
                out[i, j] = (digest[j] / 255.0) * 2 - 1  # in [-1, 1]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def _items(*texts: str, **md_per_idx: dict) -> list[RetrievedItem]:
    """Build RetrievedItems with optional per-index metadata."""
    return [
        RetrievedItem(
            text=t,
            score=0.0,
            metadata=md_per_idx.get(str(i), {"idx": i}),
        )
        for i, t in enumerate(texts)
    ]


def test_empty_index_returns_empty_list() -> None:
    r = EmbeddingRetriever(embedder=_HashEmbedder())
    assert r.retrieve("anything") == []


def test_retrieve_returns_self_at_top_when_query_in_index() -> None:
    r = EmbeddingRetriever(embedder=_HashEmbedder())
    items = _items("alpha", "beta", "gamma")
    r.build_index(items)
    hits = r.retrieve("alpha", k=3)
    assert hits[0].text == "alpha"


def test_retrieve_respects_k() -> None:
    r = EmbeddingRetriever(embedder=_HashEmbedder())
    r.build_index(_items("alpha", "beta", "gamma", "delta"))
    assert len(r.retrieve("z", k=2)) == 2
    assert len(r.retrieve("z", k=10)) == 4  # k > n -> clamp


def test_retrieve_scores_descending() -> None:
    r = EmbeddingRetriever(embedder=_HashEmbedder())
    r.build_index(_items("alpha", "beta", "gamma"))
    hits = r.retrieve("anything", k=3)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_get_by_metadata_finds_item() -> None:
    r = EmbeddingRetriever(embedder=_HashEmbedder())
    items = [
        RetrievedItem(text="x", score=0.0, metadata={"pmid": 100}),
        RetrievedItem(text="y", score=0.0, metadata={"pmid": 200}),
    ]
    r.build_index(items)
    hit = r.get_by_metadata("pmid", 200)
    assert hit is not None
    assert hit.text == "y"


def test_get_by_metadata_returns_none_when_missing() -> None:
    r = EmbeddingRetriever(embedder=_HashEmbedder())
    r.build_index(_items("a"))
    assert r.get_by_metadata("pmid", 42) is None


def test_similarity_uses_cached_index_when_item_indexed() -> None:
    r = EmbeddingRetriever(embedder=_HashEmbedder())
    r.build_index(_items("the abstract text"))
    indexed_item = r._index_items[0]
    sim = r.similarity("the abstract text", indexed_item)
    # Identical text -> similarity should be ~1.
    assert sim == pytest.approx(1.0, abs=0.05)


def test_similarity_falls_back_when_item_not_indexed() -> None:
    r = EmbeddingRetriever(embedder=_HashEmbedder())
    r.build_index(_items("alpha"))
    external = RetrievedItem(text="omega", score=0.0, metadata={})
    sim = r.similarity("omega", external)
    # Identical text vs external item -> ~1.
    assert sim == pytest.approx(1.0, abs=0.05)


def test_save_and_load_index_roundtrip(tmp_path: Path) -> None:
    r = EmbeddingRetriever(embedder=_HashEmbedder())
    r.build_index(
        [
            RetrievedItem(
                text="abstract A", score=0.0, metadata={"pmid": 1}
            ),
            RetrievedItem(
                text="abstract B", score=0.0, metadata={"pmid": 2}
            ),
        ]
    )
    path = tmp_path / "idx"
    r.save_index(path)

    r2 = EmbeddingRetriever(embedder=_HashEmbedder())
    r2.load_index(path)
    assert r2.n_items == 2
    assert r2.get_by_metadata("pmid", 2).text == "abstract B"
    # Embeddings round-tripped, so retrieve works on the reloaded index.
    hits = r2.retrieve("abstract B", k=1)
    assert hits[0].metadata["pmid"] == 2


def test_save_empty_index_raises(tmp_path: Path) -> None:
    r = EmbeddingRetriever(embedder=_HashEmbedder())
    with pytest.raises(RuntimeError):
        r.save_index(tmp_path / "idx")


def test_load_missing_index_raises(tmp_path: Path) -> None:
    r = EmbeddingRetriever(embedder=_HashEmbedder())
    with pytest.raises(FileNotFoundError):
        r.load_index(tmp_path / "missing")
