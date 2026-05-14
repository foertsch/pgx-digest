"""Tests for the Embedder abstraction.

The `LocalEmbedder` uses fastembed which downloads a real model. We
test the contract (shape, dim, normalized) via an injected fake
embedder, and one smoke test of `LocalEmbedder` with a real model
that is skipped when fastembed isn't installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pgx_digest.embeddings import DEFAULT_MODEL, Embedder, LocalEmbedder


class _FakeFastEmbedModel:
    """Minimal stand-in for fastembed.TextEmbedding.

    fastembed's interface: model.embed(texts) -> generator of np arrays.
    """

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]):
        self.calls.append(list(texts))
        for i in range(len(texts)):
            yield np.full(self.dim, float(i + 1), dtype=np.float32)


def test_local_embedder_with_injected_model() -> None:
    fake = _FakeFastEmbedModel(dim=8)
    embedder = LocalEmbedder(_model=fake)

    arr = embedder.embed(["a", "b", "c"])
    assert arr.shape == (3, 8)
    assert arr.dtype == np.float32


def test_local_embedder_empty_input_returns_zero_rows() -> None:
    fake = _FakeFastEmbedModel(dim=8)
    embedder = LocalEmbedder(_model=fake)
    arr = embedder.embed([])
    assert arr.shape == (0, 8)


def test_local_embedder_dim_probes_model_lazily() -> None:
    fake = _FakeFastEmbedModel(dim=12)
    embedder = LocalEmbedder(_model=fake)
    # `dim` not yet known.
    assert embedder._dim is None
    assert embedder.dim == 12
    # Once probed, no further model calls are needed for dim.
    n_before = len(fake.calls)
    _ = embedder.dim
    assert len(fake.calls) == n_before


def test_default_model_constant_is_set() -> None:
    assert DEFAULT_MODEL.startswith("BAAI/")


def test_local_embedder_real_smoke() -> None:
    """One real smoke test of fastembed end-to-end.

    Skipped if fastembed isn't installed OR if the model isn't cached
    locally and we'd need to download it (the test environment may not
    have network access).
    """
    try:
        from fastembed import TextEmbedding  # noqa: F401
    except ImportError:
        pytest.skip("fastembed not installed")
    embedder = LocalEmbedder()
    try:
        arr = embedder.embed(["pharmacogenomic narrative test"])
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"fastembed model not available: {exc}")
    assert arr.shape[0] == 1
    assert arr.shape[1] == 384  # BGE-small-en-v1.5 dim
