"""Tests for the PubMed + CPIC corpus retrievers.

All HTTP is mocked. One optional real-network smoke test per corpus
is gated on `PGX_LIVE_NETWORK_TESTS=1` so it doesn't run in CI by
default.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from pgx_digest.corpora.cpic import (
    CPICRetriever,
    fetch_drug_id,
    fetch_recommendations,
)
from pgx_digest.corpora.pubmed import (
    PubMedRetriever,
    fetch_abstracts,
    parse_abstracts,
)
from pgx_digest.embeddings import Embedder


# ---------------------------------------------------------------------------
# Shared hash embedder (reuse from test_retriever indirectly is fine)
# ---------------------------------------------------------------------------


class _HashEmbedder(Embedder):
    DIM = 8

    @property
    def dim(self) -> int:
        return self.DIM

    def embed(self, texts: list[str]) -> np.ndarray:
        import hashlib

        if not texts:
            return np.zeros((0, self.DIM), dtype=np.float32)
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            digest = hashlib.sha256(t.encode()).digest()
            for j in range(self.DIM):
                out[i, j] = (digest[j] / 255.0) * 2 - 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


# ---------------------------------------------------------------------------
# PubMed
# ---------------------------------------------------------------------------


# Synthetic eutils efetch XML payload covering two PMIDs. Matches the
# shape `parse_abstracts` looks for: PubmedArticleSet > PubmedArticle >
# MedlineCitation > {PMID, Article > Abstract > AbstractText}.
_PUBMED_FIXTURE = """\
<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">21716271</PMID>
      <Article>
        <ArticleTitle>Clopidogrel and CYP2C19.</ArticleTitle>
        <Abstract>
          <AbstractText>CYP2C19 poor metabolizers exhibit reduced clopidogrel active metabolite formation and a higher rate of cardiovascular events. Alternative antiplatelet agents such as prasugrel or ticagrelor are recommended.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">28198005</PMID>
      <Article>
        <ArticleTitle>Voriconazole pharmacogenetics.</ArticleTitle>
        <Abstract>
          <AbstractText>Voriconazole exposure varies markedly across CYP2C19 phenotypes. Poor metabolizers may achieve supratherapeutic concentrations; ultrarapid metabolizers may have insufficient exposure for invasive aspergillosis treatment.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


def test_parse_abstracts_extracts_two_pmids() -> None:
    parsed = parse_abstracts(_PUBMED_FIXTURE)
    assert set(parsed.keys()) == {21716271, 28198005}
    assert "antiplatelet" in parsed[21716271]
    assert "Voriconazole" in parsed[28198005]


def test_parse_abstracts_empty_text() -> None:
    assert parse_abstracts("") == {}


def test_fetch_abstracts_uses_injected_http_get() -> None:
    captured: list[str] = []

    def fake_http_get(url: str, **kwargs):
        captured.append(url)
        return _PUBMED_FIXTURE

    result = fetch_abstracts(
        [21716271, 28198005],
        http_get=fake_http_get,
        cache_dir=None,
        sleep_s=0.0,
    )
    assert set(result.keys()) == {21716271, 28198005}
    assert len(captured) == 1
    # urlencode replaces comma with %2C — check either form.
    url = captured[0]
    assert "21716271" in url and "28198005" in url
    assert "%2C" in url or "," in url


def test_fetch_abstracts_caches_to_disk(tmp_path: Path) -> None:
    calls = []

    def fake_http_get(url: str, **kwargs):
        calls.append(url)
        return _PUBMED_FIXTURE

    # First call: hits HTTP.
    result1 = fetch_abstracts(
        [21716271, 28198005],
        http_get=fake_http_get,
        cache_dir=tmp_path,
        sleep_s=0.0,
    )
    assert len(calls) == 1
    assert (tmp_path / "pmid_21716271.json").exists()
    assert (tmp_path / "pmid_28198005.json").exists()

    # Second call: served from cache, no new HTTP.
    result2 = fetch_abstracts(
        [21716271, 28198005],
        http_get=fake_http_get,
        cache_dir=tmp_path,
        sleep_s=0.0,
    )
    assert len(calls) == 1
    assert result1 == result2


def test_pubmed_retriever_build_index_for_pmids() -> None:
    def fake_http_get(url: str, **kwargs):
        return _PUBMED_FIXTURE

    r = PubMedRetriever(embedder=_HashEmbedder())
    n = r.build_index_for_pmids(
        [21716271, 28198005], http_get=fake_http_get, sleep_s=0.0
    )
    assert n == 2
    item = r.get_by_metadata("pmid", 21716271)
    assert item is not None
    assert "antiplatelet" in item.text


def test_pubmed_retriever_can_retrieve_by_query() -> None:
    """The hash embedder doesn't have semantic structure, so we can't
    assert *which* abstract comes back on top — but we can verify the
    retriever surfaces hits keyed on the right PMIDs.
    """

    def fake_http_get(url: str, **kwargs):
        return _PUBMED_FIXTURE

    r = PubMedRetriever(embedder=_HashEmbedder())
    r.build_index_for_pmids([21716271, 28198005], http_get=fake_http_get, sleep_s=0.0)
    hits = r.retrieve("anything", k=2)
    assert {h.metadata["pmid"] for h in hits} == {21716271, 28198005}


def test_pubmed_retriever_similarity_grounds_card_against_abstract() -> None:
    """When a card's recommendation is identical to the abstract text,
    similarity should be ~1; for unrelated text it should be lower.
    """

    def fake_http_get(url: str, **kwargs):
        return _PUBMED_FIXTURE

    r = PubMedRetriever(embedder=_HashEmbedder())
    r.build_index_for_pmids([21716271], http_get=fake_http_get, sleep_s=0.0)
    item = r.get_by_metadata("pmid", 21716271)
    assert item is not None
    sim_match = r.similarity(item.text, item)
    sim_other = r.similarity("totally unrelated text about cars", item)
    assert sim_match > sim_other


# ---------------------------------------------------------------------------
# CPIC
# ---------------------------------------------------------------------------


def test_fetch_drug_id_picks_exact_name_match() -> None:
    def fake_http_get_json(url: str, **kwargs):
        # CPIC API returns a list of {drugid, name}.
        return [
            {"drugid": "RxNorm:32968", "name": "clopidogrel"},
            {"drugid": "RxNorm:99999", "name": "clopidogrel-something"},
        ]

    drugid = fetch_drug_id("clopidogrel", http_get_json=fake_http_get_json)
    assert drugid == "RxNorm:32968"


def test_fetch_drug_id_returns_none_when_no_rows() -> None:
    def fake_http_get_json(url: str, **kwargs):
        return []

    assert fetch_drug_id("nosuchdrug", http_get_json=fake_http_get_json) is None


def test_fetch_recommendations_caches_per_drugid(tmp_path: Path) -> None:
    calls = []

    def fake_http_get_json(url: str, **kwargs):
        calls.append(url)
        return [
            {
                "phenotypes": {"CYP2C19": "Poor Metabolizer"},
                "drugrecommendation": "Avoid clopidogrel if possible.",
                "classification": "Strong",
                "implications": "Reduced active metabolite.",
                "population": "general",
            }
        ]

    rows1 = fetch_recommendations(
        "RxNorm:32968",
        cache_dir=tmp_path,
        http_get_json=fake_http_get_json,
    )
    assert len(rows1) == 1
    assert "Avoid clopidogrel" in rows1[0]["drugrecommendation"]
    assert len(calls) == 1
    # Cache hit on second call.
    rows2 = fetch_recommendations(
        "RxNorm:32968",
        cache_dir=tmp_path,
        http_get_json=fake_http_get_json,
    )
    assert len(calls) == 1
    assert rows2 == rows1


def test_cpic_retriever_build_index_for_drugs() -> None:
    def fake_http_get_json(url: str, **kwargs):
        if "/drug?" in url:
            return [{"drugid": "RxNorm:32968", "name": "clopidogrel"}]
        # /recommendation?
        return [
            {
                "phenotypes": {"CYP2C19": "Poor Metabolizer"},
                "drugrecommendation": "Avoid clopidogrel if possible.",
                "classification": "Strong",
            },
            {
                "phenotypes": {"CYP2C19": "Normal Metabolizer"},
                "drugrecommendation": "Use standard 75 mg/day dose.",
                "classification": "Strong",
            },
        ]

    r = CPICRetriever(embedder=_HashEmbedder())
    n = r.build_index_for_drugs(
        ["clopidogrel"],
        http_get_json=fake_http_get_json,
        sleep_s=0.0,
    )
    assert n == 2
    # Metadata carries phenotypes + drugid.
    assert r._index_items[0].metadata["drugid"] == "RxNorm:32968"
    assert r._index_items[0].metadata["gene"] == "CYP2C19"


def test_cpic_retriever_dedups_drug_names() -> None:
    """Mixed-case + duplicate names should not duplicate fetches."""
    calls = []

    def fake_http_get_json(url: str, **kwargs):
        calls.append(url)
        if "/drug?" in url:
            return [{"drugid": "RxNorm:32968", "name": "clopidogrel"}]
        return []

    r = CPICRetriever(embedder=_HashEmbedder())
    r.build_index_for_drugs(
        ["clopidogrel", "Clopidogrel", "CLOPIDOGREL"],
        http_get_json=fake_http_get_json,
        sleep_s=0.0,
    )
    # /drug? for "clopidogrel" (deduped) + /recommendation? for the drugid.
    assert len([c for c in calls if "/drug?" in c]) == 1


# ---------------------------------------------------------------------------
# Optional live-network smoke (skipped unless PGX_LIVE_NETWORK_TESTS=1)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("PGX_LIVE_NETWORK_TESTS") != "1",
    reason="set PGX_LIVE_NETWORK_TESTS=1 to enable live PubMed / CPIC tests",
)
def test_pubmed_live_smoke(tmp_path: Path) -> None:
    """One real eutils call, gated. Verifies the XML parser pipeline
    against actual NCBI output.

    Uses a known-with-abstract PMID (30093137, a research paper) and a
    known-without-abstract PMID (21716271, a CPIC guideline letter).
    Confirms both shapes parse correctly: abstracts come through with
    content; abstract-less papers silently drop out of the result.
    """
    abstracts = fetch_abstracts(
        [21716271, 30093137],
        cache_dir=tmp_path,
        sleep_s=0.0,
    )
    # PMID 30093137 (research paper) MUST come back with an abstract.
    assert 30093137 in abstracts
    assert len(abstracts[30093137]) > 100  # non-trivial abstract length
    # PMID 21716271 (CPIC guideline letter, no abstract field) MUST be
    # silently absent — that's the contract.
    assert 21716271 not in abstracts


@pytest.mark.skipif(
    os.environ.get("PGX_LIVE_NETWORK_TESTS") != "1",
    reason="set PGX_LIVE_NETWORK_TESTS=1 to enable live CPIC tests",
)
def test_cpic_live_smoke(tmp_path: Path) -> None:
    drugid = fetch_drug_id("clopidogrel")
    assert drugid is not None
    rows = fetch_recommendations(drugid, cache_dir=tmp_path)
    assert any(
        r.get("phenotypes", {}).get("CYP2C19") == "Poor Metabolizer"
        for r in rows
    )
