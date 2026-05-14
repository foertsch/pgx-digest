"""CPIC guideline recommendations as a prompt-enrichment corpus.

The Bundle carries CPIC recommendations as-extracted-by-PharmCAT. The
CPIC API (api.cpicpgx.org, PostgREST-style) is the upstream source —
querying it directly gives us the canonical text, structured per
(drug, phenotype). Useful for:

- Cross-validating PharmCAT-extracted text (drift / staleness detection)
- Augmenting the Drafter prompt with authoritative recommendation
  language across multiple phenotype variants (the model sees the
  full clinical picture, not just the patient's phenotype)

Three components:
- `fetch_drug_id(name)` — drug name -> RxNorm:<n> drugid
- `fetch_recommendations(drugid)` — recommendations for a drug, one
  per (phenotype combination)
- `CPICRetriever` — `EmbeddingRetriever` indexing recommendation
  texts, with (gene, drug, phenotype) metadata for filtering.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

from pgx_digest.embeddings import Embedder, LocalEmbedder
from pgx_digest.retriever import EmbeddingRetriever, RetrievedItem


CPIC_API: Final[str] = "https://api.cpicpgx.org/v1"
DEFAULT_USER_AGENT: Final[str] = "pgx-digest/0.1 (research)"
DEFAULT_SLEEP_S: Final[float] = 0.1  # the CPIC API doesn't publish a rate limit; be polite


class CPICFetchError(RuntimeError):
    """CPIC API returned a non-200 or unexpected payload."""


def _http_get_json(
    url: str, *, user_agent: str, timeout: float = 30.0
) -> Any:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": user_agent, "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise CPICFetchError(
                f"CPIC API returned HTTP {resp.status} for {url}"
            )
        return json.loads(resp.read().decode("utf-8"))


def fetch_drug_id(
    drug_name: str,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    http_get_json=_http_get_json,
) -> str | None:
    """Resolve a drug name (case-insensitive) to a CPIC drugid like 'RxNorm:32968'.

    Returns None if no exact-name match exists.
    """
    params = urllib.parse.urlencode(
        {
            "name": f"ilike.{drug_name}",  # case-insensitive equality
            "select": "drugid,name",
        }
    )
    url = f"{CPIC_API}/drug?{params}"
    rows = http_get_json(url, user_agent=user_agent)
    if not rows:
        return None
    # Prefer the exact-name match (the ilike is case-insensitive but
    # we don't want partial matches if the CPIC API ever returns them).
    for row in rows:
        if row.get("name", "").lower() == drug_name.lower():
            return row["drugid"]
    return rows[0].get("drugid")


def fetch_recommendations(
    drugid: str,
    *,
    cache_dir: Path | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    http_get_json=_http_get_json,
) -> list[dict[str, Any]]:
    """Fetch all CPIC recommendations for one drugid.

    Each row is `{phenotypes, drugrecommendation, classification, ...}`.
    Cached per drugid as JSON on disk.
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = _cache_path(cache_dir, drugid)
        if cached.exists():
            return json.loads(cached.read_text())

    params = urllib.parse.urlencode(
        {
            "drugid": f"eq.{drugid}",
            "select": (
                "phenotypes,drugrecommendation,classification,"
                "implications,population"
            ),
        }
    )
    url = f"{CPIC_API}/recommendation?{params}"
    rows = http_get_json(url, user_agent=user_agent)
    if not isinstance(rows, list):
        raise CPICFetchError(
            f"unexpected CPIC payload for {drugid}: {type(rows).__name__}"
        )
    if cache_dir is not None:
        _cache_path(cache_dir, drugid).write_text(json.dumps(rows))
    return rows


def _cache_path(cache_dir: Path, drugid: str) -> Path:
    safe = drugid.replace(":", "_").replace("/", "_")
    return cache_dir / f"drug_{safe}.json"


class CPICRetriever(EmbeddingRetriever):
    """Retriever over CPIC recommendation snippets, keyed by (gene, drug, phenotype)."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        cache_dir: Path | None = None,
    ) -> None:
        super().__init__(embedder or LocalEmbedder())
        self.cache_dir = cache_dir

    def build_index_for_drugs(
        self,
        drug_names: Iterable[str],
        *,
        sleep_s: float = DEFAULT_SLEEP_S,
        http_get_json=_http_get_json,
    ) -> int:
        """Resolve each drug name, fetch its CPIC recs, index them. Returns n indexed."""
        items: list[RetrievedItem] = []
        seen_drugids: set[str] = set()
        for name in sorted({n.strip().lower() for n in drug_names if n}):
            drugid = fetch_drug_id(
                name,
                http_get_json=http_get_json,
            )
            if drugid is None or drugid in seen_drugids:
                continue
            seen_drugids.add(drugid)
            rows = fetch_recommendations(
                drugid,
                cache_dir=self.cache_dir,
                http_get_json=http_get_json,
            )
            for row in rows:
                phenotypes = row.get("phenotypes") or {}
                text = row.get("drugrecommendation") or ""
                if not text:
                    continue
                # Embed the recommendation prose plus a small structured
                # preamble — improves retrieval signal when the query
                # carries phenotype/gene context.
                preamble = "; ".join(
                    f"{g}={p}" for g, p in sorted(phenotypes.items())
                )
                indexed = (
                    f"{preamble} | classification={row.get('classification', '?')}: "
                    f"{text}"
                    if preamble
                    else text
                )
                # Pick the first gene in phenotypes for metadata (the
                # most common case has exactly one gene per row).
                primary_gene = (
                    sorted(phenotypes.keys())[0] if phenotypes else None
                )
                items.append(
                    RetrievedItem(
                        text=indexed,
                        score=0.0,
                        metadata={
                            "drug": name,
                            "drugid": drugid,
                            "phenotypes": dict(phenotypes),
                            "gene": primary_gene,
                            "classification": row.get("classification"),
                            "source": "cpic",
                        },
                    )
                )
            if sleep_s > 0:
                time.sleep(sleep_s)
        self.build_index(items)
        return len(items)
