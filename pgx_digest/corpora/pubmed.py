"""PubMed abstracts as a citation-grounding corpus.

Each cited PMID in the Bundle resolves to an abstract via NCBI eutils.
The Verifier can then check that the LLM's claim about a citation is
semantically consistent with what the abstract actually says — turning
the project's typed-Verifier story from "we check that PMID N is in
the Bundle" into "we check that the LLM's claim about PMID N is
supported by PMID N's abstract."

Three components:
- `fetch_abstracts(pmids)` — eutils efetch with on-disk JSON cache.
  Uses the XML response (retmode=xml) and parses `<Abstract>` /
  `<AbstractText>` nodes — robust against the various plain-text
  layouts eutils emits.
- `parse_abstracts(text)` — extract `{pmid: abstract}` from an XML
  efetch response. Returns an empty mapping for PMIDs that have no
  abstract (CPIC guideline letters, retracted papers, etc.).
- `PubMedRetriever` — `EmbeddingRetriever` whose corpus is the union
  of fetched abstracts, keyed by PMID in metadata.

Unauthenticated eutils rate-limits at 3 req/sec. We batch many PMIDs
into a single efetch request to stay well under that — typical
ablation runs need <10 unique requests for the whole corpus.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path
from typing import Final

from pgx_digest.embeddings import Embedder, LocalEmbedder
from pgx_digest.retriever import EmbeddingRetriever, RetrievedItem


EUTILS_EFETCH: Final[str] = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
)
DEFAULT_USER_AGENT: Final[str] = "pgx-digest/0.1 (research; mailto:noreply@example.com)"
DEFAULT_BATCH: Final[int] = 50  # PMIDs per efetch call
DEFAULT_SLEEP_S: Final[float] = 0.35  # ~3 req/sec, conservative


class PubMedFetchError(RuntimeError):
    """eutils returned a non-200 or unparseable response."""


def _http_get(url: str, *, user_agent: str, timeout: float = 30.0) -> str:
    """Plain HTTP GET. Lifts urllib out of the call sites for testability."""
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise PubMedFetchError(
                f"eutils returned HTTP {resp.status} for {url}"
            )
        return resp.read().decode("utf-8")


def fetch_abstracts(
    pmids: Iterable[int],
    *,
    cache_dir: Path | None = None,
    batch_size: int = DEFAULT_BATCH,
    sleep_s: float = DEFAULT_SLEEP_S,
    user_agent: str = DEFAULT_USER_AGENT,
    http_get=_http_get,
) -> dict[int, str]:
    """Fetch PubMed abstracts for the given PMIDs.

    Returns a `{pmid: abstract_text}` mapping. PMIDs not found in
    PubMed silently drop out of the result. Cached results on disk
    (one JSON per PMID under `cache_dir`) survive across runs.

    `http_get` is injected so tests can stub the network call without
    monkeypatching urllib.
    """
    pmids_unique = sorted({int(p) for p in pmids if p})
    if not pmids_unique:
        return {}

    cache: dict[int, str] = {}

    # 1. Pull what's already cached.
    to_fetch: list[int] = []
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for pmid in pmids_unique:
            cached = _cache_path(cache_dir, pmid)
            if cached.exists():
                cache[pmid] = json.loads(cached.read_text())["abstract"]
            else:
                to_fetch.append(pmid)
    else:
        to_fetch = list(pmids_unique)

    # 2. Fetch the rest in batches.
    for i in range(0, len(to_fetch), batch_size):
        batch = to_fetch[i : i + batch_size]
        params = urllib.parse.urlencode(
            {
                "db": "pubmed",
                "id": ",".join(str(p) for p in batch),
                "rettype": "abstract",
                "retmode": "xml",
            }
        )
        url = f"{EUTILS_EFETCH}?{params}"
        text = http_get(url, user_agent=user_agent)
        parsed = parse_abstracts(text)
        for pmid in batch:
            abstract = parsed.get(pmid)
            if abstract is None:
                # Some PMIDs (retracted, missing) won't have abstracts.
                continue
            cache[pmid] = abstract
            if cache_dir is not None:
                _cache_path(cache_dir, pmid).write_text(
                    json.dumps({"pmid": pmid, "abstract": abstract})
                )
        if i + batch_size < len(to_fetch):
            time.sleep(sleep_s)

    return cache


def _cache_path(cache_dir: Path, pmid: int) -> Path:
    return cache_dir / f"pmid_{pmid}.json"


def parse_abstracts(xml_text: str) -> dict[int, str]:
    """Parse a PubMed eutils efetch XML response into {pmid: abstract}.

    Walks `<PubmedArticle>` nodes, pulls the `<PMID>` and concatenates
    all `<AbstractText>` sub-elements. Returns an empty mapping for
    PMIDs that have no abstract (CPIC guideline letters, retracted
    papers, editorials, etc.) — those PMIDs are silently dropped.
    """
    if not xml_text or not xml_text.strip():
        return {}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return {}

    out: dict[int, str] = {}
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//MedlineCitation/PMID")
        if pmid_el is None or pmid_el.text is None:
            continue
        try:
            pmid = int(pmid_el.text.strip())
        except ValueError:
            continue
        # Concatenate every <AbstractText> (some articles use
        # structured abstracts with multiple labelled sections).
        parts: list[str] = []
        for at in article.findall(".//Abstract/AbstractText"):
            label = at.attrib.get("Label", "").strip()
            # Combine the element text + any tail of nested children
            # (e.g. <sub>) into one flattened string.
            text = "".join(at.itertext()).strip()
            if not text:
                continue
            parts.append(f"{label}: {text}" if label else text)
        if not parts:
            continue
        out[pmid] = " ".join(parts)
    return out


class PubMedRetriever(EmbeddingRetriever):
    """Retriever over fetched PubMed abstracts."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        cache_dir: Path | None = None,
    ) -> None:
        super().__init__(embedder or LocalEmbedder())
        self.cache_dir = cache_dir

    def build_index_for_pmids(
        self,
        pmids: Iterable[int],
        *,
        sleep_s: float = DEFAULT_SLEEP_S,
        http_get=_http_get,
    ) -> int:
        """Fetch + embed abstracts for the given PMIDs. Returns n indexed."""
        abstracts = fetch_abstracts(
            pmids,
            cache_dir=self.cache_dir,
            sleep_s=sleep_s,
            http_get=http_get,
        )
        items = [
            RetrievedItem(
                text=abstract,
                score=0.0,
                metadata={"pmid": pmid, "source": "pubmed"},
            )
            for pmid, abstract in sorted(abstracts.items())
        ]
        self.build_index(items)
        return len(items)
