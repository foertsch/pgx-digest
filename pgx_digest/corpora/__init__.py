"""External corpora for RAG grounding.

- `pubmed`: PubMed abstracts fetched via NCBI eutils, used as
  citation-level grounding (the Verifier can check that a cited PMID's
  abstract actually supports the card's claims).
- `cpic`: CPIC guideline recommendations fetched via the CPIC API,
  used as authoritative prompt enrichment for the Drafter.

Both modules expose `Retriever` subclasses that share the embedding-
index machinery in `pgx_digest.retriever`.
"""

from pgx_digest.corpora.cpic import CPICRetriever
from pgx_digest.corpora.pubmed import PubMedRetriever

__all__ = ["CPICRetriever", "PubMedRetriever"]
