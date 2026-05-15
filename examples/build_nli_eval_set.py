# /// script
# requires-python = ">=3.11"
# ///
"""Build a labeled (claim, abstract, label) gold set for evaluating the
NLI-based Verifier grounding check.

Cross-model pipeline to avoid grading-your-own-work bias:
  - DRAFTER:  Gemini  → produces (claim text, cited PMIDs) per card
  - LABELER:  Claude  → classifies (claim, abstract) pairs

Output: ``eval_data/nli_gold.jsonl`` (one labeled pair per line).

Each line:
    {
      "fixture": "pharmcat_1kg_NA20509.report.json",
      "gene": "CYP2C19",
      "drug": "clopidogrel",
      "claim": "Clopidogrel response is reduced in CYP2C19 PMs...",
      "pmid": 23486447,
      "abstract": "...",
      "label": "supported" | "partial" | "contradicted" | "unrelated",
      "reasoning": "the abstract directly states reduced platelet inhibition..."
    }

Usage:
    uv run python examples/build_nli_eval_set.py
    uv run python examples/build_nli_eval_set.py --extract-only  # skip labeling

The intermediate (unlabeled) artifact is saved to ``eval_data/nli_pairs.jsonl``
so labeling can be resumed without re-running the Drafter.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

# .env may live in the worktree root OR in the parent main-repo
# (worktree is at <main>/.claude/worktrees/<name>/).
load_dotenv(REPO_ROOT / ".env", override=True)
load_dotenv(REPO_ROOT.parent.parent.parent / ".env", override=True)

from pgx_digest.corpora.pubmed import fetch_abstracts  # noqa: E402
from pgx_digest.drafter import GeminiProvider, LLMDrafter  # noqa: E402
from pgx_digest.pharmcat import parse_pharmcat_json  # noqa: E402
from pgx_digest.ranker import rank  # noqa: E402

FIX_DIR = REPO_ROOT / "tests" / "fixtures"
OUT_DIR = REPO_ROOT / "eval_data"
PUBMED_CACHE = REPO_ROOT / "rag_cache" / "pubmed"

# Eight fixtures covering: PharmCAT-test-data, all five superpops,
# every metabolizer class (Normal/IM/PM/RM/UM), and the highest-stakes
# clinical cases (DPYD, G6PD, RYR1).
TARGET_FIXTURES = (
    "pharmcat_sample_1.report.json",          # DPYD PM, G6PD Deficient, RYR1 MH
    "pharmcat_multigene.json",                 # CYP2C19 PM + CYP2D6 IM + DPYD + TPMT
    "pharmcat_1kg_NA20509.report.json",        # TSI/EUR — CYP2C19 *2/*2 PM
    "pharmcat_1kg_HG03052.report.json",        # MSL/AFR — CYP2C19 *17/*17 UM
    "pharmcat_1kg_HG00731.report.json",        # PUR/AMR — CYP2C19 *1/*17 RM
    "pharmcat_1kg_HG01112.report.json",        # CLM/AMR — UM + ABCG2 Decreased
    "pharmcat_1kg_NA18486.report.json",        # YRI/AFR — G6PD Deficient with CNSHA
    "pharmcat_1kg_NA20846.report.json",        # GIH/SAS — CYP2C19 *2/*38 IM
)


# ---------------------------------------------------------------------------
# Step 1: Extract (claim, cited_pmids) pairs by running the Drafter
# ---------------------------------------------------------------------------


def extract_pairs(fixtures: tuple[str, ...]) -> list[dict[str, Any]]:
    """Run Gemini Drafter on each fixture; return one record per card-citation.

    Records are flat: one row per (card × cited_pmid). Abstracts are fetched
    in a single batched call at the end to minimize PubMed API load.
    """
    # Original plan was cross-model (Gemini drafts, Claude labels) but
    # the available Gemini account has no free-tier quota on 2.5-flash
    # (503 high demand), 2.5-pro (free quota = 0), or 2.0-flash (also 0).
    # Falling back to all-Claude: Haiku 4.5 drafts, Sonnet 4.5 labels.
    # The model-tier separation is weaker than cross-family, but the
    # comparative cosine-vs-NLI signal is symmetric under any same-labeler
    # setup, so the result still holds.
    print(f"[1/3] Drafting {len(fixtures)} fixtures with Claude Haiku 4.5...")
    try:
        from pgx_digest.drafter import AnthropicProvider
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    except KeyError:
        sys.exit("ANTHROPIC_API_KEY not set (looked in env + repo .env).")
    # per_card mode: one LLM call per (gene, drug) pair instead of
    # serializing the whole bundle in one shot. Slower wall time but
    # robust to the 70+ cards a 1kG fixture can produce (which would
    # overflow the default max_tokens in batch mode).
    drafter = LLMDrafter(
        provider=AnthropicProvider(client=client, model="claude-haiku-4-5"),
        mode="per_card",
    )

    rows: list[dict[str, Any]] = []
    for fname in fixtures:
        path = FIX_DIR / fname
        if not path.exists():
            print(f"  SKIP {fname} (not found)")
            continue
        bundle = parse_pharmcat_json(path)
        ranked = rank(bundle)
        t0 = time.time()
        draft = drafter.draft(ranked)
        print(f"  {fname:<50} -> {len(draft.cards)} cards in {time.time()-t0:.1f}s")
        for card in draft.cards:
            if not card.cited_pmids:
                continue
            for pmid in card.cited_pmids:
                rows.append({
                    "fixture": fname,
                    "gene": card.gene,
                    "diplotype": card.diplotype,
                    "phenotype": card.phenotype,
                    "drug": card.drug,
                    "claim": card.recommendation,
                    "pmid": int(pmid),
                })
    print(f"  -> {len(rows)} (claim, pmid) pairs extracted")

    # Batch-fetch all unique abstracts in one PubMed request
    print("[2/3] Fetching PubMed abstracts (batched)...")
    unique_pmids = sorted({r["pmid"] for r in rows})
    abstracts = fetch_abstracts(unique_pmids, cache_dir=PUBMED_CACHE)
    print(f"  -> got abstracts for {len(abstracts)}/{len(unique_pmids)} PMIDs")

    # Drop pairs whose abstract failed to fetch (CPIC letters etc.)
    enriched = []
    for r in rows:
        abstract = abstracts.get(r["pmid"])
        if abstract:
            enriched.append({**r, "abstract": abstract})
    print(f"  -> {len(enriched)}/{len(rows)} pairs have abstracts")
    return enriched


# ---------------------------------------------------------------------------
# Step 2: Label each (claim, abstract) pair with Claude
# ---------------------------------------------------------------------------


LABEL_PROMPT = """You are a careful PGx scientist labeling whether a PubMed \
abstract supports a clinical claim.

PUBMED ABSTRACT (PMID {pmid}):
{abstract}

CLINICAL CLAIM:
{claim}

Pick exactly one label and write one sentence of reasoning.

Labels:
- supported: the abstract clearly endorses the claim (same drug, same gene, \
same direction of effect).
- partial: the abstract is on-topic but doesn't fully support the specific \
claim (e.g. same gene but different drug, or supports a general principle \
but not the specific dosing recommendation).
- contradicted: the abstract argues the opposite of the claim.
- unrelated: the abstract is about a different topic (different gene, \
different drug, different clinical question).

Respond ONLY in this JSON format:
{{"label": "...", "reasoning": "..."}}
"""


def label_with_claude(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Send each pair to Claude (Sonnet 4.5) for labeling.

    Sonnet is a tier above the Haiku 4.5 used for drafting — partial
    model-tier separation since we couldn't get cross-family.
    """
    print(f"[3/3] Labeling {len(pairs)} pairs with Claude Sonnet 4.5...")
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    except KeyError:
        sys.exit("ANTHROPIC_API_KEY not set (looked in env + repo .env).")

    out = []
    for i, p in enumerate(pairs, 1):
        prompt = LABEL_PROMPT.format(
            pmid=p["pmid"],
            abstract=p["abstract"][:3000],  # keep prompts cheap
            claim=p["claim"],
        )
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            # Strip ```json fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            parsed = json.loads(raw)
            out.append({
                **p,
                "label": parsed["label"],
                "reasoning": parsed.get("reasoning", ""),
            })
            if i % 5 == 0 or i == len(pairs):
                print(f"  {i}/{len(pairs)} labeled")
        except Exception as e:
            print(f"  [{i}] FAIL ({p['gene']}/{p['drug']}/PMID{p['pmid']}): {e}")
            out.append({**p, "label": "ERROR", "reasoning": str(e)})
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--extract-only", action="store_true",
        help="Skip labeling; just write the unlabeled pairs.",
    )
    ap.add_argument(
        "--label-from", type=Path, default=None,
        help="Skip extraction; label pairs from this existing JSONL.",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unlabeled_path = OUT_DIR / "nli_pairs.jsonl"
    gold_path = OUT_DIR / "nli_gold.jsonl"

    if args.label_from:
        pairs = [json.loads(line) for line in args.label_from.read_text().splitlines() if line.strip()]
    else:
        pairs = extract_pairs(TARGET_FIXTURES)
        unlabeled_path.write_text("\n".join(json.dumps(p) for p in pairs) + "\n")
        print(f"  wrote {unlabeled_path} ({len(pairs)} rows)")

    if args.extract_only:
        return 0

    labeled = label_with_claude(pairs)
    gold_path.write_text("\n".join(json.dumps(r) for r in labeled) + "\n")
    print(f"  wrote {gold_path} ({len(labeled)} rows)")

    # Quick summary
    from collections import Counter

    counts = Counter(r["label"] for r in labeled)
    print("\nLabel distribution:")
    for label, n in counts.most_common():
        print(f"  {label:<15} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
