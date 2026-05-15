# /// script
# requires-python = ">=3.11"
# ///
"""Re-label the NLI eval pairs with a phenotype-aware system prompt.

Replaces the v1 labeling pass (categorical {supported/partial/contradicted/
unrelated}) with:

  1. A SYSTEM prompt carrying the rubric + 5 worked examples (Anthropic
     prompt cache keeps repeated calls cheap — 90% discount on cache hits).
  2. A 1-5 score (continuous evidence-strength).
  3. Explicit patient phenotype context in the user message — the v1 prompt
     only saw the claim text, which caused the systematic mis-labeling
     of "Normal patient + standard-dose claim + genotype-aware abstract"
     as "contradicted" instead of supportive.
  4. Dedup by (claim, pmid) before labeling — the same pair appears in
     multiple fixtures, no point paying twice.
  5. ThreadPoolExecutor parallelism (8 workers) — ~2 min wall time
     instead of ~17 sequential.

Inputs/outputs (both under ``eval_data/``):
  - reads  ``nli_pairs.jsonl``  (unlabeled, ~766 rows)
  - writes ``nli_gold.jsonl``    (full labels mapped back to all rows)
  - backs up the v1 labels to   ``nli_gold_v1.jsonl`` if present
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env", override=True)
load_dotenv(REPO_ROOT.parent.parent.parent / ".env", override=True)

OUT_DIR = REPO_ROOT / "eval_data"
LABELER_MODEL = "claude-sonnet-4-5"


# ---------------------------------------------------------------------------
# System prompt: rubric + worked examples (gets prompt-cached)
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """You are a careful pharmacogenomics scientist scoring how strongly a PubMed abstract supports a specific clinical recommendation for a specific patient.

# Score rubric (1-5)

**5 — Direct, phenotype-specific support.** The abstract explicitly endorses (or its embedded guideline endorses) the recommendation given to a patient with this exact phenotype.

**4 — Supports the claim via published evidence on the same gene-drug pair for the relevant phenotype context.** The abstract supports the claim by direct statement OR by clear implication (e.g., if the abstract says "drug X is contraindicated in DEFICIENT patients", that supports a claim of "no contraindication in NORMAL patients" by exclusion).

**3 — Right gene-drug, but doesn't speak to this phenotype.** Background or mechanism paper that establishes the relationship in general terms; the specific claim isn't directly addressed.

**2 — On-topic but offers no support for the specific claim.** Same gene, different drug; or methodology paper; or focuses on a sub-question unrelated to the claim.

**1 — Off-topic OR actively contradicts the claim.** Different gene, different drug class, or directly argues against the specific recommendation. (We merge "unrelated" and "genuine contradiction" because both are equally useless as supporting evidence.)

# Critical interpretation rule

**The patient's phenotype determines what counts as support.** Many abstracts describe genotype-stratified dosing across the full spectrum (Normal/IM/PM/UM, or B/Deficient). For a Normal-Metabolizer patient with a "standard dose" recommendation, an abstract that lays out the full genotype-based dosing scheme — including "standard dose for wild-type" — is **strong support (score 5)**, not a contradiction. Do not penalize the abstract for also discussing other phenotypes.

# Worked examples

## Example A: Normal Metabolizer + standard dose + genotype-aware abstract → 5
- Patient: TPMT *1/*1, Normal Metabolizer
- Claim: "Initiate therapy with standard starting dose of mercaptopurine (75 mg/m2/day for malignancy)..."
- Abstract: "TPMT activity exhibits monogenic co-dominant inheritance... homozygous TPMT-deficient patients experience severe myelosuppression... homozygous wild-type individuals show lower active thioguanine nucleotides and less myelosuppression. We provide dosing recommendations [genotype-stratified] for thiopurines."
- Score: 5
- Reasoning: The abstract describes a CPIC guideline whose recommendation for wild-type (Normal Metabolizer) patients is standard dosing, which is exactly what the claim states. The fact that the abstract also covers other phenotypes does not diminish its direct support for this Normal-Metabolizer recommendation.

## Example B: G6PD Normal + "no reason to avoid" + abstract about deficient-patient contraindication → 4
- Patient: G6PD B/B (reference), Normal
- Claim: "No reason to avoid rasburicase based on G6PD status."
- Abstract: "Rasburicase is contraindicated in G6PD-DEFICIENT patients due to risk of acute hemolytic anemia."
- Score: 4
- Reasoning: The abstract identifies G6PD-deficient patients as the at-risk population; by implication, G6PD-Normal patients are not at increased risk, supporting the claim.

## Example C: CYP2C19 PM + alternative-drug recommendation + CPIC guideline abstract → 5
- Patient: CYP2C19 *2/*2, Poor Metabolizer
- Claim: "Consider an alternative antiplatelet (e.g., prasugrel, ticagrelor) not metabolized by CYP2C19; if clopidogrel is used, antiplatelet platelet function monitoring is recommended."
- Abstract: "We provide dosing recommendations for clopidogrel based on CYP2C19 genotype. For Poor Metabolizers, alternative antiplatelets (e.g., prasugrel, ticagrelor) are recommended."
- Score: 5

## Example D: Same gene-drug, principle only → 4
- Patient: CYP2C19 *2/*2, Poor Metabolizer
- Claim: "Consider 50% reduction in clopidogrel maintenance dose for CYP2C19 PMs."
- Abstract: "Clopidogrel is a prodrug activated by CYP2C19. Loss-of-function variants reduce active metabolite formation by 30-50%. We review the pharmacogenetic mechanism."
- Score: 4
- Reasoning: Establishes the mechanistic basis for dose reduction but doesn't state the specific 50% figure or formal recommendation.

## Example E: Off-topic / wrong drug class → 2
- Patient: CYP2C19 *2/*2, PM
- Claim about clopidogrel dosing
- Abstract: "We summarize CYP2C19 effects on PROTON PUMP INHIBITORS (omeprazole, pantoprazole)... no clopidogrel data."
- Score: 2

# Output format

Respond with ONLY a JSON object:
```json
{"score": <integer 1-5>, "reasoning": "<one sentence>"}
```

Do not include any text outside the JSON. Do not use markdown code fences."""


USER_TEMPLATE = """Patient context:
- Gene: {gene}
- Diplotype: {diplotype}
- Phenotype: {phenotype}

Claim being verified:
{claim}

PubMed abstract (PMID {pmid}):
{abstract}

Score (1-5) and one-sentence reasoning."""


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------


def _label_one(client, pair: dict[str, Any]) -> dict[str, Any]:
    """Send one (claim, abstract, phenotype) tuple to Sonnet. Returns merged dict."""
    user_msg = USER_TEMPLATE.format(
        gene=pair["gene"],
        diplotype=pair.get("diplotype", "?"),
        phenotype=pair.get("phenotype", "?"),
        claim=pair["claim"],
        pmid=pair["pmid"],
        abstract=pair["abstract"][:3500],  # cap context per call
    )
    try:
        resp = client.messages.create(
            model=LABELER_MODEL,
            max_tokens=300,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        parsed = json.loads(raw)
        return {
            **pair,
            "score": int(parsed["score"]),
            "reasoning": parsed.get("reasoning", ""),
            "_usage": {
                "cache_creation": getattr(resp.usage, "cache_creation_input_tokens", 0),
                "cache_read": getattr(resp.usage, "cache_read_input_tokens", 0),
                "input": resp.usage.input_tokens,
                "output": resp.usage.output_tokens,
            },
        }
    except Exception as e:
        return {**pair, "score": -1, "reasoning": f"ERROR: {e}"}


def main() -> int:
    pairs_path = OUT_DIR / "nli_pairs.jsonl"
    gold_path = OUT_DIR / "nli_gold.jsonl"
    v1_backup_path = OUT_DIR / "nli_gold_v1.jsonl"

    if not pairs_path.exists():
        sys.exit(f"missing {pairs_path}; run build_nli_eval_set.py --extract-only first")

    # Back up v1 labels if present
    if gold_path.exists() and not v1_backup_path.exists():
        gold_path.rename(v1_backup_path)
        print(f"  backed up v1 labels to {v1_backup_path.name}")

    rows = [json.loads(l) for l in pairs_path.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(rows)} (claim, pmid) pairs from {pairs_path.name}")

    # Dedup by (claim text, pmid) — same pair appears across fixtures
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for r in rows:
        key = (r["claim"], int(r["pmid"]))
        by_key.setdefault(key, r)
    unique = list(by_key.values())
    print(f"  dedup → {len(unique)} unique pairs to label\n")

    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Warm the cache with one sequential call before going parallel,
    # so the workers all benefit from cache hits.
    print("Warming prompt cache with one sequential call...")
    first = _label_one(client, unique[0])
    cache_creation = first.get("_usage", {}).get("cache_creation", 0)
    print(f"  cache_creation_tokens={cache_creation} (this is the rubric)\n")

    print(f"Labeling {len(unique) - 1} remaining pairs in parallel (8 workers)...")
    labeled_by_key = {(unique[0]["claim"], unique[0]["pmid"]): first}
    t0 = time.time()
    completed = 1
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {
            ex.submit(_label_one, client, p): (p["claim"], p["pmid"])
            for p in unique[1:]
        }
        for fut in as_completed(futures):
            key = futures[fut]
            labeled_by_key[key] = fut.result()
            completed += 1
            if completed % 25 == 0 or completed == len(unique):
                print(f"  {completed}/{len(unique)} ({time.time()-t0:.0f}s)")

    # Map labels back to all 766 rows
    print("\nMapping labels back to full row set...")
    full_labeled = []
    for r in rows:
        key = (r["claim"], int(r["pmid"]))
        lbl = labeled_by_key[key]
        full_labeled.append({
            **r,
            "score": lbl["score"],
            "reasoning": lbl["reasoning"],
        })

    gold_path.write_text("\n".join(json.dumps(r) for r in full_labeled) + "\n")
    print(f"  wrote {gold_path} ({len(full_labeled)} rows)\n")

    # Stats
    from collections import Counter
    scores = Counter(r["score"] for r in full_labeled)
    print("Score distribution (all rows):")
    for s in [5, 4, 3, 2, 1, -1]:
        n = scores.get(s, 0)
        if n:
            pct = 100 * n / len(full_labeled)
            print(f"  {s if s != -1 else 'ERROR':>5} {n:>4}  ({pct:.0f}%)")

    # Cache effectiveness
    total_cache_create = sum(
        l.get("_usage", {}).get("cache_creation", 0) for l in labeled_by_key.values()
    )
    total_cache_read = sum(
        l.get("_usage", {}).get("cache_read", 0) for l in labeled_by_key.values()
    )
    total_input = sum(
        l.get("_usage", {}).get("input", 0) for l in labeled_by_key.values()
    )
    print(f"\nCache stats (across {len(labeled_by_key)} unique-pair calls):")
    print(f"  cache_creation_tokens (paid 1.25×): {total_cache_create:>8}")
    print(f"  cache_read_tokens     (paid 0.10×): {total_cache_read:>8}")
    print(f"  non-cached input_tokens:            {total_input:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
