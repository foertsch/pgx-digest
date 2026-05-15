# /// script
# requires-python = ">=3.11"
# ///
"""Ablation H: cosine-similarity vs NLI claim-citation grounding.

Compares the two grounding methods against the v2 gold-labeled set
(``eval_data/nli_gold.jsonl``):

  - **cosine**: cosine similarity over fastembed BGE-small-en-v1.5
    embeddings (the current Verifier default; threshold 0.35).
  - **NLI**: PubMedBERT-MNLI-MedNLI entailment probability
    (threshold 0.5).

Both methods produce a scalar score per (claim, abstract) pair, so we
report:

  - **AUROC** treating gold score >= 4 as the positive class (good
    citation; should be accepted as grounding).
  - **Spearman + Pearson correlation** with the 1-5 gold score
    (continuous evidence-strength).
  - **Threshold-level confusion matrix** at each method's default
    threshold, and a quick scan of disagreement cases.

Output:
  - ``eval_data/ablation_h_per_pair_scores.jsonl``  (per-pair scores)
  - ``eval_results/ablation_h_cosine_vs_nli.md``    (summary + verdict)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pgx_digest.embeddings import LocalEmbedder  # noqa: E402
from pgx_digest.verifier_nli import NLIGrounder  # noqa: E402

EVAL_DATA = REPO_ROOT / "eval_data"
EVAL_RESULTS = REPO_ROOT / "eval_results"
EVAL_RESULTS.mkdir(parents=True, exist_ok=True)

# Default thresholds matching production defaults
COSINE_THRESHOLD = 0.35
NLI_THRESHOLD = 0.5
POSITIVE_GOLD = 4  # gold score >= this → positive class


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> int:
    gold_path = EVAL_DATA / "nli_gold.jsonl"
    if not gold_path.exists():
        sys.exit(f"missing {gold_path}; run label_nli_pairs_v2.py first")

    rows = [json.loads(l) for l in gold_path.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(rows)} labeled pairs")

    # ---- 1. Cosine: embed once for unique texts, dot-product per pair ----
    print("\n[1/2] Cosine via fastembed BGE-small-en-v1.5...")
    embedder = LocalEmbedder()
    unique_claims = sorted({r["claim"] for r in rows})
    unique_abstracts = sorted({r["abstract"] for r in rows})
    print(f"  embedding {len(unique_claims)} unique claims + {len(unique_abstracts)} unique abstracts")
    t0 = time.time()
    claim_emb = dict(zip(unique_claims, embedder.embed(unique_claims), strict=True))
    abs_emb = dict(zip(unique_abstracts, embedder.embed(unique_abstracts), strict=True))
    print(f"  embedding done in {time.time()-t0:.1f}s")

    for r in rows:
        r["cosine"] = _cosine_sim(claim_emb[r["claim"]], abs_emb[r["abstract"]])

    # ---- 2. NLI: PubMedBERT, dedup unique (claim, abstract) pairs ----
    print(f"\n[2/2] NLI via PubMedBERT-MNLI-MedNLI...")
    grounder = NLIGrounder()
    unique_pairs = sorted({(r["claim"], r["abstract"]) for r in rows})
    print(f"  scoring {len(unique_pairs)} unique pairs (~150ms each on CPU)...")
    t0 = time.time()
    nli_by_pair: dict[tuple[str, str], float] = {}
    for i, (claim, abstract) in enumerate(unique_pairs, 1):
        result = grounder.score(premise=abstract, hypothesis=claim)
        nli_by_pair[(claim, abstract)] = result.probs["entailment"]
        if i % 50 == 0 or i == len(unique_pairs):
            print(f"  {i}/{len(unique_pairs)} ({time.time()-t0:.0f}s)")

    for r in rows:
        r["nli_entail"] = nli_by_pair[(r["claim"], r["abstract"])]

    # Save per-pair scores (dropping the abstract to keep file small)
    out_path = EVAL_DATA / "ablation_h_per_pair_scores.jsonl"
    out_rows = [
        {k: v for k, v in r.items() if k != "abstract"} for r in rows
    ]
    out_path.write_text("\n".join(json.dumps(r) for r in out_rows) + "\n")
    print(f"\nWrote per-pair scores to {out_path}")

    # ---- 3. Metrics ----
    gold = np.array([r["score"] for r in rows])
    cos = np.array([r["cosine"] for r in rows])
    nli = np.array([r["nli_entail"] for r in rows])

    pos = (gold >= POSITIVE_GOLD).astype(int)

    # AUROC
    from sklearn.metrics import roc_auc_score, precision_recall_curve

    auroc_cos = roc_auc_score(pos, cos)
    auroc_nli = roc_auc_score(pos, nli)

    # Pearson + Spearman
    def _spearman(x: np.ndarray, y: np.ndarray) -> float:
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])

    pearson_cos = float(np.corrcoef(gold, cos)[0, 1])
    pearson_nli = float(np.corrcoef(gold, nli)[0, 1])
    spearman_cos = _spearman(gold, cos)
    spearman_nli = _spearman(gold, nli)

    # Confusion @ default thresholds
    def _cm(scores: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
        pred = (scores >= threshold).astype(int)
        tp = int(((pred == 1) & (pos == 1)).sum())
        fp = int(((pred == 1) & (pos == 0)).sum())
        tn = int(((pred == 0) & (pos == 0)).sum())
        fn = int(((pred == 0) & (pos == 1)).sum())
        return tp, fp, tn, fn

    cos_tp, cos_fp, cos_tn, cos_fn = _cm(cos, COSINE_THRESHOLD)
    nli_tp, nli_fp, nli_tn, nli_fn = _cm(nli, NLI_THRESHOLD)

    def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    cos_p, cos_r, cos_f = _prf(cos_tp, cos_fp, cos_fn)
    nli_p, nli_r, nli_f = _prf(nli_tp, nli_fp, nli_fn)

    # Disagreement analysis: cases where the two methods disagree at their thresholds
    cos_pred = cos >= COSINE_THRESHOLD
    nli_pred = nli >= NLI_THRESHOLD
    cos_yes_nli_no = (cos_pred & ~nli_pred).sum()
    nli_yes_cos_no = (~cos_pred & nli_pred).sum()
    agree = (cos_pred == nli_pred).sum()

    # ---- 4. Write summary markdown ----
    md = [
        "# Ablation H — Cosine similarity vs NLI grounding",
        "",
        "**Question**: Does a learned NLI claim-citation matcher (PubMedBERT-MNLI-MedNLI)",
        "outperform the existing cosine-similarity grounding check (fastembed BGE-small)",
        "at distinguishing genuine supporting citations from weak ones?",
        "",
        f"**Eval set**: {len(rows)} (claim, abstract, gold_score) triples extracted from",
        "running the LLM Drafter on 8 representative fixtures (4 PharmCAT-test-data +",
        "4 1000G), with gold scores assigned by Claude Sonnet 4.5 using a phenotype-aware",
        "rubric (see `eval_data/nli_gold.jsonl`). Positive class is `gold_score >= 4`",
        f"(strong support); positive rate = {pos.mean():.0%}.",
        "",
        "## Method comparison",
        "",
        "| Metric | Cosine (BGE) | NLI (PubMedBERT) | Δ |",
        "|---|---|---|---|",
        f"| **AUROC** (vs gold >= 4) | **{auroc_cos:.3f}** | **{auroc_nli:.3f}** | {auroc_nli-auroc_cos:+.3f} |",
        f"| Pearson r (vs 1-5 score) | {pearson_cos:.3f} | {pearson_nli:.3f} | {pearson_nli-pearson_cos:+.3f} |",
        f"| Spearman ρ (vs 1-5 score) | {spearman_cos:.3f} | {spearman_nli:.3f} | {spearman_nli-spearman_cos:+.3f} |",
        "",
        f"## At default thresholds (cosine={COSINE_THRESHOLD}, NLI={NLI_THRESHOLD})",
        "",
        "| | Precision | Recall | F1 | TP | FP | TN | FN |",
        "|---|---|---|---|---|---|---|---|",
        f"| Cosine | {cos_p:.2f} | {cos_r:.2f} | {cos_f:.2f} | {cos_tp} | {cos_fp} | {cos_tn} | {cos_fn} |",
        f"| NLI | {nli_p:.2f} | {nli_r:.2f} | {nli_f:.2f} | {nli_tp} | {nli_fp} | {nli_tn} | {nli_fn} |",
        "",
        "## Method disagreement",
        "",
        f"- {agree} / {len(rows)} ({100*agree/len(rows):.0f}%) pairs: methods agree",
        f"- {cos_yes_nli_no} pairs: cosine accepts, NLI rejects (NLI is stricter)",
        f"- {nli_yes_cos_no} pairs: cosine rejects, NLI accepts (NLI catches what cosine missed)",
        "",
        "## Reproducing",
        "",
        "```bash",
        "uv sync --extra nli  # pulls torch + transformers (~1.5 GB; one-time)",
        "uv run python examples/build_nli_eval_set.py  # regenerates the gold set",
        "uv run python examples/label_nli_pairs_v2.py  # re-labels with v2 rubric",
        "uv run python examples/ablation_h_cosine_vs_nli.py  # this script",
        "```",
        "",
    ]
    md_path = EVAL_RESULTS / "ablation_h_cosine_vs_nli.md"
    md_path.write_text("\n".join(md))
    print(f"Wrote summary to {md_path}")

    # ---- 5. Print to stdout ----
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Positive rate (gold >= {POSITIVE_GOLD}): {pos.mean():.0%}\n")
    print(f"{'Metric':<25} {'Cosine':>10} {'NLI':>10} {'Δ':>10}")
    print(f"{'-' * 60}")
    print(f"{'AUROC':<25} {auroc_cos:>10.3f} {auroc_nli:>10.3f} {auroc_nli-auroc_cos:>+10.3f}")
    print(f"{'Pearson r':<25} {pearson_cos:>10.3f} {pearson_nli:>10.3f} {pearson_nli-pearson_cos:>+10.3f}")
    print(f"{'Spearman ρ':<25} {spearman_cos:>10.3f} {spearman_nli:>10.3f} {spearman_nli-spearman_cos:>+10.3f}")
    print(f"\nAt thresholds cos={COSINE_THRESHOLD}, nli={NLI_THRESHOLD}:")
    print(f"{'F1':<25} {cos_f:>10.2f} {nli_f:>10.2f} {nli_f-cos_f:>+10.2f}")
    print(f"{'Precision':<25} {cos_p:>10.2f} {nli_p:>10.2f}")
    print(f"{'Recall':<25} {cos_r:>10.2f} {nli_r:>10.2f}")
    print(f"\nAgree: {agree}/{len(rows)} ({100*agree/len(rows):.0f}%)")
    print(f"Cos-yes / NLI-no: {cos_yes_nli_no}")
    print(f"NLI-yes / Cos-no: {nli_yes_cos_no}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
