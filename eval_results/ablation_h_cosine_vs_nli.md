# Ablation H — Cosine similarity vs learned NLI for claim-citation grounding

**Question.** The Verifier's PubMed-grounding check uses cosine similarity over
fastembed BGE-small-en-v1.5 embeddings with a threshold of 0.35. Cosine is a
fast, principled heuristic, but it's blunt: abstracts can be topically similar
without actually supporting a specific clinical recommendation. Would a learned
biomedical natural-language-inference (NLI) classifier — specifically
`pritamdeka/PubMedBERT-MNLI-MedNLI` — discriminate genuine supporting citations
from weak ones better than cosine?

**TL;DR — no.** The NLI model is *anti-correlated* with our gold labels
(AUROC 0.355). The diagnosis is sharper than the metric: pre-trained NLI
is built for strict logical entailment, but clinical evidence grounding
requires implicit inference that small specialized models can't do.
Cosine is also weak (AUROC 0.557) — too permissive at the default threshold
to discriminate anything — but at least it's monotonic. **The Verifier keeps
the cosine baseline.** The right architecture for this task is LLM-as-judge
(already de-facto demonstrated: the gold labels in this eval were produced
by Claude Sonnet with a phenotype-aware rubric).

## Setup

**Gold set generation** (see `examples/build_nli_eval_set.py` and
`examples/label_nli_pairs_v2.py`):

1. Run the LLM Drafter (Claude Haiku 4.5, per-card mode) on 8 representative
   fixtures (4 PharmCAT-test-data + 4 1000G samples covering 5 superpopulations
   and every metabolizer class). Extract every `(card.recommendation, cited_pmid)`
   tuple. Fetch each cited PMID's PubMed abstract.
2. Dedup to 347 unique (claim, abstract) pairs.
3. Label each with Claude Sonnet 4.5 using a system prompt containing a 1-5
   evidence-strength rubric and 5 worked examples that explicitly address the
   "Normal patient + standard-dose claim + genotype-aware abstract" failure
   mode (a v1 labeling pass without these guardrails systematically called
   such pairs "contradicted"; the v2 transition matrix showed 85% of those v1
   "contradicted" labels are actually 4-5 in v2).
4. Map labels back to all 766 row-level pairs.

**Final gold-score distribution** (n=766):

| Score | n | % | Meaning |
|---|---|---|---|
| 5 | 437 | 57% | Direct, phenotype-specific support |
| 4 | 114 | 15% | Supports via principle / clinical implication |
| 3 | 91 | 12% | Right gene-drug, indirect / background only |
| 2 | 101 | 13% | On-topic but not supportive |
| 1 | 23 | 3% | Off-topic or contradicts |

Positive class for binary evaluation: `gold >= 4` → 72%.

## Results

| Metric | Cosine (BGE) | NLI (PubMedBERT) | Δ |
|---|---|---|---|
| **AUROC** (vs gold ≥ 4) | **0.557** | **0.355** | −0.202 |
| Pearson r (vs 1-5 score) | −0.042 | −0.225 | −0.183 |
| Spearman ρ (vs 1-5 score) | −0.083 | −0.171 | −0.087 |

NLI is *worse than random* (AUROC < 0.5). It's also anti-correlated:
higher entailment probability predicts *lower* gold support.

### At default thresholds (cosine=0.35, NLI=0.5)

| | Precision | Recall | F1 | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|
| Cosine | 0.72 | 1.00 | 0.84 | 551 | 215 | 0 | 0 |
| NLI | 0.61 | 0.31 | 0.41 | 170 | 108 | 107 | 381 |

Cosine's F1 of 0.84 is misleading — it achieves it by accepting *every*
pair (recall = 1.0, true negatives = 0). The threshold of 0.35 is too low
for our domain: nearly all PGx claim-abstract pairs share enough vocabulary
to clear it.

### Method disagreement

- **278 / 766 (36%) pairs**: methods agree
- **488 pairs**: cosine accepts, NLI rejects (NLI is stricter)
- **0 pairs**: cosine rejects, NLI accepts

NLI never "catches" anything cosine missed; it just rejects more.

## Why NLI fails — the diagnosis

The damning signal is in the per-score distributions:

| Gold | n | NLI median entail prob |
|---|---|---|
| 5 (strong support) | 437 | **0.20** ← low |
| 4 | 114 | **0.02** ← very low |
| 3 | 91 | 0.16 |
| 2 (weak) | 101 | **1.00** ← high (!) |
| 1 (off-topic) | 23 | 0.45 |

Strong citations get *low* entailment scores; weak citations get *high* ones.
Two diagnostic examples:

**Example 1** (gold=5, NLI=contradiction at 1.00):
- Claim: *"Based on CACNA1S status, halogenated volatile anesthetics or
  succinylcholine are relatively contraindicated in [MHS-susceptible] persons."*
- Abstract: *"The identification in a patient of 1 of the 50 variants in RYR1
  or CACNA1S genes reviewed here should lead to a presumption of malignant
  hyperthermia susceptibility..."*

The abstract straightforwardly supports the claim *clinically* — MHS
susceptibility → avoid triggering anesthetics is textbook PGx. But the NLI
model sees two surface-different sentences ("MHS susceptibility presumed"
vs "anesthetics contraindicated") and labels them as not entailed. The
model has no clinical reasoning bridge.

**Example 2** (gold=5, NLI=neutral at 0.99):
- Claim: *"Avoid capecitabine or other 5-fluorouracil prodrug-based regimens."*
- Abstract: *"The purpose of this guideline is to provide information for the
  interpretation of clinical DPYD genotype tests so that the results can be
  used to guide dosing of fluoropyrimidines..."*

The abstract is the CPIC DPYD guideline's *introduction*. The actual
recommendation (avoid fluoropyrimidines in DPYD PMs) lives inside the
guideline body. NLI can't reason that "we present this guideline" implies
"this guideline contains the recommendation in the claim."

## The fundamental mismatch

MNLI and MedNLI train models on **strict logical entailment**: does premise X
*literally* imply hypothesis Y? Clinical citation grounding asks a different
question: does this abstract *justify* this recommendation as part of the
broader evidence base, allowing for implicit clinical inference (phenotype
mapping, "guideline endorses what guideline contains," CPIC classification
logic, etc.)?

A capable LLM can do this — Sonnet 4.5 produced sensible gold labels in this
very eval. A 110M-parameter specialized NLI model cannot.

## What we ship

1. **Verifier default stays as cosine grounding.** Raising the threshold to
   ~0.65 is a follow-up worth scoping (the cosine distribution shows weak but
   real signal; the default 0.35 is too permissive for our domain).
2. **`pgx_digest/verifier_nli.py` ships behind the `nli` optional extra**
   so anyone curious can reproduce the result. Not wired into the Verifier
   by default.
3. **The right next step is LLM-as-judge grounding** — Claude Haiku 4.5 with
   the phenotype-aware system prompt we developed here costs ~$0.001 per claim
   and demonstrably handles the implicit-inference cases. Scoped as a separate
   follow-up.

This is a useful negative result, not a failed experiment. We set up a
rigorous eval, found a clear answer, diagnosed the *why*, and identified the
concrete next architecture.

## Reproducing

```bash
uv sync --extra nli   # pulls torch + transformers (~1.5 GB; one-time)
uv run python examples/build_nli_eval_set.py     # regenerate gold set (~$0.30 + 3 min)
uv run python examples/label_nli_pairs_v2.py     # re-label with v2 rubric (~$0.76 + 3 min)
uv run python examples/ablation_h_cosine_vs_nli.py  # eval (zero API, ~1 min)
```
