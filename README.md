# pgx-digest

[![CI](https://github.com/foertsch/pgx-digest/actions/workflows/ci.yml/badge.svg)](https://github.com/foertsch/pgx-digest/actions/workflows/ci.yml)

> Privacy-first, locally-runnable pharmacogenomic reports with a token-level
> verifier that makes LLM hallucination impossible.

This project layers a verified narrative generator on top of
[PharmCAT](https://pharmcat.org/) (Stanford/Penn). PharmCAT does diplotype
calling and produces a structured JSON report against CPIC guidelines;
pgx-digest turns that JSON into a clinical-grade, patient-facing report
where every gene, star allele, phenotype, drug recommendation, and PMID
citation is verified against the Bundle the LLM was given.

The LLM cannot invent a variant, a drug, a metabolizer status, or a study.
If it tries, the Verifier rejects the draft and the pipeline retries.

## What it demonstrates

This is an ML engineering portfolio project showing **LLM-systems
architecture** — not LLM-as-magic. Specifically:

- **Deterministic core, fenced LLM.** Diplotype calls, phenotype lookups,
  and CPIC recommendations come from PharmCAT (deterministic). The LLM's
  only job is sentence-level synthesis of the narrative.
- **Typed `Bundle[T]` with provenance per row.** Every fact carries an
  identifier; the Verifier checks token-level containment.
- **Privacy-tiered model selection.** Public demo data runs on Claude
  Haiku (cloud). Real personal genome data runs on local Ollama. The
  Drafter refuses to call a cloud model on a `LOCAL_ONLY` bundle.
- **Eval harness with ablations.** Rule-based + LLM-judge tiers.
  Verifier on/off, ranker variants, model swap.

## Status

Working end-to-end on PharmCAT example data. 125 tests passing. The
framework, multi-provider Drafter (Anthropic + Gemini) with batch and
per-card modes, Bundle redaction, Triage layer (rule-based + learned
embedding classifier), extended Verifier (typed + cross-gene prose
check), eval harness, and six ablation sweeps are all in place. Next:
RAG over PubMed + CPIC for citation-level grounding; `OllamaDrafter`
for the local-LLM privacy path; second-domain instantiation to
validate framework portability.

Run the demo (uses a committed PharmCAT JSON fixture, no Docker needed):

```bash
uv run examples/run_demo.py
```

Or run end-to-end from a real VCF (requires Docker + ~1 GB image pull on
first invocation):

```bash
uv run examples/run_with_pharmcat.py path/to/sample.vcf
```

Both load a CYP2C19 *2/*2 case, call Claude Haiku, and print verified
clinical-narrative cards (clopidogrel + voriconazole). The VCF entry
point invokes the upstream PharmCAT pipeline via the `pgkb/pharmcat`
Docker image.

Run the eval harness (rule-based + LLM-judge tiers, ~$0.01 in Haiku):

```bash
uv run examples/run_evals.py
```

Run the four ablation sweeps (zero-API for Ablation A; full sweep ~$0.04):

```bash
uv run examples/run_ablations.py --only-verifier  # zero-API smoke
uv run examples/run_ablations.py --skip-gemini    # full Anthropic sweep
```

## Architecture

```
PharmCAT JSON ──┐
                ▼
   ┌─────────────────────────────────────────────┐
   │ pgx_digest framework                          │
   │  • Bundle[PGxFinding]                         │
   │  • Ranker (deterministic + LLMRanker)         │
   │  • Triage (template / llm / skip router)      │
   │  • Drafter (LLMDrafter + Provider abstraction │
   │     for Anthropic / Gemini; TemplateDrafter   │
   │     for templatable cases; OllamaDrafter      │
   │     stub for LOCAL_ONLY)                      │
   │  • Verifier (typed containment + cross-gene   │
   │     prose check)                              │
   └─────────────────────────────────────────────┘
                ▼
        Verified report
```

## Ablations

The eval harness drives six sweeps that double as the portfolio's
empirical claims.

| Ablation | What it measures | Result |
|---|---|---|
| **A** — Verifier on/off | Detection rate on controlled hallucinations (an `AdversarialDrafter` corrupts one field per card) | **100% catch** across `swap_pmid`, `swap_drug`, `swap_phenotype`, `swap_diplotype`; 0% false positives on the faithful baseline |
| **B** — Drafter model comparison | Same fixture / same prompts on Haiku, Sonnet, Gemini; judge-scored | All three pass typed verification; judge means in 3.1–3.4 (±0.2 noise band); Sonnet is ~2× slower than Haiku for marginal quality |
| **C** — Deterministic vs LLM ranker | Ordering agreement on the multi-gene fixture | Top-1 match; LLM ranker places actionable IM ahead of Normals (better clinical order than alphabetical tie-break) |
| **D** — Triage on/off | API-call reduction via deterministic templating | **-27% Drafter input tokens** on the multi-gene fixture with no judge-quality regression |
| **E** — Batch vs per-card Drafter mode | Failure-mode comparison on real PharmCAT fixtures (up to 73 cards/bundle) | Batch silently drops 5–25% of cards on large bundles; per-card mode catches each missing pair explicitly via the Verifier. Per-card is 2.2× faster wall-clock (`ThreadPoolExecutor`) and 0.27 judge points higher when it succeeds |
| **F** — Rule-based vs learned Triage | Routing-decision agreement on 141 unique (gene, drug, recommendation) triples | **99.3% agreement.** When labels are bootstrapped from rules, the embedding classifier converges to imitate. Honest scaffolding for the active-learning loop: handle-corrected labels would let the learned model diverge usefully |

## Scope (MVP)

Five CPIC Level-A genes: **CYP2C19, CYP2C9, VKORC1, TPMT, DPYD, SLCO1B1**.
~10 drugs total. Demo on NA12878 (publicly available, has published
star-allele calls).

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> ~/Documents/GitHub/pgx-digest
cd ~/Documents/GitHub/pgx-digest
cp .env.example .env
# edit .env and add ANTHROPIC_API_KEY
uv run smoke_test.py   # verifies API wiring
```

## Why this exists

The 2025 collapse of 23andMe ended consumer-facing pharmacogenomics in
the US. Their reports were template-only and didn't carry provenance.
This project sketches what a modern, privacy-first replacement looks
like — running locally on your machine, with claims that can be audited
back to their source.

**Not a medical device. Not diagnostic. For research and educational
use only.** See [`DISCLAIMER.md`](DISCLAIMER.md).

## License

MIT — see [`LICENSE`](LICENSE).
