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

Working end-to-end on PharmCAT example data. Framework, PharmCAT JSON
parser, structured-output Drafter (Claude Haiku 4.5), and Verifier all
in place. 17 tests passing. PharmCAT-to-VCF integration, Ollama local
Drafter, and the full eval harness are the next pieces.

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

## Architecture

```
PharmCAT JSON ──┐
                ▼
   ┌─────────────────────────────────┐
   │ pgx_digest framework             │
   │  • Bundle[PGxFinding]            │
   │  • Ranker (deterministic)        │
   │  • Drafter (Haiku / Ollama)      │
   │  • Verifier (typed containment)  │
   └─────────────────────────────────┘
                ▼
        Verified report
```

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
