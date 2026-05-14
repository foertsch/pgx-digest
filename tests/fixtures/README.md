# Test Fixtures

Hand-built VCF fixtures for development and unit tests. **Not for clinical
use.** These exercise the framework's input path without requiring network
access, downloads, or PharmCAT to run.

## What's here

| File | Gene | Diplotype | Phenotype | Purpose |
|---|---|---|---|---|
| `cyp2c19_star1_star2_het.vcf` | CYP2C19 | `*1/*2` | Intermediate Metabolizer | Clopidogrel use case |
| `cyp2c19_star1_star17_het.vcf` | CYP2C19 | `*1/*17` | Rapid Metabolizer | NA12878-shaped (matches GeT-RM published call) |
| `tpmt_star1_star1.vcf` | TPMT | `*1/*1` | Normal Metabolizer | Negative case — no actionable variants |
| `dpyd_star1_star2a_het.vcf` | DPYD | `*1/*2A` | Intermediate Metabolizer | 5-FU toxicity risk case |

All fixtures use **GRCh38** with `chr`-prefixed chromosomes — matching the
modern Nebula Genomics deliverable format.

## What these fixtures are NOT

- **Not biologically validated.** Variant coordinates here are approximations
  based on common references and may not match PharmCAT's authoritative
  allele definition files byte-for-byte. Before running real pipeline
  integration tests, regenerate these against PharmCAT's allele definitions:
  https://github.com/PharmGKB/PharmCAT/tree/main/src/main/resources/org/pharmgkb/pharmcat/definition
- **Not full WGS VCFs.** Each fixture contains only the positions PharmCAT
  needs to call the named diplotype. A real Nebula or 1000 Genomes VCF
  would have 3–5 million additional variant rows.
- **Not for the verifier.** Verifier unit tests (`tests/test_verifier.py`)
  use pure-Python `PGxFinding` fixtures, not VCF files. These VCFs are
  for the upstream PharmCAT integration once `pgx_digest/inputs/vcf.py`
  is wired up.

## When to upgrade

Replace these with real public-domain data when you're ready to validate
end-to-end:

1. **PharmCAT test data** — clone https://github.com/PharmGKB/PharmCAT and
   copy `src/test/resources/.../*.vcf`. These are the test cases PharmCAT
   itself validates against, including the gene-by-gene reference materials.
2. **NA12878 from 1000 Genomes** — full WGS VCF on GRCh38. Subset to PGx
   positions via PharmCAT's preprocessor. The GeT-RM consortium has
   published reference diplotype calls for this sample, so it doubles as
   an integration-test ground truth.

Both are deferred to a later phase per the project's tier-1-only test data
scope.

## Adding a new fixture

1. Pick a star allele combination from the PharmCAT allele definition
   for the gene you want to exercise.
2. Write a minimal VCF containing only the defining variants for that
   diplotype.
3. Use `chr`-prefixed chromosomes and GRCh38 coordinates.
4. Add a header comment block with the source/rationale.
5. Add a row to the table above.
