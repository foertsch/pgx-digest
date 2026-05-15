"""Pure-Python helpers for the Streamlit demo.

No `streamlit` import here — kept separate from `streamlit_app.py` so
unit tests can exercise the data path without pulling Streamlit into the
library's dependency tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pgx_digest.bundle import Bundle, PGxFinding
from pgx_digest.drafter import (
    AnthropicProvider,
    Drafter,
    GeminiProvider,
    LLMDrafter,
    Provider,
)
from pgx_digest.pharmcat import parse_pharmcat_json
from pgx_digest.pipeline import PipelineResult
from pgx_digest.ranker import rank
from pgx_digest.verifier import Verifier


@dataclass(frozen=True)
class BundleSummary:
    """Display-ready summary of a parsed Bundle."""

    n_genes: int
    n_drugs: int
    rows: tuple[dict[str, Any], ...]


def find_fixtures(fixtures_dir: Path) -> list[Path]:
    """All PharmCAT JSON fixtures under `fixtures_dir`, sorted by name."""
    if not fixtures_dir.exists():
        return []
    return sorted(fixtures_dir.glob("*.json"))


def summarize_bundle(bundle: Bundle[PGxFinding]) -> BundleSummary:
    """Build a display-ready summary of a parsed Bundle."""
    rows: list[dict[str, Any]] = []
    total_drugs = 0
    for f in bundle.items:
        drug_names = [d.drug for d in f.affected_drugs]
        rows.append(
            {
                "Gene": f.gene,
                "Diplotype": f.diplotype,
                "Phenotype": f.phenotype,
                "Drugs": ", ".join(drug_names) if drug_names else "—",
                "# Drugs": len(drug_names),
                "Confidence": f.confidence,
            }
        )
        total_drugs += len(drug_names)
    return BundleSummary(
        n_genes=len(bundle.items),
        n_drugs=total_drugs,
        rows=tuple(rows),
    )


def find_ablation_files(eval_dir: Path) -> list[Path]:
    """Markdown ablation tables — latest version of each Ablation letter.

    Walks every `*_ablations/*.md` and keeps only the newest copy of
    each unique filename. Sorted alphabetically so Ablation A → Z
    appear in order regardless of which run produced each one.

    This handles partial runs (e.g. `--skip-model` produces an A-only
    directory) — we still surface earlier runs' B/C/etc tables.
    """
    if not eval_dir.exists():
        return []
    # Walk run dirs newest-first so the first-seen file wins per name.
    run_dirs = sorted(
        (d for d in eval_dir.glob("*_ablations") if d.is_dir()),
        reverse=True,
    )
    latest_by_name: dict[str, Path] = {}
    for run_dir in run_dirs:
        for md in run_dir.glob("*.md"):
            latest_by_name.setdefault(md.name, md)
    return sorted(latest_by_name.values(), key=lambda p: p.name)


def build_provider(name: str, api_key: str | None) -> Provider:
    """Construct an Anthropic or Gemini provider.

    If `api_key` is None or empty, the underlying SDK falls back to its
    standard env var (`ANTHROPIC_API_KEY` / `GEMINI_API_KEY`).
    """
    key = api_key or None
    if name == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=key) if key else None
        return AnthropicProvider(client=client)
    if name == "gemini":
        from google import genai

        client = genai.Client(api_key=key) if key else None
        return GeminiProvider(client=client)
    raise ValueError(f"Unknown provider: {name!r}")


def run_pipeline(
    bundle: Bundle[PGxFinding],
    provider: Provider,
) -> PipelineResult:
    """Rank → Draft → Verify with an explicit provider.

    Mirrors `pgx_digest.run()` but lets the caller inject the provider
    so the UI can switch between Anthropic and Gemini at click time.
    """
    ranked = rank(bundle)
    drafter: Drafter = LLMDrafter(provider=provider)
    draft = drafter.draft(ranked)
    verification = Verifier().verify(draft, ranked)
    return PipelineResult(bundle=ranked, draft=draft, verification=verification)


__all__ = [
    "BundleSummary",
    "build_provider",
    "find_ablation_files",
    "find_fixtures",
    "parse_pharmcat_json",
    "run_pipeline",
    "summarize_bundle",
]
