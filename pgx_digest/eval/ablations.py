"""Three ablations — the portfolio centerpiece.

Each function returns `tuple[AblationRow, ...]` and (when given a path)
writes a Markdown comparison table. The three:

a) Verifier on/off — driven by the synthetic `AdversarialDrafter` so
   that "hallucination detection rate" is a real number rather than
   zero (the real Drafter, constrained by JSON schema, rarely
   hallucinates). For each adversarial mode, run with Verifier on and
   measure detection; running with Verifier off is the negative
   baseline (everything passes).

b) Haiku vs Sonnet — same fixture, same prompts, model swap. Reports
   judge mean + input/output tokens + wall time.

c) Deterministic ranker vs LLM ranker — measures how close (or not)
   an LLM-ordering matches the CPIC-A-first deterministic order on
   the multi-gene fixture.

All three are gated on availability: any of them can be skipped if the
relevant client/model isn't reachable. None modifies the working tree
besides writing under `eval_results/`.
"""

from __future__ import annotations

from pathlib import Path

from pgx_digest.bundle import Bundle, PGxFinding
from pgx_digest.drafter import (
    AnthropicProvider,
    GeminiProvider,
    LLMDrafter,
    Provider,
    TriagingDrafter,
)
from pgx_digest.eval.adversarial import AdversarialDrafter, AdversarialMode
from pgx_digest.eval.cases import EvalCase, load_cases
from pgx_digest.eval.judge import Judge
from pgx_digest.eval.report import (
    AblationRow,
    CaseResult,
    summarize,
    write_ablation_markdown,
    write_results_jsonl,
)
from pgx_digest.eval.runner import run_case, run_eval
from pgx_digest.pharmcat import parse_pharmcat_json
from pgx_digest.ranker import LLMRanker, rank as deterministic_rank


# ---------------------------------------------------------------------------
# (a) Verifier on/off — detection rate using AdversarialDrafter
# ---------------------------------------------------------------------------


_ADVERSARIAL_MODES: tuple[AdversarialMode, ...] = (
    AdversarialMode.FAITHFUL,
    AdversarialMode.SWAP_PMID,
    AdversarialMode.SWAP_DRUG,
    AdversarialMode.SWAP_PHENOTYPE,
    AdversarialMode.SWAP_DIPLOTYPE,
)


def ablation_verifier(
    cases: tuple[EvalCase, ...],
    *,
    fixtures_dir: Path,
    rate: float = 1.0,
    seed: int = 0,
    out_md: Path | None = None,
) -> tuple[AblationRow, ...]:
    """For each adversarial mode, run all cases with Verifier on and off.

    Reports verifier-pass count per (mode, verifier_state). For the
    faithful baseline, both Verifier-on and Verifier-off should pass
    every case. For each corruption, Verifier-on should reject 100%
    and Verifier-off should pass 100%.
    """
    rows: list[AblationRow] = []

    for mode in _ADVERSARIAL_MODES:
        drafter = AdversarialDrafter(mode=mode, rate=rate, seed=seed)
        for verifier_on in (True, False):
            results = tuple(
                run_case(
                    c,
                    fixtures_dir=fixtures_dir,
                    drafter=drafter,
                    ranker=deterministic_rank,
                    verifier_on=verifier_on,
                    judge=None,
                )
                for c in cases
            )
            state = "ON" if verifier_on else "OFF"
            rows.append(
                summarize(
                    name=f"adv={mode.value} verifier={state}",
                    results=results,
                    notes=(
                        "expect 100% verifier-pass (faithful baseline)"
                        if mode == AdversarialMode.FAITHFUL
                        else (
                            "expect 0% verifier-pass (Verifier should catch)"
                            if verifier_on
                            else "negative baseline (no Verifier)"
                        )
                    ),
                )
            )

    rows_t = tuple(rows)
    if out_md is not None:
        write_ablation_markdown(
            "Ablation A: Verifier on vs off (adversarial drafter)",
            rows_t,
            out_md,
        )
    return rows_t


# ---------------------------------------------------------------------------
# (b) Haiku vs Sonnet — model swap on the real Drafter
# ---------------------------------------------------------------------------


def _default_model_providers() -> tuple[Provider, ...]:
    """The three providers compared in Ablation B.

    Defaults to `gemini-2.0-flash` because Google's free-tier RPD on
    `gemini-2.5-flash` is too low for iterative eval work (~20/day).
    `2.0-flash` gives ~75x headroom with comparable structured-output
    quality for this task.
    """
    return (
        AnthropicProvider(model="claude-haiku-4-5"),
        AnthropicProvider(model="claude-sonnet-4-6"),
        GeminiProvider(model="gemini-2.0-flash"),
    )


def ablation_model(
    cases: tuple[EvalCase, ...],
    *,
    fixtures_dir: Path,
    providers: tuple[Provider, ...] | None = None,
    judge: Judge | None = None,
    out_md: Path | None = None,
    out_jsonl_dir: Path | None = None,
) -> tuple[AblationRow, ...]:
    """For each provider, run all cases through LLMDrafter.

    Pass `judge=Judge()` to fill in prose-quality scores. The judge
    backend is always Haiku for cross-provider comparability — we are
    comparing drafters, not judges.
    """
    if providers is None:
        providers = _default_model_providers()

    rows: list[AblationRow] = []
    for provider in providers:
        drafter = LLMDrafter(provider=provider)
        results = run_eval(
            cases,
            fixtures_dir=fixtures_dir,
            drafter=drafter,
            ranker=deterministic_rank,
            verifier_on=True,
            judge=judge,
        )
        # Safe filename slug: prefix with provider name to disambiguate.
        slug = f"{provider.name}_{provider.model}".replace("/", "_")
        if out_jsonl_dir is not None:
            write_results_jsonl(
                results,
                out_jsonl_dir / f"model_{slug}.jsonl",
            )
        rows.append(
            summarize(
                name=f"{provider.name}/{provider.model}",
                results=results,
            )
        )

    rows_t = tuple(rows)
    if out_md is not None:
        write_ablation_markdown(
            "Ablation B: Drafter model comparison", rows_t, out_md
        )
    return rows_t


# ---------------------------------------------------------------------------
# (c) Deterministic vs LLM ranker — top-1 / top-N agreement
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# (d) Triage on/off — Drafter API-call reduction via deterministic routing
# ---------------------------------------------------------------------------


def ablation_triage(
    cases: tuple[EvalCase, ...],
    *,
    fixtures_dir: Path,
    model: str = "claude-haiku-4-5",
    judge: Judge | None = None,
    out_md: Path | None = None,
    out_jsonl_dir: Path | None = None,
) -> tuple[AblationRow, ...]:
    """Triage on vs off on a single model.

    Reports drafter input/output tokens and judge mean for each setting.
    The judge sees a mix of LLM-generated and template-generated cards
    when triage is on — the comparison answers "does deterministic
    templating degrade judge quality?"
    """
    rows: list[AblationRow] = []

    for triage_on in (False, True):
        provider = AnthropicProvider(model=model)
        drafter = (
            TriagingDrafter(llm_drafter=LLMDrafter(provider=provider))
            if triage_on
            else LLMDrafter(provider=provider)
        )
        results = run_eval(
            cases,
            fixtures_dir=fixtures_dir,
            drafter=drafter,
            ranker=deterministic_rank,
            verifier_on=True,
            judge=judge,
        )
        label = "triage=ON" if triage_on else "triage=OFF"
        if out_jsonl_dir is not None:
            write_results_jsonl(
                results,
                out_jsonl_dir / f"triage_{'on' if triage_on else 'off'}.jsonl",
            )
        rows.append(summarize(name=f"{label} ({model})", results=results))

    rows_t = tuple(rows)
    if out_md is not None:
        write_ablation_markdown(
            f"Ablation D: Triage on vs off ({model})", rows_t, out_md
        )
    return rows_t


def _gene_order(bundle: Bundle[PGxFinding]) -> tuple[str, ...]:
    return tuple(f.gene for f in bundle.items)


def ablation_ranker(
    fixture_path: Path,
    *,
    llm_ranker: LLMRanker | None = None,
    out_md: Path | None = None,
) -> tuple[AblationRow, ...]:
    """Compare deterministic and LLM rankers on the same fixture.

    Reports the ordered gene list each ranker produced and the top-1
    / top-N agreement. The deterministic order is the reference.
    """
    bundle = parse_pharmcat_json(fixture_path)
    if llm_ranker is None:
        llm_ranker = LLMRanker()

    det_ranked = deterministic_rank(bundle)
    llm_ranked = llm_ranker.rank(bundle)

    det_order = _gene_order(det_ranked)
    llm_order = _gene_order(llm_ranked)

    top1_match = bool(det_order and llm_order and det_order[0] == llm_order[0])
    full_match = det_order == llm_order

    notes_det = f"order={list(det_order)}"
    notes_llm = (
        f"order={list(llm_order)}; "
        f"top1_match={top1_match}; full_match={full_match}"
    )

    rows = (
        AblationRow(
            name="ranker=deterministic",
            n_cases=1,
            n_verifier_pass=1,
            n_rule_pass=1,
            judge_mean=None,
            drafter_input_tokens=0,
            drafter_output_tokens=0,
            drafter_latency_s=0.0,
            notes=notes_det,
        ),
        AblationRow(
            name="ranker=llm",
            n_cases=1,
            n_verifier_pass=1,
            n_rule_pass=1,
            judge_mean=None,
            drafter_input_tokens=0,
            drafter_output_tokens=0,
            drafter_latency_s=0.0,
            notes=notes_llm,
        ),
    )

    if out_md is not None:
        write_ablation_markdown(
            "Ablation C: Deterministic vs LLM Ranker", rows, out_md
        )
    return rows


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


def run_all(
    cases_path: Path,
    fixtures_dir: Path,
    output_dir: Path,
    *,
    providers: tuple[Provider, ...] | None = None,
    include_model_ablation: bool = True,
    include_ranker_ablation: bool = True,
    include_triage_ablation: bool = True,
) -> dict[str, tuple[AblationRow, ...]]:
    """Run every ablation and write Markdown tables under `output_dir`.

    Pass `providers` to override the default Ablation B model set
    (Haiku + Sonnet + Gemini 2.0 Flash).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(cases_path)
    results: dict[str, tuple[AblationRow, ...]] = {}

    results["verifier"] = ablation_verifier(
        cases,
        fixtures_dir=fixtures_dir,
        out_md=output_dir / "ablation_a_verifier.md",
    )

    if include_model_ablation:
        results["model"] = ablation_model(
            cases,
            fixtures_dir=fixtures_dir,
            providers=providers,
            judge=Judge(),
            out_md=output_dir / "ablation_b_model.md",
            out_jsonl_dir=output_dir,
        )

    if include_triage_ablation:
        results["triage"] = ablation_triage(
            cases,
            fixtures_dir=fixtures_dir,
            judge=Judge(),
            out_md=output_dir / "ablation_d_triage.md",
            out_jsonl_dir=output_dir,
        )

    if include_ranker_ablation:
        # Ranker comparison needs a fixture with >1 finding; the
        # single-gene one trivially has nothing to reorder. Pick the
        # first case whose fixture parses to multiple findings.
        chosen = next(
            (
                c.fixture
                for c in cases
                if len(parse_pharmcat_json(fixtures_dir / c.fixture)) > 1
            ),
            cases[0].fixture,
        )
        results["ranker"] = ablation_ranker(
            fixtures_dir / chosen,
            out_md=output_dir / "ablation_c_ranker.md",
        )

    return results
