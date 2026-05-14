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
from pgx_digest.corpora import CPICRetriever, PubMedRetriever
from pgx_digest.eval.adversarial import AdversarialDrafter, AdversarialMode
from pgx_digest.pharmcat import parse_pharmcat_json as _parse_for_ablation
from pgx_digest.triage import Triage, TriageDecision
from pgx_digest.triage_ml import LearnedTriage
from pgx_digest.verifier import Verifier
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

    Defaults to `gemini-2.5-flash-lite`. Google's free-tier RPD on
    `gemini-2.5-flash` is ~20/day and on `gemini-2.0-flash` is now
    zero on at least some projects; the `-lite` variant retains
    free-tier access (~1000 RPD) with lower quality. To compare
    against the stronger Gemini variants, pass an explicit
    `providers=` tuple or use the `--gemini-model` flag on the CLI.
    """
    return (
        AnthropicProvider(model="claude-haiku-4-5"),
        AnthropicProvider(model="claude-sonnet-4-6"),
        GeminiProvider(model="gemini-2.5-flash-lite"),
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


# ---------------------------------------------------------------------------
# (e) Batch vs per-card Drafter mode
# ---------------------------------------------------------------------------


def ablation_drafter_mode(
    cases: tuple[EvalCase, ...],
    *,
    fixtures_dir: Path,
    model: str = "claude-haiku-4-5",
    judge: Judge | None = None,
    max_workers: int = 8,
    max_tokens: int = 16384,
    out_md: Path | None = None,
    out_jsonl_dir: Path | None = None,
) -> tuple[AblationRow, ...]:
    """Batch vs per-card LLMDrafter on a single model.

    Reports tokens, latency, judge mean, and verifier+rule pass rates
    for each mode. The per-card mode fans calls out via
    `ThreadPoolExecutor`; verifier pass rate is the most interesting
    number because per-card *structurally* eliminates cross-gene
    contamination (each call sees only one gene).
    """
    rows: list[AblationRow] = []

    for mode in ("batch", "per_card"):
        provider = AnthropicProvider(model=model)
        drafter = LLMDrafter(
            provider=provider,
            mode=mode,  # type: ignore[arg-type]
            max_workers=max_workers,
            max_tokens=max_tokens,
        )
        results = run_eval(
            cases,
            fixtures_dir=fixtures_dir,
            drafter=drafter,
            ranker=deterministic_rank,
            verifier_on=True,
            judge=judge,
        )
        if out_jsonl_dir is not None:
            write_results_jsonl(
                results,
                out_jsonl_dir / f"drafter_mode_{mode}.jsonl",
            )
        rows.append(
            summarize(name=f"mode={mode} ({model})", results=results)
        )

    rows_t = tuple(rows)
    if out_md is not None:
        write_ablation_markdown(
            f"Ablation E: Batch vs per-card Drafter mode ({model})",
            rows_t,
            out_md,
        )
    return rows_t


# ---------------------------------------------------------------------------
# (g) RAG on/off — CPIC context to Drafter + PubMed grounding in Verifier
# ---------------------------------------------------------------------------


def ablation_rag(
    cases: tuple[EvalCase, ...],
    *,
    fixtures_dir: Path,
    model: str = "claude-haiku-4-5",
    cache_dir: Path | None = None,
    citation_grounding_threshold: float = 0.35,
    judge: Judge | None = None,
    out_md: Path | None = None,
    out_jsonl_dir: Path | None = None,
) -> tuple[AblationRow, ...]:
    """Compare RAG-enabled vs RAG-disabled on the same cases.

    `RAG-on` configuration:
    - `LLMDrafter` gets a `CPICRetriever` built over every drug name
      that appears in any case fixture (one index, shared across cases).
    - `Verifier` gets a `PubMedRetriever` built over every PMID
      present in any case bundle (one index, shared across cases).
    - The Drafter prepends retrieved CPIC snippets to its prompt; the
      Verifier flags `cited_pmids` whose abstract-similarity falls
      below `citation_grounding_threshold`.

    `RAG-off` is the bare LLMDrafter + bare Verifier baseline.

    Reports verifier+rule pass rates, judge mean, and token usage for
    each side. The interesting numbers are (a) does RAG context
    improve judge means, and (b) does PubMed grounding catch real
    ungrounded citations vs the bare typed verifier.
    """
    # Build the union of (drug names, PMIDs) across all cases' fixtures.
    drug_names: set[str] = set()
    pmids: set[int] = set()
    unique_fixtures = {c.fixture for c in cases}
    for fixture in unique_fixtures:
        bundle = _parse_for_ablation(fixtures_dir / fixture)
        for finding in bundle.items:
            for drug in finding.affected_drugs:
                drug_names.add(drug.drug)
                pmids.update(drug.pmids)

    cpic = CPICRetriever(cache_dir=cache_dir / "cpic" if cache_dir else None)
    pubmed = PubMedRetriever(
        cache_dir=cache_dir / "pubmed" if cache_dir else None
    )
    print(
        f"[ablation G] indexing {len(drug_names)} drugs (CPIC) "
        f"and {len(pmids)} PMIDs (PubMed)..."
    )
    n_cpic = cpic.build_index_for_drugs(sorted(drug_names))
    n_pubmed = pubmed.build_index_for_pmids(sorted(pmids))
    print(
        f"[ablation G] indexed {n_cpic} CPIC recommendations + "
        f"{n_pubmed} PubMed abstracts."
    )

    rows: list[AblationRow] = []
    from pgx_digest.eval.runner import run_eval  # local to avoid cycle

    for rag_on in (False, True):
        provider = AnthropicProvider(model=model)
        drafter = LLMDrafter(
            provider=provider,
            retriever=cpic if rag_on else None,
        )
        # The Verifier is constructed inside run_case from a default
        # `Verifier()` — to inject a custom one we'd need to extend the
        # runner. For Plan B's first cut we let the Verifier in
        # run_case stay bare and instead post-hoc check grounding on
        # the produced drafts. That keeps the runner unchanged.
        results = run_eval(
            cases,
            fixtures_dir=fixtures_dir,
            drafter=drafter,
            ranker=deterministic_rank,
            verifier_on=True,
            judge=judge,
        )

        # Apply citation grounding as a post-hoc check on the cards
        # produced under rag_on; tally additional failures.
        n_grounding_failures = 0
        if rag_on:
            grounding_verifier = Verifier(
                retriever=pubmed,
                citation_grounding_threshold=citation_grounding_threshold,
            )
            for r in results:
                gr_result = grounding_verifier.verify(r.draft, r.bundle)
                n_grounding_failures += sum(
                    1
                    for f in gr_result.failures
                    if "grounding similarity" in f.reason
                )

        label = "rag=ON" if rag_on else "rag=OFF"
        if out_jsonl_dir is not None:
            write_results_jsonl(
                results,
                out_jsonl_dir / f"rag_{'on' if rag_on else 'off'}.jsonl",
            )
        row = summarize(
            name=f"{label} ({model})",
            results=results,
            notes=(
                f"grounding failures (cards × cited PMIDs): "
                f"{n_grounding_failures}"
                if rag_on
                else "baseline (no RAG)"
            ),
        )
        rows.append(row)

    rows_t = tuple(rows)
    if out_md is not None:
        write_ablation_markdown(
            f"Ablation G: RAG on vs off ({model}) "
            f"— CPIC context + PubMed grounding",
            rows_t,
            out_md,
        )
    return rows_t


# ---------------------------------------------------------------------------
# (f) Rule-based vs learned Triage — decision-level comparison
# ---------------------------------------------------------------------------


def ablation_triage_classifier(
    cases: tuple[EvalCase, ...],
    *,
    fixtures_dir: Path,
    classifier_path: Path | None = None,
    learned_triage: LearnedTriage | None = None,
    out_md: Path | None = None,
) -> tuple[AblationRow, ...]:
    """Decision-level comparison: rule-based Triage vs LearnedTriage.

    Walks every `(PGxFinding, DrugRec)` pair across every fixture in
    `cases` and records the route each Triage variant produced. Zero
    API calls — this measures the routing layer in isolation.

    Reports:
      - Per-variant routing breakdown (llm / template / skip counts)
      - Agreement rate (% of pairs where both variants agree)
      - Disagreement bucket: which pairs the learned model moved
        OUT of `llm` (cost savings) vs INTO `llm` (extra calls)

    Pass either an explicit `learned_triage` (already fit/loaded) or a
    `classifier_path` to a joblib artifact. If neither is given, the
    function tries `triage_data/classifier.joblib` in the parent of
    `fixtures_dir`.
    """
    # Resolve learned Triage
    if learned_triage is None:
        if classifier_path is None:
            classifier_path = (
                fixtures_dir.parent.parent / "triage_data" / "classifier.joblib"
            )
        if not classifier_path.exists():
            raise FileNotFoundError(
                f"No LearnedTriage classifier at {classifier_path}. "
                f"Run `uv run examples/train_triage.py` first."
            )
        learned_triage = LearnedTriage()
        learned_triage.load_classifier(classifier_path)

    rule_triage = Triage()

    # Dedup (gene, drug, recommendation) so we count one decision per
    # unique CPIC instance regardless of which fixture it came from.
    seen: set[tuple[str, str, str]] = set()
    rule_counts: dict[str, int] = {"llm": 0, "template": 0, "skip": 0}
    learned_counts: dict[str, int] = {"llm": 0, "template": 0, "skip": 0}
    agreements = 0
    total = 0
    moved_out_of_llm = 0  # learned moved a `llm` rule-decision to template/skip
    moved_into_llm = 0  # learned moved a `template`/`skip` rule-decision to llm

    unique_fixtures = {c.fixture for c in cases}
    for fixture in sorted(unique_fixtures):
        bundle = _parse_for_ablation(fixtures_dir / fixture)
        for finding in bundle.items:
            for drug in finding.affected_drugs:
                key = (finding.gene, drug.drug, drug.recommendation)
                if key in seen:
                    continue
                seen.add(key)
                rule = rule_triage.classify(finding, drug)
                learned = learned_triage.classify(finding, drug)
                rule_counts[rule.route] += 1
                learned_counts[learned.route] += 1
                total += 1
                if rule.route == learned.route:
                    agreements += 1
                else:
                    if rule.route == "llm" and learned.route != "llm":
                        moved_out_of_llm += 1
                    elif rule.route != "llm" and learned.route == "llm":
                        moved_into_llm += 1

    agreement_rate = agreements / total if total else 0.0
    api_call_delta = moved_out_of_llm - moved_into_llm  # > 0: learned saves calls

    def _row(name: str, counts: dict[str, int], notes: str) -> AblationRow:
        return AblationRow(
            name=name,
            n_cases=total,
            n_verifier_pass=counts["llm"],  # repurpose: llm column
            n_rule_pass=counts["template"],  # repurpose: template column
            judge_mean=None,
            drafter_input_tokens=counts["skip"],  # repurpose: skip column
            drafter_output_tokens=0,
            drafter_latency_s=0.0,
            notes=notes,
        )

    rows = (
        _row(
            "triage=rule-based",
            rule_counts,
            f"llm/template/skip out of {total} unique pairs",
        ),
        _row(
            "triage=learned",
            learned_counts,
            (
                f"agreement={agreement_rate:.1%}; "
                f"net API-call delta vs rule={-api_call_delta:+d} "
                f"(negative = fewer LLM calls)"
            ),
        ),
    )

    if out_md is not None:
        _write_triage_classifier_md(
            out_md,
            total=total,
            rule_counts=rule_counts,
            learned_counts=learned_counts,
            agreement_rate=agreement_rate,
            moved_out_of_llm=moved_out_of_llm,
            moved_into_llm=moved_into_llm,
        )

    return rows


def _write_triage_classifier_md(
    path: Path,
    *,
    total: int,
    rule_counts: dict[str, int],
    learned_counts: dict[str, int],
    agreement_rate: float,
    moved_out_of_llm: int,
    moved_into_llm: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Ablation F: Rule-based vs Learned Triage (decision-level)",
        "",
        f"Zero-API comparison on **{total} unique (gene, drug, recommendation)** triples",
        "drawn from every fixture in the eval set.",
        "",
        "| Variant | llm | template | skip |",
        "|---|---|---|---|",
        f"| rule-based | {rule_counts['llm']} | {rule_counts['template']} | {rule_counts['skip']} |",
        f"| learned    | {learned_counts['llm']} | {learned_counts['template']} | {learned_counts['skip']} |",
        "",
        f"**Agreement rate: {agreement_rate:.1%}**",
        "",
        f"- Learned moved {moved_out_of_llm} pairs OUT of `llm` "
        f"(template or skip — saves an API call each).",
        f"- Learned moved {moved_into_llm} pairs INTO `llm` "
        f"(extra cost; usually nuanced cases the rules called template/skip).",
        f"- **Net API-call delta vs rule-based: {moved_into_llm - moved_out_of_llm:+d}** "
        f"(negative = learned saves calls).",
    ]
    path.write_text("\n".join(lines) + "\n")


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
    include_drafter_mode_ablation: bool = True,
    include_triage_classifier_ablation: bool = True,
    include_rag_ablation: bool = True,
    classifier_path: Path | None = None,
    rag_cache_dir: Path | None = None,
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

    if include_drafter_mode_ablation:
        results["drafter_mode"] = ablation_drafter_mode(
            cases,
            fixtures_dir=fixtures_dir,
            judge=Judge(),
            out_md=output_dir / "ablation_e_drafter_mode.md",
            out_jsonl_dir=output_dir,
        )

    if include_triage_classifier_ablation:
        try:
            results["triage_classifier"] = ablation_triage_classifier(
                cases,
                fixtures_dir=fixtures_dir,
                classifier_path=classifier_path,
                out_md=output_dir / "ablation_f_triage_classifier.md",
            )
        except FileNotFoundError as exc:
            # No classifier on disk yet — skip rather than crash.
            print(f"[ablation F] skipped: {exc}")

    if include_rag_ablation:
        try:
            results["rag"] = ablation_rag(
                cases,
                fixtures_dir=fixtures_dir,
                cache_dir=rag_cache_dir,
                judge=Judge(),
                out_md=output_dir / "ablation_g_rag.md",
                out_jsonl_dir=output_dir,
            )
        except Exception as exc:  # noqa: BLE001
            # Network failures, etc — skip rather than crash the sweep.
            print(f"[ablation G] skipped: {exc}")

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
