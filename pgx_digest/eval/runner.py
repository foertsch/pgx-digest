"""Eval runner: orchestrate one or many cases through the pipeline.

Each case picks up its fixture from `tests/fixtures/<case.fixture>`,
runs (optional ranker) -> drafter -> (optional verifier) -> (optional
judge), and produces a `CaseResult` carrying everything the ablation
table needs: rule failures, verification result, judge scores, token
usage, wall time.

The runner is the single piece that knows about all the moving parts;
`ablations.py` calls it three times with different parameterizations.

Run_eval dedupes the expensive work (draft + verify + judge) per
(fixture, drafter) tuple — five of six PGx eval cases share the
multi-gene fixture, so naive serial execution would re-draft the same
24-card output five times. The cache cuts that to one draft per
fixture per provider.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from pgx_digest.bundle import Bundle, PGxFinding
from pgx_digest.drafter import Draft, Drafter, LLMDrafter, ProviderResponse
from pgx_digest.eval.cases import EvalCase, check_rules
from pgx_digest.eval.judge import Judge, JudgeResult
from pgx_digest.eval.report import CaseResult
from pgx_digest.pharmcat import parse_pharmcat_json
from pgx_digest.ranker import rank as deterministic_rank
from pgx_digest.verifier import VerificationResult, Verifier


@dataclass(frozen=True)
class _Precomputed:
    """The drafter+verifier+judge product for one (fixture, drafter) pair.

    Cases that share this pair reuse the result fields below. The
    *cost* fields (latency, token usage) are attributed only to the
    first case that triggered the actual API calls — subsequent cache
    hits report zero, so summaries don't double-count.
    """

    bundle: Bundle[PGxFinding]
    draft: Draft
    verification: VerificationResult
    judge_result: JudgeResult | None
    drafter_latency_s: float
    judge_latency_s: float
    drafter_usage: dict[str, int]
    judge_usage: dict[str, int]


class RankerFn(Protocol):
    def __call__(
        self, bundle: Bundle[PGxFinding]
    ) -> Bundle[PGxFinding]: ...


def _no_op_rank(bundle: Bundle[PGxFinding]) -> Bundle[PGxFinding]:
    return bundle


def _no_op_verify(
    _draft, _bundle
) -> VerificationResult:
    return VerificationResult(passed=True, failures=())


def _usage_dict(response) -> dict[str, int]:
    """Normalize token usage across two response shapes.

    - `ProviderResponse` (LLMDrafter.last_response): flat dataclass with
      `input_tokens`, `output_tokens`, `cache_*_tokens`.
    - Raw `anthropic.types.Message` (Judge.last_response): tokens nested
      under `.usage` with `input_tokens`, `cache_*_input_tokens`.

    Returns the same key set as the Anthropic Message format so the
    JSONL output is comparable across cases.
    """
    if response is None:
        return {}
    if isinstance(response, ProviderResponse):
        out = {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        }
        if response.cache_creation_tokens:
            out["cache_creation_input_tokens"] = response.cache_creation_tokens
        if response.cache_read_tokens:
            out["cache_read_input_tokens"] = response.cache_read_tokens
        return out
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    out = {}
    for k in (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ):
        v = getattr(usage, k, None)
        if v is not None:
            out[k] = int(v)
    return out


def _compute(
    *,
    fixture_path: Path,
    drafter: Drafter,
    ranker: RankerFn,
    verifier_on: bool,
    judge: Judge | None,
) -> _Precomputed:
    """Run draft + verify + judge once for one (fixture, drafter) pair."""
    bundle = parse_pharmcat_json(fixture_path)
    ranked = ranker(bundle)

    t0 = time.perf_counter()
    draft = drafter.draft(ranked)
    drafter_latency_s = time.perf_counter() - t0

    verification = (
        Verifier().verify(draft, ranked)
        if verifier_on
        else _no_op_verify(draft, ranked)
    )

    judge_result: JudgeResult | None = None
    judge_latency_s = 0.0
    if judge is not None:
        t1 = time.perf_counter()
        judge_result = judge.judge(ranked, draft)
        judge_latency_s = time.perf_counter() - t1

    return _Precomputed(
        bundle=ranked,
        draft=draft,
        verification=verification,
        judge_result=judge_result,
        drafter_latency_s=drafter_latency_s,
        judge_latency_s=judge_latency_s,
        drafter_usage=_usage_dict(getattr(drafter, "last_response", None)),
        judge_usage=_usage_dict(getattr(judge, "last_response", None)),
    )


def run_case(
    case: EvalCase,
    *,
    fixtures_dir: Path,
    drafter: Drafter,
    ranker: RankerFn | None = None,
    verifier_on: bool = True,
    judge: Judge | None = None,
    _cache: dict[tuple[str, int], _Precomputed] | None = None,
) -> CaseResult:
    """Run one case end-to-end and return a CaseResult.

    Parameters control the ablation knobs:

    - `drafter`: real LLMDrafter, AdversarialDrafter, or any Drafter.
    - `ranker`: ranking function (deterministic `rank`, `LLMRanker.rank`,
      or `None` -> no reordering).
    - `verifier_on`: when False, returns a vacuously-passed result.
    - `judge`: when provided, runs the prose-quality judge.

    The private `_cache` parameter is the dedup hook used by `run_eval`.
    Callers should not need to set it directly.
    """
    rank_fn: Callable[[Bundle[PGxFinding]], Bundle[PGxFinding]] = (
        ranker or _no_op_rank
    )
    fixture_path = fixtures_dir / case.fixture
    cache_key = (str(fixture_path), id(drafter))

    pre: _Precomputed
    cache_hit = _cache is not None and cache_key in _cache
    if cache_hit:
        pre = _cache[cache_key]
    else:
        pre = _compute(
            fixture_path=fixture_path,
            drafter=drafter,
            ranker=rank_fn,
            verifier_on=verifier_on,
            judge=judge,
        )
        if _cache is not None:
            _cache[cache_key] = pre

    rule_failures = check_rules(case, pre.draft)

    # Only the first case to compute a given (fixture, drafter) gets
    # the API-cost attribution. Subsequent cache hits report zero so
    # the ablation summary reflects true API spend.
    return CaseResult(
        case_id=case.id,
        bundle=pre.bundle,
        draft=pre.draft,
        verification=pre.verification,
        rule_failures=rule_failures,
        judge=pre.judge_result,
        drafter_latency_s=0.0 if cache_hit else pre.drafter_latency_s,
        judge_latency_s=0.0 if cache_hit else pre.judge_latency_s,
        drafter_usage={} if cache_hit else pre.drafter_usage,
        judge_usage={} if cache_hit else pre.judge_usage,
    )


def run_eval(
    cases: tuple[EvalCase, ...],
    *,
    fixtures_dir: Path,
    drafter: Drafter | None = None,
    ranker: RankerFn | None = deterministic_rank,
    verifier_on: bool = True,
    judge: Judge | None = None,
) -> tuple[CaseResult, ...]:
    """Run a list of cases. Default drafter = real LLMDrafter (Anthropic).

    Dedupes the expensive draft + verify + judge work per unique
    (fixture, drafter) pair across the case list — cases that share a
    fixture share their heavy lifting.
    """
    if drafter is None:
        drafter = LLMDrafter()
    cache: dict[tuple[str, int], _Precomputed] = {}
    return tuple(
        run_case(
            c,
            fixtures_dir=fixtures_dir,
            drafter=drafter,
            ranker=ranker,
            verifier_on=verifier_on,
            judge=judge,
            _cache=cache,
        )
        for c in cases
    )
