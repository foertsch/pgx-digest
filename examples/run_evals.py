# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "anthropic>=0.40",
#   "google-genai>=1.0",
#   "python-dotenv>=1.0",
# ]
# ///
"""Eval harness: rule-based + LLM-judge tiers.

Loads cases from `tests/eval_cases.jsonl`, runs each through the real
Drafter + Verifier, scores prose with the Judge, and prints a summary.
Writes per-case detail to `eval_results/<timestamp>_results.jsonl`.

Costs roughly $0.01 per full sweep on Claude Haiku 4.5 (depends on
number of cases). Pass --no-judge to skip the LLM-judge tier and run
only the rule-based assertions (no API cost for Drafter + Judge would
still spend on the Drafter call).

Run from the repo root:
    uv run examples/run_evals.py
    uv run examples/run_evals.py --no-judge
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).parent.parent
# override=True because VS Code may export a blank ANTHROPIC_API_KEY from
# launchd and python-dotenv would otherwise leave the blank in place.
# Explicit paths only — do NOT use find_dotenv() (call-stack walk breaks
# on stdin/heredoc). Second path covers running from a git worktree at
# <repo>/.claude/worktrees/<name>/, where the .env lives in the main repo.
load_dotenv(REPO_ROOT / ".env", override=True)
load_dotenv(REPO_ROOT.parent.parent.parent / ".env", override=True)
sys.path.insert(0, str(REPO_ROOT))

from pgx_digest.eval import (  # noqa: E402
    Judge,
    load_cases,
    run_eval,
    write_results_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=REPO_ROOT / "tests" / "eval_cases.jsonl",
        help="Path to eval cases JSONL.",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=REPO_ROOT / "tests" / "fixtures",
        help="Directory holding PharmCAT JSON fixtures.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "eval_results",
        help="Where to write per-case JSONL results.",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the LLM-judge tier (rule checks only).",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases)
    print(f"Loaded {len(cases)} cases from {args.cases.name}")
    judge = None if args.no_judge else Judge()

    results = run_eval(
        cases,
        fixtures_dir=args.fixtures_dir,
        judge=judge,
    )

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = write_results_jsonl(
        results,
        args.results_dir / f"{ts}_results.jsonl",
    )

    # Console summary.
    n = len(results)
    rule_pass = sum(1 for r in results if r.rule_passed)
    ver_pass = sum(1 for r in results if r.verification.passed)
    print()
    print("=" * 60)
    print(f"Rule-based: passed {rule_pass}/{n}")
    print(f"Verifier:   passed {ver_pass}/{n}")
    if judge is not None:
        means = [r.judge.scores.mean for r in results if r.judge is not None]
        if means:
            print(f"Judge mean: {sum(means)/len(means):.2f}  (across {len(means)} cases)")
    print()
    for r in results:
        flag = "PASS" if r.rule_passed and r.verification.passed else "FAIL"
        print(f"  [{flag}] {r.case_id}  ({r.drafter_latency_s:.2f}s)")
        for f in r.rule_failures:
            print(f"      rule: {f.rule}: {f.detail}")
        for f in r.verification.failures:
            print(
                f"      verifier: card={f.card_index} "
                f"{f.field}={f.value!r} ({f.reason})"
            )
    print()
    print(f"Detail: {out}")
    return 0 if (rule_pass == n and ver_pass == n) else 1


if __name__ == "__main__":
    raise SystemExit(main())
