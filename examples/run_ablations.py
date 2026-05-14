# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "anthropic>=0.40",
#   "fastembed>=0.4",
#   "google-genai>=1.0",
#   "joblib>=1.3",
#   "python-dotenv>=1.0",
#   "scikit-learn>=1.5",
# ]
# ///
"""Run the three portfolio ablations.

(a) Verifier on/off — driven by the AdversarialDrafter so detection
    rate is a real measured number (no API cost; deterministic).
(b) Haiku vs Sonnet — same fixture, both models, judge each. Reports
    judge mean + tokens + wall time. Spends API tokens (~$0.05 per
    full sweep, depending on case count).
(c) Deterministic vs LLM ranker — top-1 agreement on the multi-gene
    fixture. Cheap (one LLM call per fixture).

Writes Markdown comparison tables to `eval_results/`.

Run from the repo root:
    uv run examples/run_ablations.py
    uv run examples/run_ablations.py --skip-model      # if you want to skip B
    uv run examples/run_ablations.py --skip-ranker     # if you want to skip C
    uv run examples/run_ablations.py --only-verifier   # zero-API run
    uv run examples/run_ablations.py --skip-gemini     # Anthropic-only sweep
    uv run examples/run_ablations.py --gemini-model gemini-2.5-flash
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

from pgx_digest import AnthropicProvider, GeminiProvider  # noqa: E402
from pgx_digest.eval.ablations import run_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=REPO_ROOT / "tests" / "eval_cases.jsonl",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=REPO_ROOT / "tests" / "fixtures",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "eval_results",
    )
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-ranker", action="store_true")
    parser.add_argument("--skip-triage", action="store_true")
    parser.add_argument("--skip-drafter-mode", action="store_true")
    parser.add_argument("--skip-triage-classifier", action="store_true")
    parser.add_argument(
        "--only-verifier",
        action="store_true",
        help=(
            "Shortcut for --skip-model --skip-ranker --skip-triage "
            "--skip-drafter-mode (zero API cost)."
        ),
    )
    parser.add_argument(
        "--haiku-model",
        default="claude-haiku-4-5",
        help="Anthropic Haiku model id for Ablation B.",
    )
    parser.add_argument(
        "--sonnet-model",
        default="claude-sonnet-4-6",
        help="Anthropic Sonnet model id for Ablation B.",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-2.5-flash-lite",
        help=(
            "Gemini model id for Ablation B. Default `gemini-2.5-flash-lite` "
            "is the strongest variant that reliably has free-tier access "
            "(~1000 RPD); `gemini-2.5-flash` is ~20 RPD; `gemini-2.0-flash` "
            "is paid-only on at least some projects."
        ),
    )
    parser.add_argument(
        "--skip-gemini",
        action="store_true",
        help="Skip the Gemini row in Ablation B (Anthropic-only sweep).",
    )
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.results_dir / f"{ts}_ablations"
    out_dir.mkdir(parents=True, exist_ok=True)

    include_model = not (args.skip_model or args.only_verifier)
    include_ranker = not (args.skip_ranker or args.only_verifier)
    include_triage = not (args.skip_triage or args.only_verifier)
    include_drafter_mode = not (
        args.skip_drafter_mode or args.only_verifier
    )
    include_triage_classifier = not (
        args.skip_triage_classifier or args.only_verifier
    )

    providers = [
        AnthropicProvider(model=args.haiku_model),
        AnthropicProvider(model=args.sonnet_model),
    ]
    if not args.skip_gemini:
        providers.append(GeminiProvider(model=args.gemini_model))

    results = run_all(
        cases_path=args.cases,
        fixtures_dir=args.fixtures_dir,
        output_dir=out_dir,
        providers=tuple(providers),
        include_model_ablation=include_model,
        include_ranker_ablation=include_ranker,
        include_triage_ablation=include_triage,
        include_drafter_mode_ablation=include_drafter_mode,
        include_triage_classifier_ablation=include_triage_classifier,
    )

    print()
    print("=" * 60)
    print(f"Ablation outputs written under: {out_dir}")
    for name, rows in results.items():
        print(f"\n[{name}]")
        for r in rows:
            judge = f"{r.judge_mean:.2f}" if r.judge_mean is not None else "-"
            print(
                f"  {r.name:40s}  ver={r.n_verifier_pass}/{r.n_cases}  "
                f"rule={r.n_rule_pass}/{r.n_cases}  judge={judge}  "
                f"tokens={r.drafter_input_tokens}/{r.drafter_output_tokens}"
            )
            if r.notes:
                print(f"      ({r.notes})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
