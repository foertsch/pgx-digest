"""Eval harness for pgx-digest.

Two tiers:

- Rule-based (`cases.py`, `runner.run_case`): substring-style assertions
  against `DraftedCard` fields. Fast, no API.
- LLM-judge (`judge.py`): Claude scores each draft on a 5-axis PGx
  rubric. Pairs with the rule tier — the judge sees what the verifier
  cannot (clarity, framing, redundancy).

The same primitives drive the ablations in `ablations.py`:
verifier on/off (via the synthetic adversarial drafter), Haiku vs
Sonnet (via model swap), and deterministic vs LLM ranker (via ranker
swap).
"""

from pgx_digest.eval.adversarial import (
    AdversarialDrafter,
    AdversarialMode,
    make_faithful_draft,
)
from pgx_digest.eval.cases import (
    EvalCase,
    RuleFailure,
    check_rules,
    load_cases,
)
from pgx_digest.eval.judge import (
    Judge,
    JudgeResult,
    JudgeScores,
)
from pgx_digest.eval.report import (
    AblationRow,
    CaseResult,
    write_ablation_markdown,
    write_results_jsonl,
)
from pgx_digest.eval.runner import run_case, run_eval

__all__ = [
    "AblationRow",
    "AdversarialDrafter",
    "AdversarialMode",
    "CaseResult",
    "EvalCase",
    "Judge",
    "JudgeResult",
    "JudgeScores",
    "RuleFailure",
    "check_rules",
    "load_cases",
    "make_faithful_draft",
    "run_case",
    "run_eval",
    "write_ablation_markdown",
    "write_results_jsonl",
]
