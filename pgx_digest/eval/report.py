"""Result dataclasses and formatters for the eval / ablations harness.

`CaseResult` is the single-case record. `AblationRow` aggregates a set
of results into one comparison row (e.g. "Verifier ON, rate=100%").
Writers emit JSONL (per-case detail) and Markdown (the portfolio-facing
summary table).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pgx_digest.bundle import Bundle, PGxFinding
from pgx_digest.drafter import Draft
from pgx_digest.eval.cases import RuleFailure
from pgx_digest.verifier import VerificationResult

if TYPE_CHECKING:
    from pgx_digest.eval.judge import JudgeResult


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    bundle: Bundle[PGxFinding]
    draft: Draft
    verification: VerificationResult
    rule_failures: tuple[RuleFailure, ...]
    judge: "JudgeResult | None"
    drafter_latency_s: float
    judge_latency_s: float
    drafter_usage: dict[str, int]
    judge_usage: dict[str, int]

    @property
    def rule_passed(self) -> bool:
        return not self.rule_failures


@dataclass(frozen=True)
class AblationRow:
    """One row in an ablation table."""

    name: str
    n_cases: int
    n_verifier_pass: int
    n_rule_pass: int
    judge_mean: float | None
    drafter_input_tokens: int
    drafter_output_tokens: int
    drafter_latency_s: float
    notes: str = ""


def summarize(
    name: str,
    results: tuple[CaseResult, ...],
    notes: str = "",
) -> AblationRow:
    """Collapse a list of CaseResults into a single AblationRow."""
    n = len(results)
    n_verifier_pass = sum(1 for r in results if r.verification.passed)
    n_rule_pass = sum(1 for r in results if r.rule_passed)

    judge_means = [r.judge.scores.mean for r in results if r.judge is not None]
    judge_mean: float | None = (
        sum(judge_means) / len(judge_means) if judge_means else None
    )

    in_tok = sum(r.drafter_usage.get("input_tokens", 0) for r in results)
    out_tok = sum(r.drafter_usage.get("output_tokens", 0) for r in results)
    latency = sum(r.drafter_latency_s for r in results)

    return AblationRow(
        name=name,
        n_cases=n,
        n_verifier_pass=n_verifier_pass,
        n_rule_pass=n_rule_pass,
        judge_mean=judge_mean,
        drafter_input_tokens=in_tok,
        drafter_output_tokens=out_tok,
        drafter_latency_s=latency,
        notes=notes,
    )


def write_results_jsonl(
    results: tuple[CaseResult, ...],
    path: Path,
) -> Path:
    """Write per-case detail as JSON Lines. One row per case."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in results:
            row = {
                "case_id": r.case_id,
                "rule_passed": r.rule_passed,
                "rule_failures": [
                    {"rule": f.rule, "detail": f.detail}
                    for f in r.rule_failures
                ],
                "verifier_passed": r.verification.passed,
                "verifier_failures": [
                    {
                        "card_index": f.card_index,
                        "field": f.field,
                        "value": f.value,
                        "reason": f.reason,
                    }
                    for f in r.verification.failures
                ],
                "n_cards": len(r.draft.cards),
                "drafter_latency_s": round(r.drafter_latency_s, 3),
                "drafter_usage": r.drafter_usage,
                "judge_latency_s": round(r.judge_latency_s, 3),
                "judge_usage": r.judge_usage,
            }
            if r.judge is not None:
                row["judge"] = {
                    "scores": {
                        "patient_clarity": r.judge.scores.patient_clarity,
                        "clinical_accuracy": r.judge.scores.clinical_accuracy,
                        "actionability": r.judge.scores.actionability,
                        "safety_framing": r.judge.scores.safety_framing,
                        "conciseness": r.judge.scores.conciseness,
                        "mean": round(r.judge.scores.mean, 2),
                    },
                    "comments": r.judge.comments,
                }
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def write_ablation_markdown(
    title: str,
    rows: tuple[AblationRow, ...],
    path: Path,
) -> Path:
    """Write a Markdown comparison table. The portfolio-facing artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "Variant",
        "n",
        "Verifier pass",
        "Rule pass",
        "Judge mean",
        "Drafter tokens (in/out)",
        "Drafter latency (s)",
        "Notes",
    ]
    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        judge = (
            f"{r.judge_mean:.2f}" if r.judge_mean is not None else "—"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    r.name,
                    str(r.n_cases),
                    f"{r.n_verifier_pass}/{r.n_cases}",
                    f"{r.n_rule_pass}/{r.n_cases}",
                    judge,
                    f"{r.drafter_input_tokens}/{r.drafter_output_tokens}",
                    f"{r.drafter_latency_s:.2f}",
                    r.notes or "",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")
    return path
