"""LLM-as-judge scorer for drafted PGx narratives.

The Verifier handles token-level containment. The Judge scores what the
Verifier cannot: clarity, framing, prose quality. Five axes, 1–5 each.

Constructor takes an optional `client` so tests can inject a fake. The
real default is a fresh Anthropic client; everything else is Haiku 4.5
per the project's model-selection decision.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import anthropic

from pgx_digest.bundle import Bundle, PGxFinding
from pgx_digest.drafter import Draft


_JUDGE_SYSTEM_PROMPT = """\
You score patient-facing pharmacogenomic narrative cards. The cards
have already passed a structural verifier (every gene, drug, phenotype,
and PMID claim is known to be present in the source Bundle). Your
job is to score the *prose quality* of the narrative, on five axes
from 1 (terrible) to 5 (excellent):

- patient_clarity: Would a non-medical reader understand each card?
  1 = jargon-heavy, opaque. 5 = direct, plain English.
- clinical_accuracy: Does the narrative convey the recommendation in
  a way that is consistent with the phenotype? E.g. a "Poor
  Metabolizer" card should NOT say "increased metabolism". 5 = the
  prose accurately reflects the underlying phenotype direction.
- actionability: Is it clear what the patient should do or discuss
  with their physician? 1 = vague. 5 = each card has a concrete next
  step.
- safety_framing: Is the language appropriately hedged ("consider",
  "discuss") rather than prescriptive ("you must")? Each card should
  end with the safety footer. 5 = correctly hedged throughout.
- conciseness: Free of filler, repetition, and unnecessary preamble?
  5 = tight. 1 = bloated.

Respond with ONLY a single-line JSON object — no prose, no code fence:
{"patient_clarity": N, "clinical_accuracy": N, "actionability": N, "safety_framing": N, "conciseness": N, "comments": "<one short sentence>"}
"""


_JUDGE_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "patient_clarity": {"type": "integer"},
        "clinical_accuracy": {"type": "integer"},
        "actionability": {"type": "integer"},
        "safety_framing": {"type": "integer"},
        "conciseness": {"type": "integer"},
        "comments": {"type": "string"},
    },
    "required": [
        "patient_clarity",
        "clinical_accuracy",
        "actionability",
        "safety_framing",
        "conciseness",
        "comments",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class JudgeScores:
    patient_clarity: int
    clinical_accuracy: int
    actionability: int
    safety_framing: int
    conciseness: int

    @property
    def mean(self) -> float:
        return sum(
            (
                self.patient_clarity,
                self.clinical_accuracy,
                self.actionability,
                self.safety_framing,
                self.conciseness,
            )
        ) / 5


@dataclass(frozen=True)
class JudgeResult:
    scores: JudgeScores
    comments: str
    raw_text: str


class _AnthropicLike(Protocol):
    """Subset of the Anthropic client surface the Judge uses."""

    @property
    def messages(self) -> Any: ...


class Judge:
    """Claude-Haiku-backed prose-quality judge.

    Tests can inject a fake `client` whose `messages.create(...)` returns
    a stand-in response object exposing `.content[0].text` and `.usage`.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        max_tokens: int = 512,
        client: _AnthropicLike | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = client or anthropic.Anthropic()
        self.last_response: Any | None = None

    def judge(
        self,
        bundle: Bundle[PGxFinding],
        draft: Draft,
    ) -> JudgeResult:
        bundle_payload = json.dumps(
            [
                {
                    "gene": f.gene,
                    "diplotype": f.diplotype,
                    "phenotype": f.phenotype,
                    "drugs": [d.drug for d in f.affected_drugs],
                }
                for f in bundle.items
            ],
            sort_keys=True,
        )
        cards_payload = json.dumps(
            [asdict(c) for c in draft.cards],
            sort_keys=True,
            default=list,
        )

        user_msg = (
            f"Bundle summary:\n{bundle_payload}\n\n"
            f"Drafted cards:\n{cards_payload}"
        )

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": _JUDGE_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": _JUDGE_OUTPUT_SCHEMA,
                }
            },
            messages=[{"role": "user", "content": user_msg}],
        )
        self.last_response = response

        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)

        scores = JudgeScores(
            patient_clarity=int(data["patient_clarity"]),
            clinical_accuracy=int(data["clinical_accuracy"]),
            actionability=int(data["actionability"]),
            safety_framing=int(data["safety_framing"]),
            conciseness=int(data["conciseness"]),
        )
        return JudgeResult(
            scores=scores,
            comments=str(data.get("comments", "")),
            raw_text=text,
        )
