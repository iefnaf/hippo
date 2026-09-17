"""Fake judge with a fixed-protocol adapter (injectable verdicts).

The fake plays BOTH roles of the judge path:

- the protocol adapter (render_prompt / parse_verdict): renders a
  deterministic prompt tagged with the bound protocol id and parses
  strict yes/no output -- anything else is unparseable and fails the
  judge stage (never silently "wrong");
- the fake judge (FakeJudge.evaluate): deterministic verdict rules,
  fixed before the run via the judge plan's 'extra' mapping.

Verdict rules:
- expected_substring (default): correct iff the expected answer appears
  verbatim in the hypothesis -- enough for the smoke fixtures where
  echo'd evidence text contains the gold answer;
- always_correct / always_wrong: inject controlled verdicts;
- unparseable: emit a non-yes/no raw output so the protocol parse fails.

verdict_overrides maps question text -> bool and wins over the rule,
making per-sample verdicts injectable in tests.

The official LongMemEval protocol (prompt template + yes/no semantics,
bound commit) replaces this adapter in M2; the JudgeRequest/JudgeResult
shapes stay identical.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from eval.contracts.adapter import ResourceUsage
from eval.contracts.common import ContractModel
from eval.contracts.internal import JudgeRequest, JudgeResult
from eval.judges.base import JudgeProtocolError

FAKE_JUDGE_NAME = "fake-judge"

VERDICT_RULES = (
    "expected_substring",
    "always_correct",
    "always_wrong",
    "unparseable",
)


class FakeJudgeSpec(ContractModel):
    """A fake judge profile; fixed before the run and part of the config."""

    model: str = FAKE_JUDGE_NAME
    verdict_rule: Literal[
        "expected_substring", "always_correct", "always_wrong", "unparseable"
    ] = "expected_substring"
    verdict_overrides: dict[str, bool] = Field(default_factory=dict)

    @classmethod
    def from_judge_plan(cls, plan: Any) -> "FakeJudgeSpec":
        extra = dict(getattr(plan, "extra", {}) or {})
        return cls(
            model=plan.model,
            verdict_rule=extra.get("verdict_rule", "expected_substring"),
            verdict_overrides=dict(extra.get("verdict_overrides", {})),
        )


def render_prompt(request: JudgeRequest) -> str:
    """Render the fake protocol prompt (deterministic, protocol-tagged)."""
    return (
        f"[protocol {request.protocol_id} | fake judge]\n"
        f"Question: {request.question}\n"
        f"Answer: {request.expected_answer}\n"
        f"Hypothesis: {request.hypothesis}\n"
        f"Question type: {request.question_type}\n"
        "Is the hypothesis correct with respect to the answer? "
        "Reply with exactly yes or no."
    )


def parse_verdict(raw_output: str) -> bool:
    """Strict yes/no parse; anything else is a protocol failure."""
    stripped = raw_output.strip()
    if stripped == "yes":
        return True
    if stripped == "no":
        return False
    raise JudgeProtocolError(
        "judge_output_unparseable",
        f"judge output {raw_output!r} is neither 'yes' nor 'no'; the "
        "verdict cannot be fabricated as wrong",
    )


class FakeJudge:
    """Offline deterministic judge implementing the Judge protocol."""

    def __init__(self, spec: FakeJudgeSpec) -> None:
        self.spec = spec
        self.requests: list[JudgeRequest] = []

    def evaluate(self, request: JudgeRequest) -> JudgeResult:
        self.requests.append(request)
        if request.question in self.spec.verdict_overrides:
            correct = self.spec.verdict_overrides[request.question]
        elif self.spec.verdict_rule == "always_correct":
            correct = True
        elif self.spec.verdict_rule == "always_wrong":
            correct = False
        elif self.spec.verdict_rule == "unparseable":
            # Parse fails: the judge stage fails instead of judging wrong.
            parse_verdict("可能是对的吧")
            raise AssertionError("unreachable")  # pragma: no cover
        else:
            correct = request.expected_answer in request.hypothesis
        raw_output = "yes" if correct else "no"
        usage = ResourceUsage(
            input_tokens=len(render_prompt(request)),
            output_tokens=len(raw_output),
            llm_call_count=1,
        )
        return JudgeResult(
            correct=correct,
            raw_output=raw_output,
            model=self.spec.model,
            usage=usage,
        )


def build_fake_judge(plan: Any) -> FakeJudge:
    """Build the fake judge declared by an ExperimentConfig judge plan."""
    return FakeJudge(FakeJudgeSpec.from_judge_plan(plan))
