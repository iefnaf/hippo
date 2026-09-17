"""Configurable fake reader for the M1 offline loop.

FakeReaderSpec fixes the fake profile BEFORE a run (part of the reader
plan's 'extra' mapping, which enters the config fingerprint):

- evidence_echo (default): the hypothesis is the text of the FIRST
  retained evidence unit, or the fixed abstention sentence when nothing
  was retained. The reader demonstrably consumes the exact
  PreparedEvidence the harness kept (same rendered budget survivors,
  same order).
- fixed: always answer with a configured string (for injecting
  controlled verdicts through the judge's substring rule).
- fail: raise ReaderError so tests can exercise the failed-read path
  (recall is still computed from the prepared evidence).

The journal records every call's adapter-visible inputs so isolation
tests can assert gold, expected answers and ID mappings never reach the
reader path. Usage mirrors a real call: question+rendered evidence as
input tokens, hypothesis as output tokens, one LLM call.
"""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import model_validator

from eval.contracts.adapter import QueryContext, ResourceUsage
from eval.contracts.common import ContractModel
from eval.contracts.internal import PreparedEvidence, ReaderResult
from eval.readers.base import ReaderError

FAKE_READER_NAME = "fake-reader"

ABSTENTION_HYPOTHESIS = "无法回答"


class FakeReaderSpec(ContractModel):
    """A fake reader profile; fixed before the run and part of the config."""

    model: str = FAKE_READER_NAME
    mode: Literal["evidence_echo", "fixed", "fail"] = "evidence_echo"
    fixed_hypothesis: str = ""

    @model_validator(mode="after")
    def _fixed_needs_text(self) -> Self:
        if self.mode == "fixed" and not self.fixed_hypothesis:
            raise ValueError(
                "mode='fixed' requires a non-empty fixed_hypothesis"
            )
        return self

    @classmethod
    def from_reader_plan(cls, plan: Any) -> "FakeReaderSpec":
        extra = dict(getattr(plan, "extra", {}) or {})
        return cls(
            model=plan.model,
            mode=extra.get("mode", "evidence_echo"),
            fixed_hypothesis=extra.get("fixed_hypothesis", ""),
        )


class FakeReader:
    """Offline deterministic reader implementing the Reader protocol."""

    def __init__(self, spec: FakeReaderSpec) -> None:
        self.spec = spec
        self.journal: list[dict[str, Any]] = []

    def answer(
        self, question: QueryContext, prepared: PreparedEvidence
    ) -> ReaderResult:
        self.journal.append(
            {
                "query": question.query,
                "question_date": question.question_date,
                "rendered_text": prepared.rendered_text,
                "retained_units": len(prepared.items),
            }
        )
        if self.spec.mode == "fail":
            raise ReaderError(
                "reader_backend_failed",
                "fake reader configured to fail (mode='fail')",
                transient=True,
            )
        if self.spec.mode == "fixed":
            hypothesis = self.spec.fixed_hypothesis
        elif prepared.items:
            hypothesis = prepared.items[0].evidence.text
        else:
            hypothesis = ABSTENTION_HYPOTHESIS
        usage = ResourceUsage(
            input_tokens=len(question.query) + len(prepared.rendered_text),
            output_tokens=len(hypothesis),
            llm_call_count=1,
        )
        return ReaderResult(
            hypothesis=hypothesis,
            raw_output=hypothesis,
            model=self.spec.model,
            usage=usage,
        )


def build_fake_reader(plan: Any) -> FakeReader:
    """Build the fake reader declared by an ExperimentConfig reader plan."""
    return FakeReader(FakeReaderSpec.from_reader_plan(plan))
