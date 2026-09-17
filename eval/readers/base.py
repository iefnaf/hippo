"""Reader protocol: the fixed answering component.

The reader is the ONLY component that turns prepared evidence into an
answer. It receives the shared query context (question + dataset
question_date) and the exact PreparedEvidence the harness retained --
never gold, never the private ID mapping, never raw pre-budget evidence.
ReaderResult only exists for accepted answers; failures raise ReaderError
and never fabricate an output object.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from eval.contracts.adapter import Effect, ErrorInfo, QueryContext
from eval.contracts.internal import PreparedEvidence, ReaderResult


class ReaderError(RuntimeError):
    """Structured reader failure (backend or protocol violation)."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        transient: bool = False,
        effect: Effect = "none",
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.transient = transient
        self.effect: Effect = effect

    def to_error_info(self) -> ErrorInfo:
        return ErrorInfo(
            code=self.code,
            message=self.message,
            effect=self.effect,
            transient=self.transient,
        )


@runtime_checkable
class Reader(Protocol):
    """Answer one question from the prepared evidence actually retained."""

    def answer(
        self, question: QueryContext, prepared: PreparedEvidence
    ) -> ReaderResult: ...
