"""Judge protocol: fixed-verdict scoring of reader answers.

The judge receives a semantic JudgeRequest (question, expected answer,
hypothesis) and renders/parses through a PROTOCOL ADAPTER bound to a
fixed protocol id. It never participates in retrieval or answer
generation. JudgeResult only exists for parsed verdicts: unparseable
output raises JudgeProtocolError, which fails the judge stage instead of
defaulting to "wrong".
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from eval.contracts.adapter import Effect, ErrorInfo
from eval.contracts.internal import JudgeRequest, JudgeResult


class JudgeProtocolError(RuntimeError):
    """Judge output could not be parsed under the bound protocol."""

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
class Judge(Protocol):
    """Judge one reader hypothesis against the expected answer."""

    def evaluate(self, request: JudgeRequest) -> JudgeResult: ...
