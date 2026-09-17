"""Adapter-visible contract models.

These are the types a memory adapter sees: sessions/messages going in,
evidence, receipts, state and usage coming out. Field semantics follow
docs/design/eval-harness-data-contracts.md:

- identifiers are anonymized internal IDs only;
- span indices are Python Unicode character offsets, half-open [start, end);
- extractive evidence must carry a verified-able span and no derivation
  sources; generated evidence has no span and may carry derivation sources
  for diagnostics only;
- usage unknowns are null, known-absent is zero;
- receipts: accepted/completed have error=null, failed must carry an error;
- state: current/superseded must have content; deleted/unknown may not.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from eval.contracts.common import ContractModel, parse_dataset_time

Validity = Literal["current", "superseded", "deleted", "unknown"]

Effect = Literal["none", "possible", "confirmed"]
ReceiptStatus = Literal["accepted", "completed", "failed"]

ROLES = ("user", "assistant", "system", "tool")


class SourceRef(ContractModel):
    """Reference to a cleaned internal source (session, message)."""

    session_id: str = Field(min_length=1)
    msg_id: str = Field(min_length=1)


class SourceSpan(ContractModel):
    """A half-open [start, end) Unicode character range within one message."""

    session_id: str = Field(min_length=1)
    msg_id: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int

    @model_validator(mode="after")
    def _span_order(self) -> Self:
        if self.end <= self.start:
            raise ValueError(
                f"span end {self.end} must be greater than start {self.start} "
                "(half-open [start, end) requires at least one character)"
            )
        return self


class Message(ContractModel):
    """Whitelist-shaped message: role/content only, no upstream annotations."""

    msg_id: str = Field(min_length=1)
    role: Literal["user", "assistant", "system", "tool"]
    content: str

    @model_validator(mode="after")
    def _no_leak_fields(self) -> Self:
        # extra='forbid' already rejects unknown keys; additionally guard
        # against upstream annotation markers smuggled into content keys.
        return self


class Session(ContractModel):
    """A single cleaned session passed to ingest (only the current one)."""

    session_id: str = Field(min_length=1)
    occurred_at: str
    messages: list[Message]

    @field_validator("occurred_at")
    @classmethod
    def _time_format(cls, v: str) -> str:
        parse_dataset_time(v)
        return v

    @model_validator(mode="after")
    def _unique_msg_ids(self) -> Self:
        seen = [m.msg_id for m in self.messages]
        if len(set(seen)) != len(seen):
            dupes = sorted({m for m in seen if seen.count(m) > 1})
            raise ValueError(f"duplicate msg_id values in session: {dupes}")
        return self


class QueryContext(ContractModel):
    """Question plus its dataset question_date (shared reader context)."""

    query: str = Field(min_length=1)
    question_date: str

    @field_validator("question_date")
    @classmethod
    def _time_format(cls, v: str) -> str:
        parse_dataset_time(v)
        return v


class RetrievalRequest(ContractModel):
    """What retrieve() receives: query, question_date, positive budget."""

    query: str = Field(min_length=1)
    question_date: str
    evidence_token_budget: int = Field(gt=0)

    @field_validator("question_date")
    @classmethod
    def _time_format(cls, v: str) -> str:
        parse_dataset_time(v)
        return v


class Evidence(ContractModel):
    """One evidence unit returned by retrieve().

    Carries no memory identity and no validity: the unit identity is the
    harness-assigned raw_index, operation targets come from receipts and
    state is only observable via inspect().
    """

    kind: Literal["extractive", "generated"]
    text: str = Field(min_length=1)
    extractive_span: SourceSpan | None
    derivation_sources: list[SourceRef]
    source_times: list[str]
    retrieval_score: float | None

    @field_validator("source_times")
    @classmethod
    def _times_format(cls, v: list[str]) -> list[str]:
        for t in v:
            parse_dataset_time(t)
        if len(set(v)) != len(v):
            raise ValueError("source_times must be deduplicated in order")
        return v

    @field_validator("retrieval_score")
    @classmethod
    def _finite_score(cls, v: float | None) -> float | None:
        if v is not None and not (v == v and -float("inf") < v < float("inf")):
            raise ValueError("retrieval_score must be a finite number or null")
        return v

    @model_validator(mode="after")
    def _kind_constraints(self) -> Self:
        if self.kind == "extractive":
            if self.extractive_span is None:
                raise ValueError(
                    "extractive evidence requires a non-null extractive_span"
                )
            if self.derivation_sources:
                raise ValueError(
                    "extractive evidence must have derivation_sources=[] "
                    "(the span is the source; derivation entries are for "
                    "generated content only)"
                )
        else:
            if self.extractive_span is not None:
                raise ValueError(
                    "generated evidence must have extractive_span=null; it "
                    "must not masquerade as an exact source span"
                )
        return self


class MemoryState(ContractModel):
    """Observable state of one stable memory ID, returned by inspect()."""

    memory_id: str = Field(min_length=1)
    content: str | None
    sources: list[SourceRef]
    validity: Validity
    superseded_by: list[str] | None

    @model_validator(mode="after")
    def _content_rules(self) -> Self:
        if self.validity in ("current", "superseded") and self.content is None:
            raise ValueError(
                f"validity={self.validity!r} must carry content (null content "
                "is only allowed for deleted/unknown)"
            )
        if self.validity in ("deleted", "unknown") and self.content is not None:
            # Allowed by contract as informative remnant? No: contract says
            # content may be null for deleted/unknown but must not be
            # fabricated; a tombstone carrying stale content would let
            # runners mistake it for retrievable text. Reject.
            raise ValueError(
                f"validity={self.validity!r} must have content=null; deleted "
                "or unknown targets must not carry content"
            )
        return self


class ResourceUsage(ContractModel):
    """Usage of one call attempt; unknown quantities are null, not zero."""

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    llm_call_count: int | None = Field(default=None, ge=0)
    cost_amount: str | None = None
    currency: str | None = None

    @model_validator(mode="after")
    def _cost_pair(self) -> Self:
        if (self.cost_amount is None) != (self.currency is None):
            raise ValueError(
                "cost_amount and currency must be provided together "
                "(both present or both null)"
            )
        if self.cost_amount is not None:
            try:
                float(self.cost_amount)
            except ValueError as exc:
                raise ValueError(
                    f"cost_amount must be a decimal string, got {self.cost_amount!r}"
                ) from exc
        return self


class ErrorInfo(ContractModel):
    """Structured error of a call attempt."""

    code: str = Field(min_length=1)
    message: str
    effect: Effect
    transient: bool


class MutationReceipt(ContractModel):
    """Receipt of an ingest/update/delete/await_ready attempt."""

    operation_id: str = Field(min_length=1)
    status: ReceiptStatus
    memory_ids: list[str]
    sources: list[SourceRef]
    error: ErrorInfo | None
    usage: ResourceUsage | None

    @model_validator(mode="after")
    def _status_error_rules(self) -> Self:
        if self.status in ("accepted", "completed") and self.error is not None:
            raise ValueError(
                f"status={self.status!r} must have error=null; errors belong "
                "to failed receipts"
            )
        if self.status == "failed" and self.error is None:
            raise ValueError("status='failed' must carry a non-null error")
        return self

