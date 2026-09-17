"""Harness-internal contract models.

These types never reach a memory adapter. They cover reader input
preparation (PreparedEvidence), scoring inputs/outputs, per-attempt
records, per-sample results and the schema-versioned artifact envelopes
persisted under a run directory.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from eval.contracts.adapter import (
    ErrorInfo,
    Evidence,
    ResourceUsage,
    SourceSpan,
)
from eval.contracts.common import (
    SCHEMA_VERSION,
    ContractModel,
    SchemaVersionedModel,
    parse_run_timestamp,
)

Attribution = Literal["hit_correct", "hit_wrong", "miss_correct", "miss_wrong"]

#: Attempt classification for the usage split: a call that is part
#: of the plan's first execution is "logical"; bounded retries after
#: a failed attempt are "retry"; isolation+replay (and resume) calls
#: are "replay". Retry and replay usage both count into run totals
#: and are reported separately from logical usage.
ATTEMPT_KINDS = ("logical", "retry", "replay")

#: Stage names of the run pipeline (fixed in M1). "update", "delete"
#: and "inspect" are the operations-suite stages: explicit mutations and
#: the only state-observation channel.
STAGE_NAMES = (
    "ingest",
    "await_ready",
    "update",
    "delete",
    "inspect",
    "retrieve",
    "prepare",
    "read",
    "score",
    "judge",
)

#: Recognized stage_states values (superset of pending/completed/failed).
STAGE_STATES = ("pending", "completed", "failed")


class PreparedItem(ContractModel):
    """One retained evidence unit after validation, rendering and budget."""

    raw_index: int = Field(ge=0)
    evidence: Evidence
    verified_span: SourceSpan | None
    rendered_text: str
    token_count: int = Field(ge=0)
    retained_chars: int = Field(ge=0)
    truncated: bool

    @model_validator(mode="after")
    def _span_rules(self) -> Self:
        if self.evidence.kind == "extractive":
            if self.verified_span is None:
                raise ValueError(
                    "retained extractive items must carry a verified_span"
                )
            ev_span = self.evidence.extractive_span
            if ev_span is None or (
                ev_span.start,
                ev_span.end,
                ev_span.session_id,
                ev_span.msg_id,
            ) != (
                self.verified_span.start,
                self.verified_span.end,
                self.verified_span.session_id,
                self.verified_span.msg_id,
            ):
                raise ValueError(
                    "verified_span must equal the evidence extractive_span"
                )
        elif self.verified_span is not None:
            raise ValueError(
                "generated items must have verified_span=null; generated "
                "text is not a verifiable source span"
            )
        if self.retained_chars != len(self.evidence.text):
            raise ValueError(
                f"retained_chars {self.retained_chars} must equal "
                f"len(evidence.text) == {len(self.evidence.text)}"
            )
        return self


class PreparedEvidence(ContractModel):
    """The exact evidence actually shown to the reader."""

    rendered_text: str
    items: list[PreparedItem]
    token_count: int = Field(ge=0)
    text_token_count: int = Field(ge=0)
    budget: int | None = Field(default=None, gt=0)
    counting_mode: Literal["exact", "estimated", "test"]
    tokenizer_id: str = Field(min_length=1)
    dropped_raw_indices: list[int]

    @model_validator(mode="after")
    def _budget_rules(self) -> Self:
        if self.budget is not None and self.token_count > self.budget:
            raise ValueError(
                f"rendered token_count {self.token_count} exceeds budget "
                f"{self.budget}; the harness budget is a hard cap"
            )
        if self.counting_mode == "test" and not self.tokenizer_id.startswith(
            "test:"
        ):
            raise ValueError(
                "counting_mode='test' requires a tokenizer_id prefixed "
                "with 'test:' (M1 fake counter, not a real tokenizer)"
            )
        seen = [i.raw_index for i in self.items]
        if len(set(seen)) != len(seen):
            raise ValueError("raw_index values must be unique across items")
        all_indices = sorted(seen + list(self.dropped_raw_indices))
        if all_indices != list(range(len(all_indices))):
            raise ValueError(
                "every non-empty raw unit must be retained exactly once or "
                "dropped exactly once (indices must partition 0..n-1)"
            )
        if self.text_token_count > self.token_count:
            raise ValueError(
                "text_token_count must not exceed total token_count"
            )
        return self


class ReaderResult(ContractModel):
    """An accepted reader answer. Failures never fabricate this object."""

    hypothesis: str
    raw_output: str
    model: str | None
    usage: ResourceUsage | None


class ScoringData(ContractModel):
    """Private scoring view: gold, ID mapping and category flags."""

    expected_answer: str
    gold_source_ids: list[str]
    internal_to_official_session: dict[str, str]
    is_abstention: bool
    question_type: str
    official_fields: dict[str, object]

    @model_validator(mode="after")
    def _gold_rules(self) -> Self:
        if not self.is_abstention and not self.gold_source_ids:
            raise ValueError(
                "non-abstention samples must have non-empty gold_source_ids; "
                "a missing gold source is a data validation error, not a "
                "silent exclusion"
            )
        return self


class JudgeRequest(ContractModel):
    """Semantic judge input; the protocol adapter renders the prompt."""

    question: str
    expected_answer: str
    hypothesis: str
    question_type: str
    protocol_id: str
    protocol_fields: dict[str, object]


class JudgeResult(ContractModel):
    """An accepted judge verdict. Unparseable output fails scoring instead."""

    correct: bool
    raw_output: str
    model: str | None
    usage: ResourceUsage | None


class StageAttempt(ContractModel):
    """One real call attempt with timing and usage."""

    attempt_id: str = Field(min_length=1)
    stage: str
    operation_id: str | None
    outcome: Literal["running", "returned", "error"]
    started_at: str
    ended_at: str | None
    elapsed_ms: float | None = Field(default=None, ge=0)
    input_ref: str
    output_ref: str | None
    error: ErrorInfo | None
    usage: ResourceUsage | None
    attempt_kind: Literal["logical", "retry", "replay"] = "logical"

    @field_validator("stage")
    @classmethod
    def _known_stage(cls, v: str) -> str:
        if v not in STAGE_NAMES:
            raise ValueError(
                f"unknown stage {v!r}; known stages: {sorted(STAGE_NAMES)}"
            )
        return v

    @field_validator("started_at", "ended_at")
    @classmethod
    def _timestamp_format(cls, v: str | None) -> str | None:
        if v is not None:
            parse_run_timestamp(v)
        return v

    @model_validator(mode="after")
    def _outcome_rules(self) -> Self:
        if self.outcome == "error" and self.error is None:
            raise ValueError("outcome='error' requires a non-null error")
        if self.outcome == "returned" and self.output_ref is None:
            raise ValueError("outcome='returned' requires an output_ref")
        if self.outcome == "running" and (
            self.ended_at is not None or self.output_ref is not None
        ):
            raise ValueError(
                "outcome='running' must have ended_at=null and output_ref=null"
            )
        return self


class MetricResult(ContractModel):
    """One metric outcome for a sample; metric_id must be registered.

    Per-sample rows REUSE the aggregate-level metric_id (issue #3
    leftover, documented): e.g. a sample carrying
    verifiable_session_recall_macro=0.5 holds that sample's own recall,
    while the same id in report.json holds the macro average over the
    applicable set E. Consumers that aggregate from samples.jsonl (the
    compare command) therefore identify a metric by metric_id and know
    its per-sample semantics from the registry denominator — there is
    no separate per-sample id namespace.
    """

    metric_id: str
    status: Literal["computed", "not_supported", "not_applicable", "pending"]
    value: float | None
    reason: str | None

    @model_validator(mode="after")
    def _value_rules(self) -> Self:
        if self.status == "computed":
            if self.value is None:
                raise ValueError(
                    "status='computed' requires a numeric value"
                )
        elif self.value is not None:
            raise ValueError(
                f"status={self.status!r} must have value=null"
            )
        if self.status != "computed" and not self.reason:
            raise ValueError(
                f"status={self.status!r} requires a reason explaining N/A"
            )
        return self


class Result(ContractModel):
    """Minimal per-sample progress and artifact index."""

    run_id: str
    sample_handle: str
    namespace: str
    config_fingerprint: str
    suite: Literal["qa", "operations"]
    qa_status: (
        Literal[
            "pending",
            "scored",
            "failed",
            "context_exceeded",
            "invalid_input",
        ]
        | None
    )
    operation_status: (
        Literal["pending", "passed", "failed", "not_supported"] | None
    )
    correct: bool | None
    attribution: Attribution | None
    failed_stage: str | None
    metrics: list[MetricResult]
    stage_states: dict[str, str]
    artifact_refs: dict[str, str]
    attempts: list[StageAttempt]

    @field_validator("config_fingerprint")
    @classmethod
    def _fingerprint_shape(cls, v: str) -> str:
        if len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError(
                "config_fingerprint must be a 64-char lowercase sha256 hex"
            )
        return v

    @model_validator(mode="after")
    def _suite_rules(self) -> Self:  # metric_id registration is checked by
        # eval.metrics.validate_result_metrics (registry-aware, avoids an
        # import cycle between contracts and the metric registry).
        if self.suite == "qa":
            if self.operation_status is not None:
                raise ValueError(
                    "qa suite results must have operation_status=null"
                )
            if self.qa_status == "scored":
                if self.correct is None:
                    raise ValueError(
                        "qa_status='scored' requires correct to be a boolean"
                    )
                if self.attribution is None:
                    raise ValueError(
                        "qa_status='scored' requires an attribution class"
                    )
            else:
                if self.correct is not None:
                    raise ValueError(
                        "correct must be null unless qa_status='scored'; "
                        "failures are never fabricated into wrong answers"
                    )
                if self.attribution is not None:
                    raise ValueError(
                        "attribution is only set when qa_status='scored'"
                    )
        else:
            if self.qa_status is not None or self.correct is not None:
                raise ValueError(
                    "operations suite results must have qa_status=null and "
                    "correct=null"
                )
            if self.attribution is not None:
                raise ValueError(
                    "operations suite results have no attribution"
                )
        for stage, state in self.stage_states.items():
            if stage not in STAGE_NAMES:
                raise ValueError(
                    f"unknown stage {stage!r} in stage_states; known: "
                    f"{sorted(STAGE_NAMES)}"
                )
            if state not in STAGE_STATES:
                raise ValueError(
                    f"stage_states[{stage!r}]={state!r} not in "
                    f"{list(STAGE_STATES)}"
                )
        return self


# ---------------------------------------------------------------------------
# Artifact envelopes: every persisted artifact carries schema_version=1.
# ---------------------------------------------------------------------------


class RawEvidenceArtifact(SchemaVersionedModel):
    """Persisted raw retrieve() output for one sample."""

    run_id: str
    sample_handle: str
    stage: Literal["retrieve"] = "retrieve"
    evidence: list[Evidence]


class PreparedEvidenceArtifact(SchemaVersionedModel):
    """Persisted PreparedEvidence (what the reader actually received)."""

    run_id: str
    sample_handle: str
    stage: Literal["prepare"] = "prepare"
    prepared: PreparedEvidence


class ReaderResultArtifact(SchemaVersionedModel):
    """Persisted reader output for one sample."""

    run_id: str
    sample_handle: str
    stage: Literal["read"] = "read"
    result: ReaderResult


class JudgeRecordArtifact(SchemaVersionedModel):
    """Persisted judge input/output for auditing disputed verdicts."""

    run_id: str
    sample_handle: str
    stage: Literal["judge"] = "judge"
    request: JudgeRequest
    result: JudgeResult


class ScoringTraceArtifact(SchemaVersionedModel):
    """Private scoring trace: how recall and attribution were derived.

    Contains gold and the internal->official mapping, so like the judge
    record it is a harness-private artifact: never an adapter, reader or
    reporter-input path into any tested component. The Reporter reads it
    for category fields and micro-recall denominators only.
    """

    run_id: str
    sample_handle: str
    stage: Literal["score"] = "score"
    scoring: ScoringData
    evidence_mode: Literal[
        "extractive_declared", "generated_only", "none_baseline"
    ]
    gold_internal_sessions: list[str]
    actual_sessions_in_order: list[str]
    hit_gold_sessions: list[str]
    recall_applicable: bool
    recall_na_reason: str | None
    recall_value: float | None
    hit_criterion: Literal["gold_recall", "nonempty_evidence", "never"]
    hit: bool


class ResultArtifact(SchemaVersionedModel):
    """Persisted per-sample Result (progress, metrics, artifact refs)."""

    result: Result
