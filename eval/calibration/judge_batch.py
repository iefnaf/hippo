"""Calibration judge batch calls (live) with input discipline (issue #9).

A calibration judge call receives EXACTLY what the scorer's judge
receives: the question, the gold answer, the response, the question
type (protocol template selection) and the protocol-private
abstention flag — nothing else. Retrieval results, memory state,
evidence, conditions and run metadata never reach the judge;
assert_judge_input_discipline enforces the request shape on every
path (scorer and calibration share it).

Failure semantics mirror the scorer (#3/#5): a TRANSIENT transport
error retries a bounded number of times; an unparseable output (empty
content / no choices) is a protocol failure recorded as parse_failed —
NEVER a default 'no'. Parse-failed items are excluded from agreement
denominators and reported separately in the statistics.

Credentials: the judge plan names the environment variable
(api_key_env); run_judge_batch fails fast with an explicit message
when it is unset — an explicit skip/fail, never a silent pass.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Sequence

from pydantic import Field, model_validator

from eval.calibration.sampling import CalibrationItem, CalibrationPlanArtifact
from eval.contracts.adapter import ResourceUsage
from eval.contracts.common import ContractError, SchemaVersionedModel, now_utc
from eval.contracts.internal import JudgeRequest, JudgeResult
from eval.judges.base import JudgeProtocolError
from eval.judges.longmemeval import OFFICIAL_PROTOCOL_ID

#: The exact JudgeRequest field set a calibration call may carry
#: (identical to the scorer's request; enforced on every call).
ALLOWED_REQUEST_FIELDS = frozenset(
    {
        "question",
        "expected_answer",
        "hypothesis",
        "question_type",
        "protocol_id",
        "protocol_fields",
    }
)
ALLOWED_PROTOCOL_FIELDS = frozenset({"abstention"})


def assert_judge_input_discipline(request: JudgeRequest) -> None:
    """The judge input contract: question + gold + response only.

    Raises ContractError when the request carries anything beyond the
    protocol's needs — this is the assertable form of the acceptance
    criterion 「judge 输入只有问题、标准答案与回答」.
    """
    fields = set(request.model_dump().keys())
    extra = fields - ALLOWED_REQUEST_FIELDS
    if extra:
        raise ContractError(
            code="judge_input_discipline",
            message=(
                f"judge request carries non-protocol fields {sorted(extra)}; "
                "the judge may only see question/expected_answer/hypothesis "
                "(+ protocol-private fields)"
            ),
            location="(judge request)",
        )
    extra_fields = set(request.protocol_fields.keys()) - ALLOWED_PROTOCOL_FIELDS
    if extra_fields:
        raise ContractError(
            code="judge_input_discipline",
            message=(
                f"protocol_fields carries non-protocol keys "
                f"{sorted(extra_fields)}; only 'abstention' is part of the "
                "official protocol"
            ),
            location="(judge request)",
        )
    if request.protocol_id != OFFICIAL_PROTOCOL_ID:
        raise ContractError(
            code="judge_input_discipline",
            message=(
                f"calibration judge calls bind {OFFICIAL_PROTOCOL_ID!r}; got "
                f"{request.protocol_id!r} (no self-made prompts)"
            ),
            location="(judge request)",
        )


class CalibrationJudgeCall(SchemaVersionedModel):
    """One calibration judge call (full audit trail)."""

    item_id: str = Field(min_length=1)
    request: JudgeRequest
    verdict: bool | None = None
    raw_output: str | None = None
    parse_failed: bool = False
    response_model: str | None = None
    usage: ResourceUsage | None = None
    error: str | None = None
    attempts: int = Field(default=1, ge=1)
    started_at: str
    ended_at: str | None = None

    @model_validator(mode="after")
    def _verdict_rules(self):
        if self.parse_failed and self.verdict is not None:
            raise ValueError("a parse-failed call must not carry a verdict")
        if not self.parse_failed and self.verdict is None and self.error is None:
            raise ValueError("a call is either parsed, parse-failed or errored")
        return self


class CalibrationJudgeCallsArtifact(SchemaVersionedModel):
    """The archived batch of calibration judge calls (live run header
    diagnostics + the verdicts the statistics pair against)."""

    created_at: str
    judge_alias: str = Field(min_length=1)
    judge_base_url: str = Field(min_length=1)
    judge_temperature: float
    judge_max_output_tokens: int
    observed_response_models: tuple[str, ...] = ()
    calls: tuple[CalibrationJudgeCall, ...]

    @model_validator(mode="after")
    def _call_rules(self):
        ids = [c.item_id for c in self.calls]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate item ids in the judge batch")
        return self


def _build_request(item: CalibrationItem) -> JudgeRequest:
    """The calibration request — the scorer's exact request shape."""
    request = JudgeRequest(
        question=item.question,
        expected_answer=item.expected_answer,
        hypothesis=item.hypothesis,
        question_type=item.question_type,
        protocol_id=OFFICIAL_PROTOCOL_ID,
        protocol_fields={"abstention": item.is_abstention},
    )
    assert_judge_input_discipline(request)
    return request


def run_judge_batch(
    plan: CalibrationPlanArtifact,
    judge: Any,
    *,
    api_key_env: str,
    max_retries: int = 3,
    backoff_base_s: float = 1.0,
    clock: Callable[[], str] = now_utc,
    monotonic: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> CalibrationJudgeCallsArtifact:
    """Call the real judge for every planned item (random + boundary).

    Fails fast BEFORE any call when the credential environment variable
    is unset (explicit, not a silent pass). Transient errors retry with
    the 1s/4s/16s-shaped backoff; every attempt count is recorded.
    """
    if not os.environ.get(api_key_env):
        raise ContractError(
            code="missing_api_key",
            message=(
                f"environment variable {api_key_env!r} (the judge "
                "credential) is not set; judge calibration is a LIVE step "
                "— set it or skip explicitly"
            ),
            location="(credentials)",
        )
    calls: list[CalibrationJudgeCall] = []
    observed: set[str] = set()
    for item in plan.items:
        request = _build_request(item)
        attempts = 0
        while True:
            attempts += 1
            started_at = clock()
            try:
                result: JudgeResult = judge.evaluate(request)
            except JudgeProtocolError as exc:
                if exc.transient and attempts <= max_retries:
                    sleep(backoff_base_s * (4 ** (attempts - 1)))
                    continue
                calls.append(
                    CalibrationJudgeCall(
                        item_id=item.item_id,
                        request=request,
                        parse_failed=not exc.transient,
                        error=f"{exc.code}: {exc.message}",
                        attempts=attempts,
                        started_at=started_at,
                        ended_at=clock(),
                    )
                )
                break
            except Exception as exc:  # noqa: BLE001 - recorded per call
                calls.append(
                    CalibrationJudgeCall(
                        item_id=item.item_id,
                        request=request,
                        error=f"{type(exc).__name__}: {exc}",
                        attempts=attempts,
                        started_at=started_at,
                        ended_at=clock(),
                    )
                )
                break
            if result.model:
                observed.add(result.model)
            calls.append(
                CalibrationJudgeCall(
                    item_id=item.item_id,
                    request=request,
                    verdict=result.correct,
                    raw_output=result.raw_output,
                    response_model=result.model,
                    usage=result.usage,
                    attempts=attempts,
                    started_at=started_at,
                    ended_at=clock(),
                )
            )
            break
    return CalibrationJudgeCallsArtifact(
        created_at=clock(),
        judge_alias=judge.model,
        judge_base_url=judge.base_url,
        judge_temperature=judge.temperature,
        judge_max_output_tokens=judge.max_output_tokens,
        observed_response_models=tuple(sorted(observed)),
        calls=tuple(calls),
    )
