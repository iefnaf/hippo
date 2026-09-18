"""The calibration record: judge identity + date + criteria digest +
the decision they were computed under (issue #9).

Two acceptance criteria live here:

- 「判定的模型标识与运行日期随校准记录保存」: the record carries the
  FOUR drift identifiers of the judge (alias, observed response
  models, vendor-documented version + date) plus the calibration run
  date, mirroring eval.models.ModelVersionRecord;
- 「判据未变化时判定未被改动（内容哈希入记录）」: the record stores
  the criteria digest (rubric content + thresholds + protocol commit
  + algorithms). Decisions are pure functions of (annotations, judge
calls, criteria); verify_decision_stability proves byte-stability for
  unchanged inputs, and bind_record_to_judge_plan refuses to attach a
  record whose judge identity or protocol binding differs from the
  run's judge plan — any such change requires a NEW calibration.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

from pydantic import Field, model_validator

from eval.calibration.criteria import (
    OFFICIAL_JUDGE_MODEL,
    criteria_digest,
    criteria_payload,
)
from eval.calibration.judge_batch import CalibrationJudgeCallsArtifact
from eval.calibration.sampling import AnnotationRecord, CalibrationPlanArtifact
from eval.calibration.stats import CalibrationDecision, CalibrationStatistics
from eval.contracts.common import ContractError, SchemaVersionedModel
from eval.judges.longmemeval import OFFICIAL_PROTOCOL_ID, UPSTREAM_PROTOCOL_COMMIT

CALIBRATION_RECORD_SCHEMA_VERSION = 1


def _content_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class JudgeIdentity(SchemaVersionedModel):
    """The judge's four drift identifiers at calibration time, plus the
    fixed call-shape parameters the verdicts were produced under."""

    alias: str = Field(min_length=1)
    response_models: tuple[str, ...] = ()
    vendor_documented_version: str
    vendor_documented_on: str
    calibration_run_date: str = Field(min_length=1)
    temperature: float = 0.0
    max_output_tokens: int = Field(default=10, gt=0)


class CalibrationRecordArtifact(SchemaVersionedModel):
    """The committed calibration record (written once, never edited)."""

    schema_version: int = CALIBRATION_RECORD_SCHEMA_VERSION
    record_id: str = Field(min_length=1)
    created_at: str
    judge: JudgeIdentity
    criteria: dict[str, Any]
    criteria_digest: str = Field(min_length=64, max_length=64)
    plan_digest: str = Field(min_length=64, max_length=64)
    annotations_digest: str = Field(min_length=64, max_length=64)
    judge_calls_digest: str = Field(min_length=64, max_length=64)
    statistics: CalibrationStatistics
    decision: CalibrationDecision
    official_model_deviation: dict[str, str]

    @model_validator(mode="after")
    def _digest_rules(self):
        if self.criteria_digest != _content_digest(self.criteria):
            raise ValueError(
                "criteria_digest does not match the embedded criteria content"
            )
        return self


def _decision_payload(decision: CalibrationDecision) -> Any:
    return decision.model_dump(mode="json", exclude_none=True)


def build_calibration_record(
    *,
    plan: CalibrationPlanArtifact,
    annotations: Sequence[AnnotationRecord],
    self_consistency_annotations: Sequence[AnnotationRecord],
    judge_calls: CalibrationJudgeCallsArtifact,
    statistics: CalibrationStatistics,
    decision: CalibrationDecision,
    judge_vendor_documented_version: str,
    judge_vendor_documented_on: str,
    created_at: str,
) -> CalibrationRecordArtifact:
    """Assemble the immutable calibration record.

    The record binds the judge identity observed DURING the batch, the
    criteria digest, content digests of every input and the decision.
    """
    criteria = criteria_payload()
    annotations_payload = [
        *[
            {
                "item_id": a.item_id,
                "annotation": a.annotation,
                "note": a.note,
                "annotated_at": a.annotated_at,
            }
            for a in annotations
        ],
        *[
            {
                "item_id": a.item_id,
                "annotation": a.annotation,
                "note": a.note,
                "annotated_at": a.annotated_at,
                "round": a.round,
            }
            for a in self_consistency_annotations
        ],
    ]
    record_id = (
        "judge-calibration-"
        + hashlib.sha256(
            (
                criteria_digest()
                + "|"
                + judge_calls.judge_alias
                + "|"
                + "|".join(judge_calls.observed_response_models)
                + "|"
                + _content_digest(annotations_payload)[:32]
            ).encode("utf-8")
        ).hexdigest()[:16]
    )
    return CalibrationRecordArtifact(
        record_id=record_id,
        created_at=created_at,
        judge=JudgeIdentity(
            alias=judge_calls.judge_alias,
            response_models=judge_calls.observed_response_models,
            vendor_documented_version=judge_vendor_documented_version,
            vendor_documented_on=judge_vendor_documented_on,
            calibration_run_date=created_at[:10],
            temperature=judge_calls.judge_temperature,
            max_output_tokens=judge_calls.judge_max_output_tokens,
        ),
        criteria=criteria,
        criteria_digest=criteria_digest(),
        plan_digest=_content_digest(plan.model_dump(mode="json")),
        annotations_digest=_content_digest(annotations_payload),
        judge_calls_digest=_content_digest(judge_calls.model_dump(mode="json")),
        statistics=statistics,
        decision=decision,
        official_model_deviation={
            "official_protocol_model": OFFICIAL_JUDGE_MODEL,
            "note": (
                f"judge 绑定官方 anscheck 协议（上游 commit "
                f"{UPSTREAM_PROTOCOL_COMMIT[:12]}…）但模型为 "
                f"{judge_calls.judge_alias}，与官方验证过的 "
                f"{OFFICIAL_JUDGE_MODEL} 不同家族：不宣称与论文分数可比。"
            ),
        },
    )


def verify_decision_stability(
    previous: CalibrationRecordArtifact,
    *,
    statistics: CalibrationStatistics,
    decision: CalibrationDecision,
) -> bool:
    """Unchanged criteria + unchanged inputs => unchanged decision.

    True when the freshly computed (statistics, decision) equal the
    stored ones under the same criteria digest; False when the criteria
    changed (a NEW record is required — the stored decision is never
    edited). Structural mismatches raise: silently returning True for
    different data would forge stability.
    """
    if previous.criteria_digest != criteria_digest():
        return False
    if previous.decision.model_dump(mode="json", exclude_none=True) != _decision_payload(decision):
        raise ContractError(
            code="decision_instability",
            message=(
                "same criteria digest but a different decision: decisions "
                "must be pure functions of (annotations, judge calls, "
                "criteria) — investigate before trusting either record"
            ),
            location="(decision)",
        )
    if previous.statistics.model_dump(mode="json", exclude_none=True) != (
        statistics.model_dump(mode="json", exclude_none=True)
    ):
        raise ContractError(
            code="statistics_instability",
            message=(
                "same criteria digest but different statistics: the "
                "statistics must be pure functions of their inputs"
            ),
            location="(statistics)",
        )
    return True


def bind_record_to_judge_plan(record: CalibrationRecordArtifact, plan: Any) -> None:
    """Refuse to attach a record whose judge binding differs from the
    run's judge plan (model alias, protocol, generation parameters).

    Any mismatch means the calibration no longer describes the judge
    the run will use — a new calibration is required, never a silent
    reuse (docs: 其中任一项变化（含模型漂移）都必须重标).
    """
    problems: list[str] = []
    if record.judge.alias != plan.model:
        problems.append(
            f"judge alias {plan.model!r} != calibrated {record.judge.alias!r}"
        )
    if plan.protocol_id != OFFICIAL_PROTOCOL_ID:
        problems.append(
            f"judge plan protocol {plan.protocol_id!r} is not the official "
            "anscheck protocol"
        )
    if record.criteria.get("protocol_source_commit") != plan.protocol_source_commit:
        problems.append(
            "protocol source commit differs between the calibration record "
            "and the judge plan"
        )
    temperature = float(getattr(plan, "temperature", 0.0))
    if abs(temperature - record.judge.temperature) > 1e-9:
        problems.append(
            f"judge temperature {temperature} != calibrated "
            f"{record.judge.temperature}"
        )
    max_output_tokens = int(getattr(plan, "max_output_tokens", 10))
    if max_output_tokens != record.judge.max_output_tokens:
        problems.append(
            f"judge max_output_tokens {max_output_tokens} != calibrated "
            f"{record.judge.max_output_tokens}"
        )
    if problems:
        raise ContractError(
            code="calibration_binding_mismatch",
            message=(
                "the calibration record does not bind this judge plan: "
                + "; ".join(problems)
                + "; judge 配置变化必须重新校准"
            ),
            location="/judge",
        )
