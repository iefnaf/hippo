"""Tests for harness-internal contract models and artifact envelopes."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from eval.contracts.adapter import Evidence
from eval.contracts.internal import (
    JudgeRecordArtifact,
    MetricResult,
    PreparedEvidence,
    PreparedItem,
    ReaderResultArtifact,
    Result,
    ResultArtifact,
    StageAttempt,
    ScoringData,
)

EXTRACTIVE = {
    "kind": "extractive",
    "text": "迁移到 pnpm，以后都用 pnpm。",
    "extractive_span": {
        "session_id": "s_6f2c",
        "msg_id": "m_91ab",
        "start": 0,
        "end": 19,
    },
    "derivation_sources": [],
    "source_times": ["2026-09-03"],
    "retrieval_score": None,
}
GENERATED = {
    "kind": "generated",
    "text": "项目目前统一使用 pnpm。",
    "extractive_span": None,
    "derivation_sources": [],
    "source_times": ["2026-09-03"],
    "retrieval_score": None,
}
FP = "a" * 64


def _item(raw_index: int, evidence: dict, tokens: int = 5) -> dict:
    span = evidence["extractive_span"]
    return {
        "raw_index": raw_index,
        "evidence": evidence,
        "verified_span": dict(span) if span else None,
        "rendered_text": "[s1] " + evidence["text"],
        "token_count": tokens,
        "retained_chars": len(evidence["text"]),
        "truncated": False,
    }


def _prepared(**overrides) -> dict:
    payload = {
        "rendered_text": "[s1] 迁移到 pnpm，以后都用 pnpm。",
        "items": [_item(0, EXTRACTIVE)],
        "token_count": 10,
        "text_token_count": 8,
        "budget": 4096,
        "counting_mode": "test",
        "tokenizer_id": "test:char-v1",
        "dropped_raw_indices": [],
    }
    return payload | overrides


class TestPreparedEvidence:
    def test_valid_round_trip(self):
        prepared = PreparedEvidence.model_validate(_prepared())
        dumped = json.loads(prepared.model_dump_json())
        assert PreparedEvidence.model_validate(dumped) == prepared

    def test_budget_is_hard_cap(self):
        with pytest.raises(ValidationError, match="hard cap"):
            PreparedEvidence.model_validate(_prepared(token_count=5000))

    def test_null_budget_allowed_for_full_history(self):
        prepared = PreparedEvidence.model_validate(
            _prepared(items=[], rendered_text="", token_count=0,
                      text_token_count=0, budget=None)
        )
        assert prepared.budget is None

    def test_missing_raw_index_partition_rejected(self):
        # units 0 and 2 exist, 1 is neither retained nor dropped
        with pytest.raises(ValidationError, match="partition"):
            PreparedEvidence.model_validate(
                _prepared(
                    items=[_item(0, EXTRACTIVE), _item(2, GENERATED, 4)],
                    text_token_count=12,
                )
            )

    def test_truncated_item_must_shrink_span_and_chars(self):
        truncated = _item(0, EXTRACTIVE)
        truncated["evidence"]["text"] = EXTRACTIVE["text"][:10]
        truncated["evidence"]["extractive_span"]["end"] = 10
        truncated["verified_span"]["end"] = 10
        truncated["retained_chars"] = 10
        truncated["truncated"] = True
        prepared = PreparedEvidence.model_validate(
            _prepared(items=[truncated], text_token_count=6)
        )
        assert prepared.items[0].truncated
        assert prepared.items[0].retained_chars == 10

    def test_retained_chars_must_match_text_length(self):
        item = _item(0, EXTRACTIVE)
        item["retained_chars"] = 3
        with pytest.raises(ValidationError, match="retained_chars"):
            PreparedItem.model_validate(item)

    def test_generated_item_with_verified_span_rejected(self):
        item = _item(0, GENERATED)
        item["verified_span"] = dict(EXTRACTIVE["extractive_span"])
        with pytest.raises(ValidationError, match="generated"):
            PreparedItem.model_validate(item)

    def test_test_mode_requires_test_tokenizer(self):
        with pytest.raises(ValidationError, match="test:"):
            PreparedEvidence.model_validate(
                _prepared(tokenizer_id="deepseek-offline")
            )

    def test_no_memory_shape_is_valid(self):
        prepared = PreparedEvidence.model_validate(
            {
                "rendered_text": "",
                "items": [],
                "token_count": 0,
                "text_token_count": 0,
                "budget": 4096,
                "counting_mode": "test",
                "tokenizer_id": "test:char-v1",
                "dropped_raw_indices": [],
            }
        )
        assert prepared.items == []


class TestScoringData:
    def _payload(self, **overrides) -> dict:
        payload = {
            "expected_answer": "pnpm",
            "gold_source_ids": ["s_6f2c"],
            "internal_to_official_session": {"s_6f2c": "answer_01_20240101"},
            "is_abstention": False,
            "question_type": "knowledge-update",
            "official_fields": {},
        }
        return payload | overrides

    def test_non_abstention_requires_gold(self):
        with pytest.raises(ValidationError, match="gold"):
            ScoringData.model_validate(
                self._payload(gold_source_ids=[])
            )

    def test_abstention_with_empty_gold_is_fine(self):
        data = ScoringData.model_validate(
            self._payload(is_abstention=True, gold_source_ids=[])
        )
        assert data.is_abstention

    def test_abstention_with_nonempty_gold_is_also_fine(self):
        # recall is N/A for abstention regardless of gold contents
        data = ScoringData.model_validate(self._payload(is_abstention=True))
        assert data.gold_source_ids


class TestMetricResult:
    def test_computed_requires_value(self):
        with pytest.raises(ValidationError):
            MetricResult.model_validate(
                {"metric_id": "scored_accuracy", "status": "computed",
                 "value": None, "reason": None}
            )

    def test_na_requires_null_value_and_reason(self):
        with pytest.raises(ValidationError):
            MetricResult.model_validate(
                {"metric_id": "scored_accuracy",
                 "status": "not_applicable", "value": 0.5, "reason": "r"}
            )
        with pytest.raises(ValidationError):
            MetricResult.model_validate(
                {"metric_id": "scored_accuracy",
                 "status": "not_applicable", "value": None, "reason": None}
            )

    def test_valid_na(self):
        mr = MetricResult.model_validate(
            {"metric_id": "scored_accuracy", "status": "not_applicable",
             "value": None, "reason": "aggregate-only metric"}
        )
        assert mr.value is None


class TestStageAttempt:
    def _payload(self, **overrides) -> dict:
        payload = {
            "attempt_id": "att1",
            "stage": "retrieve",
            "operation_id": None,
            "outcome": "returned",
            "started_at": "2026-09-06T00:00:00.000000Z",
            "ended_at": "2026-09-06T00:00:01.000000Z",
            "elapsed_ms": 1000.0,
            "input_ref": "in.json",
            "output_ref": "out.json",
            "error": None,
            "usage": None,
        }
        return payload | overrides

    def test_error_outcome_requires_error(self):
        with pytest.raises(ValidationError, match="non-null error"):
            StageAttempt.model_validate(self._payload(outcome="error"))

    def test_returned_requires_output_ref(self):
        with pytest.raises(ValidationError, match="output_ref"):
            StageAttempt.model_validate(
                self._payload(output_ref=None)
            )

    def test_running_must_be_open_ended(self):
        with pytest.raises(ValidationError, match="running"):
            StageAttempt.model_validate(
                self._payload(outcome="running", output_ref=None)
            )

    def test_unknown_stage_rejected(self):
        with pytest.raises(ValidationError, match="unknown stage"):
            StageAttempt.model_validate(self._payload(stage="summarize"))

    def test_naive_timestamp_rejected(self):
        with pytest.raises(ValidationError):
            StageAttempt.model_validate(
                self._payload(started_at="2026-09-06T00:00:00")
            )


def _result(**overrides) -> dict:
    payload = {
        "run_id": "r1",
        "sample_handle": "s1",
        "namespace": "ns1",
        "config_fingerprint": FP,
        "suite": "qa",
        "qa_status": "scored",
        "operation_status": None,
        "correct": True,
        "attribution": "hit_correct",
        "failed_stage": None,
        "metrics": [],
        "stage_states": {"retrieve": "completed"},
        "artifact_refs": {},
        "attempts": [],
    }
    return payload | overrides


class TestResult:
    def test_scored_requires_correct_and_attribution(self):
        with pytest.raises(ValidationError):
            Result.model_validate(_result(correct=None))
        with pytest.raises(ValidationError):
            Result.model_validate(_result(attribution=None))

    def test_failed_never_fabricates_correct(self):
        result = Result.model_validate(
            _result(qa_status="failed", correct=None, attribution=None,
                    failed_stage="read")
        )
        assert result.correct is None
        assert result.attribution is None

    def test_pending_and_invalid_input_allowed(self):
        for status in ("pending", "invalid_input", "context_exceeded"):
            result = Result.model_validate(
                _result(qa_status=status, correct=None, attribution=None)
            )
            assert result.qa_status == status

    def test_qa_suite_rejects_operation_status(self):
        with pytest.raises(ValidationError, match="operation_status"):
            Result.model_validate(_result(operation_status="passed"))

    def test_operations_suite_nulls_qa_fields(self):
        result = Result.model_validate(
            _result(
                suite="operations",
                qa_status=None,
                correct=None,
                attribution=None,
                operation_status="passed",
            )
        )
        assert result.operation_status == "passed"

    def test_operations_suite_rejects_attribution(self):
        with pytest.raises(ValidationError, match="attribution"):
            Result.model_validate(
                _result(
                    suite="operations",
                    qa_status=None,
                    correct=None,
                    operation_status="passed",
                )
            )

    def test_fingerprint_must_be_sha256_hex(self):
        with pytest.raises(ValidationError, match="sha256"):
            Result.model_validate(_result(config_fingerprint="short"))

    def test_unknown_stage_state_rejected(self):
        with pytest.raises(ValidationError):
            Result.model_validate(
                _result(stage_states={"retrieve": "started"})
            )

    def test_all_attribution_values_accepted(self):
        for attribution in (
            "hit_correct",
            "hit_wrong",
            "miss_correct",
            "miss_wrong",
        ):
            result = Result.model_validate(_result(attribution=attribution))
            assert result.attribution == attribution


class TestArtifactEnvelopes:
    def test_every_artifact_carries_schema_version_one(self):
        for artifact in (
            ReaderResultArtifact(
                run_id="r1",
                sample_handle="s1",
                result={
                    "hypothesis": "pnpm",
                    "raw_output": "pnpm",
                    "model": "fake-judge",
                    "usage": None,
                },
            ),
            JudgeRecordArtifact(
                run_id="r1",
                sample_handle="s1",
                request={
                    "question": "q",
                    "expected_answer": "a",
                    "hypothesis": "h",
                    "question_type": "multi-session",
                    "protocol_id": "longmemeval-yes-no@1",
                    "protocol_fields": {},
                },
                result={
                    "correct": True,
                    "raw_output": "yes",
                    "model": "fake-judge",
                    "usage": None,
                },
            ),
            ResultArtifact(
                result=Result.model_validate(_result())
            ),
        ):
            dumped = json.loads(artifact.dump_json())
            assert dumped["schema_version"] == 1

    def test_artifact_round_trip_through_load_json(self):
        artifact = ResultArtifact(result=Result.model_validate(_result()))
        loaded = ResultArtifact.load_json(artifact.dump_json())
        assert loaded == artifact
