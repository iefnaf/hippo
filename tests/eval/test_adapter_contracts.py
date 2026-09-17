"""Tests for adapter-visible contract models.

Covers the checks the contract document demands: bidirectional
serialization, the two Evidence JSON examples, bad offsets, forged
sources, unknown failure effects, identity/status fields on evidence and
receipt/state/usage rules.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from eval.contracts.adapter import (
    Evidence,
    MemoryState,
    MutationReceipt,
    QueryContext,
    RetrievalRequest,
    ResourceUsage,
    Session,
    SourceRef,
    SourceSpan,
)

# -- the two Evidence examples from the data contract document ---------------

EXTRACTIVE_EXAMPLE = {
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
    "retrieval_score": 2.73,
}

GENERATED_EXAMPLE = {
    "kind": "generated",
    "text": "项目目前统一使用 pnpm。",
    "extractive_span": None,
    "derivation_sources": [{"session_id": "s_6f2c", "msg_id": "m_91ab"}],
    "source_times": ["2026-09-03"],
    "retrieval_score": None,
}

ALL_FIELDS = (
    "kind text extractive_span derivation_sources source_times retrieval_score"
).split()


def _example_payload() -> dict:
    return json.loads(json.dumps(EXTRACTIVE_EXAMPLE))


class TestBidirectionalSerialization:
    @pytest.mark.parametrize(
        "factory",
        [
            lambda: SourceRef(session_id="s1", msg_id="m1"),
            lambda: SourceSpan(
                session_id="s1", msg_id="m1", start=0, end=3
            ),
            lambda: Session(
                session_id="s1",
                occurred_at="2026-09-03",
                messages=[
                    {
                        "msg_id": "m1",
                        "role": "user",
                        "content": "hi",
                    }
                ],
            ),
            lambda: QueryContext(query="q?", question_date="2026-09-06"),
            lambda: RetrievalRequest(
                query="q?",
                question_date="2026-09-06",
                evidence_token_budget=4096,
            ),
            lambda: Evidence.model_validate(EXTRACTIVE_EXAMPLE),
            lambda: Evidence.model_validate(GENERATED_EXAMPLE),
            lambda: MemoryState(
                memory_id="mem1",
                content="c",
                sources=[],
                validity="current",
                superseded_by=[],
            ),
            lambda: ResourceUsage(input_tokens=10, output_tokens=5),
            lambda: MutationReceipt(
                operation_id="op1",
                status="completed",
                memory_ids=["mem1"],
                sources=[],
                error=None,
                usage=None,
            ),
        ],
        ids=[
            "source_ref",
            "source_span",
            "session",
            "query_context",
            "retrieval_request",
            "evidence_extractive",
            "evidence_generated",
            "memory_state",
            "resource_usage",
            "mutation_receipt",
        ],
    )
    def test_round_trip_is_lossless(self, factory):
        obj = factory()
        dumped = json.loads(obj.model_dump_json())
        loaded = type(obj).model_validate(dumped)
        assert loaded == obj

    def test_null_fields_are_explicit_not_omitted(self):
        # T | None means JSON null, never a missing key.
        ev = Evidence.model_validate(GENERATED_EXAMPLE)
        dumped = json.loads(ev.model_dump_json())
        for field in ALL_FIELDS:
            assert field in dumped
        assert dumped["extractive_span"] is None
        assert dumped["retrieval_score"] is None
        # Empty lists serialize as [], not null.
        assert dumped["derivation_sources"] == [
            {"session_id": "s_6f2c", "msg_id": "m_91ab"}
        ]


class TestDocumentExamples:
    def test_both_contract_examples_load(self):
        extractive = Evidence.model_validate(EXTRACTIVE_EXAMPLE)
        assert extractive.kind == "extractive"
        generated = Evidence.model_validate(GENERATED_EXAMPLE)
        assert generated.kind == "generated"

    def test_extractive_span_matches_python_unicode_indexing(self):
        ev = Evidence.model_validate(EXTRACTIVE_EXAMPLE)
        span = ev.extractive_span
        assert span is not None
        assert ev.text == "迁移到 pnpm，以后都用 pnpm。"
        # [start, end) over the cleaned message content: 19 chars.
        assert len(ev.text) == 19
        assert span.start == 0 and span.end == 19


class TestEvidenceRules:
    def test_bad_offset_range_is_rejected(self):
        payload = _example_payload()
        payload["extractive_span"]["start"] = 5
        payload["extractive_span"]["end"] = 3
        with pytest.raises(ValidationError, match="greater than start"):
            Evidence.model_validate(payload)

    def test_zero_length_span_is_rejected(self):
        payload = _example_payload()
        payload["extractive_span"]["end"] = 0
        with pytest.raises(ValidationError):
            Evidence.model_validate(payload)

    def test_empty_text_is_rejected(self):
        payload = _example_payload()
        payload["text"] = ""
        with pytest.raises(ValidationError):
            Evidence.model_validate(payload)

    def test_forged_derivation_sources_on_extractive(self):
        payload = _example_payload()
        payload["derivation_sources"] = [
            {"session_id": "s_6f2c", "msg_id": "m_91ab"}
        ]
        with pytest.raises(ValidationError, match="derivation_sources"):
            Evidence.model_validate(payload)

    def test_generated_with_span_is_rejected(self):
        payload = json.loads(json.dumps(GENERATED_EXAMPLE))
        payload["extractive_span"] = dict(EXTRACTIVE_EXAMPLE["extractive_span"])
        with pytest.raises(ValidationError, match="masquerade"):
            Evidence.model_validate(payload)

    def test_extractive_without_span_is_rejected(self):
        payload = _example_payload()
        payload["extractive_span"] = None
        with pytest.raises(ValidationError):
            Evidence.model_validate(payload)

    def test_identity_and_validity_fields_are_rejected(self):
        for extra in ({"memory_id": "mem1"}, {"validity": "current"}):
            payload = _example_payload() | extra
            with pytest.raises(ValidationError, match="Extra inputs"):
                Evidence.model_validate(payload)

    def test_duplicate_source_times_rejected(self):
        payload = _example_payload()
        payload["source_times"] = ["2026-09-03", "2026-09-03"]
        with pytest.raises(ValidationError, match="deduplicated"):
            Evidence.model_validate(payload)

    def test_invalid_source_time_rejected(self):
        payload = _example_payload()
        payload["source_times"] = ["09/03/2026"]
        with pytest.raises(ValidationError):
            Evidence.model_validate(payload)

    def test_non_finite_score_rejected(self):
        payload = _example_payload()
        payload["retrieval_score"] = float("inf")
        with pytest.raises(ValidationError):
            Evidence.model_validate(payload)

    def test_unknown_kind_rejected(self):
        payload = _example_payload()
        payload["kind"] = "hybrid"
        with pytest.raises(ValidationError):
            Evidence.model_validate(payload)


class TestSessionRules:
    def _payload(self) -> dict:
        return {
            "session_id": "s1",
            "occurred_at": "2026-09-03",
            "messages": [
                {"msg_id": "m1", "role": "user", "content": "a"},
                {"msg_id": "m2", "role": "assistant", "content": "b"},
            ],
        }

    def test_leaked_annotation_fields_rejected(self):
        payload = self._payload()
        payload["messages"][0]["has_answer"] = True
        with pytest.raises(ValidationError, match="Extra inputs"):
            Session.model_validate(payload)

    def test_leaked_answer_marker_in_session_id_rejected_as_extra(self):
        # Session-level annotations (e.g. answer_session_ids) are extras.
        payload = self._payload() | {"answer_session_ids": ["s1"]}
        with pytest.raises(ValidationError):
            Session.model_validate(payload)

    def test_duplicate_msg_ids_rejected(self):
        payload = self._payload()
        payload["messages"][1]["msg_id"] = "m1"
        with pytest.raises(ValidationError, match="duplicate msg_id"):
            Session.model_validate(payload)

    def test_unknown_role_rejected(self):
        payload = self._payload()
        payload["messages"][0]["role"] = "bot"
        with pytest.raises(ValidationError):
            Session.model_validate(payload)

    def test_bad_occurred_at_rejected(self):
        payload = self._payload()
        payload["occurred_at"] = "2026-09-03 10:00"
        with pytest.raises(ValidationError):
            Session.model_validate(payload)

    def test_naive_datetime_occurred_at_rejected(self):
        payload = self._payload()
        payload["occurred_at"] = "2026-09-03T10:00:00"
        with pytest.raises(ValidationError):
            Session.model_validate(payload)


class TestRetrievalRequestRules:
    def _payload(self) -> dict:
        return {
            "query": "q?",
            "question_date": "2026-09-06",
            "evidence_token_budget": 4096,
        }

    def test_budget_must_be_positive(self):
        payload = self._payload()
        payload["evidence_token_budget"] = 0
        with pytest.raises(ValidationError):
            RetrievalRequest.model_validate(payload)

    def test_bad_question_date_rejected(self):
        payload = self._payload()
        payload["question_date"] = "2026/09/06"
        with pytest.raises(ValidationError):
            RetrievalRequest.model_validate(payload)

    def test_empty_query_rejected(self):
        payload = self._payload()
        payload["query"] = ""
        with pytest.raises(ValidationError):
            RetrievalRequest.model_validate(payload)


class TestMemoryStateRules:
    def _payload(self, **overrides) -> dict:
        payload = {
            "memory_id": "mem1",
            "content": "text",
            "sources": [],
            "validity": "current",
            "superseded_by": [],
        }
        return payload | overrides

    def test_current_without_content_rejected(self):
        with pytest.raises(ValidationError, match="must carry content"):
            MemoryState.model_validate(self._payload(content=None))

    def test_deleted_with_content_rejected(self):
        with pytest.raises(ValidationError):
            MemoryState.model_validate(
                self._payload(validity="deleted", content="stale")
            )

    def test_deleted_null_content_ok(self):
        state = MemoryState.model_validate(
            self._payload(validity="deleted", content=None)
        )
        assert state.validity == "deleted"

    def test_unknown_validity_value_rejected(self):
        with pytest.raises(ValidationError):
            MemoryState.model_validate(self._payload(validity="archived"))


class TestResourceUsageRules:
    def test_unknowns_are_null_not_zero(self):
        usage = ResourceUsage()
        assert usage.input_tokens is None
        assert usage.cost_amount is None
        assert usage.currency is None

    def test_known_absent_is_zero(self):
        usage = ResourceUsage(input_tokens=0, llm_call_count=0)
        assert usage.input_tokens == 0

    def test_negative_tokens_rejected(self):
        with pytest.raises(ValidationError):
            ResourceUsage(input_tokens=-1)

    def test_cost_amount_without_currency_rejected(self):
        with pytest.raises(ValidationError, match="together"):
            ResourceUsage(cost_amount="0.42")

    def test_non_decimal_cost_rejected(self):
        with pytest.raises(ValidationError):
            ResourceUsage(cost_amount="cheap", currency="USD")


class TestMutationReceiptRules:
    def _payload(self, **overrides) -> dict:
        payload = {
            "operation_id": "op1",
            "status": "completed",
            "memory_ids": ["mem1"],
            "sources": [],
            "error": None,
            "usage": None,
        }
        return payload | overrides

    def test_accepted_with_error_rejected(self):
        error = {
            "code": "x",
            "message": "m",
            "effect": "none",
            "transient": True,
        }
        with pytest.raises(ValidationError, match="error=null"):
            MutationReceipt.model_validate(
                self._payload(status="accepted", error=error)
            )

    def test_failed_without_error_rejected(self):
        with pytest.raises(ValidationError, match="non-null error"):
            MutationReceipt.model_validate(self._payload(status="failed"))

    def test_unknown_failure_effect_rejected(self):
        error = {
            "code": "backend_down",
            "message": "lost",
            "effect": "maybe",  # not in none/possible/confirmed
            "transient": True,
        }
        with pytest.raises(ValidationError):
            MutationReceipt.model_validate(
                self._payload(status="failed", error=error)
            )

    def test_empty_memory_ids_allowed_on_pending_receipt(self):
        receipt = MutationReceipt.model_validate(
            self._payload(status="accepted", memory_ids=[])
        )
        assert receipt.memory_ids == []
