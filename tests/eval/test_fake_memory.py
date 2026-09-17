"""Fake memory adapter: capability combinations and protocol behavior."""

from __future__ import annotations

import pytest

from eval.contracts.adapter import RetrievalRequest, Session, Message
from eval.memories.base import MemoryAdapterError
from eval.memories.fake import FakeMemoryAdapter, FakeMemorySpec
from eval.memories.usage import recorded_usage


def make_session(sid: str = "s_alpha", date: str = "2026-09-03") -> Session:
    return Session(
        session_id=sid,
        occurred_at=date,
        messages=[
            Message(msg_id="m_1", role="user", content="项目使用 pnpm。"),
            Message(msg_id="m_2", role="user", content="别的项目用 npm。"),
        ],
    )


def make_request(query: str = "项目用什么包管理器？") -> RetrievalRequest:
    return RetrievalRequest(
        query=query, question_date="2026-09-06", evidence_token_budget=4096
    )


class TestCapabilityCombinations:
    def test_sync_extractive_only_minimal(self):
        spec = FakeMemorySpec(
            mutation_mode="sync", evidence_kinds=("extractive",), state_inspection=False
        )
        assert set(spec.capabilities()) == {"extractive_evidence"}

    def test_async_mixed_idempotent_inspect(self):
        spec = FakeMemorySpec(
            mutation_mode="async",
            evidence_kinds=("extractive", "generated"),
            idempotent=True,
            state_inspection=True,
        )
        assert set(spec.capabilities()) == {
            "extractive_evidence",
            "generated_evidence",
            "async_mutation",
            "operation_status",
            "idempotent_mutation",
            "state_inspection",
        }

    def test_async_implies_operation_status(self):
        spec = FakeMemorySpec(mutation_mode="async", evidence_kinds=("generated",))
        caps = spec.capabilities()
        assert "async_mutation" in caps and "operation_status" in caps

    def test_generated_only_non_idempotent_no_inspect(self):
        spec = FakeMemorySpec(
            mutation_mode="sync",
            evidence_kinds=("generated",),
            idempotent=False,
            state_inspection=False,
        )
        assert set(spec.capabilities()) == {"generated_evidence"}

    def test_empty_kinds_rejected(self):
        with pytest.raises(Exception):
            FakeMemorySpec(evidence_kinds=())

    def test_adapter_capabilities_match_spec(self):
        adapter = FakeMemoryAdapter(
            FakeMemorySpec(mutation_mode="sync", idempotent=True)
        )
        assert adapter.capabilities() == {
            "extractive_evidence",
            "idempotent_mutation",
            "state_inspection",
        }


class TestSyncVsAsync:
    def _prepared(self, mode: str) -> FakeMemoryAdapter:
        adapter = FakeMemoryAdapter(
            FakeMemorySpec(mutation_mode=mode, evidence_kinds=("extractive",))
        )
        adapter.reset("ns")
        adapter.open("ns")
        return adapter

    def test_sync_ingest_returns_completed(self):
        adapter = self._prepared("sync")
        receipt = adapter.ingest("ns", make_session(), "op-1")
        assert receipt.status == "completed"
        assert receipt.error is None
        assert receipt.memory_ids
        adapter.close("ns")

    def test_async_ingest_awaits_to_completed(self):
        adapter = self._prepared("async")
        receipt = adapter.ingest("ns", make_session(), "op-1")
        assert receipt.status == "accepted"
        second = adapter.await_ready("ns", "op-1", 300.0)
        assert second.status == "completed"
        assert second.operation_id == "op-1"
        adapter.close("ns")

    def test_async_lag_requires_multiple_polls(self):
        adapter = FakeMemoryAdapter(
            FakeMemorySpec(mutation_mode="async", async_lag=2)
        )
        adapter.reset("ns")
        adapter.open("ns")
        assert adapter.ingest("ns", make_session(), "op-1").status == "accepted"
        assert adapter.await_ready("ns", "op-1", 1.0).status == "accepted"
        assert adapter.await_ready("ns", "op-1", 1.0).status == "completed"
        adapter.close("ns")

    def test_await_ready_never_resubmits(self):
        adapter = self._prepared("async")
        adapter.ingest("ns", make_session(), "op-1")
        adapter.await_ready("ns", "op-1", 1.0)
        adapter.close("ns")
        with pytest.raises(MemoryAdapterError) as excinfo:
            adapter.await_ready("ns", "op-1", 1.0)
        assert excinfo.value.code == "namespace_not_open"

    def test_close_with_pending_rejected(self):
        adapter = self._prepared("async")
        adapter.ingest("ns", make_session(), "op-1")
        with pytest.raises(MemoryAdapterError) as excinfo:
            adapter.close("ns")
        assert excinfo.value.code == "pending_operations"


class TestIdempotency:
    def _adapter(self, idempotent: bool) -> FakeMemoryAdapter:
        adapter = FakeMemoryAdapter(
            FakeMemorySpec(mutation_mode="sync", idempotent=idempotent)
        )
        adapter.reset("ns")
        adapter.open("ns")
        return adapter

    def test_same_id_same_input_is_one_logical_mutation(self):
        adapter = self._adapter(True)
        session = make_session()
        first = adapter.ingest("ns", session, "op-1")
        second = adapter.ingest("ns", session, "op-1")
        assert first.status == second.status == "completed"
        assert adapter.stored_message_count("ns") == 2  # stored once

    def test_same_id_different_input_rejected(self):
        adapter = self._adapter(True)
        adapter.ingest("ns", make_session(), "op-1")
        with pytest.raises(MemoryAdapterError) as excinfo:
            adapter.ingest("ns", make_session(sid="s_other"), "op-1")
        assert excinfo.value.code == "operation_id_input_conflict"

    def test_non_idempotent_reexecutes_same_operation(self):
        adapter = self._adapter(False)
        session = make_session()
        first = adapter.ingest("ns", session, "op-1")
        second = adapter.ingest("ns", session, "op-1")
        # Without idempotency the same operation id is processed again: a
        # fresh receipt (not the stored terminal one) and duplicated work.
        assert first.status == second.status == "completed"
        assert first is not second
        assert adapter.stored_message_count("ns") == 4

    def test_new_operation_on_same_message_creates_new_entry(self):
        adapter = self._adapter(True)
        adapter.ingest("ns", make_session(), "op-1")
        adapter.ingest("ns", make_session(), "op-2")
        assert adapter.stored_message_count("ns") == 4


class TestPersistenceAndIsolation:
    def test_data_survives_close_and_reopen(self):
        adapter = FakeMemoryAdapter(FakeMemorySpec())
        adapter.reset("ns")
        adapter.open("ns")
        adapter.ingest("ns", make_session(), "op-1")
        adapter.close("ns")
        with pytest.raises(MemoryAdapterError):
            adapter.retrieve("ns", make_request())
        adapter.open("ns")
        evidence = adapter.retrieve("ns", make_request())
        assert evidence
        adapter.close("ns")

    def test_reset_clears_space(self):
        adapter = FakeMemoryAdapter(FakeMemorySpec())
        adapter.reset("ns")
        adapter.open("ns")
        adapter.ingest("ns", make_session(), "op-1")
        adapter.close("ns")
        adapter.reset("ns")
        adapter.open("ns")
        assert adapter.retrieve("ns", make_request()) == []
        adapter.close("ns")

    def test_namespaces_are_isolated(self):
        adapter = FakeMemoryAdapter(FakeMemorySpec())
        adapter.reset("ns_a")
        adapter.reset("ns_b")
        adapter.open("ns_a")
        adapter.ingest("ns_a", make_session(), "op-a")
        adapter.close("ns_a")
        adapter.open("ns_b")
        assert adapter.retrieve("ns_b", make_request()) == []
        adapter.close("ns_b")


class TestRetrieveEvidence:
    def test_match_mode_returns_consistent_spans(self):
        adapter = FakeMemoryAdapter(
            FakeMemorySpec(retrieval_mode="match", evidence_kinds=("extractive",))
        )
        adapter.reset("ns")
        adapter.open("ns")
        adapter.ingest("ns", make_session(), "op-1")
        evidence = adapter.retrieve("ns", make_request())
        assert evidence
        stored = {
            (s.session_id, m.msg_id): m.content
            for s in [make_session()]
            for m in s.messages
        }
        for ev in evidence:
            assert ev.kind == "extractive"
            span = ev.extractive_span
            assert stored[(span.session_id, span.msg_id)][span.start : span.end] == ev.text
            assert ev.derivation_sources == []
        adapter.close("ns")

    def test_flood_returns_all_units_with_null_scores(self):
        adapter = FakeMemoryAdapter(
            FakeMemorySpec(retrieval_mode="flood", evidence_kinds=("extractive",))
        )
        adapter.reset("ns")
        adapter.open("ns")
        adapter.ingest("ns", make_session(), "op-1")
        evidence = adapter.retrieve("ns", make_request())
        assert len(evidence) == 2
        assert all(ev.retrieval_score is None for ev in evidence)
        adapter.close("ns")

    def test_generated_summary_is_diagnostic_only(self):
        adapter = FakeMemoryAdapter(
            FakeMemorySpec(evidence_kinds=("extractive", "generated"))
        )
        adapter.reset("ns")
        adapter.open("ns")
        adapter.ingest("ns", make_session(), "op-1")
        evidence = adapter.retrieve("ns", make_request())
        generated = [ev for ev in evidence if ev.kind == "generated"]
        assert generated
        summary = generated[0]
        assert summary.extractive_span is None
        assert summary.derivation_sources
        assert summary.source_times == ["2026-09-03"]
        adapter.close("ns")

    def test_no_match_returns_empty_list(self):
        adapter = FakeMemoryAdapter(FakeMemorySpec(retrieval_mode="match"))
        adapter.reset("ns")
        adapter.open("ns")
        adapter.ingest("ns", make_session(), "op-1")
        assert (
            adapter.retrieve(
                "ns", make_request(query="星期几开会？")
            )
            == []
        )
        adapter.close("ns")


class TestInspect:
    def test_inspect_returns_current_states(self):
        adapter = FakeMemoryAdapter(FakeMemorySpec(state_inspection=True))
        adapter.reset("ns")
        adapter.open("ns")
        receipt = adapter.ingest("ns", make_session(), "op-1")
        states = adapter.inspect("ns", list(receipt.memory_ids))
        assert all(s.validity == "current" for s in states)
        assert {s.content for s in states} == {"项目使用 pnpm。", "别的项目用 npm。"}
        adapter.close("ns")

    def test_unknown_id_is_unknown_not_missing(self):
        adapter = FakeMemoryAdapter(FakeMemorySpec(state_inspection=True))
        adapter.reset("ns")
        adapter.open("ns")
        (state,) = adapter.inspect("ns", ["mem_unknown"])
        assert state.validity == "unknown"
        assert state.content is None
        adapter.close("ns")

    def test_inspection_capability_must_be_declared(self):
        adapter = FakeMemoryAdapter(FakeMemorySpec(state_inspection=False))
        adapter.reset("ns")
        adapter.open("ns")
        with pytest.raises(MemoryAdapterError) as excinfo:
            adapter.inspect("ns", ["mem_x"])
        assert excinfo.value.code == "capability_not_declared"
        adapter.close("ns")


class TestUsageReporting:
    def test_usage_reported_through_contextvar(self):
        adapter = FakeMemoryAdapter(FakeMemorySpec())
        adapter.reset("ns")
        adapter.open("ns")
        with recorded_usage() as recorder:
            receipt = adapter.ingest("ns", make_session(), "op-1")
        merged = recorder.merged()
        assert merged is not None
        assert merged.input_tokens == len("项目使用 pnpm。") + len("别的项目用 npm。")
        assert receipt.usage is not None
        with recorded_usage() as recorder:
            adapter.retrieve("ns", make_request())
        merged = recorder.merged()
        assert merged is not None
        assert merged.input_tokens == len(make_request().query)
        assert merged.llm_call_count == 0
        adapter.close("ns")

    def test_no_recorder_means_no_report_needed(self):
        adapter = FakeMemoryAdapter(FakeMemorySpec())
        adapter.reset("ns")
        adapter.open("ns")
        # Outside a recorded attempt, reporting is a no-op (never crashes).
        adapter.ingest("ns", make_session(), "op-1")
        adapter.retrieve("ns", make_request())
        adapter.close("ns")

    def test_await_ready_reports_no_build_usage(self):
        adapter = FakeMemoryAdapter(FakeMemorySpec(mutation_mode="async"))
        adapter.reset("ns")
        adapter.open("ns")
        adapter.ingest("ns", make_session(), "op-1")
        with recorded_usage() as recorder:
            adapter.await_ready("ns", "op-1", 1.0)
        assert recorder.merged() is None
        adapter.close("ns")
