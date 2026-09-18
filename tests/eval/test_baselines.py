"""The three control baselines and the context precheck (issue #8).

Fully offline: BM25 scoring is verified against rank_bm25 itself (the
upstream scoring engine), the adapters are exercised through the real
runner on the manual fixtures with the fake reader/judge, and the
context precheck is driven with offline_fake readers that declare a
context window (the M1 default of no window keeps runnable_coverage
at 1.0 and is asserted too).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rank_bm25 import BM25Okapi

from eval.config import load_config_dict
from eval.contracts.adapter import Message, RetrievalRequest, Session
from eval.judges.fake import FakeJudge, FakeJudgeSpec
from eval.memories import (
    BM25_B,
    BM25_DEFAULT_K,
    BM25_K1,
    UPSTREAM_RETRIEVAL_COMMIT,
    FullHistoryAdapter,
    NoMemoryAdapter,
    build_memory_for_plan,
    full_history_evidence,
)
from eval.memories.fake import FakeMemoryAdapter, FakeMemorySpec
from eval.prepare.context import check_context, full_history_allowance
from eval.prepare.tokens import TestCharTokenizer
from eval.prompts import get_prompt_template, public_prompt_tokens
from eval.readers.fake import FakeReader, FakeReaderSpec
from eval.runner import OfflineRunner
from eval.runs import RunStore


def session(
    sid: str,
    day: str,
    messages: list[tuple[str, str]],
) -> Session:
    return Session(
        session_id=sid,
        occurred_at=day,
        messages=[
            Message(msg_id=f"{sid}_m{i}", role=role, content=content)
            for i, (role, content) in enumerate(messages)
        ],
    )


def request(query: str, budget: int = 4096) -> RetrievalRequest:
    return RetrievalRequest(
        query=query, question_date="2026-09-06", evidence_token_budget=budget
    )


# ---------------------------------------------------------------------------
# Upstream-definition unit checks
# ---------------------------------------------------------------------------


class TestUpstreamDefinitions:
    def test_commit_and_defaults_are_pinned(self):
        assert UPSTREAM_RETRIEVAL_COMMIT == "9e0b455f4ef0e2ab8f2e582289761153549043fc"
        # rank_bm25 BM25Okapi defaults, deliberately NOT tuned
        assert (BM25_K1, BM25_B) == (1.5, 0.75)
        assert BM25_DEFAULT_K == 10

    def test_session_document_is_user_turns_joined_by_space(self):
        from eval.memories.baselines import session_document, upstream_tokenize

        s = session(
            "s1",
            "2026-09-01",
            [("user", "项目使用 npm"), ("assistant", "知道了"), ("user", "好的")],
        )
        assert session_document(s) == "项目使用 npm 好的"
        assert upstream_tokenize("a b  c") == ["a", "b", "", "c"]


class TestNoMemoryAdapter:
    def test_retrieve_always_empty_and_receipts_completed(self):
        adapter = NoMemoryAdapter()
        assert adapter.capabilities() == set()
        adapter.open("ns")
        receipt = adapter.ingest("ns", session("s1", "2026-09-01", [("user", "x")]), "op1")
        assert receipt.status == "completed"
        assert adapter.retrieve("ns", request("anything")) == []
        adapter.close("ns")
        adapter.open("ns")
        assert adapter.retrieve("ns", request("anything")) == []


class TestFullHistoryAdapter:
    def test_every_message_in_time_order_as_extractive_evidence(self):
        adapter = FullHistoryAdapter()
        assert adapter.capabilities() == {"extractive_evidence"}
        adapter.open("ns")
        s1 = session("s1", "2026-09-01", [("user", "a"), ("assistant", "b")])
        s2 = session("s2", "2026-09-02", [("user", "c")])
        adapter.ingest("ns", s1, "op1")
        adapter.ingest("ns", s2, "op2")
        evidence = adapter.retrieve("ns", request("q"))
        assert [e.text for e in evidence] == ["a", "b", "c"]
        for e, (sid, text) in zip(evidence, [("s1", "a"), ("s1", "b"), ("s2", "c")]):
            assert e.kind == "extractive"
            assert e.extractive_span is not None
            assert e.extractive_span.session_id == sid
            assert e.extractive_span.start == 0
            assert e.extractive_span.end == len(text)
            assert e.retrieval_score is None  # unranked control
        # The precheck shares the exact same construction
        assert [e.text for e in full_history_evidence([s1, s2])] == ["a", "b", "c"]

    def test_namespaces_are_isolated(self):
        adapter = FullHistoryAdapter()
        adapter.open("a")
        adapter.open("b")
        adapter.ingest("a", session("s1", "2026-09-01", [("user", "x")]), "op1")
        assert adapter.retrieve("b", request("q")) == []
        assert len(adapter.retrieve("a", request("q"))) == 1


class TestBM25Adapter:
    def _adapter_with(self, sessions, k=BM25_DEFAULT_K):
        plan = type(
            "Plan",
            (),
            {"baseline_kind": "bm25", "config": {"k": k} if k != BM25_DEFAULT_K else {}},
        )()
        adapter = build_memory_for_plan(plan)
        adapter.open("ns")
        for i, s in enumerate(sessions):
            adapter.ingest("ns", s, f"op{i}")
        return adapter

    def test_scores_match_rank_bm25_on_upstream_definition(self):
        sessions = [
            session("s1", "2026-09-01", [("user", "pnpm pnpm pnpm"), ("assistant", "noise")]),
            session("s2", "2026-09-02", [("user", "npm npm"), ("assistant", "pnpm")]),
            session("s3", "2026-09-03", [("user", "unrelated words entirely")]),
        ]
        adapter = self._adapter_with(sessions)
        evidence = adapter.retrieve("ns", request("pnpm"))
        # Top session must be s1 (highest BM25 score under the upstream
        # document/tokenizer definition); verify against rank_bm25 itself.
        corpus = [
            " ".join(m.content for m in s.messages if m.role == "user").split(" ")
            for s in sessions
        ]
        scores = BM25Okapi(corpus).get_scores("pnpm".split(" "))
        best = max(range(3), key=lambda i: scores[i])
        top_sessions = []
        for e in evidence:
            assert e.extractive_span is not None
            if not top_sessions or top_sessions[-1] != e.extractive_span.session_id:
                top_sessions.append(e.extractive_span.session_id)
        assert top_sessions[0] == sessions[best].session_id
        # every evidence carries its session's BM25 score
        by_sid = {s.session_id: scores[i] for i, s in enumerate(sessions)}
        for e in evidence:
            assert e.retrieval_score == pytest.approx(
                by_sid[e.extractive_span.session_id]
            )

    def test_hit_sessions_expand_to_message_level_evidence(self):
        sessions = [
            session("s1", "2026-09-01", [("user", "alpha"), ("assistant", "beta")]),
            session("s2", "2026-09-02", [("user", "gamma")]),
        ]
        adapter = self._adapter_with(sessions)
        evidence = adapter.retrieve("ns", request("alpha beta gamma"))
        texts = [e.text for e in evidence]
        # assistant turns are NOT in the index but ARE expanded as evidence
        assert texts == ["alpha", "beta", "gamma"]
        for e, text in zip(evidence, ["alpha", "beta", "gamma"]):
            span = e.extractive_span
            assert span.start == 0 and span.end == len(text)
            assert e.source_times  # session time rides every unit

    def test_k_cap_and_stable_tie_order(self):
        # 12 identical-document sessions: every score ties, so ordering
        # must fall back to (session time asc, session_id lexicographic).
        sessions = [
            session(f"s{i:02d}", f"2026-09-{(i % 3) + 1:02d}", [("user", "same words")])
            for i in range(12)
        ]
        adapter = self._adapter_with(sessions)
        evidence = adapter.retrieve("ns", request("same words"))
        seen = []
        for e in evidence:
            sid = e.extractive_span.session_id
            if not seen or seen[-1] != sid:
                seen.append(sid)
        assert len(seen) == 10  # k = 10 sessions, not 12
        order_key = [
            (next(s.occurred_at for s in sessions if s.session_id == sid), sid)
            for sid in seen
        ]
        assert order_key == sorted(order_key)

    def test_k_configurable_and_persistence_across_reopen(self):
        plan = type("Plan", (), {"baseline_kind": "bm25", "config": {"k": 1}})()
        adapter = build_memory_for_plan(plan)
        adapter.open("ns")
        adapter.ingest("ns", session("s1", "2026-09-01", [("user", "alpha")]), "op1")
        adapter.ingest("ns", session("s2", "2026-09-02", [("user", "alpha")]), "op2")
        adapter.close("ns")
        adapter.open("ns")  # reopen observes the persisted index
        evidence = adapter.retrieve("ns", request("alpha"))
        sessions_hit = {e.extractive_span.session_id for e in evidence}
        assert len(sessions_hit) == 1

    def test_k_must_be_positive(self):
        from eval.contracts.common import ContractError

        plan = type("Plan", (), {"baseline_kind": "bm25", "config": {"k": 0}})()
        with pytest.raises(ContractError):
            build_memory_for_plan(plan)

    def test_builder_dispatch(self):
        fake = type(
            "Plan",
            (),
            {"baseline_kind": "adapter", "config": {}, "name": "fake-memory"},
        )()
        assert isinstance(build_memory_for_plan(fake), FakeMemoryAdapter)
        assert isinstance(
            build_memory_for_plan(
                type("Plan", (), {"baseline_kind": "none", "config": {}})()
            ),
            NoMemoryAdapter,
        )


# ---------------------------------------------------------------------------
# Context precheck math
# ---------------------------------------------------------------------------


class TestCommittedRealConfigs:
    """The committed real-run configs (loading them needs no data or
    credentials): the three dev50 baselines must pin one identical
    question set — that is what makes them the same-development-set
    comparison of issue #8 AC1."""

    def test_dev50_configs_pin_one_identical_question_set(self):
        from eval.config import load_config_toml

        plans = {}
        for baseline in ("none", "bm25", "full_history"):
            config = load_config_toml(
                f"eval/configs/examples/real_dev50_{baseline}.toml"
            )
            assert config.sample_plan_id == "longmemeval-s-dev-50"
            assert len(config.sample_ids) == 50
            plans[baseline] = list(config.sample_ids)
        assert plans["none"] == plans["bm25"] == plans["full_history"]
        # the smoke-live trio pins the same fixed 8-question subset
        smoke = {}
        for baseline in ("none", "bm25", "full_history"):
            config = load_config_toml(
                f"eval/configs/examples/real_smoke_live_{baseline}.toml"
            )
            assert config.smoke_subset_ids == config.sample_ids
            smoke[baseline] = list(config.sample_ids)
        assert smoke["none"] == smoke["bm25"] == smoke["full_history"]


class TestContextMath:
    def test_components_sum_and_exceedance(self):
        check = check_context(
            prompt_question_tokens=100,
            evidence_tokens=4096,
            format_overhead_tokens=64,
            output_reserve_tokens=512,
            context_window_tokens=5000,
        )
        assert check.fits and check.total == 4772
        exceeded = check_context(
            prompt_question_tokens=100,
            evidence_tokens=4096,
            format_overhead_tokens=64,
            output_reserve_tokens=512,
            context_window_tokens=4771,
        )
        assert not exceeded.fits
        assert "EXCEEDED" in exceeded.detail

    def test_allowance_is_window_minus_fixed_parts(self):
        assert (
            full_history_allowance(
                prompt_question_tokens=100,
                format_overhead_tokens=64,
                output_reserve_tokens=512,
                context_window_tokens=1_000_000,
            )
            == 1_000_000 - 100 - 64 - 512
        )


# ---------------------------------------------------------------------------
# Runner integration on the manual fixtures (offline, fake components)
# ---------------------------------------------------------------------------


HANDLE = "smoke_single_session_user_0001"


def baseline_config(baseline: str, *, window: int | None = None, **reader_over):
    reader = {
        "model": "fake-reader",
        "model_family": "family-r",
        "base_url": "offline://fake",
        "temperature": 0.0,
        "max_output_tokens": 1024,
        "tokenizer_id": "test:char-v1",
        "counting_mode": "test",
    }
    if window is not None:
        reader["context_window_tokens"] = window
        reader["output_reserve_tokens"] = 64
        reader["format_overhead_tokens"] = 8
    reader.update(reader_over)
    memory = {
        "name": f"baseline-{baseline}",
        "baseline_kind": baseline,
        "capabilities": (
            ["extractive_evidence"] if baseline in ("bm25", "full_history") else []
        ),
        "config": {},
    }
    data = {
        "name": f"baseline-test-{baseline}",
        "dataset_plan": "manual-fixtures@1",
        "sample_plan_id": "baseline-tests",
        "sample_ids": [HANDLE],
        "smoke_subset_ids": [],
        "memory": memory,
        "reader": reader,
        "judge": {
            "model": "fake-judge",
            "model_family": "family-j",
            "base_url": "offline://fake",
            "temperature": 0.0,
            "protocol_id": "longmemeval-yes-no@1",
            "protocol_source_commit": "0" * 40,
        },
        "evidence_token_budget": 4096,
    }
    return load_config_dict(data)


def make_runner(tmp_path: Path, config, run_id="run-baseline"):
    from eval.datasets.manual import ManualDataset

    runner = OfflineRunner(
        config=config,
        dataset=ManualDataset.load_default(),
        adapter=build_memory_for_plan(config.memory),
        reader=FakeReader(FakeReaderSpec.from_reader_plan(config.reader)),
        judge=FakeJudge(FakeJudgeSpec.from_judge_plan(config.judge)),
        store=RunStore(tmp_path / "runs", run_id),
        run_id=run_id,
    )
    return runner


class TestRunnerBaselines:
    def test_none_baseline_reader_sees_no_evidence_and_recall_na(self, tmp_path):
        config = baseline_config("none", window=500)
        runner = make_runner(tmp_path, config)
        reader = runner.reader
        outcome = runner.run()
        assert outcome.scored == 1
        assert reader.journal[0]["rendered_text"] == ""
        assert reader.journal[0]["retained_units"] == 0
        # question date rides the shared query context
        assert reader.journal[0]["question_date"] == "2026-09-06"
        result = outcome.results[0].result
        recall = {m.metric_id: m for m in result.metrics}
        assert recall["verifiable_session_recall_macro"].status == "not_applicable"
        assert "ranking_baseline" in recall["verifiable_session_recall_macro"].reason
        store = RunStore(tmp_path / "runs", outcome.run_id)
        trace = json.loads(
            store.resolve_ref(result.artifact_refs["scoring"]).read_text()
        )
        assert trace["evidence_mode"] == "none_baseline"
        assert trace["hit_criterion"] == "never"
        assert trace["hit"] is False

    def test_bm25_recall_computed_and_budgeted_after_expansion(self, tmp_path):
        config = baseline_config("bm25", window=4096 + 600)
        runner = make_runner(tmp_path, config)
        outcome = runner.run()
        assert outcome.scored == 1
        result = outcome.results[0].result
        recall = {m.metric_id: m for m in result.metrics}
        # ranked, extractive-declared: recall is COMPUTED (not N/A)
        assert recall["verifiable_session_recall_macro"].status == "computed"
        assert 0.0 <= recall["verifiable_session_recall_macro"].value <= 1.0
        store = RunStore(tmp_path / "runs", outcome.run_id)
        raw = json.loads(
            store.resolve_ref(result.artifact_refs["raw_evidence"]).read_text()
        )
        prepared = json.loads(
            store.resolve_ref(result.artifact_refs["prepared_evidence"]).read_text()
        )
        # expansion: raw units are whole messages; budget applied after
        assert raw["evidence"] and all(
            e["kind"] == "extractive" for e in raw["evidence"]
        )
        assert prepared["prepared"]["token_count"] <= 4096
        assert prepared["prepared"]["counting_mode"] == "test"
        assert prepared["prepared"]["tokenizer_id"] == "test:char-v1"

    def test_full_history_unbounded_within_context_and_not_in_ranking_recall(
        self, tmp_path
    ):
        from eval.datasets.manual import ManualDataset

        dataset = ManualDataset.load_default()
        sessions = dataset.iter_sessions(HANDLE)
        total_messages = sum(len(s.messages) for s in sessions)
        # window comfortably above the full render
        config = baseline_config("full_history", window=10_000)
        runner = make_runner(tmp_path, config)
        outcome = runner.run()
        assert outcome.scored == 1
        result = outcome.results[0].result
        store = RunStore(tmp_path / "runs", outcome.run_id)
        prepared = json.loads(
            store.resolve_ref(result.artifact_refs["prepared_evidence"]).read_text()
        )["prepared"]
        # NOT bound by the 4K retrieval budget semantics: every message
        # retained whole (no truncation, no drops) under the allowance.
        assert len(prepared["items"]) == total_messages
        assert all(not item["truncated"] for item in prepared["items"])
        assert prepared["dropped_raw_indices"] == []
        assert prepared["budget"] > 4096  # context-derived allowance
        # ranking recall N/A (issue #3 leftover) but QA still scored
        recall = {m.metric_id: m for m in result.metrics}
        assert recall["verifiable_session_recall_macro"].status == "not_applicable"
        assert "ranking_baseline" in recall["verifiable_session_recall_macro"].reason
        trace = json.loads(
            store.resolve_ref(result.artifact_refs["scoring"]).read_text()
        )
        assert trace["evidence_mode"] == "full_history_control"
        assert trace["hit_criterion"] == "nonempty_evidence"
        assert trace["hit"] is True  # non-empty evidence was provided

    def test_full_history_over_limit_is_context_exceeded_not_truncated(
        self, tmp_path
    ):
        from eval.datasets.manual import ManualDataset

        dataset = ManualDataset.load_default()
        sessions = dataset.iter_sessions(HANDLE)
        tokenizer = TestCharTokenizer()
        template = get_prompt_template("offline-fake@0")
        question = dataset.get_question(HANDLE)
        prompt_tokens = public_prompt_tokens(template, question, tokenizer)
        full_tokens = tokenizer.count(
            "".join(m.content for s in sessions for m in s.messages)
        )
        # A window that cannot hold prompt + full history + overhead +
        # reserve: the sample MUST end context_exceeded, never truncated.
        tight = prompt_tokens + full_tokens + 8 + 64 - 10
        config = baseline_config("full_history", window=tight)
        runner = make_runner(tmp_path, config)
        outcome = runner.run()
        assert outcome.context_exceeded == 1
        assert outcome.scored == 0
        result = outcome.results[0].result
        assert result.qa_status == "context_exceeded"
        assert result.failed_stage == "precheck"
        assert result.stage_states["precheck"] == "failed"
        assert result.stage_states["read"] == "pending"
        precheck_attempt = next(
            a for a in result.attempts if a.stage == "precheck" and a.error
        )
        assert precheck_attempt.error.code == "context_exceeded"
        # reader never called: no reader artifact was produced
        assert "reader_result" not in result.artifact_refs
        assert runner.reader.journal == []
        # report surfaces the status and runnable_coverage denominator
        report = json.loads(
            (outcome.run_dir / "report.json").read_text(encoding="utf-8")
        )
        assert report["statuses"]["context_exceeded"] == 1
        assert report["statuses"]["runnable_coverage" if False else "planned"] == 1
        runnable = next(
            m for m in report["metrics"] if m["metric_id"] == "runnable_coverage"
        )
        assert runnable["status"] == "computed"
        assert runnable["value"] == 0.0
        pqs = next(
            m for m in report["metrics"] if m["metric_id"] == "planned_question_score"
        )
        assert pqs["value"] == 0.0  # contributes zero, not excluded

    def test_budget_bound_baseline_exceeded_by_worst_case_budget(self, tmp_path):
        # A window smaller than prompt+budget+overhead+reserve marks the
        # sample context_exceeded BEFORE any ingest or model call.
        config = baseline_config("bm25", window=100)
        runner = make_runner(tmp_path, config)
        outcome = runner.run()
        assert outcome.context_exceeded == 1
        assert "raw_evidence" not in outcome.results[0].result.artifact_refs

    def test_no_window_declared_keeps_m1_behavior(self, tmp_path):
        config = baseline_config("bm25")  # context_window_tokens = None
        runner = make_runner(tmp_path, config)
        outcome = runner.run()
        result = outcome.results[0].result
        assert "precheck" not in result.stage_states
        report = json.loads(
            (outcome.run_dir / "report.json").read_text(encoding="utf-8")
        )
        runnable = next(
            m for m in report["metrics"] if m["metric_id"] == "runnable_coverage"
        )
        assert runnable["value"] == 1.0

    def test_context_exceeded_is_terminal_for_resume(self, tmp_path):
        config = baseline_config("full_history", window=100)
        runner = make_runner(tmp_path, config, run_id="run-ce-resume")
        outcome = runner.run()
        assert outcome.context_exceeded == 1
        runner2 = make_runner(tmp_path, config, run_id="run-ce-resume")
        resumed = runner2.resume()
        assert resumed.reused == 1
        assert resumed.context_exceeded == 1
        assert resumed.re_run == 0
