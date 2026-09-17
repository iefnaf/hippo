"""M1 failure handling, idempotency and checkpointed resume (issue #5).

Covers the five acceptance criteria:

1. response loss after submit: idempotent adapters apply exactly one
   logical mutation; non-idempotent adapters isolate + replay and the
   quarantined old task never pollutes the fresh space;
2. await timeouts, unparseable judge output and transient errors all
   record stage/reason/timing/usage; failures never masquerade as empty
   evidence or fabricated verdicts;
3. low scores are never a retry reason;
4. resume only completes unfinished work and refuses incompatible
   checkpoints (fingerprint / space identity / schema version) with a
   new-run hint;
5. retry and replay usage counts into run totals and is reported
   separately from logical usage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.config import load_config_dict, load_config_toml
from eval.contracts.common import ContractError
from eval.datasets.manual import ManualDataset
from eval.judges.fake import FakeJudge, FakeJudgeSpec
from eval.memories.base import MemoryAdapterError
from eval.memories.fake import FakeMemoryAdapter, FakeMemorySpec
from eval.readers.base import ReaderError
from eval.readers.fake import FakeReader, FakeReaderSpec
from eval.runner import OfflineRunner
from eval.runs import AttemptLogArtifact, RunStore

HANDLE = "smoke_single_session_user_0001"
EXAMPLE_CONFIG = "eval/configs/examples/offline_fake.toml"
OPS_CONFIG = "eval/configs/examples/offline_fake_ops.toml"


def custom_config(**overrides):
    memory = overrides.pop(
        "memory",
        {
            "name": "fake-memory",
            "baseline_kind": "adapter",
            "capabilities": ["extractive_evidence", "state_inspection"],
            "config": {"mutation_mode": "sync", "evidence_kinds": ["extractive"]},
        },
    )
    data = {
        "name": "failure-recovery-test",
        "dataset_plan": "manual-fixtures@1",
        "sample_plan_id": "smoke-offline-8",
        "sample_ids": [HANDLE],
        "smoke_subset_ids": [],
        "memory": memory,
        "reader": {
            "model": "fake-reader",
            "model_family": "family-r",
            "base_url": "offline://fake",
            "temperature": 0.0,
            "max_output_tokens": 1024,
            "tokenizer_id": "test:char-v1",
            "counting_mode": "test",
        },
        "judge": {
            "model": "fake-judge",
            "model_family": "family-j",
            "base_url": "offline://fake",
            "temperature": 0.0,
            "protocol_id": "longmemeval-yes-no@1",
            "protocol_source_commit": "0" * 40,
        },
        "evidence_token_budget": overrides.pop("evidence_token_budget", 4096),
        "run_params": overrides.pop(
            "run_params", {"max_retries": 0, "backoff_base_s": 0.001}
        ),
    }
    data.update(overrides)
    return load_config_dict(data)


def memory_plan(spec_kwargs, baseline="adapter"):
    """Memory plan whose declared capabilities equal the spec's set."""
    spec = FakeMemorySpec(**spec_kwargs)
    return {
        "name": "fake-memory",
        "baseline_kind": baseline,
        "capabilities": sorted(spec.capabilities()),
        "config": spec_kwargs,
    }


def make_runner(
    tmp_path: Path,
    config,
    adapter=None,
    reader=None,
    judge=None,
    run_id=None,
    monotonic=None,
):
    dataset = ManualDataset.load_default()
    adapter = adapter or FakeMemoryAdapter(
        FakeMemorySpec.from_memory_plan(config.memory)
    )
    reader = reader or FakeReader(FakeReaderSpec.from_reader_plan(config.reader))
    judge = judge or FakeJudge(FakeJudgeSpec.from_judge_plan(config.judge))
    run_id = run_id or f"run-test-{config.fingerprint()[:8]}"
    store = RunStore(tmp_path / "runs", run_id)
    kwargs = {}
    if monotonic is not None:
        kwargs["monotonic"] = monotonic
    return (
        OfflineRunner(
            config=config,
            dataset=dataset,
            adapter=adapter,
            reader=reader,
            judge=judge,
            store=store,
            run_id=run_id,
            **kwargs,
        ),
        adapter,
    )


def expected_message_count(handle: str) -> int:
    return sum(
        len(session.messages)
        for session in ManualDataset.load_default().iter_sessions(handle)
    )


def namespace_of(config, handle: str) -> str:
    return ManualDataset.load_default().namespace_for(
        handle, config.sample_plan_id
    )


def load_attempt_log(run_dir: Path, result) -> AttemptLogArtifact:
    store = RunStore(run_dir.parent, run_dir.name)
    doc = json.loads(
        store.resolve_ref(result.artifact_refs["attempts"]).read_text(
            encoding="utf-8"
        )
    )
    return AttemptLogArtifact.model_validate(doc)


class FlakyRetrieveAdapter(FakeMemoryAdapter):
    """First retrieve calls fail with a transient, effect=none error."""

    def __init__(self, spec: FakeMemorySpec, fail_times: int = 1) -> None:
        super().__init__(spec)
        self.fail_times = fail_times

    def retrieve(self, namespace, request):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise MemoryAdapterError(
                "backend_overloaded",
                "simulated transient backend blip",
                transient=True,
                effect="none",
            )
        return super().retrieve(namespace, request)


class CrashingResetAdapter(FakeMemoryAdapter):
    """Simulates process death right before given samples start."""

    def __init__(self, spec: FakeMemorySpec, crash_namespaces) -> None:
        super().__init__(spec)
        self.crash_namespaces = set(crash_namespaces)

    def reset(self, namespace):
        if namespace in self.crash_namespaces:
            raise KeyboardInterrupt("simulated process crash")
        return super().reset(namespace)


class FlakyReader(FakeReader):
    """First answer calls fail with a transient reader error."""

    def __init__(self, spec: FakeReaderSpec, fail_times: int = 0) -> None:
        super().__init__(spec)
        self.fail_times = fail_times

    def answer(self, question, prepared):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ReaderError(
                "reader_backend_failed",
                "simulated transient reader failure",
                transient=True,
            )
        return super().answer(question, prepared)


class SteadyMonotonic:
    """Deterministic clock advancing a fixed step per call."""

    def __init__(self, step: float) -> None:
        self.step = step
        self.now = 0.0

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


fast_params = {"max_retries": 2, "backoff_base_s": 0.001}


class TestResponseLossIdempotency:
    """AC1: submit-response loss and the idempotency split."""

    def test_idempotent_adapter_applies_exactly_one_logical_mutation(
        self, tmp_path: Path
    ):
        config = custom_config(
            memory=memory_plan(
                {
                    "mutation_mode": "sync",
                    "evidence_kinds": ["extractive"],
                    "idempotent": True,
                    "response_loss_calls": 1,
                }
            ),
            run_params=dict(fast_params),
        )
        runner, adapter = make_runner(tmp_path, config)
        outcome = runner.run()
        assert outcome.failed == 0
        result = outcome.results[0].result
        assert result.qa_status == "scored"

        ns = namespace_of(config, HANDLE)
        n_sessions = len(ManualDataset.load_default().iter_sessions(HANDLE))
        # ONE logical mutation per session: the replayed submit returned
        # the stored receipt, so no suffixed duplicate entries exist.
        assert adapter.stored_message_count(ns) == expected_message_count(HANDLE)
        ingest_calls = [
            e for e in adapter.journal if e["method"] == "ingest"
        ]
        assert len(ingest_calls) == n_sessions + 1  # lost response + re-submit

        ingest_attempts = [
            a for a in result.attempts if a.stage == "ingest"
        ]
        assert len(ingest_attempts) == n_sessions + 1
        lost, resubmitted = ingest_attempts[0], ingest_attempts[1]
        assert lost.outcome == "error"
        assert lost.error.code == "response_lost"
        assert lost.error.effect == "possible"
        assert lost.attempt_kind == "logical"
        assert resubmitted.outcome == "returned"
        assert resubmitted.attempt_kind == "retry"

    def test_non_idempotent_adapter_isolates_and_replays(
        self, tmp_path: Path
    ):
        config = custom_config(
            memory=memory_plan(
                {
                    "mutation_mode": "async",
                    "evidence_kinds": ["extractive"],
                    "response_loss_calls": 1,
                }
            ),
            run_params=dict(fast_params),
        )
        runner, adapter = make_runner(tmp_path, config)
        outcome = runner.run()
        assert outcome.failed == 0
        result = outcome.results[0].result
        assert result.qa_status == "scored"
        assert result.stage_states["ingest"] == "completed"

        ns = namespace_of(config, HANDLE)
        # The uncertain old task was isolated with the discarded space.
        quarantined = adapter.quarantined_operation_ids(ns)
        assert len(quarantined) == 1
        # ...and it never polluted the replayed space: the live namespace
        # holds exactly the planned messages, nothing twice.
        assert adapter.stored_message_count(ns) == expected_message_count(HANDLE)
        assert quarantined[0] not in adapter._spaces.get(ns, {})

        # The isolation reset and the replayed operations are recorded
        # as recovery usage.
        log = load_attempt_log(Path(outcome.run_dir), result)
        resets = [
            e for e in log.entries if e.method == "reset"
        ]
        assert any(e.attempt_kind == "replay" for e in resets)
        replayed = [
            a for a in result.attempts if a.attempt_kind == "replay"
        ]
        assert replayed
        assert {a.stage for a in replayed} <= {"ingest", "await_ready"}
        lost = [
            a
            for a in result.attempts
            if a.stage == "ingest" and a.outcome == "error"
        ]
        assert lost[0].error.code == "response_lost"

    def test_same_operation_id_with_different_input_is_rejected(self):
        from eval.contracts.adapter import Message, Session

        adapter = FakeMemoryAdapter(
            FakeMemorySpec(
                mutation_mode="sync",
                evidence_kinds=["extractive"],
                idempotent=True,
            )
        )
        adapter.open("ns")
        first = Session(
            session_id="s1",
            occurred_at="2026-09-01",
            messages=[Message(msg_id="m1", role="user", content="约定一")],
        )
        second = Session(
            session_id="s1",
            occurred_at="2026-09-01",
            messages=[Message(msg_id="m1", role="user", content="约定二")],
        )
        adapter.ingest("ns", first, "op-1")
        with pytest.raises(MemoryAdapterError) as excinfo:
            adapter.ingest("ns", second, "op-1")
        assert excinfo.value.code == "operation_id_input_conflict"

    def test_same_id_resubmit_while_pending_never_applies_twice(self):
        from eval.contracts.adapter import Message, Session

        adapter = FakeMemoryAdapter(
            FakeMemorySpec(
                mutation_mode="async",
                evidence_kinds=["extractive"],
                idempotent=True,
            )
        )
        adapter.open("ns")
        session = Session(
            session_id="s1",
            occurred_at="2026-09-01",
            messages=[Message(msg_id="m1", role="user", content="约定一")],
        )
        first = adapter.ingest("ns", session, "op-1")
        assert first.status == "accepted"
        inflight = adapter.ingest("ns", session, "op-1")
        assert inflight.status == "accepted"
        assert adapter.stored_message_count("ns") == 1  # one logical mutation
        receipt = adapter.await_ready("ns", "op-1", 1)
        assert receipt.status == "completed"


class TestAwaitReadyTimeout:
    """AC2: waiting past the wall-clock budget is a recorded failure."""

    def test_never_ready_backend_fails_with_recorded_timeout(
        self, tmp_path: Path
    ):
        config = custom_config(
            memory=memory_plan(
                {
                    "mutation_mode": "async",
                    "evidence_kinds": ["extractive"],
                    "idempotent": True,
                    "never_ready": True,
                }
            ),
            run_params={
                "max_retries": 0,
                "backoff_base_s": 0.001,
                "await_ready_timeout_s": 0.05,
            },
        )
        runner, _ = make_runner(
            tmp_path, config, monotonic=SteadyMonotonic(0.02)
        )
        outcome = runner.run()
        result = outcome.results[0].result
        assert result.qa_status == "failed"
        assert result.failed_stage == "await_ready"

        timeout_attempts = [
            a
            for a in result.attempts
            if a.stage == "await_ready" and a.outcome == "error"
        ]
        assert len(timeout_attempts) == 1
        attempt = timeout_attempts[0]
        assert attempt.error.code == "await_ready_timeout"
        assert attempt.error.effect == "possible"
        assert attempt.elapsed_ms == pytest.approx(50.0)
        assert attempt.attempt_kind == "logical"
        # polls before the timeout were recorded with timing as well
        polls = [
            a
            for a in result.attempts
            if a.stage == "await_ready" and a.outcome == "returned"
        ]
        assert polls and all(a.elapsed_ms is not None for a in polls)
        # the failure is honest: no fabricated verdict, later stages pending
        assert result.correct is None
        assert result.attribution is None
        assert result.stage_states["read"] == "pending"
        assert result.stage_states["judge"] == "pending"

    def test_never_ready_requires_async_mutation(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            FakeMemorySpec(
                mutation_mode="sync",
                evidence_kinds=["extractive"],
                never_ready=True,
            )


class TestAttemptRecording:
    """AC2/AC3: transient retries recorded; low scores never retried."""

    def test_transient_reader_error_retries_with_full_records(
        self, tmp_path: Path
    ):
        config = custom_config(run_params=dict(fast_params))
        reader = FlakyReader(
            FakeReaderSpec.from_reader_plan(config.reader), fail_times=2
        )
        runner, _ = make_runner(tmp_path, config, reader=reader)
        outcome = runner.run()
        result = outcome.results[0].result
        assert result.qa_status == "scored"

        read_attempts = [a for a in result.attempts if a.stage == "read"]
        assert len(read_attempts) == 3
        assert [a.attempt_kind for a in read_attempts] == [
            "logical",
            "retry",
            "retry",
        ]
        failed = [a for a in read_attempts if a.outcome == "error"]
        assert {a.error.code for a in failed} == {"reader_backend_failed"}
        assert all(a.elapsed_ms is not None for a in read_attempts)
        assert read_attempts[-1].usage is not None
        assert read_attempts[-1].usage.llm_call_count == 1

    def test_transient_retrieve_error_retries(self, tmp_path: Path):
        config = custom_config(run_params=dict(fast_params))
        spec = FakeMemorySpec.from_memory_plan(config.memory)
        adapter = FlakyRetrieveAdapter(spec, fail_times=1)
        runner, _ = make_runner(tmp_path, config, adapter=adapter)
        outcome = runner.run()
        result = outcome.results[0].result
        assert result.qa_status == "scored"
        retrieve_attempts = [
            a for a in result.attempts if a.stage == "retrieve"
        ]
        assert len(retrieve_attempts) == 2
        assert [a.attempt_kind for a in retrieve_attempts] == ["logical", "retry"]
        assert retrieve_attempts[0].error.code == "backend_overloaded"
        assert retrieve_attempts[0].error.effect == "none"

    def test_unparseable_judge_output_fails_without_retry(
        self, tmp_path: Path
    ):
        config = custom_config(
            judge={
                "model": "fake-judge",
                "model_family": "family-j",
                "base_url": "offline://fake",
                "temperature": 0.0,
                "protocol_id": "longmemeval-yes-no@1",
                "protocol_source_commit": "0" * 40,
                "extra": {"verdict_rule": "unparseable"},
            },
            run_params=dict(fast_params),
        )
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        result = outcome.results[0].result
        assert result.qa_status == "failed"
        assert result.failed_stage == "judge"

        judge_attempts = [a for a in result.attempts if a.stage == "judge"]
        assert len(judge_attempts) == 1  # non-transient: never retried
        attempt = judge_attempts[0]
        assert attempt.outcome == "error"
        assert attempt.error.code == "judge_output_unparseable"
        assert attempt.elapsed_ms is not None
        # the verdict was never fabricated: no correct/attribution
        assert result.correct is None
        assert result.attribution is None

    def test_wrong_answer_is_never_a_retry_reason(self, tmp_path: Path):
        config = custom_config(
            judge={
                "model": "fake-judge",
                "model_family": "family-j",
                "base_url": "offline://fake",
                "temperature": 0.0,
                "protocol_id": "longmemeval-yes-no@1",
                "protocol_source_commit": "0" * 40,
                "extra": {"verdict_rule": "always_wrong"},
            },
            run_params=dict(fast_params),
        )
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        result = outcome.results[0].result
        assert result.qa_status == "scored"
        assert result.correct is False
        # exactly one reader call and one judge call, nothing recovered
        assert [a.attempt_kind for a in result.attempts] == ["logical"] * len(
            result.attempts
        )
        read_attempts = [a for a in result.attempts if a.stage == "read"]
        judge_attempts = [a for a in result.attempts if a.stage == "judge"]
        assert len(read_attempts) == 1
        assert len(judge_attempts) == 1


class TestUsageSplit:
    """AC5: retry/replay usage counts into totals, reported separately."""

    def test_retry_and_replay_usage_reported_separately(
        self, tmp_path: Path
    ):
        from eval.report import Reporter, render_markdown

        config = custom_config(
            memory=memory_plan(
                {
                    "mutation_mode": "async",
                    "evidence_kinds": ["extractive"],
                    "response_loss_calls": 1,
                }
            ),
            run_params=dict(fast_params),
        )
        reader = FlakyReader(
            FakeReaderSpec.from_reader_plan(config.reader), fail_times=1
        )
        runner, _ = make_runner(tmp_path, config, reader=reader)
        outcome = runner.run()
        assert outcome.failed == 0

        report = Reporter(Path(outcome.run_dir)).build()
        usage = report["cost_model"]["usage_totals"]
        assert usage["logical"]["attempts"] > 0
        assert usage["retry"]["attempts"] > 0  # the retried reader call
        assert usage["replay"]["attempts"] > 0  # the isolated replay cycle
        total = usage["total"]
        parts = ("logical", "retry", "replay")
        assert total["attempts"] == sum(usage[k]["attempts"] for k in parts)
        assert total["input_tokens"] == sum(
            usage[k]["input_tokens"] for k in parts
        )
        assert total["output_tokens"] == sum(
            usage[k]["output_tokens"] for k in parts
        )

        # the buckets equal the per-attempt sums over samples.jsonl
        result = outcome.results[0].result
        log = load_attempt_log(Path(outcome.run_dir), result)
        for kind in parts:
            entries = [e for e in log.entries if e.attempt_kind == kind]
            assert usage[kind]["attempts"] >= len(entries)
        markdown = render_markdown(report)
        assert "用量拆分" in markdown
        assert "replay（隔离重放/断点补跑）" in markdown


class TestResume:
    """AC4: checkpointed resume completes only unfinished work."""

    THREE = [
        "smoke_single_session_user_0001",
        "smoke_temporal_reasoning_0001",
        "smoke_multi_session_0001",
    ]

    def test_resume_after_crash_only_runs_missing_samples(
        self, tmp_path: Path
    ):
        config = custom_config(sample_ids=list(self.THREE))
        spec = FakeMemorySpec.from_memory_plan(config.memory)
        crash_ns = namespace_of(config, self.THREE[2])
        runner = OfflineRunner(
            config=config,
            dataset=ManualDataset.load_default(),
            adapter=CrashingResetAdapter(spec, {crash_ns}),
            reader=FakeReader(FakeReaderSpec.from_reader_plan(config.reader)),
            judge=FakeJudge(FakeJudgeSpec.from_judge_plan(config.judge)),
            store=RunStore(tmp_path / "runs", "run-crash"),
            run_id="run-crash",
        )
        with pytest.raises(KeyboardInterrupt):
            runner.run()
        run_dir = tmp_path / "runs" / "run-crash"
        lines = (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        before = {
            p.relative_to(run_dir): p.read_bytes()
            for p in sorted((run_dir / "artifacts").rglob("*.json"))
        }

        runner2, _ = make_runner(tmp_path, config, run_id="run-crash")
        outcome = runner2.resume()
        assert outcome.scored == 1
        assert outcome.re_run == 1
        assert outcome.reused == 2
        assert outcome.re_run_handles == [self.THREE[2]]
        after = {
            p.relative_to(run_dir): p.read_bytes()
            for p in sorted((run_dir / "artifacts").rglob("*.json"))
        }
        # prior artifacts survive byte-identically
        for key, payload in before.items():
            assert after[key] == payload
        lines = (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        handles = [json.loads(line)["result"]["sample_handle"] for line in lines]
        assert sorted(set(handles)) == sorted(self.THREE)
        assert (run_dir / "resume.jsonl").exists()
        assert (run_dir / "report.json").exists()

    def test_resume_of_read_failure_reuses_saved_evidence(
        self, tmp_path: Path
    ):
        config = custom_config(run_params={"max_retries": 0, "backoff_base_s": 0.001})
        flaky = FlakyReader(
            FakeReaderSpec.from_reader_plan(config.reader), fail_times=5
        )
        runner, _ = make_runner(
            tmp_path, config, reader=flaky, run_id="run-readfail"
        )
        outcome = runner.run()
        result = outcome.results[0].result
        assert result.qa_status == "failed"
        assert result.failed_stage == "read"
        prior_raw_ref = result.artifact_refs["raw_evidence"]
        prior_attempts = len(result.attempts)

        runner2, adapter2 = make_runner(
            tmp_path, config, run_id="run-readfail"
        )
        outcome2 = runner2.resume()
        resumed = outcome2.results[0].result
        assert resumed.qa_status == "scored"
        assert resumed.artifact_refs["raw_evidence"] == prior_raw_ref
        assert len(resumed.attempts) > prior_attempts  # prior attempts merged
        # the resume made NO memory calls: saved evidence was reused
        assert adapter2.received_requests == []
        methods = {e["method"] for e in adapter2.journal}
        assert "ingest" not in methods
        assert "retrieve" not in methods
        # all resume calls are marked as recovery usage
        new_attempts = resumed.attempts[prior_attempts:]
        assert new_attempts
        assert all(a.attempt_kind == "replay" for a in new_attempts)

    def test_resume_rejects_config_fingerprint_mismatch(self, tmp_path: Path):
        config = custom_config()
        runner, _ = make_runner(tmp_path, config, run_id="run-fp")
        runner.run()
        other = custom_config(evidence_token_budget=2048)
        runner2, _ = make_runner(tmp_path, other, run_id="run-fp")
        with pytest.raises(ContractError) as excinfo:
            runner2.resume()
        assert excinfo.value.code == "resume_config_fingerprint_mismatch"
        assert "新建" in excinfo.value.message or "NEW run" in excinfo.value.message

    def test_resume_rejects_incompatible_checkpoint_schema(
        self, tmp_path: Path
    ):
        config = custom_config()
        runner, _ = make_runner(tmp_path, config, run_id="run-schema")
        runner.run()
        run_dir = tmp_path / "runs" / "run-schema"
        snapshot = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        snapshot["schema_version"] = 99
        (run_dir / "config.json").write_text(
            json.dumps(snapshot, ensure_ascii=False), encoding="utf-8"
        )
        runner2, _ = make_runner(tmp_path, config, run_id="run-schema")
        with pytest.raises(ContractError) as excinfo:
            runner2.resume()
        assert excinfo.value.code == "resume_checkpoint_schema_incompatible"
        assert "新建" in excinfo.value.message or "NEW run" in excinfo.value.message

    def test_resume_rejects_space_identity_mismatch(self, tmp_path: Path):
        config = custom_config()
        runner, _ = make_runner(tmp_path, config, run_id="run-space")
        runner.run()
        run_dir = tmp_path / "runs" / "run-space"
        path = run_dir / "samples.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        doc = json.loads(lines[-1])
        doc["result"]["namespace"] = doc["result"]["namespace"] + "-forged"
        lines[-1] = json.dumps(doc, ensure_ascii=False)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        runner2, _ = make_runner(tmp_path, config, run_id="run-space")
        with pytest.raises(ContractError) as excinfo:
            runner2.resume()
        assert excinfo.value.code == "resume_space_identity_mismatch"
        assert "新建" in excinfo.value.message or "NEW run" in excinfo.value.message

    def test_operations_resume_re_runs_missing_checks(self, tmp_path: Path):
        from eval.memories.fake import build_fake_adapter
        from eval.operations import OperationsRunner

        config = load_config_toml(OPS_CONFIG)
        store = RunStore(tmp_path / "runs", "run-ops-resume")
        runner = OperationsRunner(
            config=config,
            adapter=build_fake_adapter(config.memory),
            store=store,
            run_id="run-ops-resume",
        )
        outcome = runner.run()
        assert outcome.failed == 0
        run_dir = tmp_path / "runs" / "run-ops-resume"
        path = run_dir / "samples.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 5
        path.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")

        runner2 = OperationsRunner(
            config=config,
            adapter=build_fake_adapter(config.memory),
            store=RunStore(tmp_path / "runs", "run-ops-resume"),
            run_id="run-ops-resume",
        )
        outcome2 = runner2.resume()
        assert outcome2.failed == 0
        handles = [
            json.loads(line)["result"]["sample_handle"]
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        assert sorted(set(handles)) == sorted(config.sample_ids)
        summary = json.loads(
            (run_dir / "operations_summary.json").read_text(encoding="utf-8")
        )
        assert summary["planned_checks"] == 5
        assert summary["passed"] == 5

        # Issue #5 leftover: operations resume writes the same audit
        # trail as the qa runner (resume.jsonl), recording what was
        # reused vs re-run.
        marker_path = run_dir / "resume.jsonl"
        assert marker_path.exists()
        marker = json.loads(marker_path.read_text(encoding="utf-8").splitlines()[-1])
        assert marker["suite"] == "operations"
        assert marker["run_id"] == "run-ops-resume"
        assert marker["reused"] == 2
        assert marker["re_run"] == 3
        assert len(marker["re_run_handles"]) == 3
        assert "resumed_at" in marker


class TestResumeCli:
    """AC4 via the CLI surface."""

    def test_cli_resume_completes_truncated_run(
        self, capsys, tmp_path: Path
    ):
        from eval.cli import main

        out = tmp_path / "runs"
        code = main(
            ["run", "--config", EXAMPLE_CONFIG, "--out", str(out)]
        )
        assert code == 0
        run_dir = Path(
            json.loads(capsys.readouterr().out)["run_dir"]
        )
        path = run_dir / "samples.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 8
        path.write_text("\n".join(lines[:4]) + "\n", encoding="utf-8")

        code = main(
            ["resume", "--config", EXAMPLE_CONFIG, "--run", str(run_dir)]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["command"] == "resume"
        assert payload["re_run"] == 4
        assert payload["reused"] == 4
        lines = path.read_text(encoding="utf-8").splitlines()
        handles = {json.loads(line)["result"]["sample_handle"] for line in lines}
        assert len(handles) == 8
        assert (run_dir / "report.json").exists()
        assert (run_dir / "resume.jsonl").exists()

    def test_cli_resume_rejects_changed_config(
        self, capsys, tmp_path: Path
    ):
        from eval.cli import main

        out = tmp_path / "runs"
        code = main(
            ["run", "--config", EXAMPLE_CONFIG, "--out", str(out)]
        )
        assert code == 0
        run_dir = Path(
            json.loads(capsys.readouterr().out)["run_dir"]
        )
        original = Path(EXAMPLE_CONFIG).read_text(encoding="utf-8")
        changed = original.replace("evidence_token_budget = 4096", "evidence_token_budget = 2048")
        assert changed != original
        other = tmp_path / "changed.toml"
        other.write_text(changed, encoding="utf-8")

        code = main(
            ["resume", "--config", str(other), "--run", str(run_dir)]
        )
        assert code == 2
        err = capsys.readouterr().err
        assert "resume_config_fingerprint_mismatch" in err
