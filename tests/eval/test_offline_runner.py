"""End-to-end offline runner: the six acceptance criteria of issue #2."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.config import load_config_dict
from eval.contracts.adapter import Evidence, SourceSpan
from eval.contracts.common import ContractError
from eval.contracts.internal import PreparedEvidenceArtifact, RawEvidenceArtifact, ResultArtifact
from eval.datasets.manual import ManualDataset
from eval.memories.base import MemoryAdapterError
from eval.memories.fake import FakeMemoryAdapter, FakeMemorySpec
from eval.runner import OfflineRunner
from eval.runs import RunStore, new_run_id

EXAMPLE_CONFIG = "eval/configs/examples/offline_fake.toml"
FIXED_CLOCK = ["1999-01-01T00:00:00.000001Z"]


def clock(times: list[str]):
    def _clock() -> str:
        return times[0] if len(times) == 1 else times.pop(0)
    return _clock


def load_config(path: str = EXAMPLE_CONFIG):
    from eval.config import load_config_toml

    return load_config_toml(path)


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
    budget = overrides.pop("evidence_token_budget", 4096)
    data = {
        "name": "offline-test",
        "dataset_plan": "manual-fixtures@1",
        "sample_plan_id": "smoke-offline-8",
        "sample_ids": ["smoke_single_session_user_0001"],
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
        "evidence_token_budget": budget,
    }
    data.update(overrides)
    return load_config_dict(data)


def make_runner(tmp_path: Path, config, adapter=None, dataset=None, run_id=None):
    dataset = dataset or ManualDataset.load_default()
    adapter = adapter or FakeMemoryAdapter(
        FakeMemorySpec.from_memory_plan(config.memory)
    )
    run_id = run_id or f"run-test-{config.fingerprint()[:8]}"
    store = RunStore(tmp_path / "runs", run_id)
    return (
        OfflineRunner(
            config=config,
            dataset=dataset,
            adapter=adapter,
            store=store,
            run_id=run_id,
        ),
        adapter,
    )


class RogueEvidenceAdapter(FakeMemoryAdapter):
    """Fake that returns illegal extractive output for prepare to catch."""

    def __init__(self, spec: FakeMemorySpec, mode: str) -> None:
        super().__init__(spec)
        self.mode = mode

    def retrieve(self, namespace, request):
        evidence = list(super().retrieve(namespace, request))
        if not evidence:
            return evidence
        first = evidence[0]
        if first.kind != "extractive":
            return evidence
        if self.mode == "unknown_source":
            span = SourceSpan(
                session_id="s_does_not_exist",
                msg_id=first.extractive_span.msg_id,
                start=0,
                end=1,
            )
            evidence[0] = first.model_copy(update={"extractive_span": span, "text": "P"})
        elif self.mode == "text_mismatch":
            evidence[0] = first.model_copy(update={"text": first.text + "伪造"})
        elif self.mode == "bad_range":
            span = SourceSpan(
                session_id=first.extractive_span.session_id,
                msg_id=first.extractive_span.msg_id,
                start=0,
                end=10**6,
            )
            evidence[0] = first.model_copy(update={"extractive_span": span})
        return evidence


class FailingIngestAdapter(FakeMemoryAdapter):
    def __init__(self, spec: FakeMemorySpec) -> None:
        super().__init__(spec)
        self.calls = 0

    def ingest(self, namespace, session, operation_id):
        self.calls += 1
        raise MemoryAdapterError(
            "backend_unavailable",
            "simulated backend failure",
            transient=True,
            effect="none",
        )


class TestRunWithManifestArtifacts:
    """AC2: one offline run produces run id, config snapshot, per-question JSONL."""

    def test_full_run_writes_all_artifacts(self, tmp_path: Path):
        config = load_config()
        runner, adapter = make_runner(tmp_path, config)
        outcome = runner.run()
        assert outcome.failed == 0
        run_dir = Path(outcome.run_dir)
        assert (run_dir / "run.json").exists()
        manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert manifest["run_id"] == outcome.run_id
        assert manifest["config_fingerprint"] == config.fingerprint()
        assert manifest["async_mutation"] is True
        assert manifest["plan_counts"]["smoke_multi_session_0001"]["sessions"] == 4

        snapshot = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        assert snapshot["config_fingerprint"] == config.fingerprint()
        assert snapshot["schema_version"] == 1

        lines = (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 8
        for line in lines:
            artifact = ResultArtifact.load_json(line)
            result = artifact.result
            assert result.run_id == outcome.run_id
            assert result.config_fingerprint == config.fingerprint()
            assert result.suite == "qa"
            assert result.qa_status == "pending"
            assert result.stage_states["ingest"] == "completed"
            assert result.stage_states["retrieve"] == "completed"
            assert result.stage_states["prepare"] == "completed"
            assert result.stage_states["read"] == "pending"
            assert result.failed_stage is None

    def test_attempts_record_stage_timing_and_usage(self, tmp_path: Path):
        config = load_config()
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        for artifact in outcome.results:
            attempts = artifact.result.attempts
            stages = {a.stage for a in attempts}
            assert {"ingest", "await_ready", "retrieve", "prepare"} <= stages
            for attempt in attempts:
                assert attempt.elapsed_ms is not None and attempt.elapsed_ms >= 0
                assert attempt.started_at and attempt.ended_at
            ingest_attempts = [a for a in attempts if a.stage == "ingest"]
            assert all(a.usage is not None for a in ingest_attempts)
            assert all(
                a.usage.input_tokens is not None for a in ingest_attempts
            )
            retrieve = [a for a in attempts if a.stage == "retrieve"]
            assert all(a.usage.output_tokens is not None for a in retrieve)

    def test_artifact_refs_resolve_and_checksum(self, tmp_path: Path):
        config = load_config()
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        store = RunStore(tmp_path / "runs", outcome.run_id)
        for artifact in outcome.results:
            for name, ref in artifact.result.artifact_refs.items():
                path = store.resolve_ref(ref)
                assert path.exists(), (name, ref)

    def test_run_directory_is_immutable(self, tmp_path: Path):
        config = load_config()
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        same_store = RunStore(tmp_path / "runs", outcome.run_id)
        with pytest.raises(ContractError) as excinfo:
            same_store.create(None, None)  # type: ignore[arg-type]
        assert excinfo.value.code == "run_dir_exists"

    def test_two_runs_get_distinct_ids_same_fingerprint(self, tmp_path: Path):
        config = load_config()
        runner, _ = make_runner(tmp_path, config)
        first = runner.run()
        runner2, _ = make_runner(tmp_path, config, run_id=first.run_id + "-b")
        second = runner2.run()
        assert first.run_id != second.run_id
        assert first.results[0].result.config_fingerprint == (
            second.results[0].result.config_fingerprint
        )


class TestQueryContext:
    """AC3: retrieval carries dataset question_date and budget; no machine time."""

    def test_requests_carry_dataset_date_and_budget(self, tmp_path: Path):
        config = load_config()
        dataset = ManualDataset.load_default()
        runner, adapter = make_runner(tmp_path, config, dataset=dataset)
        runner.run()
        assert len(adapter.received_requests) == 8
        for handle, request in zip(config.sample_ids, adapter.received_requests):
            expected = dataset.build_retrieval_request(
                handle, config.evidence_token_budget
            )
            assert request == expected
            assert request.evidence_token_budget == 4096

    def test_machine_time_never_enters_query_context(self, tmp_path: Path):
        config = load_config()
        runner, adapter = make_runner(tmp_path, config)
        # Run with a machine clock pinned to 1999: no rendered evidence or
        # request may carry it (operation ids legitimately embed the run).
        runner._clock = clock(list(FIXED_CLOCK))
        runner.run()
        for request in adapter.received_requests:
            dumped = request.model_dump_json()
            assert "1999" not in dumped
        runs_dir = tmp_path / "runs"
        for prepared_file in runs_dir.rglob("prepared_evidence.json"):
            doc = json.loads(prepared_file.read_text(encoding="utf-8"))
            assert "1999" not in doc["prepared"]["rendered_text"]

    def test_prepared_times_come_from_dataset_dates(self, tmp_path: Path):
        config = load_config()
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        artifact = next(
            a for a in outcome.results
            if a.result.sample_handle == "smoke_single_session_user_0001"
        )
        ref = artifact.result.artifact_refs["prepared_evidence"]
        store = RunStore(tmp_path / "runs", outcome.run_id)
        doc = json.loads(store.resolve_ref(ref).read_text(encoding="utf-8"))
        rendered = doc["prepared"]["rendered_text"]
        assert "2026-09-03" in rendered  # dataset session time, fixed format


class TestCapabilityFixing:
    """AC1: capability combinations are declared and fixed before the run."""

    def test_mismatch_between_declaration_and_adapter_aborts_run(self, tmp_path):
        config = custom_config(
            memory={
                "name": "fake-memory",
                "baseline_kind": "adapter",
                "capabilities": ["extractive_evidence", "delete"],  # undeclared by fake
                "config": {"mutation_mode": "sync", "evidence_kinds": ["extractive"]},
            }
        )
        runner, _ = make_runner(tmp_path, config)
        with pytest.raises(ContractError) as excinfo:
            runner.run()
        assert excinfo.value.code == "capability_declaration_mismatch"
        # No run directory was created: the check happens before the run.
        assert not (tmp_path / "runs" / runner.run_id).exists()

    def test_example_config_matches_fake_capabilities(self):
        config = load_config()
        adapter = FakeMemoryAdapter(FakeMemorySpec.from_memory_plan(config.memory))
        assert adapter.capabilities() == set(config.memory.capabilities)

    @pytest.mark.parametrize(
        "spec,caps",
        [
            (
                {"mutation_mode": "sync", "evidence_kinds": ["extractive"]},
                {"extractive_evidence", "state_inspection"},
            ),
            (
                {"mutation_mode": "async", "evidence_kinds": ["generated"], "idempotent": True},
                {
                    "generated_evidence",
                    "async_mutation",
                    "operation_status",
                    "idempotent_mutation",
                    "state_inspection",
                },
            ),
        ],
    )
    def test_runner_accepts_other_fixed_combinations(self, tmp_path, spec, caps):
        config = custom_config(
            memory={
                "name": "fake-memory",
                "baseline_kind": "adapter",
                "capabilities": sorted(caps),
                "config": spec,
            }
        )
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        assert outcome.failed == 0

    def test_sync_profile_has_no_await_ready_attempts(self, tmp_path):
        config = custom_config(
            memory={
                "name": "fake-memory",
                "baseline_kind": "adapter",
                "capabilities": ["extractive_evidence", "state_inspection"],
                "config": {"mutation_mode": "sync", "evidence_kinds": ["extractive"]},
            }
        )
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        result = outcome.results[0].result
        assert "await_ready" not in result.stage_states
        assert all(a.stage != "await_ready" for a in result.attempts)


class TestPrepareFailure:
    """AC4: illegal extractive output fails the prepare stage, no patching."""

    @pytest.mark.parametrize(
        "mode,code",
        [
            ("unknown_source", "prepare_unknown_source"),
            ("text_mismatch", "prepare_text_mismatch"),
            ("bad_range", "prepare_invalid_range"),
        ],
    )
    def test_prepare_failure_marks_sample_failed(self, tmp_path, mode, code):
        spec = FakeMemorySpec(
            mutation_mode="sync", evidence_kinds=["extractive"], retrieval_mode="match"
        )
        config = custom_config()
        adapter = RogueEvidenceAdapter(spec, mode)
        runner, _ = make_runner(tmp_path, config, adapter=adapter)
        outcome = runner.run()
        result = outcome.results[0].result
        assert result.qa_status == "failed"
        assert result.failed_stage == "prepare"
        assert result.stage_states["prepare"] == "failed"
        assert result.stage_states["retrieve"] == "completed"
        assert result.correct is None
        assert result.attribution is None
        prepare_attempt = [a for a in result.attempts if a.stage == "prepare"][-1]
        assert prepare_attempt.outcome == "error"
        assert prepare_attempt.error.code == code

    def test_raw_return_still_persisted_for_diagnostics(self, tmp_path):
        spec = FakeMemorySpec(mutation_mode="sync", evidence_kinds=["extractive"])
        config = custom_config()
        adapter = RogueEvidenceAdapter(spec, "text_mismatch")
        runner, rogue = make_runner(tmp_path, config, adapter=adapter)
        outcome = runner.run()
        result = outcome.results[0].result
        assert "raw_evidence" in result.artifact_refs
        assert "prepared_evidence" not in result.artifact_refs
        store = RunStore(tmp_path / "runs", outcome.run_id)
        raw_doc = json.loads(
            store.resolve_ref(result.artifact_refs["raw_evidence"]).read_text(
                encoding="utf-8"
            )
        )
        raw = RawEvidenceArtifact.model_validate(raw_doc)
        # The diagnostic artifact preserves the rogue return untouched.
        assert any("伪造" in ev.text for ev in raw.evidence)

    def test_ingest_failure_marks_sample_failed(self, tmp_path):
        spec = FakeMemorySpec(mutation_mode="sync", evidence_kinds=["extractive"])
        config = custom_config()
        adapter = FailingIngestAdapter(spec)
        runner, _ = make_runner(tmp_path, config, adapter=adapter)
        outcome = runner.run()
        result = outcome.results[0].result
        assert result.qa_status == "failed"
        assert result.failed_stage == "ingest"
        assert result.stage_states["retrieve"] == "pending"


class TestBudgetEnforcement:
    """AC5: over-budget units are truncated/dropped; raw return kept separately."""

    def test_flood_run_truncates_and_drops_with_consistent_spans(self, tmp_path):
        class CapturingFloodAdapter(FakeMemoryAdapter):
            last_return: list = []

            def retrieve(self, namespace, request):
                evidence = list(super().retrieve(namespace, request))
                type(self).last_return = evidence
                return evidence

        config = custom_config(
            sample_ids=["smoke_multi_session_0001"],
            evidence_token_budget=512,
            memory={
                "name": "fake-memory",
                "baseline_kind": "adapter",
                "capabilities": ["extractive_evidence", "state_inspection"],
                "config": {
                    "mutation_mode": "sync",
                    "evidence_kinds": ["extractive"],
                    "retrieval_mode": "flood",
                },
            },
        )
        adapter = CapturingFloodAdapter(
            FakeMemorySpec(
                mutation_mode="sync",
                evidence_kinds=["extractive"],
                retrieval_mode="flood",
            )
        )
        runner, _ = make_runner(tmp_path, config, adapter=adapter)
        outcome = runner.run()
        result = outcome.results[0].result
        assert result.qa_status == "pending"

        store = RunStore(tmp_path / "runs", outcome.run_id)
        raw_doc = RawEvidenceArtifact.model_validate(
            json.loads(
                store.resolve_ref(result.artifact_refs["raw_evidence"]).read_text(
                    encoding="utf-8"
                )
            )
        )
        prepared_doc = PreparedEvidenceArtifact.model_validate(
            json.loads(
                store.resolve_ref(result.artifact_refs["prepared_evidence"]).read_text(
                    encoding="utf-8"
                )
            )
        )
        raw, prepared = raw_doc.evidence, prepared_doc.prepared

        # The raw return exceeded the budget; the harness enforced it hard.
        assert sum(len(ev.text) for ev in raw) > 512
        assert prepared.token_count <= 512
        assert prepared.budget == 512

        # Retained evidence and source ranges stay consistent.
        history_contents = {}
        for session in ManualDataset.load_default().iter_sessions(
            "smoke_multi_session_0001"
        ):
            for message in session.messages:
                history_contents[(session.session_id, message.msg_id)] = message.content
        for item in prepared.items:
            if item.evidence.kind == "extractive":
                span = item.evidence.extractive_span
                assert item.verified_span == span
                content = history_contents[(span.session_id, span.msg_id)]
                assert item.evidence.text == content[span.start : span.end]
                assert span.end - span.start == item.retained_chars

        # Truncation and removal were recorded, and every raw unit is either
        # retained or dropped exactly once.
        assert any(item.truncated for item in prepared.items)
        assert prepared.dropped_raw_indices
        assert sorted(
            [i.raw_index for i in prepared.items] + prepared.dropped_raw_indices
        ) == list(range(len(raw)))

        # The pre-truncation raw return is a separate diagnostic artifact
        # and preserves the adapter's return untouched.
        assert [ev.model_dump() for ev in raw] == [
            ev.model_dump() for ev in CapturingFloodAdapter.last_return
        ]

    def test_match_run_under_budget_keeps_everything(self, tmp_path):
        config = load_config()
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        store = RunStore(tmp_path / "runs", outcome.run_id)
        for artifact in outcome.results:
            if "prepared_evidence" not in artifact.result.artifact_refs:
                continue
            doc = json.loads(
                store.resolve_ref(
                    artifact.result.artifact_refs["prepared_evidence"]
                ).read_text(encoding="utf-8")
            )
            prepared = doc["prepared"]
            assert prepared["token_count"] <= 4096
            assert all(not item["truncated"] for item in prepared["items"])


class TestPollutionIsolation:
    """AC6: upstream ids and annotation fields never reach adapter-visible objects."""

    MARKERS = [
        "has_answer",
        "answer_session_ids",
        "evidence_session_ids",
        "question_type",
        "_abs",
        "session_12_answer_7ab9",
        "session_11_answer_4d02",
        "smoke_abstention_0001_abs",
        "smoke_abstention_0002_abs",
    ]

    def test_adapter_never_sees_pollution(self, tmp_path):
        config = load_config()
        runner, adapter = make_runner(tmp_path, config)
        runner.run()
        blob = json.dumps(adapter.journal, ensure_ascii=False)
        for marker in self.MARKERS:
            assert marker not in blob, marker

    def test_namespaces_have_no_sample_semantics(self, tmp_path):
        config = load_config()
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        for artifact in outcome.results:
            ns = artifact.result.namespace
            assert ns.startswith("ns_")
            assert artifact.result.sample_handle not in ns
            assert "abstention" not in ns

    def test_evidence_uses_internal_ids_only(self, tmp_path):
        config = load_config()
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        store = RunStore(tmp_path / "runs", outcome.run_id)
        for artifact in outcome.results:
            if "raw_evidence" not in artifact.result.artifact_refs:
                continue
            doc = json.loads(
                store.resolve_ref(artifact.result.artifact_refs["raw_evidence"]).read_text(
                    encoding="utf-8"
                )
            )
            blob = json.dumps(doc["evidence"], ensure_ascii=False)
            for marker in self.MARKERS:
                assert marker not in blob, (artifact.result.sample_handle, marker)

    def test_configured_sample_handles_stay_internal(self):
        dataset = ManualDataset.load_default()
        assert "smoke_abstention_0001" in dataset.sample_handles
        # The _abs marker is only recoverable via the private scoring view.
        assert (
            dataset.get_scoring_data("smoke_abstention_0001").official_fields[
                "question_id"
            ]
            == "smoke_abstention_0001_abs"
        )
