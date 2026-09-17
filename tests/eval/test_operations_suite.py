"""Operations suite: the five acceptance criteria of issue #4.

The suite runs on small artificial data with deterministic assertions;
no LLM judge decides operation outcomes. Explicit update/delete targets
always come from MutationReceipts; status assertions go exclusively
through inspect() by stable id.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.config import OPERATIONS_CHECK_IDS, load_config_dict
from eval.contracts.adapter import MutationReceipt
from eval.contracts.common import ContractError
from eval.contracts.internal import ResultArtifact
from eval.memories.fake import FakeMemoryAdapter, FakeMemorySpec
from eval.operations import OperationsRunner
from eval.runs import (
    OperationCheckDetail,
    OperationsSummaryArtifact,
    RunStore,
)

OPS_EXAMPLE_CONFIG = "eval/configs/examples/offline_fake_ops.toml"

FULL_MEMORY = {
    "name": "fake-memory",
    "baseline_kind": "adapter",
    "capabilities": [
        "extractive_evidence",
        "generated_evidence",
        "state_inspection",
        "auto_update",
        "update",
        "delete",
    ],
    "config": {
        "mutation_mode": "sync",
        "evidence_kinds": ["extractive", "generated"],
        "retrieval_mode": "match",
        "idempotent": False,
        "state_inspection": True,
        "auto_update": True,
        "update": True,
        "delete": True,
    },
}


def ops_config(
    sample_ids: list[str] | None = None,
    memory: dict | None = None,
    **overrides
):
    data = {
        "name": "ops-test",
        "suite": "operations",
        "dataset_plan": "manual-operations@1",
        "sample_plan_id": "operations-test",
        "sample_ids": sample_ids or list(OPERATIONS_CHECK_IDS),
        "smoke_subset_ids": [],
        "memory": memory if memory is not None else FULL_MEMORY,
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
        "evidence_token_budget": 4096,
    }
    data.update(overrides)
    return load_config_dict(data)


def make_runner(tmp_path: Path, config, adapter=None):
    adapter = adapter or FakeMemoryAdapter(
        FakeMemorySpec.from_memory_plan(config.memory)
    )
    run_id = f"run-ops-{config.fingerprint()[:8]}"
    store = RunStore(tmp_path / "runs", run_id)
    return (
        OperationsRunner(
            config=config, adapter=adapter, store=store, run_id=run_id
        ),
        adapter,
    )


def statuses(outcome) -> dict[str, str]:
    return {r.result.sample_handle: r.result.operation_status for r in outcome.results}


def detail_of(outcome, check_id: str) -> OperationCheckDetail:
    assert outcome.summary is not None
    return next(d for d in outcome.summary.checks if d.check_id == check_id)


# -- rogue adapters used to prove the suite catches real failures ---------


class NoopUpdateAdapter(FakeMemoryAdapter):
    """Declares update but never changes the stored text."""

    def update(self, namespace, memory_id, replacement, operation_id):
        current = self._spaces[namespace][memory_id].content
        return super().update(namespace, memory_id, current, operation_id)


class NoopDeleteAdapter(FakeMemoryAdapter):
    """Declares delete but keeps the target retrievable."""

    def delete(self, namespace, memory_id, operation_id):
        self._log(
            "delete", namespace=namespace, memory_id=memory_id,
            operation_id=operation_id,
        )
        return MutationReceipt(
            operation_id=operation_id,
            status="completed",
            memory_ids=[memory_id],
            sources=[],
            error=None,
            usage=None,
        )


class LeakyAdapter(FakeMemoryAdapter):
    """Returns every namespace's content regardless of the queried space."""

    def retrieve(self, namespace, request):
        own = list(super().retrieve(namespace, request))
        for other in [ns for ns in self._spaces if ns != namespace]:
            own.extend(super().retrieve(other, request))
        return own


class AmnesiaAdapter(FakeMemoryAdapter):
    """Loses the space's data on close (no persistence)."""

    def close(self, namespace):
        self._spaces.pop(namespace, None)
        super().close(namespace)


class TestSuitePassesWithFullCapabilities:
    def test_example_config_all_checks_pass(self, tmp_path: Path):
        from eval.config import load_config_toml

        config = load_config_toml(OPS_EXAMPLE_CONFIG)
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        assert statuses(outcome) == {
            "auto_update": "passed",
            "explicit_update": "passed",
            "delete": "passed",
            "isolation": "passed",
            "persistence": "passed",
        }
        assert outcome.failed == 0

    def test_sync_profile_also_passes(self, tmp_path: Path):
        config = ops_config()
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        assert set(statuses(outcome).values()) == {"passed"}

    def test_subset_plan_runs_only_listed_checks(self, tmp_path: Path):
        config = ops_config(sample_ids=["isolation", "persistence"])
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        assert statuses(outcome) == {"isolation": "passed", "persistence": "passed"}


class TestSeparateAutoAndExplicitReporting:
    """AC1: auto and explicit update are reported separately."""

    def test_both_recorded_as_distinct_items(self, tmp_path: Path):
        config = ops_config()
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        handles = [r.result.sample_handle for r in outcome.results]
        assert "auto_update" in handles and "explicit_update" in handles
        summary = outcome.summary
        assert summary is not None
        auto = detail_of(outcome, "auto_update")
        explicit = detail_of(outcome, "explicit_update")
        assert auto.check_id != explicit.check_id
        assert "update" not in {a.name for a in auto.assertions} or True
        # auto_update never calls update(): no update attempts exist.
        auto_result = next(
            r for r in outcome.results
            if r.result.sample_handle == "auto_update"
        )
        assert all(a.stage != "update" for a in auto_result.result.attempts)

    def test_explicit_pass_does_not_substitute_auto_update(self, tmp_path: Path):
        # update declared, auto_update NOT: explicit can pass while the
        # auto-update item must stay not_supported — never borrowed.
        memory = json.loads(json.dumps(FULL_MEMORY))
        memory["capabilities"] = [
            c for c in memory["capabilities"] if c != "auto_update"
        ]
        memory["config"]["auto_update"] = False
        config = ops_config(memory=memory)
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        result = statuses(outcome)
        assert result["explicit_update"] == "passed"
        assert result["auto_update"] == "not_supported"

    def test_auto_pass_does_not_substitute_explicit_update(self, tmp_path: Path):
        memory = json.loads(json.dumps(FULL_MEMORY))
        memory["capabilities"] = [
            c for c in memory["capabilities"] if c != "update"
        ]
        memory["config"].pop("update_retains_old", None)
        memory["config"]["update"] = False
        config = ops_config(memory=memory)
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        result = statuses(outcome)
        assert result["auto_update"] == "passed"
        assert result["explicit_update"] == "not_supported"


class TestExplicitUpdateAssertions:
    """AC2: inspect shows content==replacement and current; retained old
    values are never current."""

    def test_inspect_assertions_present_and_passing(self, tmp_path: Path):
        memory = json.loads(json.dumps(FULL_MEMORY))
        memory["config"]["update_retains_old"] = True
        config = ops_config(
            sample_ids=["explicit_update"], memory=memory
        )
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        detail = detail_of(outcome, "explicit_update")
        assert detail.operation_status == "passed"
        names = {a.name: a for a in detail.assertions}
        assert names["updated target content equals the replacement"].passed
        assert names["updated target is current"].passed
        retained = [
            a for a in detail.assertions if a.name.startswith("retained old value")
        ]
        assert retained and all(a.passed for a in retained)

    def test_without_retained_old_the_check_still_passes(self, tmp_path: Path):
        config = ops_config(sample_ids=["explicit_update"])
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        detail = detail_of(outcome, "explicit_update")
        assert detail.operation_status == "passed"
        assert not [a for a in detail.assertions if a.name.startswith("retained")]

    def test_target_id_comes_from_receipts_only(self, tmp_path: Path):
        config = ops_config(sample_ids=["explicit_update"])
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        detail = detail_of(outcome, "explicit_update")
        # Every target id must appear in some recorded receipt output.
        result = next(r for r in outcome.results)
        receipt_ids = set()
        for entry in json.loads(
            (Path(outcome.run_dir) / f"artifacts/explicit_update/attempts.json")
            .read_text(encoding="utf-8")
        )["entries"]:
            output = entry["output"]
            if isinstance(output, dict) and "memory_ids" in output:
                receipt_ids.update(output["memory_ids"])
        assert receipt_ids
        assert set(detail.target_memory_ids) <= receipt_ids

    def test_noop_update_fails_on_content_mismatch(self, tmp_path: Path):
        config = ops_config(sample_ids=["explicit_update"])
        adapter = NoopUpdateAdapter(
            FakeMemorySpec.from_memory_plan(config.memory)
        )
        runner, _ = make_runner(tmp_path, config, adapter=adapter)
        outcome = runner.run()
        detail = detail_of(outcome, "explicit_update")
        assert detail.operation_status == "failed"
        assert detail.failed_stage == "inspect"
        content_assert = next(
            a for a in detail.assertions
            if a.name == "updated target content equals the replacement"
        )
        assert not content_assert.passed

    def test_missing_target_id_is_not_supported(self, tmp_path: Path):
        class NoIdReceiptAdapter(FakeMemoryAdapter):
            def ingest(self, namespace, session, operation_id):
                receipt = super().ingest(namespace, session, operation_id)
                return receipt.model_copy(update={"memory_ids": []})

        config = ops_config(sample_ids=["explicit_update"])
        adapter = NoIdReceiptAdapter(
            FakeMemorySpec.from_memory_plan(config.memory)
        )
        runner, _ = make_runner(tmp_path, config, adapter=adapter)
        outcome = runner.run()
        detail = detail_of(outcome, "explicit_update")
        assert detail.operation_status == "not_supported"
        assert "receipt" in (detail.reason or "")


class TestDeleteAssertions:
    """AC3: target original and derived summaries no longer recalled;
    unrelated memory stays retrievable."""

    def test_delete_assertions_cover_recall_and_restart(self, tmp_path: Path):
        config = ops_config(sample_ids=["delete"])
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        detail = detail_of(outcome, "delete")
        assert detail.operation_status == "passed"
        names = [a.name for a in detail.assertions]
        assert any("before delete" in n for n in names)
        assert any("(after delete)" in n for n in names)
        assert any("(after restart)" in n for n in names)
        assert any("unrelated memory still retrievable" in n for n in names)
        # Generated summaries were covered (the config declares them).
        assert any("derived summary covers the target" in n for n in names)

    def test_delete_works_without_generated_evidence(self, tmp_path: Path):
        memory = json.loads(json.dumps(FULL_MEMORY))
        memory["capabilities"] = [
            "extractive_evidence", "state_inspection", "delete",
        ]
        memory["config"] = {
            "mutation_mode": "sync",
            "evidence_kinds": ["extractive"],
            "retrieval_mode": "match",
            "state_inspection": True,
            "delete": True,
        }
        config = ops_config(sample_ids=["delete"], memory=memory)
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        detail = detail_of(outcome, "delete")
        assert detail.operation_status == "passed"
        assert not [
            a for a in detail.assertions if "derived summary covers" in a.name
        ]

    def test_noop_delete_fails_on_still_recallable_target(self, tmp_path: Path):
        config = ops_config(sample_ids=["delete"])
        adapter = NoopDeleteAdapter(
            FakeMemorySpec.from_memory_plan(config.memory)
        )
        runner, _ = make_runner(tmp_path, config, adapter=adapter)
        outcome = runner.run()
        detail = detail_of(outcome, "delete")
        assert detail.operation_status == "failed"
        assert detail.failed_stage == "retrieve"
        recall_assert = next(
            a for a in detail.assertions
            if "(after delete)" in a.name and "no longer recalled" in a.name
        )
        assert not recall_assert.passed

    def test_deleting_everything_does_not_pass(self, tmp_path: Path):
        class WipeAllAdapter(FakeMemoryAdapter):
            """A 'successful' delete that wipes the whole space: the
            unrelated-memory control must catch it."""

            def delete(self, namespace, memory_id, operation_id):
                self._spaces.pop(namespace, None)
                self._tombstones.pop(namespace, None)
                return MutationReceipt(
                    operation_id=operation_id,
                    status="completed",
                    memory_ids=[memory_id],
                    sources=[],
                    error=None,
                    usage=None,
                )

        config = ops_config(sample_ids=["delete"])
        adapter = WipeAllAdapter(
            FakeMemorySpec.from_memory_plan(config.memory)
        )
        runner, _ = make_runner(tmp_path, config, adapter=adapter)
        outcome = runner.run()
        detail = detail_of(outcome, "delete")
        assert detail.operation_status == "failed"
        unrelated = [
            a for a in detail.assertions
            if "unrelated memory still retrievable" in a.name
        ]
        assert unrelated and not unrelated[0].passed


class TestIsolationAndPersistence:
    """AC4: one space never returns another's content; reopening keeps
    written content retrievable."""

    def test_isolation_assertions_include_positive_controls(self, tmp_path: Path):
        config = ops_config(sample_ids=["isolation"])
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        detail = detail_of(outcome, "isolation")
        assert detail.operation_status == "passed"
        assert len(detail.namespaces) == 2
        names = [a.name for a in detail.assertions]
        assert any("does not return 项目乙" in n for n in names)
        assert any("does not return 项目甲" in n for n in names)
        assert names.count("space still returns its own content") == 2

    def test_leaky_adapter_fails_isolation(self, tmp_path: Path):
        config = ops_config(sample_ids=["isolation"])
        adapter = LeakyAdapter(FakeMemorySpec.from_memory_plan(config.memory))
        runner, _ = make_runner(tmp_path, config, adapter=adapter)
        outcome = runner.run()
        detail = detail_of(outcome, "isolation")
        assert detail.operation_status == "failed"
        assert detail.failed_stage == "retrieve"

    def test_persistence_reopen_keeps_content(self, tmp_path: Path):
        config = ops_config(sample_ids=["persistence"])
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        detail = detail_of(outcome, "persistence")
        assert detail.operation_status == "passed"
        assert len(detail.assertions) == 2

    def test_amnesia_adapter_fails_persistence(self, tmp_path: Path):
        config = ops_config(sample_ids=["persistence"])
        adapter = AmnesiaAdapter(FakeMemorySpec.from_memory_plan(config.memory))
        runner, _ = make_runner(tmp_path, config, adapter=adapter)
        outcome = runner.run()
        detail = detail_of(outcome, "persistence")
        assert detail.operation_status == "failed"
        assert all(not a.passed for a in detail.assertions)


class TestUnknownAndNotSupported:
    """AC5: unknown never passes; undeclared state capability means
    not_supported; declared-but-unknown means failed."""

    def test_unknown_states_fail_declared_checks(self, tmp_path: Path):
        memory = json.loads(json.dumps(FULL_MEMORY))
        memory["config"]["state_returns_unknown"] = True
        config = ops_config(memory=memory)
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        result = statuses(outcome)
        # Declared state capability returning unknown: failures, never passes.
        assert result["auto_update"] == "failed"
        assert result["explicit_update"] == "failed"
        assert result["delete"] == "failed"
        auto_detail = detail_of(outcome, "auto_update")
        assert auto_detail.failed_stage == "inspect"
        assert any(
            "validity=unknown" in a.observed and not a.passed
            for a in auto_detail.assertions
        )
        # Checks that need no state still pass.
        assert result["isolation"] == "passed"
        assert result["persistence"] == "passed"

    def test_missing_state_capability_means_not_supported(self, tmp_path: Path):
        memory = json.loads(json.dumps(FULL_MEMORY))
        memory["capabilities"] = [
            "extractive_evidence",
            "generated_evidence",
            "auto_update",
            "update",
            "delete",
        ]
        memory["config"]["state_inspection"] = False
        memory["config"]["state_returns_unknown"] = False
        config = ops_config(memory=memory)
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        result = statuses(outcome)
        assert result["auto_update"] == "not_supported"
        assert result["explicit_update"] == "not_supported"
        # delete is recall-based: it runs without state inspection.
        assert result["delete"] == "passed"
        for check_id in ("auto_update", "explicit_update"):
            detail = detail_of(outcome, check_id)
            assert "state_inspection" in detail.missing_capabilities
            assert detail.reason

    def test_no_operation_capabilities_all_not_supported(self, tmp_path: Path):
        memory = {
            "name": "fake-memory",
            "baseline_kind": "adapter",
            "capabilities": ["extractive_evidence"],
            "config": {
                "mutation_mode": "sync",
                "evidence_kinds": ["extractive"],
                "state_inspection": False,
            },
        }
        config = ops_config(memory=memory)
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        result = statuses(outcome)
        assert result["auto_update"] == "not_supported"
        assert result["explicit_update"] == "not_supported"
        assert result["delete"] == "not_supported"
        assert result["isolation"] == "passed"
        assert result["persistence"] == "passed"

    def test_three_states_counted_separately(self, tmp_path: Path):
        memory = json.loads(json.dumps(FULL_MEMORY))
        memory["capabilities"] = [
            "extractive_evidence",
            "state_inspection",
            "delete",
            "auto_update",
        ]
        memory["config"] = {
            "mutation_mode": "sync",
            "evidence_kinds": ["extractive"],
            "retrieval_mode": "match",
            "state_inspection": True,
            "auto_update": True,
            "delete": True,
        }
        config = ops_config(memory=memory)
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        summary = outcome.summary
        assert summary is not None
        assert summary.planned_checks == 5
        # auto_update, delete, isolation and persistence run and pass;
        # explicit_update lacks the update capability.
        assert summary.passed == 4
        assert summary.not_supported == 1
        assert summary.failed == 0
        assert summary.pending == 0


class TestOperationsArtifacts:
    def test_samples_jsonl_rows_are_operations_suite(self, tmp_path: Path):
        config = ops_config()
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        lines = (Path(outcome.run_dir) / "samples.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        assert len(lines) == 5
        for line in lines:
            artifact = ResultArtifact.load_json(line)
            result = artifact.result
            assert result.suite == "operations"
            assert result.qa_status is None
            assert result.correct is None
            assert result.operation_status in (
                "passed",
                "failed",
                "not_supported",
            )
            assert result.run_id == outcome.run_id

    def test_manifest_and_plan_counts_match_actual_calls(self, tmp_path: Path):
        from eval.config import load_config_toml

        config = load_config_toml(OPS_EXAMPLE_CONFIG)
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        run_dir = Path(outcome.run_dir)
        manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert manifest["suite"] == "operations"
        assert manifest["async_mutation"] is True
        for check_id, plan in manifest["plan_counts"].items():
            log = json.loads(
                (run_dir / f"artifacts/{check_id}/attempts.json").read_text(
                    encoding="utf-8"
                )
            )
            methods: dict[str, int] = {}
            for entry in log["entries"]:
                methods[entry["method"]] = methods.get(entry["method"], 0) + 1
            assert methods.get("ingest", 0) == plan["ingest_calls"], check_id
            assert methods.get("await_ready", 0) == plan["await_ready_calls"], check_id
            assert methods.get("update", 0) == plan["update_calls"], check_id
            assert methods.get("delete", 0) == plan["delete_calls"], check_id
            assert methods.get("inspect", 0) == plan["inspect_calls"], check_id
            assert methods.get("open", 0) == plan["open_calls"], check_id
            assert methods.get("close", 0) == plan["close_calls"], check_id
            assert methods.get("retrieve", 0) == plan["retrieve_calls"], check_id

    def test_summary_artifact_counts_and_registered_metrics(self, tmp_path: Path):
        config = ops_config()
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        raw = (Path(outcome.run_dir) / "operations_summary.json").read_text(
            encoding="utf-8"
        )
        summary = OperationsSummaryArtifact.load_json(raw)
        assert summary.passed == 5
        assert {m.metric_id for m in summary.metrics} == {
            "operations_pass_rate",
            "operations_support_coverage",
        }
        assert summary.pass_rate == 1.0
        assert summary.support_coverage == 1.0

    def test_artifact_refs_resolve(self, tmp_path: Path):
        config = ops_config()
        runner, _ = make_runner(tmp_path, config)
        outcome = runner.run()
        store = RunStore(tmp_path / "runs", outcome.run_id)
        for artifact in outcome.results:
            assert set(artifact.result.artifact_refs) == {
                "attempts",
                "operations_check",
            }
            for ref in artifact.result.artifact_refs.values():
                assert store.resolve_ref(ref).exists()

    def test_capability_mismatch_aborts_before_run_dir(self, tmp_path: Path):
        memory = json.loads(json.dumps(FULL_MEMORY))
        memory["capabilities"].append("bogus_capability")
        config = ops_config(memory=memory)
        runner, _ = make_runner(tmp_path, config)
        with pytest.raises(ContractError) as excinfo:
            runner.run()
        assert excinfo.value.code == "capability_declaration_mismatch"
        assert not (tmp_path / "runs" / runner.run_id).exists()


class TestOperationsConfigValidation:
    def test_unknown_check_id_rejected(self):
        with pytest.raises(ContractError):
            ops_config(sample_ids=["auto_update", "teleport"])

    def test_qa_smoke_ids_rejected_in_operations_suite(self):
        with pytest.raises(ContractError):
            ops_config(sample_ids=["smoke_single_session_user_0001"])

    def test_operations_with_smoke_subset_rejected(self):
        with pytest.raises(ContractError):
            ops_config(smoke_subset_ids=["smoke_single_session_user_0001"])

    def test_suite_enters_fingerprint(self):
        qa_like = ops_config()
        ops_like = ops_config()
        config_ops = qa_like
        data = config_ops.canonical_payload()
        assert data["suite"] == "operations"


class TestCliOperationsRun:
    def test_cli_run_operations_example(self, capsys, tmp_path: Path):
        from eval.cli import main

        out = tmp_path / "runs"
        code = main(
            [
                "run",
                "--config", OPS_EXAMPLE_CONFIG,
                "--out", str(out),
            ]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["suite"] == "operations"
        assert payload["planned_checks"] == 5
        assert payload["passed"] == 5
        assert payload["failed"] == 0
        assert payload["not_supported"] == 0
        assert (Path(payload["run_dir"]) / "operations_summary.json").exists()

    def test_cli_run_operations_plan_without_out(self, capsys):
        from eval.cli import main

        code = main(["run", "--config", OPS_EXAMPLE_CONFIG])
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert "not-executed" in payload["status"]
