"""compare command: comparability, refusal semantics, intersection.

Issue #6 acceptance criteria covered here:

- smoke-offline: two deterministic runs of the same config produce
  comparable fingerprints and scores (same-condition, equal-budget,
  zero metric deltas, identical intersection results);
- registry version mismatch refuses automatic metric alignment;
  unregistered metrics stay diagnostics;
- both sides report full planned-set results, status counts and
  coverage; the common runnable intersection carries its ID list in
  the persisted compare artifact.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from eval.compare import (
    RunSide,
    classify,
    comparability,
    compare_runs,
    load_run_side,
)
from eval.config import load_config_dict
from eval.datasets.manual import ManualDataset
from eval.judges.fake import FakeJudge, FakeJudgeSpec
from eval.memories.fake import FakeMemoryAdapter, FakeMemorySpec
from eval.readers.fake import FakeReader, FakeReaderSpec
from eval.runner import OfflineRunner
from eval.runs import RunStore

ALL_EIGHT = [
    "smoke_single_session_user_0001",
    "smoke_single_session_assistant_0001",
    "smoke_single_session_preference_0001",
    "smoke_temporal_reasoning_0001",
    "smoke_knowledge_update_0001",
    "smoke_multi_session_0001",
    "smoke_abstention_0001",
    "smoke_abstention_0002",
]


def compare_config(**overrides):
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
        "name": "compare-test",
        "dataset_plan": "manual-fixtures@1",
        "sample_plan_id": "smoke-offline-8",
        "sample_ids": list(ALL_EIGHT),
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
        "evidence_token_budget": 4096,
    }
    data.update(overrides)
    return load_config_dict(data)


def run_offline(tmp_path: Path, config, run_id: str) -> Path:
    dataset = ManualDataset.load_default()
    runner = OfflineRunner(
        config=config,
        dataset=dataset,
        adapter=FakeMemoryAdapter(
            FakeMemorySpec.from_memory_plan(config.memory)
        ),
        reader=FakeReader(FakeReaderSpec.from_reader_plan(config.reader)),
        judge=FakeJudge(FakeJudgeSpec.from_judge_plan(config.judge)),
        store=RunStore(tmp_path / "runs", run_id),
        run_id=run_id,
    )
    outcome = runner.run()
    assert outcome.failed == 0
    assert outcome.invalid_input == 0
    return outcome.run_dir


@pytest.fixture(scope="module")
def run_pair(tmp_path_factory) -> tuple[Path, Path]:
    """Two deterministic runs of the SAME config (smoke-offline shape)."""
    base = tmp_path_factory.mktemp("same")
    config = compare_config()
    left = run_offline(base, config, "run-cmp-left")
    right = run_offline(base, config, "run-cmp-right")
    return left, right


@pytest.fixture(scope="module")
def run_other_memory(tmp_path_factory) -> Path:
    """A second implementation under test: async, mixed evidence kinds."""
    base = tmp_path_factory.mktemp("other")
    config = compare_config(
        name="compare-test-impl-b",
        memory={
            "name": "fake-memory",
            "baseline_kind": "adapter",
            "capabilities": [
                "extractive_evidence",
                "generated_evidence",
                "state_inspection",
                "async_mutation",
                "idempotent_mutation",
                "operation_status",
            ],
            "config": {
                "mutation_mode": "async",
                "evidence_kinds": ["extractive", "generated"],
                "idempotent": True,
                "state_inspection": True,
                "async_lag": 1,
            },
        },
    )
    return run_offline(base, config, "run-cmp-implb")


class TestSameConditionComparison:
    def test_two_deterministic_runs_are_same_condition(
        self, run_pair, tmp_path
    ):
        left, right = run_pair
        payload = compare_runs(left, right)
        comp = payload["comparability"]
        assert comp["same_condition"] is True
        assert comp["differences"] == []
        assert comp["metrics_auto_aligned"] is True
        assert payload["same_config_fingerprint"] is True
        assert payload["comparison_kind"] == "equal_budget"
        assert "指纹相等" in payload["fingerprint_note"]

    def test_two_deterministic_runs_have_comparable_scores(
        self, run_pair
    ):
        left, right = run_pair
        payload = compare_runs(left, right)
        aligned = payload["metrics"]["aligned"]
        assert aligned
        # Every deterministic metric aligns exactly; only wall-clock
        # latency aggregates may drift between two runs of one config.
        latency_ids = {"ingest_latency_ms", "retrieve_latency_ms_p50", "retrieve_latency_ms_p95"}
        deterministic = [row for row in aligned if row["metric_id"] not in latency_ids]
        assert len(deterministic) == len(aligned) - 3
        # status and value agree exactly on both sides (N/A on both
        # sides included; a one-sided N/A would fail the value check)
        for row in deterministic:
            assert row["left"]["status"] == row["right"]["status"]
            assert row["left"]["value"] == row["right"]["value"]
        assert {row["metric_id"] for row in aligned} >= {
            "planned_question_score",
            "scored_accuracy",
            "verifiable_session_recall_macro",
            "recall_at_1",
        }
        left_score = payload["left_summary"]["full_planned_set"][
            "planned_question_score"
        ]
        right_score = payload["right_summary"]["full_planned_set"][
            "planned_question_score"
        ]
        assert left_score == right_score == 0.25

    def test_full_planned_set_status_counts_and_coverage(self, run_pair):
        left, _ = run_pair
        payload = compare_runs(left, left)
        summary = payload["left_summary"]
        assert summary["statuses"] == {
            "planned": 8,
            "scored": 8,
            "failed": 0,
            "context_exceeded": 0,
            "invalid_input": 0,
            "pending": 0,
        }
        full = summary["full_planned_set"]
        assert full["scoring_coverage"] == 1.0
        assert full["runnable_coverage"] == 1.0
        assert full["scored_accuracy"] == 0.25

    def test_intersection_ids_and_scores(self, run_pair, tmp_path):
        from eval.compare import save_comparison

        left, right = run_pair
        payload = compare_runs(left, right)
        inter = payload["intersection"]
        assert inter["available"] is True
        assert inter["scope"] == "同条件交集"
        assert inter["runnable_intersection_ids"] == sorted(ALL_EIGHT)
        assert inter["size"] == 8
        assert inter["left"]["planned_question_score"] == pytest.approx(0.25)
        assert inter["right"]["planned_question_score"] == pytest.approx(0.25)
        refs = save_comparison(payload, tmp_path / "out")
        saved = json.loads(Path(refs["compare_json"]).read_text(encoding="utf-8"))
        assert (
            saved["intersection"]["runnable_intersection_ids"] == sorted(ALL_EIGHT)
        )
        # run directories stay immutable: nothing written into them
        assert not (left / "compare.json").exists()

    def test_cross_implementation_memory_is_allowed_factor(
        self, run_pair, run_other_memory
    ):
        left, _ = run_pair
        payload = compare_runs(left, run_other_memory)
        comp = payload["comparability"]
        # Fingerprints differ (they embed the memory plan) but the runs
        # ARE comparable: every key field except memory matches.
        assert payload["same_config_fingerprint"] is False
        assert comp["same_condition"] is True
        assert comp["differences"] == []
        assert payload["comparison_kind"] == "equal_budget"
        factor = comp["experimental_factor"]
        assert factor["field"] == "/memory"
        assert factor["differences"]
        assert all(d["field"].startswith("/memory") for d in factor["differences"])
        # scores are still aligned and the intersection is defined
        assert payload["metrics"]["aligned"]
        assert payload["intersection"]["runnable_intersection_ids"]


class TestRefusals:
    def _side(self, run_dir: Path) -> RunSide:
        return load_run_side(run_dir)

    def test_registry_version_mismatch_refuses_alignment(self, run_pair, tmp_path):
        left, right = run_pair
        tampered = tmp_path / "runs" / "run-cmp-registry"
        shutil.copytree(right, tampered)
        doc = json.loads((tampered / "config.json").read_text(encoding="utf-8"))
        doc["metrics_registry_version"] = "2"
        (tampered / "config.json").write_text(
            json.dumps(doc, ensure_ascii=False), encoding="utf-8"
        )
        payload = compare_runs(left, tampered)
        comp = payload["comparability"]
        assert comp["same_condition"] is False
        fields = [d["field"] for d in comp["differences"]]
        assert "/metrics_registry_version" in fields
        assert comp["metrics_auto_aligned"] is False
        assert "注册表版本不一致" in comp["metrics_alignment_refused_reason"]
        assert payload["metrics"]["aligned"] == []
        # both sides' own full-P numbers are still reported
        assert payload["left_summary"]["available"] is True

    def test_registry_content_drift_refuses_alignment(self, run_pair, tmp_path):
        """A run produced under a DIFFERENT registry content cannot have
        its stored fingerprint reproduced now: drift is detectable and
        alignment is refused (content version is hash-bound, issue #1)."""
        left, right = run_pair
        tampered = tmp_path / "runs" / "run-cmp-drift"
        shutil.copytree(right, tampered)
        doc = json.loads((tampered / "config.json").read_text(encoding="utf-8"))
        doc["config_fingerprint"] = "f" * 64
        (tampered / "config.json").write_text(
            json.dumps(doc, ensure_ascii=False), encoding="utf-8"
        )
        payload = compare_runs(left, tampered)
        comp = payload["comparability"]
        assert comp["registry_fingerprint_reproducible"]["right"] is False
        assert comp["metrics_auto_aligned"] is False
        assert "无法在当前注册表内容下复现" in comp["metrics_alignment_refused_reason"]

    def test_unregistered_metric_is_diagnostic_only(self, run_pair, tmp_path):
        left, right = run_pair
        tampered = tmp_path / "runs" / "run-cmp-unregistered"
        shutil.copytree(right, tampered)
        report_path = tampered / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["metrics"].append(
            {
                "metric_id": "custom_vibes_score",
                "status": "computed",
                "value": 0.42,
                "denominator": "made up",
                "reason": None,
            }
        )
        report_path.write_text(
            json.dumps(report, ensure_ascii=False), encoding="utf-8"
        )
        payload = compare_runs(left, tampered)
        aligned_ids = {row["metric_id"] for row in payload["metrics"]["aligned"]}
        assert "custom_vibes_score" not in aligned_ids
        diag = [
            d for d in payload["metrics"]["diagnostics"]
            if d["kind"] == "unregistered"
        ]
        assert [d["metric_id"] for d in diag] == ["custom_vibes_score"]
        assert "未登记" in diag[0]["note"]

    @pytest.mark.parametrize(
        "field_path,override",
        [
            ("/reader/model", {"reader": {
                "model": "other-reader", "model_family": "family-r",
                "base_url": "offline://fake", "temperature": 0.0,
                "max_output_tokens": 1024, "tokenizer_id": "test:char-v1",
                "counting_mode": "test"}}),
            ("/reader/tokenizer_id", {"reader": {
                "model": "fake-reader", "model_family": "family-r",
                "base_url": "offline://fake", "temperature": 0.0,
                "max_output_tokens": 1024, "tokenizer_id": "test:char-v2",
                "counting_mode": "test"}}),
            ("/judge/protocol_id", {"judge": {
                "model": "fake-judge", "model_family": "family-j",
                "base_url": "offline://fake", "temperature": 0.0,
                "protocol_id": "other-protocol@9", "protocol_source_commit": "0" * 40}}),
            ("/evidence_token_budget", {"evidence_token_budget": 2048}),
            ("/run_params/max_retries", {"run_params": {"max_retries": 1}}),
        ],
    )
    def test_key_field_differences_refuse_same_condition_label(
        self, tmp_path, field_path, override
    ):
        config = compare_config(**override)
        left = run_offline(tmp_path, compare_config(), "run-kf-left")
        right = run_offline(tmp_path, config, "run-kf-right")
        payload = compare_runs(left, right)
        comp = payload["comparability"]
        assert comp["same_condition"] is False
        fields = [d["field"] for d in comp["differences"]]
        assert field_path in fields
        assert payload["comparison_kind"] == "not_same_condition"

    def test_sample_plan_difference_lists_only_sides(self, tmp_path):
        subset = list(ALL_EIGHT[:5])
        left = run_offline(tmp_path, compare_config(), "run-plan-left")
        right = run_offline(
            tmp_path, compare_config(sample_ids=subset), "run-plan-right"
        )
        payload = compare_runs(left, right)
        comp = payload["comparability"]
        assert comp["same_condition"] is False
        sample_diff = next(
            d for d in comp["differences"] if d["field"] == "/sample_ids"
        )
        assert sample_diff["right"]["only_on_this_side"] == []
        assert sample_diff["left"]["only_on_this_side"] == sorted(ALL_EIGHT[5:])
        inter = payload["intersection"]
        assert inter["runnable_intersection_ids"] == sorted(subset)
        assert inter["scope"].startswith("诊断性交集")


class TestIntersectionSemantics:
    def test_context_excluded_and_failed_kept(self, run_pair, tmp_path):
        """Precheck capability decides the intersection; runtime failure
        stays inside and contributes zero (never excluded by scoring)."""
        left, right = run_pair
        tampered = tmp_path / "runs" / "run-cmp-statuses"
        shutil.copytree(right, tampered)
        path = tampered / "samples.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        by_handle = {r["result"]["sample_handle"]: r for r in rows}
        ctx = by_handle["smoke_temporal_reasoning_0001"]["result"]
        ctx["qa_status"] = "context_exceeded"
        ctx["correct"] = None
        ctx["attribution"] = None
        failed = by_handle["smoke_multi_session_0001"]["result"]
        failed["qa_status"] = "failed"
        failed["correct"] = None
        failed["attribution"] = None
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        payload = compare_runs(left, tampered)
        inter = payload["intersection"]
        ids = inter["runnable_intersection_ids"]
        assert "smoke_temporal_reasoning_0001" not in ids  # precheck excludes
        assert "smoke_multi_session_0001" in ids  # failure stays inside
        assert inter["right"]["not_runnable_on_this_side"] == [
            "smoke_temporal_reasoning_0001"
        ]
        # the failed sample contributes zero but is NOT dropped; the
        # tampered side keeps its two originally-correct answers, so the
        # intersection score is 2/7 (failed counts zero, not excluded)
        assert inter["size"] == 7
        assert inter["right"]["correct"] == 2
        assert inter["right"]["planned_question_score"] == pytest.approx(2 / 7)

    def test_qa_vs_operations_compares_but_refuses_alignment(
        self, run_pair, tmp_path
    ):
        from eval.config import load_config_toml
        from eval.operations import OperationsRunner

        left, _ = run_pair
        config = load_config_toml(
            "eval/configs/examples/offline_fake_ops.toml"
        )
        store = RunStore(tmp_path / "runs", "run-cmp-ops")
        runner = OperationsRunner(
            config=config,
            adapter=FakeMemoryAdapter(
                FakeMemorySpec.from_memory_plan(config.memory)
            ),
            store=store,
            run_id="run-cmp-ops",
        )
        assert runner.run().failed == 0
        payload = compare_runs(left, store.dir)
        comp = payload["comparability"]
        assert comp["same_condition"] is False
        assert "/suite" in [d["field"] for d in comp["differences"]]
        assert comp["metrics_auto_aligned"] is False
        assert "suite" in comp["metrics_alignment_refused_reason"]
        assert payload["intersection"]["available"] is False
        # each side still reports its own shape
        assert payload["left_summary"]["suite"] == "qa"
        assert payload["right_summary"]["suite"] == "operations"
        assert payload["right_summary"]["statuses"]["passed"] == 5


class TestClassificationUnits:
    """Pure classification of equal-budget vs information controls."""

    def _side(self, baseline_kind: str) -> RunSide:
        side = RunSide(
            run_dir=Path("/tmp/x"),
            manifest={"run_id": "r", "suite": "qa", "sample_ids": []},
            config_doc={
                "metrics_registry_version": "1",
                "config": {
                    "suite": "qa",
                    "memory": {"baseline_kind": baseline_kind},
                    "sample_ids": [],
                },
            },
            report=None,
        )
        return side

    def _comp(self, same: bool = True) -> dict:
        return {
            "same_condition": same,
            "metrics_auto_aligned": True,
            "metrics_alignment_refused_reason": None,
        }

    def test_full_history_is_information_control(self):
        left, right = self._side("adapter"), self._side("full_history")
        kind = classify(left, right, self._comp())
        assert kind["comparison_kind"] == "full_history_control"
        assert "信息条件不同" in kind["note"]
        assert "等预算" not in kind["note"].replace("不标为等预算", "")

    def test_none_baseline_is_no_memory_control(self):
        left, right = self._side("none"), self._side("adapter")
        kind = classify(left, right, self._comp())
        assert kind["comparison_kind"] == "no_memory_control"

    def test_budget_bound_pair_is_equal_budget(self):
        left, right = self._side("adapter"), self._side("bm25")
        assert classify(left, right, self._comp())["comparison_kind"] == "equal_budget"

    def test_key_differences_override_kind(self):
        left, right = self._side("adapter"), self._side("adapter")
        kind = classify(left, right, self._comp(same=False))
        assert kind["comparison_kind"] == "not_same_condition"


class TestCompareCli:
    def test_cli_compare_prints_payload_and_saves_artifact(
        self, run_pair, tmp_path, capsys
    ):
        from eval.cli import main

        left, right = run_pair
        out = tmp_path / "cmp-out"
        code = main(
            ["compare", str(left), str(right), "--out", str(out)]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["command"] == "compare"
        assert payload["comparability"]["same_condition"] is True
        refs = payload["artifacts"]
        saved = json.loads(Path(refs["compare_json"]).read_text(encoding="utf-8"))
        assert saved["intersection"]["runnable_intersection_ids"] == sorted(ALL_EIGHT)
        markdown = Path(refs["compare_markdown"]).read_text(encoding="utf-8")
        assert "# Memory Eval 运行比较" in markdown
        assert "可比性检查" in markdown
        assert "共同可运行交集" in markdown
        assert "指标对齐" in markdown

    def test_cli_compare_defaults_out_to_left_parent(self, run_pair, capsys):
        from eval.cli import main

        left, right = run_pair
        code = main(["compare", str(left), str(right)])
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert Path(payload["artifacts"]["compare_json"]).parent == left.parent

    def test_cli_compare_rejects_missing_run_dir(self, tmp_path, capsys):
        from eval.cli import main

        code = main(["compare", str(tmp_path / "nope"), str(tmp_path / "nada")])
        assert code == 2
        assert "run manifest missing" in capsys.readouterr().err
