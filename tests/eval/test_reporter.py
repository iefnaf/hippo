"""Reporter: JSON + Markdown aggregation over a completed run directory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.config import load_config_dict, load_config_toml
from eval.contracts.common import ContractError
from eval.datasets.manual import ManualDataset
from eval.judges.fake import FakeJudge, FakeJudgeSpec
from eval.memories.fake import FakeMemoryAdapter, FakeMemorySpec
from eval.readers.fake import FakeReader, FakeReaderSpec
from eval.report import Reporter, render_markdown
from eval.runner import OfflineRunner
from eval.runs import RunStore

EXAMPLE_CONFIG = "eval/configs/examples/offline_fake.toml"


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
    reader_extra = overrides.pop("reader_extra", {})
    judge_extra = overrides.pop("judge_extra", {})
    data = {
        "name": "reporter-test",
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
            "extra": reader_extra,
        },
        "judge": {
            "model": "fake-judge",
            "model_family": "family-j",
            "base_url": "offline://fake",
            "temperature": 0.0,
            "protocol_id": "longmemeval-yes-no@1",
            "protocol_source_commit": "0" * 40,
            "extra": judge_extra,
        },
        "evidence_token_budget": overrides.pop("evidence_token_budget", 4096),
    }
    data.update(overrides)
    return load_config_dict(data)


def run_one(tmp_path: Path, config, dataset=None, tag="r1"):
    dataset = dataset or ManualDataset.load_default()
    adapter = FakeMemoryAdapter(FakeMemorySpec.from_memory_plan(config.memory))
    run_id = f"run-{tag}-{config.fingerprint()[:8]}"
    runner = OfflineRunner(
        config=config,
        dataset=dataset,
        adapter=adapter,
        reader=FakeReader(FakeReaderSpec.from_reader_plan(config.reader)),
        judge=FakeJudge(FakeJudgeSpec.from_judge_plan(config.judge)),
        store=RunStore(tmp_path / "runs", run_id),
        run_id=run_id,
    )
    return runner.run()


def metric(report, metric_id):
    return next(m for m in report["metrics"] if m["metric_id"] == metric_id)


class TestHappyPathReport:
    """The deterministic example run has fixed, checkable aggregates."""

    @pytest.fixture()
    def report(self, tmp_path: Path):
        config = load_config_toml(EXAMPLE_CONFIG)
        outcome = run_one(tmp_path, config, tag="happy")
        assert outcome.failed == 0 and outcome.invalid_input == 0
        return Reporter(outcome.run_dir).build()

    def test_report_files_written_and_checksummed(self, tmp_path: Path):
        config = load_config_toml(EXAMPLE_CONFIG)
        outcome = run_one(tmp_path, config, tag="files")
        run_dir = outcome.run_dir
        assert (run_dir / "report.json").exists()
        assert (run_dir / "report.md").exists()
        store = RunStore(tmp_path / "runs", outcome.run_id)
        for ref in outcome.report_refs.values():
            assert store.resolve_ref(ref).exists()
        doc = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        assert doc["header"]["run_id"] == outcome.run_id

    def test_status_counts_and_denominators(self, report):
        statuses = report["statuses"]
        assert statuses["planned"] == 8
        assert statuses["scored"] == 8
        assert statuses["failed"] == 0
        assert statuses["context_exceeded"] == 0
        assert statuses["invalid_input"] == 0
        assert statuses["pending"] == 0
        assert statuses["abstention_planned"] == 2
        assert statuses["abstention_scored"] == 2

    def test_overall_qa_metrics(self, report):
        # Deterministic fake outcomes: assistant + preference correct.
        assert metric(report, "planned_question_score")["value"] == pytest.approx(0.25)
        assert metric(report, "scored_accuracy")["value"] == pytest.approx(0.25)
        assert metric(report, "scoring_coverage")["value"] == pytest.approx(1.0)
        assert metric(report, "runnable_coverage")["value"] == pytest.approx(1.0)
        assert metric(report, "abstention_accuracy")["value"] == pytest.approx(0.0)

    def test_recall_aggregates_use_applicable_set(self, report):
        # E = 6 non-abstention questions; both abstention samples excluded.
        macro = metric(report, "verifiable_session_recall_macro")
        assert macro["value"] == pytest.approx(1.0)
        assert "|E|=6" in macro["denominator"]
        micro = metric(report, "verifiable_session_recall_micro")
        assert micro["value"] == pytest.approx(1.0)
        assert "7" in micro["denominator"]  # gold sessions across E
        assert metric(report, "recall_at_1")["value"] == pytest.approx(3.5 / 6)
        assert metric(report, "recall_at_3")["value"] == pytest.approx(1.0)
        assert metric(report, "recall_at_5")["value"] == pytest.approx(1.0)
        assert metric(report, "budgeted_session_recall")["value"] == pytest.approx(1.0)
        assert metric(report, "derivation_source_coverage")["value"] == pytest.approx(1.0)

    def test_attribution_2x2(self, report):
        attribution = report["attribution"]
        assert attribution["scored_denominator"] == 8
        assert attribution["overall"] == {
            "hit_correct": 2,
            "hit_wrong": 4,
            "miss_correct": 0,
            "miss_wrong": 2,
        }
        assert attribution["shares"]["hit_correct"] == pytest.approx(0.25)
        assert attribution["shares"]["hit_wrong"] == pytest.approx(0.5)
        assert attribution["shares"]["miss_wrong"] == pytest.approx(0.25)
        # Abstention questions never hit gold: both land in miss_wrong.
        assert attribution["by_abstention"]["true"] == {
            "hit_correct": 0,
            "hit_wrong": 0,
            "miss_correct": 0,
            "miss_wrong": 2,
        }
        # Per-question-type breakdown exists for all six categories.
        types = set(attribution["by_question_type"])
        assert types == {
            "single-session-user",
            "single-session-assistant",
            "single-session-preference",
            "temporal-reasoning",
            "knowledge-update",
            "multi-session",
        }
        # hit_wrong quantifies "found evidence but answered wrong".
        assert attribution["by_question_type"]["knowledge-update"]["hit_wrong"] == 1

    def test_only_registered_metrics_in_formal_table(self, report):
        from eval.metrics import REGISTRY

        ids = [m["metric_id"] for m in report["metrics"]]
        assert ids, "formal table must not be empty"
        assert all(m in REGISTRY for m in ids)
        assert "operations_pass_rate" not in ids  # qa-suite run

    def test_budget_composition_matches_artifacts(self, report):
        budget = report["budget"]
        assert budget["counting_mode"] == "test"
        assert budget["budget"] == 4096
        totals = budget["totals"]
        assert totals["returned_units"] == totals["retained_units"] == 22
        assert totals["dropped_units"] == 0
        assert totals["token_count"] == sum(
            row["token_count"] for row in budget["rows"]
        )
        # Format overhead (metadata/separators) is visible separately.
        assert totals["format_tokens"] > 0
        user_row = next(
            r for r in budget["rows"] if r["sample_handle"] == "smoke_single_session_user_0001"
        )
        assert user_row["returned_units"] == 3  # 2 extractive + 1 generated

    def test_cost_model_planned_and_unit_costs(self, report):
        cost = report["cost_model"]
        planned = cost["planned_calls"]
        assert planned["sessions"] == 19
        assert planned["ingest_calls"] == 19
        assert planned["await_ready_calls"] == 19  # async fake
        assert planned["open_calls"] == planned["close_calls"] == 27
        assert planned["retrieve_calls"] == 8
        assert planned["reader_calls"] == 8
        assert planned["judge_calls"] == 8
        actual = cost["actual_calls"]
        assert actual["ingest"] == 19
        assert actual["judge.evaluate"] == 8
        units = {u["method"]: u for u in cost["unit_costs"]}
        assert units["reader.answer"]["call_count"] == 8
        assert units["reader.answer"]["mean_output_tokens"] > 0
        assert units["judge.evaluate"]["mean_input_tokens"] > 0
        # Extrapolations are explicitly marked estimates.
        assert cost["estimated"] is True
        assert "reader.answer.input_tokens" in cost["estimated_totals"]
        assert (
            cost["estimated_totals"]["judge.evaluate.input_tokens"]
            == units["judge.evaluate"]["mean_input_tokens"] * 8
        )

    def test_registry_version_and_header(self, report):
        header = report["header"]
        assert header["metrics_registry_version"] == "1"
        assert header["metrics_registry_content_version"] == "metrics-registry@1"
        assert header["config_fingerprint"]
        assert header["reader_model_family"] != header["judge_model_family"]
        # Happy path: no invalid-input or pending caveats in limitations.
        assert not any("invalid_input" in n for n in report["limitations"])
        assert not any("pending" in n or "未到终态" in n for n in report["limitations"])

    def test_markdown_renders_all_sections(self, report):
        markdown = render_markdown(report)
        for section in (
            "# Memory Eval 运行汇总报告",
            "## 状态计数与分母",
            "## 正式指标（指标注册表 v1）",
            "## 检索 × 问答 2×2 联合归因",
            "## 证据预算构成（诊断）",
            "## 规模与成本模型",
            "## 限制",
        ):
            assert section in markdown
        assert "hit_wrong = 4" in markdown
        assert "metrics-registry@1" in markdown


class TestFailurePathReports:
    def test_reader_failure_recall_survives(self, tmp_path: Path):
        config = custom_config(reader_extra={"mode": "fail"})
        outcome = run_one(tmp_path, config, tag="readfail")
        assert outcome.failed == 1
        report = Reporter(outcome.run_dir).build()
        statuses = report["statuses"]
        assert statuses["failed"] == 1 and statuses["scored"] == 0
        # Failures contribute zero to the planned score, never "wrong".
        assert metric(report, "planned_question_score")["value"] == 0.0
        assert metric(report, "scored_accuracy")["status"] == "not_applicable"
        for cell in ("hit_correct", "hit_wrong", "miss_correct", "miss_wrong"):
            assert metric(report, f"attribution_{cell}")["status"] == "not_applicable"
        # Retrieval results are not erased by the reader failure.
        assert metric(report, "verifiable_session_recall_macro")["value"] == 1.0
        row = report["failures"][0]
        assert row["failed_stage"] == "read"
        assert row["error_code"] == "reader_backend_failed"

    def test_judge_unparseable_failure(self, tmp_path: Path):
        config = custom_config(judge_extra={"verdict_rule": "unparseable"})
        outcome = run_one(tmp_path, config, tag="judgefail")
        assert outcome.failed == 1
        report = Reporter(outcome.run_dir).build()
        row = report["failures"][0]
        assert row["failed_stage"] == "judge"
        assert row["error_code"] == "judge_output_unparseable"
        assert metric(report, "verifiable_session_recall_macro")["value"] == 1.0
        assert metric(report, "planned_question_score")["value"] == 0.0

    def test_invalid_gold_marks_invalid_input(self, tmp_path: Path):
        class RogueDataset(ManualDataset):
            def get_scoring_data(self, handle):
                if handle == "smoke_single_session_user_0001":
                    raise ContractError(
                        code="gold_source_unknown",
                        message="answer_session_ids do not match any session",
                    )
                return super().get_scoring_data(handle)

        config = custom_config()
        outcome = run_one(
            tmp_path, config, dataset=RogueDataset.load_default(), tag="invalid"
        )
        assert outcome.invalid_input == 1
        assert outcome.failed == 0
        report = Reporter(outcome.run_dir).build()
        assert report["statuses"]["invalid_input"] == 1
        row = report["failures"][0]
        assert row["qa_status"] == "invalid_input"
        assert row["error_code"] == "scoring_data_invalid"
        # The run is explicitly not a formal report.
        assert any("invalid_input" in note for note in report["limitations"])
        # No recall value was fabricated for the invalid sample.
        assert metric(report, "verifiable_session_recall_macro")["status"] == "not_applicable"

    def test_generated_only_adapter_recall_na_everywhere(self, tmp_path: Path):
        config = custom_config(
            memory={
                "name": "fake-memory",
                "baseline_kind": "adapter",
                "capabilities": ["generated_evidence", "state_inspection"],
                "config": {
                    "mutation_mode": "sync",
                    "evidence_kinds": ["generated"],
                },
            }
        )
        outcome = run_one(tmp_path, config, tag="genonly")
        assert outcome.scored == 1
        report = Reporter(outcome.run_dir).build()
        for metric_id in (
            "verifiable_session_recall_macro",
            "verifiable_session_recall_micro",
            "recall_at_1",
            "budgeted_session_recall",
        ):
            entry = metric(report, metric_id)
            assert entry["status"] == "not_applicable", metric_id
            assert "evidence_mode" in entry["reason"] or "E=0" in entry["reason"]
        # Hit criterion switches to non-empty evidence.
        attribution = report["attribution"]
        assert "nonempty_evidence" in attribution["hit_criteria"][0]


class TestReporterGuards:
    def test_rejects_non_run_directory(self, tmp_path: Path):
        from eval.report import ReportError

        with pytest.raises(ReportError):
            Reporter(tmp_path).build()

    def test_report_cli_command_prints_markdown(self, tmp_path: Path, capsys):
        from eval.cli import main

        config = load_config_toml(EXAMPLE_CONFIG)
        outcome = run_one(tmp_path, config, tag="cli")
        code = main(["report", "--run", str(outcome.run_dir)])
        assert code == 0
        out = capsys.readouterr().out
        assert "检索 × 问答 2×2 联合归因" in out
        assert "hit_wrong = 4" in out

    def test_report_cli_rejects_bad_dir(self, tmp_path: Path, capsys):
        from eval.cli import main

        code = main(["report", "--run", str(tmp_path / "nope")])
        assert code == 2
        assert "error" in capsys.readouterr().err
