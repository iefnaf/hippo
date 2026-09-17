"""smoke-offline: the fixed 8-question smoke subset in CI (issue #6 AC1).

The example config locks the smoke subset (one question per question_type
plus two abstention samples); two deterministic runs of that config must
produce identical fingerprints and comparable scores, the whole loop
must not open a network socket (no external model), and the compare
command must mark the pair same-condition / equal-budget with an aligned
metric table over the full 8-question intersection.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from eval.cli import main
from eval.config import DEFAULT_SMOKE_SAMPLE_IDS, load_config_toml
from eval.compare import compare_runs
from eval.datasets.manual import ManualDataset

EXAMPLE_CONFIG = "eval/configs/examples/offline_fake.toml"


class _NoNetwork:
    """Patch that makes any socket creation fail loudly."""

    def __init__(self, monkeypatch) -> None:
        self.calls = 0

        def blocked(*args, **kwargs):
            self.calls += 1
            raise AssertionError(
                "smoke-offline must not open network sockets "
                "(no external model calls)"
            )

        monkeypatch.setattr(socket, "socket", blocked)
        monkeypatch.setattr(socket, "create_connection", blocked)


def _run_once(out: Path, run_id: str) -> Path:
    from eval.datasets.manual import ManualDataset as _MD
    from eval.judges.fake import FakeJudge, FakeJudgeSpec
    from eval.memories.fake import FakeMemoryAdapter, FakeMemorySpec
    from eval.readers.fake import FakeReader, FakeReaderSpec
    from eval.runner import OfflineRunner
    from eval.runs import RunStore

    config = load_config_toml(EXAMPLE_CONFIG)
    runner = OfflineRunner(
        config=config,
        dataset=_MD.load_default(),
        adapter=FakeMemoryAdapter(FakeMemorySpec.from_memory_plan(config.memory)),
        reader=FakeReader(FakeReaderSpec.from_reader_plan(config.reader)),
        judge=FakeJudge(FakeJudgeSpec.from_judge_plan(config.judge)),
        store=RunStore(out, run_id),
        run_id=run_id,
    )
    outcome = runner.run()
    assert outcome.failed == 0
    assert outcome.invalid_input == 0
    return outcome.run_dir


class TestSmokeOffline:
    def test_smoke_subset_locked_in_config(self):
        config = load_config_toml(EXAMPLE_CONFIG)
        assert config.smoke_subset_ids == DEFAULT_SMOKE_SAMPLE_IDS
        assert config.sample_ids == DEFAULT_SMOKE_SAMPLE_IDS
        assert len(DEFAULT_SMOKE_SAMPLE_IDS) == 8
        # Six question types covered once, plus two abstention samples
        dataset = ManualDataset.load_default()
        types = {
            dataset.get_scoring_data(h).question_type
            for h in DEFAULT_SMOKE_SAMPLE_IDS
            if not dataset.get_scoring_data(h).is_abstention
        }
        assert len(types) == 6
        abstentions = [
            h
            for h in DEFAULT_SMOKE_SAMPLE_IDS
            if dataset.get_scoring_data(h).is_abstention
        ]
        assert len(abstentions) == 2

    def test_two_deterministic_runs_comparable(self, tmp_path: Path):
        left = _run_once(tmp_path / "runs", "run-smoke-a")
        right = _run_once(tmp_path / "runs", "run-smoke-b")
        left_doc = json.loads((left / "config.json").read_text(encoding="utf-8"))
        right_doc = json.loads((right / "config.json").read_text(encoding="utf-8"))
        # Same config => same fingerprint (and same registry content)
        assert left_doc["config_fingerprint"] == right_doc["config_fingerprint"]

        payload = compare_runs(left, right)
        assert payload["comparability"]["same_condition"] is True
        assert payload["comparison_kind"] == "equal_budget"
        assert payload["metrics"]["refused_reason"] is None
        # Scores comparable: deterministic metrics align exactly
        latency = {"ingest_latency_ms", "retrieve_latency_ms_p50", "retrieve_latency_ms_p95"}
        for row in payload["metrics"]["aligned"]:
            if row["metric_id"] in latency:
                continue
            assert row["left"]["value"] == row["right"]["value"], row["metric_id"]
        # Full-P headline scores equal on the whole 8-question set
        left_full = payload["left_summary"]["full_planned_set"]
        right_full = payload["right_summary"]["full_planned_set"]
        assert left_full == right_full
        assert left_full["planned_question_score"] == 0.25
        assert payload["intersection"]["size"] == 8

    def test_smoke_scores_not_formal(self, tmp_path: Path):
        run_dir = _run_once(tmp_path / "runs", "run-smoke-limit")
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        assert any("smoke" in note for note in report["limitations"])

    def test_no_network_sockets_opened(self, tmp_path: Path, monkeypatch):
        guard = _NoNetwork(monkeypatch)
        run_dir = _run_once(tmp_path / "runs", "run-smoke-offline")
        assert (run_dir / "report.json").exists()
        assert guard.calls == 0  # the run completed without touching sockets

    def test_cli_smoke_pair_via_compare(self, tmp_path: Path, capsys):
        out = tmp_path / "runs"
        for _ in range(2):
            code = main(
                ["run", "--config", EXAMPLE_CONFIG, "--out", str(out)]
            )
            assert code == 0
            capsys.readouterr()
        runs = sorted(p for p in out.iterdir() if p.is_dir())
        assert len(runs) == 2
        code = main(["compare", str(runs[0]), str(runs[1])])
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["comparability"]["same_condition"] is True
        assert payload["intersection"]["runnable_intersection_ids"] == sorted(
            DEFAULT_SMOKE_SAMPLE_IDS
        )
