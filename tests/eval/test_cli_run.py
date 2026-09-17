"""CLI run command: offline execution with --out, plan without it."""

from __future__ import annotations

import json
from pathlib import Path

from eval.cli import main


class TestRunCommand:
    def test_without_out_still_prints_offline_plan(self, capsys, tmp_path):
        code = main(
            ["run", "--config", "eval/configs/examples/offline_fake.toml"]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["command"] == "run"
        assert len(payload["config_fingerprint"]) == 64
        assert payload["metrics_registry_version"] == "1"
        assert "not-executed" in payload["status"]

    def test_with_out_executes_offline_loop(self, capsys, tmp_path: Path):
        out = tmp_path / "runs"
        code = main(
            [
                "run",
                "--config", "eval/configs/examples/offline_fake.toml",
                "--out", str(out),
            ]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        run_dir = Path(payload["run_dir"])
        assert run_dir.is_dir()
        assert (run_dir / "config.json").exists()
        assert (run_dir / "samples.jsonl").exists()
        lines = (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == payload["sample_count"] == 8
        assert payload["failed"] == 0
        assert payload["scored"] == 8
        assert payload["invalid_input"] == 0
        # The run summary now carries the report and headline metrics.
        assert (run_dir / "report.json").exists()
        assert (run_dir / "report.md").exists()
        headline = payload["headline_metrics"]
        assert headline["planned_question_score"] == 0.25
        assert headline["verifiable_session_recall_macro"] == 1.0
        assert headline["recall_at_1"] == 0.583333

    def test_report_command_prints_summary_markdown(self, capsys, tmp_path: Path):
        out = tmp_path / "runs"
        code = main(
            [
                "run",
                "--config", "eval/configs/examples/offline_fake.toml",
                "--out", str(out),
            ]
        )
        assert code == 0
        run_dir = Path(json.loads(capsys.readouterr().out)["run_dir"])
        code = main(["report", "--run", str(run_dir)])
        assert code == 0
        markdown = capsys.readouterr().out
        assert "## 正式指标（指标注册表 v1）" in markdown
        assert "planned_question_score" in markdown
        assert "## 规模与成本模型" in markdown

    def test_missing_dataset_is_structured(self, capsys, tmp_path: Path):
        out = tmp_path / "runs"
        code = main(
            [
                "run",
                "--config", "eval/configs/examples/offline_fake.toml",
                "--out", str(out),
                "--dataset", str(tmp_path / "nope.json"),
            ]
        )
        assert code == 2
        err = capsys.readouterr().err
        assert "dataset_missing" in err

    def test_invalid_config_exits_two(self, capsys):
        try:
            main(
                [
                    "run",
                    "--config", "eval/configs/examples/invalid_bad_family.toml",
                    "--out", "/tmp/should-not-run",
                ]
            )
        except SystemExit as exc:
            assert exc.code == 2
        else:  # pragma: no cover
            raise AssertionError("expected SystemExit")
        assert "config" in capsys.readouterr().err
