"""smoke-live: the fixed 8-question subset through the REAL models.

These tests cost money and need credentials + the pinned dataset and
tokenizer files. Every test skips with an explicit reason (naming the
missing prerequisite) instead of passing silently; run them with:

    export DEEPSEEK_API_KEY=... ZAI_API_KEY=...
    uv run python scripts/fetch_longmemeval.py
    uv run python scripts/fetch_deepseek_tokenizer.py
    uv run pytest tests/eval/test_live_smoke.py -r s

Cost control (docs/design/eval-harness.md, 已确定事项-冒烟数据): 8
questions per baseline (6 question types + 2 abstention) plus the fixed
10-probe drift set per run; smoke scores never enter formal reports.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from eval.config import load_config_toml
from eval.datasets.longmemeval import DEFAULT_LONGMEMEVAL_S_PATH
from eval.prepare.tokens import DEFAULT_DEEPSEEK_TOKENIZER_PATH

SMOKE_CONFIGS = {
    "none": "eval/configs/examples/real_smoke_live_none.toml",
    "bm25": "eval/configs/examples/real_smoke_live_bm25.toml",
    "full_history": "eval/configs/examples/real_smoke_live_full_history.toml",
}


def _config(kind: str):
    return load_config_toml(SMOKE_CONFIGS[kind])


def _skip_reasons(kind: str) -> list[str]:
    config = _config(kind)
    reasons: list[str] = []
    if not os.environ.get(config.reader.api_key_env):
        reasons.append(
            f"reader credential env {config.reader.api_key_env} not set"
        )
    if not os.environ.get(config.judge.api_key_env):
        reasons.append(
            f"judge credential env {config.judge.api_key_env} not set"
        )
    if not DEFAULT_LONGMEMEVAL_S_PATH.exists():
        reasons.append(
            "pinned LongMemEval-S file missing (scripts/fetch_longmemeval.py)"
        )
    if not DEFAULT_DEEPSEEK_TOKENIZER_PATH.exists():
        reasons.append(
            "pinned DeepSeek tokenizer missing "
            "(scripts/fetch_deepseek_tokenizer.py)"
        )
    return reasons


def _live(kind: str):
    reasons = _skip_reasons(kind)
    return pytest.mark.skipif(
        bool(reasons),
        reason=f"live smoke skipped ({'; '.join(reasons)}) — explicit skip, "
        "not a silent pass",
    )


def _run_live(kind: str, tmp_path: Path):
    from eval.datasets import load_dataset_for_config
    from eval.judges import build_judge_for_plan
    from eval.memories import build_memory_for_plan
    from eval.readers import build_reader_for_plan
    from eval.runner import OfflineRunner
    from eval.runs import RunStore

    config = _config(kind)
    runner = OfflineRunner(
        config=config,
        dataset=load_dataset_for_config(config),
        adapter=build_memory_for_plan(config.memory),
        reader=build_reader_for_plan(config.reader),
        judge=build_judge_for_plan(config.judge),
        store=RunStore(tmp_path / "runs", f"run-live-{kind}"),
        run_id=f"run-live-{kind}",
    )
    return runner.run()


class TestLiveDriftProbes:
    """The version record and probe outputs against the real reader."""

    @_live("bm25")
    def test_probes_execute_and_version_record_observes_model(self, tmp_path):
        from eval.models import run_reader_probes
        from eval.readers import build_reader_for_plan

        config = _config("bm25")
        reader = build_reader_for_plan(config.reader)
        probe = run_reader_probes(
            run_id="probe-live",
            probe_set_id=config.reader.probe_set_id,
            reader=reader,
            tokenizer_id=config.reader.tokenizer_id,
            counting_mode=config.reader.counting_mode,
        )
        assert len(probe.calls) == 10
        assert all(call.error is None for call in probe.calls)
        assert all(call.output for call in probe.calls)
        # The rolling alias resolved to something concrete this run
        assert reader.observed_models
        assert probe.calls[0].response_model in reader.observed_models


@pytest.mark.parametrize("kind", ["none", "bm25", "full_history"])
class TestLiveSmokeBaselines:
    """Each baseline runs the full 8-question loop against real models.

    Asserts the M2 machinery end to end: every question scored, exact
    counting with server-usage calibration recorded per call, the
    four-identifier version record with observed response models, the
    archived probe outputs, and the per-baseline control-role/N-A rules.
    """

    def test_smoke_loop(self, kind, tmp_path):
        reasons = _skip_reasons(kind)
        if reasons:
            pytest.skip(
                f"live smoke skipped ({'; '.join(reasons)}) — explicit "
                "skip, not a silent pass"
            )
        outcome = _run_live(kind, tmp_path)
        assert outcome.scored == 8, [r.result.qa_status for r in outcome.results]
        assert outcome.failed == 0
        assert outcome.invalid_input == 0
        # 1M window: the S smoke subset must fully run (design: full
        # history is expected runnable on S under this reader)
        assert outcome.context_exceeded == 0

        run_dir = outcome.run_dir
        versions = json.loads(
            (run_dir / "model_versions.json").read_text(encoding="utf-8")
        )
        # Four identifiers observed against the REAL endpoints
        assert versions["reader"]["response_model"]
        assert versions["judge"]["response_model"]
        assert versions["probe"] is not None
        probe_doc = json.loads(
            (run_dir / "artifacts" / "model_probes.json").read_text(
                encoding="utf-8"
            )
        )
        assert len(probe_doc["calls"]) == 10
        assert all(call["error"] is None for call in probe_doc["calls"])

        report = json.loads(
            (run_dir / "report.json").read_text(encoding="utf-8")
        )
        calibration = report["token_calibration"]
        assert calibration["calls_with_calibration"] == 8
        assert calibration["calls_with_server_usage"] == 8
        # Exact counting: budget enforced locally, deltas recorded
        assert calibration["counting_modes"] == ["exact"]

        header = report["header"]
        if kind in ("none", "full_history"):
            assert header["baseline_control_role"]["equal_budget"] is False
            recall = {
                m["metric_id"]: m
                for m in report["metrics"]
                if m["metric_id"] == "verifiable_session_recall_macro"
            }
            assert (
                recall["verifiable_session_recall_macro"]["status"]
                == "not_applicable"
            )
        else:
            assert header["baseline_control_role"]["equal_budget"] is True
