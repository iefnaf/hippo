"""LIVE calibration judge-batch smoke (issue #9).

Costs money and needs ZAI_API_KEY. Skips with an explicit reason when
the credential is missing — an explicit skip, never a silent pass.
Run with:

    export ZAI_API_KEY=...
    uv run pytest tests/eval/test_judge_calibration_live.py -r s

The smoke exercises the exact production path: a tiny plan (2 items)
through run_judge_batch against the real OpenAIChatJudge bound to the
official anscheck protocol, then statistics + decision + record on the
REAL judge outputs. No dataset files are needed.
"""

from __future__ import annotations

import os

import pytest

from eval.calibration.criteria import criteria_digest
from eval.calibration.sampling import (
    CalibrationCandidate,
    CalibrationPlanArtifact,
    build_calibration_plan,
)

JUDGE_CONFIG = "eval/configs/examples/real_dev50_bm25.toml"


def _skip_reasons() -> list[str]:
    from eval.config import load_config_toml

    config = load_config_toml(JUDGE_CONFIG)
    reasons: list[str] = []
    if not os.environ.get(config.judge.api_key_env):
        reasons.append(
            f"judge credential env {config.judge.api_key_env} not set"
        )
    return reasons


def _live():
    reasons = _skip_reasons()
    return pytest.mark.skipif(
        bool(reasons),
        reason=(
            f"live calibration smoke skipped ({'; '.join(reasons)}) — "
            "explicit skip, not a silent pass"
        ),
    )


def _tiny_plan() -> CalibrationPlanArtifact:
    candidates = [
        CalibrationCandidate(
            condition=cond,
            run_id=f"live-{cond}",
            # Both conditions answer the SAME planned sample (shared
            # handle): disjoint handles would leave an empty common
            # population and the plan build refuses to proceed.
            sample_handle="live-001",
            question="Which package manager does the project use now?",
            expected_answer="pnpm",
            hypothesis="The project now uses pnpm.",
            question_type="multi-session",
            is_abstention=False,
        )
        for cond in ("A", "B")
    ]
    return build_calibration_plan(
        candidates,
        created_at="2026-09-18T00:00:00Z",
        strict=False,
        random_size=2,
        boundary_size=0,
        self_consistency_size=1,
    )


class TestLiveJudgeBatch:
    @_live()
    def test_batch_statistics_decision_and_record(self, tmp_path):
        from eval.calibration.judge_batch import run_judge_batch
        from eval.calibration.record import build_calibration_record
        from eval.calibration.sampling import AnnotationRecord
        from eval.calibration.stats import compute_statistics, decide_calibration
        from eval.config import load_config_toml
        from eval.judges import build_judge_for_plan

        config = load_config_toml(JUDGE_CONFIG)
        judge = build_judge_for_plan(config.judge)
        plan = _tiny_plan()
        artifact = run_judge_batch(
            plan, judge, api_key_env=config.judge.api_key_env, sleep=lambda _s: None
        )
        assert len(artifact.calls) == 2
        # the real endpoint answered with a parseable yes/no on both items
        assert all(call.verdict is not None for call in artifact.calls)
        assert artifact.observed_response_models

        # the correct gold answer must be judged yes by the real model
        yes_calls = [c for c in artifact.calls if c.verdict]
        assert yes_calls, "expected the real judge to accept the pnpm answer"

        annotations = [
            AnnotationRecord(
                item_id=i.item_id,
                annotation="yes" if "pnpm" in i.hypothesis else "no",
                annotated_at="2026-09-18",
            )
            for i in plan.items
        ]
        statistics = compute_statistics(
            items=list(plan.items),
            annotations=annotations,
            judge_calls=artifact.calls,
        )
        decision = decide_calibration(statistics)
        record = build_calibration_record(
            plan=plan,
            annotations=annotations,
            self_consistency_annotations=[],
            judge_calls=artifact,
            statistics=statistics,
            decision=decision,
            judge_vendor_documented_version=config.judge.vendor_documented_version,
            judge_vendor_documented_on=config.judge.vendor_documented_on,
            created_at="2026-09-18T12:00:00Z",
        )
        assert record.criteria_digest == criteria_digest()
        assert record.judge.alias == config.judge.model
        assert record.judge.response_models
        out = tmp_path / "calibration.json"
        out.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        assert out.exists()
