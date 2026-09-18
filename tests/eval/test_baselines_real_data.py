"""The three baselines over the REAL pinned LongMemEval-S histories.

Offline (fake reader/judge, no network) but real data: the ~45-50
session haystacks per question exercise BM25 at its real scale and the
full-history precheck at its real ~120k-token size. Skips with an
explicit reason when the pinned data file has not been fetched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.config import load_config_dict
from eval.datasets.longmemeval import DEFAULT_LONGMEMEVAL_S_PATH
from eval.judges.fake import FakeJudge, FakeJudgeSpec
from eval.memories import build_memory_for_plan
from eval.readers.fake import FakeReader, FakeReaderSpec
from eval.runner import OfflineRunner
from eval.runs import RunStore

pytestmark = pytest.mark.skipif(
    not DEFAULT_LONGMEMEVAL_S_PATH.exists(),
    reason="pinned LongMemEval-S file not fetched; run "
    "scripts/fetch_longmemeval.py (explicit skip, not a silent pass)",
)

SMOKE_IDS = (
    "1de5cff2",
    "4dfccbf8",
    "6c49646a",
    "8550ddae",
    "a89d7624",
    "ba61f0b9",
    "f4f1d8a4_abs",
    "gpt4_70e84552_abs",
)


def _config(baseline: str, *, window: int | None, counting: str = "test"):
    tokenizer = {
        "test": "test:char-v1",
        "estimated": "estimated:chars4-v1",
    }[counting]
    reader = {
        "model": "fake-reader",
        "model_family": "family-r",
        "base_url": "offline://fake",
        "temperature": 0.0,
        "max_output_tokens": 1024,
        "tokenizer_id": tokenizer,
        "counting_mode": counting,
    }
    if window is not None:
        reader["context_window_tokens"] = window
        reader["format_overhead_tokens"] = 64
        reader["output_reserve_tokens"] = 2048
    return load_config_dict(
        {
            "name": f"real-baseline-{baseline}",
            "dataset_plan": "longmemeval-s-cleaned@1",
            "sample_plan_id": "longmemeval-s-smoke-8",
            "sample_ids": list(SMOKE_IDS),
            "smoke_subset_ids": [],
            "memory": {
                "name": f"baseline-{baseline}",
                "baseline_kind": baseline,
                "capabilities": (
                    ["extractive_evidence"]
                    if baseline in ("bm25", "full_history")
                    else []
                ),
                "config": {},
            },
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
    )


def _run(tmp_path: Path, config, run_id: str):
    from eval.datasets.longmemeval import LongMemEvalDataset

    runner = OfflineRunner(
        config=config,
        dataset=LongMemEvalDataset.load_default(),
        adapter=build_memory_for_plan(config.memory),
        reader=FakeReader(FakeReaderSpec.from_reader_plan(config.reader)),
        judge=FakeJudge(FakeJudgeSpec.from_judge_plan(config.judge)),
        store=RunStore(tmp_path / "runs", run_id),
        run_id=run_id,
    )
    return runner.run()


class TestRealDataBaselines:
    def test_bm25_runs_all_real_questions_with_ranked_recall(self, tmp_path):
        config = _config("bm25", window=4096 + 4096)
        outcome = _run(tmp_path, config, "run-real-bm25")
        assert outcome.scored == 8
        assert outcome.failed == 0
        assert outcome.context_exceeded == 0
        report = json.loads(
            (outcome.run_dir / "report.json").read_text(encoding="utf-8")
        )
        metrics = {m["metric_id"]: m for m in report["metrics"]}
        # Ranked baseline: recall is computed over the real gold sets
        assert metrics["verifiable_session_recall_macro"]["status"] == "computed"
        # BM25 keeps at most 10 sessions; evidence stays inside the 4K cap
        prepared_docs = list((outcome.run_dir / "artifacts").glob(
            "*/prepared_evidence.json"
        ))
        assert len(prepared_docs) == 8
        for path in prepared_docs:
            prepared = json.loads(path.read_text(encoding="utf-8"))["prepared"]
            assert prepared["token_count"] <= 4096
            sessions = {
                item["evidence"]["extractive_span"]["session_id"]
                for item in prepared["items"]
            }
            assert len(sessions) <= 10

    def test_full_history_fits_a_1m_window_and_is_never_truncated(self, tmp_path):
        config = _config("full_history", window=1_000_000)
        outcome = _run(tmp_path, config, "run-real-full")
        assert outcome.scored == 8
        assert outcome.context_exceeded == 0
        for artifact in outcome.results:
            result = artifact.result
            store = RunStore(tmp_path / "runs", outcome.run_id)
            prepared = json.loads(
                store.resolve_ref(
                    result.artifact_refs["prepared_evidence"]
                ).read_text(encoding="utf-8")
            )["prepared"]
            # Real histories are ~120k tokens: fully retained, unbounded
            # by the 4K retrieval budget, no truncation, no drops
            assert prepared["token_count"] > 100_000
            assert prepared["budget"] > 4096
            assert prepared["dropped_raw_indices"] == []
            assert all(not item["truncated"] for item in prepared["items"])

    def test_full_history_over_tight_window_marks_context_exceeded(self, tmp_path):
        # A 32k window cannot hold ~120k-token histories: every question
        # must be context_exceeded BEFORE any reader call, and runnable
        # coverage reflects the real denominator.
        config = _config("full_history", window=32_000)
        outcome = _run(tmp_path, config, "run-real-ce")
        assert outcome.context_exceeded == 8
        assert outcome.scored == 0
        report = json.loads(
            (outcome.run_dir / "report.json").read_text(encoding="utf-8")
        )
        metrics = {m["metric_id"]: m for m in report["metrics"]}
        assert metrics["runnable_coverage"]["value"] == 0.0
        assert metrics["planned_question_score"]["value"] == 0.0
        assert metrics["scored_accuracy"]["status"] == "not_applicable"

    def test_none_baseline_scores_all_real_questions(self, tmp_path):
        config = _config("none", window=8192)
        outcome = _run(tmp_path, config, "run-real-none")
        assert outcome.scored == 8
        report = json.loads(
            (outcome.run_dir / "report.json").read_text(encoding="utf-8")
        )
        metrics = {m["metric_id"]: m for m in report["metrics"]}
        assert (
            metrics["verifiable_session_recall_macro"]["status"]
            == "not_applicable"
        )


class TestSameQuestionSetAcrossBaselines:
    """AC1: the three baselines run the SAME dev-set question set.

    The committed dev50 configs pin one identical sample list (checked
    in tests/eval/test_baselines.py so CI always runs it); a 12-question
    slice of that list then runs through all three baselines offline,
    each producing its own report over the identical planned set.
    """

    def test_three_baselines_report_over_identical_planned_set(self, tmp_path):
        import json as _json

        split = _json.loads(
            Path(
                "eval/datasets/splits/longmemeval_s_cleaned/split.json"
            ).read_text(encoding="utf-8")
        )
        shared = sorted(split["dev_ids"])[:12]
        planned = None
        for baseline in ("none", "bm25", "full_history"):
            config = _config(baseline, window=1_000_000)
            config = config.model_copy(
                update={
                    "sample_ids": tuple(shared),
                    "sample_plan_id": "longmemeval-s-dev-50",
                }
            )
            outcome = _run(tmp_path, config, f"run-shared-{baseline}")
            report = _json.loads(
                (outcome.run_dir / "report.json").read_text(encoding="utf-8")
            )
            assert report["statuses"]["planned"] == 12
            assert report["statuses"]["scored"] == 12
            ids = {r.result.sample_handle for r in outcome.results}
            if planned is None:
                planned = ids
            else:
                assert ids == planned  # identical question set
            # each baseline's report labels its own control role
            assert report["header"]["baseline_control_role"]["role"] == {
                "none": "no_memory_control",
                "bm25": "equal_budget_baseline",
                "full_history": "full_history_control",
            }[baseline]


class TestQuestionAndPromptOutsideEvidenceBudget:
    """AC2: the question and public prompt never enter the evidence
    budget (they ride the shared query context, not the evidence render),
    while the reader call usage still accounts for them."""

    def test_prepared_render_excludes_question_and_prompt(self, tmp_path):
        from eval.datasets.longmemeval import LongMemEvalDataset

        dataset = LongMemEvalDataset.load_default()
        config = _config("bm25", window=8192)
        outcome = _run(tmp_path, config, "run-budget-scope")
        assert outcome.scored == 8
        store = RunStore(tmp_path / "runs", outcome.run_id)
        for artifact in outcome.results:
            question = dataset.get_question(artifact.result.sample_handle)
            prepared = json.loads(
                store.resolve_ref(
                    artifact.result.artifact_refs["prepared_evidence"]
                ).read_text(encoding="utf-8")
            )["prepared"]
            # Evidence budget covers ONLY the evidence render
            assert prepared["token_count"] <= 4096
            # The question text is not part of the budgeted render
            assert question.query not in prepared["rendered_text"]
            # ...but the reader call usage counts question + render (the
            # fake reader mirrors a real call's input accounting)
            reader_doc = json.loads(
                store.resolve_ref(
                    artifact.result.artifact_refs["reader_result"]
                ).read_text(encoding="utf-8")
            )
            usage = reader_doc["result"]["usage"]
            assert usage["input_tokens"] == (
                len(question.query) + prepared["token_count"]
            )
