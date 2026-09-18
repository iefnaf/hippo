"""QA scorer: recall math, applicability, attribution and failure rules."""

from __future__ import annotations

import pytest

from eval.contracts.adapter import Evidence, SourceSpan
from eval.contracts.common import ContractError
from eval.contracts.internal import ReaderResult, ScoringData
from eval.datasets.manual import ManualDataset
from eval.judges.fake import FakeJudge, FakeJudgeSpec
from eval.prepare.evidence import build_history_index, prepare_evidence
from eval.scorers.qa import (
    QAScorer,
    ScoringDataError,
    retained_extractive_sessions,
    session_recall,
    session_recall_at,
)

USER = "smoke_single_session_user_0001"
ABSTENTION = "smoke_abstention_0001"
MULTI = "smoke_multi_session_0001"


def _evidence(session, message, text=None):
    text = text if text is not None else message.content
    return Evidence(
        kind="extractive",
        text=text,
        extractive_span=SourceSpan(
            session_id=session.session_id,
            msg_id=message.msg_id,
            start=0,
            end=len(text),
        ),
        derivation_sources=[],
        source_times=[session.occurred_at],
        retrieval_score=None,
    )


def _sessions(handle):
    return ManualDataset.load_default().iter_sessions(handle)


def _prepared(handle, indices, budget=4096):
    """Prepared evidence built from the given session indices, in order."""
    sessions = _sessions(handle)
    chosen = [sessions[i] for i in indices]
    evidence = [
        _evidence(session, message)
        for session in chosen
        for message in session.messages
    ]
    return prepare_evidence(
        evidence, build_history_index(sessions), budget=budget
    )


def _scorer(dataset=None, judge=None, **kwargs):
    defaults = dict(
        dataset=dataset or ManualDataset.load_default(),
        judge=judge or FakeJudge(FakeJudgeSpec()),
        extractive_declared=True,
        baseline_kind="adapter",
        protocol_id="longmemeval-yes-no@1",
    )
    defaults.update(kwargs)
    return QAScorer(**defaults)


def _answer(text):
    return ReaderResult(
        hypothesis=text, raw_output=text, model="fake-reader", usage=None
    )


class TestRecallHelpers:
    def test_session_recall_and_at_k(self):
        gold = ["g1", "g2"]
        ordered = ["x", "g1", "y", "g2"]
        assert session_recall(gold, ordered) == 1.0
        assert session_recall_at(gold, ordered, 1) == 0.0
        assert session_recall_at(gold, ordered, 3) == 0.5
        assert session_recall_at(gold, ordered, 5) == 1.0

    def test_distinct_sessions_deduplicated_in_order(self):
        gold = ["g1"]
        ordered = ["g1", "g1", "g1", "x"]
        assert session_recall_at(gold, ordered, 2) == 1.0
        assert session_recall_at(gold, ordered, 1) == 1.0

    def test_retained_extractive_sessions_ignores_generated(self):
        prepared = _prepared(USER, [1, 2])
        sessions = retained_extractive_sessions(prepared)
        assert len(sessions) == 2  # distinct sessions, extractive only

    def test_empty_gold_raises(self):
        with pytest.raises(ValueError):
            session_recall([], ["x"])


class TestRecallComputation:
    """AC: recall counts only actually retained original-text evidence."""

    def test_gold_session_hit_and_miss(self):
        # Sessions of USER: [0]=haystack, [1]=npm, [2]=pnpm(gold), [3]=haystack
        scorer = _scorer()
        question = ManualDataset.load_default().get_question(USER)
        both = scorer.score_sample(
            run_id="r",
            handle=USER,
            question=question,
            prepared=_prepared(USER, [1, 2]),
            reader_result=_answer("迁移到 pnpm，以后都用 pnpm。"),
        )
        metrics = {m.metric_id: m for m in both.metrics}
        assert metrics["verifiable_session_recall_macro"].value == 1.0
        assert metrics["budgeted_session_recall"].value == 1.0
        assert both.trace.hit_gold_sessions == both.trace.gold_internal_sessions

        only_npm = scorer.score_sample(
            run_id="r",
            handle=USER,
            question=question,
            prepared=_prepared(USER, [1]),
            reader_result=_answer("项目使用 npm。"),
        )
        metrics = {m.metric_id: m for m in only_npm.metrics}
        assert metrics["verifiable_session_recall_macro"].value == 0.0
        assert only_npm.trace.hit_gold_sessions == []

    def test_dropped_gold_unit_does_not_count(self):
        # A giant haystack unit eats the budget and the gold unit is dropped
        # entirely: its session must NOT count as retained evidence.
        scorer = _scorer()
        question = ManualDataset.load_default().get_question(MULTI)
        sessions = _sessions(MULTI)
        giant = _evidence(sessions[2], sessions[2].messages[0])  # 例会备份
        gold_unit = _evidence(sessions[1], sessions[1].messages[0])  # gold
        assert len(giant.text) > 1000
        prepared = prepare_evidence(
            [giant, gold_unit], build_history_index(sessions), budget=400
        )
        assert prepared.dropped_raw_indices == [1]
        scoring = scorer.score_sample(
            run_id="r",
            handle=MULTI,
            question=question,
            prepared=prepared,
            reader_result=_answer("民宿老板确认：整栋最多住 8 人，带院子。"),
        )
        metrics = {m.metric_id: m for m in scoring.metrics}
        assert metrics["verifiable_session_recall_macro"].value == 0.0
        assert scoring.trace.actual_sessions_in_order == [
            sessions[2].session_id
        ]

    def test_truncated_gold_unit_still_counts(self):
        # A gold unit truncated to a prefix still counts: its source was kept.
        scorer = _scorer()
        question = ManualDataset.load_default().get_question(USER)
        sessions = _sessions(USER)
        history = build_history_index(sessions)
        unit = _evidence(sessions[2], sessions[2].messages[0])
        prepared = None
        for extra in range(1, 120):
            candidate = prepare_evidence(
                [unit], history, budget=len(unit.text) + extra
            )
            if any(item.truncated for item in candidate.items):
                prepared = candidate
                break
        assert prepared is not None, "no budget produced a truncated unit"
        scoring = scorer.score_sample(
            run_id="r",
            handle=USER,
            question=question,
            prepared=prepared,
            reader_result=_answer("迁移到 pnpm"),
        )
        metrics = {m.metric_id: m for m in scoring.metrics}
        assert metrics["verifiable_session_recall_macro"].value == 1.0

    def test_multi_gold_partial_recall(self):
        scorer = _scorer()
        question = ManualDataset.load_default().get_question(MULTI)
        # MULTI gold = sessions 0 and 1; retain only session 1.
        scoring = scorer.score_sample(
            run_id="r",
            handle=MULTI,
            question=question,
            prepared=_prepared(MULTI, [1]),
            reader_result=_answer("民宿老板确认：整栋最多住 8 人，带院子。"),
        )
        metrics = {m.metric_id: m for m in scoring.metrics}
        assert metrics["verifiable_session_recall_macro"].value == pytest.approx(
            0.5
        )
        assert len(scoring.trace.gold_internal_sessions) == 2
        assert len(scoring.trace.hit_gold_sessions) == 1

    def test_generated_derivation_sources_never_enter_recall(self):
        scorer = _scorer()
        question = ManualDataset.load_default().get_question(USER)
        sessions = _sessions(USER)
        from eval.contracts.adapter import SourceRef

        generated = Evidence(
            kind="generated",
            text="项目目前统一使用 pnpm。",
            extractive_span=None,
            derivation_sources=[
                SourceRef(
                    session_id=sessions[2].session_id,
                    msg_id=sessions[2].messages[0].msg_id,
                )
            ],
            source_times=[sessions[2].occurred_at],
            retrieval_score=None,
        )
        prepared = prepare_evidence(
            [generated], build_history_index(sessions), budget=128
        )
        scoring = scorer.score_sample(
            run_id="r",
            handle=USER,
            question=question,
            prepared=prepared,
            reader_result=_answer("pnpm"),
        )
        metrics = {m.metric_id: m for m in scoring.metrics}
        # Self-reported derivation sources are diagnostics only.
        assert metrics["verifiable_session_recall_macro"].value == 0.0
        assert scoring.trace.actual_sessions_in_order == []
        assert metrics["derivation_source_coverage"].value == 1.0


class TestApplicability:
    """AC: abstention recall N/A but still judged; missing gold is an error."""

    def test_abstention_recall_na_but_answer_judged(self):
        scorer = _scorer()
        dataset = ManualDataset.load_default()
        scoring = scorer.score_sample(
            run_id="r",
            handle=ABSTENTION,
            question=dataset.get_question(ABSTENTION),
            prepared=_prepared(ABSTENTION, [0, 1]),
            reader_result=_answer("把仓库的镜像源换成国内的。"),
        )
        metrics = {m.metric_id: m for m in scoring.metrics}
        for metric_id in (
            "verifiable_session_recall_macro",
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "budgeted_session_recall",
        ):
            assert metrics[metric_id].status == "not_applicable"
            assert metrics[metric_id].value is None
            assert "abstention" in metrics[metric_id].reason
        # The answer still went through the judge.
        assert scoring.judge_call is not None
        assert scoring.correct is False  # substring rule misses
        assert scoring.attribution == "miss_wrong"  # empty gold: never a hit
        assert scoring.trace.recall_applicable is False

    def test_non_extractive_adapter_recall_na(self):
        scorer = _scorer(extractive_declared=False)
        dataset = ManualDataset.load_default()
        scoring = scorer.score_sample(
            run_id="r",
            handle=USER,
            question=dataset.get_question(USER),
            prepared=_prepared(USER, [2]),
            reader_result=_answer("pnpm"),
        )
        metrics = {m.metric_id: m for m in scoring.metrics}
        assert (
            metrics["verifiable_session_recall_macro"].status
            == "not_applicable"
        )
        assert "evidence_mode" in metrics["verifiable_session_recall_macro"].reason
        # Non-verifiable hit criterion: non-empty retained evidence.
        assert scoring.trace.hit_criterion == "nonempty_evidence"
        assert scoring.trace.hit is True
        assert scoring.attribution == "hit_correct"

    def test_none_baseline_never_hits(self):
        scorer = _scorer(baseline_kind="none")
        dataset = ManualDataset.load_default()
        scoring = scorer.score_sample(
            run_id="r",
            handle=USER,
            question=dataset.get_question(USER),
            prepared=_prepared(USER, [2]),
            reader_result=_answer("pnpm"),
        )
        assert scoring.trace.hit_criterion == "never"
        assert scoring.trace.hit is False
        assert scoring.attribution == "miss_correct"

    def test_missing_gold_raises_data_validation_error(self):
        class RogueDataset(ManualDataset):
            def get_scoring_data(self, handle):
                if handle == USER:
                    raise ContractError(
                        code="gold_source_unknown",
                        message="answer_session_ids do not match sessions",
                    )
                return super().get_scoring_data(handle)

        scorer = _scorer(dataset=RogueDataset.load_default())
        dataset = ManualDataset.load_default()
        with pytest.raises(ScoringDataError) as excinfo:
            scorer.score_sample(
                run_id="r",
                handle=USER,
                question=dataset.get_question(USER),
                prepared=_prepared(USER, [2]),
                reader_result=_answer("pnpm"),
            )
        assert excinfo.value.code == "scoring_data_invalid"

    def test_gold_id_missing_from_mapping_raises(self):
        scoring_data = ScoringData(
            expected_answer="pnpm",
            gold_source_ids=["s_ghost"],
            internal_to_official_session={"s_real": "session_x"},
            is_abstention=False,
            question_type="single-session-user",
            official_fields={},
        )

        class GhostDataset:
            def get_scoring_data(self, handle):
                return scoring_data

        scorer = _scorer(dataset=GhostDataset())
        dataset = ManualDataset.load_default()
        with pytest.raises(ScoringDataError) as excinfo:
            scorer.score_sample(
                run_id="r",
                handle=USER,
                question=dataset.get_question(USER),
                prepared=None,
                reader_result=None,
            )
        assert "mapping" in excinfo.value.message

    def test_empty_retained_set_scores_zero_not_excluded(self):
        scorer = _scorer()
        dataset = ManualDataset.load_default()
        scoring = scorer.score_sample(
            run_id="r",
            handle=USER,
            question=dataset.get_question(USER),
            prepared=None,  # e.g. ingest/retrieve failed earlier
            reader_result=None,
        )
        metrics = {m.metric_id: m for m in scoring.metrics}
        assert metrics["verifiable_session_recall_macro"].status == "computed"
        assert metrics["verifiable_session_recall_macro"].value == 0.0
        assert scoring.correct is None
        assert scoring.attribution is None


class TestJudgePath:
    def test_judge_verdict_and_injection(self):
        judge = FakeJudge(
            FakeJudgeSpec(
                verdict_overrides={"这个项目现在使用什么包管理器？": True}
            )
        )
        scorer = _scorer(judge=judge)
        dataset = ManualDataset.load_default()
        scoring = scorer.score_sample(
            run_id="r",
            handle=USER,
            question=dataset.get_question(USER),
            prepared=_prepared(USER, [2]),
            reader_result=_answer("项目使用 npm。"),  # substring says wrong
        )
        assert scoring.correct is True  # injected override wins
        assert scoring.attribution == "hit_correct"
        assert scoring.judge_call.error is None
        assert scoring.judge_call.result.raw_output == "yes"

    def test_unparseable_judge_fails_not_wrong(self):
        scorer = _scorer(judge=FakeJudge(FakeJudgeSpec(verdict_rule="unparseable")))
        dataset = ManualDataset.load_default()
        scoring = scorer.score_sample(
            run_id="r",
            handle=USER,
            question=dataset.get_question(USER),
            prepared=_prepared(USER, [2]),
            reader_result=_answer("pnpm"),
        )
        assert scoring.judge_call.error is not None
        assert scoring.judge_call.error.code == "judge_output_unparseable"
        assert scoring.judge_call.result is None
        assert scoring.correct is None  # never fabricated into wrong
        assert scoring.attribution is None
        # Recall survives the judge failure.
        metrics = {m.metric_id: m for m in scoring.metrics}
        assert metrics["verifiable_session_recall_macro"].value == 1.0

    def test_no_reader_answer_skips_judge(self):
        scorer = _scorer()
        dataset = ManualDataset.load_default()
        scoring = scorer.score_sample(
            run_id="r",
            handle=USER,
            question=dataset.get_question(USER),
            prepared=_prepared(USER, [2]),
            reader_result=None,
        )
        assert scoring.judge_call is None
        assert scoring.correct is None
        assert scoring.attribution is None

    def test_judge_request_carries_protocol_not_gold_sources(self):
        judge = FakeJudge(FakeJudgeSpec())
        scorer = _scorer(judge=judge)
        dataset = ManualDataset.load_default()
        scorer.score_sample(
            run_id="r",
            handle=USER,
            question=dataset.get_question(USER),
            prepared=_prepared(USER, [2]),
            reader_result=_answer("pnpm"),
        )
        request = judge.requests[0]
        assert request.protocol_id == "longmemeval-yes-no@1"
        assert request.expected_answer == "pnpm"  # official protocol needs it
        assert request.protocol_fields == {"abstention": False}  # M2: the
        # official protocol selects its abstention template from this
        # protocol-private flag; gold ids and ID mappings still never ride
        # along (asserted below)
        dumped = request.model_dump_json()
        assert "gold_source_ids" not in dumped
        assert "internal_to_official" not in dumped


class TestAttributionCells:
    def test_cell_combinations(self):
        from eval.scorers.qa import attribution_cell

        assert attribution_cell(True, True) == "hit_correct"
        assert attribution_cell(False, True) == "hit_wrong"
        assert attribution_cell(True, False) == "miss_correct"
        assert attribution_cell(False, False) == "miss_wrong"
