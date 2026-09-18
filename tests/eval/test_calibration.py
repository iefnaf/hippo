"""Judge calibration (issue #9): sampling, worksheets, statistics,
threshold decision, records, judge batch discipline and the
report/compare degradation channel.

Everything here is offline. The LIVE judge batch smoke lives in
tests/eval/test_judge_calibration_live.py and skips explicitly without
ZAI_API_KEY.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
from pathlib import Path

import pytest

from eval.calibration.criteria import (
    BOUNDARY_SAMPLE_SIZE,
    RANDOM_SAMPLE_SIZE,
    RUBRIC_ID,
    RUBRIC_SHA256,
    SELF_CONSISTENCY_SIZE,
    THRESHOLDS,
    criteria_digest,
    criteria_payload,
)
from eval.calibration.integration import (
    QA_CONCLUSION_METRIC_IDS,
    is_downgraded,
    judge_calibration_doc,
)
from eval.calibration.judge_batch import (
    CalibrationJudgeCall,
    CalibrationJudgeCallsArtifact,
    assert_judge_input_discipline,
    run_judge_batch,
)
from eval.calibration.record import (
    CalibrationRecordArtifact,
    bind_record_to_judge_plan,
    build_calibration_record,
    verify_decision_stability,
)
from eval.calibration.sampling import (
    WORKSHEET_COLUMNS,
    CalibrationCandidate,
    CalibrationPlanArtifact,
    build_calibration_plan,
    import_annotations,
    render_worksheet_csv,
    worksheet_rows,
)
from eval.calibration.stats import (
    CalibrationDecision,
    CalibrationStatistics,
    ConfusionCounts,
    SubsetAgreement,
    decide_calibration,
    wilson_interval,
)
from eval.contracts.adapter import ResourceUsage
from eval.contracts.common import ContractError
from eval.contracts.internal import JudgeRequest, JudgeResult
from eval.judges.longmemeval import (
    OFFICIAL_PROTOCOL_ID,
    render_official_prompt,
)

TYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "temporal-reasoning",
    "knowledge-update",
    "multi-session",
]


def make_candidate(condition: str, i: int, *, hypothesis: str | None = None) -> CalibrationCandidate:
    abst = i % 10 == 0
    return CalibrationCandidate(
        condition=condition,
        run_id=f"run-{condition}",
        sample_handle=f"q{i:03d}" + ("_abs" if abst else ""),
        question=f"Question {i}?",
        expected_answer=f"gold {i}",
        hypothesis=hypothesis or (f"gold {i}" if i % 2 == 0 else f"wrong {i} long enough"),
        question_type=TYPES[i % 6],
        is_abstention=abst,
    )


def make_population(n: int = 50) -> list[CalibrationCandidate]:
    return [make_candidate(c, i) for c in ("A", "B") for i in range(n)]


def build_plan(strict: bool = True, **kwargs):
    return build_calibration_plan(
        make_population(),
        created_at="2026-09-18T00:00:00Z",
        strict=strict,
        **kwargs,
    )


class FakeJudge:
    """Deterministic offline judge with the real client's attributes."""

    model = "glm-5.3"
    base_url = "https://api.z.ai/api/paas/v4"
    temperature = 0.0
    max_output_tokens = 10

    def __init__(self, agree_with_gold: bool = True) -> None:
        self.agree_with_gold = agree_with_gold
        self.requests: list[JudgeRequest] = []

    def evaluate(self, request: JudgeRequest) -> JudgeResult:
        self.requests.append(request)
        if self.agree_with_gold:
            yes = request.expected_answer in request.hypothesis
        else:
            yes = False
        raw = "Yes." if yes else "No."
        return JudgeResult(
            correct=yes, raw_output=raw, model=self.model, usage=None
        )


class TestCriteria:
    def test_thresholds_match_the_confirmed_design_values(self):
        assert THRESHOLDS == {
            "overall_agreement_min": 0.85,
            "overall_wilson_lower_min": 0.75,
            "per_question_type_agreement_min": 0.70,
            "cross_condition_max_diff": 0.05,
            "self_consistency_min": 0.85,
            "undecided_ratio_max": 0.10,
        }

    def test_sample_sizes_are_the_confirmed_ones(self):
        assert RANDOM_SAMPLE_SIZE == 100
        assert BOUNDARY_SAMPLE_SIZE == 20
        assert SELF_CONSISTENCY_SIZE == 20

    def test_criteria_digest_binds_rubric_protocol_and_thresholds(self):
        payload = criteria_payload()
        assert payload["rubric_sha256"] == RUBRIC_SHA256
        assert payload["protocol_id"] == OFFICIAL_PROTOCOL_ID
        # The digest changes when any criterion content changes.
        mutated = dict(payload)
        mutated["thresholds"] = {**payload["thresholds"], "overall_agreement_min": 0.9}
        import hashlib

        mutated_digest = hashlib.sha256(
            json.dumps(mutated, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        assert mutated_digest != criteria_digest()
        assert criteria_digest() == criteria_digest()  # stable


class TestWilsonInterval:
    def test_known_reference_values(self):
        assert wilson_interval(85, 100) == pytest.approx((0.767163, 0.906941), abs=1e-5)
        assert wilson_interval(95, 100) == pytest.approx((0.888248, 0.978457), abs=1e-5)
        assert wilson_interval(10, 10) == pytest.approx((0.72246, 1.0), abs=1e-5)
        assert wilson_interval(0, 10) == pytest.approx((0.0, 0.27754), abs=1e-5)

    def test_empty_denominator_is_none_never_a_point_estimate(self):
        assert wilson_interval(0, 0) is None

    def test_perfect_agreement_with_small_n_cannot_clear_the_wilson_gate(self):
        # n=10 all-agree: low bound 0.722 < 0.75 — honest statistics, the
        # design's abstention-subset gate deliberately bites here.
        lo, _hi = wilson_interval(10, 10)
        assert lo < 0.75


class TestSampling:
    def test_plan_shape_and_stratification(self):
        plan = build_plan()
        assert plan.random_size == 100
        assert plan.boundary_size == 20
        random_items = [i for i in plan.items if i.cohort == "random"]
        assert {i.question_type for i in random_items} == set(TYPES)
        assert any(i.is_abstention for i in random_items)
        assert len(plan.self_consistency_item_ids) == 20
        # both conditions present in the random cohort
        assert {i.condition for i in random_items} == {"A", "B"}

    def test_deterministic_same_seed_same_plan(self):
        assert build_plan().plan_id == build_plan().plan_id
        other = build_calibration_plan(
            make_population(), created_at="2026-09-19T00:00:00Z", seed="other"
        )
        assert other.plan_id != build_plan().plan_id

    def test_boundary_cohort_contains_abstention_and_hard_cases(self):
        plan = build_plan()
        boundary = [i for i in plan.items if i.cohort == "boundary"]
        assert len(boundary) == 20
        assert any(i.is_abstention for i in boundary)
        assert all(i.selection_reason for i in boundary)

    def test_strict_refuses_short_population(self):
        with pytest.raises(ContractError, match="random cohort needs"):
            build_calibration_plan(
                make_population(20), created_at="2026-09-18T00:00:00Z", strict=True
            )

    def test_strict_refuses_single_condition(self):
        with pytest.raises(ContractError, match="exactly two"):
            build_calibration_plan(
                [make_candidate("A", i) for i in range(50)],
                created_at="2026-09-18T00:00:00Z",
            )

    def test_strict_refuses_condition_handle_mismatch(self):
        left = [make_candidate("A", i) for i in range(50)]
        right = [make_candidate("B", i) for i in range(49)]  # missing q049
        with pytest.raises(ContractError, match="SAME planned samples"):
            build_calibration_plan(
                left + right, created_at="2026-09-18T00:00:00Z", strict=True
            )

    def test_strict_requires_all_six_types(self):
        # 60 questions per side minus the 10 preference ones: a full
        # 100-pair population that misses ONE question type.
        population = [
            make_candidate(c, i)
            for c in ("A", "B")
            for i in range(60)
            if TYPES[i % 6] != "single-session-preference"
        ]
        assert len(population) == 100
        with pytest.raises(ContractError, match="six question types"):
            build_calibration_plan(
                population, created_at="2026-09-18T00:00:00Z", strict=True
            )


class TestWorksheetDeidentification:
    """The worksheet is the annotator's ONLY view; it must be blind."""

    def test_columns_are_exactly_the_allowed_set(self):
        assert WORKSHEET_COLUMNS == (
            "item_id",
            "question",
            "gold_answer",
            "response",
            "question_type",
            "is_abstention",
            "annotation",
            "note",
            "annotated_at",
        )

    def test_worksheet_leaks_nothing(self):
        plan = build_plan()
        text = render_worksheet_csv(worksheet_rows(plan, "random"))
        # no condition labels, no run ids, no sample handles, no judge data
        for leak in ("run-A", "run-B", "condition", "sample_handle", "verdict", "raw_output"):
            assert leak not in text, leak
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        assert len(rows) == 100
        # neutral sequential ids assigned after the shuffle
        ids = [r["item_id"] for r in rows]
        assert ids == [f"c{i:04d}" for i in range(1, 101)]
        # presentation order is NOT the (condition, handle) order
        assert [r["question"] for r in rows] != sorted(
            r["question"] for r in rows
        )

    def test_cohort_worksheets_are_disjoint_in_item_ids(self):
        plan = build_plan()
        random_ids = {r.item_id for r in worksheet_rows(plan, "random")}
        boundary_ids = {r.item_id for r in worksheet_rows(plan, "boundary")}
        assert random_ids & boundary_ids == set()

    def test_self_consistency_worksheet_is_the_planned_subset(self):
        plan = build_plan()
        rows = worksheet_rows(plan, "self_consistency")
        assert {r.item_id for r in rows} == set(plan.self_consistency_item_ids)


class TestAnnotationImport:
    def _filled(self, plan, label="yes"):
        text = render_worksheet_csv(worksheet_rows(plan, "random"))
        rows = list(csv.DictReader(io.StringIO(text)))
        fields = list(rows[0].keys())
        for row in rows:
            row["annotation"] = label
            row["annotated_at"] = "2026-09-19"
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return buf.getvalue()

    def test_round_trip(self):
        plan = build_plan()
        records = import_annotations(
            self._filled(plan),
            expected_item_ids=[
                i.item_id for i in plan.items if i.cohort == "random"
            ],
        )
        assert len(records) == 100
        assert all(r.annotation == "yes" for r in records)

    def test_rejects_identifying_extra_columns(self):
        plan = build_plan()
        text = self._filled(plan)
        text = text.replace(
            "item_id,", "condition,item_id", 1
        ).replace(
            "c0001,", "A,c0001", 1
        )
        with pytest.raises(ContractError, match="columns"):
            import_annotations(
                text,
                expected_item_ids=[
                    i.item_id for i in plan.items if i.cohort == "random"
                ],
            )

    def test_rejects_unknown_and_missing_items(self):
        plan = build_plan()
        random_ids = [i.item_id for i in plan.items if i.cohort == "random"]
        text = self._filled(plan)
        # unknown item id
        with pytest.raises(ContractError, match="not part of this cohort"):
            import_annotations(
                text.replace("c0001,", "c9999,"), expected_item_ids=random_ids
            )
        # missing rows
        with pytest.raises(ContractError, match="missing"):
            import_annotations(
                "\n".join(text.splitlines()[:-1]), expected_item_ids=random_ids
            )

    def test_rejects_empty_and_bad_labels(self):
        plan = build_plan()
        random_ids = [i.item_id for i in plan.items if i.cohort == "random"]
        with pytest.raises(ContractError, match="empty annotation"):
            import_annotations(
                render_worksheet_csv(worksheet_rows(plan, "random")),
                expected_item_ids=random_ids,
            )
        with pytest.raises(ContractError, match="must be one of"):
            import_annotations(
                self._filled(plan, label="maybe"), expected_item_ids=random_ids
            )


class TestJudgeInputDiscipline:
    """AC1: the judge sees ONLY question + gold + response."""

    def test_calibration_request_shape_matches_the_scorer(self):
        from eval.calibration.judge_batch import _build_request

        item = build_plan().items[0]
        request = _build_request(item)
        assert_judge_input_discipline(request)
        assert request.protocol_fields == {"abstention": item.is_abstention}
        # the rendered official prompt exposes exactly the three texts
        prompt = render_official_prompt(request)
        assert item.question in prompt
        assert item.expected_answer in prompt
        assert item.hypothesis in prompt

    def test_forbidden_protocol_fields_are_rejected(self):
        request = JudgeRequest(
            question="q",
            expected_answer="g",
            hypothesis="h",
            question_type="multi-session",
            protocol_id=OFFICIAL_PROTOCOL_ID,
            protocol_fields={
                "abstention": False,
                "retrieved_evidence": ["session_1 evidence"],
            },
        )
        with pytest.raises(ContractError, match="non-protocol keys"):
            assert_judge_input_discipline(request)

    def test_self_made_protocol_is_rejected(self):
        request = JudgeRequest(
            question="q",
            expected_answer="g",
            hypothesis="h",
            question_type="multi-session",
            protocol_id="my-own-prompt@1",
            protocol_fields={},
        )
        with pytest.raises(ContractError, match="self-made prompts"):
            assert_judge_input_discipline(request)

    def test_worksheet_never_carries_evidence_content(self):
        # The calibration population is built from judge records; the
        # worksheet projection drops everything but the five fields.
        plan = build_plan()
        text = render_worksheet_csv(worksheet_rows(plan, "random"))
        for leak in ("evidence", "prepared", "memory", "retrieval"):
            assert leak not in text.lower()


class _SleepRecorder:
    def __init__(self) -> None:
        self.slept: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.slept.append(seconds)


class TestJudgeBatch:
    def test_batch_covers_every_planned_item_with_audit_rows(self, monkeypatch):
        monkeypatch.setenv("ZAI_API_KEY", "offline-test")
        plan = build_plan()
        judge = FakeJudge()
        artifact = run_judge_batch(
            plan,
            judge,
            api_key_env="ZAI_API_KEY",
            sleep=_SleepRecorder(),
        )
        assert len(artifact.calls) == 120  # 100 random + 20 boundary
        assert all(c.verdict is not None for c in artifact.calls)
        assert artifact.observed_response_models == ("glm-5.3",)
        assert artifact.judge_temperature == 0.0

    def test_missing_credential_fails_fast_before_any_call(self, monkeypatch):
        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        plan = build_plan()
        judge = FakeJudge()
        with pytest.raises(ContractError, match="ZAI_API_KEY"):
            run_judge_batch(plan, judge, api_key_env="ZAI_API_KEY")
        assert judge.requests == []

    def test_transient_errors_retry_with_backoff(self, monkeypatch):
        monkeypatch.setenv("ZAI_API_KEY", "offline-test")
        from eval.judges.base import JudgeProtocolError

        plan = build_calibration_plan(
            make_population(),
            created_at="2026-09-18T00:00:00Z",
            strict=False,
            random_size=2,
            boundary_size=0,
        )
        sleep = _SleepRecorder()

        class FlakyJudge(FakeJudge):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def evaluate(self, request):
                self.calls += 1
                if self.calls == 1:
                    raise JudgeProtocolError(
                        "judge_http_503", "boom", transient=True
                    )
                return super().evaluate(request)

        artifact = run_judge_batch(
            plan, FlakyJudge(), api_key_env="ZAI_API_KEY", sleep=sleep
        )
        assert sleep.slept == [1.0]
        assert artifact.calls[0].attempts == 2
        assert artifact.calls[0].verdict is not None

    def test_unparseable_output_is_a_recorded_failure_never_a_no(self, monkeypatch):
        """AC4: unparseable output fails the stage; it is NOT judged wrong."""
        monkeypatch.setenv("ZAI_API_KEY", "offline-test")
        from eval.judges.base import JudgeProtocolError

        plan = build_calibration_plan(
            make_population(),
            created_at="2026-09-18T00:00:00Z",
            strict=False,
            random_size=1,
            boundary_size=0,
        )

        class EmptyJudge(FakeJudge):
            def evaluate(self, request):
                raise JudgeProtocolError(
                    "judge_empty_content",
                    "empty content",
                    transient=False,
                )

        artifact = run_judge_batch(
            plan, EmptyJudge(), api_key_env="ZAI_API_KEY", sleep=_SleepRecorder()
        )
        call = artifact.calls[0]
        assert call.parse_failed is True
        assert call.verdict is None
        assert call.error is not None


class TestStatistics:
    def _calls(self, plan, verdicts: dict[str, bool]):
        calls = []
        for item in plan.items:
            verdict = verdicts.get(item.item_id, True)
            calls.append(
                CalibrationJudgeCall(
                    item_id=item.item_id,
                    request=JudgeRequest(
                        question=item.question,
                        expected_answer=item.expected_answer,
                        hypothesis=item.hypothesis,
                        question_type=item.question_type,
                        protocol_id=OFFICIAL_PROTOCOL_ID,
                        protocol_fields={"abstention": item.is_abstention},
                    ),
                    verdict=verdict,
                    raw_output="Yes." if verdict else "No.",
                    response_model="glm-5.3",
                    usage=ResourceUsage(input_tokens=1, output_tokens=1, llm_call_count=1),
                    started_at="2026-09-18T00:00:00Z",
                )
            )
        return calls

    def _annotations(self, plan, labels: dict[str, str]):
        from eval.calibration.sampling import AnnotationRecord

        return [
            AnnotationRecord(
                item_id=i.item_id,
                annotation=labels.get(i.item_id, "yes"),
                annotated_at="2026-09-19",
            )
            for i in plan.items
            if i.cohort == "random"
        ]

    def test_perfect_agreement_full_statistics(self):
        from eval.calibration.stats import compute_statistics

        plan = build_plan()
        stats = compute_statistics(
            items=list(plan.items),
            annotations=self._annotations(plan, {}),
            judge_calls=self._calls(plan, {}),
        )
        assert stats.overall.n == 100
        assert stats.overall.agreement == 1.0
        assert stats.undecided_count == 0
        assert stats.judge_parse_failures == 0
        assert stats.confusion.judge_lenient == 0
        assert stats.confusion.judge_strict == 0
        # every question type present
        assert set(stats.per_question_type) == set(TYPES)
        # abstention subset is the abstention items of the random cohort
        abst_n = sum(1 for i in plan.items if i.cohort == "random" and i.is_abstention)
        assert stats.abstention_subset.n == abst_n

    def test_undecided_excluded_from_denominator_but_counted(self):
        from eval.calibration.stats import compute_statistics

        plan = build_plan()
        random_ids = [i.item_id for i in plan.items if i.cohort == "random"]
        labels = {random_ids[0]: "cannot_judge", random_ids[1]: "cannot_judge"}
        stats = compute_statistics(
            items=list(plan.items),
            annotations=self._annotations(plan, labels),
            judge_calls=self._calls(plan, {}),
        )
        assert stats.undecided_count == 2
        assert stats.undecided_ratio == pytest.approx(0.02)
        assert stats.overall.n == 98

    def test_parse_failures_never_count_as_disagreement_or_no(self):
        from eval.calibration.stats import compute_statistics

        plan = build_plan()
        calls = list(self._calls(plan, {}))
        broken = calls[0]
        calls[0] = CalibrationJudgeCall(
            item_id=broken.item_id,
            request=broken.request,
            parse_failed=True,
            error="judge_empty_content: empty",
            started_at="2026-09-18T00:00:00Z",
        )
        stats = compute_statistics(
            items=list(plan.items),
            annotations=self._annotations(plan, {}),
            judge_calls=calls,
        )
        assert stats.judge_parse_failures == 1
        assert stats.overall.n == 99
        assert stats.overall.agreement == 1.0  # not a fabricated 'no'

    def test_confusion_matrix_lenient_and_strict(self):
        from eval.calibration.stats import compute_statistics

        plan = build_plan()
        random_ids = [i.item_id for i in plan.items if i.cohort == "random"]
        # human says no on two items where judge says yes -> lenient
        # human says yes on one item where judge says no -> strict
        labels = {random_ids[0]: "no", random_ids[1]: "no", random_ids[2]: "yes"}
        verdicts = {random_ids[2]: False}
        stats = compute_statistics(
            items=list(plan.items),
            annotations=self._annotations(plan, labels),
            judge_calls=self._calls(plan, verdicts),
        )
        assert stats.confusion.judge_lenient == 2
        assert stats.confusion.judge_strict == 1
        assert stats.overall.agreement == pytest.approx(97 / 100)

    def test_cross_condition_difference_and_self_consistency(self):
        from eval.calibration.sampling import AnnotationRecord
        from eval.calibration.stats import compute_statistics

        plan = build_plan()
        random_items = [i for i in plan.items if i.cohort == "random"]
        # judge disagrees on every condition-B item
        verdicts = {
            i.item_id: False
            for i in random_items
            if i.condition == "B"
        }
        second_round = [
            AnnotationRecord(
                item_id=item_id,
                annotation="yes",
                annotated_at="2026-09-20",
                round="self_consistency",
            )
            for item_id in plan.self_consistency_item_ids
        ]
        stats = compute_statistics(
            items=list(plan.items),
            annotations=self._annotations(plan, {}),
            judge_calls=self._calls(plan, verdicts),
            self_consistency_annotations=second_round,
        )
        assert stats.per_condition["A"].agreement == 1.0
        assert stats.per_condition["B"].agreement == 0.0
        assert stats.cross_condition_diff == 1.0
        assert stats.self_consistency.agreement == 1.0

    def test_boundary_annotations_are_diagnostics_only(self):
        from eval.calibration.sampling import AnnotationRecord
        from eval.calibration.stats import compute_statistics

        plan = build_plan()
        boundary_items = [i for i in plan.items if i.cohort == "boundary"]
        annotations = self._annotations(plan, {}) + [
            AnnotationRecord(item_id=i.item_id, annotation="no", annotated_at="2026-09-19")
            for i in boundary_items
        ]
        # judge says yes everywhere -> boundary all lenient
        stats = compute_statistics(
            items=list(plan.items),
            annotations=annotations,
            judge_calls=self._calls(plan, {}),
        )
        assert stats.boundary_subset is not None
        assert stats.boundary_subset.agreement == 0.0
        assert stats.boundary_confusion.judge_lenient == len(boundary_items)
        # the decision gates never read the boundary subset
        decision = decide_calibration(stats)
        assert all(g.name != "boundary" for g in decision.gates)

    def test_unpaired_inputs_raise_at_decision_time(self):
        from eval.calibration.stats import compute_statistics

        plan = build_plan()
        random_ids = {i.item_id for i in plan.items if i.cohort == "random"}
        dropped = sorted(random_ids)[:3]
        calls = [c for c in self._calls(plan, {}) if c.item_id not in set(dropped)]
        stats = compute_statistics(
            items=list(plan.items),
            annotations=self._annotations(plan, {}),
            judge_calls=calls,
        )
        assert stats.unpaired_items
        with pytest.raises(ContractError, match="unpaired"):
            decide_calibration(stats)


def _subset(n: int, agreement: float) -> SubsetAgreement:
    lo, hi = wilson_interval(round(agreement * n), n)
    return SubsetAgreement(n=n, agreement=agreement, wilson_low=lo, wilson_high=hi)


def make_stats(
    *,
    overall_n=100,
    overall_agreement=0.87,
    abst_n=10,
    abst_agreement=1.0,
    per_type=None,
    cross_diff=0.02,
    self_n=20,
    self_agreement=0.95,
    undecided_ratio=0.02,
):
    pairs = round(overall_agreement * overall_n)
    return CalibrationStatistics(
        overall=_subset(overall_n, overall_agreement),
        confusion=ConfusionCounts(
            judge_yes_human_yes=pairs - 2,
            judge_lenient=1,
            judge_strict=1,
            judge_no_human_no=overall_n - pairs,
        ),
        per_question_type={
            qtype: _subset(n, agreement)
            for qtype, (n, agreement) in (per_type or {
                t: (16, 0.9) for t in TYPES
            }).items()
        },
        abstention_subset=_subset(abst_n, abst_agreement),
        per_condition={"A": _subset(50, 0.88), "B": _subset(50, 0.86)},
        cross_condition_diff=cross_diff,
        self_consistency=_subset(self_n, self_agreement) if self_n else None,
        undecided_count=round(undecided_ratio * overall_n),
        undecided_ratio=undecided_ratio,
        judge_parse_failures=0,
        unpaired_items=(),
    )


class TestDecision:
    def test_pass_case(self):
        decision = decide_calibration(make_stats())
        assert decision.verdict == "passed"
        assert decision.effective_overall_threshold == 0.85
        assert not decision.threshold_adjusted_for_self_consistency

    def test_point_estimate_below_threshold_fails(self):
        decision = decide_calibration(make_stats(overall_agreement=0.84))
        assert decision.verdict == "failed"

    def test_point_estimate_without_interval_is_never_enough(self):
        stats = make_stats()
        # strip the interval: same point estimate, no lower bound
        stripped = stats.model_copy(
            update={
                "overall": SubsetAgreement(
                    n=stats.overall.n, agreement=stats.overall.agreement
                )
            }
        )
        decision = decide_calibration(stripped)
        assert decision.verdict == "failed"
        assert any(
            g.name == "overall_agreement" and not g.passed for g in decision.gates
        )

    def test_wilson_lower_bound_gate(self):
        # 0.86 with n=100: low bound 0.783... wait compute: agreement above
        # 0.85 but a LOW lower bound requires a small n; n=40, 0.875 -> low
        decision = decide_calibration(make_stats(overall_n=40, overall_agreement=0.875))
        lo = wilson_interval(35, 40)[0]
        if lo < 0.75:
            assert decision.verdict == "failed"
        else:  # pragma: no cover - guard against reference drift
            assert decision.verdict == "passed"

    def test_undecided_over_10pct_is_inconclusive(self):
        decision = decide_calibration(make_stats(undecided_ratio=0.12))
        assert decision.verdict == "inconclusive"
        assert any("不确定" in n for n in decision.notes)

    def test_low_type_marks_only_that_type_unavailable(self):
        per_type = {t: (16, 0.9) for t in TYPES}
        per_type["temporal-reasoning"] = (10, 0.6)
        decision = decide_calibration(make_stats(per_type=per_type))
        assert decision.verdict == "passed"  # overall still fine
        assert decision.unavailable_question_types == ("temporal-reasoning",)

    def test_abstention_gate_marks_subset_unavailable_not_failed(self):
        decision = decide_calibration(make_stats(abst_agreement=0.6))
        assert decision.verdict == "passed"
        assert decision.abstention_subset_unavailable is True

    def test_cross_condition_flag_is_a_note_not_a_failure(self):
        decision = decide_calibration(make_stats(cross_diff=0.12))
        assert decision.verdict == "passed"
        assert decision.cross_condition_affected is True
        assert any("差异化误差" in n for n in decision.notes)

    def test_low_self_consistency_lowers_the_overall_threshold(self):
        stats = make_stats(self_agreement=0.80)
        decision = decide_calibration(stats)
        assert decision.threshold_adjusted_for_self_consistency is True
        assert decision.effective_overall_threshold == pytest.approx(0.80)
        assert decision.self_consistency_cap == pytest.approx(0.80)
        assert any("下调" in n for n in decision.notes)

    def test_no_self_consistency_keeps_threshold(self):
        decision = decide_calibration(make_stats(self_n=0))
        assert not decision.threshold_adjusted_for_self_consistency


class TestRecord:
    def _record(self, stats=None, decision=None):
        from eval.calibration.stats import compute_statistics

        plan = build_plan()
        if stats is None:
            calls = CalibrationJudgeCallsArtifact(
                created_at="2026-09-18T01:00:00Z",
                judge_alias="glm-5.3",
                judge_base_url="https://api.z.ai/api/paas/v4",
                judge_temperature=0.0,
                judge_max_output_tokens=10,
                observed_response_models=("glm-5.3",),
                calls=(
                    CalibrationJudgeCall(
                        item_id=i.item_id,
                        request=JudgeRequest(
                            question=i.question,
                            expected_answer=i.expected_answer,
                            hypothesis=i.hypothesis,
                            question_type=i.question_type,
                            protocol_id=OFFICIAL_PROTOCOL_ID,
                            protocol_fields={"abstention": i.is_abstention},
                        ),
                        verdict=True,
                        raw_output="Yes.",
                        response_model="glm-5.3",
                        started_at="2026-09-18T01:00:00Z",
                    )
                    for i in plan.items
                ),
            )
            annotations = [
                {
                    "item_id": i.item_id,
                    "annotation": "yes",
                    "note": "",
                    "annotated_at": "2026-09-18",
                }
                for i in plan.items
                if i.cohort == "random"
            ]
            from eval.calibration.sampling import AnnotationRecord

            annotation_records = [
                AnnotationRecord(
                    item_id=i.item_id,
                    annotation="yes",
                    annotated_at="2026-09-18",
                )
                for i in plan.items
                if i.cohort == "random"
            ]
            stats = compute_statistics(
                items=list(plan.items),
                annotations=annotation_records,
                judge_calls=calls.calls,
            )
            decision = decide_calibration(stats)
            return (
                build_calibration_record(
                    plan=plan,
                    annotations=annotation_records,
                    self_consistency_annotations=[],
                    judge_calls=calls,
                    statistics=stats,
                    decision=decision,
                    judge_vendor_documented_version="GLM-5.3",
                    judge_vendor_documented_on="2026-09-18",
                    created_at="2026-09-20T10:00:00Z",
                ),
                plan,
                annotation_records,
                calls,
            )
        raise NotImplementedError

    def test_record_carries_model_identity_and_run_date(self):
        record, *_ = self._record()
        assert record.judge.alias == "glm-5.3"
        assert record.judge.response_models == ("glm-5.3",)
        assert record.judge.vendor_documented_version == "GLM-5.3"
        assert record.judge.calibration_run_date == "2026-09-20"
        assert record.criteria_digest == criteria_digest()
        assert "GPT-4o" in record.official_model_deviation["note"]

    def test_same_criteria_and_inputs_yield_stable_decision(self):
        from eval.calibration.stats import compute_statistics, decide_calibration

        record, plan, annotations, calls = self._record()
        stats2 = compute_statistics(
            items=list(plan.items),
            annotations=annotations,
            judge_calls=calls.calls,
        )
        decision2 = decide_calibration(stats2)
        assert verify_decision_stability(
            record, statistics=stats2, decision=decision2
        )

    def test_changed_inputs_raise_instability(self):
        from eval.calibration.stats import compute_statistics, decide_calibration

        record, plan, annotations, calls = self._record()
        flipped = [
            a.model_copy(update={"annotation": "no"})
            if a.item_id == annotations[0].item_id
            else a
            for a in annotations
        ]
        stats2 = compute_statistics(
            items=list(plan.items),
            annotations=flipped,
            judge_calls=calls.calls,
        )
        decision2 = decide_calibration(stats2)
        with pytest.raises(ContractError, match="instability"):
            verify_decision_stability(record, statistics=stats2, decision=decision2)

    def test_record_round_trips_through_json(self):
        record, *_ = self._record()
        loaded = CalibrationRecordArtifact.load_json(record.model_dump_json())
        assert loaded == record


class TestBinding:
    def _plan(self, **overrides):
        from eval.config import JudgePlan

        fields = {
            "model": "glm-5.3",
            "model_family": "glm",
            "base_url": "https://api.z.ai/api/paas/v4",
            "temperature": 0.0,
            "protocol_id": OFFICIAL_PROTOCOL_ID,
            "protocol_source_commit": criteria_payload()["protocol_source_commit"],
            "api": "openai_chat",
            "api_key_env": "ZAI_API_KEY",
            "max_output_tokens": 10,
            "vendor_documented_version": "GLM-5.3",
            "vendor_documented_on": "2026-09-18",
        }
        fields.update(overrides)
        return JudgePlan(**fields)

    def _record_for(self, alias="glm-5.3", temperature=0.0, max_tokens=10):
        record, *_ = TestRecord()._record()
        return record.model_copy(
            update={
                "judge": record.judge.model_copy(
                    update={
                        "alias": alias,
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    }
                )
            }
        )

    def test_matching_plan_binds(self):
        record = self._record_for()
        bind_record_to_judge_plan(record, self._plan())  # no raise

    def test_alias_mismatch_requires_recalibration(self):
        record = self._record_for()
        with pytest.raises(ContractError, match="alias"):
            bind_record_to_judge_plan(record, self._plan(model="glm-4.7"))

    def test_temperature_mismatch_requires_recalibration(self):
        record = self._record_for()
        with pytest.raises(ContractError, match="temperature"):
            bind_record_to_judge_plan(record, self._plan(temperature=0.3))

    def test_max_tokens_mismatch_requires_recalibration(self):
        record = self._record_for()
        with pytest.raises(ContractError, match="max_output_tokens"):
            bind_record_to_judge_plan(record, self._plan(max_output_tokens=100))


class TestDegradationChannel:
    """AC3: failed configs degrade QA conclusions to diagnostics."""

    def test_offline_fake_is_not_applicable_and_stays_formal(self):
        doc = judge_calibration_doc("offline_fake", None)
        assert doc["status"] == "not_applicable"
        assert doc["qa_conclusions"] == "formal"
        assert not is_downgraded(doc)

    def test_real_judge_without_record_is_not_calibrated(self):
        doc = judge_calibration_doc("openai_chat", None)
        assert doc["status"] == "not_calibrated"
        assert is_downgraded(doc)
        assert "GPT-4o" not in doc["note"]  # deviation note lives with records

    def _record_doc(self, verdict: str) -> dict:
        record, *_ = TestRecord()._record()
        doc = json.loads(record.model_dump_json())
        doc["decision"]["verdict"] = verdict
        return doc

    @pytest.mark.parametrize("verdict", ["failed", "inconclusive"])
    def test_failed_and_inconclusive_downgrade_with_gpt4o_deviation(self, verdict):
        doc = judge_calibration_doc("openai_chat", self._record_doc(verdict))
        assert doc["status"] == verdict
        assert is_downgraded(doc)
        assert "GPT-4o" in doc["downgrade_note"]
        assert "诊断" in doc["downgrade_note"]

    def test_passed_record_keeps_conclusions_formal_with_limitation_note(self):
        doc = judge_calibration_doc("openai_chat", self._record_doc("passed"))
        assert doc["status"] == "passed"
        assert not is_downgraded(doc)
        assert "不宣称与论文分数可比" in doc["note"]
        assert doc["calibration_run_date"]
        assert doc["criteria_digest"] == criteria_digest()

    def test_qa_conclusion_metric_ids_are_registered(self):
        from eval.metrics import get_metric

        for metric_id in QA_CONCLUSION_METRIC_IDS:
            get_metric(metric_id)


class TestRunReportDegradation:
    """End to end over a REAL offline run directory."""

    @pytest.fixture()
    def offline_run(self, tmp_path: Path) -> Path:
        from eval.cli import main

        out = tmp_path / "runs"
        code = main(
            [
                "run",
                "--config", "eval/configs/examples/offline_fake.toml",
                "--out", str(out),
            ]
        )
        assert code == 0
        run_dirs = sorted(out.glob("run-*"))
        assert len(run_dirs) == 1
        return run_dirs[0]

    def _patched_copy(self, run_dir: Path, tmp_path: Path, api: str, record=None) -> Path:
        # The copy keeps the run DIRECTORY NAME (Reporter resolves the
        # samples through run_id); the patched copy lives under a new
        # parent so it never collides with the original.
        copy = tmp_path / f"copies-{api}-{record is not None}" / run_dir.name
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(run_dir, copy)
        config_doc = json.loads((copy / "config.json").read_text(encoding="utf-8"))
        config_doc["config"]["judge"]["api"] = api
        (copy / "config.json").write_text(
            json.dumps(config_doc, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if record is not None:
            (copy / "judge_calibration.json").write_text(
                json.dumps(record, ensure_ascii=False), encoding="utf-8"
            )
        return copy

    def test_offline_run_report_has_not_applicable_block(self, offline_run):
        from eval.report import Reporter

        report = Reporter(offline_run).build()
        block = report["judge_calibration"]
        assert block["status"] == "not_applicable"
        assert block["qa_conclusions"] == "formal"
        # no QA metric row is downgraded
        assert not any(
            m.get("downgraded_to_diagnostic") for m in report["metrics"]
        )

    def test_real_judge_run_without_record_downgrades(self, offline_run, tmp_path):
        from eval.report import Reporter

        copy = self._patched_copy(offline_run, tmp_path, "openai_chat", None)
        report = Reporter(copy).build()
        block = report["judge_calibration"]
        assert block["status"] == "not_calibrated"
        assert block["qa_conclusions"] == "downgraded_to_diagnostic"
        downgraded = {
            m["metric_id"]
            for m in report["metrics"]
            if m.get("downgraded_to_diagnostic")
        }
        assert downgraded == set(QA_CONCLUSION_METRIC_IDS)
        # retrieval metrics stay formal
        assert any(
            m["metric_id"] == "verifiable_session_recall_macro"
            and not m.get("downgraded_to_diagnostic")
            for m in report["metrics"]
        )
        assert any("降级" in note for note in report["limitations"])
        markdown = __import__("eval.report", fromlist=["render_markdown"]).render_markdown(report)
        assert "Judge 校准（诊断）" in markdown
        assert "已降级为诊断项" in markdown

    def test_failed_record_downgrades_with_deviation_note(self, offline_run, tmp_path):
        from eval.report import Reporter

        record_doc = TestDegradationChannel()._record_doc("failed")
        copy = self._patched_copy(offline_run, tmp_path, "openai_chat", record_doc)
        report = Reporter(copy).build()
        block = report["judge_calibration"]
        assert block["status"] == "failed"
        assert block["qa_conclusions"] == "downgraded_to_diagnostic"
        assert "GPT-4o" in block["downgrade_note"]
        assert any(
            m["metric_id"] == "planned_question_score"
            and m.get("downgraded_to_diagnostic")
            for m in report["metrics"]
        )

    def test_passed_record_keeps_formal(self, offline_run, tmp_path):
        from eval.report import Reporter

        record_doc = TestDegradationChannel()._record_doc("passed")
        copy = self._patched_copy(offline_run, tmp_path, "openai_chat", record_doc)
        report = Reporter(copy).build()
        block = report["judge_calibration"]
        assert block["status"] == "passed"
        assert block["qa_conclusions"] == "formal"
        assert not any(
            m.get("downgraded_to_diagnostic") for m in report["metrics"]
        )


class TestCompareDegradation:
    @staticmethod
    def _make_judge_real(run_dir: Path, record_doc: dict | None) -> None:
        """Rewrite a finished offline run as a REAL-judge run.

        The judge plan gets a complete openai_chat section and BOTH the
        config artifact and the manifest are re-fingerprinted, so the
        patched run stays internally consistent for the comparison
        machinery (fingerprints reproduce, both sides share one judge
        plan -> same condition holds)."""
        from eval.config import load_config_dict
        from eval.report import Reporter

        config_doc = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        judge = config_doc["config"]["judge"]
        from eval.calibration.criteria import criteria_payload

        judge.update(
            {
                "model": "glm-5.3",
                "model_family": "glm",
                "base_url": "https://api.z.ai/api/paas/v4",
                "temperature": 0.0,
                "protocol_id": OFFICIAL_PROTOCOL_ID,
                "protocol_source_commit": criteria_payload()["protocol_source_commit"],
                "api": "openai_chat",
                "api_key_env": "ZAI_API_KEY",
                "vendor_documented_version": "GLM-5.3",
                "vendor_documented_on": "2026-09-18",
                "max_output_tokens": 10,
                "request_timeout_s": 180.0,
            }
        )
        config = load_config_dict(config_doc["config"])
        config_doc["config"] = json.loads(config.model_dump_json())
        config_doc["config_fingerprint"] = config.fingerprint()
        (run_dir / "config.json").write_text(
            json.dumps(config_doc, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        manifest["config_fingerprint"] = config.fingerprint()
        (run_dir / "run.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if record_doc is not None:
            (run_dir / "judge_calibration.json").write_text(
                json.dumps(record_doc, ensure_ascii=False), encoding="utf-8"
            )
        report = Reporter(run_dir).build()
        (run_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def test_downgraded_side_moves_qa_metrics_to_diagnostics(self, tmp_path):
        from eval.cli import main
        from eval.compare import compare_runs

        out = tmp_path / "runs"
        for _ in range(2):
            code = main(
                [
                    "run",
                    "--config", "eval/configs/examples/offline_fake.toml",
                    "--out", str(out),
                ]
            )
            assert code == 0
        left_dir, right_dir = sorted(out.glob("run-*"))

        # Both sides share the SAME real judge plan (same condition);
        # only the RIGHT side carries a FAILED calibration record.
        record_doc = TestDegradationChannel()._record_doc("failed")
        self._make_judge_real(left_dir, None)
        self._make_judge_real(right_dir, record_doc)

        payload = compare_runs(str(left_dir), str(right_dir))
        assert payload["comparability"]["same_condition"] is True
        metrics = payload["metrics"]
        degraded = {
            d["metric_id"]
            for d in metrics["diagnostics"]
            if d["kind"] == "judge_calibration_downgraded"
        }
        assert degraded == set(QA_CONCLUSION_METRIC_IDS)
        aligned_ids = {row["metric_id"] for row in metrics["aligned"]}
        assert not aligned_ids & set(QA_CONCLUSION_METRIC_IDS)
        # retrieval metrics stay aligned
        assert "verifiable_session_recall_macro" in aligned_ids
        assert payload["right"]["judge_calibration_status"] == "failed"
        # real judge WITHOUT a record is itself degraded (not_calibrated)
        assert payload["left"]["judge_calibration_status"] == "not_calibrated"

    def test_both_formal_sides_keep_qa_metrics_aligned(self, tmp_path):
        from eval.cli import main
        from eval.compare import compare_runs

        out = tmp_path / "runs"
        for _ in range(2):
            code = main(
                [
                    "run",
                    "--config", "eval/configs/examples/offline_fake.toml",
                    "--out", str(out),
                ]
            )
            assert code == 0
        left_dir, right_dir = sorted(out.glob("run-*"))
        payload = compare_runs(str(left_dir), str(right_dir))
        aligned_ids = {row["metric_id"] for row in payload["metrics"]["aligned"]}
        assert "planned_question_score" in aligned_ids
        assert not any(
            d.get("kind") == "judge_calibration_downgraded"
            for d in payload["metrics"]["diagnostics"]
        )


class TestRunnerAttachment:
    def test_refuses_to_attach_to_offline_fake_config(self, tmp_path, monkeypatch):
        from eval.config import load_config_toml
        from eval.datasets import load_dataset_for_config
        from eval.memories import build_memory_for_plan
        from eval.readers import build_reader_for_plan
        from eval.judges import build_judge_for_plan
        from eval.runner import OfflineRunner
        from eval.runs import RunStore

        config = load_config_toml("eval/configs/examples/offline_fake.toml")
        record_doc = TestDegradationChannel()._record_doc("passed")
        store = RunStore(tmp_path / "runs", "run-attach-test")
        runner = OfflineRunner(
            config=config,
            dataset=load_dataset_for_config(config),
            adapter=build_memory_for_plan(config.memory),
            reader=build_reader_for_plan(config.reader),
            judge=build_judge_for_plan(config.judge),
            store=store,
            run_id="run-attach-test",
            judge_calibration=record_doc,
        )
        with pytest.raises(ContractError, match="only attach to a real judge"):
            runner._attach_judge_calibration()

    def test_refuses_mismatched_judge_binding(self, tmp_path, monkeypatch):
        from eval.config import load_config_toml
        from eval.datasets import load_dataset_for_config
        from eval.judges import build_judge_for_plan
        from eval.memories import build_memory_for_plan
        from eval.readers import build_reader_for_plan
        from eval.runner import OfflineRunner
        from eval.runs import RunStore

        config = load_config_toml("eval/configs/examples/real_smoke_live_bm25.toml")
        record_doc = TestDegradationChannel()._record_doc("passed")
        # a DIFFERENT judge alias than the config declares
        record_doc["judge"]["alias"] = "glm-4.7"
        store = RunStore(tmp_path / "runs", "run-bind-test")
        runner = OfflineRunner(
            config=config,
            dataset=load_dataset_for_config(config),
            adapter=build_memory_for_plan(config.memory),
            reader=build_reader_for_plan(config.reader),
            judge=build_judge_for_plan(config.judge),
            store=store,
            run_id="run-bind-test",
            judge_calibration=record_doc,
        )
        with pytest.raises(ContractError, match="calibration_binding_mismatch"):
            runner._attach_judge_calibration()


class TestRenderMarkdown:
    def test_render_contains_the_required_sections(self):
        from eval.calibration.render import render_calibration_markdown

        record, *_ = TestRecord()._record()
        text = render_calibration_markdown(record)
        assert "# Judge 校准报告" in text
        assert "阈值判定" in text
        assert "混淆矩阵" in text
        assert "Wilson" in text
        assert "GPT-4o" in text
        assert "校准运行日期" in text
