"""QA scorer: verifiable recall from actual evidence + judge verdicts.

The scorer is the only component that reads the dataset's private
ScoringData view (gold sessions, internal->official mapping, abstention
flag, category). It computes retrieval metrics PROGRAM-side from the
exact PreparedEvidence the reader consumed, then (when a reader answer
exists) asks the judge through the protocol adapter.

Recall semantics (from the design doc, 评分与指标):

- R is the set of sessions of RETAINED extractive evidence items -- the
  same prepared items the reader saw. Dropped units never count; a
  partially truncated unit still counts (its source was retained).
  Generated derivation sources never enter R.
- recall = |G intersect R| / |G| over internal session ids; gold ids and
  retained source ids must be covered by the private mapping (an
  invalid mapping is a data validation error, never a silent exclusion).
- Recall@k restricts R to the first k DISTINCT sessions in adapter
  return order.
- Abstention samples: every recall metric is N/A (gold is empty by
  definition) while the answer still goes to the judge.
- Adapters that do not declare extractive_evidence: recall metrics are
  N/A (evidence_mode); attribution "hit" then means non-empty retained
  evidence, the no-memory baseline is always a miss.

Failure rules: invalid scoring data raises ScoringDataError (the runner
marks the sample invalid_input); a judge protocol failure is returned in
the judge call record (qa_status=failed, recall metrics survive).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from eval.contracts.adapter import ErrorInfo, QueryContext, ResourceUsage
from eval.contracts.common import ContractError, now_utc
from eval.contracts.internal import (
    JudgeRequest,
    JudgeResult,
    MetricResult,
    PreparedEvidence,
    ReaderResult,
    ScoringData,
    ScoringTraceArtifact,
)
from eval.judges.base import Judge, JudgeProtocolError

#: Per-sample retrieval metric ids emitted by the scorer (all registered).
RECALL_METRIC_IDS = (
    "verifiable_session_recall_macro",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "budgeted_session_recall",
)
RECALL_KS = {"recall_at_1": 1, "recall_at_3": 3, "recall_at_5": 5}

ABSTENTION_NA_REASON = (
    "not_applicable: abstention samples never enter recall metrics "
    "(gold source set is empty by definition)"
)
EVIDENCE_MODE_NA_REASON = (
    "evidence_mode: the implementation does not declare "
    "extractive_evidence, so verifiable session recall is N/A"
)
#: Unranked controls (no-memory / full history) never enter ranking
#: retrieval metrics (issue #3 leftover; registry na_condition
#: 'ranking_baseline'). They still take part in auxiliary QA.
RANKING_BASELINE_NA_REASON = (
    "ranking_baseline: the {kind} control is unranked and never enters "
    "ranking retrieval metrics (registry condition 'ranking_baseline'); "
    "it still takes part in auxiliary QA"
)


class ScoringDataError(ContractError):
    """Private scoring data is invalid: a data validation error."""


def session_recall(gold: Sequence[str], retained_sessions: Sequence[str]) -> float:
    """|G intersect R| / |G|; both as session id collections."""
    if not gold:
        raise ValueError("session_recall requires a non-empty gold set")
    return len(set(gold) & set(retained_sessions)) / len(set(gold))


def session_recall_at(
    gold: Sequence[str], ordered_sessions: Sequence[str], k: int
) -> float:
    """Recall over the first k distinct sessions in return order."""
    first_k: list[str] = []
    seen: set[str] = set()
    for sid in ordered_sessions:
        if sid not in seen:
            seen.add(sid)
            first_k.append(sid)
            if len(first_k) == k:
                break
    return session_recall(gold, first_k)


def retained_extractive_sessions(prepared: PreparedEvidence | None) -> list[str]:
    """Distinct sessions of retained extractive items, in return order."""
    ordered: list[str] = []
    seen: set[str] = set()
    if prepared is not None:
        for item in prepared.items:
            span = item.evidence.extractive_span
            if item.evidence.kind == "extractive" and span is not None:
                if span.session_id not in seen:
                    seen.add(span.session_id)
                    ordered.append(span.session_id)
    return ordered


def attribution_cell(correct: bool, hit: bool) -> str:
    if hit:
        return "hit_correct" if correct else "hit_wrong"
    return "miss_correct" if correct else "miss_wrong"


@dataclass(frozen=True)
class JudgeCallRecord:
    """Everything the runner needs to audit one judge call attempt."""

    request: JudgeRequest
    result: JudgeResult | None
    error: ErrorInfo | None
    started_at: str
    ended_at: str
    elapsed_ms: float
    usage: ResourceUsage | None


@dataclass(frozen=True)
class SampleScoring:
    """Scorer output for one sample; metrics always registry ids.

    judge_attempts carries EVERY judge call attempt in order (bounded
    transient retries included); judge_call exposes the final one.
    """

    metrics: list[MetricResult]
    correct: bool | None
    attribution: str | None
    trace: ScoringTraceArtifact
    judge_attempts: tuple[JudgeCallRecord, ...] = ()

    @property
    def judge_call(self) -> JudgeCallRecord | None:
        return self.judge_attempts[-1] if self.judge_attempts else None


class QAScorer:
    """Scores one QA sample: recall (program) + verdict (judge)."""

    def __init__(
        self,
        *,
        dataset: Any,
        judge: Judge,
        extractive_declared: bool,
        baseline_kind: str,
        protocol_id: str,
        clock: Callable[[], str] = now_utc,
        monotonic: Callable[[], float] = time.perf_counter,
        max_retries: int = 0,
        backoff_base_s: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._dataset = dataset
        self._judge = judge
        self._extractive_declared = extractive_declared
        self._baseline_kind = baseline_kind
        self._protocol_id = protocol_id
        self._clock = clock
        self._monotonic = monotonic
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s
        self._sleep = sleep

    def _backoff_seconds(self, retry_index: int) -> float:
        """1s/4s/16s-shaped deterministic backoff (base * 4**k)."""
        return self._backoff_base_s * (4**retry_index)

    # -- private scoring view ----------------------------------------------

    def _scoring_data(self, handle: str) -> ScoringData:
        try:
            scoring = self._dataset.get_scoring_data(handle)
        except ContractError as exc:
            raise ScoringDataError(
                code="scoring_data_invalid",
                message=f"private scoring view for {handle!r} is invalid: "
                f"{exc.message}",
                location=f"/{handle}/scoring_data",
            ) from exc
        except Exception as exc:  # noqa: BLE001 - data errors are structural
            raise ScoringDataError(
                code="scoring_data_invalid",
                message=(
                    f"private scoring view for {handle!r} failed to load: "
                    f"{type(exc).__name__}: {exc}"
                ),
                location=f"/{handle}/scoring_data",
            ) from exc
        mapping = scoring.internal_to_official_session
        problems: list[str] = []
        if not scoring.is_abstention and not scoring.gold_source_ids:
            problems.append("non-abstention sample has empty gold_source_ids")
        missing_gold = [
            sid for sid in scoring.gold_source_ids if sid not in mapping
        ]
        if missing_gold:
            problems.append(
                f"gold sessions missing from the ID mapping: {missing_gold}"
            )
        if problems:
            raise ScoringDataError(
                code="scoring_data_invalid",
                message=(
                    f"private scoring view for {handle!r} is invalid: "
                    + "; ".join(problems)
                ),
                location=f"/{handle}/scoring_data",
            )
        return scoring

    # -- scoring -------------------------------------------------------------

    def score_sample(
        self,
        *,
        run_id: str,
        handle: str,
        question: QueryContext,
        prepared: PreparedEvidence | None,
        reader_result: ReaderResult | None,
    ) -> SampleScoring:
        scoring = self._scoring_data(handle)
        mapping = scoring.internal_to_official_session

        ordered_sessions = retained_extractive_sessions(prepared)
        unmapped = [sid for sid in ordered_sessions if sid not in mapping]
        if unmapped:
            raise ScoringDataError(
                code="scoring_data_invalid",
                message=(
                    f"retained extractive sessions absent from the ID "
                    f"mapping for {handle!r}: {unmapped}"
                ),
                location=f"/{handle}/scoring_data",
            )

        gold = list(scoring.gold_source_ids)
        # Applicable set E for ranking retrieval metrics (issue #3
        # leftover, consumed here): non-abstention, declared extractive,
        # AND a ranked implementation. The unranked controls (no-memory,
        # full history) never enter ranking recall even when they declare
        # extractive evidence — the registry carries the N/A condition
        # 'ranking_baseline' for exactly this case.
        unranked_baseline = self._baseline_kind in ("none", "full_history")
        applicable = (
            (not scoring.is_abstention)
            and self._extractive_declared
            and not unranked_baseline
        )
        if applicable:
            na_reason: str | None = None
            recall = session_recall(gold, ordered_sessions)
        else:
            if scoring.is_abstention:
                na_reason = ABSTENTION_NA_REASON
            elif unranked_baseline:
                na_reason = RANKING_BASELINE_NA_REASON.format(
                    kind=self._baseline_kind
                )
            else:
                na_reason = EVIDENCE_MODE_NA_REASON
            recall = None

        metrics: list[MetricResult] = []
        for metric_id in RECALL_METRIC_IDS:
            if applicable:
                if metric_id in RECALL_KS:
                    value = session_recall_at(
                        gold, ordered_sessions, RECALL_KS[metric_id]
                    )
                else:
                    assert recall is not None  # applicable implies non-empty gold
                    value = recall
                metrics.append(
                    MetricResult(
                        metric_id=metric_id, status="computed", value=value, reason=None
                    )
                )
            else:
                metrics.append(
                    MetricResult(
                        metric_id=metric_id,
                        status="not_applicable",
                        value=None,
                        reason=na_reason,
                    )
                )

        generated_units = [
            item
            for item in (prepared.items if prepared is not None else [])
            if item.evidence.kind == "generated"
        ]
        with_sources = [
            item for item in generated_units if item.evidence.derivation_sources
        ]
        if generated_units:
            metrics.append(
                MetricResult(
                    metric_id="derivation_source_coverage",
                    status="computed",
                    value=len(with_sources) / len(generated_units),
                    reason=None,
                )
            )
        else:
            metrics.append(
                MetricResult(
                    metric_id="derivation_source_coverage",
                    status="not_applicable",
                    value=None,
                    reason=(
                        "not_applicable: no retained generated evidence "
                        "units in this sample"
                    ),
                )
            )

        # Evidence-side hit for the 2x2 attribution (fixed per mode). The
        # full-history control has no VERIFIABLE gold comparison (its
        # "hit" is not retrieval quality): hit = non-empty evidence, like
        # other unranked controls (docs: 联合归因 命中口径).
        if self._baseline_kind == "none":
            evidence_mode = "none_baseline"
            hit_criterion = "never"
            hit = False
        elif self._baseline_kind == "full_history":
            evidence_mode = "full_history_control"
            hit_criterion = "nonempty_evidence"
            hit = bool(prepared is not None and prepared.items)
        elif self._extractive_declared:
            evidence_mode = "extractive_declared"
            hit_criterion = "gold_recall"
            # Abstention gold is empty: no gold hit is possible -> miss.
            hit = bool(applicable and recall is not None and recall > 0.0)
        else:
            evidence_mode = "generated_only"
            hit_criterion = "nonempty_evidence"
            hit = bool(prepared is not None and prepared.items)

        correct: bool | None = None
        attribution: str | None = None
        judge_attempts: tuple[JudgeCallRecord, ...] = ()
        if reader_result is not None:
            judge_attempts = self._judge_attempts(
                handle, scoring, question, reader_result
            )
            final = judge_attempts[-1]
            if final.result is not None:
                correct = final.result.correct
                attribution = attribution_cell(correct, hit)

        trace = ScoringTraceArtifact(
            run_id=run_id,
            sample_handle=handle,
            scoring=scoring,
            evidence_mode=evidence_mode,
            gold_internal_sessions=gold,
            actual_sessions_in_order=ordered_sessions,
            hit_gold_sessions=sorted(set(gold) & set(ordered_sessions)),
            recall_applicable=applicable,
            recall_na_reason=na_reason,
            recall_value=recall,
            hit_criterion=hit_criterion,
            hit=hit,
        )
        return SampleScoring(
            metrics=metrics,
            correct=correct,
            attribution=attribution,
            trace=trace,
            judge_attempts=judge_attempts,
        )

    def _judge_attempts(
        self,
        handle: str,
        scoring: ScoringData,
        question: QueryContext,
        reader_result: ReaderResult,
    ) -> tuple[JudgeCallRecord, ...]:
        """Bounded transient retries around the judge call.

        Only TRANSIENT errors retry (temporary API failures); an
        unparseable verdict is a protocol failure that fails the judge
        stage instead of being retried into a fabricated verdict, and a
        low score is never a retry reason.
        """
        records: list[JudgeCallRecord] = []
        retries = 0
        while True:
            record = self._single_judge_attempt(
                handle, scoring, question, reader_result
            )
            records.append(record)
            if record.error is None or not record.error.transient:
                break
            if retries >= self._max_retries:
                break
            self._sleep(self._backoff_seconds(retries))
            retries += 1
        return tuple(records)

    def _single_judge_attempt(
        self,
        handle: str,
        scoring: ScoringData,
        question: QueryContext,
        reader_result: ReaderResult,
    ) -> JudgeCallRecord:
        request = JudgeRequest(
            question=question.query,
            expected_answer=scoring.expected_answer,
            hypothesis=reader_result.hypothesis,
            question_type=scoring.question_type,
            protocol_id=self._protocol_id,
            # The abstention flag is a private scoring datum the official
            # protocol needs to select its unanswerable-question template
            # (M2); fake protocols ignore it. It rides in protocol_fields
            # (protocol-private data), never in the free-form prompt.
            protocol_fields={"abstention": scoring.is_abstention},
        )
        started_at = self._clock()
        t0 = self._monotonic()
        error: ErrorInfo | None = None
        result: JudgeResult | None = None
        try:
            result = self._judge.evaluate(request)
        except JudgeProtocolError as exc:
            error = exc.to_error_info()
        except Exception as exc:  # noqa: BLE001 - recorded as attempt error
            error = ErrorInfo(
                code="judge_exception",
                message=f"{type(exc).__name__}: {exc}",
                effect="none",
                transient=True,
            )
        elapsed_ms = round((self._monotonic() - t0) * 1000.0, 3)
        return JudgeCallRecord(
            request=request,
            result=result,
            error=error,
            started_at=started_at,
            ended_at=self._clock(),
            elapsed_ms=elapsed_ms,
            usage=result.usage if result is not None else None,
        )
