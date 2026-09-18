"""Calibration statistics and the threshold decision (issue #9).

Everything here is a PURE function of (annotations, judge calls,
thresholds): the same inputs always produce the same statistics and
the same decision, which is what makes 「判据未变化时判定不被改动」
verifiable (record.verify_decision_stability).

Pairing semantics:

- an agreement pair is (judge_label, human_label) over the RANDOM
  cohort where BOTH sides are decidable: human cannot_judge labels are
  excluded from the agreement denominator and reported as the
  undecided ratio; judge calls whose output could not be parsed are
  scored-stage failures (issue #3/#5 semantics) — recorded and
  reported, NEVER counted as a judge 'no' or as a disagreement;
- boundary-cohort annotations never enter any agreement statistic
  (they only feed the lenient/strict diagnostic);
- per-condition agreement powers the cross-condition difference.

Threshold decision (user-confirmed values in criteria.THRESHOLDS):

- undecided ratio > 10%                       -> inconclusive
- overall agreement >= 0.85 AND Wilson lower
  >= 0.75 (point estimate alone is never enough) -> overall gate
- self-consistency < 0.85 lowers the effective overall threshold to
  the self-consistency value and the cap is written into the decision
- any question type < 0.70                    -> that type's QA
  conclusions are marked unavailable (calibration may still pass)
- the abstention subset uses the OVERALL gates; failing them marks
  the abstention subset unavailable
- cross-condition agreement difference > 0.05 -> the comparison is
  flagged as affected by judge differential error (a note, not a
  failure)
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from pydantic import Field, model_validator

from eval.calibration.criteria import THRESHOLDS, WILSON_Z
from eval.calibration.sampling import AnnotationRecord, CalibrationItem
from eval.contracts.common import ContractError, SchemaVersionedModel


def wilson_interval(
    successes: int, n: int, z: float = WILSON_Z
) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion.

    Returns None when n == 0 (no claim may be made from an empty
    denominator — reporting a point estimate without an interval never
    passes the thresholds).
    """
    if n <= 0:
        return None
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


class SubsetAgreement(SchemaVersionedModel):
    """Agreement over one subset (n is the paired, decidable count)."""

    n: int = Field(ge=0)
    agreement: float | None = None
    wilson_low: float | None = None
    wilson_high: float | None = None


class ConfusionCounts(SchemaVersionedModel):
    """2x2 human x judge confusion over paired decidable items.

    judge_lenient = judge said yes while the human said no;
    judge_strict = judge said no while the human said yes.
    """

    judge_yes_human_yes: int = Field(ge=0)
    judge_lenient: int = Field(ge=0)
    judge_strict: int = Field(ge=0)
    judge_no_human_no: int = Field(ge=0)

    @model_validator(mode="after")
    def _sum_matches(self):
        total = (
            self.judge_yes_human_yes
            + self.judge_lenient
            + self.judge_strict
            + self.judge_no_human_no
        )
        if total == 0:
            raise ValueError("the confusion matrix needs at least one pair")
        return self


class CalibrationStatistics(SchemaVersionedModel):
    """Everything the decision and the report need (pure data)."""

    overall: SubsetAgreement
    confusion: ConfusionCounts
    per_question_type: dict[str, SubsetAgreement]
    abstention_subset: SubsetAgreement
    per_condition: dict[str, SubsetAgreement]
    cross_condition_diff: float | None
    self_consistency: SubsetAgreement | None
    #: Boundary-cohort diagnostics (NEVER enter the decision gates; the
    #: design says boundary items only diagnose lenient/strict bias).
    boundary_subset: SubsetAgreement | None = None
    boundary_confusion: ConfusionCounts | None = None
    undecided_count: int = Field(ge=0)
    undecided_ratio: float | None
    judge_parse_failures: int = Field(ge=0)
    unpaired_items: tuple[str, ...] = ()


class ThresholdGate(SchemaVersionedModel):
    """One named gate with its inputs and whether it held."""

    name: str
    passed: bool
    observed: str
    required: str


class CalibrationDecision(SchemaVersionedModel):
    """The threshold verdict: passed / failed / inconclusive."""

    verdict: str = Field(min_length=1)  # passed | failed | inconclusive
    gates: tuple[ThresholdGate, ...]
    effective_overall_threshold: float
    threshold_adjusted_for_self_consistency: bool
    self_consistency_cap: float | None
    unavailable_question_types: tuple[str, ...]
    abstention_subset_unavailable: bool
    cross_condition_affected: bool
    notes: tuple[str, ...]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _label_of(annotation: str) -> str | None:
    """yes/no stay; cannot_judge (and anything else) is undecided."""
    if annotation in ("yes", "no"):
        return annotation
    return None


def _subset(pairs: Sequence[tuple[bool, str]]) -> SubsetAgreement:
    """Agreement + Wilson interval over (judge_correct, human_label)."""
    if not pairs:
        return SubsetAgreement(n=0, agreement=None)
    matches = sum(1 for judge_yes, human in pairs if (judge_yes and human == "yes") or (not judge_yes and human == "no"))
    interval = wilson_interval(matches, len(pairs))
    return SubsetAgreement(
        n=len(pairs),
        agreement=matches / len(pairs),
        wilson_low=interval[0] if interval else None,
        wilson_high=interval[1] if interval else None,
    )


def _confusion(pairs: Sequence[tuple[bool, str]]) -> ConfusionCounts:
    counts = {
        "judge_yes_human_yes": 0,
        "judge_lenient": 0,
        "judge_strict": 0,
        "judge_no_human_no": 0,
    }
    for judge_yes, human in pairs:
        if judge_yes and human == "yes":
            counts["judge_yes_human_yes"] += 1
        elif judge_yes:
            counts["judge_lenient"] += 1
        elif human == "yes":
            counts["judge_strict"] += 1
        else:
            counts["judge_no_human_no"] += 1
    return ConfusionCounts(**counts)


def compute_statistics(
    *,
    items: Sequence[CalibrationItem],
    annotations: Sequence[AnnotationRecord],
    judge_calls: Sequence[Any],
    self_consistency_annotations: Sequence[AnnotationRecord] = (),
) -> CalibrationStatistics:
    """Aggregate annotations vs judge calls into CalibrationStatistics.

    judge_calls are CalibrationJudgeCall objects (or equal dicts); only
    their item_id, verdict (bool | None) and parse status are read.
    Only the RANDOM cohort enters the agreement statistics.
    """
    first_by_item = {a.item_id: a for a in annotations}
    calls_by_item = {c.item_id: c for c in judge_calls}
    random_items = [i for i in items if i.cohort == "random"]
    boundary_items = [i for i in items if i.cohort == "boundary"]

    pairs: list[tuple[bool, str]] = []
    undecided = 0
    parse_failures = 0
    unpaired: list[str] = []
    type_pairs: dict[str, list[tuple[bool, str]]] = {}
    abst_pairs: list[tuple[bool, str]] = []
    cond_pairs: dict[str, list[tuple[bool, str]]] = {}
    for item in random_items:
        call = calls_by_item.get(item.item_id)
        annotation = first_by_item.get(item.item_id)
        if call is None or annotation is None:
            unpaired.append(item.item_id)
            continue
        verdict = getattr(call, "verdict", None)
        parse_failed = bool(getattr(call, "parse_failed", False))
        if parse_failed or verdict is None:
            # Unparseable judge output: a scored-stage failure, never a
            # 'no' and never a disagreement (issue #3/#5 semantics).
            parse_failures += 1
            continue
        human = _label_of(annotation.annotation)
        if human is None:
            undecided += 1
            continue
        pair = (verdict, human)
        pairs.append(pair)
        type_pairs.setdefault(item.question_type, []).append(pair)
        cond_pairs.setdefault(item.condition, []).append(pair)
        if item.is_abstention:
            abst_pairs.append(pair)

    conditions = sorted(cond_pairs)
    per_condition = {c: _subset(cond_pairs[c]) for c in conditions}
    if len(conditions) == 2:
        a, b = (per_condition[c].agreement for c in conditions)
        cross_diff = abs(a - b) if a is not None and b is not None else None
    else:
        cross_diff = None

    self_pairs: list[tuple[bool, str]] = []
    self_stats: SubsetAgreement | None = None
    if self_consistency_annotations:
        second_by_item = {a.item_id: a for a in self_consistency_annotations}
        for item_id, second in second_by_item.items():
            first = first_by_item.get(item_id)
            if first is None:
                unpaired.append(item_id)
                continue
            left = _label_of(first.annotation)
            right = _label_of(second.annotation)
            if left is None or right is None:
                continue  # undecided on either round: not comparable
            # Self-consistency compares HUMAN round 1 vs round 2.
            self_pairs.append((left == "yes", right))
        self_stats = _subset(self_pairs)

    # Boundary diagnostics: paired boundary annotations (when the
    # annotator labeled the boundary worksheet) never touch the gates.
    boundary_pairs: list[tuple[bool, str]] = []
    for item in boundary_items:
        call = calls_by_item.get(item.item_id)
        annotation = first_by_item.get(item.item_id)
        if call is None or annotation is None:
            continue
        verdict = getattr(call, "verdict", None)
        if verdict is None or getattr(call, "parse_failed", False):
            continue
        human = _label_of(annotation.annotation)
        if human is None:
            continue
        boundary_pairs.append((verdict, human))
    boundary_subset = _subset(boundary_pairs) if boundary_pairs else None
    boundary_confusion = _confusion(boundary_pairs) if boundary_pairs else None

    return CalibrationStatistics(
        overall=_subset(pairs),
        confusion=_confusion(pairs),
        per_question_type={t: _subset(ps) for t, ps in sorted(type_pairs.items())},
        abstention_subset=_subset(abst_pairs),
        per_condition=per_condition,
        cross_condition_diff=cross_diff,
        self_consistency=self_stats,
        boundary_subset=boundary_subset,
        boundary_confusion=boundary_confusion,
        undecided_count=undecided,
        undecided_ratio=(undecided / len(random_items)) if random_items else None,
        judge_parse_failures=parse_failures,
        unpaired_items=tuple(sorted(set(unpaired))),
    )


# ---------------------------------------------------------------------------
# Threshold decision
# ---------------------------------------------------------------------------


def decide_calibration(
    statistics: CalibrationStatistics,
    *,
    thresholds: dict[str, float] | None = None,
) -> CalibrationDecision:
    """Apply the fixed thresholds; a pure function of the statistics."""
    th = dict(THRESHOLDS if thresholds is None else thresholds)
    gates: list[ThresholdGate] = []
    notes: list[str] = []

    if statistics.unpaired_items:
        raise ContractError(
            code="calibration_unpaired",
            message=(
                "statistics carry unpaired items (annotation or judge call "
                f"missing): {list(statistics.unpaired_items)[:5]}… — fix the "
                "inputs before deciding"
            ),
            location="(statistics)",
        )

    # Effective overall threshold: a self-consistency below the minimum
    # lowers the bar and the cap is written out explicitly.
    effective = th["overall_agreement_min"]
    cap: float | None = None
    adjusted = False
    self_stats = statistics.self_consistency
    if self_stats is not None and self_stats.agreement is not None:
        gates.append(
            ThresholdGate(
                name="self_consistency",
                passed=self_stats.agreement >= th["self_consistency_min"],
                observed=f"自身一致率 {self_stats.agreement:.4f} (n={self_stats.n})",
                required=f">= {th['self_consistency_min']}",
            )
        )
        if self_stats.agreement < th["self_consistency_min"]:
            effective = min(effective, self_stats.agreement)
            cap = self_stats.agreement
            adjusted = True
            notes.append(
                f"自身一致率 {self_stats.agreement:.4f} 低于 "
                f"{th['self_consistency_min']}：总体阈值下调为 {effective:.4f}，"
                "一致率结论的可信上限即自身一致率。"
            )

    undecided_ratio = statistics.undecided_ratio
    undecided_ok = undecided_ratio is not None and undecided_ratio <= th["undecided_ratio_max"]
    gates.append(
        ThresholdGate(
            name="undecided_ratio",
            passed=bool(undecided_ok),
            observed=(
                f"无法判定比例 {undecided_ratio:.4f}"
                if undecided_ratio is not None
                else "无样本"
            ),
            required=f"<= {th['undecided_ratio_max']}",
        )
    )

    overall = statistics.overall
    if overall.agreement is None or overall.wilson_low is None:
        gates.append(
            ThresholdGate(
                name="overall_agreement",
                passed=False,
                observed="无配对样本",
                required=f">= {effective}（点估计）且 Wilson 95% 下界 >= "
                f"{th['overall_wilson_lower_min']}",
            )
        )
        overall_passed = False
    else:
        overall_passed = (
            overall.agreement >= effective
            and overall.wilson_low >= th["overall_wilson_lower_min"]
        )
        gates.append(
            ThresholdGate(
                name="overall_agreement",
                passed=overall_passed,
                observed=(
                    f"一致率 {overall.agreement:.4f}，Wilson 95% 下界 "
                    f"{overall.wilson_low:.4f} (n={overall.n})"
                ),
                required=f">= {effective}（点估计）且 Wilson 95% 下界 >= "
                f"{th['overall_wilson_lower_min']}",
            )
        )

    abst = statistics.abstention_subset
    abst_unavailable = False
    if abst.agreement is None:
        abst_unavailable = True
        notes.append("拒答子集没有可配对样本：拒答结论不可用。")
    else:
        abst_passed = (
            abst.agreement >= effective
            and (abst.wilson_low or 0.0) >= th["overall_wilson_lower_min"]
        )
        gates.append(
            ThresholdGate(
                name="abstention_subset",
                passed=abst_passed,
                observed=(
                    f"拒答子集一致率 {abst.agreement:.4f}，Wilson 95% 下界 "
                    f"{abst.wilson_low:.4f} (n={abst.n})"
                ),
                required=f"门槛同总体：>= {effective} 且下界 >= "
                f"{th['overall_wilson_lower_min']}",
            )
        )
        if not abst_passed:
            abst_unavailable = True
            notes.append("拒答子集未达总体门槛：拒答子集的问答结论标为不可用。")

    unavailable_types: list[str] = []
    for qtype, subset in statistics.per_question_type.items():
        if subset.agreement is None:
            unavailable_types.append(qtype)
            continue
        if subset.agreement < th["per_question_type_agreement_min"]:
            unavailable_types.append(qtype)
            notes.append(
                f"题型 {qtype} 一致率 {subset.agreement:.4f} (n={subset.n}) 低于 "
                f"{th['per_question_type_agreement_min']}：该题型结论标为不可用。"
            )

    cross_affected = statistics.cross_condition_diff is not None and (
        statistics.cross_condition_diff > th["cross_condition_max_diff"]
    )
    if statistics.cross_condition_diff is not None:
        gates.append(
            ThresholdGate(
                name="cross_condition_diff",
                passed=not cross_affected,
                observed=f"跨条件一致率差 {statistics.cross_condition_diff:.4f}",
                required=f"<= {th['cross_condition_max_diff']}",
            )
        )
        if cross_affected:
            notes.append(
                "跨条件一致率差超过 "
                f"{th['cross_condition_max_diff']}：两个实现条件的比较受 "
                "judge 差异化误差影响，须随结论标注。"
            )

    if not undecided_ok:
        verdict = "inconclusive"
        notes.append(
            "无法判定比例超过阈值：校准结论为不确定，须补充可判定标注后再判。"
        )
    elif not overall_passed:
        verdict = "failed"
    else:
        verdict = "passed"

    return CalibrationDecision(
        verdict=verdict,
        gates=tuple(gates),
        effective_overall_threshold=effective,
        threshold_adjusted_for_self_consistency=adjusted,
        self_consistency_cap=cap,
        unavailable_question_types=tuple(sorted(unavailable_types)),
        abstention_subset_unavailable=abst_unavailable,
        cross_condition_affected=cross_affected,
        notes=tuple(notes),
    )
