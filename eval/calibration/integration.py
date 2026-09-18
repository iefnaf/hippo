"""Judge-calibration integration with the report/compare diagnostic
channel (issue #9, acceptance criterion 3).

A run whose judge calibration did NOT pass must not carry formal QA
conclusions: the judge-dependent metrics (planned-question score,
scored accuracy, abstention accuracy and the four attribution cells)
are downgraded to explicitly marked DIAGNOSTIC items, the deviation
from the official GPT-4o judge is stated, and no comparability with
the paper's scores is claimed. Retrieval-side and coverage metrics
stay formal — they are program-computed and never touch the judge.

Degradation states:

- passed                 -> QA conclusions formal (subset exclusions
                            still apply per-type / abstention);
- failed / inconclusive  -> downgraded (calibration ran, did not pass);
- not_calibrated         -> downgraded (a REAL judge without any
                            calibration record attached);
- not_applicable         -> offline fake judge (M1 configs): no GPT-4o
                            claim exists, nothing to degrade.
"""

from __future__ import annotations

from typing import Any

#: Judge-dependent QA metrics (registry v1 ids). These are the ones a
#: failed calibration downgrades to diagnostics.
QA_CONCLUSION_METRIC_IDS = (
    "planned_question_score",
    "scored_accuracy",
    "abstention_accuracy",
    "attribution_hit_correct",
    "attribution_hit_wrong",
    "attribution_miss_correct",
    "attribution_miss_wrong",
)

#: Statuses whose QA conclusions are downgraded to diagnostics.
_DOWNGRADED_STATUSES = frozenset({"failed", "inconclusive", "not_calibrated"})

_DOWNGRADE_NOTE = (
    "judge 校准未达标：该侧 judge 依赖的问答结论（计划题整体得分、成功"
    "评分题准确率、拒答准确率、联合归因四格）已降级为诊断项，不进入正式"
    "比较与排名；judge 与官方验证过的 GPT-4o 不同家族，不宣称与论文分数"
    "可比。"
)


def judge_calibration_doc(judge_api: str, record_doc: dict[str, Any] | None) -> dict[str, Any]:
    """The run-header judge-calibration block (pure function).

    judge_api: the config's judge api ('openai_chat' | 'offline_fake').
    record_doc: the parsed calibration record JSON (run-header
    artifact) or None when the run carries none.
    """
    if record_doc is None:
        if judge_api == "openai_chat":
            return {
                "kind": "diagnostic",
                "status": "not_calibrated",
                "qa_conclusions": "downgraded_to_diagnostic",
                "judge_alias": None,
                "note": (
                    "真实 judge（openai_chat）运行未附带 judge 校准记录："
                    "问答结论按未校准处理，降级为诊断项；正式结论前须完成 "
                    "100+20 人工校准（run --judge-calibration 附上记录）。"
                ),
            }
        return {
            "kind": "diagnostic",
            "status": "not_applicable",
            "qa_conclusions": "formal",
            "judge_alias": None,
            "note": (
                "离线 fake judge（M1 配置）：校准只对真实 judge 有意义；"
                "fake 成绩本就不代表任何实现性能。"
            ),
        }

    decision = record_doc.get("decision", {})
    verdict = decision.get("verdict", "unknown")
    judge = record_doc.get("judge", {})
    statistics = record_doc.get("statistics", {})
    overall = statistics.get("overall", {})
    doc: dict[str, Any] = {
        "kind": "diagnostic",
        "status": verdict,
        "qa_conclusions": (
            "formal" if verdict == "passed" else "downgraded_to_diagnostic"
        ),
        "record_id": record_doc.get("record_id"),
        "criteria_digest": record_doc.get("criteria_digest"),
        "judge_alias": judge.get("alias"),
        "judge_response_models": list(judge.get("response_models", [])),
        "judge_vendor_documented_version": judge.get("vendor_documented_version"),
        "calibration_run_date": judge.get("calibration_run_date"),
        "overall_agreement": overall.get("agreement"),
        "overall_wilson_low": overall.get("wilson_low"),
        "overall_n": overall.get("n"),
        "undecided_ratio": statistics.get("undecided_ratio"),
        "self_consistency": (
            statistics.get("self_consistency", {}) or {}
        ).get("agreement"),
        "cross_condition_diff": statistics.get("cross_condition_diff"),
        "unavailable_question_types": list(
            decision.get("unavailable_question_types", [])
        ),
        "abstention_subset_unavailable": decision.get(
            "abstention_subset_unavailable"
        ),
        "cross_condition_affected": decision.get("cross_condition_affected"),
        "official_model_deviation": record_doc.get("official_model_deviation", {}),
        "notes": list(decision.get("notes", [])),
    }
    if verdict in _DOWNGRADED_STATUSES:
        doc["downgrade_note"] = _DOWNGRADE_NOTE
    elif verdict == "passed":
        doc["note"] = (
            "judge 校准通过：问答结论保持正式。单标注者只有自身一致性，"
            "没有评分者间一致性；judge 与官方 GPT-4o 不同家族，仍不宣称与"
            "论文分数可比。"
        )
    else:
        doc["qa_conclusions"] = "downgraded_to_diagnostic"
        doc["downgrade_note"] = (
            f"校准记录判定为未知状态 {verdict!r}：按降级处理。"
        )
    return doc


def is_downgraded(calibration_doc: dict[str, Any] | None) -> bool:
    """True when the doc's QA conclusions are downgraded to diagnostics."""
    if not isinstance(calibration_doc, dict):
        return False
    return calibration_doc.get("qa_conclusions") == "downgraded_to_diagnostic"


def downgrade_reason(calibration_doc: dict[str, Any] | None) -> str:
    """The note to attach to downgraded rows / diagnostics."""
    if isinstance(calibration_doc, dict) and calibration_doc.get("downgrade_note"):
        return str(calibration_doc["downgrade_note"])
    return _DOWNGRADE_NOTE
