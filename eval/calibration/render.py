"""Human-readable rendering of a calibration record (calibration.md)."""

from __future__ import annotations

from typing import Any


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_calibration_markdown(record: Any) -> str:
    """Render the calibration record as the auditable markdown report."""
    stats = record.statistics
    decision = record.decision
    judge = record.judge
    lines: list[str] = []
    lines.append("# Judge 校准报告")
    lines.append("")
    lines.append(
        _table(
            ["字段", "值"],
            [
                ["记录 ID", record.record_id],
                ["生成时间", record.created_at],
                ["判据摘要 (criteria_digest)", record.criteria_digest[:16] + "…"],
                [
                    "judge（别名 / 响应 model）",
                    f"{judge.alias} / {', '.join(judge.response_models) or '未观测'}",
                ],
                [
                    "厂商文档版本 / 核对日期",
                    f"{judge.vendor_documented_version} / {judge.vendor_documented_on}",
                ],
                ["校准运行日期", judge.calibration_run_date],
                ["调用参数", f"temperature={judge.temperature}, max_tokens={judge.max_output_tokens}"],
                ["官方模型偏离", record.official_model_deviation.get("note", "")],
            ],
        )
    )
    lines.append("")
    lines.append("## 阈值判定")
    lines.append("")
    verdict_text = {
        "passed": "**达标（passed）**",
        "failed": "**不达标（failed）**——问答结论降级为诊断项",
        "inconclusive": "**不确定（inconclusive）**——问答结论降级为诊断项",
    }.get(decision.verdict, decision.verdict)
    lines.append(f"- 判定：{verdict_text}")
    lines.append(
        f"- 生效总体阈值：{decision.effective_overall_threshold:.2f}"
        + (
            "（因自身一致率低于 0.85 已下调，可信上限见下）"
            if decision.threshold_adjusted_for_self_consistency
            else ""
        )
    )
    if decision.self_consistency_cap is not None:
        lines.append(
            f"- 自身一致率上限：{decision.self_consistency_cap:.4f}"
        )
    lines.append("")
    lines.append(
        _table(
            ["门槛", "观测值", "要求", "通过"],
            [
                [
                    gate.name,
                    gate.observed,
                    gate.required,
                    "是" if gate.passed else "**否**",
                ]
                for gate in decision.gates
            ],
        )
    )
    lines.append("")
    if decision.unavailable_question_types:
        lines.append(
            "- 结论不可用题型：" + "、".join(decision.unavailable_question_types)
        )
    if decision.cross_condition_affected:
        lines.append("- 跨条件比较受 judge 差异化误差影响（差 > 0.05），随结论标注。")
    for note in decision.notes:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## 一致率统计（随机样本 100 条口径）")
    lines.append("")
    lines.append(
        _table(
            ["口径", "n", "一致率", "Wilson 95% 区间"],
            [
                [
                    "总体",
                    str(stats.overall.n),
                    _fmt(stats.overall.agreement),
                    f"[{_fmt(stats.overall.wilson_low)}, {_fmt(stats.overall.wilson_high)}]",
                ],
                *[
                    [
                        f"题型 {qtype}",
                        str(subset.n),
                        _fmt(subset.agreement),
                        f"[{_fmt(subset.wilson_low)}, {_fmt(subset.wilson_high)}]",
                    ]
                    for qtype, subset in stats.per_question_type.items()
                ],
                [
                    "拒答子集",
                    str(stats.abstention_subset.n),
                    _fmt(stats.abstention_subset.agreement),
                    f"[{_fmt(stats.abstention_subset.wilson_low)}, {_fmt(stats.abstention_subset.wilson_high)}]",
                ],
                *[
                    [
                        f"条件 {cond}",
                        str(subset.n),
                        _fmt(subset.agreement),
                        f"[{_fmt(subset.wilson_low)}, {_fmt(subset.wilson_high)}]",
                    ]
                    for cond, subset in stats.per_condition.items()
                ],
                *(
                    [
                        [
                            "自身一致率（人工复标）",
                            str(stats.self_consistency.n),
                            _fmt(stats.self_consistency.agreement),
                            f"[{_fmt(stats.self_consistency.wilson_low)}, {_fmt(stats.self_consistency.wilson_high)}]",
                        ]
                    ]
                    if stats.self_consistency is not None
                    else []
                ),
                *(
                    [
                        [
                            "边界样本（仅诊断，不入判定）",
                            str(stats.boundary_subset.n),
                            _fmt(stats.boundary_subset.agreement),
                            "—",
                        ]
                    ]
                    if stats.boundary_subset is not None
                    else []
                ),
            ],
        )
    )
    lines.append("")
    lines.append("## 混淆矩阵（judge × 人工，随机样本）")
    lines.append("")
    c = stats.confusion
    lines.append(
        _table(
            ["", "人工 yes", "人工 no"],
            [
                ["judge yes", str(c.judge_yes_human_yes), f"judge 偏宽 {c.judge_lenient}"],
                ["judge no", f"judge 偏严 {c.judge_strict}", str(c.judge_no_human_no)],
            ],
        )
    )
    lines.append("")
    lines.append(
        "## 其他计数（不入门槛）"
    )
    lines.append("")
    lines.append(
        _table(
            ["项", "值"],
            [
                ["无法判定（cannot_judge）数量 / 比例", f"{stats.undecided_count} / {_fmt(stats.undecided_ratio)}"],
                ["judge 不可解析输出（评分阶段失败，不默认判错）", str(stats.judge_parse_failures)],
                ["跨条件一致率差", _fmt(stats.cross_condition_diff)],
            ],
        )
    )
    lines.append("")
    lines.append("## 限制")
    lines.append("")
    lines.append(
        "- 单人标注：只有自身一致性，没有评分者间一致性；一致率结论的可信"
        "上限即自身一致率。"
    )
    lines.append(
        "- judge 与官方验证过的 GPT-4o 不同家族：即使校准通过，也不宣称与"
        "论文分数可比。"
    )
    lines.append(
        "- 校准记录绑定 judge 模型版本、prompt 版本与生成参数；任一项变化"
        "（含模型漂移）必须重新校准。"
    )
    lines.append("")
    return "\n".join(lines)
