"""Fixed calibration criteria: rubric, thresholds and the criteria digest.

Judge calibration (issue #9, docs/design/eval-harness.md 「Judge 校准」)
must run BEFORE judge verdicts enter formal conclusions, because our
judge deliberately deviates from the official GPT-4o: it binds the
official LongMemEval anscheck protocol (prompt templates + yes/no
parsing semantics, upstream commit pinned in eval.judges.longmemeval)
but runs a different model family (anti self-preference). Scores are
therefore never claimed comparable with the paper.

This module fixes every judgment CRITERION as content, not just as
code behavior:

- the annotation RUBRIC the human annotator applies (single annotator,
  blind, random order, cannot_judge allowed);
- the numeric THRESHOLDS that turn statistics into a passed / failed /
  inconclusive decision (user-confirmed values from the design doc);
- the CRITERIA DIGEST — a sha256 over rubric content + thresholds +
  protocol binding + algorithm ids — stored inside every calibration
  record. A record's decision may only be trusted under an unchanged
  criteria digest; any edit to the rubric, thresholds or protocol
  binding produces a different digest and requires a NEW record (a
  stored decision is never edited in place).
"""

from __future__ import annotations

import hashlib
import json

from eval.judges.longmemeval import OFFICIAL_PROTOCOL_ID, UPSTREAM_PROTOCOL_COMMIT

#: Stable id of the annotation rubric; content changes need a new id.
RUBRIC_ID = "judge-calibration-rubric@1"

#: Stable id of the threshold set; value changes need a new id.
THRESHOLDS_ID = "judge-calibration-thresholds@1"

#: Selection / statistics algorithm binding (part of the criteria).
CALIBRATION_ALGORITHMS = {
    "sampling": "calibration-stratified-hashrank@1",
    "worksheet_order": "calibration-worksheet-hashrank@1",
    "statistics": "calibration-stats@1",
    "decision": "calibration-decision@1",
}

#: Wilson score interval z value for the 95% level (fixed).
WILSON_Z = 1.96

#: User-confirmed thresholds (docs/design/eval-harness.md, Judge 校准):
#: 总体一致率 >= 0.85 且 Wilson 95% 下界 >= 0.75；任一题型 < 0.70 时只把
#: 该题型的结论标为不可用；拒答子集门槛同总体；跨条件一致率差 <= 0.05；
#: 自身一致率 < 0.85 时下调总体阈值并写明上限；“无法判定”比例 > 10% 时
#: 校准结论标为不确定。只报点估计不报区间不算通过。
THRESHOLDS: dict[str, float] = {
    "overall_agreement_min": 0.85,
    "overall_wilson_lower_min": 0.75,
    "per_question_type_agreement_min": 0.70,
    "cross_condition_max_diff": 0.05,
    "self_consistency_min": 0.85,
    "undecided_ratio_max": 0.10,
}

#: Fixed sample sizes (user-confirmed): 100 random + 20 boundary; 20 of
#: the random items are re-annotated (>= 1 day apart) for the
#: self-consistency estimate.
RANDOM_SAMPLE_SIZE = 100
BOUNDARY_SAMPLE_SIZE = 20
SELF_CONSISTENCY_SIZE = 20

#: The official judge model our judge deviates from; recorded verbatim
#: with every calibration record and every downgraded conclusion.
OFFICIAL_JUDGE_MODEL = "GPT-4o"

#: Fixed rubric content (rendered verbatim into rubric.md; the sha256
#: of this exact text enters the criteria digest).
RUBRIC_MD = """# Judge 校准人工标注 rubric（judge-calibration-rubric@1）

## 任务

你会看到一道题（question）、标准答案（gold_answer）与一个模型的回答
（response）。请独立判断该回答是否正确，在 annotation 列填入 yes / no /
cannot_judge 三者之一。判断语义与官方 LongMemEval anscheck 协议一致：
你扮演的是“人工 judge”，用来校准自动 judge 是否可信。

## 标注纪律（盲标）

- 你不知道 judge 的结论，也不知道回答来自哪个实现条件；表格顺序已随机
  打乱，条目 ID（c0001…）不携带任何条件信息。
- 不参照其他条目、不回看自己此前的标注；每条独立判断。
- 拿不准时标 cannot_judge，不要猜；该比例会计入报告（超过 10% 会使校准
  结论变为“不确定”）。
- 首轮与间隔复标（自身一致性）分开进行：复标表在间隔至少一天后填写，
  且不查看首轮结果。

## 判定语义（与官方协议逐字对齐）

- **yes**：回答包含正确答案；回答等价于正确答案，或包含得出正确答案的
  全部中间步骤。
- **no**：回答只包含答案所需信息的**子集**（部分正确不算对），或答案
  错误、答非所问。
- **cannot_judge**：无法据题面与标准答案判断（如标准答案本身存疑、回答
  语言混乱到无法比较）。

分题型补充：

- **temporal-reasoning**：不因“差一天/差一周”的天数 off-by-one 判错
  （如答案 18 天、回答 19 天仍算正确）。
- **knowledge-update**：回答包含旧信息但给出了更新后的正确答案时，仍算
  正确；只有过时答案才算错。
- **single-session-preference**：gold_answer 是期望个性化回答的 rubric；
  回答不需要覆盖 rubric 的每一点，只要正确回忆并使用了用户的个人信息
  即算正确。
- **拒答题（is_abstention = true）**：yes = 回答正确识别了该题不可回答
  （可以说信息不完整、或给了其他信息但明确指出所问信息缺失）；no = 回答
  编造了答案。

## 输出格式

worksheet CSV 的 annotation 列：yes / no / cannot_judge（小写）；note 列
可写简短理由（可选）；annotated_at 列填标注日期（YYYY-MM-DD）。
"""

RUBRIC_SHA256 = hashlib.sha256(RUBRIC_MD.encode("utf-8")).hexdigest()


def criteria_payload() -> dict[str, object]:
    """The canonical criteria content a decision is bound to."""
    return {
        "rubric_id": RUBRIC_ID,
        "rubric_sha256": RUBRIC_SHA256,
        "thresholds_id": THRESHOLDS_ID,
        "thresholds": dict(THRESHOLDS),
        "wilson_z": WILSON_Z,
        "protocol_id": OFFICIAL_PROTOCOL_ID,
        "protocol_source_commit": UPSTREAM_PROTOCOL_COMMIT,
        "official_judge_model": OFFICIAL_JUDGE_MODEL,
        "sample_sizes": {
            "random": RANDOM_SAMPLE_SIZE,
            "boundary": BOUNDARY_SAMPLE_SIZE,
            "self_consistency": SELF_CONSISTENCY_SIZE,
        },
        "algorithms": dict(CALIBRATION_ALGORITHMS),
    }


def criteria_digest() -> str:
    """sha256 over the canonical criteria rendering.

    Stored in every calibration record: a decision computed under one
    digest is never edited or reused under another.
    """
    payload = json.dumps(
        criteria_payload(), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
