# Judge 校准人工标注工作流（M2，issue #9）

状态：随 issue #9 落成的操作文档。判据（rubric、阈值、协议绑定）以代码内固定内容为准（`eval/calibration/criteria.py`）；本文档描述怎么用，不重复定义判据。设计依据见 [eval-harness.md](eval-harness.md)「Judge 校准」。

## 这件事在解决什么问题

本 harness 的 judge（`glm-5.3`，Z.ai）绑定官方 LongMemEval anscheck 协议，但与官方论文验证过的 GPT-4o 分属不同模型家族。judge 判定进入正式结论前，必须先由人工盲标校准：一致率与区间达标才允许问答结论保持正式；不达标则问答结论自动降级为诊断项并标注与官方模型的偏离，不宣称与论文分数可比。

人工盲标是**用户环节**：工具产出抽样清单、表格模板与导入方式，不代替人标注。

## 前置条件

1. 两个条件的 dev50 运行已完成（同一样本清单，例如 bm25 与 none）：

   ```bash
   export DEEPSEEK_API_KEY=... ZAI_API_KEY=...
   uv run hippo-eval run --config eval/configs/examples/real_dev50_bm25.toml --out runs
   uv run hippo-eval run --config eval/configs/examples/real_dev50_none.toml --out runs
   ```

2. 两次运行的全部样本都有 judge 记录（`plan` 会在缺失时拒绝执行；先 `resume` 补完）。

## 五步流程

### 1. 生成抽样清单与表格（离线，固定种子）

```bash
uv run python scripts/judge_calibration.py plan \
  --run-a runs/<run-bm25> --run-b runs/<run-none> --out calib
```

产物（`calib/`）：

| 文件 | 内容 | 谁可见 |
| --- | --- | --- |
| `plan.json` | 100 条随机分层样本 + 20 条边界样本的完整映射（条件、run、样本、题目） | harness 私有 |
| `rubric.md` | 固定标注 rubric（judge-calibration-rubric@1） | 标注者必读 |
| `worksheet_random.csv` | 首轮盲标表：100 行随机顺序 | 标注者 |
| `worksheet_boundary.csv` | 边界样本表：20 行（只诊断偏宽/偏严，不入一致率） | 标注者 |
| `worksheet_selfconsistency.csv` | 复标表：从 100 条中抽 20 条，乱序重排 | 标注者（隔天） |

脱敏保证：worksheet 只有 `item_id, question, gold_answer, response, question_type, is_abstention, annotation, note, annotated_at` 九列；不含条件标签、run 信息、judge 结论或任何检索/记忆内容；条目 ID 为乱序后分配的中性编号。

### 2. 人工盲标（用户环节）

1. 先读 `calib/rubric.md`（判定语义、yes/no/cannot_judge 定义、分题型补充）。
2. 填 `worksheet_random.csv` 的 `annotation` 列（每行必填 yes / no / cannot_judge，拿不准就标 cannot_judge，不要猜）；`note` 可选；`annotated_at` 填日期。
3. 另找时间填 `worksheet_boundary.csv`（同格式）。
4. **间隔至少一天**、不回看首轮结果，填 `worksheet_selfconsistency.csv`（衡量标注者自身一致性；单标注者只有自身一致性，没有评分者间一致性）。

可以把填好的表另存为 `*.filled.csv`，原模板留档。

### 3. judge 批量调用（live，需 ZAI_API_KEY）

```bash
uv run python scripts/judge_calibration.py judge \
  --dir calib --config eval/configs/examples/real_dev50_bm25.toml
```

- 凭证缺失时**显式 fail-fast**，不会静默跳过或假装成功。
- 对 120 条样本逐条调用官方协议 prompt（输入只有问题、标准答案与回答）；瞬时错误按 1s/4s/16s 退避重试。
- 不可解析输出记录为 `parse_failed`（评分阶段失败语义，不默认判错），统计时单独计数并排除出一致率分母。
- 产物：`calib/judge_calls.json`（每条调用的完整留档：prompt 请求、原始输出、判定、响应 model、用量、重试次数）。

### 4. 对齐产出报告（离线）

```bash
uv run python scripts/judge_calibration.py report \
  --dir calib \
  --config eval/configs/examples/real_dev50_bm25.toml \
  --annotations calib/worksheet_random.filled.csv \
  --boundary-annotations calib/worksheet_boundary.filled.csv \
  --self-consistency calib/worksheet_selfconsistency.filled.csv
```

产物：`calib/calibration.json`（机器可读记录）与 `calib/calibration.md`（人读报告：阈值判定、门槛明细、一致率与 Wilson 95% 区间、混淆矩阵、分题型与拒答子集、跨条件差、边界诊断）。

导入校验：列集合不符（含夹带识别性列）、未知/缺失条目、空标注或非法标签都会被拒绝并指出位置。

### 5. 把记录附到正式运行

```bash
uv run hippo-eval run --config eval/configs/examples/real_dev50_bm25.toml \
  --out runs --judge-calibration calib/calibration.json
```

- 记录绑定 judge 别名、协议 commit、temperature/max_tokens；不匹配即拒绝附上（配置变了必须重新校准）。
- 报告头出现「Judge 校准（诊断）」块：`passed` → 问答结论保持正式；`failed`/`inconclusive`/未附记录 → 问答结论（计划题整体得分、成功评分题准确率、拒答准确率、联合归因四格）标记为**降级诊断项**，并标注与官方 GPT-4o 的偏离；`compare` 中这些指标同样移入诊断通道，不进入正式比较与排名。

## 判定阈值（固定，改动即判据变更）

| 门槛 | 值 | 含义 |
| --- | --- | --- |
| 总体一致率 | ≥ 0.85 且 Wilson 95% 下界 ≥ 0.75 | 只报点估计不算通过 |
| 任一题型 | < 0.70 | 仅该题型结论标为不可用 |
| 拒答子集 | 门槛同总体 | 未达则拒答子集结论不可用 |
| 跨条件一致率差 | ≤ 0.05 | 超出则比较受差异化误差影响，随结论标注 |
| 自身一致率 | ≥ 0.85 | 低于则总体阈值下调至该值并写明上限 |
| 无法判定比例 | ≤ 10% | 超过则校准结论为不确定 |

注意：dev50 只有约 10 条拒答题进入校准，即便全部一致，Wilson 下界也达不到 0.75——拒答子集通常会被标为不可用，这是统计上的诚实结果，不是校准失败。

## 记录的不可变性

`calibration.json` 保存 judge 的四项版本标识（别名、观测响应 model、厂商标注版本与日期）与校准运行日期，以及判据内容哈希（rubric + 阈值 + 协议 commit + 算法）。判据未变且输入未变时重算的判定逐字节一致（`verify_decision_stability`）；判据任何变化都会改变哈希，旧记录对新判据失效，必须重新校准——判定从不被就地改写。
