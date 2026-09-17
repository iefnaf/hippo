# hippo

Hippo 计划开发供 pi、Codex 和 Claude Code 使用的共享 memory 层。本仓库当前包含
memory eval harness 的离线骨架（M1）：数据契约（pydantic v2 单一真源）、指标注册表
v1、配置指纹、validate 命令，以及完整的离线单题闭环（逐会话写入 → 重开实例 → 检索 →
证据准备 → 固定 reader 回答 → Scorer 可核验 session recall + fake judge 判定 →
JSON/Markdown 汇总报告，全程 fake 组件、不调外部模型）。

- 设计文档：`docs/design/eval-harness.md`、`docs/design/eval-harness-data-contracts.md`
- 运行测试：`uv run pytest`
- 结构校验：`uv run hippo-eval validate --config eval/configs/examples/offline_fake.toml`
- 离线闭环：`uv run hippo-eval run --config eval/configs/examples/offline_fake.toml --out runs`
  （产出 run ID、不可变配置快照、逐题 JSONL 与 `report.json` / `report.md` 汇总：
  状态计数与分母、计划题整体得分、成功评分题准确率、可核验 recall（宏/微/Recall@k）、
  检索×问答 2×2 联合归因、证据预算构成与规模成本模型；不含 `--out` 时仅打印计划）
- 操作能力套件：`uv run hippo-eval run --config eval/configs/examples/offline_fake_ops.toml --out runs`
  （自动更新、显式更新、删除、隔离与重启后持久化；通过/失败/不支持分态记录并汇总为
  `operations_summary.json`，不经 LLM judge；配置的 `suite` 字段选择 qa 或 operations 运行器）
- 重看报告：`uv run hippo-eval report --run runs/<run_id>`
- 断点恢复：`uv run hippo-eval resume --config <config> --run runs/<run_id>`
  （校验配置指纹/空间身份/检查点版本，只补未完成步骤，成功产物逐字节不变）
- 比较两个 run：`uv run hippo-eval compare runs/<left> runs/<right> [--out DIR]`
  （按除 memory 外的关键项判定可比性——样本清单、reader/judge、prompt 协议、
  预算、tokenizer、运行参数、指标注册表版本；不一致列出差异并拒绝标记同条件；
  注册表版本不一致或指纹无法在当前注册表内容下复现时拒绝指标自动对齐，
  未登记指标只作诊断；区分等预算比较与完整历史/无记忆对照；给出双方完整
  题目集合成绩、状态数量、覆盖率与共同可运行交集，交集 ID 清单随
  `compare-*.json/.md` 产物保存，run 目录保持不可变）
- 冒烟：`uv run hippo-eval run --config eval/configs/examples/offline_fake.toml --out runs`
  连跑两次后用 compare 验证指纹与成绩可比较（smoke-offline；8 题固定清单已
  锁定在配置，CI 在 GitHub Actions 常跑，全程不调用外部模型）
