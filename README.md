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
- 重看报告：`uv run hippo-eval report --run runs/<run_id>`
