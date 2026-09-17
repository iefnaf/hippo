# hippo

Hippo 计划开发供 pi、Codex 和 Claude Code 使用的共享 memory 层。本仓库当前包含
memory eval harness 的离线骨架（M1）：数据契约（pydantic v2 单一真源）、指标注册表
v1、配置指纹、validate 命令，以及离线单题闭环（逐会话写入 → 重开实例 → 检索 →
证据准备，全程 fake 组件、不调外部模型）。

- 设计文档：`docs/design/eval-harness.md`、`docs/design/eval-harness-data-contracts.md`
- 运行测试：`uv run pytest`
- 结构校验：`uv run hippo-eval validate --config eval/configs/examples/offline_fake.toml`
- 离线闭环：`uv run hippo-eval run --config eval/configs/examples/offline_fake.toml --out runs`
  （产出 run ID、不可变配置快照与逐题 JSONL；不含 `--out` 时仅打印计划）
