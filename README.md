# hippo

Hippo 计划开发供 pi、Codex 和 Claude Code 使用的共享 memory 层。本仓库当前包含
memory eval harness 的离线骨架（M1）：数据契约（pydantic v2 单一真源）、指标注册表
v1、配置指纹与 validate 命令。

- 设计文档：`docs/design/eval-harness.md`、`docs/design/eval-harness-data-contracts.md`
- 运行测试：`uv run pytest`
- 结构校验：`uv run hippo-eval validate --config eval/configs/examples/offline_fake.toml`
