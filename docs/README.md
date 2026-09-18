# 项目文档

本项目计划基于《Agent Memory — The 5-Layer Playbook》的思想，开发供 pi、Codex 和 Claude Code 使用的 agent memory 层插件工具。

## 目录

- [references/](references/)：论文及其他参考资料。
- [design/](design/)：后续的需求、架构和设计文档。

## 参考资料

- [Agent Memory — The 5-Layer Playbook](references/Agent%20Memory%20%E2%80%94%20The%205-Layer%20Playbook.pdf)

## 设计文档

- [Memory Eval Harness 第一版设计](design/eval-harness.md)
- [Eval Harness 数据契约草案](design/eval-harness-data-contracts.md)：公共输入输出、证据结构、回执及 harness 内部评分与记录类型。
- [Eval Harness 单题调用图](design/diagrams/eval-harness-calls.html)：可缩放、点选查看调用与数据说明；[图源](design/diagrams/eval-harness-calls.sequence.json)。
- [Judge 校准人工标注工作流](design/judge-calibration-workflow.md)：M2 校准的操作文档——抽样清单、盲标表格模板、导入方式与 live 执行步骤。

## 写作约定

设计文档、架构方案与实施计划使用项目内的 [write-design-doc 技能](../.agents/skills/write-design-doc/SKILL.md)。[AGENTS.md](../AGENTS.md) 规定了触发条件。技能以 [Refactoring English 的设计文档文章](https://refactoringenglish.com/excerpts/write-an-effective-design-doc/)为方法依据，并包含本项目的一致性检查。
