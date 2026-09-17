"""Result comparison of two run directories (M1 compare command).

Comparability follows docs/design/eval-harness.md (工程结构与产物, 比较
命令按指标检查条件): the comparison fixes reader/judge (incl. prompt
protocol), sample plan, budget, tokenizer/counting mode, run parameters
and the metric registry version; the memory implementation and its
configuration are the ALLOWED experimental factor. Full config-fingerprint
equality is therefore NOT the comparability criterion (the fingerprint
embeds the memory plan, so two implementations under test never share
one — issue #1 leftover): comparability is "every key field except
memory identical", with differences listed and the result refused the
same-condition label when any key field differs.

- Registry version mismatch refuses automatic metric alignment; a run
  whose stored fingerprint cannot be reproduced under the current
  registry content is treated the same way (definition drift since the
  run). Unregistered metric ids never enter the aligned table — they
  are reported as explicitly marked diagnostics only.
- The comparison distinguishes equal-budget comparisons (both sides
  budget-bound retrieval implementations) from controls with different
  information conditions (full-history baseline) and no-memory
  controls; neither control is labeled equal-budget.
- Both sides always report their full planned-set results, status
counts and coverage; additionally the common precheck-runnable
  intersection is computed with its ID list persisted in the compare
  artifact. The intersection is decided by precheck capability
  (context_exceeded), never by scoring success or answer correctness;
  run failures inside the intersection still contribute zero.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.metrics import UnknownMetricError, get_metric

#: Artifact schema version for compare payloads.
COMPARE_SCHEMA_VERSION = 1

#: Baseline kinds bound by the shared evidence-token budget. "none" and
#: "full_history" are control conditions with different information
#: conditions, never equal-budget comparisons.
BUDGET_BOUND_BASELINES = frozenset({"bm25", "adapter"})
CONTROL_BASELINES = frozenset({"none", "full_history"})

_MISSING = object()


class CompareError(ValueError):
    """A run directory cannot take part in a comparison."""


# ---------------------------------------------------------------------------
# Loading one run side
# ---------------------------------------------------------------------------


@dataclass
class RunSide:
    """Everything the comparison reads from one run directory."""

    run_dir: Path
    manifest: dict[str, Any]
    config_doc: dict[str, Any]
    report: dict[str, Any] | None
    #: handle -> {"qa_status": ..., "correct": bool | None} (last wins)
    samples: dict[str, dict[str, Any]] = field(default_factory=dict)
    operations_summary: dict[str, Any] | None = None

    @property
    def run_id(self) -> str:
        return self.manifest["run_id"]

    @property
    def suite(self) -> str:
        return self.manifest.get("suite", "qa")

    @property
    def config(self) -> dict[str, Any]:
        return self.config_doc["config"]

    @property
    def registry_version(self) -> Any:
        return self.config_doc.get("metrics_registry_version")

    @property
    def registry_content_version(self) -> Any:
        if self.report is not None:
            return self.report.get("header", {}).get(
                "metrics_registry_content_version"
            )
        return None

    @property
    def planned_ids(self) -> list[str]:
        return list(self.manifest.get("sample_ids", []))

    def runnable(self, handle: str) -> bool | None:
        """Precheck-runnable = not context_exceeded (qa suite).

        Runtime failures and pending states do NOT change the runnable
        set; None means the handle (or suite) has no precheck notion.
        """
        sample = self.samples.get(handle)
        if sample is None:
            return None
        return sample.get("qa_status") != "context_exceeded"


def _read_json(path: Path, what: str) -> dict[str, Any]:
    if not path.exists():
        raise CompareError(f"{what} missing: {path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompareError(f"{what} unreadable: {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise CompareError(f"{what} is not a JSON object: {path}")
    return doc


def load_run_side(run_dir: str | Path) -> RunSide:
    """Load one run directory for comparison (structural checks only)."""
    run_dir = Path(run_dir)
    manifest = _read_json(run_dir / "run.json", "run manifest")
    config_doc = _read_json(run_dir / "config.json", "config snapshot")
    side = RunSide(run_dir=run_dir, manifest=manifest, config_doc=config_doc, report=None)

    report_path = run_dir / "report.json"
    if report_path.exists():
        try:
            side.report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CompareError(f"report.json unreadable: {run_dir}: {exc}") from exc

    ops_path = run_dir / "operations_summary.json"
    if ops_path.exists():
        try:
            side.operations_summary = json.loads(
                ops_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise CompareError(
                f"operations_summary.json unreadable: {run_dir}: {exc}"
            ) from exc

    samples_path = run_dir / "samples.jsonl"
    if samples_path.exists():
        for line in samples_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)["result"]
            except (json.JSONDecodeError, KeyError) as exc:
                raise CompareError(
                    f"samples.jsonl row unreadable in {run_dir}: {exc}"
                ) from exc
            side.samples[row["sample_handle"]] = {
                "qa_status": row.get("qa_status"),
                "operation_status": row.get("operation_status"),
                "correct": row.get("correct"),
            }
    return side


# ---------------------------------------------------------------------------
# Comparability checks
# ---------------------------------------------------------------------------


def _render(value: Any) -> Any:
    if value is _MISSING:
        return "<missing>"
    if isinstance(value, (list, tuple)):
        return [_render(v) for v in value]
    return value


def _leaf_diffs(prefix: str, left: Any, right: Any) -> list[dict[str, Any]]:
    """Differences at JSON-leaf granularity under a field prefix."""
    if isinstance(left, dict) and isinstance(right, dict):
        out: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            out.extend(
                _leaf_diffs(
                    f"{prefix}/{key}",
                    left.get(key, _MISSING),
                    right.get(key, _MISSING),
                )
            )
        return out
    if left != right:
        return [{"field": prefix, "left": _render(left), "right": _render(right)}]
    return []


def _sorted_ids(config: dict[str, Any]) -> list[str]:
    return sorted(config.get("sample_ids", []))


#: Key comparability fields (everything except the memory plan and the
#: config name): any difference refuses the same-condition label.
def _key_field_diffs(left: RunSide, right: RunSide) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    lc, rc = left.config, right.config

    if lc.get("suite") != rc.get("suite"):
        diffs.append(
            {"field": "/suite", "left": lc.get("suite"), "right": rc.get("suite")}
        )
    for path in ("dataset_plan", "sample_plan_id", "evidence_token_budget"):
        if lc.get(path) != rc.get(path):
            diffs.append(
                {"field": f"/{path}", "left": lc.get(path), "right": rc.get(path)}
            )
    if _sorted_ids(lc) != _sorted_ids(rc):
        only_left = sorted(set(_sorted_ids(lc)) - set(_sorted_ids(rc)))
        only_right = sorted(set(_sorted_ids(rc)) - set(_sorted_ids(lc)))
        diffs.append(
            {
                "field": "/sample_ids",
                "left": {"count": len(_sorted_ids(lc)), "only_on_this_side": only_left},
                "right": {"count": len(_sorted_ids(rc)), "only_on_this_side": only_right},
            }
        )
    if sorted(lc.get("smoke_subset_ids", [])) != sorted(
        rc.get("smoke_subset_ids", [])
    ):
        diffs.append(
            {
                "field": "/smoke_subset_ids",
                "left": sorted(lc.get("smoke_subset_ids", [])),
                "right": sorted(rc.get("smoke_subset_ids", [])),
            }
        )
    for section in ("reader", "judge", "run_params"):
        diffs.extend(_leaf_diffs(f"/{section}", lc.get(section), rc.get(section)))

    # Metric registry: version mismatch refuses metric auto-alignment
    # and breaks the same-condition label (design: 指标注册表).
    if left.registry_version != right.registry_version:
        diffs.append(
            {
                "field": "/metrics_registry_version",
                "left": left.registry_version,
                "right": right.registry_version,
            }
        )
    left_content = left.registry_content_version
    right_content = right.registry_content_version
    if left_content is not None and right_content is not None:
        if left_content != right_content:
            diffs.append(
                {
                    "field": "/metrics_registry_content_version",
                    "left": left_content,
                    "right": right_content,
                }
            )
    return diffs


def _fingerprint_reproducible(side: RunSide) -> bool:
    """Stored fingerprint vs recomputed under the CURRENT registry.

    A mismatch means the metric registry content moved after this run
    was produced; its metric values are not safely alignable with the
    other side (definition drift, issue #1 leftover: registry content
    is hash-bound into the fingerprint).
    """
    from eval.config import ExperimentConfig

    stored = side.config_doc.get("config_fingerprint")
    try:
        config = ExperimentConfig.model_validate(side.config)
    except Exception:  # noqa: BLE001 - structural problem, reported below
        return False
    try:
        return config.fingerprint() == stored
    except Exception:  # pragma: no cover - defensive
        return False


def comparability(left: RunSide, right: RunSide) -> dict[str, Any]:
    """Key-field comparability (memory excluded) + registry alignment."""
    differences = _key_field_diffs(left, right)
    same_condition = not differences

    left_repro = _fingerprint_reproducible(left)
    right_repro = _fingerprint_reproducible(right)
    registry_mismatch = any(
        d["field"].startswith("/metrics_registry") for d in differences
    )
    drift_reasons: list[str] = []
    if registry_mismatch:
        drift_reasons.append(
            "双方指标注册表版本不一致：名称相同但版本不同的指标不能自动对齐"
        )
    if not left_repro:
        drift_reasons.append(
            f"左侧 run {left.run_id} 的配置指纹无法在当前注册表内容下复现"
            "（该 run 产生于不同的注册表内容）"
        )
    if not right_repro:
        drift_reasons.append(
            f"右侧 run {right.run_id} 的配置指纹无法在当前注册表内容下复现"
            "（该 run 产生于不同的注册表内容）"
        )
    if left.suite != right.suite:
        drift_reasons.append(
            f"suite 不一致（{left.suite} vs {right.suite}）：没有可对齐的共同正式指标"
        )

    metrics_auto_aligned = not drift_reasons
    return {
        "same_condition": same_condition,
        "criterion": (
            "可比性判据：除 memory 计划与配置名外的关键项（样本清单、"
            "reader/judge、prompt 协议、预算、tokenizer/计数模式、运行参数、"
            "指标注册表版本）全部一致；配置指纹包含 memory 计划，指纹相等"
            "不是可比性判据"
        ),
        "differences": differences,
        "metrics_auto_aligned": metrics_auto_aligned,
        "metrics_alignment_refused_reason": (
            "；".join(drift_reasons) if drift_reasons else None
        ),
        "registry_fingerprint_reproducible": {
            "left": left_repro,
            "right": right_repro,
        },
        "experimental_factor": {
            "field": "/memory",
            "note": (
                "memory 实现及其配置是允许变化的实验因素：计入指纹差异，"
                "但不构成条件不一致"
            ),
            "left": left.config.get("memory"),
            "right": right.config.get("memory"),
            "differences": _leaf_diffs(
                "/memory", left.config.get("memory"), right.config.get("memory")
            ),
        },
    }


# ---------------------------------------------------------------------------
# Comparison-kind classification
# ---------------------------------------------------------------------------


def classify(left: RunSide, right: RunSide, comp: dict[str, Any]) -> dict[str, str]:
    """equal-budget vs information-condition controls vs not comparable."""
    if not comp["same_condition"]:
        return {
            "comparison_kind": "not_same_condition",
            "note": "关键配置存在差异（见 differences）：结果不标记为同条件比较，"
            "以下数字仅为双方各自完整题目集合上的报告与诊断。",
        }
    kinds = {left.config.get("memory", {}).get("baseline_kind"),
             right.config.get("memory", {}).get("baseline_kind")}
    if "full_history" in kinds:
        return {
            "comparison_kind": "full_history_control",
            "note": "完整历史对照：不受 4K 证据预算约束，与 4K 检索的回答比较属于"
            "信息条件不同的对照，不标为等预算。",
        }
    if "none" in kinds:
        return {
            "comparison_kind": "no_memory_control",
            "note": "无记忆对照：reader 不使用历史证据，不参加等预算检索比较。",
        }
    if kinds <= BUDGET_BOUND_BASELINES:
        return {
            "comparison_kind": "equal_budget",
            "note": "等预算比较：双方实现受同一证据预算、tokenizer 与计数规则约束。",
        }
    return {
        "comparison_kind": "equal_budget",
        "note": "等预算比较（按声明基线类型判定）。",
    }


# ---------------------------------------------------------------------------
# Per-side summary and the runnable intersection
# ---------------------------------------------------------------------------


def side_summary(side: RunSide) -> dict[str, Any]:
    if side.suite == "operations":
        summary = side.operations_summary
        if summary is None:
            return {
                "suite": "operations",
                "run_id": side.run_id,
                "available": False,
                "reason": "operations_summary.json 缺失",
            }
        return {
            "suite": "operations",
            "run_id": side.run_id,
            "available": True,
            "statuses": {
                "planned": summary["planned_checks"],
                "passed": summary["passed"],
                "failed": summary["failed"],
                "not_supported": summary["not_supported"],
                "pending": summary["pending"],
            },
            "metrics": summary.get("metrics", []),
        }
    report = side.report
    if report is None:
        statuses = {
            "planned": len(side.planned_ids),
            "scored": sum(
                1 for s in side.samples.values() if s["qa_status"] == "scored"
            ),
            "failed": sum(
                1 for s in side.samples.values() if s["qa_status"] == "failed"
            ),
            "context_exceeded": sum(
                1
                for s in side.samples.values()
                if s["qa_status"] == "context_exceeded"
            ),
            "invalid_input": sum(
                1
                for s in side.samples.values()
                if s["qa_status"] == "invalid_input"
            ),
            "pending": sum(
                1 for s in side.samples.values() if s["qa_status"] == "pending"
            ),
        }
        return {
            "suite": "qa",
            "run_id": side.run_id,
            "available": False,
            "reason": "report.json 缺失（运行未完成汇总）；状态计数直接取自 samples.jsonl",
            "statuses": statuses,
            "metrics": [],
        }
    metrics_by_id = {m["metric_id"]: m for m in report.get("metrics", [])}

    def value(metric_id: str) -> Any:
        metric = metrics_by_id.get(metric_id)
        if metric is None or metric.get("status") != "computed":
            return None
        return metric.get("value")

    statuses = dict(report.get("statuses", {}))
    return {
        "suite": "qa",
        "run_id": side.run_id,
        "available": True,
        "statuses": {
            key: statuses.get(key)
            for key in (
                "planned",
                "scored",
                "failed",
                "context_exceeded",
                "invalid_input",
                "pending",
            )
        },
        "full_planned_set": {
            "planned_question_score": value("planned_question_score"),
            "scored_accuracy": value("scored_accuracy"),
            "scoring_coverage": value("scoring_coverage"),
            "runnable_coverage": value("runnable_coverage"),
            "abstention_accuracy": value("abstention_accuracy"),
        },
        "metrics": report.get("metrics", []),
    }


def runnable_intersection(left: RunSide, right: RunSide) -> dict[str, Any]:
    """Common precheck-runnable questions, with the ID list kept.

    Precheck capability decides membership (context_exceeded samples
    are out); scoring success or answer correctness never do. Run
    failures inside the intersection still contribute zero to the
    intersection score (planned-question semantics).
    """
    if left.suite != "qa" or right.suite != "qa":
        return {
            "available": False,
            "reason": "可运行交集只对 qa 套件双方定义",
            "runnable_intersection_ids": [],
        }
    left_ids, right_ids = set(left.planned_ids), set(right.planned_ids)
    common = sorted(left_ids & right_ids)
    intersection = sorted(
        handle
        for handle in common
        if left.runnable(handle) and right.runnable(handle)
    )
    same_planned = left_ids == right_ids

    def side_stats(side: RunSide) -> dict[str, Any]:
        rows = [side.samples[h] for h in intersection if h in side.samples]
        scored = sum(1 for row in rows if row["qa_status"] == "scored")
        correct = sum(
            1 for row in rows if row["qa_status"] == "scored" and row["correct"]
        )
        return {
            "scored": scored,
            "correct": correct,
            "planned_question_score": (
                correct / len(intersection) if intersection else None
            ),
            "not_runnable_on_this_side": sorted(
                h
                for h in common
                if h in side.samples and not side.runnable(h)
            ),
        }

    return {
        "available": True,
        "common_planned_ids": common,
        "left_only_ids": sorted(left_ids - right_ids),
        "right_only_ids": sorted(right_ids - left_ids),
        "runnable_intersection_ids": intersection,
        "size": len(intersection),
        "scope": (
            "同条件交集"
            if same_planned
            else "诊断性交集（双方完整题目集合不同，不替代任何一方原报告）"
        ),
        "left": side_stats(left),
        "right": side_stats(right),
        "note": (
            "交集按预检能力（context_exceeded）决定，不按评分成功或答案正确"
            "决定；交集内的运行失败仍按零分计入 planned_question_score。"
        ),
    }


# ---------------------------------------------------------------------------
# Metric alignment
# ---------------------------------------------------------------------------


def align_metrics(
    left: RunSide, right: RunSide, comp: dict[str, Any]
) -> dict[str, Any]:
    """Align formal metrics; unregistered ids are diagnostics only."""
    diagnostics: list[dict[str, Any]] = []
    if not comp["metrics_auto_aligned"]:
        return {
            "aligned": [],
            "diagnostics": diagnostics,
            "refused_reason": comp["metrics_alignment_refused_reason"],
        }

    def metric_rows(side: RunSide) -> dict[str, dict[str, Any]]:
        if side.suite == "operations":
            return {
                m["metric_id"]: m
                for m in (side.operations_summary or {}).get("metrics", [])
            }
        if side.report is None:
            return {}
        return {m["metric_id"]: m for m in side.report.get("metrics", [])}

    left_rows, right_rows = metric_rows(left), metric_rows(right)
    aligned: list[dict[str, Any]] = []
    def registered(metric_id: str) -> bool:
        try:
            get_metric(metric_id)
        except UnknownMetricError:
            return False
        return True

    for metric_id in left_rows:
        if not registered(metric_id):
            # Unregistered metrics are explicitly marked diagnostics and
            # never enter the aligned (formal) table — regardless of
            # whether the other side reports them too.
            diagnostics.append(
                {
                    "metric_id": metric_id,
                    "kind": "unregistered",
                    "note": "未登记指标：只作明确标记的诊断项，不进入正式比较与排名",
                    "left": left_rows[metric_id],
                    "right": right_rows.get(metric_id),
                }
            )
            continue
        if metric_id not in right_rows:
            diagnostics.append(
                {
                    "metric_id": metric_id,
                    "kind": "one_side_only",
                    "note": f"仅左侧报告该指标（右侧无 {metric_id}）",
                    "left": left_rows[metric_id],
                }
            )
            continue
        lrow, rrow = left_rows[metric_id], right_rows[metric_id]
        both_computed = (
            lrow.get("status") == "computed" and rrow.get("status") == "computed"
        )
        aligned.append(
            {
                "metric_id": metric_id,
                "left": {"status": lrow.get("status"), "value": lrow.get("value")},
                "right": {"status": rrow.get("status"), "value": rrow.get("value")},
                "delta": (
                    rrow.get("value") - lrow.get("value") if both_computed else None
                ),
            }
        )
    for metric_id in sorted(set(right_rows) - set(left_rows)):
        if not registered(metric_id):
            diagnostics.append(
                {
                    "metric_id": metric_id,
                    "kind": "unregistered",
                    "note": "未登记指标：只作明确标记的诊断项，不进入正式比较与排名",
                    "right": right_rows[metric_id],
                }
            )
            continue
        diagnostics.append(
            {
                "metric_id": metric_id,
                "kind": "one_side_only",
                "note": f"仅右侧报告该指标（左侧无 {metric_id}）",
                "right": right_rows[metric_id],
            }
        )
    return {"aligned": aligned, "diagnostics": diagnostics, "refused_reason": None}


# ---------------------------------------------------------------------------
# Comparison payload
# ---------------------------------------------------------------------------


def _side_header(side: RunSide) -> dict[str, Any]:
    return {
        "run_id": side.run_id,
        "run_dir": str(side.run_dir),
        "config_name": side.config.get("name"),
        "config_fingerprint": side.config_doc.get("config_fingerprint"),
        "suite": side.suite,
        "memory_name": side.config.get("memory", {}).get("name"),
        "memory_baseline_kind": side.config.get("memory", {}).get(
            "baseline_kind"
        ),
        "metrics_registry_version": side.registry_version,
        "metrics_registry_content_version": side.registry_content_version,
    }


def compare_runs(left_dir: str | Path, right_dir: str | Path) -> dict[str, Any]:
    """Build the full comparison payload for two run directories."""
    left = load_run_side(left_dir)
    right = load_run_side(right_dir)
    comp = comparability(left, right)
    kind = classify(left, right, comp)
    payload = {
        "schema_version": COMPARE_SCHEMA_VERSION,
        "command": "compare",
        "left": _side_header(left),
        "right": _side_header(right),
        "same_config_fingerprint": (
            left.config_doc.get("config_fingerprint")
            == right.config_doc.get("config_fingerprint")
        ),
        "fingerprint_note": (
            "配置指纹包含 memory 计划：实现 A/B 的指纹天然不同，指纹相等"
            "不是可比性判据；跨实现可比性见 comparability.same_condition"
        ),
        "comparability": comp,
        **kind,
        "left_summary": side_summary(left),
        "right_summary": side_summary(right),
        "intersection": runnable_intersection(left, right),
        "metrics": align_metrics(left, right, comp),
        "limitations": [
            "M1 fake 成绩仅验证 harness 行为，不代表任何 memory 实现的性能。",
            "双方各自完整题目集合上的成绩优先；交集数字不覆盖任何一方原报告。",
            "未登记指标只作诊断项，不进入正式比较与排名。",
        ],
    }
    planned = set(left.planned_ids) & set(right.planned_ids)
    smoke = set(left.config.get("smoke_subset_ids", [])) | set(
        right.config.get("smoke_subset_ids", [])
    )
    if planned and planned <= smoke:
        payload["limitations"].insert(
            0,
            "双方题目集合属于 smoke 子集：冒烟成绩不进入正式报告。",
        )
    return payload


# ---------------------------------------------------------------------------
# Rendering and persistence
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text if text else "0"
    return str(value)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any]) -> str:
    left, right = payload["left"], payload["right"]
    comp = payload["comparability"]
    lines: list[str] = []
    lines.append("# Memory Eval 运行比较")
    lines.append("")
    lines.append(
        _table(
            ["", "左", "右"],
            [
                ["run ID", left["run_id"], right["run_id"]],
                ["配置", left["config_name"], right["config_name"]],
                ["配置指纹", left["config_fingerprint"][:12] + "…", right["config_fingerprint"][:12] + "…"],
                ["suite", left["suite"], right["suite"]],
                ["memory", f"{left['memory_name']}（{left['memory_baseline_kind']}）", f"{right['memory_name']}（{right['memory_baseline_kind']}）"],
                ["指标注册表", str(left["metrics_registry_version"]), str(right["metrics_registry_version"])],
            ],
        )
    )
    lines.append("")
    lines.append("## 可比性检查")
    lines.append("")
    verdict = (
        "同条件比较（除 memory 外关键项全部一致）"
        if comp["same_condition"]
        else "**拒绝标记为同条件比较**（下列关键项不一致）"
    )
    lines.append(f"- 判定：{verdict}")
    lines.append(f"- 比较类型：{payload['comparison_kind']}——{payload['note']}")
    lines.append(f"- 判据：{comp['criterion']}")
    aligned_note = (
        "已按 metric_id 自动对齐"
        if comp["metrics_auto_aligned"]
        else f"**拒绝自动对齐**：{comp['metrics_alignment_refused_reason']}"
    )
    lines.append(f"- 指标对齐：{aligned_note}")
    if comp["differences"]:
        lines.append("")
        lines.append("关键项差异：")
        lines.append("")
        lines.append(
            _table(
                ["字段", "左", "右"],
                [
                    [d["field"], _fmt(d["left"]), _fmt(d["right"])]
                    for d in comp["differences"]
                ],
            )
        )
    factor = comp["experimental_factor"]
    if factor["differences"]:
        lines.append("")
        lines.append(
            "实验因素差异（memory 计划，允许变化）："
            + "、".join(d["field"] for d in factor["differences"])
        )
    lines.append("")
    lines.append("## 双方完整题目集合成绩")
    lines.append("")
    for label, summary in (("左", payload["left_summary"]), ("右", payload["right_summary"])):
        lines.append(f"- **{label}（{summary['run_id']}）**")
        if not summary.get("available"):
            lines.append(f"  - {summary.get('reason', '报告不可用')}")
            continue
        statuses = summary["statuses"]
        if summary["suite"] == "qa":
            full = summary["full_planned_set"]
            lines.append(
                "  - 状态：planned={planned} scored={scored} failed={failed} "
                "context_exceeded={ce} invalid_input={ii} pending={pg}".format(
                    planned=statuses.get("planned"),
                    scored=statuses.get("scored"),
                    failed=statuses.get("failed"),
                    ce=statuses.get("context_exceeded"),
                    ii=statuses.get("invalid_input"),
                    pg=statuses.get("pending"),
                )
            )
            lines.append(
                "  - 计划题整体得分 {pqs}；成功评分题准确率 {acc}；"
                "评分覆盖率 {sc}；可运行覆盖率 {rc}".format(
                    pqs=_fmt(full["planned_question_score"]),
                    acc=_fmt(full["scored_accuracy"]),
                    sc=_fmt(full["scoring_coverage"]),
                    rc=_fmt(full["runnable_coverage"]),
                )
            )
        else:
            lines.append(
                "  - 状态：planned={planned} passed={passed} failed={failed} "
                "not_supported={ns} pending={pg}".format(
                    planned=statuses.get("planned"),
                    passed=statuses.get("passed"),
                    failed=statuses.get("failed"),
                    ns=statuses.get("not_supported"),
                    pg=statuses.get("pending"),
                )
            )
    lines.append("")
    lines.append("## 共同可运行交集")
    lines.append("")
    inter = payload["intersection"]
    if not inter.get("available"):
        lines.append(f"- {inter.get('reason')}")
    else:
        lines.append(
            f"- 交集 {inter['size']} 题（{inter['scope']}）；{inter['note']}"
        )
        lines.append(
            _table(
                ["", "scored", "correct", "整体得分"],
                [
                    [
                        "左",
                        str(inter["left"]["scored"]),
                        str(inter["left"]["correct"]),
                        _fmt(inter["left"]["planned_question_score"]),
                    ],
                    [
                        "右",
                        str(inter["right"]["scored"]),
                        str(inter["right"]["correct"]),
                        _fmt(inter["right"]["planned_question_score"]),
                    ],
                ],
            )
        )
        lines.append("")
        lines.append(
            "交集 ID 清单：" + ("、".join(inter["runnable_intersection_ids"]) or "（空）")
        )
        if inter["left_only_ids"] or inter["right_only_ids"]:
            lines.append(
                f"仅左侧：{inter['left_only_ids'] or '无'}；仅右侧：{inter['right_only_ids'] or '无'}"
            )
    lines.append("")
    lines.append("## 指标对齐")
    lines.append("")
    metrics = payload["metrics"]
    if metrics.get("refused_reason"):
        lines.append(f"拒绝自动对齐：{metrics['refused_reason']}")
    elif metrics["aligned"]:
        lines.append(
            _table(
                ["metric_id", "左", "右", "差值（右−左）"],
                [
                    [
                        row["metric_id"],
                        _fmt(row["left"]["value"]) if row["left"]["status"] == "computed" else "N/A",
                        _fmt(row["right"]["value"]) if row["right"]["status"] == "computed" else "N/A",
                        _fmt(row["delta"]),
                    ]
                    for row in metrics["aligned"]
                ],
            )
        )
    else:
        lines.append("（无共同指标行）")
    if metrics["diagnostics"]:
        lines.append("")
        lines.append("诊断项（不进入正式比较）：")
        for item in metrics["diagnostics"]:
            lines.append(f"- {item['metric_id']}：{item['note']}")
    lines.append("")
    lines.append("## 限制")
    lines.append("")
    for note in payload["limitations"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _safe_run_id(run_id: str) -> str:
    return "".join(
        ch if (ch.isalnum() or ch in "-_.") else "_" for ch in run_id
    )


def save_comparison(
    payload: dict[str, Any], out_dir: str | Path
) -> dict[str, str]:
    """Persist the comparison artifact (JSON + Markdown) beside the runs.

    Run directories are immutable, so the comparison artifact (which
    must carry the intersection ID list) is written to a separate
    output directory, never into the compared runs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"compare-{_safe_run_id(payload['left']['run_id'])}"
        f"-vs-{_safe_run_id(payload['right']['run_id'])}"
    )
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return {"compare_json": str(json_path), "compare_markdown": str(md_path)}
