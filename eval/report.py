"""Run summary reporter: JSON + Markdown aggregation.

The reporter reads ONLY persisted run artifacts (manifest, config
snapshot, per-sample JSONL and their referenced artifacts); it never
triggers retrieval, answering or re-scoring. Every metric entering the
formal table comes from the v1 metric registry; the evidence budget
composition and the scale/cost model are clearly marked diagnostics.

Aggregation rules follow the design doc (适用范围、状态与统计分母):

- planned question score: correct / |P|; failed, context_exceeded and
  invalid_input all contribute zero (never folded into "wrong");
- scored accuracy: correct / scored, N/A when scored==0;
- runnable coverage: (|P| - context_exceeded) / |P|;
- retrieval recall aggregates run over the applicable set E
  (non-abstention, extractive-declared, valid gold); micro recall uses
  gold session counts as denominator;
- the 2x2 attribution uses ONLY scored questions; failed,
  context_exceeded, pending and invalid_input are listed separately;
- budget composition reports adapter-returned units, units entering the
  reader, dropped/truncated units and their tokens (diagnostic, test
  counting mode only in M1);
- the scale & cost model compares planned call counts with actual ones
  and extrapolates fake unit costs (explicitly marked estimates).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from eval.contracts.internal import (
    PreparedEvidenceArtifact,
    RawEvidenceArtifact,
    ReaderResultArtifact,
    ResultArtifact,
    ScoringTraceArtifact,
)
from eval.metrics import REGISTRY_CONTENT_VERSION, get_metric
from eval.runs import AttemptLogArtifact, RunStore

#: Formal metric ids reported for a qa-suite run (all registered v1).
FORMAL_METRIC_ORDER = (
    "planned_question_score",
    "scored_accuracy",
    "scoring_coverage",
    "runnable_coverage",
    "abstention_accuracy",
    "attribution_hit_correct",
    "attribution_hit_wrong",
    "attribution_miss_correct",
    "attribution_miss_wrong",
    "verifiable_session_recall_macro",
    "verifiable_session_recall_micro",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "budgeted_session_recall",
    "derivation_source_coverage",
    "retrieval_returned_units",
    "retrieval_reader_units",
    "retrieval_dropped_units",
    "evidence_text_tokens",
    "evidence_format_tokens",
    "ingest_latency_ms",
    "retrieve_latency_ms_p50",
    "retrieve_latency_ms_p95",
)

ATTRIBUTION_CELLS = (
    "hit_correct",
    "hit_wrong",
    "miss_correct",
    "miss_wrong",
)

HIT_CRITERION_NOTES = {
    "gold_recall": (
        "gold_recall：实现声明提供原文检索，命中 = 4K 预算内实际保留的原文证据与 "
        "gold 会话集合有交集（session recall > 0）；拒答题 gold 为空，恒为未命中"
    ),
    "nonempty_evidence": (
        "nonempty_evidence：无可核验 gold 来源的对照（纯生成/完整历史），"
        "命中 = 实际进入了 Reader 的证据非空"
    ),
    "never": "never：无记忆基线恒为未命中",
}

#: Control-role labels (issue #8 AC5): every run report states whether
#: the run is an equal-budget comparison member or a control with a
#: different information condition.
BASELINE_CONTROL_ROLES = {
    "none": {
        "role": "no_memory_control",
        "equal_budget": False,
        "note": "无记忆对照：reader 不使用历史证据，不参加等预算检索比较。",
    },
    "full_history": {
        "role": "full_history_control",
        "equal_budget": False,
        "note": (
            "完整历史对照：不受 4K 证据预算约束（按上下文窗口整量提供），"
            "与 4K 检索基线的回答比较属于信息条件不同的对照，不标为等预算；"
            "不参加排名检索指标（recall 为 N/A）。"
        ),
    },
    "bm25": {
        "role": "equal_budget_baseline",
        "equal_budget": True,
        "note": "等预算检索基线：与 hippo 等实现受同一 4K 证据预算约束。",
    },
    "adapter": {
        "role": "equal_budget_baseline",
        "equal_budget": True,
        "note": "被测实现：受 4K 证据预算约束，可与等预算基线同条件比较。",
    },
}


def baseline_control_role(baseline_kind: str) -> dict[str, Any]:
    """The run's comparison role label (equal-budget member vs control)."""
    role = BASELINE_CONTROL_ROLES.get(baseline_kind)
    if role is None:
        return {
            "role": "unknown",
            "equal_budget": None,
            "note": f"未知基线类型 {baseline_kind!r}。",
        }
    return dict(role)


class ReportError(ValueError):
    """The run directory cannot be summarized."""


# ---------------------------------------------------------------------------
# Report models
# ---------------------------------------------------------------------------


class ReportHeader(dict):
    """Plain key/value header (kept a dict for simple JSON rendering)."""


class MetricAggregate(dict):
    """{metric_id, status, value, denominator, reason}."""


class SummaryReport(dict):
    """The full report payload (also persisted with schema_version)."""


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


class Reporter:
    """Builds the summary report from one run directory."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        if not (self.run_dir / "run.json").exists():
            raise ReportError(f"not a run directory: {self.run_dir}")
        manifest = json.loads((self.run_dir / "run.json").read_text(encoding="utf-8"))
        self.manifest = manifest
        config_doc = json.loads((self.run_dir / "config.json").read_text(encoding="utf-8"))
        self.config_doc = config_doc
        self.store = RunStore(self.run_dir.parent, manifest["run_id"])

    # -- loading ------------------------------------------------------------

    def _load_results(self) -> list[Any]:
        # Last-wins per sample handle: a resumed sample appends a new
        # Result line and the latest terminal record supersedes its
        # checkpointed predecessor.
        latest: dict[str, Any] = {}
        for line in self.store.read_result_lines():
            result = ResultArtifact.load_json(json.dumps(line)).result
            latest[result.sample_handle] = result
        results = list(latest.values())
        planned_ids = list(self.manifest["sample_ids"])
        seen = [r.sample_handle for r in results]
        if sorted(seen) != sorted(planned_ids):
            raise ReportError(
                f"samples.jsonl handles {sorted(seen)} do not match the "
                f"planned sample ids {sorted(planned_ids)}"
            )
        return results

    def _load_ref(self, ref: str | None, model: Any) -> Any | None:
        if ref is None:
            return None
        doc = json.loads(self.store.resolve_ref(ref).read_text(encoding="utf-8"))
        return model.load_json(json.dumps(doc))

    # -- build --------------------------------------------------------------

    def build(self) -> SummaryReport:
        if self.manifest.get("suite") == "operations":
            return self._build_operations_report()
        return self._build_qa_report()

    # -- operations suite ----------------------------------------------------

    def _build_operations_report(self) -> SummaryReport:
        """Suite-dispatched report for operations runs (issue #4 leftover).

        Operations run directories were previously forced through the
        qa-shaped renderer; they now render their own shape from the
        persisted operations summary: per-check status table, separate
        passed/failed/not_supported/pending counts and the two
        registered operations metrics.
        """
        import json as _json

        summary_path = self.run_dir / "operations_summary.json"
        if not summary_path.exists():
            raise ReportError(
                f"operations run directory {self.run_dir} has no "
                "operations_summary.json; re-run or resume it first"
            )
        summary = _json.loads(summary_path.read_text(encoding="utf-8"))
        config = self.config_doc["config"]
        header = {
            "run_id": self.manifest["run_id"],
            "command": self.manifest.get("command", "run"),
            "created_at": self.manifest["created_at"],
            "config_name": self.manifest["config_name"],
            "config_fingerprint": self.manifest["config_fingerprint"],
            "dataset_plan": self.manifest["dataset_plan"],
            "sample_plan_id": self.manifest["sample_plan_id"],
            "suite": "operations",
            "evidence_token_budget": self.manifest["evidence_token_budget"],
            "memory_name": config["memory"]["name"],
            "memory_baseline_kind": config["memory"]["baseline_kind"],
            "metrics_registry_version": self.manifest["metrics_registry_version"],
            "metrics_registry_content_version": REGISTRY_CONTENT_VERSION,
        }
        statuses = {
            "planned": summary["planned_checks"],
            "passed": summary["passed"],
            "failed": summary["failed"],
            "not_supported": summary["not_supported"],
            "pending": summary["pending"],
        }
        metrics: list[MetricAggregate] = []
        denominator = summary["passed"] + summary["failed"]
        if denominator:
            metrics.append(
                MetricAggregate(
                    metric_id="operations_pass_rate",
                    status="computed",
                    value=_round(summary["pass_rate"]),
                    denominator=f"passed+failed={denominator}",
                    reason=None,
                )
            )
        else:
            metrics.append(
                MetricAggregate(
                    metric_id="operations_pass_rate",
                    status="not_applicable",
                    value=None,
                    denominator="",
                    reason="denominator_zero：没有任何已执行（passed/failed）的检查",
                )
            )
        if summary["planned_checks"]:
            metrics.append(
                MetricAggregate(
                    metric_id="operations_support_coverage",
                    status="computed",
                    value=_round(summary["support_coverage"]),
                    denominator=f"计划检查={summary['planned_checks']}",
                    reason=None,
                )
            )
        for metric in metrics:
            get_metric(metric["metric_id"])  # registry guard
        checks = [
            {
                "check_id": check["check_id"],
                "status": check["operation_status"],
                "namespaces": list(check["namespaces"]),
                "target_memory_ids": list(check["target_memory_ids"]),
                "missing_capabilities": list(check["missing_capabilities"]),
                "failed_stage": check["failed_stage"],
                "reason": check["reason"],
                "assertions": [
                    {
                        "name": a["name"],
                        "passed": a["passed"],
                        "expected": a["expected"],
                        "observed": a["observed"],
                    }
                    for a in check["assertions"]
                ],
            }
            for check in summary["checks"]
        ]
        limitations = [
            "M1 操作检查为 fake 路径的程序化断言，不代表任何 memory 实现的性能。",
            "通过率分母为 passed+failed；not_supported 不进分母也不被隐藏。",
            "正式指标全部来自指标注册表。",
        ]
        if summary["pending"]:
            limitations.insert(
                0,
                "本次运行存在未到终态（pending）的检查：报告仅为中间进度。",
            )
        return SummaryReport(
            schema_version=1,
            header=header,
            statuses=statuses,
            metrics=metrics,
            checks=checks,
            limitations=limitations,
        )

    def _build_qa_report(self) -> SummaryReport:
        results = self._load_results()
        traces: dict[str, ScoringTraceArtifact | None] = {}
        prepared: dict[str, PreparedEvidenceArtifact | None] = {}
        raws: dict[str, RawEvidenceArtifact | None] = {}
        logs: dict[str, AttemptLogArtifact | None] = {}
        for result in results:
            refs = result.artifact_refs
            traces[result.sample_handle] = self._load_ref(
                refs.get("scoring"), ScoringTraceArtifact
            )
            prepared[result.sample_handle] = self._load_ref(
                refs.get("prepared_evidence"), PreparedEvidenceArtifact
            )
            raws[result.sample_handle] = self._load_ref(
                refs.get("raw_evidence"), RawEvidenceArtifact
            )
            logs[result.sample_handle] = self._load_ref(refs.get("attempts"), AttemptLogArtifact)

        config = self.config_doc["config"]
        statuses = self._status_block(results, traces)
        metrics = self._formal_metrics(results, traces, prepared, raws, logs, statuses)
        attribution = self._attribution_block(results, traces)
        budget = self._budget_composition(prepared, raws)
        cost_model = self._cost_model(results, logs)
        failures = self._failure_rows(results, logs)
        reader_results = {
            result.sample_handle: self._load_ref(
                result.artifact_refs.get("reader_result"), ReaderResultArtifact
            )
            for result in results
        }
        header = {
            "run_id": self.manifest["run_id"],
            "command": self.manifest.get("command", "run"),
            "created_at": self.manifest["created_at"],
            "code_version": self.manifest.get("code_version", ""),
            "config_name": self.manifest["config_name"],
            "config_fingerprint": self.manifest["config_fingerprint"],
            "dataset_plan": self.manifest["dataset_plan"],
            "sample_plan_id": self.manifest["sample_plan_id"],
            "suite": "qa",
            "evidence_token_budget": self.manifest["evidence_token_budget"],
            "memory_name": config["memory"]["name"],
            "memory_baseline_kind": config["memory"]["baseline_kind"],
            "baseline_control_role": baseline_control_role(
                config["memory"]["baseline_kind"]
            ),
            "reader_model": config["reader"]["model"],
            "reader_model_family": config["reader"]["model_family"],
            "judge_model": config["judge"]["model"],
            "judge_model_family": config["judge"]["model_family"],
            "judge_protocol_id": config["judge"]["protocol_id"],
            "counting_mode": config["reader"]["counting_mode"],
            "tokenizer_id": config["reader"]["tokenizer_id"],
            "model_versions": self._model_versions_doc(),
            "context_precheck": self._context_precheck_doc(),
            "metrics_registry_version": self.manifest["metrics_registry_version"],
            "metrics_registry_content_version": REGISTRY_CONTENT_VERSION,
        }
        limitations = [
            "M1 fake 成绩仅验证 harness 行为，不代表任何 memory 实现的性能。",
            (
                f"证据计数模式为 {config['reader']['counting_mode']}"
                "（M1 为 1 token/字符的测试计数器）：预算与 tokens 数字"
                "不与真实模型结果比较。"
            ),
            "规模与成本模型中的外推值为估算值，已与实测值分开标注。",
            "正式指标全部来自指标注册表；证据预算构成与成本模型为诊断项。",
            "smoke 子集成绩不进入正式报告。",
        ]
        if config["reader"]["counting_mode"] == "estimated":
            limitations.insert(
                0,
                "估算计数模式（estimated）：预算与 tokens 数字为估算口径，"
                "与精确计数（exact）结果分开比较，不并表。",
            )
        if config["judge"].get("api") == "openai_chat":
            limitations.insert(
                0,
                "judge 与官方论文验证过的 GPT-4o 不同家族（偏离已随运行头"
                "版本记录留档）：不宣称与论文分数可比，正式结论前须完成"
                "“Judge 校准”的 100+20 人工判定。",
            )
        if statuses["invalid_input"]:
            limitations.insert(
                0,
                "本次运行包含 invalid_input 样本（数据校验错误），不构成完整 "
                "正式报告；修正数据后需开启新 run。",
            )
        if statuses["pending"]:
            limitations.insert(
                0,
                "本次运行存在未到终态（pending）的样本：报告仅为中间进度。",
            )
        return SummaryReport(
            schema_version=1,
            header=header,
            statuses=statuses,
            metrics=metrics,
            attribution=attribution,
            budget=budget,
            cost_model=cost_model,
            failures=failures,
            token_calibration=self._calibration_block(reader_results),
            limitations=limitations,
        )

    # -- blocks ---------------------------------------------------------------

    def _model_versions_doc(self) -> dict[str, Any] | None:
        """The archived four-identifier version record, if written."""
        path = self.run_dir / "model_versions.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _context_precheck_doc(self) -> dict[str, Any]:
        """The configured context precheck inputs (or the M1 no-limit case)."""
        reader = self.config_doc["config"]["reader"]
        if reader.get("context_window_tokens") is None:
            return {
                "enabled": False,
                "note": "未声明上下文窗口（M1 离线配置）：预检未执行，可运行覆盖率恒为 1。",
            }
        return {
            "enabled": True,
            "context_window_tokens": reader["context_window_tokens"],
            "format_overhead_tokens": reader.get("format_overhead_tokens"),
            "output_reserve_tokens": reader.get("output_reserve_tokens"),
            "components": (
                "公共 prompt+问题 + 证据（等预算基线取预算上界/完整历史取全量渲染）"
                "+ 消息格式开销 + 固定输出预留"
            ),
        }

    def _calibration_block(
        self, reader_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Token counting calibration diagnostics (explicitly NOT a formal
        metric: local count vs server usage per reader call).

        Estimated-mode calls never merge into exact-mode comparisons; the
        counting mode is reported with the block.
        """
        rows = [
            artifact.result.calibration
            for artifact in reader_results.values()
            if artifact is not None and artifact.result.calibration is not None
        ]
        with_server = [c for c in rows if c.server_prompt_tokens is not None]
        mismatches = [
            c for c in with_server if (c.delta_tokens or 0) != 0
        ]
        deltas = [abs(c.delta_tokens or 0) for c in with_server]
        counting_modes = sorted({c.counting_mode for c in rows})
        return {
            "kind": "diagnostic",
            "note": (
                "计数校准（诊断项，不入正式指标）：预算执行以本地计数为准，"
                "服务端 usage 仅用于校准；差值不为零即记录。"
            ),
            "calls_with_calibration": len(rows),
            "calls_with_server_usage": len(with_server),
            "mismatch_calls": len(mismatches),
            "mean_abs_delta_tokens": (
                round(sum(deltas) / len(deltas), 3) if deltas else None
            ),
            "max_abs_delta_tokens": max(deltas) if deltas else None,
            "counting_modes": counting_modes,
            "sample_deltas": [
                {
                    "local": c.local_prompt_tokens,
                    "server": c.server_prompt_tokens,
                    "delta": c.delta_tokens,
                }
                for c in with_server[:10]
            ],
        }

    def _status_block(self, results: list[Any], traces: dict[str, Any]) -> dict[str, Any]:
        counts = {"scored": 0, "failed": 0, "context_exceeded": 0, "invalid_input": 0, "pending": 0}
        for result in results:
            counts[result.qa_status] += 1
        failed_stages: dict[str, int] = {}
        for result in results:
            if result.failed_stage:
                failed_stages[result.failed_stage] = (
                    failed_stages.get(result.failed_stage, 0) + 1
                )
        planned = len(results)
        abstention_planned = sum(
            1
            for r in results
            if (traces.get(r.sample_handle) is not None)
            and traces[r.sample_handle].scoring.is_abstention
        )
        abstention_scored = sum(
            1
            for r in results
            if r.qa_status == "scored"
            and (traces.get(r.sample_handle) is not None)
            and traces[r.sample_handle].scoring.is_abstention
        )
        return {
            "planned": planned,
            "scored": counts["scored"],
            "failed": counts["failed"],
            "context_exceeded": counts["context_exceeded"],
            "invalid_input": counts["invalid_input"],
            "pending": counts["pending"],
            "failed_stages": failed_stages,
            "abstention_planned": abstention_planned,
            "abstention_scored": abstention_scored,
        }

    def _formal_metrics(
        self,
        results: list[Any],
        traces: dict[str, Any],
        prepared: dict[str, Any],
        raws: dict[str, Any],
        logs: dict[str, Any],
        statuses: dict[str, Any],
    ) -> list[MetricAggregate]:
        planned = statuses["planned"]
        scored = statuses["scored"]
        correct = sum(1 for r in results if r.qa_status == "scored" and r.correct)
        aggregates: dict[str, MetricAggregate] = {}

        def computed(metric_id: str, value: float | None, denominator: str) -> None:
            aggregates[metric_id] = MetricAggregate(
                metric_id=metric_id,
                status="computed",
                value=_round(value),
                denominator=denominator,
                reason=None,
            )

        def na(metric_id: str, reason: str, denominator: str = "") -> None:
            aggregates[metric_id] = MetricAggregate(
                metric_id=metric_id,
                status="not_applicable",
                value=None,
                denominator=denominator,
                reason=reason,
            )

        # -- QA overall -----------------------------------------------------
        if planned:
            computed("planned_question_score", correct / planned, f"|P|={planned}")
            computed("scoring_coverage", scored / planned, f"|P|={planned}")
            runnable = planned - statuses["context_exceeded"]
            computed(
                "runnable_coverage",
                runnable / planned,
                f"|P|={planned}（预检可运行 {runnable}）",
            )
        if scored:
            computed("scored_accuracy", correct / scored, f"scored={scored}")
        else:
            na("scored_accuracy", "denominator_zero：没有成功评分的题目")
        abstention_planned = statuses["abstention_planned"]
        if abstention_planned:
            correct_abs = sum(
                1
                for r in results
                if r.qa_status == "scored"
                and r.correct
                and (traces.get(r.sample_handle) is not None)
                and traces[r.sample_handle].scoring.is_abstention
            )
            computed(
                "abstention_accuracy",
                correct_abs / abstention_planned,
                f"计划拒答题={abstention_planned}",
            )
        else:
            na("abstention_accuracy", "denominator_zero：计划题目中没有拒答题")

        # -- attribution shares ----------------------------------------------
        cells = {c: 0 for c in ATTRIBUTION_CELLS}
        for r in results:
            if r.qa_status == "scored":
                cells[r.attribution] += 1
        for cell in ATTRIBUTION_CELLS:
            metric_id = f"attribution_{cell}"
            if scored:
                computed(metric_id, cells[cell] / scored, f"scored={scored}")
            else:
                na(metric_id, "denominator_zero：没有成功评分的题目")

        # -- recall aggregates -------------------------------------------------
        by_metric: dict[str, list[float]] = {}
        for r in results:
            for metric in r.metrics:
                if metric.status == "computed" and metric.value is not None:
                    by_metric.setdefault(metric.metric_id, []).append(metric.value)
        for metric_id in (
            "verifiable_session_recall_macro",
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "budgeted_session_recall",
        ):
            values = by_metric.get(metric_id, [])
            if values:
                computed(metric_id, _mean(values), f"|E|={len(values)} 道适用题")
            else:
                na(
                    metric_id,
                    "denominator_zero / evidence_mode：没有适用题（非拒答、声明"
                    "原文检索且 gold 有效），E=0",
                )
        applicable_traces = [t for t in traces.values() if t is not None and t.recall_applicable]
        total_gold = sum(len(t.gold_internal_sessions) for t in applicable_traces)
        total_hit = sum(len(t.hit_gold_sessions) for t in applicable_traces)
        if total_gold:
            computed(
                "verifiable_session_recall_micro",
                total_hit / total_gold,
                f"gold 会话总数={total_gold}",
            )
        else:
            modes = {
                t.evidence_mode for t in traces.values() if t is not None
            }
            if modes and modes <= {"generated_only", "none_baseline"}:
                reason = (
                    "evidence_mode / denominator_zero：没有任何声明原文检索"
                    "的适用题（E=0），gold 会话总数为 0"
                )
            else:
                reason = "denominator_zero：适用题的 gold 会话总数为 0"
            na("verifiable_session_recall_micro", reason)

        # -- derivation coverage / budget composition --------------------------
        gen_total = sum(
            len(
                [i for i in p.prepared.items if i.evidence.kind == "generated"]
            )
            for p in prepared.values()
            if p is not None
        )
        gen_with_sources = sum(
            len(
                [
                    i
                    for i in p.prepared.items
                    if i.evidence.kind == "generated"
                    and i.evidence.derivation_sources
                ]
            )
            for p in prepared.values()
            if p is not None
        )
        if gen_total:
            computed(
                "derivation_source_coverage",
                gen_with_sources / gen_total,
                f"保留的生成内容单元={gen_total}",
            )
        else:
            na(
                "derivation_source_coverage",
                "denominator_zero：没有保留的生成内容单元（诊断性指标）",
            )

        raw_returns = [len(r.evidence) for r in raws.values() if r is not None]
        if raw_returns:
            computed(
                "retrieval_returned_units",
                _mean([float(v) for v in raw_returns]),
                f"有返回的检索次数={len(raw_returns)}",
            )
        else:
            na("retrieval_returned_units", "denominator_zero：没有成功的检索")
        prepared_list = [p.prepared for p in prepared.values() if p is not None]
        if prepared_list:
            computed(
                "retrieval_reader_units",
                _mean([float(len(p.items)) for p in prepared_list]),
                f"完成准备的检索次数={len(prepared_list)}",
            )
            computed(
                "retrieval_dropped_units",
                _mean([float(len(p.dropped_raw_indices)) for p in prepared_list]),
                f"完成准备的检索次数={len(prepared_list)}",
            )
            computed(
                "evidence_text_tokens",
                _mean([float(p.text_token_count) for p in prepared_list]),
                f"完成准备的检索次数={len(prepared_list)}",
            )
            computed(
                "evidence_format_tokens",
                _mean([float(p.token_count - p.text_token_count) for p in prepared_list]),
                f"完成准备的检索次数={len(prepared_list)}",
            )
        else:
            for metric_id in (
                "retrieval_reader_units",
                "retrieval_dropped_units",
                "evidence_text_tokens",
                "evidence_format_tokens",
            ):
                na(metric_id, "denominator_zero：没有完成准备的检索")

        # -- latency -----------------------------------------------------------
        logical_ingest = self._logical_ingest_latencies(results)
        if logical_ingest:
            computed(
                "ingest_latency_ms",
                _mean(logical_ingest),
                f"逻辑写入（含等待与重试）={len(logical_ingest)}",
            )
        else:
            na("ingest_latency_ms", "denominator_zero：没有写入尝试")
        retrieve_latencies = [
            float(a.elapsed_ms)
            for r in results
            for a in r.attempts
            if a.stage == "retrieve" and a.outcome == "returned" and a.elapsed_ms is not None
        ]
        p50 = _percentile(retrieve_latencies, 0.50)
        p95 = _percentile(retrieve_latencies, 0.95)
        if retrieve_latencies:
            computed(
                "retrieve_latency_ms_p50",
                p50,
                f"成功逻辑检索={len(retrieve_latencies)}",
            )
            computed(
                "retrieve_latency_ms_p95",
                p95,
                f"成功逻辑检索={len(retrieve_latencies)}",
            )
        else:
            na("retrieve_latency_ms_p50", "denominator_zero：没有成功的检索")
            na("retrieve_latency_ms_p95", "denominator_zero：没有成功的检索")

        ordered: list[MetricAggregate] = []
        for metric_id in FORMAL_METRIC_ORDER:
            get_metric(metric_id)  # registry guard: only registered ids
            aggregate = aggregates.get(metric_id)
            if aggregate is not None:
                ordered.append(aggregate)
        return ordered

    def _logical_ingest_latencies(self, results: list[Any]) -> list[float]:
        latencies: list[float] = []
        for result in results:
            current: list[float] | None = None
            for attempt in result.attempts:
                if attempt.stage == "ingest":
                    if current is not None:
                        latencies.append(sum(current))
                    current = []
                if attempt.stage in ("ingest", "await_ready") and current is not None:
                    if attempt.elapsed_ms is not None:
                        current.append(attempt.elapsed_ms)
            if current is not None:
                latencies.append(sum(current))
        return latencies

    def _attribution_block(
        self, results: list[Any], traces: dict[str, Any]
    ) -> dict[str, Any]:
        scored_results = [r for r in results if r.qa_status == "scored"]

        def empty() -> dict[str, int]:
            return {c: 0 for c in ATTRIBUTION_CELLS}

        overall = empty()
        by_abstention: dict[str, dict[str, int]] = {"true": empty(), "false": empty()}
        by_type: dict[str, dict[str, int]] = {}
        for result in scored_results:
            overall[result.attribution] += 1
            trace = traces.get(result.sample_handle)
            if trace is None:
                continue
            key = "true" if trace.scoring.is_abstention else "false"
            by_abstention[key][result.attribution] += 1
            by_type.setdefault(trace.scoring.question_type, empty())[
                result.attribution
            ] += 1
        scored = len(scored_results)
        shares = {
            c: (_round(overall[c] / scored) if scored else None)
            for c in ATTRIBUTION_CELLS
        }
        criteria = sorted(
            {
                t.hit_criterion
                for t in traces.values()
                if t is not None
            }
        )
        criterion_notes = [HIT_CRITERION_NOTES[c] for c in criteria]
        excluded = {
            "failed": sum(1 for r in results if r.qa_status == "failed"),
            "context_exceeded": sum(
                1 for r in results if r.qa_status == "context_exceeded"
            ),
            "pending": sum(1 for r in results if r.qa_status == "pending"),
            "invalid_input": sum(
                1 for r in results if r.qa_status == "invalid_input"
            ),
        }
        return {
            "scored_denominator": scored,
            "overall": overall,
            "shares": shares,
            "by_abstention": by_abstention,
            "by_question_type": {
                qtype: cells for qtype, cells in sorted(by_type.items())
            },
            "hit_criteria": criterion_notes,
            "excluded_from_cells": excluded,
        }

    def _budget_composition(
        self, prepared: dict[str, Any], raws: dict[str, Any]
    ) -> dict[str, Any]:
        handles = list(self.manifest["sample_ids"])
        rows: list[dict[str, Any]] = []
        for handle in handles:
            raw = raws.get(handle)
            prep = prepared.get(handle)
            counting = prep.prepared.counting_mode if prep is not None else None
            returned = len(raw.evidence) if raw is not None else None
            retained = len(prep.prepared.items) if prep is not None else None
            dropped = (
                len(prep.prepared.dropped_raw_indices) if prep is not None else None
            )
            truncated_units = (
                sum(1 for i in prep.prepared.items if i.truncated)
                if prep is not None
                else None
            )
            token_count = prep.prepared.token_count if prep is not None else None
            text_tokens = (
                prep.prepared.text_token_count if prep is not None else None
            )
            format_tokens = (
                token_count - text_tokens
                if token_count is not None and text_tokens is not None
                else None
            )
            truncated_removed: int | None = None
            dropped_text: int | None = None
            if prep is not None and raw is not None and counting == "test":
                by_index = {i: ev for i, ev in enumerate(raw.evidence)}
                truncated_removed = 0
                for item in prep.prepared.items:
                    if item.truncated and item.raw_index in by_index:
                        truncated_removed += len(by_index[item.raw_index].text) - len(
                            item.evidence.text
                        )
                dropped_text = sum(
                    len(by_index[i].text)
                    for i in prep.prepared.dropped_raw_indices
                    if i in by_index
                )
            rows.append(
                {
                    "sample_handle": handle,
                    "returned_units": returned,
                    "retained_units": retained,
                    "dropped_units": dropped,
                    "truncated_units": truncated_units,
                    "token_count": token_count,
                    "text_tokens": text_tokens,
                    "format_tokens": format_tokens,
                    "truncated_removed_tokens": truncated_removed,
                    "dropped_text_tokens": dropped_text,
                }
            )

        def total() -> dict[str, Any]:
            keys = (
                "returned_units",
                "retained_units",
                "dropped_units",
                "truncated_units",
                "token_count",
                "text_tokens",
                "format_tokens",
                "truncated_removed_tokens",
                "dropped_text_tokens",
            )
            totals: dict[str, Any] = {"sample_handle": "TOTAL"}
            for key in keys:
                known = [row[key] for row in rows if row[key] is not None]
                totals[key] = sum(known) if known else None
            return totals

        first_prep = next((p.prepared for p in prepared.values() if p is not None), None)
        return {
            "counting_mode": first_prep.counting_mode if first_prep else None,
            "tokenizer_id": first_prep.tokenizer_id if first_prep else None,
            "budget": self.manifest["evidence_token_budget"],
            "rows": rows,
            "totals": total(),
            "note": (
                "诊断项：adapter 返回条数、进入 Reader 条数、被移除与截断的 "
                "tokens（test 计数模式按字符计）。失败阶段的样本相应字段为 "
                "null，不计入合计。碎片化返回的元数据开销见 "
                "format_tokens。"
            ),
        }

    def _usage_split(self, logs: dict[str, Any]) -> dict[str, Any]:
        """Usage by attempt kind: logical vs recovery, both in totals.

        Retry and replay (including resume re-runs) resource consumption
        counts into the run TOTAL and is reported SEPARATELY from the
        logical operation usage (issue #5 AC5). Unknown quantities stay
        unknown: a bucket sums only reported values and also reports how
        many attempts carried usage at all.
        """

        class _Agg:
            def __init__(self) -> None:
                self.attempts = 0
                self.with_usage = 0
                self.input_tokens = 0
                self.output_tokens = 0
                self.llm_calls = 0

            def add(self, usage: Any) -> None:
                self.attempts += 1
                if usage is None:
                    return
                self.with_usage += 1
                if usage.input_tokens is not None:
                    self.input_tokens += usage.input_tokens
                if usage.output_tokens is not None:
                    self.output_tokens += usage.output_tokens
                if usage.llm_call_count is not None:
                    self.llm_calls += usage.llm_call_count

        buckets = {"logical": _Agg(), "retry": _Agg(), "replay": _Agg()}
        total = _Agg()
        for log in logs.values():
            if log is None:
                continue
            for entry in log.entries:
                kind = getattr(entry, "attempt_kind", "logical") or "logical"
                bucket = buckets.get(kind, buckets["logical"])
                bucket.add(entry.usage)
                total.add(entry.usage)

        def render(agg: _Agg) -> dict[str, Any]:
            return {
                "attempts": agg.attempts,
                "attempts_with_usage": agg.with_usage,
                "input_tokens": agg.input_tokens,
                "output_tokens": agg.output_tokens,
                "llm_call_count": agg.llm_calls,
            }

        return {
            "logical": render(buckets["logical"]),
            "retry": render(buckets["retry"]),
            "replay": render(buckets["replay"]),
            "total": render(total),
            "note": (
                "logical=按计划首次执行的调用；retry=有限重试；replay=隔离重放"
                "与断点补跑；total=run 总量（logical+retry+replay）。"
                "未上报用量的尝试只计入 attempts，不补零。"
            ),
        }

    def _cost_model(self, results: list[Any], logs: dict[str, Any]) -> dict[str, Any]:
        planned: dict[str, int] = {}
        for counts in self.manifest["plan_counts"].values():
            for key, value in counts.items():
                planned[key] = planned.get(key, 0) + value
        method_entries: dict[str, list[Any]] = {}
        for log in logs.values():
            if log is None:
                continue
            for entry in log.entries:
                method_entries.setdefault(entry.method, []).append(entry)
        interesting = (
            "ingest",
            "await_ready",
            "open",
            "close",
            "retrieve",
            "reader.answer",
            "judge.evaluate",
        )
        unit_costs = []
        for method in interesting:
            entries = method_entries.get(method, [])
            elapsed = [
                float(e.elapsed_ms) for e in entries if e.elapsed_ms is not None
            ]
            inputs = [
                e.usage.input_tokens
                for e in entries
                if e.usage is not None and e.usage.input_tokens is not None
            ]
            outputs = [
                e.usage.output_tokens
                for e in entries
                if e.usage is not None and e.usage.output_tokens is not None
            ]
            unit_costs.append(
                {
                    "method": method,
                    "call_count": len(entries),
                    "mean_elapsed_ms": _round(_mean(elapsed), 3),
                    "mean_input_tokens": _round(_mean([float(v) for v in inputs])),
                    "mean_output_tokens": _round(_mean([float(v) for v in outputs])),
                }
            )
        actual_calls = {
            method: len(entries) for method, entries in sorted(method_entries.items())
        }
        usage_totals = self._usage_split(logs)
        planned_key_for_method = {
            "ingest": "ingest_calls",
            "await_ready": "await_ready_calls",
            "open": "open_calls",
            "close": "close_calls",
            "retrieve": "retrieve_calls",
            "reader.answer": "reader_calls",
            "judge.evaluate": "judge_calls",
        }
        estimated: dict[str, float] = {}
        for unit in unit_costs:
            key = planned_key_for_method[unit["method"]]
            count = planned.get(key, 0)
            if unit["mean_elapsed_ms"] is not None:
                estimated[f"{unit['method']}.elapsed_ms"] = _round(
                    unit["mean_elapsed_ms"] * count, 3
                )
            if unit["mean_input_tokens"] is not None:
                estimated[f"{unit['method']}.input_tokens"] = _round(
                    unit["mean_input_tokens"] * count
                )
            if unit["mean_output_tokens"] is not None:
                estimated[f"{unit['method']}.output_tokens"] = _round(
                    unit["mean_output_tokens"] * count
                )
        return {
            "planned_calls": dict(sorted(planned.items())),
            "actual_calls": actual_calls,
            "unit_costs": unit_costs,
            "estimated_totals": dict(sorted(estimated.items())),
            "usage_totals": usage_totals,
            "estimated": True,
            "note": (
                "planned_calls 是配置的函数（run.json plan_counts 求和）；"
                "unit_costs 为 fake 路径实测单位成本（均值）；"
                "estimated_totals = 单位成本 × 计划调用次数，明确为估算值，"
                "不与实测混写。"
            ),
        }

    def _failure_rows(
        self, results: list[Any], logs: dict[str, Any]
    ) -> list[dict[str, Any]]:
        rows = []
        for result in results:
            if result.qa_status not in ("failed", "invalid_input", "pending"):
                continue
            error_code = None
            error_message = None
            log = logs.get(result.sample_handle)
            candidates = []
            if log is not None:
                candidates = [
                    e
                    for e in log.entries
                    if e.error is not None
                    and (e.stage == result.failed_stage or e.method == "dataset.get_scoring_data")
                ]
            if candidates:
                last = candidates[-1]
                error_code = last.error.code
                error_message = last.error.message
            rows.append(
                {
                    "sample_handle": result.sample_handle,
                    "qa_status": result.qa_status,
                    "failed_stage": result.failed_stage,
                    "error_code": error_code,
                    "error_message": error_message,
                }
            )
        return rows


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt(value: float | None) -> str:
    if value is None:
        return "N/A"
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_markdown(report: SummaryReport) -> str:
    if report["header"].get("suite") == "operations":
        return _render_markdown_operations(report)
    return _render_markdown_qa(report)


def _render_markdown_operations(report: SummaryReport) -> str:
    header = report["header"]
    statuses = report["statuses"]
    lines: list[str] = []
    lines.append("# Memory Eval 操作能力套件报告")
    lines.append("")
    lines.append(
        _table(
            ["字段", "值"],
            [
                ["run ID", header["run_id"]],
                ["配置", f"{header['config_name']}（指纹 {header['config_fingerprint'][:12]}…）"],
                ["数据计划", f"{header['dataset_plan']} / {header['sample_plan_id']}"],
                ["memory", f"{header['memory_name']}（{header['memory_baseline_kind']}）"],
                ["指标注册表", f"v{header['metrics_registry_version']}（{header['metrics_registry_content_version']}）"],
            ],
        )
    )
    lines.append("")
    lines.append("## 状态计数")
    lines.append("")
    lines.append(
        _table(
            ["口径", "数量", "说明"],
            [
                ["计划检查", str(statuses["planned"]), "运行前固定的检查清单"],
                ["passed", str(statuses["passed"]), "确定性断言全部通过"],
                ["failed", str(statuses["failed"]), "断言失败或声明能力但无法给出明确状态"],
                ["not_supported", str(statuses["not_supported"]), "未声明所需能力；不进通过率分母"],
                ["pending", str(statuses["pending"]), "未到终态（中间进度）"],
            ],
        )
    )
    lines.append("")
    lines.append(f"## 正式指标（指标注册表 v{header['metrics_registry_version']}）")
    lines.append("")
    lines.append(
        _table(
            ["metric_id", "状态", "数值", "分母 / 说明"],
            [
                [
                    m["metric_id"],
                    "computed" if m["status"] == "computed" else "N/A",
                    _fmt(m["value"]) if m["status"] == "computed" else "—",
                    m["reason"] or m["denominator"],
                ]
                for m in report["metrics"]
            ],
        )
    )
    lines.append("")
    lines.append("## 逐项检查")
    lines.append("")
    checks = report.get("checks", [])
    lines.append(
        _table(
            ["检查", "状态", "断言数", "缺失能力", "失败阶段", "原因"],
            [
                [
                    c["check_id"],
                    c["status"],
                    str(len(c["assertions"])),
                    ", ".join(c["missing_capabilities"]) or "—",
                    c["failed_stage"] or "—",
                    (c["reason"] or "—")[:60],
                ]
                for c in checks
            ],
        )
    )
    lines.append("")
    lines.append("断言明细（程序化断言，不经 LLM judge）：")
    lines.append("")
    for check in checks:
        lines.append(f"- **{check['check_id']}**（{check['status']}）")
        for a in check["assertions"]:
            mark = "✓" if a["passed"] else "✗"
            lines.append(f"  - {mark} {a['name']}: 期望 {a['expected']}，实际 {a['observed']}")
    lines.append("")
    lines.append("## 限制")
    lines.append("")
    for note in report["limitations"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _render_model_versions_lines(header: dict[str, Any]) -> list[str]:
    """Markdown rows for the four-identifier version record + probes."""
    lines: list[str] = []
    versions = header.get("model_versions")
    if versions is None:
        lines.append(
            "- 版本记录缺失（该 run 未写入 model_versions.json；新 run 均会写入）"
        )
        return lines
    for role in ("reader", "judge"):
        record = versions.get(role, {})
        lines.append(
            "- {role}：别名 {alias}；响应 model 字段 {resp}；厂商标注 {vendor}"
            "（{when}）；运行日期 {date}".format(
                role=role,
                alias=record.get("alias"),
                resp=record.get("response_model") or "未观测",
                vendor=record.get("vendor_documented_version") or "未记录",
                when=record.get("vendor_documented_on") or "未记录",
                date=record.get("run_date"),
            )
        )
    probe = versions.get("probe")
    if probe is None:
        lines.append("- 漂移探测：本 run 未配置探测集")
    else:
        ok = sum(1 for c in probe.get("calls", []) if not c.get("error"))
        digest = str(probe.get("probe_digest", ""))[:12]
        lines.append(
            "- 漂移探测：{sid}（{ok}/{n} 条成功，digest {digest}…）；"
            "两次 run 探测输出明显不同即判为模型变更".format(
                sid=probe.get("probe_set_id"),
                ok=ok,
                n=len(probe.get("calls", [])),
                digest=digest,
            )
        )
    lines.append(
        "- 代码版本：{code}".format(
            code=header.get("code_version") or versions.get("code_version") or "unknown"
        )
    )
    return lines


def _render_markdown_qa(report: SummaryReport) -> str:
    header = report["header"]
    statuses = report["statuses"]
    role = header.get("baseline_control_role", {})
    lines: list[str] = []
    lines.append("# Memory Eval 运行汇总报告")
    lines.append("")
    lines.append(
        _table(
            ["字段", "值"],
            [
                ["run ID", header["run_id"]],
                ["配置", f"{header['config_name']}（指纹 {header['config_fingerprint'][:12]}…）"],
                ["数据计划", f"{header['dataset_plan']} / {header['sample_plan_id']}"],
                ["证据预算", f"{header['evidence_token_budget']} tokens（{header['counting_mode']} 计数，tokenizer {header.get('tokenizer_id', '—')}）"],
                ["memory", f"{header['memory_name']}（{header['memory_baseline_kind']}）"],
                ["对照类型", f"{role.get('role', '—')}（等预算：{'是' if role.get('equal_budget') else '否'}）——{role.get('note', '')}"],
                ["reader / judge", f"{header['reader_model']} / {header['judge_model']}（协议 {header['judge_protocol_id']}）"],
                ["指标注册表", f"v{header['metrics_registry_version']}（{header['metrics_registry_content_version']}）"],
            ],
        )
    )
    lines.append("")
    lines.append("## 模型版本与漂移探测（四项标识）")
    lines.append("")
    lines.extend(_render_model_versions_lines(header))
    lines.append("## 状态计数与分母")
    lines.append("")
    failed_stages = ", ".join(
        f"{stage}={count}" for stage, count in sorted(statuses["failed_stages"].items())
    ) or "无"
    lines.append(
        _table(
            ["口径", "数量", "说明"],
            [
                ["计划题目 |P|", str(statuses["planned"]), "运行前固定的题目集合"],
                ["scored", str(statuses["scored"]), "回答已成功评分（可对可错）"],
                ["failed", str(statuses["failed"]), f"失败阶段：{failed_stages}"],
                ["context_exceeded", str(statuses["context_exceeded"]), "预检不可运行，单列不并入答错"],
                ["invalid_input", str(statuses["invalid_input"]), "数据校验错误，不静默排除"],
                ["pending", str(statuses["pending"]), "未到终态（中间进度）"],
                ["拒答题（计划/已评分）", f"{statuses['abstention_planned']} / {statuses['abstention_scored']}", "recall 为 N/A 但参与回答评分"],
            ],
        )
    )
    lines.append("")
    lines.append(f"## 正式指标（指标注册表 v{header['metrics_registry_version']}）")
    lines.append("")
    lines.append(
        _table(
            ["metric_id", "状态", "数值", "分母 / 说明"],
            [
                [
                    m["metric_id"],
                    "computed" if m["status"] == "computed" else "N/A",
                    _fmt(m["value"]) if m["status"] == "computed" else "—",
                    m["reason"] or m["denominator"],
                ]
                for m in report["metrics"]
            ],
        )
    )
    lines.append("")
    lines.append("## 检索 × 问答 2×2 联合归因")
    lines.append("")
    attribution = report["attribution"]
    overall = attribution["overall"]
    shares = attribution["shares"]
    scored = attribution["scored_denominator"]
    lines.append(
        _table(
            ["", "回答正确", "回答错误"],
            [
                [
                    "证据命中",
                    f"hit_correct = {overall['hit_correct']}（{_fmt(shares['hit_correct'])}）",
                    f"hit_wrong = {overall['hit_wrong']}（{_fmt(shares['hit_wrong'])}）",
                ],
                [
                    "证据未命中",
                    f"miss_correct = {overall['miss_correct']}（{_fmt(shares['miss_correct'])}）",
                    f"miss_wrong = {overall['miss_wrong']}（{_fmt(shares['miss_wrong'])}）",
                ],
            ],
        )
    )
    lines.append("")
    lines.append(f"分母：scored={scored}。命中口径：" + "；".join(attribution["hit_criteria"]))
    excluded = attribution["excluded_from_cells"]
    excluded_text = ", ".join(f"{k}={v}" for k, v in excluded.items() if v) or "无"
    lines.append("不进四格的状态（单列，不并入答错）：" + excluded_text)
    lines.append("")
    by_abstention = attribution["by_abstention"]
    lines.append(
        _table(
            ["子集", "hit_correct", "hit_wrong", "miss_correct", "miss_wrong"],
            [
                [
                    "拒答题",
                    str(by_abstention["true"]["hit_correct"]),
                    str(by_abstention["true"]["hit_wrong"]),
                    str(by_abstention["true"]["miss_correct"]),
                    str(by_abstention["true"]["miss_wrong"]),
                ],
                [
                    "非拒答题",
                    str(by_abstention["false"]["hit_correct"]),
                    str(by_abstention["false"]["hit_wrong"]),
                    str(by_abstention["false"]["miss_correct"]),
                    str(by_abstention["false"]["miss_wrong"]),
                ],
                *[
                    [
                        qtype,
                        str(cells["hit_correct"]),
                        str(cells["hit_wrong"]),
                        str(cells["miss_correct"]),
                        str(cells["miss_wrong"]),
                    ]
                    for qtype, cells in attribution["by_question_type"].items()
                ],
            ],
        )
    )
    lines.append("")
    lines.append("## 证据预算构成（诊断）")
    budget = report["budget"]
    lines.append("")
    lines.append(
        f"计数模式 {budget['counting_mode']}（tokenizer {budget['tokenizer_id']}），"
        f"预算 {budget['budget']} tokens。{budget['note']}"
    )
    lines.append("")
    rows = budget["rows"] + [budget["totals"]]
    lines.append(
        _table(
            [
                "样本",
                "返回",
                "进 Reader",
                "移除",
                "截断单元",
                "总 tokens",
                "文本 tokens",
                "格式 tokens",
                "截断移除",
                "移除文本",
            ],
            [
                [
                    row["sample_handle"],
                    *("—" if row[k] is None else str(row[k]) for k in (
                        "returned_units",
                        "retained_units",
                        "dropped_units",
                        "truncated_units",
                        "token_count",
                        "text_tokens",
                        "format_tokens",
                        "truncated_removed_tokens",
                        "dropped_text_tokens",
                    )),
                ]
                for row in rows
            ],
        )
    )
    lines.append("")
    lines.append("## 规模与成本模型")
    cost = report["cost_model"]
    lines.append("")
    calibration = report.get("token_calibration")
    if calibration:
        lines.append(calibration["note"])
        lines.append("")
        lines.append(
            _table(
                ["口径", "数值"],
                [
                    ["带校准的 reader 调用", str(calibration["calls_with_calibration"])],
                    ["带服务端 usage 的调用", str(calibration["calls_with_server_usage"])],
                    ["计数不一致（差值≠0）", str(calibration["mismatch_calls"])],
                    ["平均 |差值| tokens", _fmt(calibration["mean_abs_delta_tokens"])],
                    ["最大 |差值| tokens", _fmt(calibration["max_abs_delta_tokens"])],
                    ["计数模式", "、".join(calibration["counting_modes"]) or "—"],
                ],
            )
        )
        lines.append("")
    precheck = header.get("context_precheck") or {}
    if precheck.get("enabled"):
        lines.append(
            "上下文预检：窗口 {win} tokens（消息格式开销 {fo}、输出预留 {res}）；"
            "构成 = {comp}；超限题记 context_exceeded 单列。".format(
                win=precheck.get("context_window_tokens"),
                fo=precheck.get("format_overhead_tokens"),
                res=precheck.get("output_reserve_tokens"),
                comp=precheck.get("components"),
            )
        )
    else:
        lines.append("上下文预检：" + precheck.get("note", "未启用"))
    lines.append("")
    lines.append(cost["note"])
    lines.append("")
    planned = cost["planned_calls"]
    actual = cost["actual_calls"]
    lines.append(
        _table(
            ["调用", "计划次数", "实际次数"],
            [
                [method, str(planned.get(key, 0)), str(actual.get(method, 0))]
                for method, key in (
                    ("ingest", "ingest_calls"),
                    ("await_ready", "await_ready_calls"),
                    ("open", "open_calls"),
                    ("close", "close_calls"),
                    ("retrieve", "retrieve_calls"),
                    ("reader.answer", "reader_calls"),
                    ("judge.evaluate", "judge_calls"),
                )
            ],
        )
    )
    lines.append("")
    lines.append(
        _table(
            ["方法", "调用数", "均耗时 ms", "均输入 tokens", "均输出 tokens"],
            [
                [
                    unit["method"],
                    str(unit["call_count"]),
                    _fmt(unit["mean_elapsed_ms"]),
                    _fmt(unit["mean_input_tokens"]),
                    _fmt(unit["mean_output_tokens"]),
                ]
                for unit in cost["unit_costs"]
            ],
        )
    )
    lines.append("")
    lines.append("外推总量（估算 = 单位成本 × 计划次数）：")
    lines.append("")
    lines.append(
        _table(
            ["项", "估算值"],
            [[key, _fmt(value)] for key, value in cost["estimated_totals"].items()],
        )
    )
    lines.append("")
    lines.append("用量拆分（重试与重放计入 total 并单列）：")
    lines.append("")
    usage = report["cost_model"]["usage_totals"]
    lines.append(
        _table(
            ["类别", "尝试数", "有用量尝试", "输入 tokens", "输出 tokens", "LLM 调用"],
            [
                [
                    label,
                    str(bucket["attempts"]),
                    str(bucket["attempts_with_usage"]),
                    _fmt(bucket["input_tokens"]),
                    _fmt(bucket["output_tokens"]),
                    _fmt(bucket["llm_call_count"]),
                ]
                for label, bucket in (
                    ("logical（按计划首次调用）", usage["logical"]),
                    ("retry（有限重试）", usage["retry"]),
                    ("replay（隔离重放/断点补跑）", usage["replay"]),
                    ("total（run 总量）", usage["total"]),
                )
            ],
        )
    )
    lines.append("")
    lines.append(usage["note"])
    if report["failures"]:
        lines.append("")
        lines.append("## 失败与无效样本")
        lines.append("")
        lines.append(
            _table(
                ["样本", "状态", "失败阶段", "错误码", "错误信息"],
                [
                    [
                        row["sample_handle"],
                        row["qa_status"],
                        row["failed_stage"] or "—",
                        row["error_code"] or "—",
                        row["error_message"] or "—",
                    ]
                    for row in report["failures"]
                ],
            )
        )
    lines.append("")
    lines.append("## 限制")
    lines.append("")
    for note in report["limitations"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)
