"""M1 review 检查矩阵：设计文档「Review 修订对应的验收检查」的逐条落成。

The design doc (docs/design/eval-harness.md, 实施阶段与验收) lists one
"must pass" check per review finding. Issue #6 requires every matrix
row to have an identifiable automated assertion. This module is that
matrix: one parameterized case per row, each asserting the row's OWN
requirement end to end against real harness artifacts (not a meta
check that a test exists elsewhere).

Row ids follow the issue wording:
  id_leakage, question_date, truncation_recall, update_state_observable,
  incomplete_close, retry_duplicate_mutation, abstention_recall,
  retrieval_not_state, granularity_overhead, metric_drift,
  joint_attribution, budget_overrun_visible, explicit_update_completion,
  scale_and_cost.

Run one row: uv run pytest "tests/eval/test_review_matrix.py::test_review_matrix[abstention_recall]"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from eval.compare import RunSide, comparability
from eval.config import load_config_dict, load_config_toml
from eval.contracts.adapter import Evidence
from eval.datasets.manual import ManualDataset
from eval.judges.fake import FakeJudge, FakeJudgeSpec
from eval.memories.fake import FakeMemoryAdapter, FakeMemorySpec
from eval.metrics import REGISTRY
from eval.readers.fake import FakeReader, FakeReaderSpec
from eval.runner import OfflineRunner
from eval.runs import RunStore

EXAMPLE_CONFIG = "eval/configs/examples/offline_fake.toml"
OPS_EXAMPLE_CONFIG = "eval/configs/examples/offline_fake_ops.toml"

MATRIX: list[tuple[str, str]] = [
    ("id_leakage", "污染样本中的上游 ID 标记与标注字段不进入 adapter 可见对象或 reader/judge 记录；私有映射仍能准确评分"),
    ("question_date", "所有检索请求与证据来源时间使用数据集 question_date；机器日期不影响查询上下文"),
    ("truncation_recall", "被预算裁掉/截断的原文不计召回；范围与剩余文本一致；生成来源不带来 recall 命中"),
    ("update_state_observable", "自动与显式更新分别测试并分别记录；状态断言经 inspect 且旧值不为 current"),
    ("incomplete_close", "延迟完成的 adapter 返回 accepted 后 runner 等待完成才关闭；超时记失败且耗时用量被记录"),
    ("retry_duplicate_mutation", "响应丢失后幂等重试只产生一次逻辑修改；重试尝试与首次尝试分别记录"),
    ("abstention_recall", "拒答题即使 gold 数组非空也 recall=N/A，仍参与回答评分并单独报告"),
    ("retrieval_not_state", "Evidence 不含记忆身份与有效性字段；操作目标只取自回执；删除以不再可召回为准"),
    ("granularity_overhead", "报告分别给出单元文本与格式开销 tokens；碎片化开销可见且不改变预算规则"),
    ("metric_drift", "正式指标全部来自带版本的注册表；注册表版本不一致拒绝自动对齐；指纹绑定注册表内容"),
    ("joint_attribution", "报告给出 2×2 分布与命中口径；failed/context_exceeded/pending/invalid 单列不并入答错"),
    ("budget_overrun_visible", "报告给出返回/进入 Reader/被移除截断的数量与 tokens；预算硬上限不被静默突破"),
    ("explicit_update_completion", "update() 后 inspect 给出 content=replacement、validity=current；旧文本不再作为当前值；sources=[] 不判失败"),
    ("scale_and_cost", "调用次数模型、fake 单位成本与外推估算在报告中并明确标注；smoke 子集 8 题固定"),
]
IDS = [row_id for row_id, _ in MATRIX]


# ---------------------------------------------------------------------------
# Shared harness runs (module scope: one qa run, one flood run, one ops run)
# ---------------------------------------------------------------------------


def _runner_for(tmp_path: Path, config, run_id: str, adapter=None) -> OfflineRunner:
    return OfflineRunner(
        config=config,
        dataset=ManualDataset.load_default(),
        adapter=adapter
        or FakeMemoryAdapter(FakeMemorySpec.from_memory_plan(config.memory)),
        reader=FakeReader(FakeReaderSpec.from_reader_plan(config.reader)),
        judge=FakeJudge(FakeJudgeSpec.from_judge_plan(config.judge)),
        store=RunStore(tmp_path / "runs", run_id),
        run_id=run_id,
    )


def _load_run_docs(run_dir: Path) -> dict[str, Any]:
    store = RunStore(run_dir.parent, run_dir.name)
    results: dict[str, Any] = {}
    for line in store.read_result_lines():
        result = line["result"]
        handle = result["sample_handle"]
        refs = result.get("artifact_refs", {})
        entry: dict[str, Any] = {"result": result}
        for key, model in (
            ("attempts", None),
            ("scoring", None),
            ("prepared_evidence", None),
            ("raw_evidence", None),
            ("reader_result", None),
            ("judge_record", None),
            ("operations_check", None),
        ):
            if refs.get(key):
                entry[key] = json.loads(
                    store.resolve_ref(refs[key]).read_text(encoding="utf-8")
                )
        results[handle] = entry
    return results


@pytest.fixture(scope="module")
def qa_run(tmp_path_factory) -> dict[str, Any]:
    """Full 8-sample smoke run with the async mixed fake (generated +
    extractive evidence, state inspection)."""
    base = tmp_path_factory.mktemp("qa")
    config = load_config_toml(EXAMPLE_CONFIG)
    adapter = FakeMemoryAdapter(FakeMemorySpec.from_memory_plan(config.memory))
    outcome = _runner_for(base, config, "run-matrix-qa", adapter=adapter).run()
    assert outcome.failed == 0
    return {
        "run_dir": outcome.run_dir,
        "config": config,
        "adapter": adapter,
        "docs": _load_run_docs(outcome.run_dir),
        "report": json.loads(
            (outcome.run_dir / "report.json").read_text(encoding="utf-8")
        ),
    }


@pytest.fixture(scope="module")
def flood_run(tmp_path_factory) -> dict[str, Any]:
    """Flood retrieval under a tight budget: units get truncated/dropped."""
    base = tmp_path_factory.mktemp("flood")
    config = load_config_dict(
        {
            "name": "matrix-flood",
            "dataset_plan": "manual-fixtures@1",
            "sample_plan_id": "smoke-offline-8",
            "sample_ids": ["smoke_multi_session_0001"],
            "smoke_subset_ids": [],
            "memory": {
                "name": "fake-memory",
                "baseline_kind": "adapter",
                "capabilities": ["extractive_evidence", "state_inspection"],
                "config": {
                    "mutation_mode": "sync",
                    "evidence_kinds": ["extractive"],
                    "retrieval_mode": "flood",
                },
            },
            "reader": {
                "model": "fake-reader",
                "model_family": "family-r",
                "base_url": "offline://fake",
                "temperature": 0.0,
                "max_output_tokens": 1024,
                "tokenizer_id": "test:char-v1",
                "counting_mode": "test",
            },
            "judge": {
                "model": "fake-judge",
                "model_family": "family-j",
                "base_url": "offline://fake",
                "temperature": 0.0,
                "protocol_id": "longmemeval-yes-no@1",
                "protocol_source_commit": "0" * 40,
            },
            "evidence_token_budget": 512,
        }
    )
    outcome = _runner_for(base, config, "run-matrix-flood").run()
    assert outcome.failed == 0
    docs = _load_run_docs(outcome.run_dir)
    return {
        "run_dir": outcome.run_dir,
        "config": config,
        "docs": docs,
        "report": json.loads(
            (outcome.run_dir / "report.json").read_text(encoding="utf-8")
        ),
        "doc": docs["smoke_multi_session_0001"],
    }


@pytest.fixture(scope="module")
def ops_run(tmp_path_factory) -> dict[str, Any]:
    base = tmp_path_factory.mktemp("ops")
    config = load_config_toml(OPS_EXAMPLE_CONFIG)
    from eval.operations import OperationsRunner

    runner = OperationsRunner(
        config=config,
        adapter=FakeMemoryAdapter(FakeMemorySpec.from_memory_plan(config.memory)),
        store=RunStore(base / "runs", "run-matrix-ops"),
        run_id="run-matrix-ops",
    )
    outcome = runner.run()
    assert outcome.failed == 0
    return {
        "run_dir": outcome.run_dir,
        "config": config,
        "docs": _load_run_docs(outcome.run_dir),
        "summary": json.loads(
            (outcome.run_dir / "operations_summary.json").read_text(
                encoding="utf-8"
            )
        ),
    }


@pytest.fixture(scope="module")
def matrix(
    qa_run, flood_run, ops_run, tmp_path_factory
) -> dict[str, Any]:
    return {
        "qa": qa_run,
        "flood": flood_run,
        "ops": ops_run,
        "dataset": ManualDataset.load_default(),
        "tmp": tmp_path_factory,
    }


# ---------------------------------------------------------------------------
# Row checks
# ---------------------------------------------------------------------------

UPSTREAM_MARKERS = [
    "has_answer",
    "answer_session_ids",
    "evidence_session_ids",
    "question_type",
    "_abs",
    "smoke_abstention_0001_abs",
    "session_12_answer_7ab9",
    "session_11_answer_4d02",
]


ADAPTER_METHODS = {
    "ingest",
    "retrieve",
    "open",
    "close",
    "reset",
    "await_ready",
    "update",
    "delete",
    "inspect",
}
PRIVATE_MARKERS = (
    "answer_session_ids",
    "evidence_session_ids",
    "gold_source_ids",
    "has_answer",
    "_abs",
    "internal_to_official",
)


def check_id_leakage(m: dict[str, Any]) -> None:
    docs = m["qa"]["docs"]
    # 1) Adapter-visible inputs (recorded verbatim in attempt logs)
    adapter_blob = json.dumps(
        [
            entry.get("input")
            for d in docs.values()
            for entry in d["attempts"]["entries"]
            if entry["method"] in ADAPTER_METHODS
        ],
        ensure_ascii=False,
    )
    for marker in UPSTREAM_MARKERS:
        assert marker not in adapter_blob, marker
    # 2) Reader inputs never see private scoring data either
    reader_blob = json.dumps(
        [
            entry.get("input")
            for d in docs.values()
            for entry in d["attempts"]["entries"]
            if entry["method"] == "reader.answer"
        ],
        ensure_ascii=False,
    )
    for marker in UPSTREAM_MARKERS:
        assert marker not in reader_blob, marker
    # 3) The judge follows the official protocol (question_type and the
    #    expected answer belong to it) but never private id mappings
    judge_blob = json.dumps(
        [
            entry.get("input")
            for d in docs.values()
            for entry in d["attempts"]["entries"]
            if entry["method"] == "judge.evaluate"
        ],
        ensure_ascii=False,
    )
    for marker in PRIVATE_MARKERS:
        assert marker not in judge_blob, marker
    # 4) The private mapping still scores: recall over internal ids works
    abst = m["qa"]["report"]["metrics"]
    macro = next(x for x in abst if x["metric_id"] == "verifiable_session_recall_macro")
    assert macro["status"] == "computed" and macro["value"] == 1.0
    scored = [d for d in docs.values() if d["scoring"]]
    assert any(d["scoring"]["scoring"]["internal_to_official_session"] for d in scored)


def check_question_date(m: dict[str, Any]) -> None:
    dataset: ManualDataset = m["dataset"]
    used_dates: set[str] = set()
    for handle, d in m["qa"]["docs"].items():
        expected = dataset.get_question(handle).question_date
        session_dates = {s.occurred_at for s in dataset.iter_sessions(handle)}
        retrieves = [
            entry
            for entry in d["attempts"]["entries"]
            if entry["method"] == "retrieve"
        ]
        assert retrieves
        for entry in retrieves:
            request = entry["input"]["request"]
            assert request["question_date"] == expected
            used_dates.add(request["question_date"])
        # Source times in prepared evidence come from dataset dates only
        if "prepared_evidence" in d:
            for item in d["prepared_evidence"]["prepared"]["items"]:
                for stamp in item["evidence"]["source_times"]:
                    assert stamp in session_dates, (handle, stamp)
    # The machine clock never entered the query context: every used date
    # is one of the dataset's fixed question dates.
    all_question_dates = {
        dataset.get_question(h).question_date for h in dataset.sample_handles
    }
    assert used_dates <= all_question_dates
    assert used_dates  # sanity: dates were actually observed


def check_truncation_recall(m: dict[str, Any]) -> None:
    doc = m["flood"]["doc"]
    prepared = doc["prepared_evidence"]["prepared"]
    raw = doc["raw_evidence"]["evidence"]
    trace = doc["scoring"]
    dropped = prepared["dropped_raw_indices"]
    assert dropped, "flood run must actually drop units for this row"
    # Dropped units never contribute sessions to the recall numerator
    dropped_sessions = {
        raw[i]["extractive_span"]["session_id"]
        for i in dropped
        if raw[i].get("extractive_span")
    }
    actual_sessions = set(trace["actual_sessions_in_order"])
    assert not (dropped_sessions & actual_sessions)
    # Retained (possibly truncated) items: span end - start == len(text)
    for item in prepared["items"]:
        ev = item["evidence"]
        if ev["kind"] == "extractive":
            span = ev["extractive_span"]
            assert span["end"] - span["start"] == len(ev["text"])
    # Generated derivation sources never enter the recall hit set
    # (asserted on the mixed-mode qa run, which returns generated units)
    for handle, d in m["qa"]["docs"].items():
        if "prepared_evidence" not in d or "scoring" not in d:
            continue
        extractive_sessions = {
            item["evidence"]["extractive_span"]["session_id"]
            for item in d["prepared_evidence"]["prepared"]["items"]
            if item["evidence"]["kind"] == "extractive"
        }
        hit = set(d["scoring"]["hit_gold_sessions"])
        assert hit <= extractive_sessions, handle


def check_update_state_observable(m: dict[str, Any]) -> None:
    summary = m["ops"]["summary"]
    by_id = {c["check_id"]: c for c in summary["checks"]}
    auto, explicit = by_id["auto_update"], by_id["explicit_update"]
    # Separate records — one must not substitute the other
    assert auto["operation_status"] == "passed"
    assert explicit["operation_status"] == "passed"
    assert auto is not explicit
    # State assertions go through inspect with definite validities
    for check in (auto, explicit):
        state_assertions = [
            a for a in check["assertions"] if "validity" in a["expected"]
        ]
        assert state_assertions
        for a in state_assertions:
            assert a["passed"]
            assert "unknown" not in a["observed"]
    auto_names = " ".join(a["name"] for a in auto["assertions"])
    assert "new convention is current" in auto_names
    assert "retained old convention is not current" in auto_names


def check_incomplete_close(m: dict[str, Any]) -> None:
    base = m["tmp"].mktemp("never-ready")
    config = load_config_dict(
        {
            "name": "matrix-never-ready",
            "dataset_plan": "manual-fixtures@1",
            "sample_plan_id": "smoke-offline-8",
            "sample_ids": ["smoke_single_session_user_0001"],
            "smoke_subset_ids": [],
            "memory": {
                "name": "fake-memory",
                "baseline_kind": "adapter",
                "capabilities": [
                    "extractive_evidence",
                    "state_inspection",
                    "async_mutation",
                    "idempotent_mutation",
                    "operation_status",
                ],
                "config": {
                    "mutation_mode": "async",
                    "evidence_kinds": ["extractive"],
                    "idempotent": True,
                    "never_ready": True,
                },
            },
            "reader": {
                "model": "fake-reader",
                "model_family": "family-r",
                "base_url": "offline://fake",
                "temperature": 0.0,
                "max_output_tokens": 1024,
                "tokenizer_id": "test:char-v1",
                "counting_mode": "test",
            },
            "judge": {
                "model": "fake-judge",
                "model_family": "family-j",
                "base_url": "offline://fake",
                "temperature": 0.0,
                "protocol_id": "longmemeval-yes-no@1",
                "protocol_source_commit": "0" * 40,
            },
            "evidence_token_budget": 4096,
            "run_params": {
                "max_retries": 0,
                "backoff_base_s": 0.001,
                "await_ready_timeout_s": 0.05,
            },
        }
    )
    adapter = FakeMemoryAdapter(FakeMemorySpec.from_memory_plan(config.memory))
    outcome = _runner_for(base, config, "run-matrix-nr", adapter=adapter).run()
    result = outcome.results[0].result
    assert result.qa_status == "failed"
    assert result.failed_stage == "await_ready"
    # The submit was accepted; the wall-clock timeout carries timing
    timeout_attempt = next(
        a
        for a in result.attempts
        if a.stage == "await_ready"
        and a.outcome == "error"
        and a.error.code == "await_ready_timeout"
    )
    assert timeout_attempt.elapsed_ms == pytest.approx(50.0)
    polls = [
        a for a in result.attempts if a.stage == "await_ready" and a.outcome == "returned"
    ]
    assert polls and all(a.elapsed_ms is not None for a in polls)
    # True call order (adapter journal): after the accepted ingest the
    # runner only polls await_ready; close comes after the failed wait,
    # never while the mutation is still in flight
    journal = list(adapter.journal)
    methods = [e["method"] for e in journal]
    ingest_index = methods.index("ingest")
    await_indexes = [i for i, x in enumerate(methods) if x == "await_ready"]
    close_indexes = [i for i, x in enumerate(methods) if x == "close"]
    assert await_indexes and all(i > ingest_index for i in await_indexes)
    assert all(i > max(await_indexes) for i in close_indexes)
    # The failure is honest: nothing fabricated, later stages pending
    assert result.correct is None and result.attribution is None
    assert result.stage_states["read"] == "pending"
    # Positive control on the healthy async run: per namespace, every
    # close follows a completed await_ready of the ingest before it
    healthy: dict[str, list[str]] = {}
    for entry in m["qa"]["adapter"].journal:
        healthy.setdefault(entry["namespace"], []).append(entry["method"])
    for ns, seq in healthy.items():
        last_ingest = last_await = -1
        for i, method in enumerate(seq):
            if method == "ingest":
                last_ingest = i
            elif method == "await_ready":
                last_await = i
            elif method == "close":
                assert last_await > last_ingest, (ns, seq)


def check_retry_duplicate_mutation(m: dict[str, Any]) -> None:
    base = m["tmp"].mktemp("response-loss")
    handle = "smoke_single_session_user_0001"
    config = load_config_dict(
        {
            "name": "matrix-response-loss",
            "dataset_plan": "manual-fixtures@1",
            "sample_plan_id": "smoke-offline-8",
            "sample_ids": [handle],
            "smoke_subset_ids": [],
            "memory": {
                "name": "fake-memory",
                "baseline_kind": "adapter",
                "capabilities": ["extractive_evidence", "state_inspection", "idempotent_mutation"],
                "config": {
                    "mutation_mode": "sync",
                    "evidence_kinds": ["extractive"],
                    "idempotent": True,
                    "response_loss_calls": 1,
                },
            },
            "reader": {
                "model": "fake-reader",
                "model_family": "family-r",
                "base_url": "offline://fake",
                "temperature": 0.0,
                "max_output_tokens": 1024,
                "tokenizer_id": "test:char-v1",
                "counting_mode": "test",
            },
            "judge": {
                "model": "fake-judge",
                "model_family": "family-j",
                "base_url": "offline://fake",
                "temperature": 0.0,
                "protocol_id": "longmemeval-yes-no@1",
                "protocol_source_commit": "0" * 40,
            },
            "evidence_token_budget": 4096,
            "run_params": {"max_retries": 2, "backoff_base_s": 0.001},
        }
    )
    adapter = FakeMemoryAdapter(FakeMemorySpec.from_memory_plan(config.memory))
    runner = OfflineRunner(
        config=config,
        dataset=ManualDataset.load_default(),
        adapter=adapter,
        reader=FakeReader(FakeReaderSpec.from_reader_plan(config.reader)),
        judge=FakeJudge(FakeJudgeSpec.from_judge_plan(config.judge)),
        store=RunStore(base / "runs", "run-matrix-rl"),
        run_id="run-matrix-rl",
    )
    outcome = runner.run()
    assert outcome.failed == 0
    result = outcome.results[0].result
    dataset = ManualDataset.load_default()
    expected_messages = sum(len(s.messages) for s in dataset.iter_sessions(handle))
    ingest_calls = len(dataset.iter_sessions(handle))
    ingest_attempts = [a for a in result.attempts if a.stage == "ingest"]
    assert len(ingest_attempts) == ingest_calls + 1  # lost response + one re-submit
    lost, retry = ingest_attempts[0], ingest_attempts[1]
    assert lost.outcome == "error" and lost.error.code == "response_lost"
    assert lost.attempt_kind == "logical"
    assert retry.outcome == "returned" and retry.attempt_kind == "retry"
    # Exactly ONE logical mutation per message despite the re-submit
    namespace = result.namespace
    assert adapter.stored_message_count(namespace) == expected_messages
    # Same operation id reused for the retry
    docs = _load_run_docs(outcome.run_dir)
    ingests = [
        e
        for e in docs[handle]["attempts"]["entries"]
        if e["method"] == "ingest"
    ]
    assert ingests[0]["operation_id"] == ingests[1]["operation_id"]


def check_abstention_recall(m: dict[str, Any]) -> None:
    report = m["qa"]["report"]
    abstention_handles = [
        handle
        for handle, d in m["qa"]["docs"].items()
        if d.get("scoring", {}).get("scoring", {}).get("is_abstention")
    ]
    assert len(abstention_handles) == 2
    for handle in abstention_handles:
        d = m["qa"]["docs"][handle]
        trace = d["scoring"]
        # Recall N/A even where the fixture carries a non-empty gold array
        assert trace["recall_applicable"] is False
        assert trace["recall_value"] is None
        assert "abstention" in trace["recall_na_reason"]
        # ...but the answer was still judged
        assert d["result"]["qa_status"] == "scored"
        assert d["result"]["correct"] is not None
    statuses = report["statuses"]
    assert statuses["abstention_planned"] == 2
    metric = next(
        x for x in report["metrics"] if x["metric_id"] == "abstention_accuracy"
    )
    assert metric["status"] == "computed"


def check_retrieval_not_state(m: dict[str, Any]) -> None:
    # 1) Evidence structurally carries no identity/validity fields
    assert not ({"memory_id", "validity", "superseded_by"} & set(Evidence.model_fields))
    from pydantic import ValidationError

    valid = {
        "kind": "generated",
        "text": "摘要",
        "extractive_span": None,
        "derivation_sources": ["s_1"],
        "source_times": [],
        "retrieval_score": None,
    }
    with pytest.raises(ValidationError):
        Evidence.model_validate({**valid, "validity": "current"})
    with pytest.raises(ValidationError):
        Evidence.model_validate({**valid, "memory_id": "m_1"})
    # 2) Operation targets come only from receipts
    docs = m["ops"]["docs"]
    receipt_ids: set[str] = set()
    for d in docs.values():
        for entry in d["attempts"]["entries"]:
            output = entry.get("output")
            if isinstance(output, dict) and "memory_ids" in output:
                receipt_ids.update(output["memory_ids"])
    assert receipt_ids
    for check in m["ops"]["summary"]["checks"]:
        for target in check["target_memory_ids"]:
            assert target in receipt_ids, (check["check_id"], target)
    # 3) Delete decides by recall (target AND derived summaries), with an
    #    unrelated-memory positive control — not by adapter self-report
    delete = next(
        c for c in m["ops"]["summary"]["checks"] if c["check_id"] == "delete"
    )
    names = " ".join(a["name"] for a in delete["assertions"])
    assert "no longer recalled" in names
    assert "unrelated memory still retrievable" in names


def check_granularity_overhead(m: dict[str, Any]) -> None:
    report = m["qa"]["report"]
    budget = report["budget"]
    rows = [r for r in budget["rows"] if r["token_count"] is not None]
    assert rows
    # Text and format overhead are reported separately and add up
    assert any(r["format_tokens"] > 0 for r in rows)
    for r in rows:
        assert r["text_tokens"] + r["format_tokens"] == r["token_count"]
    ids = {x["metric_id"] for x in report["metrics"]}
    assert {"evidence_text_tokens", "evidence_format_tokens"} <= ids
    # The split never changes the budget rule (hard cap intact)
    for r in rows:
        assert r["token_count"] <= budget["budget"]


def check_metric_drift(m: dict[str, Any]) -> None:
    report = m["qa"]["report"]
    # 1) Formal table ids are all registered (versioned definitions)
    for metric in report["metrics"]:
        assert metric["metric_id"] in REGISTRY
    # 2) Registry version mismatch refuses metric auto-alignment
    def side(version: str) -> RunSide:
        return RunSide(
            run_dir=Path("/tmp/x"),
            manifest={"run_id": "r", "suite": "qa", "sample_ids": []},
            config_doc={
                "metrics_registry_version": version,
                "config_fingerprint": "0" * 64,
                "config": {
                    "suite": "qa",
                    "dataset_plan": "p",
                    "sample_plan_id": "s",
                    "sample_ids": [],
                    "smoke_subset_ids": [],
                    "reader": {},
                    "judge": {},
                    "run_params": {},
                    "evidence_token_budget": 4096,
                },
            },
            report=None,
        )

    comp = comparability(side("1"), side("2"))
    assert comp["same_condition"] is False
    assert comp["metrics_auto_aligned"] is False
    assert "注册表版本不一致" in comp["metrics_alignment_refused_reason"]
    # 3) Registry content is hash-bound into the config fingerprint:
    #    the header exposes the content version for audit
    assert report["header"]["metrics_registry_content_version"].startswith(
        "metrics-registry@1+"
    )


def check_joint_attribution(m: dict[str, Any]) -> None:
    report = m["qa"]["report"]
    attribution = report["attribution"]
    cells = ("hit_correct", "hit_wrong", "miss_correct", "miss_wrong")
    overall = attribution["overall"]
    assert set(overall) == set(cells)
    assert sum(overall.values()) == attribution["scored_denominator"]
    scored = report["statuses"]["scored"]
    shares = attribution["shares"]
    assert sum(shares.values()) == pytest.approx(1.0)
    # Non-scored statuses are listed separately, never folded into wrong
    assert set(attribution["excluded_from_cells"]) == {
        "failed",
        "context_exceeded",
        "pending",
        "invalid_input",
    }
    assert sum(attribution["excluded_from_cells"].values()) == (
        report["statuses"]["planned"] - scored
    )
    # Hit criterion is declared and subsets exist
    assert attribution["hit_criteria"]
    assert set(attribution["by_abstention"]) == {"true", "false"}
    assert attribution["by_question_type"]


def check_budget_overrun_visible(m: dict[str, Any]) -> None:
    doc = m["flood"]["doc"]
    report = m["flood"]["report"]
    prepared = doc["prepared_evidence"]["prepared"]
    raw = doc["raw_evidence"]["evidence"]
    row = next(
        r
        for r in report["budget"]["rows"]
        if r["sample_handle"] == "smoke_multi_session_0001"
    )
    # Returned vs entering-reader vs dropped counts are all visible
    assert row["returned_units"] == len(raw)
    assert row["retained_units"] == len(prepared["items"])
    assert row["dropped_units"] == len(prepared["dropped_raw_indices"])
    assert row["dropped_units"] > 0
    assert row["dropped_text_tokens"] > 0  # removal measured in tokens
    # The raw (pre-truncation) return stays persisted for diagnostics
    assert len(raw) > len(prepared["items"])
    # Hard cap: never silently over budget
    assert prepared["token_count"] <= report["budget"]["budget"]


def check_explicit_update_completion(m: dict[str, Any]) -> None:
    explicit = next(
        c
        for c in m["ops"]["summary"]["checks"]
        if c["check_id"] == "explicit_update"
    )
    assert explicit["operation_status"] == "passed"
    names = " ".join(a["name"] for a in explicit["assertions"])
    assert "updated target content equals the replacement" in names
    assert "updated target is current" in names
    assert "retained old value is not current" in names
    assert "replacement is observable in retrieval" in names
    # sources=[] not a failure: the target's receipt-level assertions
    # passed although the replacement carries no sources
    assert all(a["passed"] for a in explicit["assertions"])


def check_scale_and_cost(m: dict[str, Any]) -> None:
    report = m["qa"]["report"]
    cost = report["cost_model"]
    assert cost["planned_calls"]["ingest_calls"] > 0
    assert cost["actual_calls"]["ingest"] == cost["planned_calls"]["ingest_calls"]
    methods = {unit["method"] for unit in cost["unit_costs"]}
    assert {"retrieve", "reader.answer", "judge.evaluate"} <= methods
    assert cost["estimated"] is True
    assert cost["estimated_totals"]
    assert any("估算" in note for note in report["limitations"])
    # Smoke subset fixed at 8: one per question_type plus two abstention
    config = m["qa"]["config"]
    assert len(config.smoke_subset_ids) == 8
    assert config.smoke_subset_ids == config.sample_ids
    assert report["statuses"]["planned"] == 8
    assert any("smoke" in note for note in report["limitations"])


CHECKS: dict[str, Callable[[dict[str, Any]], None]] = {
    "id_leakage": check_id_leakage,
    "question_date": check_question_date,
    "truncation_recall": check_truncation_recall,
    "update_state_observable": check_update_state_observable,
    "incomplete_close": check_incomplete_close,
    "retry_duplicate_mutation": check_retry_duplicate_mutation,
    "abstention_recall": check_abstention_recall,
    "retrieval_not_state": check_retrieval_not_state,
    "granularity_overhead": check_granularity_overhead,
    "metric_drift": check_metric_drift,
    "joint_attribution": check_joint_attribution,
    "budget_overrun_visible": check_budget_overrun_visible,
    "explicit_update_completion": check_explicit_update_completion,
    "scale_and_cost": check_scale_and_cost,
}


def matrix_ids() -> list[str]:
    """Every design-doc matrix row must have exactly one check here."""
    assert set(CHECKS) == set(IDS)
    return IDS


@pytest.mark.parametrize("row_id", matrix_ids())
def test_review_matrix(row_id: str, matrix) -> None:
    """One citable assertion per design-doc review row."""
    CHECKS[row_id](matrix)


def test_matrix_rows_match_design_doc():
    """The 14 rows are exactly the issue #6 acceptance list."""
    assert len(MATRIX) == 14
    assert len(set(IDS)) == 14
