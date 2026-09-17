"""M1 operations-capability suite: deterministic checks over the adapter.

Implements the 操作 section of docs/design/eval-harness.md with small
artificial data and programmatic assertions only — no LLM judge ever
decides whether an operation succeeded:

- auto_update: consecutive ingests of an old and a new convention, with
  NO update() call; after reopening, the new convention must inspect as
  current and a retained old value as superseded/deleted (never
  current, never unknown).
- explicit_update: the target id comes from the ingest receipt;
  update() replaces the full text; afterwards inspect([target]) must
  give content == replacement and validity == current, extra ids in
  the update receipt (retained old values) must not be current, other
  memories must be unchanged, the replacement must be observable in
  retrieval and no derived summary may still embed the old convention.
- delete: the target id comes from the receipt; after completion
  neither the original text nor derived summaries containing it may be
  recalled, unrelated memories must stay retrievable, and all of this
  must hold after closing and reopening; with state_inspection declared
  the target must inspect as deleted (unknown is a failure).
- isolation: a query strongly indicating project B must not return B
  content from A's space (and vice versa), with positive controls that
  each space still returns its own content — an always-empty retrieve
  cannot pass.
- persistence: content written before close() is retrievable after
  open() again.

Status assertions go exclusively through inspect() by stable id;
retrieval results never carry validity, and explicit update/delete
targets are always receipt-issued ids. passed / failed / not_supported
are recorded per check item and counted separately in the run summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.config import OPERATIONS_CHECK_IDS
from eval.contracts.adapter import (
    ErrorInfo,
    Evidence,
    MemoryState,
    MutationReceipt,
    RetrievalRequest,
    Session,
    Message,
)
from eval.contracts.internal import (
    MetricResult,
    Result,
    ResultArtifact,
    StageAttempt,
)
from eval.datasets.manual import namespace_for
from eval.runner import RunnerBase, _StageFailure, _UncertainMutation
from eval.runs import (
    AttemptEntry,
    AttemptLogArtifact,
    CheckAssertion,
    OperationCheckDetail,
    OperationsSummaryArtifact,
    RunManifest,
    SamplePlanCounts,
)

#: Fixed question date for operations probes (data-determined, never the
#: machine clock).
OPS_QUESTION_DATE = "2026-09-06"

# -- small artificial data ---------------------------------------------------

AUTO_UPDATE_OLD_SESSION = Session(
    session_id="s_ops_auto_old",
    occurred_at="2026-09-01",
    messages=[Message(msg_id="m_ops_auto_1", role="user", content="包管理器约定：使用 npm。")],
)
AUTO_UPDATE_NEW_SESSION = Session(
    session_id="s_ops_auto_new",
    occurred_at="2026-09-03",
    messages=[
        Message(
            msg_id="m_ops_auto_2",
            role="user",
            content="包管理器约定：迁移到 pnpm，以后都用 pnpm。",
        )
    ],
)

UPDATE_TARGET_SESSION = Session(
    session_id="s_ops_update_target",
    occurred_at="2026-09-02",
    messages=[
        Message(
            msg_id="m_ops_update_1",
            role="user",
            content="部署约定：版本再发布前跑 uv run pytest。",
        )
    ],
)
UPDATE_UNRELATED_SESSION = Session(
    session_id="s_ops_update_other",
    occurred_at="2026-09-03",
    messages=[
        Message(msg_id="m_ops_update_2", role="user", content="无关事实：绿植每周五浇水。")
    ],
)
UPDATE_REPLACEMENT = "部署约定：用 ruff，发布前跑 uv run pytest。"
#: Distinctive markers: OLD occurs only in the pre-update text, NEW only
#: in the replacement. Both sit INSIDE the first 16 characters of their
#: texts so they survive the fake's summary prefixes — that is what
#: makes the before/after summary assertions discriminative instead of
#: vacuously true (tests/eval guard the marker positions).
UPDATE_OLD_MARKER = "再发布"
UPDATE_NEW_MARKER = "ruff"
UPDATE_TOPIC_QUERY = "部署约定 发布流程是什么"

DELETE_TARGET_SESSION = Session(
    session_id="s_ops_del_target",
    occurred_at="2026-09-04",
    messages=[
        Message(
            msg_id="m_ops_del_1",
            role="user",
            content="缓存约定：30 秒过期，Redis 统一。",
        )
    ],
)
DELETE_UNRELATED_SESSION = Session(
    session_id="s_ops_del_other",
    occurred_at="2026-09-05",
    messages=[
        Message(msg_id="m_ops_del_2", role="user", content="无关事实：站会改到每周四上午。")
    ],
)
DELETE_TARGET_MARKER = "30 秒过期"
DELETE_UNRELATED_MARKER = "站会"
DELETE_TARGET_QUERY = "缓存约定 过期时间是多少"
DELETE_UNRELATED_QUERY = "站会 在什么时间"

ISOLATION_A_SESSION = Session(
    session_id="s_ops_iso_a",
    occurred_at="2026-09-01",
    messages=[
        Message(msg_id="m_ops_iso_1", role="user", content="项目甲的构建工具是 Bazel。")
    ],
)
ISOLATION_B_SESSION = Session(
    session_id="s_ops_iso_b",
    occurred_at="2026-09-02",
    messages=[
        Message(msg_id="m_ops_iso_2", role="user", content="项目乙的构建工具是 Buck2。")
    ],
)
ISOLATION_A_MARKER = "Bazel"
ISOLATION_B_MARKER = "Buck2"
ISOLATION_A_NAME = "项目甲"
ISOLATION_B_NAME = "项目乙"
#: Cross queries mention the OTHER project by name and tool; own queries
#: ask about the space's own project.
ISOLATION_QUERY_ABOUT_B = "项目乙的构建工具是 Buck2 吗"
ISOLATION_QUERY_ABOUT_A = "项目甲的构建工具是 Bazel 吗"
ISOLATION_QUERY_OWN_A = "项目甲的构建工具是什么"
ISOLATION_QUERY_OWN_B = "项目乙的构建工具是什么"

PERSIST_SESSION = Session(
    session_id="s_ops_persist",
    occurred_at="2026-09-01",
    messages=[
        Message(
            msg_id="m_ops_persist_1",
            role="user",
            content="文档约定：设计文档放在 docs/design 目录。",
        ),
        Message(msg_id="m_ops_persist_2", role="user", content="无关事实：咖啡机在三层茶水间。"),
    ],
)
PERSIST_QUERY_DOCS = "文档约定 设计文档放在哪里"
PERSIST_QUERY_COFFEE = "咖啡机 在哪里"
PERSIST_DOCS_MARKER = "docs/design"
PERSIST_COFFEE_MARKER = "咖啡机"

#: Optional capabilities each check needs before it may run at all.
CHECK_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "auto_update": ("auto_update", "state_inspection"),
    "explicit_update": ("update", "state_inspection"),
    "delete": ("delete",),
    "isolation": (),
    "persistence": (),
}

#: Stages each check touches (used for initial pending stage_states).
CHECK_STAGES: dict[str, tuple[str, ...]] = {
    "auto_update": ("ingest", "await_ready", "inspect"),
    "explicit_update": ("ingest", "await_ready", "update", "inspect", "retrieve"),
    "delete": ("ingest", "await_ready", "delete", "inspect", "retrieve"),
    "isolation": ("ingest", "await_ready", "retrieve"),
    "persistence": ("ingest", "await_ready", "retrieve"),
}


class _CheckFailure(Exception):
    """A deterministic assertion failed at a stage."""

    def __init__(self, stage: str, error: ErrorInfo) -> None:
        super().__init__(f"{stage}: {error.code}")
        self.stage = stage
        self.error = error


class _NotSupported(Exception):
    """The check cannot run (recorded as not_supported, never as failed)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class _CheckRun:
    """Mutable state of one check item while it executes."""

    check_id: str
    namespace: str
    namespaces: list[str] = field(default_factory=list)
    open_namespaces: set[str] = field(default_factory=set)
    attempts: list[StageAttempt] = field(default_factory=list)
    entries: list[AttemptEntry] = field(default_factory=list)
    stage_states: dict[str, str] = field(default_factory=dict)
    assertions: list[CheckAssertion] = field(default_factory=list)
    target_ids: list[str] = field(default_factory=list)
    #: attempt classification for this execution cycle: replay cycles
    #: (isolation+replay of an uncertain mutation) mark their calls as
    #: recovery usage.
    attempt_kind: str = "logical"


@dataclass
class OperationsOutcome:
    run_id: str
    run_dir: Path
    results: list[ResultArtifact] = field(default_factory=list)
    summary: OperationsSummaryArtifact | None = None

    @property
    def failed(self) -> int:
        return sum(
            1 for r in self.results if r.result.operation_status == "failed"
        )

    def payload(self) -> dict[str, Any]:
        summary = self.summary
        assert summary is not None
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "suite": "operations",
            "planned_checks": summary.planned_checks,
            "passed": summary.passed,
            "failed": summary.failed,
            "not_supported": summary.not_supported,
            "pending": summary.pending,
            "operations_pass_rate": summary.pass_rate,
            "operations_support_coverage": summary.support_coverage,
            "checks": {
                r.result.sample_handle: r.result.operation_status
                for r in self.results
            },
        }


class OperationsRunner(RunnerBase):
    """Runs the deterministic operations suite declared by the config.

    The config's sample_ids are the planned check items; each produces
    one Result row (suite="operations") plus an attempts log and a
    check-detail artifact, and the run ends with an operations summary
    counting passed / failed / not_supported separately.
    """

    def run(self) -> OperationsOutcome:
        self._check_capabilities()
        capabilities = set(self.adapter.capabilities())
        async_mutation = "async_mutation" in capabilities
        manifest = RunManifest(
            run_id=self.run_id,
            suite="operations",
            created_at=self._clock(),
            config_name=self.config.name,
            config_fingerprint=self.config.fingerprint(),
            metrics_registry_version=self.config.canonical_payload()[
                "metrics_registry_version"
            ],
            dataset_plan=self.config.dataset_plan,
            sample_plan_id=self.config.sample_plan_id,
            sample_ids=tuple(self.config.sample_ids),
            evidence_token_budget=self.config.evidence_token_budget,
            memory_declaration=self.config.memory.model_dump(mode="json"),
            async_mutation=async_mutation,
            plan_counts=self._plan_counts(capabilities),
        )
        from eval.config import ConfigArtifact

        self.store.create(
            manifest,
            ConfigArtifact(
                config=self.config, config_fingerprint=self.config.fingerprint()
            ),
        )
        outcome = OperationsOutcome(run_id=self.run_id, run_dir=self.store.dir)
        details: list[OperationCheckDetail] = []
        for check_id in self.config.sample_ids:
            artifact, detail = self._run_check(check_id, capabilities)
            outcome.results.append(artifact)
            details.append(detail)
        outcome.summary = self._build_summary(details)
        self.store.write_operations_summary(outcome.summary)
        return outcome

    # -- planning -----------------------------------------------------------

    def _plan_counts(
        self, capabilities: set[str]
    ) -> dict[str, SamplePlanCounts]:
        """Expected calls per check when it runs (scale model)."""
        is_async = "async_mutation" in capabilities
        has_state = "state_inspection" in capabilities
        # sessions, ingest, update, delete, inspect, open/close, retrieve
        base: dict[str, dict[str, int]] = {
            "auto_update": dict(
                sessions=2, ingest=2, update=0, delete=0, inspect=1,
                open=3, retrieve=0,
            ),
            "explicit_update": dict(
                sessions=2, ingest=2, update=1, delete=0,
                inspect=2 if has_state else 0, open=4, retrieve=2,
            ),
            "delete": dict(
                sessions=2, ingest=2, update=0, delete=1,
                inspect=2 if has_state else 0, open=4, retrieve=5,
            ),
            "isolation": dict(
                sessions=2, ingest=2, update=0, delete=0, inspect=0,
                open=4, retrieve=4,
            ),
            "persistence": dict(
                sessions=1, ingest=1, update=0, delete=0, inspect=0,
                open=2, retrieve=2,
            ),
        }
        counts: dict[str, SamplePlanCounts] = {}
        for check_id in self.config.sample_ids:
            if check_id not in base:
                continue
            spec = base[check_id]
            mutations = spec["ingest"] + spec["update"] + spec["delete"]
            counts[check_id] = SamplePlanCounts(
                sessions=spec["sessions"],
                ingest_calls=spec["ingest"],
                await_ready_calls=mutations if is_async else 0,
                update_calls=spec["update"],
                delete_calls=spec["delete"],
                inspect_calls=spec["inspect"],
                open_calls=spec["open"],
                close_calls=spec["open"],
                retrieve_calls=spec["retrieve"],
            )
        return counts

    def _build_summary(
        self, details: list[OperationCheckDetail]
    ) -> OperationsSummaryArtifact:
        passed = sum(1 for d in details if d.operation_status == "passed")
        failed = sum(1 for d in details if d.operation_status == "failed")
        not_supported = sum(
            1 for d in details if d.operation_status == "not_supported"
        )
        pending = sum(1 for d in details if d.operation_status == "pending")
        attempted = passed + failed
        pass_rate = (passed / attempted) if attempted else None
        coverage = (attempted / len(details)) if details else None
        metrics = [
            MetricResult(
                metric_id="operations_pass_rate",
                status="computed" if attempted else "not_applicable",
                value=pass_rate,
                reason=(
                    None
                    if attempted
                    else "denominator_zero: no passed+failed checks"
                ),
            ),
            MetricResult(
                metric_id="operations_support_coverage",
                status="computed" if details else "not_applicable",
                value=coverage,
                reason=(
                    None if details else "denominator_zero: no planned checks"
                ),
            ),
        ]
        return OperationsSummaryArtifact(
            run_id=self.run_id,
            planned_checks=len(details),
            passed=passed,
            failed=failed,
            not_supported=not_supported,
            pending=pending,
            pass_rate=pass_rate,
            support_coverage=coverage,
            metrics=metrics,
            checks=details,
        )

    # -- per-check dispatch ---------------------------------------------------

    def _run_check(
        self, check_id: str, capabilities: set[str]
    ) -> tuple[ResultArtifact, OperationCheckDetail]:
        namespace = namespace_for(self.config.sample_plan_id, check_id)
        self._lifecycle_entries = []
        required = CHECK_REQUIREMENTS[check_id]
        missing = tuple(c for c in required if c not in capabilities)

        status = "passed"
        reason: str | None = None
        failed_stage: str | None = None
        ctx = _CheckRun(
            check_id=check_id,
            namespace=namespace,
            namespaces=[namespace],
            stage_states={stage: "pending" for stage in CHECK_STAGES[check_id]},
        )
        if missing:
            # Undeclared optional capabilities are decided BEFORE the run:
            # the item is not_supported, not a failure, and never a pass.
            status = "not_supported"
            reason = (
                f"capability not declared before the run: {list(missing)}; "
                f"check requires {list(required)}"
            )
        else:
            replays = 0
            while True:
                try:
                    getattr(self, f"_check_{check_id}")(ctx)
                    break
                except _NotSupported as exc:
                    status = "not_supported"
                    reason = exc.reason
                    break
                except _UncertainMutation as exc:
                    # Same isolation+replay policy as the QA loop: reset
                    # every space the failed cycle touched and re-run the
                    # deterministic check from its own operation log.
                    if exc.stage in ctx.stage_states:
                        ctx.stage_states[exc.stage] = "failed"
                    if replays >= self.config.run_params.max_sample_replays:
                        status = "failed"
                        failed_stage = exc.stage
                        reason = (
                            f"uncertain mutation outcome at {exc.stage} "
                            f"({exc.error.code}) and the isolation+replay "
                            f"budget of "
                            f"{self.config.run_params.max_sample_replays} is "
                            "exhausted; recorded as failed, never guessed"
                        )
                        break
                    replays += 1
                    stale = list(dict.fromkeys(ctx.namespaces + [namespace]))
                    for ns in stale:
                        self._record_lifecycle(
                            "reset",
                            ns,
                            lambda ns=ns: self.adapter.reset(ns),
                            attempt_kind="replay",
                        )
                    ctx = _CheckRun(
                        check_id=check_id,
                        namespace=namespace,
                        namespaces=[namespace],
                        stage_states={
                            stage: "pending" for stage in CHECK_STAGES[check_id]
                        },
                        attempts=ctx.attempts,
                        entries=ctx.entries,
                        assertions=ctx.assertions,
                        attempt_kind="replay",
                    )
                except (_StageFailure, _CheckFailure) as exc:
                    status = "failed"
                    failed_stage = exc.stage
                    ctx.stage_states[exc.stage] = "failed"
                    break
                finally:
                    for ns in sorted(ctx.open_namespaces):
                        self._record_lifecycle(
                            "close", ns, lambda ns=ns: self.adapter.close(ns)
                        )
                    ctx.open_namespaces.clear()

        entries = self._lifecycle_entries + ctx.entries
        log_artifact = AttemptLogArtifact(
            run_id=self.run_id,
            sample_handle=check_id,
            namespace=namespace,
            entries=entries,
        )
        detail = OperationCheckDetail(
            run_id=self.run_id,
            check_id=check_id,
            operation_status=status,  # type: ignore[arg-type]
            namespaces=tuple(ctx.namespaces),
            target_memory_ids=tuple(ctx.target_ids),
            required_capabilities=required,
            missing_capabilities=missing,
            reason=reason,
            failed_stage=failed_stage,
            assertions=ctx.assertions,
        )
        refs = self.store.write_sample_artifacts(
            check_id, attempts_log=log_artifact, operations_check=detail
        )
        result = Result(
            run_id=self.run_id,
            sample_handle=check_id,
            namespace=namespace,
            config_fingerprint=self.config.fingerprint(),
            suite="operations",
            qa_status=None,
            operation_status=status,  # type: ignore[arg-type]
            correct=None,
            attribution=None,
            failed_stage=failed_stage,
            metrics=[],
            stage_states=ctx.stage_states,
            artifact_refs=refs,
            attempts=ctx.attempts,
        )
        artifact = ResultArtifact(result=result)
        self.store.append_result(artifact)
        return artifact, detail

    # -- shared steps ---------------------------------------------------------

    def _open(self, ctx: _CheckRun, namespace: str) -> None:
        self._record_lifecycle(
            "open", namespace, lambda: self.adapter.open(namespace)
        )
        ctx.open_namespaces.add(namespace)

    def _close(self, ctx: _CheckRun, namespace: str) -> None:
        self._record_lifecycle(
            "close", namespace, lambda: self.adapter.close(namespace)
        )
        ctx.open_namespaces.discard(namespace)

    def _ingest_cycle(
        self, ctx: _CheckRun, namespace: str, session: Session, seq: int
    ) -> MutationReceipt:
        """open -> ingest (with completion confirmation) -> close."""
        self._open(ctx, namespace)
        operation_id = self._op_id(
            namespace, "ingest", seq, session.model_dump(mode="json")
        )
        receipt = self._submit_mutation(
            stage="ingest",
            method="ingest",
            namespace=namespace,
            operation_id=operation_id,
            input_payload={
                "session": session.model_dump(mode="json"),
                "operation_id": operation_id,
            },
            fn=lambda: self.adapter.ingest(namespace, session, operation_id),
            attempts=ctx.attempts,
            entries=ctx.entries,
            stage_states=ctx.stage_states,
            attempt_kind=ctx.attempt_kind,
        )
        self._close(ctx, namespace)
        return receipt

    def _retrieve(
        self, ctx: _CheckRun, namespace: str, query: str
    ) -> list[Evidence]:
        request = RetrievalRequest(
            query=query,
            question_date=OPS_QUESTION_DATE,
            evidence_token_budget=self.config.evidence_token_budget,
        )
        result = self._record_call_with_retries(
            stage="retrieve",
            method="retrieve",
            namespace=namespace,
            operation_id=None,
            input_payload={"request": request.model_dump(mode="json")},
            fn=lambda: self.adapter.retrieve(namespace, request),
            attempts=ctx.attempts,
            entries=ctx.entries,
            retryable=lambda error: error.transient,
            attempt_kind=ctx.attempt_kind,
        )
        assert result is not None
        return list(result)

    def _inspect(
        self, ctx: _CheckRun, namespace: str, memory_ids: list[str]
    ) -> dict[str, MemoryState]:
        result = self._record_call_with_retries(
            stage="inspect",
            method="inspect",
            namespace=namespace,
            operation_id=None,
            input_payload={"memory_ids": list(memory_ids)},
            fn=lambda: self.adapter.inspect(namespace, memory_ids),
            attempts=ctx.attempts,
            entries=ctx.entries,
            retryable=lambda error: error.transient,
            attempt_kind=ctx.attempt_kind,
        )
        assert result is not None
        returned = {state.memory_id: state for state in result}
        for mid in memory_ids:
            if mid not in returned:
                raise _CheckFailure(
                    "inspect",
                    ErrorInfo(
                        code="inspect_missing_target",
                        message=(
                            f"inspect() did not return a state for requested "
                            f"id {mid!r}; missing rows cannot replace "
                            "definite states"
                        ),
                        effect="confirmed",
                        transient=False,
                    ),
                )
        return returned

    def _assert(
        self,
        ctx: _CheckRun,
        stage: str,
        name: str,
        passed: bool,
        expected: str,
        observed: str,
    ) -> None:
        ctx.assertions.append(
            CheckAssertion(
                name=name, passed=passed, expected=expected, observed=observed
            )
        )
        if not passed:
            raise _CheckFailure(
                stage,
                ErrorInfo(
                    code="operation_assertion_failed",
                    message=f"{name}: expected {expected}, observed {observed}",
                    effect="confirmed",
                    transient=False,
                ),
            )

    def _select_target(
        self,
        ctx: _CheckRun,
        namespace: str,
        receipt: MutationReceipt,
        marker: str,
        capabilities: set[str],
    ) -> str:
        """Pick the explicit-operation target among receipt ids.

        The id ALWAYS comes from a MutationReceipt. With state
        inspection available the marker-containing candidate is chosen
        deterministically; otherwise the first receipt id.
        """
        if not receipt.memory_ids:
            raise _NotSupported(
                "ingest receipt provided no stable memory id; explicit "
                "operations require a receipt-issued target id"
            )
        if "state_inspection" in capabilities:
            states = self._inspect(ctx, namespace, list(receipt.memory_ids))
            for mid in receipt.memory_ids:
                content = states[mid].content or ""
                if marker in content:
                    ctx.target_ids.append(mid)
                    return mid
        ctx.target_ids.append(receipt.memory_ids[0])
        return receipt.memory_ids[0]

    # -- check implementations -------------------------------------------------

    def _check_auto_update(self, ctx: _CheckRun) -> None:
        """Consecutive ingests only — update() is never called here."""
        ns = ctx.namespaces[0]
        self._record_lifecycle("reset", ns, lambda: self.adapter.reset(ns))
        old_receipt = self._ingest_cycle(ctx, ns, AUTO_UPDATE_OLD_SESSION, 0)
        new_receipt = self._ingest_cycle(ctx, ns, AUTO_UPDATE_NEW_SESSION, 1)

        # Observe states after closing and reopening the space.
        self._open(ctx, ns)
        new_ids = list(new_receipt.memory_ids)
        old_ids = [
            mid for mid in old_receipt.memory_ids if mid not in set(new_ids)
        ]
        states = self._inspect(ctx, ns, new_ids + old_ids)
        for mid in new_ids:
            state = states[mid]
            self._assert(
                ctx,
                "inspect",
                f"new convention is current ({mid})",
                state.validity == "current",
                "validity=current",
                f"validity={state.validity}",
            )
        for mid in old_ids:
            state = states[mid]
            self._assert(
                ctx,
                "inspect",
                f"retained old convention is not current ({mid})",
                state.validity in ("superseded", "deleted"),
                "validity in (superseded, deleted)",
                f"validity={state.validity}",
            )
        ctx.stage_states["inspect"] = "completed"
        self._close(ctx, ns)

    def _check_explicit_update(self, ctx: _CheckRun) -> None:
        ns = ctx.namespaces[0]
        capabilities = set(self.adapter.capabilities())
        self._record_lifecycle("reset", ns, lambda: self.adapter.reset(ns))
        target_receipt = self._ingest_cycle(ctx, ns, UPDATE_TARGET_SESSION, 0)
        other_receipt = self._ingest_cycle(ctx, ns, UPDATE_UNRELATED_SESSION, 1)

        self._open(ctx, ns)
        # Positive control BEFORE updating, mirroring the delete check:
        # the old convention (and, when summaries are declared, a derived
        # summary embedding it) must be observable now — otherwise the
        # post-update "no stale summary" assertion would be vacuous.
        pre_evidence = self._retrieve(ctx, ns, UPDATE_TOPIC_QUERY)
        self._assert(
            ctx,
            "retrieve",
            "old convention observable before update",
            any(UPDATE_OLD_MARKER in e.text for e in pre_evidence),
            f"some evidence contains {UPDATE_OLD_MARKER!r}",
            "recalled"
            if any(UPDATE_OLD_MARKER in e.text for e in pre_evidence)
            else "not recalled",
        )
        if "generated_evidence" in capabilities:
            self._assert(
                ctx,
                "retrieve",
                "derived summary embeds the old convention before update",
                any(
                    e.kind == "generated" and UPDATE_OLD_MARKER in e.text
                    for e in pre_evidence
                ),
                f"some generated evidence contains {UPDATE_OLD_MARKER!r}",
                "covered"
                if any(
                    e.kind == "generated"
                    and UPDATE_OLD_MARKER in e.text
                    for e in pre_evidence
                )
                else "not covered",
            )
        target_id = self._select_target(
            ctx, ns, target_receipt, UPDATE_OLD_MARKER, capabilities
        )
        operation_id = self._op_id(
            ns,
            "update",
            0,
            {"memory_id": target_id, "replacement": UPDATE_REPLACEMENT},
        )
        update_receipt = self._submit_mutation(
            stage="update",
            method="update",
            namespace=ns,
            operation_id=operation_id,
            input_payload={
                "memory_id": target_id,
                "replacement": UPDATE_REPLACEMENT,
                "operation_id": operation_id,
            },
            fn=lambda: self.adapter.update(
                ns, target_id, UPDATE_REPLACEMENT, operation_id
            ),
            attempts=ctx.attempts,
            entries=ctx.entries,
            stage_states=ctx.stage_states,
            attempt_kind=ctx.attempt_kind,
        )
        self._close(ctx, ns)

        # Completion means observable after reopening: inspect and
        # retrieve in a freshly opened instance.
        self._open(ctx, ns)
        retained = [mid for mid in update_receipt.memory_ids if mid != target_id]
        unrelated_ids = list(other_receipt.memory_ids)
        states = self._inspect(ctx, ns, [target_id, *retained, *unrelated_ids])
        target_state = states[target_id]
        self._assert(
            ctx,
            "inspect",
            "updated target content equals the replacement",
            target_state.content == UPDATE_REPLACEMENT,
            f"content=={UPDATE_REPLACEMENT!r}",
            f"content={target_state.content!r}",
        )
        self._assert(
            ctx,
            "inspect",
            "updated target is current",
            target_state.validity == "current",
            "validity=current",
            f"validity={target_state.validity}",
        )
        for mid in retained:
            state = states[mid]
            self._assert(
                ctx,
                "inspect",
                f"retained old value is not current ({mid})",
                state.validity in ("superseded", "deleted"),
                "validity in (superseded, deleted)",
                f"validity={state.validity}",
            )
        unrelated_content = UPDATE_UNRELATED_SESSION.messages[0].content
        for mid in unrelated_ids:
            state = states[mid]
            self._assert(
                ctx,
                "inspect",
                f"unrelated memory unchanged ({mid})",
                state.content == unrelated_content and state.validity == "current",
                f"content=={unrelated_content!r} and validity=current",
                f"content={state.content!r}, validity={state.validity}",
            )
        ctx.stage_states["inspect"] = "completed"

        evidence = self._retrieve(ctx, ns, UPDATE_TOPIC_QUERY)
        texts = [e.text for e in evidence]
        self._assert(
            ctx,
            "retrieve",
            "replacement is observable in retrieval",
            any(UPDATE_NEW_MARKER in text for text in texts),
            f"some evidence contains {UPDATE_NEW_MARKER!r}",
            "present" if any(UPDATE_NEW_MARKER in t for t in texts) else "absent",
        )
        stale_summaries = [
            e.text
            for e in evidence
            if e.kind == "generated" and UPDATE_OLD_MARKER in e.text
        ]
        self._assert(
            ctx,
            "retrieve",
            "no derived summary still embeds the old convention",
            not stale_summaries,
            f"no generated evidence contains {UPDATE_OLD_MARKER!r}",
            f"{len(stale_summaries)} generated units still contain it",
        )
        ctx.stage_states["retrieve"] = "completed"
        self._close(ctx, ns)

    def _check_delete(self, ctx: _CheckRun) -> None:
        ns = ctx.namespaces[0]
        capabilities = set(self.adapter.capabilities())
        self._record_lifecycle("reset", ns, lambda: self.adapter.reset(ns))
        target_receipt = self._ingest_cycle(ctx, ns, DELETE_TARGET_SESSION, 0)
        other_receipt = self._ingest_cycle(ctx, ns, DELETE_UNRELATED_SESSION, 1)

        self._open(ctx, ns)
        pre_evidence = self._retrieve(ctx, ns, DELETE_TARGET_QUERY)
        self._assert(
            ctx,
            "retrieve",
            "target original text is recallable before delete",
            any(DELETE_TARGET_MARKER in e.text for e in pre_evidence),
            f"some evidence contains {DELETE_TARGET_MARKER!r}",
            "recalled"
            if any(DELETE_TARGET_MARKER in e.text for e in pre_evidence)
            else "not recalled",
        )
        if "generated_evidence" in capabilities:
            self._assert(
                ctx,
                "retrieve",
                "derived summary covers the target before delete",
                any(
                    e.kind == "generated" and DELETE_TARGET_MARKER in e.text
                    for e in pre_evidence
                ),
                f"some generated evidence contains {DELETE_TARGET_MARKER!r}",
                "covered"
                if any(
                    e.kind == "generated"
                    and DELETE_TARGET_MARKER in e.text
                    for e in pre_evidence
                )
                else "not covered",
            )
        target_id = self._select_target(
            ctx, ns, target_receipt, DELETE_TARGET_MARKER, capabilities
        )
        operation_id = self._op_id(ns, "delete", 0, {"memory_id": target_id})
        self._submit_mutation(
            stage="delete",
            method="delete",
            namespace=ns,
            operation_id=operation_id,
            input_payload={
                "memory_id": target_id,
                "operation_id": operation_id,
            },
            fn=lambda: self.adapter.delete(ns, target_id, operation_id),
            attempts=ctx.attempts,
            entries=ctx.entries,
            stage_states=ctx.stage_states,
            attempt_kind=ctx.attempt_kind,
        )
        self._assert_absent_and_unrelated(ctx, ns, "after delete")
        self._close(ctx, ns)

        # Deletion must survive closing and reopening the instance.
        self._open(ctx, ns)
        self._assert_absent_and_unrelated(ctx, ns, "after restart")
        if "state_inspection" in capabilities:
            states = self._inspect(ctx, ns, [target_id])
            state = states[target_id]
            self._assert(
                ctx,
                "inspect",
                "deleted target inspects as deleted",
                state.validity == "deleted",
                "validity=deleted",
                f"validity={state.validity}",
            )
            ctx.stage_states["inspect"] = "completed"
        self._close(ctx, ns)

    def _assert_absent_and_unrelated(
        self, ctx: _CheckRun, namespace: str, phase: str
    ) -> None:
        """The target is gone from recall; unrelated memory stays."""
        target_evidence = self._retrieve(ctx, namespace, DELETE_TARGET_QUERY)
        leaked = [
            e.text
            for e in target_evidence
            if DELETE_TARGET_MARKER in e.text
        ]
        self._assert(
            ctx,
            "retrieve",
            f"target text and derived summaries no longer recalled ({phase})",
            not leaked,
            f"no evidence (any kind) contains {DELETE_TARGET_MARKER!r}",
            f"{len(leaked)} units still contain it",
        )
        unrelated_evidence = self._retrieve(ctx, namespace, DELETE_UNRELATED_QUERY)
        kept = [
            e.text
            for e in unrelated_evidence
            if DELETE_UNRELATED_MARKER in e.text
        ]
        self._assert(
            ctx,
            "retrieve",
            f"unrelated memory still retrievable ({phase})",
            bool(kept),
            f"some evidence contains {DELETE_UNRELATED_MARKER!r}",
            "recalled" if kept else "not recalled",
        )
        ctx.stage_states["retrieve"] = "completed"

    def _check_isolation(self, ctx: _CheckRun) -> None:
        ns_a = namespace_for(self.config.sample_plan_id, "isolation#a")
        ns_b = namespace_for(self.config.sample_plan_id, "isolation#b")
        ctx.namespaces = [ns_a, ns_b]
        self._record_lifecycle("reset", ns_a, lambda: self.adapter.reset(ns_a))
        self._record_lifecycle("reset", ns_b, lambda: self.adapter.reset(ns_b))
        self._ingest_cycle(ctx, ns_a, ISOLATION_A_SESSION, 0)
        self._ingest_cycle(ctx, ns_b, ISOLATION_B_SESSION, 0)

        for own_ns, other_name, other_tool, own_tool, cross_query, own_query in (
            (
                ns_a,
                ISOLATION_B_NAME,
                ISOLATION_B_MARKER,
                ISOLATION_A_MARKER,
                ISOLATION_QUERY_ABOUT_B,
                ISOLATION_QUERY_OWN_A,
            ),
            (
                ns_b,
                ISOLATION_A_NAME,
                ISOLATION_A_MARKER,
                ISOLATION_B_MARKER,
                ISOLATION_QUERY_ABOUT_A,
                ISOLATION_QUERY_OWN_B,
            ),
        ):
            self._open(ctx, own_ns)
            cross = self._retrieve(ctx, own_ns, cross_query)
            leaked = [
                e.text
                for e in cross
                if other_tool in e.text or other_name in e.text
            ]
            self._assert(
                ctx,
                "retrieve",
                f"space does not return {other_name} content",
                not leaked,
                f"no evidence contains {other_tool!r} or {other_name!r}",
                f"{len(leaked)} units leak the other space",
            )
            own = self._retrieve(ctx, own_ns, own_query)
            self._assert(
                ctx,
                "retrieve",
                "space still returns its own content",
                any(own_tool in e.text for e in own),
                f"some evidence contains {own_tool!r}",
                "recalled"
                if any(own_tool in e.text for e in own)
                else "not recalled",
            )
            ctx.stage_states["retrieve"] = "completed"
            self._close(ctx, own_ns)

    def _check_persistence(self, ctx: _CheckRun) -> None:
        ns = ctx.namespaces[0]
        self._record_lifecycle("reset", ns, lambda: self.adapter.reset(ns))
        self._ingest_cycle(ctx, ns, PERSIST_SESSION, 0)

        # Reopen the closed instance: written content must be retrievable.
        self._open(ctx, ns)
        for query, marker in (
            (PERSIST_QUERY_DOCS, PERSIST_DOCS_MARKER),
            (PERSIST_QUERY_COFFEE, PERSIST_COFFEE_MARKER),
        ):
            evidence = self._retrieve(ctx, ns, query)
            self._assert(
                ctx,
                "retrieve",
                f"written content retrievable after reopen ({marker!r})",
                any(marker in e.text for e in evidence),
                f"some evidence contains {marker!r}",
                "recalled"
                if any(marker in e.text for e in evidence)
                else "not recalled",
            )
        ctx.stage_states["retrieve"] = "completed"
        self._close(ctx, ns)


def planned_check_ids() -> tuple[str, ...]:
    """The fixed M1 operations check plan (also config-validated)."""
    return OPERATIONS_CHECK_IDS
