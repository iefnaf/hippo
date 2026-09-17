"""Offline single-question loop: ingest -> reopen -> retrieve -> prepare.

The M1 runner drives one isolated space per sample through the protocol
fixed in the design document:

1. per-session write: open -> ingest (ONLY the current session) ->
   completion confirmation (await_ready when the receipt is accepted)
   -> close, then the next session;
2. query phase: reopen the same space, build the retrieval request from
   the dataset question_date and the fixed token budget (machine time
   never enters query context), retrieve, then run reader input
   preparation against the cleaned history;
3. persistence: per-sample artifacts (attempt log, pre-truncation raw
   evidence, post-budget prepared evidence) and one Result JSONL line
   with stage states, call attempts, timing and usage.

Reader/judge stages stay pending in this milestone (M1 辅助问答 lands in
a later issue); a failed ingest/retrieve/prepare marks the sample
failed with its failed_stage while its artifacts are still persisted.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from eval.contracts.adapter import ErrorInfo, Evidence, MutationReceipt
from eval.contracts.common import ContractError, now_utc
from eval.contracts.internal import (
    PreparedEvidence,
    PreparedEvidenceArtifact,
    RawEvidenceArtifact,
    Result,
    ResultArtifact,
    StageAttempt,
)
from eval.memories.base import MemoryAdapter, MemoryAdapterError
from eval.memories.usage import recorded_usage
from eval.prepare.evidence import (
    HistoryIndex,
    PrepareError,
    build_history_index,
    prepare_evidence,
)
from eval.runs import (
    AttemptEntry,
    AttemptLogArtifact,
    RunManifest,
    RunStore,
    SamplePlanCounts,
    inline_ref,
)
from eval.prepare.tokens import TestCharTokenizer

QueryStages = ("ingest", "await_ready", "retrieve", "prepare")


class _StageFailure(Exception):
    """Internal control flow: the sample failed at a stage."""

    def __init__(self, stage: str, error: ErrorInfo) -> None:
        super().__init__(f"{stage}: {error.code}")
        self.stage = stage
        self.error = error


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


@dataclass
class RunOutcome:
    run_id: str
    run_dir: Path
    results: list[ResultArtifact] = field(default_factory=list)
    failed: int = 0

    def summary(self) -> dict[str, Any]:
        completed = sum(
            1
            for r in self.results
            if r.result.stage_states.get("prepare") == "completed"
        )
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "samples": len(self.results),
            "prepare_completed": completed,
            "failed": self.failed,
        }


def _operation_id(run_id: str, namespace: str, stage: str, seq: int, payload: Any) -> str:
    """Stable operation id: run + space + sequence + input checksum."""
    digest = hashlib.sha256(
        json.dumps(_jsonable(payload), sort_keys=True, ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()[:12]
    return f"{run_id}:{namespace}:{stage}:{seq:03d}:{digest}"


class OfflineRunner:
    """Runs the offline loop for every configured sample."""

    def __init__(
        self,
        *,
        config: Any,
        dataset: Any,
        adapter: MemoryAdapter,
        store: RunStore,
        run_id: str,
        clock: Callable[[], str] = now_utc,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.adapter = adapter
        self.store = store
        self.run_id = run_id
        self._clock = clock
        self._monotonic = monotonic
        self._attempt_seq = 0

    # -- pre-run checks ----------------------------------------------------

    def _pre_run_checks(self) -> None:
        self.dataset.require_handles(self.config.sample_ids)
        declared = set(self.config.memory.capabilities)
        provided = set(self.adapter.capabilities())
        if declared != provided:
            raise ContractError(
                code="capability_declaration_mismatch",
                message=(
                    f"memory plan declares {sorted(declared)} but the "
                    f"adapter provides {sorted(provided)}; capabilities are "
                    "declared and fixed before the run"
                ),
                location="/memory/capabilities",
            )

    def _plan_counts(self, async_mutation: bool) -> dict[str, SamplePlanCounts]:
        counts: dict[str, SamplePlanCounts] = {}
        for handle in self.config.sample_ids:
            n_sessions = len(self.dataset.iter_sessions(handle))
            counts[handle] = SamplePlanCounts(
                sessions=n_sessions,
                ingest_calls=n_sessions,
                await_ready_calls=n_sessions if async_mutation else 0,
                open_calls=n_sessions + 1,
                close_calls=n_sessions + 1,
                retrieve_calls=1,
            )
        return counts

    # -- run ---------------------------------------------------------------

    def run(self) -> RunOutcome:
        self._pre_run_checks()
        async_mutation = "async_mutation" in set(self.adapter.capabilities())
        manifest = RunManifest(
            run_id=self.run_id,
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
            plan_counts=self._plan_counts(async_mutation),
        )
        from eval.config import ConfigArtifact

        self.store.create(
            manifest,
            ConfigArtifact(
                config=self.config, config_fingerprint=self.config.fingerprint()
            ),
        )
        outcome = RunOutcome(run_id=self.run_id, run_dir=self.store.dir)
        for handle in self.config.sample_ids:
            artifact = self._run_sample(handle, async_mutation)
            outcome.results.append(artifact)
            if artifact.result.qa_status == "failed":
                outcome.failed += 1
        return outcome

    # -- attempt recording -------------------------------------------------

    def _next_attempt_id(self, stage: str) -> str:
        self._attempt_seq += 1
        return f"att_{self._attempt_seq:04d}_{stage}"

    def _record_call(
        self,
        *,
        stage: str,
        method: str,
        namespace: str,
        operation_id: str | None,
        input_payload: dict[str, Any],
        fn: Callable[[], Any],
        error_transform: Callable[[Exception], ErrorInfo] | None = None,
    ) -> tuple[Any | None, StageAttempt, AttemptEntry, ErrorInfo | None]:
        attempt_id = self._next_attempt_id(stage)
        started_at = self._clock()
        t0 = self._monotonic()
        error: ErrorInfo | None = None
        result: Any = None
        with recorded_usage() as recorder:
            try:
                result = fn()
            except MemoryAdapterError as exc:
                error = exc.to_error_info()
            except _StageFailure:
                raise
            except Exception as exc:  # noqa: BLE001 - recorded, never bare
                if error_transform is not None:
                    error = error_transform(exc)
                else:
                    error = ErrorInfo(
                        code="adapter_exception",
                        message=f"{type(exc).__name__}: {exc}",
                        effect="possible",
                        transient=True,
                    )
        elapsed_ms = round((self._monotonic() - t0) * 1000.0, 3)
        ended_at = self._clock()
        usage = recorder.merged()
        outcome = "error" if error is not None else "returned"
        output_json = None if error is not None else _jsonable(result)
        stage_attempt = StageAttempt(
            attempt_id=attempt_id,
            stage=stage,
            operation_id=operation_id,
            outcome=outcome,  # type: ignore[arg-type]
            started_at=started_at,
            ended_at=ended_at,
            elapsed_ms=elapsed_ms,
            input_ref=inline_ref(input_payload),
            output_ref=(None if error is not None else inline_ref(_jsonable(result))),
            error=error,
            usage=usage,
        )
        entry = AttemptEntry(
            attempt_id=attempt_id,
            stage=stage,
            method=method,
            namespace=namespace,
            operation_id=operation_id,
            started_at=started_at,
            ended_at=ended_at,
            elapsed_ms=elapsed_ms,
            input=input_payload,
            output=output_json,
            error=error,
            usage=usage,
        )
        return result, stage_attempt, entry, error

    def _record_lifecycle(
        self, method: str, namespace: str, fn: Callable[[], None]
    ) -> None:
        started_at = self._clock()
        t0 = self._monotonic()
        error: ErrorInfo | None = None
        try:
            fn()
        except MemoryAdapterError as exc:
            error = exc.to_error_info()
        except Exception as exc:  # noqa: BLE001 - logged, never bare
            error = ErrorInfo(
                code="adapter_exception",
                message=f"{type(exc).__name__}: {exc}",
                effect="possible",
                transient=True,
            )
        elapsed_ms = round((self._monotonic() - t0) * 1000.0, 3)
        self._lifecycle_entries.append(
            AttemptEntry(
                attempt_id=f"att_{self._attempt_seq:04d}_{method}",
                stage=None,
                method=method,
                namespace=namespace,
                operation_id=None,
                started_at=started_at,
                ended_at=self._clock(),
                elapsed_ms=elapsed_ms,
                input={"namespace": namespace},
                output=None,
                error=error,
                usage=None,
            )
        )
        self._attempt_seq += 1
        if error is not None:
            # Lifecycle calls failing after the sample already failed are
            # diagnostics; they never overwrite the failed stage.
            pass

    # -- per-sample loop ----------------------------------------------------

    def _run_sample(self, handle: str, async_mutation: bool) -> ResultArtifact:
        namespace = self.dataset.namespace_for(handle, self.config.sample_plan_id)
        self._lifecycle_entries: list[AttemptEntry] = []
        stage_states: dict[str, str] = {
            "ingest": "pending",
            "retrieve": "pending",
            "prepare": "pending",
            "read": "pending",
            "score": "pending",
            "judge": "pending",
        }
        attempts: list[StageAttempt] = []
        entries: list[AttemptEntry] = []
        failed_stage: str | None = None
        failure: ErrorInfo | None = None
        raw_evidence: list[Evidence] | None = None
        prepared: PreparedEvidence | None = None
        sessions = self.dataset.iter_sessions(handle)

        self._record_lifecycle("reset", namespace, lambda: self.adapter.reset(namespace))

        try:
            for i, session in enumerate(sessions):
                self._record_lifecycle(
                    "open", namespace, lambda: self.adapter.open(namespace)
                )
                operation_id = _operation_id(
                    self.run_id, namespace, "ingest", i, session.model_dump(mode="json")
                )
                receipt, attempt, entry, error = self._record_call(
                    stage="ingest",
                    method="ingest",
                    namespace=namespace,
                    operation_id=operation_id,
                    input_payload={
                        "session": session.model_dump(mode="json"),
                        "operation_id": operation_id,
                    },
                    fn=lambda: self.adapter.ingest(namespace, session, operation_id),
                )
                attempts.append(attempt)
                entries.append(entry)
                if error is not None:
                    raise _StageFailure("ingest", error)
                assert receipt is not None
                self._check_receipt_identity(receipt, operation_id, "ingest")
                final: MutationReceipt = receipt
                if receipt.status == "accepted":
                    final = self._await_ready(
                        namespace, operation_id, attempts, entries
                    )
                    stage_states["await_ready"] = "completed"
                if final.status != "completed":
                    failure = final.error or ErrorInfo(
                        code="mutation_not_completed",
                        message=(
                            f"ingest ended in status {final.status!r} without "
                            "a completion confirmation"
                        ),
                        effect="possible",
                        transient=False,
                    )
                    raise _StageFailure("ingest", failure)
                self._record_lifecycle(
                    "close", namespace, lambda: self.adapter.close(namespace)
                )
            stage_states["ingest"] = "completed"

            # Query phase: reopen the persisted space and retrieve.
            self._record_lifecycle(
                "open", namespace, lambda: self.adapter.open(namespace)
            )
            request = self.dataset.build_retrieval_request(
                handle, self.config.evidence_token_budget
            )
            raw, attempt, entry, error = self._record_call(
                stage="retrieve",
                method="retrieve",
                namespace=namespace,
                operation_id=None,
                input_payload={"request": request.model_dump(mode="json")},
                fn=lambda: self.adapter.retrieve(namespace, request),
            )
            attempts.append(attempt)
            entries.append(entry)
            if error is not None:
                raise _StageFailure("retrieve", error)
            assert raw is not None
            raw_evidence = list(raw)
            stage_states["retrieve"] = "completed"

            history = build_history_index(sessions)
            tokenizer = TestCharTokenizer()
            prepared, attempt, entry, error = self._record_call(
                stage="prepare",
                method="prepare_evidence",
                namespace=namespace,
                operation_id=None,
                input_payload={
                    "raw_count": len(raw_evidence),
                    "budget": self.config.evidence_token_budget,
                    "history_messages": history.message_count(),
                },
                fn=lambda: prepare_evidence(
                    raw_evidence,
                    history,
                    budget=self.config.evidence_token_budget,
                    tokenizer=tokenizer,
                ),
                error_transform=_prepare_error_transform,
            )
            attempts.append(attempt)
            entries.append(entry)
            if error is not None:
                raise _StageFailure("prepare", error)
            assert prepared is not None
            stage_states["prepare"] = "completed"
        except _StageFailure as exc:
            failed_stage = exc.stage
            failure = exc.error
            stage_states[exc.stage] = "failed"
        finally:
            # Best-effort close; lifecycle failures land in the log only.
            self._record_lifecycle(
                "close", namespace, lambda: self.adapter.close(namespace)
            )

        entries = self._lifecycle_entries + entries

        log_artifact = AttemptLogArtifact(
            run_id=self.run_id,
            sample_handle=handle,
            namespace=namespace,
            entries=entries,
        )
        raw_artifact = (
            RawEvidenceArtifact(
                run_id=self.run_id,
                sample_handle=handle,
                evidence=raw_evidence,
            )
            if raw_evidence is not None
            else None
        )
        prepared_artifact = (
            PreparedEvidenceArtifact(
                run_id=self.run_id,
                sample_handle=handle,
                prepared=prepared,
            )
            if prepared is not None
            else None
        )
        refs = self.store.write_sample_artifacts(
            handle,
            attempts_log=log_artifact,
            raw_evidence=raw_artifact,
            prepared_evidence=prepared_artifact,
        )
        result = Result(
            run_id=self.run_id,
            sample_handle=handle,
            namespace=namespace,
            config_fingerprint=self.config.fingerprint(),
            suite="qa",
            qa_status="failed" if failed_stage is not None else "pending",
            operation_status=None,
            correct=None,
            attribution=None,
            failed_stage=failed_stage,
            metrics=[],
            stage_states=stage_states,
            artifact_refs=refs,
            attempts=attempts,
        )
        artifact = ResultArtifact(result=result)
        self.store.append_result(artifact)
        return artifact

    def _await_ready(
        self,
        namespace: str,
        operation_id: str,
        attempts: list[StageAttempt],
        entries: list[AttemptEntry],
    ) -> MutationReceipt:
        timeout = self.config.run_params.await_ready_timeout_s
        while True:
            receipt, attempt, entry, error = self._record_call(
                stage="await_ready",
                method="await_ready",
                namespace=namespace,
                operation_id=operation_id,
                input_payload={
                    "operation_id": operation_id,
                    "timeout_s": timeout,
                },
                fn=lambda: self.adapter.await_ready(namespace, operation_id, timeout),
            )
            attempts.append(attempt)
            entries.append(entry)
            if error is not None:
                raise _StageFailure("await_ready", error)
            assert receipt is not None
            self._check_receipt_identity(receipt, operation_id, "await_ready")
            if receipt.status in ("completed", "failed"):
                return receipt
            # still accepted: keep waiting on the SAME operation id

    def _check_receipt_identity(
        self, receipt: MutationReceipt, operation_id: str, stage: str
    ) -> None:
        if receipt.operation_id != operation_id:
            raise _StageFailure(
                stage,
                ErrorInfo(
                    code="receipt_operation_mismatch",
                    message=(
                        f"receipt echoes operation id {receipt.operation_id!r} "
                        f"but the call used {operation_id!r}"
                    ),
                    effect="possible",
                    transient=False,
                ),
            )


def _prepare_error_transform(exc: Exception) -> ErrorInfo:
    if isinstance(exc, PrepareError):
        return ErrorInfo(
            code=f"prepare_{exc.code}",
            message=exc.message,
            effect="none",
            transient=False,
        )
    return ErrorInfo(
        code="prepare_exception",
        message=f"{type(exc).__name__}: {exc}",
        effect="none",
        transient=False,
    )
