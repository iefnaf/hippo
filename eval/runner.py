"""Offline single-question loop: ingest -> reopen -> retrieve -> prepare
-> read -> score/judge -> report.

The M1 runner drives one isolated space per sample through the protocol
fixed in the design document:

1. per-session write: open -> ingest (ONLY the current session) ->
   completion confirmation (await_ready when the receipt is accepted)
   -> close, then the next session;
2. query phase: reopen the same space, build the retrieval request from
   the dataset question_date and the fixed token budget (machine time
   never enters query context), retrieve, then run reader input
   preparation against the cleaned history;
3. answer phase: the fixed reader answers from the exact PreparedEvidence
   the harness retained (never gold, never the private ID mapping);
4. scoring: the scorer computes verifiable recall from that same
   prepared evidence (private ScoringData view only it reads) and asks
   the judge through the protocol adapter; abstention samples keep
   recall N/A but still receive verdicts; invalid scoring data marks the
   sample invalid_input instead of silently excluding it;
5. persistence: per-sample artifacts (attempt log, raw evidence,
   prepared evidence, reader result, judge record, scoring trace) and one
   Result JSONL line per sample; after every planned sample reached a
   terminal state the Reporter writes report.json / report.md.

Reader or judge failures never erase retrieval results: recall metrics
survive in the Result while qa_status records the failed stage. Ingest,
retrieve and prepare failures still score recall with an empty retained
set (zero, not excluded) per the design's denominator rules.

Failure handling (issue #5): transient read-only/reader/judge errors
retry a bounded, configured number of times with every attempt
recorded (stage, reason, timing, usage); answer quality is never a
retry reason. Mutations use stable operation ids: transient errors
with effect=none re-submit the same id; uncertain outcomes
(effect=possible, e.g. a lost submit response) re-submit only on
idempotent adapters and otherwise isolate the old task (namespace
reset) and replay the operation log, bounded by max_sample_replays.
await_ready runs under a runner-side wall clock whose expiry is
recorded as await_ready_timeout. Retries and replays are marked
attempt_kind='retry'/'replay' and their usage reports separately from
logical usage while still counting into run totals.

RunnerBase carries the machinery shared with the operations suite
(eval.operations): attempt recording with usage capture, lifecycle
logging, mutation submission with completion confirmation and the
pre-run capability cross-check. OfflineRunner adds the reader/scorer/
judge pipeline for the qa suite; OperationsRunner (eval.operations)
drives the deterministic operations checks instead.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from eval.contracts.adapter import ErrorInfo, Evidence, MutationReceipt, ResourceUsage
from eval.contracts.common import ContractError, now_utc
from eval.contracts.internal import (
    JudgeRecordArtifact,
    PreparedEvidence,
    PreparedEvidenceArtifact,
    RawEvidenceArtifact,
    ReaderResult,
    ReaderResultArtifact,
    Result,
    ResultArtifact,
    StageAttempt,
)
from eval.judges.base import Judge
from eval.memories.base import MemoryAdapter, MemoryAdapterError
from eval.memories.usage import merge_resource_usage, recorded_usage
from eval.prepare.evidence import (
    HistoryIndex,
    PrepareError,
    build_history_index,
    prepare_evidence,
)
from eval.readers.base import Reader, ReaderError
from eval.runs import (
    AttemptEntry,
    AttemptLogArtifact,
    RunManifest,
    RunStore,
    SamplePlanCounts,
    inline_ref,
)
from eval.prepare.tokens import TestCharTokenizer
from eval.scorers.qa import QAScorer, SampleScoring, ScoringDataError

QueryStages = ("ingest", "await_ready", "retrieve", "prepare", "read", "score", "judge")


class _StageFailure(Exception):
    """Internal control flow: the sample failed at a stage."""

    def __init__(self, stage: str, error: ErrorInfo) -> None:
        super().__init__(f"{stage}: {error.code}")
        self.stage = stage
        self.error = error


class _AwaitTimeout(Exception):
    """await_ready exceeded the runner-side wall-clock budget."""

    def __init__(self, error: ErrorInfo) -> None:
        super().__init__(f"await_ready: {error.code}")
        self.error = error


class _UncertainMutation(Exception):
    """A mutation outcome is uncertain and the adapter offers no
    idempotent re-submit.

    The runner must NOT re-submit the modification; the caller isolates
    the old task (namespace reset) and replays the operation log into
    the fresh space (docs/design/eval-harness.md, 失败处理与恢复).
    """

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
    invalid_input: int = 0
    scored: int = 0
    re_run: int = 0
    reused: int = 0
    re_run_handles: list[str] = field(default_factory=list)
    report_refs: dict[str, str] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "samples": len(self.results),
            "scored": self.scored,
            "failed": self.failed,
            "invalid_input": self.invalid_input,
            "pending": sum(
                1 for r in self.results if r.result.qa_status == "pending"
            ),
            "re_run": self.re_run,
            "reused": self.reused,
            "re_run_handles": list(self.re_run_handles),
            "report": dict(self.report_refs),
        }


@dataclass
class _SampleResume:
    """Checkpointed state of one sample being resumed.

    saved_* carry artifacts of stages already completed by the prior
    invocation; prior attempts/entries are merged into the resumed
    record so retry+replay usage keeps counting into run totals. Every
    call the resume makes is marked attempt_kind='replay'.
    """

    saved_raw: list[Evidence] | None = None
    saved_prepared: PreparedEvidence | None = None
    saved_reader: ReaderResult | None = None
    prior_attempts: list[StageAttempt] = field(default_factory=list)
    prior_entries: list[AttemptEntry] = field(default_factory=list)
    prior_refs: dict[str, str] = field(default_factory=dict)


def _operation_id(run_id: str, namespace: str, stage: str, seq: int, payload: Any) -> str:
    """Stable operation id: run + space + sequence + input checksum."""
    digest = hashlib.sha256(
        json.dumps(_jsonable(payload), sort_keys=True, ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()[:12]
    return f"{run_id}:{namespace}:{stage}:{seq:03d}:{digest}"


def _reader_error_transform(exc: Exception) -> ErrorInfo:
    if isinstance(exc, ReaderError):
        return exc.to_error_info()
    return ErrorInfo(
        code="reader_exception",
        message=f"{type(exc).__name__}: {exc}",
        effect="none",
        transient=True,
    )


class RunnerBase:
    """Shared run machinery: attempt recording, mutations, capability gate.

    Both the offline QA runner and the operations suite record every
    adapter call as a StageAttempt plus a full-payload AttemptEntry,
    submit mutations with completion confirmation (accepted receipts are
    awaited on the same operation id, never re-submitted) and refuse to
    start when the config's capability declaration disagrees with the
    adapter.
    """

    def __init__(
        self,
        *,
        config: Any,
        adapter: MemoryAdapter,
        store: RunStore,
        run_id: str,
        clock: Callable[[], str] = now_utc,
        monotonic: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.adapter = adapter
        self.store = store
        self.run_id = run_id
        self._clock = clock
        self._monotonic = monotonic
        self._sleep = sleep
        self._attempt_seq = 0
        self._lifecycle_entries: list[AttemptEntry] = []

    # -- pre-run checks ----------------------------------------------------

    def _check_capabilities(self) -> None:
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

    # -- attempt recording -------------------------------------------------

    def _next_attempt_id(self, stage: str) -> str:
        self._attempt_seq += 1
        return f"att_{self._attempt_seq:04d}_{stage}"

    def _op_id(self, namespace: str, stage: str, seq: int, payload: Any) -> str:
        return _operation_id(self.run_id, namespace, stage, seq, payload)

    def _backoff_seconds(self, retry_index: int) -> float:
        """Deterministic 1s/4s/16s-shaped backoff (base * 4**k)."""
        return self.config.run_params.backoff_base_s * (4**retry_index)

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
        usage_from: Callable[[Any], ResourceUsage | None] | None = None,
        attempt_kind: str = "logical",
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
        if error is None and usage_from is not None and result is not None:
            reported = usage_from(result)
            if reported is not None:
                usage = (
                    reported
                    if usage is None
                    else merge_resource_usage([usage, reported])
                )
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
            attempt_kind=attempt_kind,  # type: ignore[arg-type]
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
            attempt_kind=attempt_kind,  # type: ignore[arg-type]
        )
        return result, stage_attempt, entry, error

    def _record_lifecycle(
        self,
        method: str,
        namespace: str,
        fn: Callable[[], None],
        attempt_kind: str = "logical",
    ) -> None:
        self._attempt_seq += 1
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
                attempt_kind=attempt_kind,  # type: ignore[arg-type]
            )
        )
        if error is not None:
            # Lifecycle calls failing after the sample already failed are
            # diagnostics; they never overwrite the failed stage.
            pass

    # -- mutation submission ------------------------------------------------

    def _mutation_retry_allowed(self, error: ErrorInfo, idempotent: bool) -> bool:
        """Transient re-submit rule for mutation attempts.

        effect=none (definitely not applied, never will be): bounded
        re-submit with the same operation id. effect=possible (uncertain
        outcome): only an idempotent adapter may see the same id again —
        it dedups to one logical mutation; without idempotency the same
        id must never be re-sent (isolation+replay instead).
        """
        if not error.transient:
            return False
        if error.effect == "none":
            return True
        return error.effect == "possible" and idempotent

    def _record_call_with_retries(
        self,
        *,
        stage: str,
        method: str,
        namespace: str,
        operation_id: str | None,
        input_payload: dict[str, Any],
        fn: Callable[[], Any],
        attempts: list[StageAttempt],
        entries: list[AttemptEntry],
        error_transform: Callable[[Exception], ErrorInfo] | None = None,
        usage_from: Callable[[Any], ResourceUsage | None] | None = None,
        retryable: Callable[[ErrorInfo], bool] | None = None,
        attempt_kind: str = "logical",
    ) -> Any:
        """Record a call with bounded retries on transient errors.

        Used for read-only adapter calls and reader answers: every
        attempt (failed or not) is recorded with stage, reason, timing
        and usage; retries are marked attempt_kind='retry' so recovery
        usage is reported separately. A non-transient error fails the
        stage immediately, and nothing about answer quality is ever a
        retry reason.
        """
        retries = 0
        kind = attempt_kind
        while True:
            result, attempt, entry, error = self._record_call(
                stage=stage,
                method=method,
                namespace=namespace,
                operation_id=operation_id,
                input_payload=input_payload,
                fn=fn,
                error_transform=error_transform,
                usage_from=usage_from,
                attempt_kind=kind,
            )
            attempts.append(attempt)
            entries.append(entry)
            if error is None:
                return result
            if (
                retryable is not None
                and retryable(error)
                and retries < self.config.run_params.max_retries
            ):
                self._sleep(self._backoff_seconds(retries))
                retries += 1
                kind = "retry"
                continue
            raise _StageFailure(stage, error)

    def _submit_mutation(
        self,
        *,
        stage: str,
        method: str,
        namespace: str,
        operation_id: str,
        input_payload: dict[str, Any],
        fn: Callable[[], Any],
        attempts: list[StageAttempt],
        entries: list[AttemptEntry],
        stage_states: dict[str, str] | None = None,
        attempt_kind: str = "logical",
    ) -> MutationReceipt:
        """Submit one mutation and confirm completion before returning.

        Retry policy (docs/design/eval-harness.md, 失败处理与恢复):
        transient+effect=none re-submits the SAME operation id; an
        uncertain outcome (effect=possible, e.g. a lost submit response)
        re-submits only on idempotent adapters (one logical mutation)
        and otherwise raises _UncertainMutation so the caller isolates
        the old task and replays; an already-accepted async task is
        only ever awaited on its own operation id; await_ready is
        bounded by a runner-side wall clock and its timeout is recorded
        as a stage failure (await_ready_timeout).
        """
        idempotent = "idempotent_mutation" in set(self.adapter.capabilities())
        max_retries = self.config.run_params.max_retries
        tries = 0
        kind = attempt_kind
        while True:
            receipt, attempt, entry, error = self._record_call(
                stage=stage,
                method=method,
                namespace=namespace,
                operation_id=operation_id,
                input_payload=input_payload,
                fn=fn,
                attempt_kind=kind,
            )
            attempts.append(attempt)
            entries.append(entry)
            if error is not None:
                if (
                    self._mutation_retry_allowed(error, idempotent)
                    and tries < max_retries
                ):
                    self._sleep(self._backoff_seconds(tries))
                    tries += 1
                    kind = "retry"
                    continue
                if error.effect == "possible" and not idempotent:
                    raise _UncertainMutation(stage, error)
                raise _StageFailure(stage, error)
            assert receipt is not None
            self._check_receipt_identity(receipt, operation_id, stage)
            final: MutationReceipt = receipt
            if receipt.status == "accepted":
                try:
                    final = self._await_ready(
                        namespace, operation_id, attempts, entries, attempt_kind=kind
                    )
                except _AwaitTimeout as exc:
                    if idempotent and tries < max_retries:
                        # Same-id re-submit: for an idempotent async
                        # backend this is a status query of the original
                        # operation, never a new mutation.
                        tries += 1
                        kind = "retry"
                        continue
                    if not idempotent:
                        raise _UncertainMutation("await_ready", exc.error) from exc
                    raise _StageFailure("await_ready", exc.error) from exc
                if stage_states is not None:
                    stage_states["await_ready"] = "completed"
            if final.status != "completed":
                failure = final.error or ErrorInfo(
                    code="mutation_not_completed",
                    message=(
                        f"{method} ended in status {final.status!r} without "
                        "a completion confirmation"
                    ),
                    effect="possible",
                    transient=False,
                )
                raise _StageFailure(stage, failure)
            if stage_states is not None:
                stage_states[stage] = "completed"
            return final

    def _await_ready(
        self,
        namespace: str,
        operation_id: str,
        attempts: list[StageAttempt],
        entries: list[AttemptEntry],
        attempt_kind: str = "logical",
    ) -> MutationReceipt:
        """Poll the SAME operation id under a runner-side wall clock.

        The adapter timeout bounds a single poll; this loop bounds the
        TOTAL wait via self._monotonic so a backend that accepts but
        never finishes cannot hang the run (issue #2 close-out note).
        Exceeding the budget records an await_ready_timeout attempt
        (stage, reason, timing, usage) and raises _AwaitTimeout —
        waiting past the configured budget is a recorded failure, never
        an unbounded hang.
        """
        timeout = self.config.run_params.await_ready_timeout_s
        deadline = self._monotonic() + timeout
        while True:
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                self._record_await_timeout(
                    namespace,
                    operation_id,
                    attempts,
                    entries,
                    waited_s=timeout,
                    attempt_kind=attempt_kind,
                )
                raise _AwaitTimeout(
                    ErrorInfo(
                        code="await_ready_timeout",
                        message=(
                            f"operation {operation_id!r} did not reach a "
                            f"terminal state within the runner-side wall-"
                            f"clock budget of {timeout:g}s; recorded as a "
                            "failure (effect uncertain)"
                        ),
                        effect="possible",
                        transient=False,
                    )
                )
            poll_timeout = round(min(timeout, remaining), 6)
            receipt, attempt, entry, error = self._record_call(
                stage="await_ready",
                method="await_ready",
                namespace=namespace,
                operation_id=operation_id,
                input_payload={
                    "operation_id": operation_id,
                    "timeout_s": poll_timeout,
                },
                fn=lambda: self.adapter.await_ready(
                    namespace, operation_id, poll_timeout
                ),
                attempt_kind=attempt_kind,
            )
            attempts.append(attempt)
            entries.append(entry)
            if error is not None:
                if error.transient:
                    # Keep polling the SAME operation within the budget.
                    self._sleep(min(remaining, self._backoff_seconds(0)))
                    continue
                raise _StageFailure("await_ready", error)
            assert receipt is not None
            self._check_receipt_identity(receipt, operation_id, "await_ready")
            if receipt.status in ("completed", "failed"):
                return receipt
            # still accepted: keep waiting on the SAME operation id

    def _record_await_timeout(
        self,
        namespace: str,
        operation_id: str,
        attempts: list[StageAttempt],
        entries: list[AttemptEntry],
        *,
        waited_s: float,
        attempt_kind: str,
    ) -> None:
        """Record the wall-clock timeout itself as a failed attempt."""
        error = ErrorInfo(
            code="await_ready_timeout",
            message=(
                f"operation {operation_id!r} did not reach a terminal state "
                f"within the runner-side wall-clock budget of {waited_s:g}s; "
                "recorded as a failure (effect uncertain)"
            ),
            effect="possible",
            transient=False,
        )
        started_at = self._clock()
        attempt_id = self._next_attempt_id("await_ready")
        attempts.append(
            StageAttempt(
                attempt_id=attempt_id,
                stage="await_ready",
                operation_id=operation_id,
                outcome="error",
                started_at=started_at,
                ended_at=self._clock(),
                elapsed_ms=round(waited_s * 1000.0, 3),
                input_ref=inline_ref(
                    {"operation_id": operation_id, "timeout_s": waited_s}
                ),
                output_ref=None,
                error=error,
                usage=None,
                attempt_kind=attempt_kind,  # type: ignore[arg-type]
            )
        )
        entries.append(
            AttemptEntry(
                attempt_id=attempt_id,
                stage="await_ready",
                method="await_ready.wall_clock",
                namespace=namespace,
                operation_id=operation_id,
                started_at=started_at,
                ended_at=self._clock(),
                elapsed_ms=round(waited_s * 1000.0, 3),
                input={
                    "operation_id": operation_id,
                    "timeout_s": waited_s,
                },
                output=None,
                error=error,
                usage=None,
                attempt_kind=attempt_kind,  # type: ignore[arg-type]
            )
        )

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


class OfflineRunner(RunnerBase):
    """Runs the offline QA loop for every configured sample.

    After prepare the fixed reader answers from the exact retained
    evidence and the scorer computes recall plus the judge verdict; the
    operations suite never uses this runner (see eval.operations).
    """

    def __init__(
        self,
        *,
        config: Any,
        dataset: Any,
        adapter: MemoryAdapter,
        reader: Reader,
        judge: Judge,
        store: RunStore,
        run_id: str,
        clock: Callable[[], str] = now_utc,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> None:
        super().__init__(
            config=config,
            adapter=adapter,
            store=store,
            run_id=run_id,
            clock=clock,
            monotonic=monotonic,
        )
        self.dataset = dataset
        self.reader = reader
        self._scorer = QAScorer(
            dataset=dataset,
            judge=judge,
            extractive_declared="extractive_evidence"
            in set(config.memory.capabilities),
            baseline_kind=config.memory.baseline_kind,
            protocol_id=config.judge.protocol_id,
            clock=clock,
            monotonic=monotonic,
            max_retries=config.run_params.max_retries,
            backoff_base_s=config.run_params.backoff_base_s,
            sleep=self._sleep,
        )

    # -- pre-run checks ----------------------------------------------------

    def _pre_run_checks(self) -> None:
        self.dataset.require_handles(self.config.sample_ids)
        self._check_capabilities()

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
                reader_calls=1,
                judge_calls=1,
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
        return self._execute(async_mutation, resume=None)

    def resume(self) -> RunOutcome:
        """Continue a checkpointed run; only unfinished work executes.

        verify_resume_compatibility gates reuse (config fingerprint,
        space identity, checkpoint schema versions); terminal samples
        and their artifacts are reused untouched, failed samples resume
        from their first unfinished stage using saved evidence, and all
        recovery calls are reported as replay usage.
        """
        self._pre_run_checks()
        async_mutation = "async_mutation" in set(self.adapter.capabilities())
        from eval.runs import verify_resume_compatibility

        prior = verify_resume_compatibility(
            self.config,
            self.store,
            namespace_for_handle=lambda handle: self.dataset.namespace_for(
                handle, self.config.sample_plan_id
            ),
        )
        self.store.open_for_resume()
        outcome = self._execute(async_mutation, resume=prior["results"])
        self.store.append_resume_marker(
            {
                "resumed_at": self._clock(),
                "run_id": self.run_id,
                "re_run": outcome.re_run,
                "reused": outcome.reused,
                "re_run_handles": outcome.re_run_handles,
            }
        )
        return outcome

    def _terminal(self, result: Result) -> bool:
        # scored and invalid_input are terminal: invalid scoring data is
        # a data error to fix with a NEW run, not something to retry.
        return result.qa_status in ("scored", "invalid_input")

    def _load_ref_model(self, ref: str | None, model: Any) -> Any | None:
        if ref is None:
            return None
        doc = json.loads(self.store.resolve_ref(ref).read_text(encoding="utf-8"))
        return model.load_json(json.dumps(doc))

    def _resume_plan(self, prior: Any) -> _SampleResume:
        """Decide which saved stages a resumed sample may reuse."""
        result = prior.result
        plan = _SampleResume(
            prior_attempts=list(result.attempts),
            prior_refs=dict(result.artifact_refs),
        )
        log = self._load_ref_model(
            result.artifact_refs.get("attempts"), AttemptLogArtifact
        )
        plan.prior_entries = list(log.entries) if log is not None else []
        saved_raw = self._load_ref_model(
            result.artifact_refs.get("raw_evidence"), RawEvidenceArtifact
        )
        saved_prepared = self._load_ref_model(
            result.artifact_refs.get("prepared_evidence"),
            PreparedEvidenceArtifact,
        )
        saved_reader = self._load_ref_model(
            result.artifact_refs.get("reader_result"), ReaderResultArtifact
        )
        failed = result.failed_stage
        if failed in ("read", "score", "judge") and saved_prepared is not None:
            # Saved retrieval evidence is reused directly; the memory
            # space is never rebuilt for reader/judge recovery.
            plan.saved_prepared = saved_prepared.prepared
            if saved_raw is not None:
                plan.saved_raw = list(saved_raw.evidence)
            if failed in ("score", "judge") and saved_reader is not None:
                plan.saved_reader = saved_reader.result
            return plan
        if failed == "prepare" and saved_raw is not None:
            plan.saved_raw = list(saved_raw.evidence)
        # everything else (no line, pending, or failed before retrieve):
        # full re-run of the sample from its own operation log
        return plan

    def _execute(
        self, async_mutation: bool, resume: dict[str, Any] | None
    ) -> RunOutcome:
        outcome = RunOutcome(run_id=self.run_id, run_dir=self.store.dir)
        for handle in self.config.sample_ids:
            prior = resume.get(handle) if resume is not None else None
            if prior is not None and self._terminal(prior.result):
                outcome.results.append(prior)
                outcome.reused += 1
                continue
            plan = self._resume_plan(prior) if prior is not None else None
            artifact = self._run_sample(handle, async_mutation, resume=plan)
            outcome.results.append(artifact)
            if artifact.result.qa_status == "failed":
                outcome.failed += 1
            elif artifact.result.qa_status == "invalid_input":
                outcome.invalid_input += 1
            elif artifact.result.qa_status == "scored":
                outcome.scored += 1
            if resume is not None:
                outcome.re_run += 1
                outcome.re_run_handles.append(handle)
        from eval.report import Reporter, render_markdown

        reporter = Reporter(self.store.dir)
        report = reporter.build()
        outcome.report_refs = self.store.write_report(
            json.dumps(report, ensure_ascii=False, indent=2),
            render_markdown(report),
            overwrite=resume is not None,
        )
        return outcome

    # -- per-sample loop ----------------------------------------------------

    def _ingest_phase(
        self,
        handle: str,
        namespace: str,
        sessions: list[Any],
        attempts: list[StageAttempt],
        entries: list[AttemptEntry],
        stage_states: dict[str, str],
        base_kind: str = "logical",
    ) -> None:
        """open -> ingest (exactly the current session) -> completion
        confirmation -> close, per session.

        An uncertain mutation outcome on a NON-idempotent adapter raises
        _UncertainMutation: the namespace is then reset (isolating the
        old task together with the uncertain space) and the whole
        operation log is replayed into the fresh space, bounded by
        run_params.max_sample_replays. Replay calls are marked
        attempt_kind='replay' so their usage reports as recovery.
        """
        replays = 0
        kind = base_kind
        while True:
            try:
                for i, session in enumerate(sessions):
                    self._record_lifecycle(
                        "open",
                        namespace,
                        lambda: self.adapter.open(namespace),
                        attempt_kind=kind,
                    )
                    operation_id = self._op_id(
                        namespace, "ingest", i, session.model_dump(mode="json")
                    )
                    self._submit_mutation(
                        stage="ingest",
                        method="ingest",
                        namespace=namespace,
                        operation_id=operation_id,
                        input_payload={
                            "session": session.model_dump(mode="json"),
                            "operation_id": operation_id,
                        },
                        fn=lambda: self.adapter.ingest(
                            namespace, session, operation_id
                        ),
                        attempts=attempts,
                        entries=entries,
                        stage_states=stage_states,
                        attempt_kind=kind,
                    )
                    self._record_lifecycle(
                        "close",
                        namespace,
                        lambda: self.adapter.close(namespace),
                        attempt_kind=kind,
                    )
                stage_states["ingest"] = "completed"
                return
            except _UncertainMutation as exc:
                if exc.stage in stage_states:
                    stage_states[exc.stage] = "failed"
                if replays >= self.config.run_params.max_sample_replays:
                    raise _StageFailure(exc.stage, exc.error) from exc
                replays += 1
                # Isolation: discard the uncertain space; the adapter
                # quarantines its in-flight tasks with it.
                self._record_lifecycle(
                    "reset",
                    namespace,
                    lambda: self.adapter.reset(namespace),
                    attempt_kind="replay",
                )
                kind = "replay"

    def _retrieve_stage(
        self,
        handle: str,
        namespace: str,
        attempts: list[StageAttempt],
        entries: list[AttemptEntry],
        attempt_kind: str = "logical",
    ) -> list[Evidence]:
        """Read-only retrieve with bounded transient retries."""
        request = self.dataset.build_retrieval_request(
            handle, self.config.evidence_token_budget
        )
        raw = self._record_call_with_retries(
            stage="retrieve",
            method="retrieve",
            namespace=namespace,
            operation_id=None,
            input_payload={"request": request.model_dump(mode="json")},
            fn=lambda: self.adapter.retrieve(namespace, request),
            attempts=attempts,
            entries=entries,
            retryable=lambda error: error.transient,
            attempt_kind=attempt_kind,
        )
        assert raw is not None
        return list(raw)

    def _prepare_stage(
        self,
        namespace: str,
        raw_evidence: list[Evidence],
        history: HistoryIndex,
        attempts: list[StageAttempt],
        entries: list[AttemptEntry],
        attempt_kind: str = "logical",
    ) -> PreparedEvidence:
        """Budget preparation is deterministic: no retries, no patching."""
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
            attempt_kind=attempt_kind,
        )
        attempts.append(attempt)
        entries.append(entry)
        if error is not None:
            raise _StageFailure("prepare", error)
        assert prepared is not None
        return prepared

    def _run_sample(
        self,
        handle: str,
        async_mutation: bool,
        resume: _SampleResume | None = None,
    ) -> ResultArtifact:
        namespace = self.dataset.namespace_for(handle, self.config.sample_plan_id)
        self._lifecycle_entries = []
        stage_states: dict[str, str] = {
            "ingest": "pending",
            "retrieve": "pending",
            "prepare": "pending",
            "read": "pending",
            "score": "pending",
            "judge": "pending",
        }
        base_kind = "replay" if resume is not None else "logical"
        attempts: list[StageAttempt] = (
            list(resume.prior_attempts) if resume is not None else []
        )
        entries: list[AttemptEntry] = (
            list(resume.prior_entries) if resume is not None else []
        )
        failed_stage: str | None = None
        failure: ErrorInfo | None = None
        invalid_input = False
        raw_evidence: list[Evidence] | None = (
            resume.saved_raw if resume is not None else None
        )
        prepared: PreparedEvidence | None = (
            resume.saved_prepared if resume is not None else None
        )
        reader_result: ReaderResult | None = (
            resume.saved_reader if resume is not None else None
        )
        new_raw = False
        new_prepared = False
        new_reader = False
        if resume is not None:
            if resume.saved_prepared is not None:
                stage_states["retrieve"] = "completed"
                stage_states["prepare"] = "completed"
            elif resume.saved_raw is not None:
                stage_states["retrieve"] = "completed"
        scoring: SampleScoring | None = None
        sessions = self.dataset.iter_sessions(handle)
        # The memory space is rebuilt only when no saved evidence can
        # carry the sample forward (nothing saved, or the failure was
        # before retrieve completed).
        memory_phase = (
            resume is None
            or (resume.saved_raw is None and resume.saved_prepared is None)
        )

        if memory_phase:
            self._record_lifecycle(
                "reset", namespace, lambda: self.adapter.reset(namespace)
            )
            try:
                self._ingest_phase(
                    handle,
                    namespace,
                    sessions,
                    attempts,
                    entries,
                    stage_states,
                    base_kind=base_kind,
                )

                # Query phase: reopen the persisted space and retrieve.
                self._record_lifecycle(
                    "open", namespace, lambda: self.adapter.open(namespace)
                )
                raw = self._retrieve_stage(
                    handle, namespace, attempts, entries, attempt_kind=base_kind
                )
                raw_evidence = list(raw)
                new_raw = True
                stage_states["retrieve"] = "completed"

                history = build_history_index(sessions)
                prepared = self._prepare_stage(
                    namespace, raw_evidence, history, attempts, entries,
                    attempt_kind=base_kind,
                )
                new_prepared = True
                stage_states["prepare"] = "completed"
            except _StageFailure as exc:
                failed_stage = exc.stage
                failure = exc.error
                stage_states[exc.stage] = "failed"
        elif resume is not None and resume.saved_prepared is None:
            # Failed at prepare with the raw return already saved: re-run
            # the deterministic preparation only; no adapter calls.
            try:
                history = build_history_index(sessions)
                assert raw_evidence is not None
                prepared = self._prepare_stage(
                    namespace, raw_evidence, history, attempts, entries,
                    attempt_kind=base_kind,
                )
                new_prepared = True
                stage_states["prepare"] = "completed"
            except _StageFailure as exc:
                failed_stage = exc.stage
                failure = exc.error
                stage_states[exc.stage] = "failed"

        try:
            # Answer phase: the reader consumes the exact retained evidence.
            question = self.dataset.get_question(handle)
            if failed_stage is None and prepared is not None:
                if reader_result is not None:
                    # Resumed with an accepted reader answer: only the
                    # scoring/judge stages remain.
                    stage_states["read"] = "completed"
                else:
                    try:
                        reader_result = self._record_call_with_retries(
                            stage="read",
                            method="reader.answer",
                            namespace=namespace,
                            operation_id=None,
                            input_payload={
                                "question": question.model_dump(mode="json"),
                                "prepared_token_count": prepared.token_count,
                                "prepared_units": len(prepared.items),
                                "tokenizer_id": prepared.tokenizer_id,
                            },
                            fn=lambda: self.reader.answer(question, prepared),
                            attempts=attempts,
                            entries=entries,
                            error_transform=_reader_error_transform,
                            usage_from=lambda r: r.usage,
                            retryable=lambda error: error.transient,
                            attempt_kind=base_kind,
                        )
                        new_reader = True
                        stage_states["read"] = "completed"
                    except _StageFailure as exc:
                        failed_stage = exc.stage
                        failure = exc.error
                        stage_states["read"] = "failed"
                        reader_result = None

            # Scoring: recall from the same prepared evidence + judge
            # verdict. Runs even after ingest/retrieve/prepare/read
            # failures: recall is computed with an empty retained set
            # (zero), and verdicts are only attempted when a reader
            # answer exists.
            try:
                scoring = self._scorer.score_sample(
                    run_id=self.run_id,
                    handle=handle,
                    question=question,
                    prepared=prepared,
                    reader_result=reader_result,
                )
                stage_states["score"] = "completed"
            except ScoringDataError as exc:
                invalid_input = True
                failed_stage = "score"
                failure = ErrorInfo(
                    code=exc.code,
                    message=exc.message,
                    effect="none",
                    transient=False,
                )
                stage_states["score"] = "failed"
                scoring = None
                self._lifecycle_entries.append(
                    AttemptEntry(
                        attempt_id=self._next_attempt_id("score"),
                        stage="score",
                        method="dataset.get_scoring_data",
                        namespace=namespace,
                        operation_id=None,
                        started_at=self._clock(),
                        ended_at=self._clock(),
                        elapsed_ms=0.0,
                        input={"sample_handle": handle},
                        output=None,
                        error=failure,
                        usage=None,
                    )
                )

            if scoring is not None and scoring.judge_attempts:
                for idx, call in enumerate(scoring.judge_attempts):
                    attempt_id = self._next_attempt_id("judge")
                    kind = "logical" if idx == 0 else "retry"
                    attempts.append(
                        StageAttempt(
                            attempt_id=attempt_id,
                            stage="judge",
                            operation_id=None,
                            outcome=("error" if call.error is not None else "returned"),
                            started_at=call.started_at,
                            ended_at=call.ended_at,
                            elapsed_ms=call.elapsed_ms,
                            input_ref=inline_ref(
                                call.request.model_dump(mode="json")
                            ),
                            output_ref=(
                                None
                                if call.error is not None
                                else inline_ref(call.result.model_dump(mode="json"))
                            ),
                            error=call.error,
                            usage=call.usage,
                            attempt_kind=kind,
                        )
                    )
                    entries.append(
                        AttemptEntry(
                            attempt_id=attempt_id,
                            stage="judge",
                            method="judge.evaluate",
                            namespace=namespace,
                            operation_id=None,
                            started_at=call.started_at,
                            ended_at=call.ended_at,
                            elapsed_ms=call.elapsed_ms,
                            input={"request": call.request.model_dump(mode="json")},
                            output=(
                                None
                                if call.error is not None
                                else call.result.model_dump(mode="json")
                            ),
                            error=call.error,
                            usage=call.usage,
                            attempt_kind=kind,
                        )
                    )
                final_call = scoring.judge_attempts[-1]
                if final_call.error is not None:
                    failed_stage = "judge"
                    failure = final_call.error
                    stage_states["judge"] = "failed"
                else:
                    stage_states["judge"] = "completed"
        finally:
            # Best-effort close; lifecycle failures land in the log only.
            if memory_phase:
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
            if raw_evidence is not None and new_raw
            else None
        )
        prepared_artifact = (
            PreparedEvidenceArtifact(
                run_id=self.run_id,
                sample_handle=handle,
                prepared=prepared,
            )
            if prepared is not None and new_prepared
            else None
        )
        reader_artifact = (
            ReaderResultArtifact(
                run_id=self.run_id,
                sample_handle=handle,
                result=reader_result,
            )
            if reader_result is not None and new_reader
            else None
        )
        successful_judge = (
            next(
                (c for c in scoring.judge_attempts if c.result is not None),
                None,
            )
            if scoring is not None
            else None
        )
        judge_artifact = (
            JudgeRecordArtifact(
                run_id=self.run_id,
                sample_handle=handle,
                request=successful_judge.request,
                result=successful_judge.result,
            )
            if successful_judge is not None
            else None
        )
        refs = self.store.write_sample_artifacts(
            handle,
            attempts_log=log_artifact,
            raw_evidence=raw_artifact,
            prepared_evidence=prepared_artifact,
            reader_result=reader_artifact,
            judge_record=judge_artifact,
            scoring_trace=(
                scoring.trace if scoring is not None else None
            ),
        )
        if resume is not None:
            # Reused checkpoint artifacts keep their original refs;
            # this round's writes (always under a .r<N> directory in
            # resume mode) override whatever it re-produced.
            merged = dict(resume.prior_refs)
            merged.update(refs)
            refs = merged

        metrics = scoring.metrics if scoring is not None else []
        if failed_stage is None:
            assert scoring is not None and scoring.correct is not None
            qa_status = "scored"
            correct: bool | None = scoring.correct
            attribution = scoring.attribution
        else:
            qa_status = "invalid_input" if invalid_input else "failed"
            correct = None
            attribution = None
        result = Result(
            run_id=self.run_id,
            sample_handle=handle,
            namespace=namespace,
            config_fingerprint=self.config.fingerprint(),
            suite="qa",
            qa_status=qa_status,  # type: ignore[arg-type]
            operation_status=None,
            correct=correct,
            attribution=attribution,  # type: ignore[arg-type]
            failed_stage=failed_stage,
            metrics=metrics,
            stage_states=stage_states,
            artifact_refs=refs,
            attempts=attempts,
        )
        from eval.metrics import validate_result_metrics

        validate_result_metrics(result)
        artifact = ResultArtifact(result=result)
        self.store.append_result(artifact)
        return artifact


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
