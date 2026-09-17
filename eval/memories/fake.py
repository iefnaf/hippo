"""Configurable fake memory adapter for the M1 offline loop.

FakeMemorySpec fixes a capability combination BEFORE a run (sync/async
mutation, extractive/generated evidence, idempotent vs not, optional
state inspection); FakeMemoryAdapter implements the base protocol
deterministically and offline:

- stores cleaned messages per namespace; data survives close() and is
  observable after open() again (persistence), while process-internal
  state is dropped;
- sync mode returns completed receipts from ingest; async mode returns
  accepted and completes after async_lag await_ready polls, which never
  re-submit;
- idempotent mode registers operation ids: same id + same input returns
  the stored receipt, same id + different input is rejected;
- close() with pending operations is rejected: completion confirmation
  cannot be replaced by closing;
- retrieve never mutates; 'match' ranks by deterministic query/content
  overlap, 'flood' returns every stored unit (used to exercise budget
  truncation);
- inspect exposes per-message stable memory states (current/unknown).

The journal records every call with its adapter-visible arguments for
isolation tests; received_requests keeps RetrievalRequest objects.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from eval.contracts.adapter import (
    Evidence,
    MemoryState,
    MutationReceipt,
    ResourceUsage,
    RetrievalRequest,
    Session,
    SourceRef,
    SourceSpan,
)
from eval.contracts.common import ContractModel
from eval.memories.base import MemoryAdapterError
from eval.memories.usage import report_usage

FAKE_MEMORY_NAME = "fake-memory"

_EVIDENCE_KINDS = ("extractive", "generated")


def _cjk_chars(text: str) -> set[str]:
    return {ch for ch in text if "\u4e00" <= ch <= "\u9fff"}


def _ascii_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _overlap_score(query: str, content: str) -> float:
    """Deterministic relevance: CJK char overlap + 2 * word overlap."""
    q_cjk, c_cjk = _cjk_chars(query), _cjk_chars(content)
    q_w, c_w = _ascii_words(query), _ascii_words(content)
    return float(len(q_cjk & c_cjk) + 2 * len(q_w & c_w))


class FakeMemorySpec(ContractModel):
    """A fake adapter profile; fixed before the run and part of the config."""

    mutation_mode: Literal["sync", "async"] = "sync"
    evidence_kinds: tuple[Literal["extractive", "generated"], ...] = (
        "extractive",
    )
    retrieval_mode: Literal["match", "flood"] = "match"
    idempotent: bool = False
    state_inspection: bool = True
    async_lag: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def _kinds_nonempty(self) -> Self:
        if not self.evidence_kinds:
            raise ValueError("evidence_kinds must declare at least one kind")
        if len(set(self.evidence_kinds)) != len(self.evidence_kinds):
            raise ValueError("evidence_kinds must be deduplicated")
        return self

    def capabilities(self) -> frozenset[str]:
        caps: set[str] = set()
        if "extractive" in self.evidence_kinds:
            caps.add("extractive_evidence")
        if "generated" in self.evidence_kinds:
            caps.add("generated_evidence")
        if self.mutation_mode == "async":
            # async_mutation must come with operation_status (contract).
            caps.update({"async_mutation", "operation_status"})
        if self.idempotent:
            caps.add("idempotent_mutation")
        if self.state_inspection:
            caps.add("state_inspection")
        return frozenset(caps)

    @classmethod
    def from_memory_plan(cls, plan: Any) -> "FakeMemorySpec":
        if plan.name != FAKE_MEMORY_NAME:
            raise MemoryAdapterError(
                "unsupported_memory_name",
                f"the fake factory only builds {FAKE_MEMORY_NAME!r}, got "
                f"{plan.name!r}",
                effect="none",
            )
        raw = dict(plan.config)
        kinds = raw.pop("evidence_kinds", ("extractive",))
        if isinstance(kinds, str):
            kinds = (kinds,)
        return cls(evidence_kinds=tuple(kinds), **raw)


class _StoredMessage:
    __slots__ = ("memory_id", "session_id", "msg_id", "content", "occurred_at", "order")

    def __init__(
        self,
        memory_id: str,
        session_id: str,
        msg_id: str,
        content: str,
        occurred_at: str,
        order: tuple[int, int, int],
    ) -> None:
        self.memory_id = memory_id
        self.session_id = session_id
        self.msg_id = msg_id
        self.content = content
        self.occurred_at = occurred_at
        self.order = order


class _PendingOp:
    __slots__ = ("operation_id", "namespace", "remaining", "receipt", "input_digest")

    def __init__(
        self,
        operation_id: str,
        namespace: str,
        remaining: int,
        receipt: MutationReceipt,
        input_digest: str,
    ) -> None:
        self.operation_id = operation_id
        self.namespace = namespace
        self.remaining = remaining
        self.receipt = receipt
        self.input_digest = input_digest


def _digest(payload: Any) -> str:
    import json

    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class FakeMemoryAdapter:
    """Offline deterministic adapter implementing the base protocol."""

    def __init__(self, spec: FakeMemorySpec) -> None:
        self.spec = spec
        # namespace -> {memory_id -> stored message}; survives close().
        self._spaces: dict[str, dict[str, _StoredMessage]] = {}
        self._open: set[str] = set()
        self._pending: dict[str, _PendingOp] = {}
        self._idem_registry: dict[str, tuple[str, MutationReceipt]] = {}
        self.journal: list[dict[str, Any]] = []
        self.received_requests: list[RetrievalRequest] = []

    # -- helpers -----------------------------------------------------------

    def _log(self, method: str, **payload: Any) -> None:
        self.journal.append({"method": method, **payload})

    def _require_open(self, namespace: str) -> None:
        if namespace not in self._open:
            raise MemoryAdapterError(
                "namespace_not_open",
                f"namespace {namespace!r} must be open()ed before use",
                effect="none",
            )

    def _space(self, namespace: str) -> dict[str, _StoredMessage]:
        return self._spaces.setdefault(namespace, {})



    def _memory_id(self, namespace: str, session_id: str, msg_id: str) -> str:
        seed = f"{namespace}|{session_id}|{msg_id}"
        return "mem_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]

    def _receipt(
        self,
        operation_id: str,
        status: str,
        memory_ids: list[str],
        sources: list[SourceRef],
        usage: ResourceUsage | None,
    ) -> MutationReceipt:
        return MutationReceipt(
            operation_id=operation_id,
            status=status,  # type: ignore[arg-type]
            memory_ids=memory_ids,
            sources=sources,
            error=None,
            usage=usage,
        )

    # -- protocol ----------------------------------------------------------

    def capabilities(self) -> set[str]:
        return set(self.spec.capabilities())

    def reset(self, namespace: str) -> None:
        self._log("reset", namespace=namespace)
        self._spaces.pop(namespace, None)
        self._open.discard(namespace)
        for op_id in [
            oid for oid, p in self._pending.items() if p.namespace == namespace
        ]:
            del self._pending[op_id]

    def open(self, namespace: str) -> None:
        self._log("open", namespace=namespace)
        self._open.add(namespace)

    def ingest(
        self, namespace: str, session: Session, operation_id: str
    ) -> MutationReceipt:
        self._require_open(namespace)
        payload = session.model_dump(mode="json")
        self._log(
            "ingest",
            namespace=namespace,
            session=payload,
            operation_id=operation_id,
        )
        digest = _digest(payload)
        if self.spec.idempotent and operation_id in self._idem_registry:
            known_digest, receipt = self._idem_registry[operation_id]
            if known_digest != digest:
                raise MemoryAdapterError(
                    "operation_id_input_conflict",
                    f"operation {operation_id!r} was already submitted with "
                    "different input; the same id must map to identical input",
                    effect="none",
                )
            # Same id + same input: exactly one logical mutation; the
            # terminal receipt is returned without storing again.
            return receipt

        space = self._space(namespace)
        memory_ids: list[str] = []
        sources: list[SourceRef] = []
        session_order = len({m.session_id for m in space.values()})
        for i, message in enumerate(session.messages):
            base_id = self._memory_id(namespace, session.session_id, message.msg_id)
            mid = base_id
            if mid in space:
                # A genuinely new write of the same message (different
                # operation) creates a NEW logical entry with a stable
                # suffixed id; evidence spans still reference the cleaned
                # (session_id, msg_id), not the memory id.
                suffix = 2
                while f"{base_id}~{suffix}" in space:
                    suffix += 1
                mid = f"{base_id}~{suffix}"
            space[mid] = _StoredMessage(
                memory_id=mid,
                session_id=session.session_id,
                msg_id=message.msg_id,
                content=message.content,
                occurred_at=session.occurred_at,
                order=(session_order, i, suffix if base_id in space else 0),
            )
            memory_ids.append(mid)
            sources.append(
                SourceRef(session_id=session.session_id, msg_id=message.msg_id)
            )

        usage = ResourceUsage(
            input_tokens=sum(len(m.content) for m in session.messages),
            output_tokens=0,
            llm_call_count=0,
        )
        report_usage(usage)
        if self.spec.mutation_mode == "sync":
            receipt = self._receipt(
                operation_id, "completed", memory_ids, sources, usage
            )
            if self.spec.idempotent:
                self._idem_registry[operation_id] = (digest, receipt)
            return receipt
        receipt = self._receipt(operation_id, "accepted", memory_ids, sources, usage)
        self._pending[operation_id] = _PendingOp(
            operation_id, namespace, self.spec.async_lag, receipt, digest
        )
        return receipt

    def await_ready(
        self, namespace: str, operation_id: str, timeout: float
    ) -> MutationReceipt:
        self._require_open(namespace)
        self._log(
            "await_ready", namespace=namespace, operation_id=operation_id, timeout=timeout
        )
        pending = self._pending.get(operation_id)
        if pending is None:
            if self.spec.idempotent and operation_id in self._idem_registry:
                # Querying a finished operation returns its terminal state
                # without re-submitting.
                return self._idem_registry[operation_id][1]
            raise MemoryAdapterError(
                "unknown_operation",
                f"operation {operation_id!r} is not pending in {namespace!r}; "
                "await_ready must not submit new mutations",
                effect="none",
            )
        pending.remaining -= 1
        if pending.remaining > 0:
            return pending.receipt
        receipt = self._receipt(
            pending.operation_id,
            "completed",
            list(pending.receipt.memory_ids),
            list(pending.receipt.sources),
            usage=None,  # waiting never re-counts build usage
        )
        del self._pending[operation_id]
        if self.spec.idempotent:
            self._idem_registry[operation_id] = (pending.input_digest, receipt)
        return receipt

    def close(self, namespace: str) -> None:
        self._log("close", namespace=namespace)
        pending_here = [
            p.operation_id for p in self._pending.values() if p.namespace == namespace
        ]
        if pending_here:
            raise MemoryAdapterError(
                "pending_operations",
                f"cannot close {namespace!r} with pending operations "
                f"{pending_here}; close never replaces completion confirmation",
                effect="possible",
            )
        self._open.discard(namespace)

    def retrieve(
        self, namespace: str, request: RetrievalRequest
    ) -> list[Evidence]:
        self._require_open(namespace)
        self._log("retrieve", namespace=namespace, request=request.model_dump(mode="json"))
        self.received_requests.append(request)
        space = self._space(namespace)
        messages = sorted(space.values(), key=lambda m: (m.occurred_at, m.order))

        if self.spec.retrieval_mode == "match":
            scored = [
                (m, _overlap_score(request.query, m.content)) for m in messages
            ]
            matched = [(m, s) for m, s in scored if s > 0]
            matched.sort(key=lambda pair: (-pair[1], pair[0].occurred_at, pair[0].order))
            selected = matched
            scores = {id(m): s for m, s in matched}
        else:
            selected = [(m, 0.0) for m in messages]
            scores = {}

        evidence: list[Evidence] = []
        if "extractive" in self.spec.evidence_kinds:
            for m, score in selected:
                evidence.append(
                    Evidence(
                        kind="extractive",
                        text=m.content,
                        extractive_span=SourceSpan(
                            session_id=m.session_id,
                            msg_id=m.msg_id,
                            start=0,
                            end=len(m.content),
                        ),
                        derivation_sources=[],
                        source_times=[m.occurred_at],
                        retrieval_score=(
                            float(score) if self.spec.retrieval_mode == "match" else None
                        ),
                    )
                )
        if "generated" in self.spec.evidence_kinds:
            top = [m for m, _ in selected[:3]]
            if top:
                evidence.append(
                    Evidence(
                        kind="generated",
                        text="摘要：" + "；".join(m.content[:16] for m in top),
                        extractive_span=None,
                        derivation_sources=[
                            SourceRef(session_id=m.session_id, msg_id=m.msg_id)
                            for m in top
                        ],
                        source_times=_dedup([m.occurred_at for m in top]),
                        retrieval_score=None,
                    )
                )

        usage = ResourceUsage(
            input_tokens=len(request.query),
            output_tokens=sum(len(e.text) for e in evidence),
            llm_call_count=1 if any(e.kind == "generated" for e in evidence) else 0,
        )
        report_usage(usage)
        return evidence

    def inspect(self, namespace: str, memory_ids: list[str]) -> list[MemoryState]:
        if not self.spec.state_inspection:
            raise MemoryAdapterError(
                "capability_not_declared",
                "this fake does not declare state_inspection",
                effect="none",
            )
        self._require_open(namespace)
        self._log("inspect", namespace=namespace, memory_ids=list(memory_ids))
        space = self._space(namespace)
        states: list[MemoryState] = []
        for mid in memory_ids:
            stored = space.get(mid)
            if stored is None:
                states.append(
                    MemoryState(
                        memory_id=mid,
                        content=None,
                        sources=[],
                        validity="unknown",
                        superseded_by=None,
                    )
                )
            else:
                states.append(
                    MemoryState(
                        memory_id=mid,
                        content=stored.content,
                        sources=[
                            SourceRef(
                                session_id=stored.session_id, msg_id=stored.msg_id
                            )
                        ],
                        validity="current",
                        superseded_by=[],
                    )
                )
        return states

    # -- test/diagnostic helpers -------------------------------------------

    def stored_message_count(self, namespace: str) -> int:
        return len(self._spaces.get(namespace, {}))

    def pending_operation_ids(self) -> list[str]:
        return sorted(self._pending)


def _dedup(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def build_fake_adapter(plan: Any) -> FakeMemoryAdapter:
    """Build the fake adapter declared by an ExperimentConfig memory plan."""
    spec = FakeMemorySpec.from_memory_plan(plan)
    return FakeMemoryAdapter(spec)
