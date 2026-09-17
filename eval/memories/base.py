"""Memory adapter protocol and errors.

The protocol mirrors docs/design/eval-harness.md, section Memory Adapter:
capabilities() declares the fixed capability set before a run; reset/
open/close manage the per-sample isolated space; ingest writes exactly
one session; retrieve is read-only; await_ready is provided by async
adapters (sync adapters return completed receipts directly); update and
delete are the explicit operations exercised by the operations suite;
inspect is the only state-observation channel.

update/delete are the explicit operations of the M1 operations suite;
inspect is the only state-observation channel and must answer by stable
memory id (a missing row may never replace a definite state).

Adapters raise MemoryAdapterError for protocol violations and backend
failures; the runner converts them into structured ErrorInfo attempt
records. A successful retrieve with no match is an empty list, never an
error.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from eval.contracts.adapter import (
    Evidence,
    MemoryState,
    MutationReceipt,
    RetrievalRequest,
    Session,
)
from eval.contracts.adapter import Effect

#: First-version capability vocabulary (data contract document). Unknown
#: strings may appear as extension diagnostics but can never make a test
#: pass; the runner requires adapter capabilities to equal the config
#: declaration before a run starts.
CAPABILITY_VOCABULARY: tuple[str, ...] = (
    "extractive_evidence",
    "generated_evidence",
    "state_inspection",
    "auto_update",
    "update",
    "delete",
    "async_mutation",
    "idempotent_mutation",
    "operation_status",
)

#: Capabilities every connected adapter must provide (base contract);
#: they are implicit and never part of the declared optional set.
BASE_CAPABILITIES: tuple[str, ...] = (
    "ingest",
    "read_only_retrieve",
    "isolated_namespaces",
)


class MemoryAdapterError(RuntimeError):
    """Structured adapter failure.

    effect follows the ErrorInfo contract: 'none' means the call
    definitely did not take effect and will not later; 'possible' means
    the outcome is uncertain; 'confirmed' is reserved for effects the
    adapter could observe (M1 fakes never emit it).
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        transient: bool = False,
        effect: Effect = "possible",
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.transient = transient
        self.effect: Effect = effect

    def to_error_info(self):
        from eval.contracts.adapter import ErrorInfo

        return ErrorInfo(
            code=self.code,
            message=self.message,
            effect=self.effect,
            transient=self.transient,
        )


@runtime_checkable
class MemoryAdapter(Protocol):
    """Semantic interface every memory adapter implements."""

    def capabilities(self) -> set[str]: ...

    def reset(self, namespace: str) -> None:
        """Empty the space; previously accepted tasks must not re-write."""
        ...

    def open(self, namespace: str) -> None: ...

    def ingest(
        self, namespace: str, session: Session, operation_id: str
    ) -> MutationReceipt: ...

    def retrieve(
        self, namespace: str, request: RetrievalRequest
    ) -> list[Evidence]: ...

    def close(self, namespace: str) -> None: ...

    def await_ready(
        self, namespace: str, operation_id: str, timeout: float
    ) -> MutationReceipt:
        """Poll/wait the SAME operation; must never re-submit it."""
        ...

    def inspect(
        self, namespace: str, memory_ids: list[str]
    ) -> list[MemoryState]: ...
