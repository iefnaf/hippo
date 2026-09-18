"""The three control baselines (M2, issue #8): no-memory, BM25, full
history.

- **无记忆（none）**: the reader sees the shared query context only;
  retrieve() always returns an empty list. Ranking recall is N/A
  (registry condition ``ranking_baseline``); attribution is never-hit.

- **BM25**: follows the upstream `flat-bm25` definition VERBATIM
  (LongMemEval src/retrieval/run_retrieval.py at the pinned commit
  below, session granularity):
    - document unit = one session; the document text is the session's
      USER-turn contents joined by a single space (upstream filters
      ``interact['role'] == 'user'`` — assistant/tool turns are not in
      the index);
    - tokenization = ``doc.split(" ")`` on documents and query (plain
      whitespace split; no lowercasing, no stemming);
    - scoring = rank_bm25.BM25Okapi with its DEFAULT parameters
      (k1=1.5, b=0.75, epsilon=0.25) — deliberately NOT tuned;
    - ranking = top-k sessions with ties broken by (score desc,
      session time asc, session_id lexicographic) for determinism
      (upstream's plain argsort tie order is unspecified);
    - hit sessions expand, in session order, into MESSAGE-level
      extractive evidence units (one full-message span per unit); the
      harness budget executes AFTER expansion.

- **完整历史（full history）**: retrieve() returns every message of
  every session in time order as message-level extractive evidence,
  ignoring the retrieval budget hint. The harness precheck decides
  runnability against the reader context window; an over-limit
  question is ``context_exceeded`` — never silently truncated. Its
  results are reported separately and never enter ranking recall
  (issue #3 leftover, consumed by the scorer via the registry's
  ``ranking_baseline`` N/A condition).

All three keep per-namespace in-process state across open/close (the
protocol's reopen-observes-persisted-data holds inside one run;
baselines are controls, not the system under test).
"""

from __future__ import annotations

from typing import Any

from eval.contracts.adapter import (
    Evidence,
    Message,
    MutationReceipt,
    RetrievalRequest,
    Session,
    SourceRef,
    SourceSpan,
)
from eval.contracts.common import ContractError

#: Upstream retrieval implementation this baseline is bound to.
UPSTREAM_RETRIEVAL_COMMIT = "9e0b455f4ef0e2ab8f2e582289761153549043fc"
UPSTREAM_RETRIEVAL_URL = (
    "https://github.com/xiaowu0162/LongMemEval/blob/"
    f"{UPSTREAM_RETRIEVAL_COMMIT}/src/retrieval/run_retrieval.py"
)

#: Fixed BM25 parameters (rank_bm25 BM25Okapi defaults; NOT tuned).
BM25_K1 = 1.5
BM25_B = 0.75
BM25_EPSILON = 0.25
#: Sessions returned per query (design: k=10, an M2 recheck parameter).
BM25_DEFAULT_K = 10


def session_document(session: Session) -> str:
    """Upstream session-granularity document: user turns joined by ' '."""
    return " ".join(
        msg.content for msg in session.messages if msg.role == "user"
    )


def upstream_tokenize(text: str) -> list[str]:
    """Upstream tokenization: plain single-space split."""
    return text.split(" ")


def message_evidence(
    session: Session, message: Message, *, retrieval_score: float | None
) -> Evidence:
    """One message as a full-span extractive evidence unit."""
    return Evidence(
        kind="extractive",
        text=message.content,
        extractive_span=SourceSpan(
            session_id=session.session_id,
            msg_id=message.msg_id,
            start=0,
            end=len(message.content),
        ),
        derivation_sources=[],
        source_times=[session.occurred_at],
        retrieval_score=retrieval_score,
    )


def full_history_evidence(sessions: list[Session]) -> list[Evidence]:
    """Every message of every session, in time order, as evidence.

    Shared by the full-history adapter and the harness precheck so both
    construct the identical unit list.
    """
    return [
        message_evidence(session, message, retrieval_score=None)
        for session in sessions
        for message in session.messages
    ]


class _SessionBaselineAdapter:
    """Shared namespace-scoped machinery for the session baselines."""

    def __init__(self, *, capabilities: frozenset[str]) -> None:
        self._capabilities = set(capabilities)
        self._sessions: dict[str, list[Session]] = {}

    def capabilities(self) -> set[str]:
        return set(self._capabilities)

    def reset(self, namespace: str) -> None:
        self._sessions.pop(namespace, None)

    def open(self, namespace: str) -> None:
        self._sessions.setdefault(namespace, [])

    def close(self, namespace: str) -> None:
        # In-process persistence: state survives close/reopen within a run.
        return None

    def ingest(
        self, namespace: str, session: Session, operation_id: str
    ) -> MutationReceipt:
        # Sync baseline: the write is complete when ingest returns.
        self._sessions.setdefault(namespace, []).append(session)
        return MutationReceipt(
            operation_id=operation_id,
            status="completed",
            memory_ids=[],
            sources=[
                SourceRef(session_id=session.session_id, msg_id=m.msg_id)
                for m in session.messages
            ],
            error=None,
            usage=None,
        )

    def await_ready(
        self, namespace: str, operation_id: str, timeout: float
    ) -> MutationReceipt:
        # Sync adapters never return 'accepted'; the runner never calls
        # this, but the protocol shape must exist.
        return MutationReceipt(
            operation_id=operation_id,
            status="completed",
            memory_ids=[],
            sources=[],
            error=None,
            usage=None,
        )

    def inspect(self, namespace: str, memory_ids: list[str]) -> list[Any]:
        raise NotImplementedError("baselines expose no operable memories")

    def _require_namespace(self, namespace: str) -> list[Session]:
        sessions = self._sessions.get(namespace)
        if sessions is None:
            raise ContractError(
                code="namespace_not_open",
                message=(
                    f"baseline namespace {namespace!r} was not opened; "
                    "the runner opens every space before writing"
                ),
                location="(namespace)",
            )
        return sessions


class NoMemoryAdapter(_SessionBaselineAdapter):
    """无记忆基线: no history evidence ever reaches the reader."""

    def __init__(self) -> None:
        super().__init__(capabilities=frozenset())
        self.ingested_sessions: dict[str, int] = {}

    def ingest(
        self, namespace: str, session: Session, operation_id: str
    ) -> MutationReceipt:
        # The protocol still walks every session through the memory
        # interface; this control just keeps nothing retrievable.
        self.ingested_sessions[namespace] = (
            self.ingested_sessions.get(namespace, 0) + 1
        )
        return MutationReceipt(
            operation_id=operation_id,
            status="completed",
            memory_ids=[],
            sources=[],
            error=None,
            usage=None,
        )

    def retrieve(
        self, namespace: str, request: RetrievalRequest
    ) -> list[Evidence]:
        self._require_namespace(namespace)
        return []


class BM25Adapter(_SessionBaselineAdapter):
    """BM25 基线: upstream flat-bm25 over session documents."""

    def __init__(self, *, k: int = BM25_DEFAULT_K) -> None:
        super().__init__(capabilities=frozenset({"extractive_evidence"}))
        if k < 1:
            raise ContractError(
                code="baseline_config_invalid",
                message=f"bm25 k must be >= 1, got {k}",
                location="/memory/config/k",
            )
        self.k = k

    def retrieve(
        self, namespace: str, request: RetrievalRequest
    ) -> list[Evidence]:
        from rank_bm25 import BM25Okapi

        sessions = self._require_namespace(namespace)
        # Upstream builds BM25 over the whole corpus per query; the
        # accumulated per-session documents are exactly that corpus.
        corpus = [upstream_tokenize(session_document(s)) for s in sessions]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(upstream_tokenize(request.query))
        order = sorted(
            range(len(sessions)),
            key=lambda i: (
                -float(scores[i]),
                sessions[i].occurred_at,
                sessions[i].session_id,
            ),
        )
        evidence: list[Evidence] = []
        for i in order[: self.k]:
            session = sessions[i]
            for message in session.messages:  # expand in session order
                evidence.append(
                    message_evidence(
                        session, message, retrieval_score=float(scores[i])
                    )
                )
        return evidence


class FullHistoryAdapter(_SessionBaselineAdapter):
    """完整历史基线: every message as evidence, budget-unbounded.

    The retrieval request's evidence_token_budget is a hint this control
    deliberately ignores; the harness precheck (against the reader
    context window) decides runnability, and an over-limit question is
    context_exceeded — never silently truncated.
    """

    def __init__(self) -> None:
        super().__init__(capabilities=frozenset({"extractive_evidence"}))

    def retrieve(
        self, namespace: str, request: RetrievalRequest
    ) -> list[Evidence]:
        sessions = self._require_namespace(namespace)
        return full_history_evidence(sessions)


def build_baseline_adapter(plan: Any):
    """Build the baseline adapter a memory plan's baseline_kind declares.

    BM25's k comes from the memory plan config (default 10, an M2
    recheck parameter); k1/b are upstream defaults and are deliberately
    NOT configurable (tuning them would be a different baseline).
    """
    kind = plan.baseline_kind
    config = dict(getattr(plan, "config", {}) or {})
    if kind == "none":
        return NoMemoryAdapter()
    if kind == "bm25":
        return BM25Adapter(k=int(config.get("k", BM25_DEFAULT_K)))
    if kind == "full_history":
        return FullHistoryAdapter()
    raise ContractError(
        code="unknown_baseline_kind",
        message=(
            f"baseline_kind {kind!r} is not a harness baseline; use "
            "build_fake_adapter for the 'adapter' fake"
        ),
        location="/memory/baseline_kind",
    )


def build_memory_for_plan(plan: Any):
    """Build the memory component a memory plan declares (all kinds)."""
    if plan.baseline_kind == "adapter":
        from eval.memories.fake import build_fake_adapter

        return build_fake_adapter(plan)
    return build_baseline_adapter(plan)
