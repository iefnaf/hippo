"""Data contract single source of truth (pydantic v2).

The dataclasses in docs/design/eval-harness-data-contracts.md are semantic
sketches; these models are the authoritative, runtime-validated types.
Adapter-visible models mirror the contract document exactly; harness-internal
models cover reader preparation, scoring and per-run records.
"""

from eval.contracts.common import (
    SCHEMA_VERSION,
    ContractError,
    ContractModel,
    SchemaVersionedModel,
    parse_dataset_time,
    parse_run_timestamp,
)
from eval.contracts.adapter import (
    Validity,
    ErrorInfo,
    Evidence,
    MemoryState,
    Message,
    MutationReceipt,
    QueryContext,
    ResourceUsage,
    RetrievalRequest,
    Session,
    SourceRef,
    SourceSpan,
)
from eval.contracts.internal import (
    PreparedEvidence,
    PreparedItem,
    ReaderResult,
    ScoringData,
    JudgeRequest,
    JudgeResult,
    StageAttempt,
    Attribution,
    Result,
    MetricResult,
    RawEvidenceArtifact,
    PreparedEvidenceArtifact,
    ReaderResultArtifact,
    JudgeRecordArtifact,
    ResultArtifact,
)

__all__ = [
    "SCHEMA_VERSION",
    "ContractError",
    "ContractModel",
    "SchemaVersionedModel",
    "parse_dataset_time",
    "parse_run_timestamp",
    # adapter-visible
    "Validity",
    "SourceRef",
    "SourceSpan",
    "Message",
    "Session",
    "QueryContext",
    "RetrievalRequest",
    "Evidence",
    "MemoryState",
    "ResourceUsage",
    "ErrorInfo",
    "MutationReceipt",
    # harness-internal
    "PreparedItem",
    "PreparedEvidence",
    "ReaderResult",
    "ScoringData",
    "JudgeRequest",
    "JudgeResult",
    "StageAttempt",
    "Attribution",
    "Result",
    "MetricResult",
    # artifact envelopes
    "RawEvidenceArtifact",
    "PreparedEvidenceArtifact",
    "ReaderResultArtifact",
    "JudgeRecordArtifact",
    "ResultArtifact",
]
