"""Memory adapters: protocol, usage reporting and offline fakes."""

from eval.memories.base import (
    BASE_CAPABILITIES,
    CAPABILITY_VOCABULARY,
    MemoryAdapter,
    MemoryAdapterError,
)
from eval.memories.baselines import (
    BM25_B,
    BM25_DEFAULT_K,
    BM25_EPSILON,
    BM25_K1,
    UPSTREAM_RETRIEVAL_COMMIT,
    UPSTREAM_RETRIEVAL_URL,
    FullHistoryAdapter,
    NoMemoryAdapter,
    build_baseline_adapter,
    build_memory_for_plan,
    full_history_evidence,
)
from eval.memories.fake import (
    FAKE_MEMORY_NAME,
    FakeMemoryAdapter,
    FakeMemorySpec,
    build_fake_adapter,
)
from eval.memories.usage import (
    UsageRecorder,
    merge_resource_usage,
    recorded_usage,
    report_usage,
)

__all__ = [
    "BASE_CAPABILITIES",
    "CAPABILITY_VOCABULARY",
    "MemoryAdapter",
    "MemoryAdapterError",
    "BM25_B",
    "BM25_DEFAULT_K",
    "BM25_EPSILON",
    "BM25_K1",
    "UPSTREAM_RETRIEVAL_COMMIT",
    "UPSTREAM_RETRIEVAL_URL",
    "FullHistoryAdapter",
    "NoMemoryAdapter",
    "FAKE_MEMORY_NAME",
    "FakeMemoryAdapter",
    "FakeMemorySpec",
    "build_baseline_adapter",
    "build_fake_adapter",
    "build_memory_for_plan",
    "full_history_evidence",
    "UsageRecorder",
    "merge_resource_usage",
    "recorded_usage",
    "report_usage",
]
