"""Memory adapters: protocol, usage reporting and offline fakes."""

from eval.memories.base import (
    BASE_CAPABILITIES,
    CAPABILITY_VOCABULARY,
    MemoryAdapter,
    MemoryAdapterError,
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
    "FAKE_MEMORY_NAME",
    "FakeMemoryAdapter",
    "FakeMemorySpec",
    "build_fake_adapter",
    "UsageRecorder",
    "merge_resource_usage",
    "recorded_usage",
    "report_usage",
]
