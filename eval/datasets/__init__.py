"""Dataset adapters; the M1 offline dataset is the manual fixture set."""

from eval.datasets.manual import (
    ABSTENTION_SUFFIX,
    DEFAULT_DATASET_PATH,
    ManualDataset,
    internal_msg_id,
    internal_session_id,
    namespace_for,
)

__all__ = [
    "ABSTENTION_SUFFIX",
    "DEFAULT_DATASET_PATH",
    "ManualDataset",
    "internal_msg_id",
    "internal_session_id",
    "namespace_for",
]
