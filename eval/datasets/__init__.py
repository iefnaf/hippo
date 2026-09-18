"""Dataset adapters: the M1 manual fixtures and the M2 pinned
LongMemEval-S cleaned file, plus the dataset registry keyed by
`dataset_plan` (used by the run/resume CLI)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from eval.contracts.common import ContractError
from eval.datasets.longmemeval import (
    ABSTENTION_SUFFIX as LONGMEMEVAL_ABSTENTION_SUFFIX,
)
from eval.datasets.longmemeval import (
    DATASET_PLAN as LONGMEMEVAL_S_DATASET_PLAN,
)
from eval.datasets.longmemeval import (
    DEFAULT_LONGMEMEVAL_S_PATH,
    LongMemEvalDataset,
)
from eval.datasets.manual import (
    ABSTENTION_SUFFIX,
    DEFAULT_DATASET_PATH,
    ManualDataset,
    internal_msg_id,
    internal_session_id,
    namespace_for,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from eval.config import ExperimentConfig

__all__ = [
    "ABSTENTION_SUFFIX",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_LONGMEMEVAL_S_PATH",
    "LONGMEMEVAL_S_DATASET_PLAN",
    "LongMemEvalDataset",
    "ManualDataset",
    "internal_msg_id",
    "internal_session_id",
    "load_dataset_for_config",
    "namespace_for",
]


def load_dataset_for_config(
    config: "ExperimentConfig", *, path: str | Path | None = None
) -> "ManualDataset | LongMemEvalDataset":
    """Load the dataset adapter a config's `dataset_plan` declares.

    `path` overrides the plan's default file (the CLI --dataset flag);
    unknown plans fail structurally instead of silently falling back to
    the manual fixtures.
    """
    if config.dataset_plan == LONGMEMEVAL_S_DATASET_PLAN:
        return LongMemEvalDataset.from_file(path or DEFAULT_LONGMEMEVAL_S_PATH)
    if config.dataset_plan.startswith("manual-"):
        return ManualDataset.from_file(path or DEFAULT_DATASET_PATH)
    raise ContractError(
        code="unknown_dataset_plan",
        message=(
            f"dataset_plan {config.dataset_plan!r} matches no known dataset "
            f"adapter; known: {LONGMEMEVAL_S_DATASET_PLAN} (pinned "
            "LongMemEval-S, fetched via scripts/fetch_longmemeval.py) or a "
            "manual-* fixture plan"
        ),
        location="/dataset_plan",
    )
