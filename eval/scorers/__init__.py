"""scorers package: program metrics plus judge-driven verdicts."""

from eval.scorers.qa import (
    ABSTENTION_NA_REASON,
    EVIDENCE_MODE_NA_REASON,
    RECALL_METRIC_IDS,
    JudgeCallRecord,
    QAScorer,
    SampleScoring,
    ScoringDataError,
    attribution_cell,
    retained_extractive_sessions,
    session_recall,
    session_recall_at,
)

__all__ = [
    "ABSTENTION_NA_REASON",
    "EVIDENCE_MODE_NA_REASON",
    "RECALL_METRIC_IDS",
    "JudgeCallRecord",
    "QAScorer",
    "SampleScoring",
    "ScoringDataError",
    "attribution_cell",
    "retained_extractive_sessions",
    "session_recall",
    "session_recall_at",
]
