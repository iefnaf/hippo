"""Judge calibration (issue #9): sampling, blind annotation, statistics,
threshold decision and the record that binds judge identity + date.

The judge follows the official LongMemEval anscheck protocol but is a
different model family than the official GPT-4o, so before its verdicts
enter formal QA conclusions it must be calibrated against a single blind
human annotator (docs/design/eval-harness.md, 「Judge 校准」):

- 100 random stratified samples (dev-side outputs of two conditions)
  estimate the overall agreement with a Wilson 95% interval;
- 20 deliberately-picked boundary samples diagnose lenient/strict bias
  and NEVER merge into the agreement statistics;
- 20 of the random items are re-annotated after >= 1 day to estimate
  the annotator's own consistency (a single annotator means
  self-consistency only — no inter-rater agreement exists, and the
  calibration report says so explicitly);
- the fixed thresholds (criteria.py) turn the statistics into passed /
  failed / inconclusive; failed or inconclusive configurations degrade
  their QA conclusions to diagnostics (report/compare channel) and
  never claim comparability with the paper's GPT-4o-validated scores.

Judge input discipline: a calibration judge call receives ONLY the
question, the gold answer and the response (plus the protocol-private
abstention flag) — exactly the scorer's request shape. Retrieval
results, memory state and evidence never reach the judge; the offline
tests assert this on every code path added here.
"""

from eval.calibration.criteria import (
    BOUNDARY_SAMPLE_SIZE,
    CALIBRATION_ALGORITHMS,
    OFFICIAL_JUDGE_MODEL,
    RANDOM_SAMPLE_SIZE,
    RUBRIC_ID,
    RUBRIC_MD,
    RUBRIC_SHA256,
    SELF_CONSISTENCY_SIZE,
    THRESHOLDS,
    THRESHOLDS_ID,
    WILSON_Z,
    criteria_digest,
    criteria_payload,
)
from eval.calibration.integration import (
    QA_CONCLUSION_METRIC_IDS,
    is_downgraded,
    judge_calibration_doc,
)
from eval.calibration.judge_batch import (
    CalibrationJudgeCall,
    CalibrationJudgeCallsArtifact,
    assert_judge_input_discipline,
    run_judge_batch,
)
from eval.calibration.record import (
    CalibrationRecordArtifact,
    bind_record_to_judge_plan,
    build_calibration_record,
    verify_decision_stability,
)
from eval.calibration.sampling import (
    AnnotationRecord,
    CalibrationCandidate,
    CalibrationItem,
    CalibrationPlanArtifact,
    WorksheetRow,
    build_calibration_plan,
    collect_run_outputs,
    import_annotations,
    render_worksheet_csv,
    worksheet_rows,
)
from eval.calibration.stats import (
    CalibrationDecision,
    CalibrationStatistics,
    decide_calibration,
    compute_statistics,
    wilson_interval,
)

__all__ = [
    "BOUNDARY_SAMPLE_SIZE",
    "CALIBRATION_ALGORITHMS",
    "CalibrationCandidate",
    "CalibrationDecision",
    "CalibrationItem",
    "CalibrationJudgeCall",
    "CalibrationJudgeCallsArtifact",
    "CalibrationPlanArtifact",
    "CalibrationRecordArtifact",
    "CalibrationStatistics",
    "OFFICIAL_JUDGE_MODEL",
    "QA_CONCLUSION_METRIC_IDS",
    "RANDOM_SAMPLE_SIZE",
    "RUBRIC_ID",
    "RUBRIC_MD",
    "RUBRIC_SHA256",
    "SELF_CONSISTENCY_SIZE",
    "THRESHOLDS",
    "THRESHOLDS_ID",
    "WILSON_Z",
    "WorksheetRow",
    "assert_judge_input_discipline",
    "compute_statistics",
    "bind_record_to_judge_plan",
    "build_calibration_plan",
    "build_calibration_record",
    "collect_run_outputs",
    "criteria_digest",
    "criteria_payload",
    "decide_calibration",
    "import_annotations",
    "is_downgraded",
    "judge_calibration_doc",
    "render_worksheet_csv",
    "run_judge_batch",
    "verify_decision_stability",
    "worksheet_rows",
    "wilson_interval",
]
