"""Metric registry v1.

Every metric entering reports and comparisons is defined here with a
stable metric_id, version, owning suite, evidence mode, count unit,
denominator, N/A condition and whether it is a ranking metric. The
registry version is part of the config fingerprint; comparison commands
require identical registry versions and refuse to auto-align metrics
whose name matches but version differs. Unregistered metrics are
diagnostics only.

Metric set (from docs/design/eval-harness.md, section 评分与指标):
- verifiable session recall macro/micro averages
- Recall@k for k in {1, 3, 5}
- session recall over all actual extractive evidence within the 4K budget
- derivation_source_coverage (diagnostic coverage of self-reported
  generation sources; never called recall)
- planned-question overall score
- accuracy over scored questions
- runnable coverage
- abstention accuracy
- per-question attribution distribution (2x2)
- operations pass rate and support coverage
- resource and budget composition items
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from eval.contracts.common import ContractModel

#: Registry schema/artifact version. Any change to metric definitions, k
#: sets or N/A semantics bumps this (it is a content-derived version: the
#: registry_version constant and the content fingerprint below both move).
REGISTRY_VERSION = "1"

#: Version string used by the config fingerprint; binds registry content.
REGISTRY_CONTENT_VERSION = "metrics-registry@1"

EvidenceMode = Literal[
    "extractive",   # requires verifiable extractive evidence
    "any",          # independent of evidence mode
    "generated",    # only meaningful for generated content diagnostics
]

MetricUnit = Literal[
    "ratio",        # numerator/denominator in [0, 1]
    "count",        # absolute count
    "distribution", # categorical distribution (shares summing to 1)
    "tokens",       # token quantity
    "ms",           # duration in milliseconds
]

Suite = Literal["qa", "operations", "all"]

NA_CONDITIONS = {
    "gold_empty": "gold source set is empty although the sample is not abstention (data error)",
    "not_applicable": "metric semantics do not apply (e.g. recall of abstention questions)",
    "evidence_mode": "declared evidence mode does not support this metric (e.g. extractive recall of a purely generated implementation)",
    "denominator_zero": "denominator is zero (e.g. no scored questions)",
    "capability_missing": "required optional capability not declared before the run",
    "ranking_baseline": "unranked baseline (no-memory / full-history) never enters ranking retrieval metrics",
    "usage_unknown": "usage was not reported by the component",
}


class MetricDefinition(ContractModel):
    """Definition of one formal metric. Immutable; versioned."""

    metric_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    suite: Suite
    evidence_mode: EvidenceMode
    unit: MetricUnit
    denominator: str = Field(min_length=1)
    na_conditions: tuple[str, ...] = Field(default=())
    ranking: bool
    description: str = Field(min_length=1)

    @model_validator(mode="after")
    def _na_known(self) -> Self:
        unknown = [c for c in self.na_conditions if c not in NA_CONDITIONS]
        if unknown:
            raise ValueError(
                f"metric {self.metric_id}: unknown N/A conditions {unknown}; "
                f"known: {sorted(NA_CONDITIONS)}"
            )
        return self


def _m(
    metric_id: str,
    suite: Suite,
    evidence_mode: EvidenceMode,
    unit: MetricUnit,
    denominator: str,
    na: tuple[str, ...],
    ranking: bool,
    description: str,
) -> MetricDefinition:
    return MetricDefinition(
        metric_id=metric_id,
        version=1,
        suite=suite,
        evidence_mode=evidence_mode,
        unit=unit,
        denominator=denominator,
        na_conditions=na,
        ranking=ranking,
        description=description,
    )


#: v1 registry entries. Ordering is stable and part of the content
#: fingerprint; do not reorder without bumping REGISTRY_VERSION.
_METRICS: tuple[MetricDefinition, ...] = (
    _m(
        "verifiable_session_recall_macro",
        "qa",
        "extractive",
        "ratio",
        "applicable sample count |E| (non-abstention, valid gold, declared extractive)",
        ("not_applicable", "evidence_mode", "denominator_zero", "gold_empty", "ranking_baseline"),
        True,
        "Macro (per-question arithmetic mean) of verifiable session recall over actual extractive evidence kept within budget.",
    ),
    _m(
        "verifiable_session_recall_micro",
        "qa",
        "extractive",
        "ratio",
        "total gold sessions across applicable samples",
        ("not_applicable", "evidence_mode", "denominator_zero", "gold_empty", "ranking_baseline"),
        False,
        "Micro recall: sum(|G intersect R|) / sum(|G|) over applicable samples; reported alongside, never replacing the macro metric.",
    ),
    _m(
        "recall_at_1",
        "qa",
        "extractive",
        "ratio",
        "applicable sample count |E|",
        ("not_applicable", "evidence_mode", "denominator_zero", "gold_empty", "ranking_baseline"),
        True,
        "Recall@1: gold coverage by the first distinct session among actual extractive evidence in return order.",
    ),
    _m(
        "recall_at_3",
        "qa",
        "extractive",
        "ratio",
        "applicable sample count |E|",
        ("not_applicable", "evidence_mode", "denominator_zero", "gold_empty", "ranking_baseline"),
        True,
        "Recall@3 over the first three distinct sessions of actual extractive evidence in return order.",
    ),
    _m(
        "recall_at_5",
        "qa",
        "extractive",
        "ratio",
        "applicable sample count |E|",
        ("not_applicable", "evidence_mode", "denominator_zero", "gold_empty", "ranking_baseline"),
        True,
        "Recall@5 over the first five distinct sessions of actual extractive evidence in return order.",
    ),
    _m(
        "budgeted_session_recall",
        "qa",
        "extractive",
        "ratio",
        "applicable sample count |E|",
        ("not_applicable", "evidence_mode", "denominator_zero", "gold_empty", "ranking_baseline"),
        True,
        "Session recall computed over ALL actual extractive evidence retained within the 4K token budget (no k-cut).",
    ),
    _m(
        "derivation_source_coverage",
        "qa",
        "generated",
        "ratio",
        "generated evidence units with declared derivation sources",
        ("not_applicable", "denominator_zero", "usage_unknown"),
        False,
        "DIAGNOSTIC coverage of self-reported derivation sources of generated content; explicitly not evidence recall and never ranked.",
    ),
    _m(
        "planned_question_score",
        "qa",
        "any",
        "ratio",
        "planned question count |P|",
        ("denominator_zero",),
        False,
        "Overall score over planned questions: correct / |P|; failed and context_exceeded questions contribute zero.",
    ),
    _m(
        "scored_accuracy",
        "qa",
        "any",
        "ratio",
        "scored question count",
        ("denominator_zero",),
        False,
        "Accuracy over scored questions only; reported together with scored/failed/context_exceeded counts and scoring coverage.",
    ),
    _m(
        "scoring_coverage",
        "qa",
        "any",
        "ratio",
        "planned question count |P|",
        ("denominator_zero",),
        False,
        "scored / |P|: share of planned questions that reached a judge verdict.",
    ),
    _m(
        "runnable_coverage",
        "qa",
        "any",
        "ratio",
        "planned question count |P|",
        ("denominator_zero",),
        False,
        "Precheck-runnable questions / |P|; runtime API/judge failures do not change the runnable set.",
    ),
    _m(
        "abstention_accuracy",
        "qa",
        "any",
        "ratio",
        "planned abstention questions",
        ("denominator_zero",),
        False,
        "Judge accuracy over abstention-marked questions (they still receive answers and judge verdicts).",
    ),
    _m(
        "attribution_hit_correct",
        "qa",
        "any",
        "distribution",
        "scored questions",
        ("denominator_zero",),
        False,
        "Share of scored questions classified hit_correct (gold-evidence hit AND correct answer) in the 2x2 attribution distribution.",
    ),
    _m(
        "attribution_hit_wrong",
        "qa",
        "any",
        "distribution",
        "scored questions",
        ("denominator_zero",),
        False,
        "Share of scored questions classified hit_wrong (evidence found but answer wrong).",
    ),
    _m(
        "attribution_miss_correct",
        "qa",
        "any",
        "distribution",
        "scored questions",
        ("denominator_zero",),
        False,
        "Share of scored questions classified miss_correct (no evidence hit, answer still correct).",
    ),
    _m(
        "attribution_miss_wrong",
        "qa",
        "any",
        "distribution",
        "scored questions",
        ("denominator_zero",),
        False,
        "Share of scored questions classified miss_wrong (no evidence hit, answer wrong).",
    ),
    _m(
        "operations_pass_rate",
        "operations",
        "any",
        "ratio",
        "passed + failed operation checks",
        ("denominator_zero", "capability_missing"),
        True,
        "Operations pass rate: passed / (passed + failed); unsupported items are never hidden and stay out of the denominator.",
    ),
    _m(
        "operations_support_coverage",
        "operations",
        "any",
        "ratio",
        "planned operation checks",
        ("denominator_zero",),
        False,
        "(passed + failed) / planned operation checks: how much of the plan the implementation actually supports.",
    ),
    _m(
        "retrieval_returned_units",
        "qa",
        "any",
        "count",
        "retrieval calls",
        ("capability_missing",),
        False,
        "Number of evidence units returned by the adapter per retrieval (explanatory, never ranked).",
    ),
    _m(
        "retrieval_reader_units",
        "qa",
        "any",
        "count",
        "retrieval calls",
        ("capability_missing",),
        False,
        "Number of evidence units that actually entered the reader within budget.",
    ),
    _m(
        "retrieval_dropped_units",
        "qa",
        "any",
        "count",
        "retrieval calls",
        ("capability_missing",),
        False,
        "Number of units dropped entirely by budget enforcement (explanatory).",
    ),
    _m(
        "evidence_text_tokens",
        "qa",
        "any",
        "tokens",
        "retrieval calls",
        ("capability_missing",),
        False,
        "Tokens of retained unit texts (PreparedEvidence.text_token_count aggregate).",
    ),
    _m(
        "evidence_format_tokens",
        "qa",
        "any",
        "tokens",
        "retrieval calls",
        ("capability_missing",),
        False,
        "Rendering/metadata overhead tokens: total minus text tokens (diagnostic for unit-granularity differences).",
    ),
    _m(
        "ingest_latency_ms",
        "all",
        "any",
        "ms",
        "ingest attempts",
        ("usage_unknown",),
        False,
        "Per-ingest latency including await_ready waits and retries.",
    ),
    _m(
        "retrieve_latency_ms_p50",
        "qa",
        "any",
        "ms",
        "successfully completed logical retrievals",
        ("denominator_zero", "usage_unknown"),
        False,
        "Median latency of successful logical retrievals including retries and waits; sample count reported alongside.",
    ),
    _m(
        "retrieve_latency_ms_p95",
        "qa",
        "any",
        "ms",
        "successfully completed logical retrievals",
        ("denominator_zero", "usage_unknown"),
        False,
        "p95 latency of successful logical retrievals; failed retrievals reported separately.",
    ),
)

#: metric_id -> definition (insertion order preserved).
REGISTRY: dict[str, MetricDefinition] = {m.metric_id: m for m in _METRICS}

#: Formal ranking metric ids (used by comparison guards).
RANKING_METRIC_IDS = frozenset(
    m.metric_id for m in _METRICS if m.ranking
)

_QA_METRIC_IDS = frozenset(
    m.metric_id for m in _METRICS if m.suite in ("qa", "all")
)
_OPS_METRIC_IDS = frozenset(
    m.metric_id for m in _METRICS if m.suite in ("operations", "all")
)


class UnknownMetricError(ValueError):
    """Raised when an unregistered metric id enters formal results."""

    def __init__(self, metric_id: str) -> None:
        self.metric_id = metric_id
        super().__init__(
            f"metric {metric_id!r} is not registered; unregistered metrics "
            "may only appear as explicitly marked diagnostics and never "
            "enter comparisons or rankings"
        )


def get_metric(metric_id: str) -> MetricDefinition:
    """Return the registered definition or raise UnknownMetricError."""
    try:
        return REGISTRY[metric_id]
    except KeyError as exc:
        raise UnknownMetricError(metric_id) from exc


def validate_metric_id(metric_id: str) -> None:
    """Raise UnknownMetricError if metric_id is not registered."""
    get_metric(metric_id)


def validate_result_metrics(result) -> None:
    """Validate every MetricResult in a Result against the registry.

    - metric_id must be registered;
    - the metric's suite must match the result suite;
    - registry version consistency is enforced at config level via the
      fingerprint (see eval.config).
    """
    from eval.contracts.internal import Result  # local import avoids cycle

    if not isinstance(result, Result):
        raise TypeError(f"expected Result, got {type(result).__name__}")
    allowed = _QA_METRIC_IDS if result.suite == "qa" else _OPS_METRIC_IDS
    for metric in result.metrics:
        definition = get_metric(metric.metric_id)
        if metric.metric_id not in allowed:
            raise ValueError(
                f"metric {metric.metric_id!r} (suite={definition.suite}) is "
                f"not valid for suite={result.suite!r} result of sample "
                f"{result.sample_handle!r}"
            )


def registry_entries() -> tuple[MetricDefinition, ...]:
    """Stable ordered tuple of all v1 definitions."""
    return _METRICS
