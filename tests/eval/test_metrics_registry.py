"""Tests for the metric registry v1.

Acceptance criteria: the registry lists the metrics required by the
design document; unregistered metrics are rejected from formal
comparison; registry version participates in the config fingerprint.
"""

from __future__ import annotations

import pytest

from eval.contracts.internal import MetricResult, Result
from eval.metrics import (
    REGISTRY,
    REGISTRY_CONTENT_VERSION,
    REGISTRY_VERSION,
    RANKING_METRIC_IDS,
    UnknownMetricError,
    get_metric,
    registry_entries,
    validate_result_metrics,
)

REQUIRED_METRICS = {
    # retrieval (verifiable extractive recall)
    "verifiable_session_recall_macro",
    "verifiable_session_recall_micro",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "budgeted_session_recall",
    # generated-source diagnostics (explicitly not recall)
    "derivation_source_coverage",
    # qa aggregates
    "planned_question_score",
    "scored_accuracy",
    "runnable_coverage",
    "abstention_accuracy",
    # attribution distribution
    "attribution_hit_correct",
    "attribution_hit_wrong",
    "attribution_miss_correct",
    "attribution_miss_wrong",
    # operations
    "operations_pass_rate",
    "operations_support_coverage",
}

FP = "b" * 64


class TestRegistryContent:
    def test_all_design_metrics_registered(self):
        missing = REQUIRED_METRICS - set(REGISTRY)
        assert not missing, f"missing required metrics: {sorted(missing)}"

    def test_recall_k_set_is_exactly_1_3_5(self):
        ks = {
            m.metric_id.split("recall_at_")[-1]
            for m in REGISTRY.values()
            if m.metric_id.startswith("recall_at_")
        }
        assert ks == {"1", "3", "5"}

    def test_every_definition_has_required_fields(self):
        for metric in REGISTRY.values():
            assert metric.version >= 1
            assert metric.suite in ("qa", "operations", "all")
            assert metric.unit
            assert metric.denominator
            assert metric.description
            assert isinstance(metric.ranking, bool)
            for condition in metric.na_conditions:
                assert condition  # non-empty strings only

    def test_registry_version_is_v1_and_stable(self):
        assert REGISTRY_VERSION == "1"
        assert REGISTRY_CONTENT_VERSION == "metrics-registry@1"

    def test_entries_are_stable_across_calls(self):
        first = [m.metric_id for m in registry_entries()]
        second = [m.metric_id for m in registry_entries()]
        assert first == second
        assert len(first) == len(REGISTRY)

    def test_ranking_metrics_flagged(self):
        assert "verifiable_session_recall_macro" in RANKING_METRIC_IDS
        assert "derivation_source_coverage" not in RANKING_METRIC_IDS
        assert "planned_question_score" not in RANKING_METRIC_IDS

    def test_derivation_coverage_is_explicitly_diagnostic(self):
        metric = get_metric("derivation_source_coverage")
        assert metric.evidence_mode == "generated"
        assert not metric.ranking

    def test_get_metric_returns_definition(self):
        metric = get_metric("budgeted_session_recall")
        assert metric.suite == "qa"
        assert metric.evidence_mode == "extractive"

    def test_unknown_metric_raises_structured_error(self):
        with pytest.raises(UnknownMetricError, match="not registered"):
            get_metric("macro_f1")


def _result(suite: str = "qa", metric_id: str = "scored_accuracy") -> Result:
    metric = MetricResult.model_validate(
        {
            "metric_id": metric_id,
            "status": "not_applicable",
            "value": None,
            "reason": "test",
        }
    )
    return Result.model_validate(
        {
            "run_id": "r1",
            "sample_handle": "s1",
            "namespace": "ns1",
            "config_fingerprint": FP,
            "suite": suite,
            "qa_status": "scored" if suite == "qa" else None,
            "operation_status": "passed" if suite == "operations" else None,
            "correct": True if suite == "qa" else None,
            "attribution": "hit_correct" if suite == "qa" else None,
            "failed_stage": None,
            "metrics": [metric],
            "stage_states": {},
            "artifact_refs": {},
            "attempts": [],
        }
    )


class TestResultMetricValidation:
    def test_unregistered_metric_rejected(self):
        with pytest.raises(UnknownMetricError):
            validate_result_metrics(_result(metric_id="macro_f1"))

    def test_registered_metric_accepted(self):
        validate_result_metrics(_result())
        validate_result_metrics(
            _result(suite="operations", metric_id="operations_pass_rate")
        )

    def test_suite_mismatch_rejected(self):
        with pytest.raises(ValueError, match="not valid for suite"):
            validate_result_metrics(
                _result(suite="operations", metric_id="scored_accuracy")
            )

    def test_qa_metric_rejected_in_operations_result(self):
        with pytest.raises(ValueError):
            validate_result_metrics(
                _result(metric_id="operations_pass_rate")
            )

    def test_rejects_non_result_input(self):
        with pytest.raises(TypeError):
            validate_result_metrics("not-a-result")
