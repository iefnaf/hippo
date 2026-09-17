"""Experiment configuration and config fingerprint.

An experiment config fixes everything that must not vary within one
comparison condition: dataset plan (sample ids and smoke subset), memory
adapter declaration (baseline kind, evidence mode, capabilities), reader
and judge model identities, prompt/budget/tokenizer settings and the
run/compare parameters. The fingerprint is a deterministic sha256 over a
canonical JSON rendering; it embeds the metric registry version, so two
configs with different registry versions can never share a fingerprint
and runs remain comparable with themselves across invocations.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from eval.contracts.common import ContractError, ContractModel
from eval.metrics import (
    REGISTRY_CONTENT_VERSION,
    REGISTRY_VERSION,
    validate_metric_id,
)

#: Supported artifact schema version for config snapshots (first version 1).
CONFIG_SCHEMA_VERSION = 1

BASELINE_KINDS = ("none", "bm25", "full_history", "adapter")
QUESTION_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "temporal-reasoning",
    "knowledge-update",
    "multi-session",
)
DEFAULT_SMOKE_SAMPLE_IDS = (
    "smoke_single_session_user_0001",
    "smoke_single_session_assistant_0001",
    "smoke_single_session_preference_0001",
    "smoke_temporal_reasoning_0001",
    "smoke_knowledge_update_0001",
    "smoke_multi_session_0001",
    "smoke_abstention_0001",
    "smoke_abstention_0002",
)

#: Check items of the M1 operations suite (docs/design/eval-harness.md,
#: section 操作). An operations-suite config plans exactly these ids.
OPERATIONS_CHECK_IDS = (
    "auto_update",
    "explicit_update",
    "delete",
    "isolation",
    "persistence",
)


class MemoryPlan(ContractModel):
    """Declared memory implementation under test (fixed before the run)."""

    name: str = Field(min_length=1)
    baseline_kind: Literal["none", "bm25", "full_history", "adapter"]
    capabilities: tuple[str, ...] = Field(default=())
    config: dict[str, Any] = Field(default_factory=dict)


class ReaderPlan(ContractModel):
    """Reader model identity and generation parameters (fixed)."""

    model: str = Field(min_length=1)
    model_family: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    temperature: float
    max_output_tokens: int = Field(gt=0)
    tokenizer_id: str = Field(min_length=1)
    counting_mode: Literal["exact", "estimated", "test"]
    extra: dict[str, Any] = Field(default_factory=dict)


class JudgePlan(ContractModel):
    """Judge model identity and official protocol binding (fixed)."""

    model: str = Field(min_length=1)
    model_family: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    temperature: float
    protocol_id: str = Field(min_length=1)
    protocol_source_commit: str = Field(min_length=1)
    extra: dict[str, Any] = Field(default_factory=dict)


class RunParams(ContractModel):
    """Retry/timeout/concurrency parameters; changing any is a config change."""

    max_retries: int = Field(default=3, ge=0)
    backoff_base_s: float = Field(default=1.0, gt=0)
    await_ready_timeout_s: float = Field(default=300.0, gt=0)
    question_concurrency: int = Field(default=1, ge=1)


class ExperimentConfig(ContractModel):
    """Full experiment configuration.

    model_config sets extra='forbid' via ContractModel so a typo in a key
    surfaces as a located validation error instead of being ignored.
    """

    name: str = Field(min_length=1)
    suite: Literal["qa", "operations"] = "qa"
    dataset_plan: str = Field(min_length=1)
    sample_plan_id: str = Field(min_length=1)
    #: qa: dataset sample handles; operations: OPERATIONS_CHECK_IDS.
    sample_ids: tuple[str, ...] = Field(min_length=1)
    smoke_subset_ids: tuple[str, ...] = Field(default=DEFAULT_SMOKE_SAMPLE_IDS)
    memory: MemoryPlan
    reader: ReaderPlan
    judge: JudgePlan
    evidence_token_budget: int = Field(default=4096, gt=0)
    run_params: RunParams = Field(default_factory=RunParams)
    notes: str = ""

    @field_validator("sample_ids", "smoke_subset_ids")
    @classmethod
    def _unique_ids(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(v)) != len(v):
            dupes = sorted({s for s in v if v.count(s) > 1})
            raise ValueError(f"duplicate sample ids: {dupes}")
        return v

    @model_validator(mode="after")
    def _suite_rules(self) -> Self:
        if self.suite == "operations":
            unknown = [
                s for s in self.sample_ids if s not in OPERATIONS_CHECK_IDS
            ]
            if unknown:
                raise ValueError(
                    f"operations-suite sample_ids must come from "
                    f"{list(OPERATIONS_CHECK_IDS)}; unknown: {unknown}"
                )
            if self.smoke_subset_ids:
                raise ValueError(
                    "operations-suite configs must leave smoke_subset_ids "
                    "empty; the smoke subset is a qa-suite concept"
                )
        return self

    @model_validator(mode="after")
    def _smoke_rules(self) -> Self:
        if self.smoke_subset_ids:
            unknown = [
                s for s in self.smoke_subset_ids if s not in set(self.sample_ids)
            ]
            if unknown:
                raise ValueError(
                    f"smoke_subset_ids not part of sample_ids: {unknown}"
                )
            if len(self.smoke_subset_ids) != 8:
                raise ValueError(
                    "smoke subset must be exactly 8 questions (one per "
                    "question_type plus two abstention)"
                )
        return self

    @model_validator(mode="after")
    def _families_differ(self) -> Self:
        if self.reader.model_family == self.judge.model_family:
            raise ValueError(
                f"reader and judge must belong to different model families "
                f"(both are {self.reader.model_family!r}); same-family "
                "reader/judge invites self-preference bias"
            )
        return self

    # -- fingerprint --------------------------------------------------------

    def canonical_payload(self) -> dict[str, Any]:
        """Content that defines comparability; note this is not run output."""
        return {
            "config_schema_version": CONFIG_SCHEMA_VERSION,
            "name": self.name,
            "suite": self.suite,
            "dataset_plan": self.dataset_plan,
            "sample_plan_id": self.sample_plan_id,
            "sample_ids": list(self.sample_ids),
            "smoke_subset_ids": list(self.smoke_subset_ids),
            "memory": self.memory.model_dump(mode="json"),
            "reader": self.reader.model_dump(mode="json"),
            "judge": self.judge.model_dump(mode="json"),
            "evidence_token_budget": self.evidence_token_budget,
            "run_params": self.run_params.model_dump(mode="json"),
            # Metric registry version: definition drift must break
            # comparability even when every other field matches.
            "metrics_registry_version": REGISTRY_VERSION,
            "metrics_registry_content_version": REGISTRY_CONTENT_VERSION,
        }

    def fingerprint(self) -> str:
        """Deterministic sha256 over the canonical JSON rendering.

        Two invocations on the same config produce the same fingerprint;
        canonicalization (sorted keys, no whitespace) guarantees order
        independence of dict fields.
        """
        payload = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ConfigArtifact(ContractModel):
    """Immutable config snapshot artifact (schema_version included on dump)."""

    schema_version: int = CONFIG_SCHEMA_VERSION
    config: ExperimentConfig
    config_fingerprint: str
    metrics_registry_version: str = REGISTRY_VERSION

    @model_validator(mode="after")
    def _fingerprint_matches(self) -> Self:
        actual = self.config.fingerprint()
        if self.config_fingerprint != actual:
            raise ValueError(
                f"config_fingerprint mismatch: snapshot says "
                f"{self.config_fingerprint}, config hashes to {actual}"
            )
        return self


def _toml_value(value: Any) -> Any:
    """Normalize TOML-parsed values for pydantic (lists -> tuples)."""
    if isinstance(value, list):
        return tuple(_toml_value(v) for v in value)
    if isinstance(value, dict):
        return {k: _toml_value(v) for k, v in value.items()}
    return value


class _ConfigLoadError(ValueError):
    pass


def _locate_toml_error(exc: Exception, cls_name: str) -> ContractError:
    return ContractError(
        code="config_validation",
        message=f"{cls_name}: {exc}",
        location="/config",
    )


def load_config_dict(data: dict[str, Any]) -> ExperimentConfig:
    """Validate a raw config mapping; raises ContractError with location."""
    try:
        return ExperimentConfig.model_validate(_toml_value(data))
    except Exception as exc:  # pydantic ValidationError or ValueError
        raise _locate_toml_error(exc, "ExperimentConfig") from exc


def load_config_toml(path: str | Path) -> ExperimentConfig:
    """Load and validate a TOML config file; structured errors on failure."""
    path = Path(path)
    if not path.exists():
        raise ContractError(
            code="config_missing",
            message=f"config file not found: {path}",
            location="(path)",
        )
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ContractError(
            code="config_invalid_toml",
            message=f"{path}: {exc}",
            location="(toml)",
        ) from exc
    return load_config_dict(raw)


def load_config_artifact_json(raw: str | bytes) -> ConfigArtifact:
    """Load a persisted config snapshot; version-checked."""
    import json as _json

    from eval.contracts.common import error_to_contract_error

    try:
        doc = _json.loads(raw)
    except _json.JSONDecodeError as exc:
        raise ContractError(
            code="invalid_json",
            message=f"config artifact is not valid JSON: {exc.msg}",
            location=f"(line {exc.lineno}, column {exc.colno})",
        ) from exc
    if not isinstance(doc, dict):
        raise ContractError(
            code="invalid_type",
            message="config artifact root must be a JSON object",
            location="",
        )
    version = doc.get("schema_version")
    if version != CONFIG_SCHEMA_VERSION:
        raise ContractError(
            code="unsupported_schema_version",
            message=(
                f"config artifact schema_version {version!r} is not supported "
                f"(expected {CONFIG_SCHEMA_VERSION}); refusing to parse"
            ),
            location="/schema_version",
        )
    try:
        return ConfigArtifact.model_validate(_toml_value(doc))
    except Exception as exc:
        raise _locate_toml_error(exc, "ConfigArtifact") from exc


def assert_metric_ids_registered(metric_ids: list[str]) -> None:
    """Guard used by reporters: every formal metric id must be registered."""
    for metric_id in metric_ids:
        validate_metric_id(metric_id)
