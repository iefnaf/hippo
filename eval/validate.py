"""Structural validation of configs and manual fixtures.

validate_config / validate_fixture / validate_all never raise bare
exceptions for bad input: they collect located ContractError items into a
ValidationReport with per-document ok/fail rows and non-zero exit
semantics for the CLI.

Two fixture shapes are supported:

- run artifacts (raw_evidence / prepared_evidence / reader_result /
  judge_record / result): schema-versioned artifact documents as persisted
  under a run directory;
- contract object fixtures for adapter-visible types (session, evidence,
  retrieval_request, ...): wrapped in a ContractFixture envelope that
  carries schema_version and kind so every fixture file is versioned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from eval.config import load_config_toml
from eval.contracts.adapter import (
    ErrorInfo,
    Evidence,
    MemoryState,
    Message,
    MutationReceipt,
    QueryContext,
    RetrievalRequest,
    ResourceUsage,
    Session,
    SourceRef,
    SourceSpan,
)
from eval.contracts.common import (
    SCHEMA_VERSION,
    ContractError,
    ContractModel,
    error_to_contract_error,
)
from eval.contracts.internal import (
    JudgeRecordArtifact,
    PreparedEvidenceArtifact,
    RawEvidenceArtifact,
    ReaderResultArtifact,
    ResultArtifact,
    ScoringTraceArtifact,
)

#: Fixture kind -> artifact model (each carries schema_version=1 on dump).
ARTIFACT_MODELS: dict[str, type] = {
    "raw_evidence": RawEvidenceArtifact,
    "prepared_evidence": PreparedEvidenceArtifact,
    "reader_result": ReaderResultArtifact,
    "judge_record": JudgeRecordArtifact,
    "scoring_trace": ScoringTraceArtifact,
    "result": ResultArtifact,
}

#: Adapter-visible contract object kinds loadable via a versioned envelope.
CONTRACT_OBJECT_MODELS: dict[str, type] = {
    "session": Session,
    "message": Message,
    "evidence": Evidence,
    "source_ref": SourceRef,
    "source_span": SourceSpan,
    "query_context": QueryContext,
    "retrieval_request": RetrievalRequest,
    "memory_state": MemoryState,
    "resource_usage": ResourceUsage,
    "error_info": ErrorInfo,
    "mutation_receipt": MutationReceipt,
}

KNOWN_KINDS = sorted(set(ARTIFACT_MODELS) | set(CONTRACT_OBJECT_MODELS))


class ContractFixture(ContractModel):
    """Envelope for fixtures of adapter-visible contract objects."""

    schema_version: int = SCHEMA_VERSION
    kind: str = Field(min_length=1)
    payload: dict[str, Any]

    @classmethod
    def load_json(cls, raw: str | bytes) -> "ContractFixture":
        import json

        try:
            doc = json.loads(raw)
        except Exception as exc:  # noqa: BLE001 - reported structurally below
            raise ContractError(
                code="invalid_json",
                message=f"fixture is not valid JSON: {exc}",
                location="(document)",
            ) from exc
        if not isinstance(doc, dict):
            raise ContractError(
                code="invalid_type",
                message="fixture root must be a JSON object",
                location="",
            )
        version = doc.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ContractError(
                code="unsupported_schema_version",
                message=(
                    f"fixture schema_version {version!r} is not supported "
                    f"(expected {SCHEMA_VERSION})"
                ),
                location="/schema_version",
            )
        try:
            return cls.model_validate(doc)
        except Exception as exc:
            raise error_to_contract_error(
                exc, "ContractFixture"
            ) from exc


class DocumentReport:
    """Outcome for one validated document."""

    def __init__(
        self,
        path: str,
        kind: str,
        ok: bool,
        errors: list[ContractError] | None = None,
    ) -> None:
        self.path = path
        self.kind = kind
        self.ok = ok
        self.errors = errors or []

    def summary_line(self) -> str:
        status = "ok" if self.ok else "FAIL"
        head = f"[{status}] {self.kind} {self.path}"
        if self.ok:
            return head
        lines = [head]
        for err in self.errors:
            lines.append(
                f"       {err.code} at {err.location or '(root)'}: {err.message}"
            )
            for k, v in err.details.items():
                lines.append(f"         ({k}) {v}")
        return "\n".join(lines)


class ValidationReport:
    """Aggregate report across configs and fixtures."""

    def __init__(self) -> None:
        self.documents: list[DocumentReport] = []

    @property
    def ok(self) -> bool:
        return all(d.ok for d in self.documents)

    def exit_code(self) -> int:
        return 0 if self.ok else 1

    def render(self) -> str:
        lines = [d.summary_line() for d in self.documents]
        total = len(self.documents)
        failed = sum(1 for d in self.documents if not d.ok)
        lines.append("")
        lines.append(f"{total - failed}/{total} documents valid, {failed} failed")
        return "\n".join(lines)


def _read_text(path: Path, kind: str) -> tuple[str | None, ContractError | None]:
    try:
        return path.read_text(encoding="utf-8"), None
    except OSError as exc:
        return None, ContractError("fixture_unreadable", str(exc), "(path)")


def _detect_kind(path: Path) -> str:
    """Pick the fixture kind from dotted filename components.

    Accepts kind.name.json, name.kind.json and kind.json; the first
    component matching a known kind wins.
    """
    for component in path.name.split("."):
        if component in ARTIFACT_MODELS or component in CONTRACT_OBJECT_MODELS:
            return component
    return ""


def validate_config(path: str | Path) -> DocumentReport:
    """Validate one TOML experiment config; located errors on failure."""
    path = Path(path)
    try:
        config = load_config_toml(path)
    except ContractError as exc:
        return DocumentReport(str(path), "config", False, [exc])
    errors: list[ContractError] = []
    if config.fingerprint() != config.fingerprint():
        errors.append(
            ContractError(
                code="fingerprint_unstable",
                message="fingerprint() is not deterministic for this config",
                location="(fingerprint)",
            )
        )
    if (
        config.reader.counting_mode == "test"
        and not config.reader.tokenizer_id.startswith("test:")
    ):
        errors.append(
            ContractError(
                code="counting_mode_mismatch",
                message=(
                    "reader.counting_mode='test' requires tokenizer_id "
                    "prefixed with 'test:' (M1 fake counter)"
                ),
                location="/reader/tokenizer_id",
            )
        )
    if (
        config.memory.baseline_kind == "bm25"
        and "extractive_evidence" not in set(config.memory.capabilities)
    ):
        errors.append(
            ContractError(
                code="capability_declaration_missing",
                message=(
                    "BM25 baseline must declare the 'extractive_evidence' "
                    "capability to enter extractive recall comparisons"
                ),
                location="/memory/capabilities",
            )
        )
    return DocumentReport(str(path), "config", not errors, errors)


def validate_fixture(path: str | Path, kind: str | None = None) -> DocumentReport:
    """Validate one JSON fixture (artifact or contract-object envelope)."""
    path = Path(path)
    detected = kind or _detect_kind(path)
    if detected not in ARTIFACT_MODELS and detected not in CONTRACT_OBJECT_MODELS:
        return DocumentReport(
            str(path),
            detected or "unknown",
            False,
            [
                ContractError(
                    code="unknown_fixture_kind",
                    message=(
                        f"fixture kind {detected!r} (from {path.name}) is not "
                        f"known; expected one of {KNOWN_KINDS} as a suffix "
                        "like evidence.name.json or name.evidence.json"
                    ),
                    location="(kind)",
                )
            ],
        )
    raw, read_error = _read_text(path, detected)
    if read_error is not None:
        return DocumentReport(str(path), detected, False, [read_error])
    assert raw is not None

    errors: list[ContractError] = []
    if detected in ARTIFACT_MODELS:
        model = ARTIFACT_MODELS[detected]
        try:
            artifact = model.load_json(raw)
        except ContractError as exc:
            errors.append(exc)
        else:
            if detected == "result":
                from eval.metrics import UnknownMetricError, validate_result_metrics

                try:
                    validate_result_metrics(artifact.result)
                except UnknownMetricError as exc:
                    errors.append(
                        ContractError(
                            code="unregistered_metric",
                            message=str(exc),
                            location="/result/metrics",
                        )
                    )
    else:
        try:
            fixture = ContractFixture.load_json(raw)
        except ContractError as exc:
            errors.append(exc)
        else:
            if fixture.kind != detected:
                errors.append(
                    ContractError(
                        code="fixture_kind_mismatch",
                        message=(
                            f"envelope kind {fixture.kind!r} does not match "
                            f"file-derived kind {detected!r}"
                        ),
                        location="/kind",
                    )
                )
            else:
                model = CONTRACT_OBJECT_MODELS[detected]
                try:
                    model.model_validate(fixture.payload)
                except Exception as exc:
                    err = error_to_contract_error(exc, model.__name__)
                    errors.append(
                        ContractError(
                            code=err.code,
                            message=err.message,
                            location="/payload" + err.location,
                        )
                    )
    return DocumentReport(str(path), detected, not errors, errors)


def validate_all(
    config_paths: list[str | Path],
    fixture_paths: list[str | Path],
) -> ValidationReport:
    """Validate configs and fixtures, aggregating one report."""
    report = ValidationReport()
    for p in config_paths:
        report.documents.append(validate_config(p))
    for p in fixture_paths:
        report.documents.append(validate_fixture(p))
    return report
