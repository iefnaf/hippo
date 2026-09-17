"""Shared validation concerns for eval data contracts.

Implements the general conventions from the data contract document: schema
versioning (first version is 1), timestamp formats (dataset dates vs UTC
run timestamps), and structured validation errors with JSON-pointer style
locators.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, ValidationError

#: First artifact schema version. Incompatible artifact changes bump this.
SCHEMA_VERSION = 1

#: Official dataset date format ('YYYY-MM-DD').
DATE_FORMAT = "%Y-%m-%d"

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?(Z|[+-]\d{2}:\d{2})$"
)

DATETIME_LOCAL_MSG = (
    "timestamp is UTC-only; use 'YYYY-MM-DDTHH:MM:SS[.ffffff]Z' "
    "or an explicit +/-HH:MM offset instead of a naive local time"
)


def parse_dataset_time(value: str) -> tuple[_dt.date | _dt.datetime, str]:
    """Parse a dataset date or timezone-aware datetime string.

    Dataset times keep their native precision ('YYYY-MM-DD' or a datetime
    with explicit timezone). Naive datetimes are rejected: we must not
    invent a timezone for data that lacks one, nor substitute machine-local
    time. Returns (parsed, mode) where mode is 'date' or 'datetime'.
    """
    if _DATE_RE.fullmatch(value):
        return _dt.datetime.strptime(value, DATE_FORMAT).date(), "date"
    if _DATETIME_RE.fullmatch(value):
        parsed = _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed, "datetime"
    if _DATE_RE.match(value) or "T" in value or " " in value.strip():
        raise ValueError(
            "dataset times must be 'YYYY-MM-DD' or "
            "'YYYY-MM-DDTHH:MM:SS[.ffffff](Z|+/-HH:MM)'; naive datetimes "
            f"are not allowed: {value!r}"
        )
    raise ValueError(
        f"invalid dataset time {value!r}: expected 'YYYY-MM-DD' or "
        "'YYYY-MM-DDTHH:MM:SS[.ffffff](Z|+/-HH:MM)'"
    )


def parse_run_timestamp(value: str) -> _dt.datetime:
    """Parse a UTC run-attempt timestamp ('...Z' or explicit offset)."""
    try:
        parsed = _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"invalid timestamp {value!r}: expected "
            "'YYYY-MM-DDTHH:MM:SS[.ffffff]Z' (UTC)"
        ) from exc
    if parsed.tzinfo is None:
        raise ValueError(DATETIME_LOCAL_MSG)
    return parsed


def now_utc() -> str:
    """Current UTC time as a contract-formatted timestamp string."""
    return (
        _dt.datetime.now(_dt.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%S.%f")
        + "Z"
    )


class ContractModel(BaseModel):
    """Base class for all contract models.

    Uses extra='forbid' so unknown fields fail validation instead of
    silently leaking upstream annotations (e.g. has_answer) into
    adapter-visible objects.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True,
    )


class SchemaVersionedModel(ContractModel):
    """Base for artifact envelope models carrying a schema version."""

    schema_version: int = SCHEMA_VERSION

    @classmethod
    def load_json(cls, raw: str | bytes) -> Self:
        """Load and validate a JSON document.

        Raises ContractError with a located message on failure.
        """
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ContractError(
                code="invalid_json",
                message=f"document is not valid JSON: {exc.msg}",
                location=f"(line {exc.lineno}, column {exc.colno})",
            ) from exc
        if not isinstance(doc, dict):
            raise ContractError(
                code="invalid_type",
                message=(
                    "expected a JSON object at document root, got "
                    + type(doc).__name__
                ),
                location="",
            )
        version = doc.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ContractError(
                code="unsupported_schema_version",
                message=(
                    f"artifact schema_version {version!r} is not supported "
                    f"(expected {SCHEMA_VERSION}); refusing to parse"
                ),
                location="/schema_version",
            )
        try:
            return cls.model_validate(doc)
        except ValidationError as exc:
            raise error_to_contract_error(exc, cls.__name__) from exc

    def dump_json(self) -> str:
        """Serialize with schema_version included."""
        return self.model_dump_json()


class ContractError(ValueError):
    """Structured, locatable contract validation error.

    Never a bare exception: callers get code/message/location (a
    JSON-pointer style path into the document) plus optional per-field
    details.
    """

    def __init__(
        self,
        code: str,
        message: str,
        location: str = "",
        details: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.location = location
        self.details = details or {}

    def __str__(self) -> str:
        loc = f" at {self.location}" if self.location else ""
        suffix = (
            "; ".join(f"{k}: {v}" for k, v in self.details.items())
            if self.details
            else ""
        )
        return f"{self.code}{loc}: {self.message}" + (
            f" ({suffix})" if suffix else ""
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "details": dict(self.details),
        }


def error_to_contract_error(exc: ValidationError, cls_name: str) -> ContractError:
    """Turn a pydantic ValidationError into a located ContractError."""
    errors = exc.errors()
    if not errors:
        return ContractError(
            "contract_validation", f"{cls_name}: validation failed", "/"
        )
    err = errors[0]
    loc_parts = [str(part) for part in err.get("loc", ())]
    loc = "/" + "/".join(loc_parts)
    field = ".".join(loc_parts) if loc_parts else cls_name
    return ContractError(
        code=f"contract_validation:{err.get('type', 'value_error')}",
        message=f"{cls_name}.{field}: {err.get('msg', 'validation failed')}",
        location=loc or "/",
        details={
            str(i): f"{'/'.join(str(p) for p in e.get('loc', ()))}: "
            f"{e.get('msg')}"
            for i, e in enumerate(errors[1:4])
        },
    )
