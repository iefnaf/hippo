"""Model version records and drift probes (M2, issue #8).

Reader and judge models are ROLLING ALIASES: the vendor can repoint a
name while the config text stays identical. Reproducibility strategy is
"drift discoverable, history auditable", not bit-level reproducibility:

- every formal run records FOUR version identifiers per model: the
  configured alias, the ``model`` field observed in server responses,
  the vendor-documented version (with its documentation date) and the
  run date (docs/design/eval-harness.md, Token 预算与模型配置);
- every formal run executes a FIXED probe set (first version: 10
  prompts with declared expected behavior) through the real reader and
  archives the outputs; clearly different probe outputs between two
  runs mean the model behind the alias changed;
- the comparison command treats differing observed response models as a
  configuration change (not same condition), never silently mixing
  runs across drift.

Probe prompts deliberately cover mixed languages, dates, list
restructuring and abstention so a repointed model is unlikely to pass
unchanged. Probes never enter sample metrics; they are run-header
diagnostics.
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import Field, model_validator

from eval.contracts.adapter import QueryContext, ResourceUsage
from eval.contracts.common import ContractError, SchemaVersionedModel, now_utc
from eval.contracts.internal import PreparedEvidence


# ---------------------------------------------------------------------------
# Probe sets (fixed; a changed set is a new probe-set id)
# ---------------------------------------------------------------------------

#: Fixed date used for every probe question (declared, never machine time).
PROBE_QUESTION_DATE = "2024-01-01"

#: First probe set: 10 prompts with declared expected behavior.
READER_DRIFT_PROBE_V1: tuple[dict[str, str], ...] = (
    {
        "prompt_id": "echo-en-1",
        "prompt": (
            "Repeat exactly the following sentence and nothing else: "
            "The migration to pnpm finished on 2026-09-03."
        ),
        "expected_behavior": "verbatim echo of the given sentence",
    },
    {
        "prompt_id": "echo-zh-1",
        "prompt": "原样复述下面这句话，不要添加任何内容：项目已经从 npm 迁移到 pnpm。",
        "expected_behavior": "verbatim echo of the given sentence (Chinese)",
    },
    {
        "prompt_id": "date-arithmetic-1",
        "prompt": (
            "If a sprint starts on 2026-09-01 and lasts 14 days, on which "
            "date does it end? Answer with the date only."
        ),
        "expected_behavior": "2026-09-14 (calendar arithmetic)",
    },
    {
        "prompt_id": "list-restructure-1",
        "prompt": (
            "List the following tools in alphabetical order, one per "
            "line, nothing else: pytest, uv, ruff, mypy."
        ),
        "expected_behavior": "four lines: mypy, pytest, ruff, uv",
    },
    {
        "prompt_id": "abstain-1",
        "prompt": (
            "You have no evidence about the CEO of Acme Corp. What is the "
            "CEO's name? If the evidence does not contain the answer, say "
            "you cannot answer."
        ),
        "expected_behavior": "explicit abstention (no invented name)",
    },
    {
        "prompt_id": "sum-arithmetic-1",
        "prompt": "Compute 17 * 23 and answer with the number only.",
        "expected_behavior": "391",
    },
    {
        "prompt_id": "lang-mix-1",
        "prompt": (
            "Answer with one word, in English: pnpm、npm、yarn 三个工具中 "
            "按字典序排在最前面的是哪个（按英文名比较）？"
        ),
        "expected_behavior": "npm (dictionary order on English names)",
    },
    {
        "prompt_id": "format-follow-1",
        "prompt": (
            "Answer with exactly three comma-separated words describing "
            "what a tokenizer does: no sentence, no period."
        ),
        "expected_behavior": "three comma-separated words",
    },
    {
        "prompt_id": "quote-boundary-1",
        "prompt": (
            "How many times does the word 'cache' appear in: 'clear the "
            "cache, then cache the result'? Answer with the number only."
        ),
        "expected_behavior": "2",
    },
    {
        "prompt_id": "date-format-1",
        "prompt": (
            "Today's date is given as 2024-01-01. Which weekday is it? "
            "Answer with the English weekday name only."
        ),
        "expected_behavior": "Monday",
    },
)

PROBE_SETS: dict[str, tuple[dict[str, str], ...]] = {
    "reader-drift-probe@1": READER_DRIFT_PROBE_V1,
}


def get_probe_set(probe_set_id: str) -> tuple[dict[str, str], ...]:
    """Return the fixed probe set or raise ValueError (config error)."""
    try:
        return PROBE_SETS[probe_set_id]
    except KeyError as exc:
        raise ValueError(
            f"unknown probe set id {probe_set_id!r}; known: "
            f"{sorted(PROBE_SETS)} (a changed probe set is a new id, "
            "otherwise archived probe outputs stop being comparable)"
        ) from exc


def probe_digest(probe_set_id: str) -> str:
    """Stable digest over a probe set's ids, prompts and expectations."""
    import hashlib
    import json

    payload = json.dumps(
        [probe_set_id, [dict(p) for p in get_probe_set(probe_set_id)]],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Version records and artifacts
# ---------------------------------------------------------------------------


class ModelVersionRecord(SchemaVersionedModel):
    """The four drift identifiers for one model role.

    alias is the configured rolling alias; response_model is what the
    server actually answered with (observed during the run); vendor_*
    document what the vendor's docs claimed at documentation time.
    """

    role: str = Field(min_length=1)
    alias: str = Field(min_length=1)
    response_model: str | None = None
    vendor_documented_version: str = ""
    vendor_documented_on: str = ""
    run_date: str = Field(min_length=1)


class ProbeCallRecord(SchemaVersionedModel):
    """One probe call's prompt, output, observed model and usage."""

    prompt_id: str = Field(min_length=1)
    prompt: str
    expected_behavior: str
    output: str | None = None
    response_model: str | None = None
    usage: ResourceUsage | None = None
    error: str | None = None


class ModelProbeArtifact(SchemaVersionedModel):
    """Archived drift-probe outputs of one run (run-header diagnostic)."""

    run_id: str = Field(min_length=1)
    probe_set_id: str = Field(min_length=1)
    probe_digest: str = Field(min_length=1)
    created_at: str
    calls: list[ProbeCallRecord]

    @model_validator(mode="after")
    def _call_rules(self):
        ok = sum(1 for c in self.calls if c.error is None)
        if not self.calls or ok == 0:
            raise ValueError(
                "a probe artifact with zero successful probe calls must "
                "not be written; the run fails fast instead"
            )
        return self


class ModelVersionsArtifact(SchemaVersionedModel):
    """Run-header model version record (four identifiers per role).

    Written once per run: declared identities at run start plus the
    response models observed during the run and the archived probe
    outputs (when a probe set is configured). Drift between two runs is
    discovered by comparing these records, never by trusting the alias.
    """

    run_id: str = Field(min_length=1)
    created_at: str
    code_version: str
    reader: ModelVersionRecord
    judge: ModelVersionRecord
    probe: ModelProbeArtifact | None = None


def empty_probe_prepared(tokenizer_id: str, counting_mode: str) -> PreparedEvidence:
    """Evidence-less PreparedEvidence for probe calls (fixed shape)."""
    return PreparedEvidence(
        rendered_text="",
        items=[],
        token_count=0,
        text_token_count=0,
        budget=None,
        counting_mode=counting_mode,  # type: ignore[arg-type]
        tokenizer_id=tokenizer_id,
        dropped_raw_indices=[],
    )


def run_reader_probes(
    *,
    run_id: str,
    probe_set_id: str,
    reader: Any,
    tokenizer_id: str,
    counting_mode: str,
    clock: Callable[[], str] = now_utc,
) -> ModelProbeArtifact:
    """Execute the fixed probe set through the real reader path.

    Every probe goes through reader.answer() with the shared query
    context shape (fixed declared question date, no evidence), so probes
    exercise exactly the production request path. Errors are recorded
    per call; if EVERY probe fails the run fails fast (a broken endpoint
    must not burn a whole sample run).
    """
    from eval.readers.base import ReaderError

    calls: list[ProbeCallRecord] = []
    for item in get_probe_set(probe_set_id):
        question = QueryContext(
            query=item["prompt"], question_date=PROBE_QUESTION_DATE
        )
        try:
            result = reader.answer(
                question, empty_probe_prepared(tokenizer_id, counting_mode)
            )
            calls.append(
                ProbeCallRecord(
                    prompt_id=item["prompt_id"],
                    prompt=item["prompt"],
                    expected_behavior=item["expected_behavior"],
                    output=result.hypothesis,
                    response_model=result.model,
                    usage=result.usage,
                    error=None,
                )
            )
        except ReaderError as exc:
            calls.append(
                ProbeCallRecord(
                    prompt_id=item["prompt_id"],
                    prompt=item["prompt"],
                    expected_behavior=item["expected_behavior"],
                    error=f"{exc.code}: {exc.message}",
                )
            )
        except Exception as exc:  # noqa: BLE001 - recorded per probe
            calls.append(
                ProbeCallRecord(
                    prompt_id=item["prompt_id"],
                    prompt=item["prompt"],
                    expected_behavior=item["expected_behavior"],
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    ok = sum(1 for c in calls if c.error is None)
    if ok == 0:
        raise ContractError(
            code="model_probe_failed",
            message=(
                f"every probe of set {probe_set_id!r} failed; refusing to "
                "start the sample run against a broken endpoint (first "
                f"error: {calls[0].error})"
            ),
            location="(probes)",
        )
    return ModelProbeArtifact(
        run_id=run_id,
        probe_set_id=probe_set_id,
        probe_digest=probe_digest(probe_set_id),
        created_at=clock(),
        calls=calls,
    )
