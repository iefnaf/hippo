"""Run directory layout, artifact persistence and checksummed references.

Every run produces a directory:

    <out>/<run_id>/
      run.json              RunManifest (plan model: expected call counts)
      config.json           immutable ConfigArtifact snapshot
      samples.jsonl         one ResultArtifact per sample (stage states,
                            attempts with timing and usage, artifact refs)
      report.json/.md       run summary (statuses, registered metrics,
                            2x2 attribution, budget composition, cost model)
      artifacts/<handle>/   attempts.json, raw_evidence.json (pre-truncation
                            diagnostic), prepared_evidence.json (post-budget),
                            reader_result.json, judge_record.json (private:
                            carries gold), scoring.json (private trace)

Artifact references are checksummed strings 'sha256:<hex>:<relative
path>'; inline values (receipts, prepared objects) are referenced as
'sha256:<hex>:inline' so every persisted record can be verified by
digest without a second copy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from eval.config import ConfigArtifact
from eval.contracts.adapter import ErrorInfo, ResourceUsage
from eval.contracts.common import (
    SCHEMA_VERSION,
    ContractError,
    ContractModel,
    SchemaVersionedModel,
)
from eval.contracts.internal import MetricResult, ResultArtifact


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inline_ref(value: Any) -> str:
    """Checksummed reference to an in-record value (no separate file)."""
    if hasattr(value, "model_dump_json"):
        payload = value.model_dump_json()
    else:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return f"sha256:{_sha256_bytes(payload.encode('utf-8'))}:inline"


class SamplePlanCounts(SchemaVersionedModel):
    """Expected call counts for one sample; a function of the config.

    For operations-suite checks the counts describe the calls the check
    makes when it actually runs (a check gated out by missing
    capabilities is not_supported and makes no calls).
    """

    sessions: int = Field(ge=1)
    ingest_calls: int = Field(ge=1)
    await_ready_calls: int = Field(ge=0)
    update_calls: int = Field(default=0, ge=0)
    delete_calls: int = Field(default=0, ge=0)
    inspect_calls: int = Field(default=0, ge=0)
    open_calls: int = Field(ge=1)
    close_calls: int = Field(ge=1)
    retrieve_calls: int = Field(ge=0)
    reader_calls: int = 0
    judge_calls: int = 0


class RunManifest(SchemaVersionedModel):
    """Run-level manifest written before the first sample runs."""

    run_id: str = Field(min_length=1)
    command: str = "run"
    suite: Literal["qa", "operations"] = "qa"
    created_at: str
    config_name: str
    config_fingerprint: str
    metrics_registry_version: str
    dataset_plan: str
    sample_plan_id: str
    sample_ids: tuple[str, ...]
    evidence_token_budget: int = Field(gt=0)
    memory_declaration: dict[str, Any]
    async_mutation: bool
    plan_counts: dict[str, SamplePlanCounts]


class AttemptEntry(SchemaVersionedModel):
    """One recorded call attempt (or lifecycle call) with full payloads.

    The Result JSONL keeps contract-shaped StageAttempt rows; this log
    additionally embeds inputs and outputs (sessions, requests, receipts,
    raw evidence) so every attempt is auditable end to end. It is a
    private harness artifact and never an adapter input.
    """

    attempt_id: str = Field(min_length=1)
    stage: str | None
    method: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    operation_id: str | None
    started_at: str
    ended_at: str | None
    elapsed_ms: float | None = Field(default=None, ge=0)
    input: dict[str, Any]
    output: Any = None
    error: ErrorInfo | None = None
    usage: ResourceUsage | None = None
    attempt_kind: Literal["logical", "retry", "replay"] = "logical"


class AttemptLogArtifact(SchemaVersionedModel):
    """Persisted per-sample attempt log."""

    run_id: str
    sample_handle: str
    namespace: str
    entries: list[AttemptEntry]


class CheckAssertion(ContractModel):
    """One deterministic assertion of an operations-suite check.

    Assertions are computed by the program only; no LLM judge
    participates in deciding whether an operation succeeded.
    """

    name: str = Field(min_length=1)
    passed: bool
    expected: str
    observed: str


class OperationCheckDetail(SchemaVersionedModel):
    """Persisted detail of one operations-suite check item.

    target_memory_ids lists the stable ids the check used as explicit
    operation targets; every one of them originates from a
    MutationReceipt, never from retrieval output.
    """

    run_id: str
    check_id: str
    operation_status: Literal["pending", "passed", "failed", "not_supported"]
    namespaces: tuple[str, ...]
    target_memory_ids: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    missing_capabilities: tuple[str, ...]
    reason: str | None
    failed_stage: str | None
    assertions: list[CheckAssertion]


class OperationsSummaryArtifact(SchemaVersionedModel):
    """Run-level operations summary: pass/fail/not-supported are counted
    separately and never merged; pending items keep the run incomplete."""

    run_id: str
    planned_checks: int = Field(ge=1)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    not_supported: int = Field(ge=0)
    pending: int = Field(ge=0)
    pass_rate: float | None = Field(default=None, ge=0, le=1)
    support_coverage: float | None = Field(default=None, ge=0, le=1)
    metrics: list[MetricResult]
    checks: list[OperationCheckDetail]


class RunStore:
    """Creates the run directory and persists artifacts immutably."""

    def __init__(self, base_dir: Path, run_id: str) -> None:
        self.base_dir = Path(base_dir)
        self.run_id = run_id
        self.dir = self.base_dir / run_id
        self._created = False
        self._resuming = False

    # -- lifecycle ---------------------------------------------------------

    def create(self, manifest: RunManifest, config_artifact: ConfigArtifact) -> None:
        if self._created:
            raise ContractError(
                code="run_already_created",
                message=f"run store for {self.run_id} was already created",
                location="(store)",
            )
        if self.dir.exists():
            raise ContractError(
                code="run_dir_exists",
                message=(
                    f"run directory {self.dir} already exists; runs are "
                    "immutable, start a new run instead"
                ),
                location="(run dir)",
            )
        (self.dir / "artifacts").mkdir(parents=True, exist_ok=False)
        self._write_json("run.json", manifest.model_dump_json())
        # Immutable config snapshot: written exactly once.
        self._write_json("config.json", config_artifact.model_dump_json())
        self._created = True

    def _write_json(self, rel: str, payload: str, overwrite: bool = False) -> str:
        path = self.dir / rel
        if path.exists():
            if not (overwrite and self._resuming):
                raise ContractError(
                    code="artifact_exists",
                    message=(
                        f"artifact {rel} already exists in run {self.run_id}; "
                        "run artifacts are immutable"
                    ),
                    location=f"/{rel}",
                )
            path.unlink()
        path.write_text(payload, encoding="utf-8")
        return f"sha256:{_sha256_bytes(payload.encode('utf-8'))}:{rel}"

    def _write_bytes(self, rel: str, payload: bytes, overwrite: bool = False) -> str:
        path = self.dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if not (overwrite and self._resuming):
                raise ContractError(
                    code="artifact_exists",
                    message=f"artifact {rel} already exists in run {self.run_id}",
                    location=f"/{rel}",
                )
            path.unlink()
        path.write_bytes(payload)
        return f"sha256:{_sha256_bytes(payload)}:{rel}"

    def open_for_resume(self) -> None:
        """Reopen an existing run directory for checkpointed resume.

        Structural gate only (run.json present); semantic compatibility
        (config fingerprint, space identity, checkpoint schema versions)
        is checked by verify_resume_compatibility before this call. In
        resume mode the per-sample artifact writer suffixes re-run
        handles with .r<N> rounds instead of refusing, and summary
        artifacts may be overwritten.
        """
        if self._created:
            raise ContractError(
                code="run_already_created",
                message=f"run store for {self.run_id} was already created",
                location="(store)",
            )
        if not (self.dir / "run.json").exists():
            raise ContractError(
                code="resume_run_missing",
                message=(
                    f"run directory {self.dir} has no run.json; there is "
                    "no checkpoint to resume"
                ),
                location="(run dir)",
            )
        self._resuming = True
        self._created = True

    def append_resume_marker(self, doc: dict[str, Any]) -> None:
        """Append one audit line to resume.jsonl (created on first use)."""
        if not self._resuming:
            raise ContractError(
                code="not_resuming",
                message="resume markers can only be written in resume mode",
                location="(store)",
            )
        path = self.dir / "resume.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(doc, ensure_ascii=False, sort_keys=True) + "\n")

    # -- artifacts ---------------------------------------------------------

    def write_sample_artifacts(
        self,
        sample_handle: str,
        *,
        attempts_log: AttemptLogArtifact,
        raw_evidence: Any | None = None,
        prepared_evidence: Any | None = None,
        operations_check: Any | None = None,
        reader_result: Any | None = None,
        judge_record: Any | None = None,
        scoring_trace: Any | None = None,
    ) -> dict[str, str]:
        """Persist per-sample artifacts; returns checksummed refs.

        reader_result / judge_record / scoring_trace are private harness
        artifacts (the latter two carry gold): they are auditable records
        and never inputs to any tested component.
        """
        refs: dict[str, str] = {}
        safe = _safe_handle(sample_handle)
        if self._resuming:
            # A re-run sample keeps its prior artifacts untouched and
            # writes this round under a suffixed directory.
            base = safe
            n = 1
            while (self.dir / "artifacts" / f"{base}.r{n}" / "attempts.json").exists():
                n += 1
            safe = f"{base}.r{n}"
        refs["attempts"] = self._write_bytes(
            f"artifacts/{safe}/attempts.json",
            attempts_log.model_dump_json().encode("utf-8"),
        )
        if raw_evidence is not None:
            refs["raw_evidence"] = self._write_bytes(
                f"artifacts/{safe}/raw_evidence.json",
                raw_evidence.model_dump_json().encode("utf-8"),
            )
        if prepared_evidence is not None:
            refs["prepared_evidence"] = self._write_bytes(
                f"artifacts/{safe}/prepared_evidence.json",
                prepared_evidence.model_dump_json().encode("utf-8"),
            )
        if operations_check is not None:
            refs["operations_check"] = self._write_bytes(
                f"artifacts/{safe}/operations_check.json",
                operations_check.model_dump_json().encode("utf-8"),
            )
        if reader_result is not None:
            refs["reader_result"] = self._write_bytes(
                f"artifacts/{safe}/reader_result.json",
                reader_result.model_dump_json().encode("utf-8"),
            )
        if judge_record is not None:
            refs["judge_record"] = self._write_bytes(
                f"artifacts/{safe}/judge_record.json",
                judge_record.model_dump_json().encode("utf-8"),
            )
        if scoring_trace is not None:
            refs["scoring"] = self._write_bytes(
                f"artifacts/{safe}/scoring.json",
                scoring_trace.model_dump_json().encode("utf-8"),
            )
        return refs

    def write_operations_summary(self, summary: Any, overwrite: bool = False) -> str:
        """Persist the run-level operations summary; returns its ref."""
        return self._write_bytes(
            "operations_summary.json",
            summary.model_dump_json().encode("utf-8"),
            overwrite=overwrite,
        )

    def write_report(
        self, report_json: str, report_markdown: str, overwrite: bool = False
    ) -> dict[str, str]:
        """Persist the run summary report (JSON + Markdown)."""
        return {
            "report_json": self._write_json(
                "report.json", report_json, overwrite=overwrite
            ),
            "report_markdown": self._write_bytes(
                "report.md", report_markdown.encode("utf-8"), overwrite=overwrite
            ),
        }

    def append_result(self, result_artifact: Any) -> None:
        line = result_artifact.model_dump_json()
        path = self.dir / "samples.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # -- reads -------------------------------------------------------------

    def read_config_artifact(self) -> dict[str, Any]:
        return json.loads((self.dir / "config.json").read_text(encoding="utf-8"))

    def read_manifest(self) -> dict[str, Any]:
        return json.loads((self.dir / "run.json").read_text(encoding="utf-8"))

    def read_result_lines(self) -> list[dict[str, Any]]:
        path = self.dir / "samples.jsonl"
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def resolve_ref(self, ref: str) -> Path:
        """Resolve a checksummed file reference within this run."""
        try:
            _scheme, digest, rel = ref.split(":", 2)
        except ValueError as exc:
            raise ContractError(
                code="invalid_artifact_ref",
                message=f"artifact ref {ref!r} is not 'sha256:<hex>:<path>'",
                location="(ref)",
            ) from exc
        path = (self.dir / rel).resolve()
        if not str(path).startswith(str(self.dir.resolve())):
            raise ContractError(
                code="artifact_ref_escapes_run",
                message=f"artifact ref {ref!r} escapes the run directory",
                location="(ref)",
            )
        if _sha256_bytes(path.read_bytes()) != digest:
            raise ContractError(
                code="artifact_checksum_mismatch",
                message=f"artifact {rel} does not match its reference digest",
                location=f"/{rel}",
            )
        return path


def _safe_handle(handle: str) -> str:
    keep = [ch if (ch.isalnum() or ch in "-_.") else "_" for ch in handle]
    return "".join(keep)


def new_run_id(config_fingerprint: str, *, clock) -> str:
    """Unique run id: timestamp (UTC) plus a fingerprint slice."""
    stamp = clock().replace("-", "").replace(":", "").replace(".", "").replace("Z", "Z")
    return f"run-{stamp}-{config_fingerprint[:8]}"


def _require_checkpoint_schema(doc: dict[str, Any], name: str) -> None:
    """Refuse checkpoints written by an incompatible artifact schema."""
    version = doc.get("schema_version") if isinstance(doc, dict) else None
    if version != SCHEMA_VERSION:
        raise ContractError(
            code="resume_checkpoint_schema_incompatible",
            message=(
                f"{name} carries schema_version {version!r} but this "
                f"harness supports {SCHEMA_VERSION}; the checkpoint format "
                "is incompatible — refuse to reuse it and start a NEW run"
            ),
            location=f"/{name}/schema_version",
        )


def verify_resume_compatibility(
    config: Any,
    store: "RunStore",
    *,
    namespace_for_handle: Any,
) -> dict[str, Any]:
    """Validate an existing run directory for resume; refuse on mismatch.

    Checks (docs/design/eval-harness.md, 失败处理与恢复): checkpoint
    schema versions, config fingerprint, suite/sample-plan identity
    (space identity) and every recorded sample's namespace. Successful
    artifacts are NEVER overwritten by resume; this function only
    decides what may be reused. Every rejection raises ContractError
    with a message advising a NEW run. Returns
    {"manifest": dict, "results": {handle: ResultArtifact}} where
    results keeps the LAST line per handle.
    """
    if not (store.dir / "run.json").exists():
        raise ContractError(
            code="resume_run_missing",
            message=(
                f"run directory {store.dir} has no run.json; there is no "
                "checkpoint to resume — start a NEW run instead"
            ),
            location="(run dir)",
        )
    raw_manifest = json.loads((store.dir / "run.json").read_text(encoding="utf-8"))
    _require_checkpoint_schema(raw_manifest, "run.json")
    raw_config = json.loads((store.dir / "config.json").read_text(encoding="utf-8"))
    _require_checkpoint_schema(raw_config, "config.json")
    manifest = RunManifest.model_validate(raw_manifest)
    if manifest.run_id != store.run_id:
        raise ContractError(
            code="resume_run_id_mismatch",
            message=(
                f"run.json declares run id {manifest.run_id!r} but the "
                f"directory is {store.run_id!r}; refusing to mix runs — "
                "start a NEW run instead"
            ),
            location="/run_id",
        )
    if manifest.config_fingerprint != config.fingerprint():
        raise ContractError(
            code="resume_config_fingerprint_mismatch",
            message=(
                f"config fingerprint {config.fingerprint()} does not match "
                f"the run snapshot {manifest.config_fingerprint}; resume "
                "never mixes configurations — start a NEW run for the "
                "changed config"
            ),
            location="/config_fingerprint",
        )
    if manifest.suite != config.suite:
        raise ContractError(
            code="resume_suite_mismatch",
            message=(
                f"run was created for suite {manifest.suite!r} but the "
                f"config declares {config.suite!r}; start a NEW run instead"
            ),
            location="/suite",
        )
    if (
        manifest.sample_plan_id != config.sample_plan_id
        or list(manifest.sample_ids) != list(config.sample_ids)
        or manifest.dataset_plan != config.dataset_plan
    ):
        raise ContractError(
            code="resume_sample_plan_mismatch",
            message=(
                "the run's sample plan (dataset_plan/sample_plan_id/"
                "sample_ids) differs from the config; the persisted space "
                "identities do not match — start a NEW run instead"
            ),
            location="/sample_plan_id",
        )
    results: dict[str, ResultArtifact] = {}
    for line in store.read_result_lines():
        _require_checkpoint_schema(line, "samples.jsonl")
        artifact = ResultArtifact.load_json(json.dumps(line))
        result = artifact.result
        if result.run_id != store.run_id:
            raise ContractError(
                code="resume_run_id_mismatch",
                message=(
                    f"samples.jsonl carries run id {result.run_id!r} but "
                    f"the run is {store.run_id!r}; start a NEW run instead"
                ),
                location="/run_id",
            )
        expected_ns = namespace_for_handle(result.sample_handle)
        if result.namespace != expected_ns:
            raise ContractError(
                code="resume_space_identity_mismatch",
                message=(
                    f"sample {result.sample_handle!r} recorded namespace "
                    f"{result.namespace!r} but the resumed plan maps it to "
                    f"{expected_ns!r}; the persisted space identity does "
                    "not match — start a NEW run instead"
                ),
                location=f"/{result.sample_handle}/namespace",
            )
        results[result.sample_handle] = artifact
    return {"manifest": raw_manifest, "results": results}

