"""Calibration sampling: the plan, run-output collection and the
DE-IDENTIFIED blind annotation worksheets (issue #9).

The calibration population is the dev-side outputs of TWO run
conditions (e.g. bm25 vs none on the dev50 split): the design fixes
100 random samples = 50 dev questions x 2 conditions, stratified over
the six question types and the abstention subset. Selection is a pure
function of the input multiset and the fixed seed (sha256 hash
ranking, the same determinism strategy as eval.datasets.split — no
RNG implementation detail is ever observable).

Two cohorts, never merged in statistics (docs/design/eval-harness.md,
Judge 校准):

- RANDOM cohort (100): estimates the overall agreement. When the
  population exceeds the target, members are stratified-selected
  proportionally (largest remainder, hash-ranked inside each stratum).
- BOUNDARY cohort (20): deliberately picked hard cases — abstention
  items, responses likely PARTIAL (shorter than the gold answer;
  official protocol says subset -> no), refusal-phrased and
  verbose-hedged responses. Diagnoses judge lenient/strict only.

De-identification discipline for the worksheets:

- the annotator sees ONLY item_id, question, gold_answer, response,
  question_type, is_abstention — never the condition, the run, the
  judge verdict or any evidence/retrieval content;
- item ids are neutral sequential ids assigned AFTER the presentation
  shuffle, so the id order carries no information;
- each cohort gets its own worksheet (boundary items may overlap the
  random cohort when the population is only 100 pairs; the overlap is
  recorded in the plan and the two worksheets never share a row).

The plan artifact (harness-private) keeps the full mapping item ->
(condition, run, sample); the annotation import accepts only what a
worksheet can contain.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import Field, model_validator

from eval.calibration.criteria import (
    BOUNDARY_SAMPLE_SIZE,
    RANDOM_SAMPLE_SIZE,
    SELF_CONSISTENCY_SIZE,
)
from eval.contracts.common import ContractError, SchemaVersionedModel
from eval.datasets.split import hash_rank

QUESTION_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "temporal-reasoning",
    "knowledge-update",
    "multi-session",
)

CALIBRATION_SEED = "hippo-judge-calibration@1"

#: Refusal / uncertainty phrasings (EN + ZH) for the boundary bucket
#: "refusal_phrasing" (deterministic substring match, no regex).
REFUSAL_MARKERS = (
    "cannot answer",
    "can't answer",
    "not sure",
    "no information",
    "not mentioned",
    "unable to determine",
    "i don't know",
    "无法回答",
    "无法确定",
    "不能确定",
    "没有提到",
    "未提及",
    "不知道",
    "信息不足",
)

#: Verbose-hedged boundary bucket: responses at least this many
#: characters are unusually hedged phrasing against a QA gold answer.
VERBOSE_MIN_CHARS = 1200

#: Partial-answer proxy: a TERSE response (< this many characters) on a
#: multi-fact question type likely omits intermediate steps (official
#: protocol: subset -> no). Real LongMemEval gold answers are short
#: facts while responses are sentences, so "shorter than gold" almost
#: never fires; terseness is the workable deterministic proxy.
PARTIAL_MAX_CHARS = 80
MULTI_FACT_TYPES = frozenset(
    {"multi-session", "knowledge-update", "temporal-reasoning"}
)

#: Worksheet CSV columns — the EXACT set the annotator may see.
WORKSHEET_COLUMNS = (
    "item_id",
    "question",
    "gold_answer",
    "response",
    "question_type",
    "is_abstention",
    "annotation",
    "note",
    "annotated_at",
)

ANNOTATION_LABELS = ("yes", "no", "cannot_judge")


class CalibrationCandidate(SchemaVersionedModel):
    """One annotatable output: a (question, condition) pair.

    Built from a run's judge_record artifacts (question, gold answer,
    hypothesis and the abstention flag all live there). The condition
    label is an opaque A/B marker; it never enters any worksheet.
    """

    condition: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    sample_handle: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_answer: str
    hypothesis: str = Field(min_length=1)
    question_type: str
    is_abstention: bool


class CalibrationItem(SchemaVersionedModel):
    """One planned calibration item (harness-private: carries the
    condition mapping; the worksheet exposes only the de-identified
    fields)."""

    item_id: str = Field(min_length=1)
    cohort: Literal["random", "boundary"]
    selection_reason: str = Field(min_length=1)
    condition: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    sample_handle: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_answer: str
    hypothesis: str = Field(min_length=1)
    question_type: str
    is_abstention: bool
    #: True when this boundary item also belongs to the random cohort
    #: (structurally unavoidable with a 100-pair population).
    overlaps_random: bool = False


class CalibrationPlanArtifact(SchemaVersionedModel):
    """The committed sampling manifest (fixed seed, reproducible)."""

    plan_id: str = Field(min_length=1)
    algorithm: str = Field(min_length=1)
    seed: str = Field(min_length=1)
    created_at: str
    condition_runs: dict[str, str]
    random_size: int = Field(ge=1)
    boundary_size: int = Field(ge=0)
    items: tuple[CalibrationItem, ...]
    self_consistency_item_ids: tuple[str, ...]

    @model_validator(mode="after")
    def _invariants(self):
        ids = [i.item_id for i in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate item ids in the calibration plan")
        random_ids = {i.item_id for i in self.items if i.cohort == "random"}
        boundary_ids = {i.item_id for i in self.items if i.cohort == "boundary"}
        if random_ids & boundary_ids:
            raise ValueError("an item id cannot be in both cohorts")
        if len(random_ids) != self.random_size:
            raise ValueError(
                f"random cohort has {len(random_ids)} items, expected "
                f"{self.random_size}"
            )
        if len(boundary_ids) != self.boundary_size:
            raise ValueError(
                f"boundary cohort has {len(boundary_ids)} items, expected "
                f"{self.boundary_size}"
            )
        if not set(self.self_consistency_item_ids) <= random_ids:
            raise ValueError(
                "self-consistency items must come from the random cohort"
            )
        if len(set(self.self_consistency_item_ids)) != len(
            self.self_consistency_item_ids
        ):
            raise ValueError("duplicate self-consistency item ids")
        for item in self.items:
            if item.cohort == "boundary" and item.overlaps_random:
                if not any(
                    other.cohort == "random"
                    and other.condition == item.condition
                    and other.sample_handle == item.sample_handle
                    for other in self.items
                ):
                    raise ValueError(
                        f"boundary item {item.item_id} claims overlap but no "
                        "random item matches (condition, sample)"
                    )
        return self


class WorksheetRow(SchemaVersionedModel):
    """One de-identified worksheet row (annotation may be empty until
    the human fills it in)."""

    item_id: str = Field(min_length=1)
    question: str
    gold_answer: str
    response: str
    question_type: str
    is_abstention: bool
    annotation: str = ""
    note: str = ""
    annotated_at: str = ""

    @model_validator(mode="after")
    def _annotation_rules(self):
        if self.annotation and self.annotation not in ANNOTATION_LABELS:
            raise ValueError(
                f"annotation must be one of {list(ANNOTATION_LABELS)}, "
                f"got {self.annotation!r}"
            )
        return self


class AnnotationRecord(SchemaVersionedModel):
    """An imported annotation (de-identified: keyed by item id only)."""

    item_id: str = Field(min_length=1)
    annotation: str
    note: str = ""
    annotated_at: str = ""
    round: Literal["first", "self_consistency"] = "first"

    @model_validator(mode="after")
    def _label_rules(self):
        if self.annotation not in ANNOTATION_LABELS:
            raise ValueError(
                f"annotation must be one of {list(ANNOTATION_LABELS)}, "
                f"got {self.annotation!r}"
            )
        return self


# ---------------------------------------------------------------------------
# Run-output collection
# ---------------------------------------------------------------------------


def _resolve_ref(run_dir: Path, ref: str) -> Path:
    """Resolve a checksummed artifact ref inside one run directory."""
    try:
        _scheme, digest, rel = ref.split(":", 2)
    except ValueError as exc:
        raise ContractError(
            code="invalid_artifact_ref",
            message=f"artifact ref {ref!r} is not 'sha256:<hex>:<path>'",
            location="(ref)",
        ) from exc
    path = (run_dir / rel).resolve()
    if not str(path).startswith(str(run_dir.resolve())):
        raise ContractError(
            code="artifact_ref_escapes_run",
            message=f"artifact ref {ref!r} escapes the run directory",
            location="(ref)",
        )
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ContractError(
            code="artifact_checksum_mismatch",
            message=f"artifact {rel} does not match its reference digest",
            location=f"/{rel}",
        )
    return path


def collect_run_outputs(
    run_dir: str | Path, condition: str
) -> tuple[list[CalibrationCandidate], dict[str, Any]]:
    """Collect annotatable outputs from one run directory.

    Reads the persisted judge_record artifacts (they carry question,
    gold answer, hypothesis, question type and the protocol-private
    abstention flag — everything the judge saw, nothing more). Samples
    WITHOUT a judge record (e.g. a failed judge stage) are counted in
    the returned stats: a strict plan build then refuses to proceed
    (calibration needs the full dev-side population).
    """
    run_dir = Path(run_dir)
    if not (run_dir / "samples.jsonl").exists():
        raise ContractError(
            code="run_missing",
            message=f"{run_dir} has no samples.jsonl; not a qa run directory",
            location="(run dir)",
        )
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    run_id = manifest["run_id"]
    candidates: list[CalibrationCandidate] = []
    stats: dict[str, Any] = {
        "run_id": run_id,
        "planned": 0,
        "without_judge_record": [],
    }
    for line in (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        stats["planned"] += 1
        doc = json.loads(line)
        result = doc.get("result", doc)
        handle = result["sample_handle"]
        ref = result.get("artifact_refs", {}).get("judge_record")
        if ref is None:
            stats["without_judge_record"].append(handle)
            continue
        record = json.loads(_resolve_ref(run_dir, ref).read_text(encoding="utf-8"))
        request = record["request"]
        candidates.append(
            CalibrationCandidate(
                condition=condition,
                run_id=run_id,
                sample_handle=handle,
                question=request["question"],
                expected_answer=request["expected_answer"],
                hypothesis=request["hypothesis"],
                question_type=request["question_type"],
                is_abstention=bool(request.get("protocol_fields", {}).get("abstention")),
            )
        )
    return candidates, stats


# ---------------------------------------------------------------------------
# Plan building
# ---------------------------------------------------------------------------


def _stratum(candidate: CalibrationCandidate) -> str:
    return f"{candidate.question_type}|{'abs' if candidate.is_abstention else 'ans'}"


def _allocate_stratified(totals: dict[str, int], target: int) -> dict[str, int]:
    """Largest-remainder proportional allocation (deterministic)."""
    total_n = sum(totals.values())
    if target > total_n:
        raise ContractError(
            code="calibration_infeasible",
            message=f"target {target} exceeds the population {total_n}",
            location="(allocation)",
        )
    quota = {k: n * target / total_n for k, n in totals.items()}
    alloc = {k: min(n, int(q)) for k, q, n in zip(totals, quota.values(), totals.values())}
    order = sorted(totals, key=lambda k: (-(quota[k] - alloc[k]), k))
    i = 0
    while sum(alloc.values()) < target:
        k = order[i % len(order)]
        if alloc[k] < totals[k]:
            alloc[k] += 1
        i += 1
    while sum(alloc.values()) > target:
        donors = sorted((k for k in totals if alloc[k] > 0), key=lambda k: (-(alloc[k] - quota[k]), k))
        alloc[donors[0]] -= 1
    return alloc


def _boundary_bucket(candidate: CalibrationCandidate) -> str | None:
    """Deterministic boundary heuristic for one candidate (or None)."""
    if candidate.is_abstention:
        return "abstention"
    response = candidate.hypothesis.strip()
    gold = candidate.expected_answer.strip()
    if 0 < len(response) < len(gold):
        return "partial_answer"
    if len(response) < PARTIAL_MAX_CHARS and (
        candidate.question_type in MULTI_FACT_TYPES or len(gold) > len(response)
    ):
        return "partial_answer"
    lowered = response.lower()
    if any(marker in lowered for marker in REFUSAL_MARKERS):
        return "refusal_phrasing"
    if len(response) >= VERBOSE_MIN_CHARS:
        return "verbose_hedged"
    return None


def _presentation_rank(seed: str, scope: str, key: str) -> str:
    return hash_rank(seed, scope, key)


def _neutral_ids(count: int, start: int) -> list[str]:
    return [f"c{start + i:04d}" for i in range(count)]


def build_calibration_plan(
    candidates: Sequence[CalibrationCandidate],
    *,
    created_at: str,
    seed: str = CALIBRATION_SEED,
    random_size: int = RANDOM_SAMPLE_SIZE,
    boundary_size: int = BOUNDARY_SAMPLE_SIZE,
    self_consistency_size: int = SELF_CONSISTENCY_SIZE,
    strict: bool = True,
) -> CalibrationPlanArtifact:
    """Build the calibration sampling plan from two-condition outputs.

    Strict mode (production) requires: two conditions covering the SAME
    sample handles with a judge record everywhere, the full random-size
    population, all six question types and at least one abstention
    item in the random cohort, and the boundary cohort filled to size.
    ``strict=False`` relaxes the size/coverage minimums for fixtures.
    """
    conditions = sorted({c.condition for c in candidates})
    if len(conditions) != 2:
        raise ContractError(
            code="calibration_infeasible",
            message=(
                f"calibration needs exactly two run conditions, got "
                f"{conditions}"
            ),
            location="(conditions)",
        )
    by_condition: dict[str, dict[str, CalibrationCandidate]] = {
        cond: {} for cond in conditions
    }
    for candidate in candidates:
        by_condition[candidate.condition].setdefault(
            candidate.sample_handle, candidate
        )
    handles_a = set(by_condition[conditions[0]])
    handles_b = set(by_condition[conditions[1]])
    if strict and handles_a != handles_b:
        raise ContractError(
            code="calibration_infeasible",
            message=(
                "the two conditions must cover the SAME planned samples "
                f"(only-a: {sorted(handles_a - handles_b)[:5]}, only-b: "
                f"{sorted(handles_b - handles_a)[:5]})"
            ),
            location="(conditions)",
        )
    population = sorted(
        (
            by_condition[c][h]
            for c in conditions
            for h in sorted(set(handles_a) & set(handles_b))
            if h in by_condition[c]
        ),
        key=lambda c: (c.condition, c.sample_handle),
    )
    if strict and not population:
        raise ContractError(
            code="calibration_infeasible",
            message="no common (question, condition) pairs to calibrate on",
            location="(population)",
        )

    # -- random cohort: stratified hash-ranked selection ------------------
    members: dict[str, list[CalibrationCandidate]] = {}
    for candidate in population:
        members.setdefault(_stratum(candidate), []).append(candidate)
    totals = {k: len(v) for k, v in members.items()}
    target = min(random_size, len(population))
    if strict and len(population) < random_size:
        raise ContractError(
            code="calibration_infeasible",
            message=(
                f"the random cohort needs {random_size} items; the two "
                f"conditions only provide {len(population)} outputs"
            ),
            location="(population)",
        )
    alloc = _allocate_stratified(totals, target)
    random_selected: list[CalibrationCandidate] = []
    for key in sorted(members):
        ranked = sorted(
            members[key],
            key=lambda c: (hash_rank(seed, f"random|{key}", c.sample_handle + "|" + c.condition), c.sample_handle),
        )
        random_selected.extend(ranked[: alloc[key]])

    if strict:
        types_covered = {c.question_type for c in random_selected}
        missing = [t for t in QUESTION_TYPES if t not in types_covered]
        if missing:
            raise ContractError(
                code="calibration_infeasible",
                message=(
                    "the random cohort must cover all six question types; "
                    f"missing {missing}"
                ),
                location="(strata)",
            )
        if not any(c.is_abstention for c in random_selected):
            raise ContractError(
                code="calibration_infeasible",
                message="the random cohort must contain abstention items",
                location="(strata)",
            )

    # -- boundary cohort: deterministic hard-case buckets -----------------
    random_keys = {(c.condition, c.sample_handle) for c in random_selected}
    buckets: dict[str, list[CalibrationCandidate]] = {}
    for candidate in population:
        bucket = _boundary_bucket(candidate)
        if bucket is None:
            continue
        buckets.setdefault(bucket, []).append(candidate)
    for bucket in buckets:
        buckets[bucket].sort(
            key=lambda c: (
                hash_rank(seed, f"boundary|{bucket}", c.sample_handle + "|" + c.condition),
                c.sample_handle,
            )
        )
    boundary_selected: list[CalibrationCandidate] = []
    boundary_reasons: dict[int, str] = {}
    # Rotate buckets in a fixed order; prefer items NOT already in the
    # random cohort while the population allows it.
    bucket_order = ("abstention", "partial_answer", "refusal_phrasing", "verbose_hedged")
    positions = {b: 0 for b in bucket_order}
    guard = 0
    while len(boundary_selected) < boundary_size and guard < 4 * boundary_size + len(population):
        guard += 1
        progressed = False
        for bucket in bucket_order:
            if len(boundary_selected) >= boundary_size:
                break
            pool = buckets.get(bucket, [])
            pos = positions[bucket]
            while pos < len(pool):
                candidate = pool[pos]
                pos += 1
                if (candidate.condition, candidate.sample_handle) not in random_keys:
                    positions[bucket] = pos
                    boundary_selected.append(candidate)
                    boundary_reasons[id(candidate)] = bucket
                    progressed = True
                    break
            positions[bucket] = pos
        if not progressed:
            break
    # Population exhausted without random-cohort overlap: allow overlap
    # (recorded per item) — structural with a 100-pair population.
    if len(boundary_selected) < boundary_size:
        for bucket in bucket_order:
            if len(boundary_selected) >= boundary_size:
                break
            for candidate in buckets.get(bucket, []):
                if len(boundary_selected) >= boundary_size:
                    break
                if candidate in boundary_selected:
                    continue
                boundary_selected.append(candidate)
                boundary_reasons[id(candidate)] = bucket
    # Deterministic backfill: the boundary cohort keeps its fixed size
    # even when hard-case buckets run dry (reason recorded verbatim).
    if len(boundary_selected) < boundary_size:
        chosen = {(c.condition, c.sample_handle) for c in boundary_selected}
        backfill = sorted(
            (c for c in population if (c.condition, c.sample_handle) not in chosen),
            key=lambda c: (
                hash_rank(seed, "boundary|backfill", c.sample_handle + "|" + c.condition),
                c.sample_handle,
            ),
        )
        for candidate in backfill[: boundary_size - len(boundary_selected)]:
            boundary_selected.append(candidate)
            boundary_reasons[id(candidate)] = "hashrank_backfill"
    if strict and len(boundary_selected) < boundary_size:
        raise ContractError(
            code="calibration_infeasible",
            message=(
                f"the boundary cohort needs {boundary_size} items; only "
                f"{len(boundary_selected)} hard cases exist in the population"
            ),
            location="(boundary)",
        )
    if strict and not any(c.is_abstention for c in boundary_selected):
        raise ContractError(
            code="calibration_infeasible",
            message="the boundary cohort must contain at least one abstention item",
            location="(boundary)",
        )

    # -- neutral ids assigned AFTER the presentation shuffle ---------------
    random_shuffled = sorted(
        random_selected,
        key=lambda c: (
            _presentation_rank(seed, "order|random", c.sample_handle + "|" + c.condition),
            c.sample_handle,
        ),
    )
    boundary_shuffled = sorted(
        boundary_selected,
        key=lambda c: (
            _presentation_rank(seed, "order|boundary", c.sample_handle + "|" + c.condition),
            c.sample_handle,
        ),
    )
    random_ids = _neutral_ids(len(random_shuffled), 1)
    boundary_ids = _neutral_ids(len(boundary_shuffled), 1 + len(random_shuffled))

    items: list[CalibrationItem] = []
    for item_id, candidate in zip(random_ids, random_shuffled):
        items.append(
            CalibrationItem(
                item_id=item_id,
                cohort="random",
                selection_reason="stratified_hashrank",
                condition=candidate.condition,
                run_id=candidate.run_id,
                sample_handle=candidate.sample_handle,
                question=candidate.question,
                expected_answer=candidate.expected_answer,
                hypothesis=candidate.hypothesis,
                question_type=candidate.question_type,
                is_abstention=candidate.is_abstention,
            )
        )
    for item_id, candidate in zip(boundary_ids, boundary_shuffled):
        items.append(
            CalibrationItem(
                item_id=item_id,
                cohort="boundary",
                selection_reason=boundary_reasons.get(id(candidate), "boundary"),
                condition=candidate.condition,
                run_id=candidate.run_id,
                sample_handle=candidate.sample_handle,
                question=candidate.question,
                expected_answer=candidate.expected_answer,
                hypothesis=candidate.hypothesis,
                question_type=candidate.question_type,
                is_abstention=candidate.is_abstention,
                overlaps_random=(
                    (candidate.condition, candidate.sample_handle) in random_keys
                ),
            )
        )

    # -- self-consistency subset: hash-ranked members of the random cohort
    self_ids = [
        item.item_id
        for item in sorted(
            (i for i in items if i.cohort == "random"),
            key=lambda i: (
                hash_rank(seed, "selfconsistency", i.condition + "|" + i.sample_handle),
                i.item_id,
            ),
        )[: min(self_consistency_size, len(random_ids))]
    ]

    plan_id = "judge-calibration-plan-" + hashlib.sha256(
        json.dumps(
            [seed, conditions, [i.model_dump(mode="json") for i in items]],
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]

    return CalibrationPlanArtifact(
        plan_id=plan_id,
        algorithm="calibration-stratified-hashrank@1",
        seed=seed,
        created_at=created_at,
        condition_runs={
            cond: by_condition[cond][sorted(by_condition[cond])[0]].run_id
            for cond in conditions
        },
        random_size=len(random_ids),
        boundary_size=len(boundary_ids),
        items=tuple(items),
        self_consistency_item_ids=tuple(self_ids),
    )


# ---------------------------------------------------------------------------
# Worksheets (de-identified) and annotation import
# ---------------------------------------------------------------------------


def worksheet_rows(plan: CalibrationPlanArtifact, cohort: str) -> list[WorksheetRow]:
    """The de-identified worksheet rows of one cohort, in presentation
    order (the plan already fixed the shuffled order and neutral ids)."""
    if cohort not in ("random", "boundary", "self_consistency"):
        raise ValueError(f"unknown cohort {cohort!r}")
    items = [i for i in plan.items if i.cohort == cohort]
    if cohort == "self_consistency":
        wanted = set(plan.self_consistency_item_ids)
        items = [i for i in plan.items if i.item_id in wanted]
    rows = [
        WorksheetRow(
            item_id=item.item_id,
            question=item.question,
            gold_answer=item.expected_answer,
            response=item.hypothesis,
            question_type=item.question_type,
            is_abstention=item.is_abstention,
        )
        for item in items
    ]
    if cohort == "self_consistency":
        rows.sort(
            key=lambda r: (
                _presentation_rank(plan.seed, "order|selfconsistency", r.item_id),
                r.item_id,
            )
        )
    return rows


def render_worksheet_csv(rows: Sequence[WorksheetRow]) -> str:
    """Render worksheet rows to CSV (the exact column set only)."""
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(WORKSHEET_COLUMNS), lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "item_id": row.item_id,
                "question": row.question,
                "gold_answer": row.gold_answer,
                "response": row.response,
                "question_type": row.question_type,
                "is_abstention": str(row.is_abstention).lower(),
                "annotation": row.annotation,
                "note": row.note,
                "annotated_at": row.annotated_at,
            }
        )
    return buffer.getvalue()


def import_annotations(
    csv_text: str,
    *,
    expected_item_ids: Sequence[str],
    round: Literal["first", "self_consistency"] = "first",
) -> list[AnnotationRecord]:
    """Import a FILLED worksheet (annotation column completed).

    Validates the column set (extra identifying columns are rejected —
    a worksheet is only allowed to carry what the annotator may see),
    the item ids and every annotation label. Rows without an
    annotation are reported as missing instead of silently dropped.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    if tuple(reader.fieldnames or ()) != WORKSHEET_COLUMNS:
        raise ContractError(
            code="worksheet_columns_invalid",
            message=(
                f"worksheet columns must be exactly {list(WORKSHEET_COLUMNS)}, "
                f"got {list(reader.fieldnames or ())}"
            ),
            location="(worksheet)",
        )
    expected = set(expected_item_ids)
    records: list[AnnotationRecord] = []
    seen: set[str] = set()
    for row in reader:
        item_id = row["item_id"]
        if item_id not in expected:
            raise ContractError(
                code="worksheet_item_unknown",
                message=f"worksheet row {item_id!r} is not part of this cohort",
                location="(worksheet)",
            )
        if item_id in seen:
            raise ContractError(
                code="worksheet_item_duplicate",
                message=f"worksheet carries item {item_id!r} twice",
                location="(worksheet)",
            )
        seen.add(item_id)
        annotation = (row.get("annotation") or "").strip().lower()
        if not annotation:
            raise ContractError(
                code="worksheet_annotation_missing",
                message=(
                    f"item {item_id!r} has an empty annotation; every row "
                    "must be labeled (cannot_judge is the explicit unsure "
                    "label)"
                ),
                location="(worksheet)",
            )
        try:
            records.append(
                AnnotationRecord(
                    item_id=item_id,
                    annotation=annotation,
                    note=(row.get("note") or "").strip(),
                    annotated_at=(row.get("annotated_at") or "").strip(),
                    round=round,
                )
            )
        except Exception as exc:  # pydantic label validation -> ContractError
            raise ContractError(
                code="worksheet_annotation_invalid",
                message=f"item {item_id!r}: {exc}",
                location="(worksheet)",
            ) from exc
    missing = sorted(expected - seen)
    if missing:
        raise ContractError(
            code="worksheet_items_missing",
            message=(
                f"worksheet is missing {len(missing)} items: "
                f"{missing[:10]}{'…' if len(missing) > 10 else ''}"
            ),
            location="(worksheet)",
        )
    return records
