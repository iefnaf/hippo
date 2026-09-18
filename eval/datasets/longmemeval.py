"""LongMemEval-S (cleaned) dataset adapter: pinned download, upstream
cleaning, anonymization and dataset-level validation.

The pinned source record (issue #7) fixes the exact bytes the harness
scores against:

- repository   HuggingFace `xiaowu0162/longmemeval-cleaned` (dataset card
  MIT; the upstream code repository is MIT as well);
- revision     98d7416c24c778c2fee6e6f3006e7a073259d48f (pinned; downloads
  always resolve through this revision, never a moving branch);
- file         longmemeval_s_cleaned.json, sha256 d6f21ea9... , 277,383,467
  bytes, verified to contain 500 questions with 30 `_abs` abstention
  questions and the field list in EXPECTED_SAMPLE_FIELDS;
- verified on  2026-09-18 against the pinned revision (checksum, license
  and per-question structure check; see scripts/fetch_longmemeval.py and
  the skipped-without-data tests in tests/eval/test_longmemeval_real.py).

The data file itself never enters Git (data/ is gitignored); only this
pin record, the fetch script and the committed split manifests do.

Upstream shape (differs from the manual fixture!): one question entry
carries parallel arrays `haystack_session_ids` / `haystack_dates` /
`haystack_sessions` (each session is a list of turns, evidence turns
may carry a `has_answer` annotation) plus `answer_session_ids`. All
timestamps use the upstream format `YYYY/MM/DD (Www) HH:MM`.

Cleaning follows the manual-fixture isolation pattern (#2) and extends
it to the real shape:

- dates normalize to the contract date format `YYYY-MM-DD`. The upstream
  timestamps are timezone-naive simulation wall-clocks; the contract
  forbids fabricating a timezone, so precision reduces to the day. The
  raw string is preserved in the private scoring view, and ingestion
  order keeps the upstream (timestamp-sorted) order within a day via a
  stable sort, so no answer-relevant ordering is lost.
- session/message ids become stable hash-derived internal ids (fixed
  seed, never depending on gold membership). Identity is per haystack
  SLOT: the pinned file repeats 13 filler sessions at two timestamps
  (identical content, non-gold); both occurrences are kept — deleting
  one would alter the history the upstream protocol scores — so ids
  derive from (official id, slot index) and the mapping folds the twins
  back onto their shared official id;
- messages are rebuilt from a role/content whitelist — `has_answer` and
  any other annotation key is dropped;
- gold answers, official ids, raw timestamps and mappings live only in
  the private ScoringData view consumed by the Scorer.

Handles: unlike the manual fixture (which strips the `_abs` suffix),
handles keep the official `question_id` VERBATIM. The real file reuses
the same base id for an abstention twin of a normal question (29 base
ids carry both `xxxxxxxx` and `xxxxxxxx_abs`), so stripping the suffix
would make handles ambiguous. The abstention marker still never reaches
adapter-visible objects: namespaces are hash-derived from the handle
and sessions carry only anonymized internal ids; `is_abstention` stays
a private scoring-view flag.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from eval.config import QUESTION_TYPES
from eval.contracts.adapter import Message, QueryContext, RetrievalRequest, Session
from eval.contracts.common import ContractError, ContractModel, SchemaVersionedModel
from eval.contracts.internal import ScoringData
from eval.datasets.manual import namespace_for  # shared, dataset-agnostic

# ---------------------------------------------------------------------------
# Pinned source record (committed; the data file itself is not)
# ---------------------------------------------------------------------------

DATASET_PLAN = "longmemeval-s-cleaned@1"

SOURCE_REPO_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
UPSTREAM_CODE_URL = "https://github.com/xiaowu0162/LongMemEval"
SOURCE_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
FILE_NAME = "longmemeval_s_cleaned.json"
#: Revision-pinned resolve URL; never a moving branch.
SOURCE_FILE_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/"
    f"{SOURCE_REVISION}/{FILE_NAME}"
)
FILE_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
FILE_SIZE_BYTES = 277_383_467
#: Both the dataset card and the upstream code repository are MIT.
DATASET_LICENSE = "MIT"
#: Date the pinned revision was verified (checksum + structure + license).
VERIFIED_ON = "2026-09-18"

#: Structure verified on the pinned revision (VERIFIED_ON).
EXPECTED_TOTAL_QUESTIONS = 500
EXPECTED_ABSTENTION_QUESTIONS = 30
#: Every question entry carries exactly these keys (sorted).
EXPECTED_SAMPLE_FIELDS = (
    "answer",
    "answer_session_ids",
    "haystack_dates",
    "haystack_session_ids",
    "haystack_sessions",
    "question",
    "question_date",
    "question_id",
    "question_type",
)
#: Turn payloads may add the `has_answer` annotation upstream; anything
#: else is revision drift the pin check must surface.
EXPECTED_TURN_FIELDS = ("content", "has_answer", "role")

ABSTENTION_SUFFIX = "_abs"

#: Anonymization seed; the algorithm never looks at gold membership.
ANONYMIZATION_SEED = "hippo-longmemeval-s@1"

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LONGMEMEVAL_S_PATH = _REPO_ROOT / "data" / "longmemeval" / FILE_NAME

# ---------------------------------------------------------------------------
# Upstream timestamp normalization
# ---------------------------------------------------------------------------

_LME_DATE_RE = re.compile(r"^(\d{4})/(\d{2})/(\d{2})$")
_LME_TIMESTAMP_RE = re.compile(
    r"^(\d{4})/(\d{2})/(\d{2}) \([A-Za-z]{3}\) (\d{2}):(\d{2})$"
)


def normalize_lme_time(value: str) -> str:
    """Normalize an upstream timestamp to the contract date format.

    Accepts `YYYY/MM/DD (Www) HH:MM` (the format every question_date and
    haystack_date uses on the pinned revision) and plain `YYYY/MM/DD`.
    Returns `YYYY-MM-DD`; anything else fails — the parser is deliberately
    not a general date parser, so a changed upstream format surfaces as a
    validation error instead of silently mis-normalizing.
    """
    m = _LME_TIMESTAMP_RE.fullmatch(value) or _LME_DATE_RE.fullmatch(value)
    if m is None:
        raise ValueError(
            f"not an upstream LongMemEval timestamp {value!r}: expected "
            "'YYYY/MM/DD (Www) HH:MM' or 'YYYY/MM/DD'"
        )
    year, month, day = (int(m.group(i)) for i in (1, 2, 3))
    try:
        return _dt.date(year, month, day).isoformat()
    except ValueError as exc:
        raise ValueError(
            f"upstream timestamp {value!r} is not a real calendar date"
        ) from exc


# ---------------------------------------------------------------------------
# Upstream models (permissive: pollution included, never forwarded)
# ---------------------------------------------------------------------------


class _UpstreamModel(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)


class UpstreamTurn(_UpstreamModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def _known_role(cls, v: str) -> str:
        allowed = ("user", "assistant", "system", "tool")
        if v not in allowed:
            raise ValueError(f"upstream turn role {v!r} not in {list(allowed)}")
        return v


class UpstreamQuestion(_UpstreamModel):
    question_id: str = Field(min_length=1)
    question_type: str
    question: str = Field(min_length=1)
    question_date: str
    #: 32 counting answers arrive as JSON numbers on the pinned revision.
    answer: str | int
    haystack_session_ids: list[str]
    haystack_dates: list[str]
    haystack_sessions: list[list[UpstreamTurn]]
    answer_session_ids: list[str] = Field(default_factory=list)

    @field_validator("question_type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        if v not in QUESTION_TYPES:
            raise ValueError(
                f"question_type {v!r} not among the six upstream values "
                f"{list(QUESTION_TYPES)}"
            )
        return v

    @field_validator("question_date")
    @classmethod
    def _qdate(cls, v: str) -> str:
        normalize_lme_time(v)
        return v

    @field_validator("haystack_dates")
    @classmethod
    def _hdates(cls, v: list[str]) -> list[str]:
        for d in v:
            normalize_lme_time(d)
        return v

    @model_validator(mode="after")
    def _aligned_arrays(self) -> "UpstreamQuestion":
        n = len(self.haystack_session_ids)
        if not (n == len(self.haystack_dates) == len(self.haystack_sessions)):
            raise ValueError(
                "parallel haystack arrays misaligned: "
                f"ids={len(self.haystack_session_ids)} "
                f"dates={len(self.haystack_dates)} "
                f"sessions={len(self.haystack_sessions)}"
            )
        if n == 0:
            raise ValueError("haystack must contain at least one session")
        for i, turns in enumerate(self.haystack_sessions):
            if not turns:
                raise ValueError(f"haystack session #{i} has no turns")
        return self


# ---------------------------------------------------------------------------
# Cleaning (same isolation pattern as the manual fixture)
# ---------------------------------------------------------------------------


def _stable_id(kind: str, *parts: str) -> str:
    seed = "|".join((ANONYMIZATION_SEED, kind, *parts))
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]


def internal_session_id(official_session_id: str, slot: int) -> str:
    """Internal id of one haystack slot (slot = upstream array index).

    Slot-scoped because the pinned file repeats a few filler session ids
    at two timestamps; the twins are distinct ingestion occurrences.
    """
    return "s_" + _stable_id("session", official_session_id, str(slot))


def internal_msg_id(official_session_id: str, slot: int, index: int) -> str:
    return "m_" + _stable_id(
        "message", official_session_id, str(slot), str(index)
    )


class CleanedSample:
    """One sample split into adapter-visible and private-scoring views."""

    def __init__(
        self,
        handle: str,
        question_type: str,
        is_abstention: bool,
        query_context: QueryContext,
        sessions: list[Session],
        scoring: ScoringData,
    ) -> None:
        self.handle = handle
        self.question_type = question_type
        self.is_abstention = is_abstention
        self.query_context = query_context
        self.sessions = sessions
        self.scoring = scoring


def _clean_question(upstream: UpstreamQuestion) -> CleanedSample:
    is_abstention = upstream.question_id.endswith(ABSTENTION_SUFFIX)

    # Stable sort by normalized date preserves the upstream order (the
    # pinned S file is timestamp-sorted) within one day.
    order = sorted(
        range(len(upstream.haystack_session_ids)),
        key=lambda i: normalize_lme_time(upstream.haystack_dates[i]),
    )
    sessions: list[Session] = []
    mapping: dict[str, str] = {}
    for i in order:
        official_sid = upstream.haystack_session_ids[i]
        sid = internal_session_id(official_sid, i)
        if sid in mapping:
            raise ContractError(
                code="session_id_collision",
                message=(
                    f"anonymized session id {sid!r} collides for upstream "
                    f"{official_sid!r}"
                ),
                location=f"/{upstream.question_id}/{official_sid}",
            )
        mapping[sid] = official_sid
        messages = [
            Message(
                msg_id=internal_msg_id(official_sid, i, j),
                role=turn.role,
                content=turn.content,
            )
            for j, turn in enumerate(upstream.haystack_sessions[i])
        ]
        sessions.append(
            Session(
                session_id=sid,
                occurred_at=normalize_lme_time(upstream.haystack_dates[i]),
                messages=messages,
            )
        )

    # First occurrence in time order wins for gold mapping; duplicated
    # official ids (identical filler twins) fold onto that one internal id.
    internal_by_official: dict[str, str] = {}
    for internal, official in mapping.items():
        internal_by_official.setdefault(official, internal)
    missing_gold = [
        sid for sid in upstream.answer_session_ids if sid not in internal_by_official
    ]
    if missing_gold and not is_abstention:
        # Non-abstention samples without a resolvable gold source are data
        # validation errors, never silent exclusions (invalid_input).
        raise ContractError(
            code="gold_source_unknown",
            message=(
                f"answer_session_ids {missing_gold} do not match any "
                "haystack session of this question"
            ),
            location=f"/{upstream.question_id}/answer_session_ids",
        )
    gold_source_ids = [
        internal_by_official[sid]
        for sid in upstream.answer_session_ids
        if sid in internal_by_official
    ]

    scoring = ScoringData(
        expected_answer=str(upstream.answer),
        gold_source_ids=gold_source_ids,
        internal_to_official_session=mapping,
        is_abstention=is_abstention,
        question_type=upstream.question_type,
        official_fields={
            "question_id": upstream.question_id,
            "answer": upstream.answer,  # raw upstream value (may be int)
            "answer_session_ids": list(upstream.answer_session_ids),
            "question_date_raw": upstream.question_date,
        },
    )
    query_context = QueryContext(
        query=upstream.question,
        question_date=normalize_lme_time(upstream.question_date),
    )
    return CleanedSample(
        handle=upstream.question_id,
        question_type=upstream.question_type,
        is_abstention=is_abstention,
        query_context=query_context,
        sessions=sessions,
        scoring=scoring,
    )


# ---------------------------------------------------------------------------
# Dataset: lazy per-question cleaning over the raw pinned file
# ---------------------------------------------------------------------------


class SplitItem(ContractModel):
    """Stratification metadata for one question (no gold, no sessions)."""

    handle: str = Field(min_length=1)
    question_type: str
    is_abstention: bool

    @field_validator("question_type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        if v not in QUESTION_TYPES:
            raise ValueError(f"question_type {v!r} not in {list(QUESTION_TYPES)}")
        return v


class LongMemEvalDataset:
    """Dataset adapter over the pinned LongMemEval-S cleaned file.

    The raw JSON array is loaded once; per-question pydantic cleaning is
    lazy (an LRU of recently cleaned samples) so indexing 500 questions
    with ~48 sessions each stays cheap for split generation, while runs
    still pay full validation per touched sample.
    """

    _CACHE_SIZE = 16

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self._entries: dict[str, dict[str, Any]] = {}
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ContractError(
                    code="invalid_type",
                    message=f"dataset entry {i} is not a JSON object",
                    location=f"/{i}",
                )
            qid = entry.get("question_id")
            if not isinstance(qid, str) or not qid:
                raise ContractError(
                    code="dataset_validation",
                    message=f"entry {i} has no string question_id",
                    location=f"/{i}/question_id",
                )
            if qid in self._entries:
                raise ContractError(
                    code="duplicate_sample_handle",
                    message=(
                        f"question_id {qid!r} appears twice; handles are the "
                        "official question ids verbatim"
                    ),
                    location=f"/{qid}",
                )
            self._entries[qid] = entry
        if not self._entries:
            raise ContractError(
                code="empty_dataset",
                message="dataset file contains no questions",
                location="",
            )
        self._cache: "OrderedDict[str, CleanedSample]" = OrderedDict()

    # -- loading -----------------------------------------------------------

    @classmethod
    def from_json(cls, raw: str | bytes) -> "LongMemEvalDataset":
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ContractError(
                code="invalid_json",
                message=f"dataset file is not valid JSON: {exc.msg}",
                location=f"(line {exc.lineno}, column {exc.colno})",
            ) from exc
        if not isinstance(doc, list):
            raise ContractError(
                code="invalid_type",
                message="LongMemEval dataset root must be a JSON array of questions",
                location="",
            )
        return cls(doc)

    @classmethod
    def from_file(cls, path: str | Path) -> "LongMemEvalDataset":
        path = Path(path)
        if not path.exists():
            raise ContractError(
                code="dataset_missing",
                message=(
                    f"LongMemEval-S data file not found: {path}; run "
                    "`uv run python scripts/fetch_longmemeval.py` to fetch "
                    "the pinned revision"
                ),
                location="(path)",
            )
        return cls.from_json(path.read_text(encoding="utf-8"))

    @classmethod
    def load_default(cls) -> "LongMemEvalDataset":
        return cls.from_file(DEFAULT_LONGMEMEVAL_S_PATH)

    # -- views -------------------------------------------------------------

    @property
    def sample_handles(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def require_handles(self, expected: "list[str] | tuple[str, ...]") -> None:
        known = set(self._entries)
        wanted = set(expected)
        unknown = sorted(wanted - known)
        if unknown:
            raise ContractError(
                code="unknown_sample_handle",
                message=(
                    f"config lists sample handles absent from the dataset: "
                    f"{unknown}; dataset provides {len(known)} questions"
                ),
                location="/sample_ids",
            )

    def _clean(self, handle: str) -> CleanedSample:
        cached = self._cache.get(handle)
        if cached is not None:
            self._cache.move_to_end(handle)
            return cached
        entry = self._entries.get(handle)
        if entry is None:
            raise ContractError(
                code="unknown_sample_handle",
                message=f"no question with handle {handle!r}",
                location=f"/{handle}",
            )
        try:
            upstream = UpstreamQuestion.model_validate(entry)
            sample = _clean_question(upstream)
        except ContractError:
            raise
        except Exception as exc:  # noqa: BLE001 - located below
            raise ContractError(
                code="dataset_validation",
                message=f"{handle}: {exc}",
                location=f"/{handle}",
            ) from exc
        self._cache[handle] = sample
        if len(self._cache) > self._CACHE_SIZE:
            self._cache.popitem(last=False)
        return sample

    def iter_sessions(self, handle: str) -> list[Session]:
        """Cleaned sessions in time order; the runner writes ONE at a time."""
        return list(self._clean(handle).sessions)

    def get_question(self, handle: str) -> QueryContext:
        return self._clean(handle).query_context

    def get_scoring_data(self, handle: str) -> ScoringData:
        """Private scoring view; never handed to memory or reader paths."""
        return self._clean(handle).scoring

    def namespace_for(self, sample_handle: str, sample_plan_id: str) -> str:
        """Semantic-free namespace for one sample space (hash-derived)."""
        return namespace_for(sample_plan_id, sample_handle)

    def build_retrieval_request(
        self, handle: str, evidence_token_budget: int
    ) -> RetrievalRequest:
        ctx = self.get_question(handle)
        return RetrievalRequest(
            query=ctx.query,
            question_date=ctx.question_date,
            evidence_token_budget=evidence_token_budget,
        )

    def iter_split_items(self) -> list[SplitItem]:
        """Stratification metadata for every question, cheap and total."""
        items: list[SplitItem] = []
        for handle in self.sample_handles:
            entry = self._entries[handle]
            items.append(
                SplitItem(
                    handle=handle,
                    question_type=entry["question_type"],
                    is_abstention=handle.endswith(ABSTENTION_SUFFIX),
                )
            )
        return items


# ---------------------------------------------------------------------------
# Dataset-level validation against the pinned expectations
# ---------------------------------------------------------------------------


class DatasetSummary(SchemaVersionedModel):
    """Structural summary of a loaded LongMemEval-S file."""

    total_questions: int = Field(ge=1)
    abstention_questions: int = Field(ge=0)
    question_type_counts: dict[str, int]
    abstention_by_type: dict[str, int]
    observed_sample_fields: tuple[str, ...]
    observed_turn_fields: tuple[str, ...]
    sessions_total: int = Field(ge=0)
    sessions_per_question_min: int = Field(ge=0)
    sessions_per_question_max: int = Field(ge=0)
    turns_with_has_answer: int = Field(ge=0)


def validate_dataset(dataset: LongMemEvalDataset) -> DatasetSummary:
    """Full clean of every question; first structural error is located.

    This is the expensive total pass (fetch script, real-data tests and
    milestone gates). It observes the union of upstream field names so a
    changed revision shows up in the pin check instead of being silently
    tolerated by the permissive upstream models.
    """
    type_counts: dict[str, int] = {}
    abs_counts: dict[str, int] = {}
    sample_fields: set[str] = set()
    turn_fields: set[str] = set()
    has_answer_turns = 0
    sessions_total = 0
    sessions_min: int | None = None
    sessions_max = 0
    for handle in dataset.sample_handles:
        entry = dataset._entries[handle]
        sample_fields.update(entry.keys())
        for turns in entry.get("haystack_sessions", []):
            for turn in turns:
                turn_fields.update(turn.keys())
                if "has_answer" in turn:
                    has_answer_turns += 1
        sample = dataset._clean(handle)
        type_counts[sample.question_type] = (
            type_counts.get(sample.question_type, 0) + 1
        )
        if sample.is_abstention:
            abs_counts[sample.question_type] = (
                abs_counts.get(sample.question_type, 0) + 1
            )
        n = len(sample.sessions)
        sessions_total += n
        sessions_min = n if sessions_min is None else min(sessions_min, n)
        sessions_max = max(sessions_max, n)
    assert sessions_min is not None
    return DatasetSummary(
        total_questions=len(dataset.sample_handles),
        abstention_questions=sum(abs_counts.values()),
        question_type_counts=dict(sorted(type_counts.items())),
        abstention_by_type=dict(sorted(abs_counts.items())),
        observed_sample_fields=tuple(sorted(sample_fields)),
        observed_turn_fields=tuple(sorted(turn_fields)),
        sessions_total=sessions_total,
        sessions_per_question_min=sessions_min,
        sessions_per_question_max=sessions_max,
        turns_with_has_answer=has_answer_turns,
    )


def check_pinned_expectations(summary: DatasetSummary) -> None:
    """Fail with every pin mismatch at once (total, abstention, fields)."""
    problems: list[str] = []
    if summary.total_questions != EXPECTED_TOTAL_QUESTIONS:
        problems.append(
            f"total questions {summary.total_questions} != pinned "
            f"{EXPECTED_TOTAL_QUESTIONS}"
        )
    if summary.abstention_questions != EXPECTED_ABSTENTION_QUESTIONS:
        problems.append(
            f"abstention questions {summary.abstention_questions} != pinned "
            f"{EXPECTED_ABSTENTION_QUESTIONS}"
        )
    if tuple(summary.observed_sample_fields) != EXPECTED_SAMPLE_FIELDS:
        problems.append(
            f"sample fields {list(summary.observed_sample_fields)} != pinned "
            f"{list(EXPECTED_SAMPLE_FIELDS)}"
        )
    extra_turn = sorted(set(summary.observed_turn_fields) - set(EXPECTED_TURN_FIELDS))
    if extra_turn:
        problems.append(
            f"turn fields beyond the known whitelist {extra_turn}; the "
            "cleaning whitelist must be re-audited before accepting them"
        )
    if problems:
        raise ContractError(
            code="dataset_pin_mismatch",
            message=(
                "the loaded file does not match the pinned LongMemEval-S "
                "revision: " + "; ".join(problems)
            ),
            location="(dataset)",
        )
