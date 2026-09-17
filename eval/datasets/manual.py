"""Manual-fixture dataset adapter (offline, polluted upstream shape).

The fixture file mirrors the LongMemEval upstream shape INCLUDING the
pollution the design document warns about: session-level 'answer'
fields, message-level 'has_answer' annotations, 'answer_'-prefixed
evidence session IDs, question-level answer_session_ids /
evidence_session_ids and '_abs' suffixed abstention question IDs.

Cleaning rules (single source for every baseline):

- sessions merge evidence + haystack, sorted by (date, official id);
- all session/message ids become stable anonymized internal ids
  (hash-derived, seed fixed, never depending on gold membership);
- messages are rebuilt from a role/content whitelist; every other
  upstream key (has_answer, tool payloads, ...) is dropped;
- the harness assigns internal message ids;
- namespaces derive from sample_plan_id + sample handle by hash and
  carry no sample semantics;
- gold answers, official ids, mappings and abstention flags live only
  in the private ScoringData view consumed by the Scorer.

Sample handles strip the '_abs' suffix: the abstention marker stays in
the private view (and in the official question id), never in adapter
input.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from eval.config import QUESTION_TYPES
from eval.contracts.adapter import Message, QueryContext, RetrievalRequest, Session
from eval.contracts.common import ContractError, parse_dataset_time
from eval.contracts.internal import ScoringData

#: Fixed anonymization seed; the algorithm never looks at gold.
ANONYMIZATION_SEED = "hippo-manual-dataset@1"
NAMESPACE_SEED = "hippo-namespace@1"

ABSTENTION_SUFFIX = "_abs"

DEFAULT_DATASET_PATH = Path(__file__).parent / "manual" / "smoke_samples.json"


def _stable_id(kind: str, *parts: str) -> str:
    seed = "|".join((ANONYMIZATION_SEED, kind, *parts))
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]


def internal_session_id(official_session_id: str) -> str:
    return "s_" + _stable_id("session", official_session_id)


def internal_msg_id(official_session_id: str, index: int) -> str:
    return "m_" + _stable_id("message", official_session_id, str(index))


def namespace_for(sample_plan_id: str, sample_handle: str) -> str:
    seed = "|".join((NAMESPACE_SEED, sample_plan_id, sample_handle))
    return "ns_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]


class _UpstreamModel(BaseModel):
    """Upstream objects are loaded permissively (pollution included)."""

    model_config = ConfigDict(extra="allow", frozen=True)


class UpstreamMessage(_UpstreamModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def _known_role(cls, v: str) -> str:
        allowed = ("user", "assistant", "system", "tool")
        if v not in allowed:
            raise ValueError(
                f"upstream message role {v!r} not in {list(allowed)}"
            )
        return v


class UpstreamSession(_UpstreamModel):
    session_id: str = Field(min_length=1)
    date: str
    messages: list[UpstreamMessage]

    @field_validator("date")
    @classmethod
    def _date_format(cls, v: str) -> str:
        parsed, mode = parse_dataset_time(v)
        if mode != "date":
            raise ValueError(
                f"upstream session date must be 'YYYY-MM-DD', got {v!r}"
            )
        return v


class UpstreamSample(_UpstreamModel):
    question_id: str = Field(min_length=1)
    question_type: str
    question: str = Field(min_length=1)
    question_date: str
    answer: str
    answer_session_ids: list[str] = Field(default_factory=list)
    evidence_session_ids: list[str] = Field(default_factory=list)
    sessions: list[UpstreamSession]

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
    def _date_format(cls, v: str) -> str:
        parse_dataset_time(v)
        return v


class CleanedSample:
    """One sample split into adapter-visible and private-scoring views."""

    def __init__(
        self,
        handle: str,
        official_question_id: str,
        question_type: str,
        is_abstention: bool,
        query_context: QueryContext,
        sessions: list[Session],
        scoring: ScoringData,
    ) -> None:
        self.handle = handle
        self.official_question_id = official_question_id
        self.question_type = question_type
        self.is_abstention = is_abstention
        self.query_context = query_context
        self.sessions = sessions
        self.scoring = scoring


def _clean_sample(upstream: UpstreamSample) -> CleanedSample:
    handle = (
        upstream.question_id[: -len(ABSTENTION_SUFFIX)]
        if upstream.question_id.endswith(ABSTENTION_SUFFIX)
        else upstream.question_id
    )
    is_abstention = upstream.question_id.endswith(ABSTENTION_SUFFIX)

    ordered = sorted(
        upstream.sessions, key=lambda s: (s.date, s.session_id)
    )
    sessions: list[Session] = []
    mapping: dict[str, str] = {}
    for upstream_session in ordered:
        sid = internal_session_id(upstream_session.session_id)
        if sid in mapping:
            raise ContractError(
                code="session_id_collision",
                message=(
                    f"anonymized session id {sid!r} collides for upstream "
                    f"{upstream_session.session_id!r}"
                ),
                location=f"/{handle}/{upstream_session.session_id}",
            )
        mapping[sid] = upstream_session.session_id
        messages = [
            Message(
                msg_id=internal_msg_id(upstream_session.session_id, i),
                role=m.role,
                content=m.content,
            )
            for i, m in enumerate(upstream_session.messages)
        ]
        sessions.append(
            Session(
                session_id=sid,
                occurred_at=upstream_session.date,
                messages=messages,
            )
        )

    official_by_internal = mapping
    internal_by_official = {v: k for k, v in mapping.items()}
    missing_gold = [
        sid
        for sid in upstream.answer_session_ids
        if sid not in internal_by_official
    ]
    if missing_gold and not is_abstention:
        raise ContractError(
            code="gold_source_unknown",
            message=(
                f"answer_session_ids {missing_gold} do not match any "
                "upstream session of this sample"
            ),
            location=f"/{handle}/answer_session_ids",
        )
    gold_source_ids = [
        internal_by_official[sid]
        for sid in upstream.answer_session_ids
        if sid in internal_by_official
    ]

    scoring = ScoringData(
        expected_answer=upstream.answer,
        gold_source_ids=gold_source_ids,
        internal_to_official_session=official_by_internal,
        is_abstention=is_abstention,
        question_type=upstream.question_type,
        official_fields={
            "question_id": upstream.question_id,
            "answer": upstream.answer,
            "answer_session_ids": list(upstream.answer_session_ids),
            "evidence_session_ids": list(upstream.evidence_session_ids),
        },
    )
    query_context = QueryContext(
        query=upstream.question, question_date=upstream.question_date
    )
    return CleanedSample(
        handle=handle,
        official_question_id=upstream.question_id,
        question_type=upstream.question_type,
        is_abstention=is_abstention,
        query_context=query_context,
        sessions=sessions,
        scoring=scoring,
    )


class ManualDataset:
    """Offline dataset over the manual fixture file."""

    def __init__(self, samples: list[CleanedSample]) -> None:
        self._samples: dict[str, CleanedSample] = {}
        for sample in samples:
            if sample.handle in self._samples:
                raise ContractError(
                    code="duplicate_sample_handle",
                    message=f"sample handle {sample.handle!r} appears twice",
                    location=f"/{sample.handle}",
                )
            self._samples[sample.handle] = sample

    # -- loading -----------------------------------------------------------

    @classmethod
    def from_json(cls, raw: str | bytes) -> "ManualDataset":
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ContractError(
                code="invalid_json",
                message=f"dataset fixture is not valid JSON: {exc.msg}",
                location=f"(line {exc.lineno}, column {exc.colno})",
            ) from exc
        if not isinstance(doc, dict) or not isinstance(doc.get("samples"), list):
            raise ContractError(
                code="invalid_type",
                message="dataset fixture root must be {'samples': [...]}",
                location="",
            )
        samples: list[CleanedSample] = []
        for i, entry in enumerate(doc["samples"]):
            try:
                upstream = UpstreamSample.model_validate(entry)
            except Exception as exc:  # noqa: BLE001 - located below
                raise ContractError(
                    code="dataset_validation",
                    message=f"samples[{i}]: {exc}",
                    location=f"/samples/{i}",
                ) from exc
            samples.append(_clean_sample(upstream))
        dataset = cls(samples)
        if not dataset.sample_handles:
            raise ContractError(
                code="empty_dataset",
                message="dataset fixture contains no samples",
                location="/samples",
            )
        return dataset

    @classmethod
    def from_file(cls, path: str | Path) -> "ManualDataset":
        path = Path(path)
        if not path.exists():
            raise ContractError(
                code="dataset_missing",
                message=f"dataset fixture not found: {path}",
                location="(path)",
            )
        return cls.from_json(path.read_text(encoding="utf-8"))

    @classmethod
    def load_default(cls) -> "ManualDataset":
        return cls.from_file(DEFAULT_DATASET_PATH)

    # -- views -------------------------------------------------------------

    @property
    def sample_handles(self) -> tuple[str, ...]:
        return tuple(self._samples)

    def require_handles(self, expected: list[str] | tuple[str, ...]) -> None:
        known = set(self._samples)
        wanted = set(expected)
        unknown = sorted(wanted - known)
        if unknown:
            raise ContractError(
                code="unknown_sample_handle",
                message=(
                    f"config lists sample handles absent from the dataset: "
                    f"{unknown}; dataset provides {sorted(known)}"
                ),
                location="/sample_ids",
            )

    def _get(self, handle: str) -> CleanedSample:
        try:
            return self._samples[handle]
        except KeyError as exc:
            raise ContractError(
                code="unknown_sample_handle",
                message=f"no sample with handle {handle!r}",
                location=f"/{handle}",
            ) from exc

    def iter_sessions(self, handle: str) -> list[Session]:
        """Cleaned sessions in time order; the runner writes ONE at a time."""
        return list(self._get(handle).sessions)

    def get_question(self, handle: str) -> QueryContext:
        return self._get(handle).query_context

    def get_scoring_data(self, handle: str) -> ScoringData:
        """Private scoring view; never handed to memory or reader paths."""
        return self._get(handle).scoring

    def namespace_for(self, sample_handle: str, sample_plan_id: str) -> str:
        """Semantic-free namespace for one sample space (hash-derived)."""
        return namespace_for(sample_plan_id, sample_handle)

    def build_retrieval_request(
        self, handle: str, evidence_token_budget: int
    ) -> RetrievalRequest:
        """Retrieval request from the dataset question and the fixed budget.

        Only dataset-provided values enter this request: machine time is
        never consulted for query context.
        """
        ctx = self.get_question(handle)
        return RetrievalRequest(
            query=ctx.query,
            question_date=ctx.question_date,
            evidence_token_budget=evidence_token_budget,
        )
