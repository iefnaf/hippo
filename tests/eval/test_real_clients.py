"""Real reader/judge clients against a fake transport (offline, no network).

Every behavioral property of the OpenAI-compatible clients is tested
through an injectable transport: request shape (credential handling,
fixed public prompt, unified query context), response parsing, error
classification (transient vs fatal), counting calibration against the
server usage, and the official judge protocol binding. The LIVE smoke
of the same components (real endpoint, real credential) lives in
tests/eval/test_live_smoke.py and skips without credentials.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from eval.config import JudgePlan, ReaderPlan
from eval.contracts.adapter import QueryContext
from eval.contracts.internal import (
    JudgeRequest,
    PreparedEvidence,
    ReaderResult,
)
from eval.judges.longmemeval import (
    OFFICIAL_PROTOCOL_ID,
    UPSTREAM_PROTOCOL_COMMIT,
    parse_official_verdict,
    render_official_prompt,
)
from eval.judges.openai_judge import OpenAIChatJudge
from eval.readers.openai_reader import OpenAIChatReader, TransportError


# ---------------------------------------------------------------------------
# Helpers: plans, fake transport, prepared evidence
# ---------------------------------------------------------------------------


def reader_plan(**overrides: Any) -> ReaderPlan:
    data: dict[str, Any] = {
        "model": "deepseek-flash",
        "model_family": "deepseek",
        "base_url": "https://api.example.com",
        "temperature": 0.0,
        "max_output_tokens": 512,
        "tokenizer_id": "estimated:chars4-v1",
        "counting_mode": "estimated",
        "api": "openai_chat",
        "api_key_env": "TEST_READER_KEY",
        "context_window_tokens": 100_000,
        "output_reserve_tokens": 512,
        "prompt_template_id": "longmemeval-reader-zh@1",
        "vendor_documented_version": "DeepSeek-V4.1-Flash",
        "vendor_documented_on": "2026-09-01",
    }
    data.update(overrides)
    return ReaderPlan.model_validate(data)


def judge_plan(**overrides: Any) -> JudgePlan:
    data: dict[str, Any] = {
        "model": "glm-5.3",
        "model_family": "glm",
        "base_url": "https://api.example.com",
        "temperature": 0.0,
        "protocol_id": OFFICIAL_PROTOCOL_ID,
        "protocol_source_commit": UPSTREAM_PROTOCOL_COMMIT,
        "api": "openai_chat",
        "api_key_env": "TEST_JUDGE_KEY",
        "vendor_documented_version": "GLM-5.3",
        "vendor_documented_on": "2026-09-01",
    }
    data.update(overrides)
    return JudgePlan.model_validate(data)


class FakeTransport:
    """Records requests; replies with queued responses or errors.

    The recorded request must NEVER contain the credential: only the
    transport sees it (as an argument), and tests assert exactly that.
    """

    def __init__(
        self,
        responses: list[Any],
        *,
        api_key: str = "secret-key-value",
    ) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.seen_keys: list[str] = []
        self.api_key = api_key

    def __call__(self, url: str, api_key: str, payload: dict, timeout: float):
        self.requests.append(
            {"url": url, "payload": payload, "timeout": timeout}
        )
        self.seen_keys.append(api_key)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def ok_response(content: str, *, model: str = "deepseek-flash-2026-08-01", prompt_tokens: int = 50) -> dict:
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 7,
            "total_tokens": prompt_tokens + 7,
        },
    }


def prepared(text: str = "[1] 原文证据：pnpm\n") -> PreparedEvidence:
    return PreparedEvidence(
        rendered_text=text,
        items=[],
        token_count=4,
        text_token_count=4,
        budget=4096,
        counting_mode="estimated",
        tokenizer_id="estimated:chars4-v1",
        dropped_raw_indices=[],
    )


QUESTION = QueryContext(query="这个项目现在用什么包管理器？", question_date="2026-09-06")


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    """Both credential env vars exist for every test in this module."""
    monkeypatch.setenv("TEST_READER_KEY", "secret-key-value")
    monkeypatch.setenv("TEST_JUDGE_KEY", "judge-secret")


# ---------------------------------------------------------------------------
# Reader client
# ---------------------------------------------------------------------------


class TestOpenAIChatReader:
    def test_request_shape_and_unified_query_context(self):
        transport = FakeTransport([ok_response("pnpm")])
        reader = OpenAIChatReader(reader_plan(), transport=transport)
        result = reader.answer(QUESTION, prepared())
        req = transport.requests[0]
        assert req["url"] == "https://api.example.com/chat/completions"
        payload = req["payload"]
        assert payload["model"] == "deepseek-flash"
        assert payload["temperature"] == 0.0
        assert payload["max_tokens"] == 512
        assert payload["stream"] is False
        system, user = payload["messages"]
        assert system["role"] == "system"
        # Unified query context: question AND dataset question_date AND the
        # exact retained evidence all ride the user message.
        assert "2026-09-06" in user["content"]
        assert QUESTION.query in user["content"]
        assert "[1] 原文证据：pnpm" in user["content"]
        assert result.hypothesis == "pnpm"
        assert result.model == "deepseek-flash-2026-08-01"

    def test_credential_only_via_env_and_never_in_requests_or_results(self, monkeypatch):
        monkeypatch.setenv("TEST_READER_KEY", "secret-key-value")
        transport = FakeTransport([ok_response("pnpm")])
        reader = OpenAIChatReader(reader_plan(), transport=transport)
        result = reader.answer(QUESTION, prepared())
        assert transport.seen_keys == ["secret-key-value"]
        blob = json.dumps(
            [req["payload"] for req in transport.requests]
        ) + result.raw_output + json.dumps(result.model_dump(mode="json"))
        assert "secret-key-value" not in blob
        assert "TEST_READER_KEY" in reader.api_key_env  # name only, by design

    def test_missing_api_key_is_fatal_with_name_only(self, monkeypatch):
        monkeypatch.delenv("TEST_READER_KEY", raising=False)
        reader = OpenAIChatReader(reader_plan(), transport=FakeTransport([]))
        from eval.readers.base import ReaderError

        with pytest.raises(ReaderError) as excinfo:
            reader.answer(QUESTION, prepared())
        assert excinfo.value.code == "missing_api_key"
        assert excinfo.value.transient is False
        assert "TEST_READER_KEY" in str(excinfo.value)
        assert "secret" not in str(excinfo.value)

    def test_counting_calibration_against_server_usage(self):
        transport = FakeTransport([ok_response("pnpm", prompt_tokens=1234)])
        reader = OpenAIChatReader(reader_plan(), transport=transport)
        result = reader.answer(QUESTION, prepared())
        assert result.calibration is not None
        assert result.calibration.counting_mode == "estimated"
        assert result.calibration.tokenizer_id == "estimated:chars4-v1"
        assert result.calibration.server_prompt_tokens == 1234
        # delta = server - local, recorded not hidden
        assert (
            result.calibration.delta_tokens
            == 1234 - result.calibration.local_prompt_tokens
        )
        assert "local count" in result.calibration.note

    def test_calibration_null_delta_without_server_usage(self):
        response = ok_response("pnpm")
        del response["usage"]
        transport = FakeTransport([response])
        reader = OpenAIChatReader(reader_plan(), transport=transport)
        result = reader.answer(QUESTION, prepared())
        assert result.calibration is not None
        assert result.calibration.server_prompt_tokens is None
        assert result.calibration.delta_tokens is None
        assert result.usage is None

    def test_observed_models_collected_for_version_record(self):
        transport = FakeTransport(
            [
                ok_response("a", model="deepseek-flash-2026-08-01"),
                ok_response("b", model="deepseek-flash-2026-09-01"),
            ]
        )
        reader = OpenAIChatReader(reader_plan(), transport=transport)
        reader.answer(QUESTION, prepared())
        reader.answer(QUESTION, prepared())
        assert reader.observed_models == {
            "deepseek-flash-2026-08-01",
            "deepseek-flash-2026-09-01",
        }

    def test_http_error_classification(self, monkeypatch):
        monkeypatch.setenv("TEST_READER_KEY", "k")
        for status, transient in ((429, True), (503, True), (500, True), (401, False), (400, False)):
            transport = FakeTransport(
                [TransportError("http", f"HTTP {status}", status=status, body="boom")]
            )
            reader = OpenAIChatReader(reader_plan(), transport=transport)
            from eval.readers.base import ReaderError

            with pytest.raises(ReaderError) as excinfo:
                reader.answer(QUESTION, prepared())
            assert excinfo.value.code == f"reader_http_{status}", status
            assert excinfo.value.transient is transient, status

    def test_timeout_and_connection_errors_transient(self):
        for err in (TransportError("timeout", "t"), TransportError("connection", "c")):
            transport = FakeTransport([err])
            reader = OpenAIChatReader(reader_plan(), transport=transport)
            from eval.readers.base import ReaderError

            with pytest.raises(ReaderError) as excinfo:
                reader.answer(QUESTION, prepared())
            assert excinfo.value.transient is True

    def test_invalid_and_empty_content_are_fatal(self):
        for bad in ({}, {"choices": []}, {"choices": [{"message": {"content": ""}}]}, {"choices": [{"message": {"content": None}}]}):
            transport = FakeTransport([bad])
            reader = OpenAIChatReader(reader_plan(), transport=transport)
            from eval.readers.base import ReaderError

            with pytest.raises(ReaderError) as excinfo:
                reader.answer(QUESTION, prepared())
            assert excinfo.value.transient is False


# ---------------------------------------------------------------------------
# Official judge protocol
# ---------------------------------------------------------------------------


class TestOfficialProtocol:
    def test_protocol_is_bound_to_upstream_commit(self):
        assert UPSTREAM_PROTOCOL_COMMIT == "9e0b455f4ef0e2ab8f2e582289761153549043fc"
        assert OFFICIAL_PROTOCOL_ID == "longmemeval-anscheck@1"

    def test_templates_match_upstream_wording(self):
        request = JudgeRequest(
            question="Q?",
            expected_answer="pnpm",
            hypothesis="pnpm",
            question_type="multi-session",
            protocol_id=OFFICIAL_PROTOCOL_ID,
            protocol_fields={},
        )
        prompt = render_official_prompt(request)
        # Verbatim upstream anchors (any template drift must be visible).
        assert prompt.startswith(
            "I will give you a question, a correct answer, and a response "
            "from a model."
        )
        assert "Question: Q?" in prompt
        assert "Correct Answer: pnpm" in prompt
        assert "Model Response: pnpm" in prompt
        assert prompt.endswith(
            "Is the model response correct? Answer yes or no only."
        )

    def test_question_type_selects_official_template(self):
        base = dict(
            question="Q",
            expected_answer="A",
            hypothesis="H",
            protocol_id=OFFICIAL_PROTOCOL_ID,
            protocol_fields={},
        )
        # temporal-reasoning carries the off-by-one clause
        prompt = render_official_prompt(
            JudgeRequest(question_type="temporal-reasoning", **base)
        )
        assert "off-by-one" in prompt
        # knowledge-update carries the updated-answer clause
        prompt = render_official_prompt(
            JudgeRequest(question_type="knowledge-update", **base)
        )
        assert "updated answer" in prompt
        # preference renders the answer as a rubric
        prompt = render_official_prompt(
            JudgeRequest(question_type="single-session-preference", **base)
        )
        assert "Rubric: A" in prompt
        assert "Correct Answer" not in prompt
        # standard three types share the standard template
        std = render_official_prompt(
            JudgeRequest(question_type="multi-session", **base)
        )
        for qtype in ("single-session-user", "single-session-assistant"):
            assert (
                render_official_prompt(JudgeRequest(question_type=qtype, **base))
                == std
            )

    def test_abstention_flag_selects_unanswerable_template(self):
        request = JudgeRequest(
            question="Q",
            expected_answer="E",
            hypothesis="H",
            question_type="multi-session",
            protocol_id=OFFICIAL_PROTOCOL_ID,
            protocol_fields={"abstention": True},
        )
        prompt = render_official_prompt(request)
        assert "unanswerable question" in prompt
        assert "Explanation: E" in prompt
        assert "Does the model correctly identify the question as unanswerable?" in prompt

    def test_official_parse_semantics(self):
        # Verbatim port of upstream 'yes' in response.lower()
        assert parse_official_verdict("yes") is True
        assert parse_official_verdict("Yes.") is True
        assert parse_official_verdict("  YES ") is True
        assert parse_official_verdict("no") is False
        assert parse_official_verdict("No.") is False
        assert parse_official_verdict("") is False
        assert parse_official_verdict("probably not") is False


class TestOpenAIChatJudge:
    def _request(self, **overrides: Any) -> JudgeRequest:
        base = dict(
            question="Q?",
            expected_answer="pnpm",
            hypothesis="pnpm",
            question_type="multi-session",
            protocol_id=OFFICIAL_PROTOCOL_ID,
            protocol_fields={"abstention": False},
        )
        base.update(overrides)
        return JudgeRequest(**base)

    def test_judge_request_uses_official_prompt_and_parse(self, monkeypatch):
        monkeypatch.setenv("TEST_JUDGE_KEY", "judge-secret")
        transport = FakeTransport(
            [ok_response("yes", model="glm-5.3-2026-09", prompt_tokens=99)]
        )
        judge = OpenAIChatJudge(judge_plan(), transport=transport)
        result = judge.evaluate(self._request())
        req = transport.requests[0]
        payload = req["payload"]
        # Upstream call shape: single user message, temperature 0, tiny max
        assert [m["role"] for m in payload["messages"]] == ["user"]
        assert payload["messages"][0]["content"].startswith(
            "I will give you a question"
        )
        assert payload["temperature"] == 0.0
        assert payload["max_tokens"] == 10
        assert payload["n"] == 1
        assert result.correct is True
        assert result.raw_output == "yes"
        assert result.model == "glm-5.3-2026-09"
        assert result.usage is not None and result.usage.input_tokens == 99
        assert judge.observed_models == {"glm-5.3-2026-09"}
        assert "judge-secret" not in json.dumps(payload)

    def test_official_no_semantics(self, monkeypatch):
        monkeypatch.setenv("TEST_JUDGE_KEY", "k")
        transport = FakeTransport([ok_response("no")])
        judge = OpenAIChatJudge(judge_plan(), transport=transport)
        assert judge.evaluate(self._request()).correct is False

    def test_empty_judge_content_is_protocol_failure_not_silent_no(self, monkeypatch):
        monkeypatch.setenv("TEST_JUDGE_KEY", "k")
        transport = FakeTransport([ok_response("  ")])
        judge = OpenAIChatJudge(judge_plan(), transport=transport)
        from eval.judges.base import JudgeProtocolError

        with pytest.raises(JudgeProtocolError) as excinfo:
            judge.evaluate(self._request())
        assert excinfo.value.code == "judge_empty_content"

    def test_real_judge_rejects_non_official_protocol(self):
        with pytest.raises(ValueError, match="official"):
            OpenAIChatJudge(
                judge_plan(protocol_id="longmemeval-yes-no@1"),
                transport=FakeTransport([]),
            )


# ---------------------------------------------------------------------------
# Config gating
# ---------------------------------------------------------------------------


class TestPlanGating:
    def test_real_reader_plan_requires_every_drift_and_context_field(self):
        with pytest.raises(Exception):
            reader_plan(api_key_env="")
        with pytest.raises(Exception):
            reader_plan(context_window_tokens=None)
        with pytest.raises(Exception):
            reader_plan(output_reserve_tokens=None)
        with pytest.raises(Exception):
            reader_plan(prompt_template_id="offline-fake@0")
        with pytest.raises(Exception):
            reader_plan(counting_mode="test")
        with pytest.raises(Exception):
            reader_plan(vendor_documented_version="")
        with pytest.raises(Exception):
            reader_plan(vendor_documented_on="")
        reader_plan()  # fully-specified real plan validates

    def test_fake_plan_cannot_declare_credentials(self):
        with pytest.raises(Exception):
            reader_plan(api="offline_fake", api_key_env="X")
        with pytest.raises(Exception):
            judge_plan(api="offline_fake", api_key_env="X")

    def test_unknown_prompt_template_or_probe_set_fails_validation(self):
        with pytest.raises(Exception, match="unknown prompt template"):
            reader_plan(api="offline_fake", prompt_template_id="nope@9")
        with pytest.raises(Exception, match="unknown probe set"):
            reader_plan(api="offline_fake", probe_set_id="nope@9")

    def test_fingerprint_includes_new_drift_fields(self):
        # The experiment fingerprint embeds the full reader/judge dumps,
        # so any drift-identifier change moves it (verified via the
        # canonical payloads those fingerprints hash).
        import json

        def payload(plan: ReaderPlan) -> str:
            return json.dumps(plan.model_dump(mode="json"), sort_keys=True)

        base = reader_plan()
        other = reader_plan(vendor_documented_version="DeepSeek-V4.2-Flash")
        third = reader_plan(api_key_env="OTHER_READER_KEY")
        assert payload(base) != payload(other)
        assert payload(base) != payload(third)
