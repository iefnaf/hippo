"""Real judge over an OpenAI-compatible endpoint (M2).

The judge renders its prompt through a PROTOCOL ADAPTER bound to a
fixed protocol id — never a free-form prompt. The official adapter
(eval.judges.longmemeval) is bound to the pinned upstream commit; the
offline fake adapter keeps its own strict yes/no protocol for M1
configs. The real client accepts ONLY the official protocol id: a real
judge run with any other protocol is a configuration error, not a
silent prompt substitution.

Credentials follow the reader convention: the environment variable
NAME lives in the judge plan (api_key_env); the value never enters the
config snapshot, artifacts or error messages.
"""

from __future__ import annotations
import json
import os
from typing import Any, Callable

from eval.contracts.adapter import ResourceUsage
from eval.contracts.internal import JudgeRequest, JudgeResult
from eval.judges.base import JudgeProtocolError
from eval.judges.longmemeval import (
    OFFICIAL_PROTOCOL_ID,
    parse_official_verdict,
    render_official_prompt,
)
from eval.readers.openai_reader import TransportError, Transport, default_transport


class OpenAIChatJudge:
    """Judge protocol implementation against /chat/completions."""

    def __init__(
        self,
        plan: Any,
        *,
        transport: Transport = default_transport,
    ) -> None:
        if plan.protocol_id != OFFICIAL_PROTOCOL_ID:
            raise ValueError(
                f"the real judge supports only protocol "
                f"{OFFICIAL_PROTOCOL_ID!r} (official LongMemEval anscheck, "
                f"bound to its upstream commit); got {plan.protocol_id!r} — "
                "a real judge with a self-made prompt is a configuration "
                "error"
            )
        self.plan = plan
        self.model = plan.model
        self.base_url = str(plan.base_url).rstrip("/")
        self.api_key_env = plan.api_key_env
        self.temperature = plan.temperature
        self.max_output_tokens = plan.max_output_tokens
        self.request_timeout_s = float(getattr(plan, "request_timeout_s", 120.0))
        self.protocol_id = plan.protocol_id
        self._render = render_official_prompt
        self._parse = parse_official_verdict
        self._transport = transport
        self.observed_models: set[str] = set()
        self.last_response_model: str | None = None

    def _api_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise JudgeProtocolError(
                "missing_api_key",
                f"environment variable {self.api_key_env!r} (the judge "
                "credential) is not set; only the variable NAME is stored "
                "in the config",
                transient=False,
                effect="none",
            )
        return key

    def evaluate(self, request: JudgeRequest) -> JudgeResult:
        prompt = self._render(request)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "n": 1,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "stream": False,
        }
        try:
            response = self._transport(
                f"{self.base_url}/chat/completions",
                self._api_key(),
                payload,
                self.request_timeout_s,
            )
        except TransportError as exc:
            raise self._transport_error(exc) from exc

        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise JudgeProtocolError(
                "judge_response_invalid",
                "response carries no choices; the verdict cannot be "
                "fabricated",
                transient=False,
                effect="none",
            )
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise JudgeProtocolError(
                "judge_empty_content",
                "judge response has no non-empty content field (official "
                "parsing is defined over non-empty responses)",
                transient=False,
                effect="none",
            )
        raw_output = content.strip()
        correct = self._parse(raw_output)
        response_model = response.get("model")
        if isinstance(response_model, str) and response_model:
            self.observed_models.add(response_model)
            self.last_response_model = response_model
        usage_doc = response.get("usage")
        usage = None
        if isinstance(usage_doc, dict):
            usage = ResourceUsage(
                input_tokens=usage_doc.get("prompt_tokens"),
                output_tokens=usage_doc.get("completion_tokens"),
                llm_call_count=1,
            )
        return JudgeResult(
            correct=correct,
            raw_output=raw_output,
            model=response_model if isinstance(response_model, str) else None,
            usage=usage,
        )

    def _transport_error(self, exc: TransportError) -> JudgeProtocolError:
        if exc.kind == "http":
            status = exc.status or 0
            from eval.readers.openai_reader import TRANSIENT_HTTP_STATUSES

            transient = status in TRANSIENT_HTTP_STATUSES or status >= 500
            return JudgeProtocolError(
                f"judge_http_{status}",
                f"HTTP {status} from chat/completions: {exc.body[:400]}",
                transient=transient,
                effect="none",
            )
        return JudgeProtocolError(
            f"judge_{exc.kind}",
            str(exc),
            transient=True,
            effect="none",
        )


def build_openai_judge(plan: Any, **kwargs: Any) -> OpenAIChatJudge:
    """Build the judge an openai_chat judge plan declares."""
    return OpenAIChatJudge(plan, **kwargs)
