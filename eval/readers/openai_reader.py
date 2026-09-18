"""Real reader over an OpenAI-compatible chat-completions endpoint (M2).

Stdlib HTTP only (no vendor SDK): the request/response shapes are fully
under harness control and nothing but the Authorization header ever
sees the credential. The API key is read from the environment variable
NAMED in the reader plan (``api_key_env``); the value never enters the
config snapshot, logs, artifacts or error messages.

Call shape per the design doc (统一查询上下文与提问时间): one system
message (the fixed public prompt), one user message rendering the
dataset question, its question_date and the exact PreparedEvidence the
harness retained. Question and public prompt are outside the evidence
budget (they are not evidence) but inside the reported call usage.

Counting calibration: the local prompt-token count (configured
counting mode) is compared against the server-reported
``usage.prompt_tokens`` and the delta is recorded on the ReaderResult;
budget enforcement keeps using the LOCAL count.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable

from eval.contracts.adapter import QueryContext, ResourceUsage
from eval.contracts.internal import PreparedEvidence, ReaderResult, TokenCalibration
from eval.prepare.tokens import build_token_counter
from eval.prompts import get_prompt_template
from eval.readers.base import ReaderError

#: Callable(url, api_key, payload, timeout_s) -> parsed JSON response dict.
Transport = Callable[[str, str, dict[str, Any], float], dict[str, Any]]

#: HTTP statuses treated as transient (bounded retry is allowed).
TRANSIENT_HTTP_STATUSES = frozenset({408, 409, 425, 429})


class TransportError(RuntimeError):
    """HTTP-level failure of one model call attempt."""

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        status: int | None = None,
        body: str = "",
    ) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind  # 'http' | 'timeout' | 'connection'
        self.status = status
        self.body = body


def default_transport(
    url: str, api_key: str, payload: dict[str, Any], timeout_s: float
) -> dict[str, Any]:
    """POST the JSON payload with a Bearer header; map errors to
    TransportError (status/body preserved for error classification)."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:2000]
        except Exception:  # pragma: no cover - body is best-effort
            pass
        raise TransportError(
            "http", f"HTTP {exc.code} from {url}", status=exc.code, body=body
        ) from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, TimeoutError) or "timed out" in str(reason).lower():
            raise TransportError("timeout", f"request to {url} timed out") from exc
        raise TransportError("connection", f"cannot reach {url}: {reason}") from exc
    except json.JSONDecodeError as exc:
        raise TransportError(
            "connection", f"non-JSON response from {url}: {exc.msg}"
        ) from exc


class OpenAIChatReader:
    """Reader protocol implementation against /chat/completions."""

    def __init__(
        self,
        plan: Any,
        *,
        transport: Transport = default_transport,
        clock: Callable[[], str] | None = None,
    ) -> None:
        from eval.prepare.tokens import TokenizerUnavailable

        self.plan = plan
        self.model = plan.model
        self.base_url = str(plan.base_url).rstrip("/")
        self.api_key_env = plan.api_key_env
        self.temperature = plan.temperature
        self.max_output_tokens = plan.max_output_tokens
        self.request_timeout_s = float(getattr(plan, "request_timeout_s", 120.0))
        self.template = get_prompt_template(plan.prompt_template_id)
        self._transport = transport
        self.observed_models: set[str] = set()
        self.last_response_model: str | None = None
        try:
            self._counter = build_token_counter(
                plan.counting_mode, tokenizer_id=plan.tokenizer_id
            )
        except TokenizerUnavailable:
            # Calibration needs a local count; without the pinned file the
            # run must fail loudly at the first call, not silently skip.
            raise

    # -- request/response ----------------------------------------------------

    def _messages(
        self, question: QueryContext, prepared: PreparedEvidence
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.template.system},
            {
                "role": "user",
                "content": self.template.render_user(
                    question, prepared.rendered_text
                ),
            },
        ]

    def _api_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ReaderError(
                "missing_api_key",
                f"environment variable {self.api_key_env!r} (the reader "
                "credential) is not set; the value is never stored in "
                "config or artifacts, only its variable name is",
                transient=False,
                effect="none",
            )
        return key

    def answer(
        self, question: QueryContext, prepared: PreparedEvidence
    ) -> ReaderResult:
        messages = self._messages(question, prepared)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "stream": False,
        }
        local_prompt_tokens = self._counter.count(
            self.template.system
        ) + self._counter.count(messages[1]["content"])
        try:
            response = self._transport(
                f"{self.base_url}/chat/completions",
                self._api_key(),
                payload,
                self.request_timeout_s,
            )
        except TransportError as exc:
            raise self._transport_error(exc) from exc

        content = self._extract_content(response)
        usage = self._extract_usage(response)
        response_model = response.get("model")
        if isinstance(response_model, str) and response_model:
            self.observed_models.add(response_model)
            self.last_response_model = response_model
        server_prompt = usage.input_tokens if usage is not None else None
        calibration = TokenCalibration(
            counting_mode=self.plan.counting_mode,
            tokenizer_id=self._counter.tokenizer_id,
            local_prompt_tokens=local_prompt_tokens,
            server_prompt_tokens=server_prompt,
            delta_tokens=(
                server_prompt - local_prompt_tokens
                if server_prompt is not None
                else None
            ),
        )
        return ReaderResult(
            hypothesis=content,
            raw_output=json.dumps(response, ensure_ascii=False, sort_keys=True),
            model=response_model if isinstance(response_model, str) else None,
            usage=usage,
            calibration=calibration,
        )

    # -- parsing helpers ------------------------------------------------------

    @staticmethod
    def _extract_content(response: dict[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ReaderError(
                "reader_response_invalid",
                "response carries no choices; the answer cannot be "
                "fabricated",
                transient=False,
                effect="none",
            )
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            # Some servers put the answer in a reasoning field and leave
            # content empty; an empty final answer is still a protocol
            # failure (never silently "answered").
            raise ReaderError(
                "reader_empty_content",
                "response message has no non-empty content field",
                transient=False,
                effect="none",
            )
        return content.strip()

    @staticmethod
    def _extract_usage(response: dict[str, Any]) -> ResourceUsage | None:
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return None
        return ResourceUsage(
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            llm_call_count=1,
        )

    def _transport_error(self, exc: TransportError) -> ReaderError:
        if exc.kind == "http":
            status = exc.status or 0
            transient = status in TRANSIENT_HTTP_STATUSES or status >= 500
            return ReaderError(
                f"reader_http_{status}",
                f"HTTP {status} from chat/completions: {exc.body[:400]}",
                transient=transient,
                effect="none",
            )
        return ReaderError(
            f"reader_{exc.kind}",
            exc.args[0] if exc.args else str(exc),
            transient=True,
            effect="none",
        )


def build_openai_reader(plan: Any, **kwargs: Any) -> OpenAIChatReader:
    """Build the reader an openai_chat reader plan declares."""
    return OpenAIChatReader(plan, **kwargs)
