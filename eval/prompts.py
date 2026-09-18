"""Fixed public prompt templates (single source of truth).

The public prompt is shared context every reader call prepends; it is
NOT part of the evidence token budget but IS part of the total call
usage and of the context precheck. Comparison requires identical public
prompt templates, so the template id (and its content hash) lives in the
reader plan and therefore in the config fingerprint.

The user template interpolates exactly three placeholders:
{question_date} {question} {evidence}. Rendering happens in one place
(this module); the offline fake readers ignore the rendered text but
the precheck still counts it, keeping fake and real paths honest.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from eval.contracts.adapter import QueryContext
    from eval.prepare.tokens import TokenCounter


@dataclass(frozen=True)
class PromptTemplate:
    """One fixed public prompt template with a stable id and hash."""

    template_id: str
    system: str
    user_template: str

    @property
    def content_sha256(self) -> str:
        payload = self.system + "\x00" + self.user_template
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def render_user(
        self, question: "QueryContext", evidence_text: str
    ) -> str:
        return self.user_template.format(
            question_date=question.question_date,
            question=question.query,
            evidence=evidence_text,
        )


#: Offline/fake template: kept for M1 configs so the precheck has a
#: deterministic stand-in; no real model consumes it.
OFFLINE_FAKE_TEMPLATE = PromptTemplate(
    template_id="offline-fake@0",
    system="[offline fake reader prompt]",
    user_template=(
        "提问时间：{question_date}\n\n问题：{question}\n\n历史证据：\n{evidence}"
    ),
)

#: M2 real-reader public prompt (Chinese, repo convention). Fixed
#: wording; any change is a new template id and a config change.
READER_PUBLIC_TEMPLATE = PromptTemplate(
    template_id="longmemeval-reader-zh@1",
    system=(
        "你是一个依据历史会话证据回答问题的助手。"
        "仅依据提供的证据作答；证据不足以回答问题时，明确说明无法回答。"
    ),
    user_template=(
        "提问时间：{question_date}\n\n"
        "问题：\n{question}\n\n"
        "历史证据：\n{evidence}\n\n"
        "请依据以上历史证据直接回答问题，不要编造证据中不存在的信息。"
    ),
)

PROMPT_TEMPLATES: dict[str, PromptTemplate] = {
    t.template_id: t for t in (OFFLINE_FAKE_TEMPLATE, READER_PUBLIC_TEMPLATE)
}


def get_prompt_template(template_id: str) -> PromptTemplate:
    """Return the fixed template or raise ValueError (config error)."""
    try:
        return PROMPT_TEMPLATES[template_id]
    except KeyError as exc:
        raise ValueError(
            f"unknown prompt template id {template_id!r}; known: "
            f"{sorted(PROMPT_TEMPLATES)} (a new template id is a config "
            "change — public prompt templates must stay fixed)"
        ) from exc


def public_prompt_tokens(
    template: PromptTemplate,
    question: "QueryContext",
    counter: "TokenCounter",
) -> int:
    """Tokens of system + user template with the question but NO evidence.

    The evidence placeholder is rendered empty, so the returned count is
    the question+public-prompt part of the context (a slight overestimate
    of the final render's fixed text; conservative for the precheck).
    """
    user = template.render_user(question, evidence_text="")
    return counter.count(template.system) + counter.count(user)
