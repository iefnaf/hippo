"""Official LongMemEval answer-check protocol adapter (M2).

The prompt templates and the yes/no parsing semantics are copied
VERBATIM from the upstream evaluation script
src/evaluation/evaluate_qa.py (get_anscheck_prompt and the
``'yes' in response.lower()`` label rule), bound to the pinned commit
below. The harness does not claim comparability with the paper's
GPT-4o-validated scores: our judge is a different model family by
design (anti-self-preference), and that deviation is recorded with
every run (see eval.models and the design doc, Judge 校准).

Template selection follows upstream exactly:

- abstention questions (``protocol_fields['abstention']``) use the
  unanswerable-question template;
- single-session-user / single-session-assistant / multi-session use
  the standard template;
- temporal-reasoning, knowledge-update and single-session-preference
  have their own templates (the preference template renders the answer
  field as the rubric, as upstream does).

Parsing semantics (official): the verdict is yes iff the lowercased,
stripped response CONTAINS 'yes'. A bare 'no' — or any text without
'yes' — is a no. This differs from the offline fake judge's strict
equality parse; the fake is its own protocol, this module is the
official one. The parse never raises: upstream has no unparseable
case. An EMPTY content is still surfaced as a protocol failure by the
client (an empty reply is an endpoint anomaly, not a 'no').
"""

from __future__ import annotations

from eval.contracts.internal import JudgeRequest

OFFICIAL_PROTOCOL_ID = "longmemeval-anscheck@1"

#: Upstream implementation commit this protocol is bound to.
UPSTREAM_PROTOCOL_COMMIT = "9e0b455f4ef0e2ab8f2e582289761153549043fc"
UPSTREAM_PROTOCOL_URL = (
    "https://github.com/xiaowu0162/LongMemEval/blob/"
    f"{UPSTREAM_PROTOCOL_COMMIT}/src/evaluation/evaluate_qa.py"
)

_STANDARD_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "multi-session",
)

_TEMPLATE_STANDARD = (
    "I will give you a question, a correct answer, and a response from a "
    "model. Please answer yes if the response contains the correct "
    "answer. Otherwise, answer no. If the response is equivalent to the "
    "correct answer or contains all the intermediate steps to get the "
    "correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer "
    "no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: "
    "{}\n\nIs the model response correct? Answer yes or no only."
)

_TEMPLATE_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a "
    "model. Please answer yes if the response contains the correct "
    "answer. Otherwise, answer no. If the response is equivalent to the "
    "correct answer or contains all the intermediate steps to get the "
    "correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer "
    "no. In addition, do not penalize off-by-one errors for the number "
    "of days. If the question asks for the number of days/weeks/months, "
    "etc., and the model makes off-by-one errors (e.g., predicting 19 "
    "days when the answer is 18), the model's response is still correct. "
    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs "
    "the model response correct? Answer yes or no only."
)

_TEMPLATE_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a "
    "model. Please answer yes if the response contains the correct "
    "answer. Otherwise, answer no. If the response contains some "
    "previous information along with an updated answer, the response "
    "should be considered as correct as long as the updated answer is "
    "the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel "
    "Response: {}\n\nIs the model response correct? Answer yes or no only."
)

_TEMPLATE_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized "
    "response, and a response from a model. Please answer yes if the "
    "response satisfies the desired response. Otherwise, answer no. The "
    "model does not need to reflect all the points in the rubric. The "
    "response is correct as long as it recalls and utilizes the user's "
    "personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel "
    "Response: {}\n\nIs the model response correct? Answer yes or no only."
)

_TEMPLATE_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a "
    "response from a model. Please answer yes if the model correctly "
    "identifies the question as unanswerable. The model could say that "
    "the information is incomplete, or some other information is given "
    "but the asked information is not.\n\nQuestion: {}\n\nExplanation: "
    "{}\n\nModel Response: {}\n\nDoes the model correctly identify the "
    "question as unanswerable? Answer yes or no only."
)


def render_official_prompt(request: JudgeRequest) -> str:
    """Render the official protocol prompt for one judge request."""
    abstention = bool(request.protocol_fields.get("abstention", False))
    question = request.question
    answer = request.expected_answer
    response = request.hypothesis
    if abstention:
        template = _TEMPLATE_ABSTENTION
    elif request.question_type in _STANDARD_TYPES:
        template = _TEMPLATE_STANDARD
    elif request.question_type == "temporal-reasoning":
        template = _TEMPLATE_TEMPORAL
    elif request.question_type == "knowledge-update":
        template = _TEMPLATE_KNOWLEDGE_UPDATE
    elif request.question_type == "single-session-preference":
        template = _TEMPLATE_PREFERENCE
    else:
        raise ValueError(
            f"question_type {request.question_type!r} has no official "
            "anscheck template (upstream would raise NotImplementedError)"
        )
    return template.format(question, answer, response)


def parse_official_verdict(raw_output: str) -> bool:
    """Official label rule: yes iff the stripped, lowercased response
    contains 'yes' (verbatim port of ``'yes' in response.lower()``)."""
    return "yes" in raw_output.strip().lower()
