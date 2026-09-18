"""Token counting for reader input preparation and context prechecks.

Three counting modes exist and are NEVER mixed in one comparison
(docs/design/eval-harness.md, Token 预算与模型配置):

- ``test``       the deterministic M1 fake counter (one token per Unicode
                 character, id ``test:char-v1``); validates harness
                 behavior offline and is never compared with real model
                 usage;
- ``exact``      the official DeepSeek offline tokenizer loaded from a
                 revision-pinned ``tokenizer.json`` (id
                 ``deepseek-offline:v1``); budget enforcement uses this
                 count and server ``usage`` values only CALIBRATE it;
- ``estimated``  a documented heuristic (ceil(chars/4), id
                 ``estimated:chars4-v1``) for configs that cannot pin the
                 real tokenizer; results are marked estimated and are
                 never compared with exact-mode runs (the counting mode
                 is a key comparability field).

The pin record below fixes repo/revision/sha256 of the tokenizer file.
The file itself never enters Git (data/ is ignored); fetch it with
scripts/fetch_deepseek_tokenizer.py. Loading verifies the sha256 so a
silently replaced file fails loudly instead of miscounting.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Protocol

TEST_TOKENIZER_ID = "test:char-v1"
#: Exact-mode official offline tokenizer (DeepSeek published tokenizer;
#: matches the API-side token accounting documented by the vendor).
DEEPSEEK_TOKENIZER_ID = "deepseek-offline:v1"
#: Estimated-mode heuristic counter.
ESTIMATED_TOKENIZER_ID = "estimated:chars4-v1"

# ---------------------------------------------------------------------------
# Pin record for the DeepSeek offline tokenizer (committed; file is not)
# ---------------------------------------------------------------------------

DEEPSEEK_TOKENIZER_REPO = "deepseek-ai/DeepSeek-V3"
#: Pinned repository revision; downloads always resolve through it.
DEEPSEEK_TOKENIZER_REVISION = "e815299b0bcbac849fa540c768ef21845365c9eb"
DEEPSEEK_TOKENIZER_FILE = "tokenizer.json"
DEEPSEEK_TOKENIZER_SHA256 = (
    "621ac2e32d0dba658404412318818aaa8ce8cda492e59830109d8da6b517fb41"
)
DEEPSEEK_TOKENIZER_SIZE_BYTES = 7_847_652
#: Revision-pinned resolve URL; never a moving branch.
DEEPSEEK_TOKENIZER_URL = (
    f"https://huggingface.co/{DEEPSEEK_TOKENIZER_REPO}/resolve/"
    f"{DEEPSEEK_TOKENIZER_REVISION}/{DEEPSEEK_TOKENIZER_FILE}"
)
#: Model card license (checked on the pinned revision).
DEEPSEEK_TOKENIZER_LICENSE = "MIT"
#: Date the pinned revision was verified (checksum + license).
DEEPSEEK_TOKENIZER_VERIFIED_ON = "2026-09-18"

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEEPSEEK_TOKENIZER_PATH = (
    _REPO_ROOT / "data" / "tokenizers" / "deepseek-v3" / "tokenizer.json"
)


class TokenizerUnavailable(RuntimeError):
    """The pinned tokenizer file is missing or does not match its pin."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class TokenCounter(Protocol):
    """Minimal counter protocol: monotonically growing with text length."""

    tokenizer_id: str

    def count(self, text: str) -> int: ...


class TestCharTokenizer:
    """Deterministic offline counter: one token per Unicode character."""

    tokenizer_id: str = TEST_TOKENIZER_ID

    def count(self, text: str) -> int:
        return len(text)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"TestCharTokenizer({self.tokenizer_id!r})"


class DeepSeekOfflineTokenizer:
    """Exact counter over the pinned official DeepSeek tokenizer file.

    The file is sha256-verified on load; a missing file is an explicit
    ``TokenizerUnavailable`` (run scripts/fetch_deepseek_tokenizer.py),
    never a silent fallback to another counting mode.
    """

    tokenizer_id: str = DEEPSEEK_TOKENIZER_ID

    def __init__(self, path: str | Path | None = None) -> None:
        path = Path(path) if path is not None else DEFAULT_DEEPSEEK_TOKENIZER_PATH
        if not path.exists():
            raise TokenizerUnavailable(
                f"pinned DeepSeek tokenizer file not found: {path}; "
                "run `uv run python scripts/fetch_deepseek_tokenizer.py` "
                "to fetch the pinned revision (counting_mode='exact' "
                "requires it)"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != DEEPSEEK_TOKENIZER_SHA256:
            raise TokenizerUnavailable(
                f"tokenizer file {path} has sha256 {digest} but the pin "
                f"expects {DEEPSEEK_TOKENIZER_SHA256}; re-fetch with "
                "scripts/fetch_deepseek_tokenizer.py (a silently replaced "
                "file must fail loudly, not miscount)"
            )
        try:
            from tokenizers import Tokenizer  # deferred optional import
        except ImportError as exc:  # pragma: no cover - dependency declared
            raise TokenizerUnavailable(
                "the `tokenizers` package is required for exact counting; "
                "run `uv sync`"
            ) from exc
        self.path = path
        self._tokenizer = Tokenizer.from_file(str(path))

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"DeepSeekOfflineTokenizer({self.tokenizer_id!r})"


class HeuristicEstimatingTokenizer:
    """Estimated counter: ceil(Unicode characters / 4).

    A deliberately coarse, deterministic approximation for configs that
    cannot pin the real tokenizer. Monotone (binary search in
    prepare_evidence stays sound). Runs produced under this counter are
    marked counting_mode='estimated' and never compared with exact-mode
    runs (key comparability field).
    """

    tokenizer_id: str = ESTIMATED_TOKENIZER_ID

    def count(self, text: str) -> int:
        return math.ceil(len(text) / 4)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"HeuristicEstimatingTokenizer({self.tokenizer_id!r})"


def build_token_counter(
    counting_mode: str,
    *,
    tokenizer_id: str | None = None,
    path: str | Path | None = None,
) -> TokenCounter:
    """Build the counter a config's counting mode declares.

    The returned counter's tokenizer_id must match the configured
    tokenizer_id; a mismatch is a configuration error (the config would
    otherwise claim one tokenizer while counting with another).
    """
    if counting_mode == "test":
        return TestCharTokenizer()
    if counting_mode == "exact":
        counter: TokenCounter = DeepSeekOfflineTokenizer(path)
    elif counting_mode == "estimated":
        counter = HeuristicEstimatingTokenizer()
    else:
        raise ValueError(f"unknown counting mode {counting_mode!r}")
    if tokenizer_id is not None and counter.tokenizer_id != tokenizer_id:
        raise ValueError(
            f"counting_mode={counting_mode!r} provides tokenizer "
            f"{counter.tokenizer_id!r} but the config pins "
            f"{tokenizer_id!r}; fix the config so the counting mode and "
            "tokenizer id agree"
        )
    return counter
