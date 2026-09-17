"""Token counting for reader input preparation.

M1 is fully offline: the only counter is a deterministic test tokenizer
counting one token per Unicode character (tokenizer_id 'test:char-v1',
counting_mode 'test'). Real tokenizer integration (DeepSeek offline
tokenizer, exact/estimated modes) lands in M2; the counting-mode split
keeps test numbers from ever being compared with real model usage.
"""

from __future__ import annotations

from typing import Protocol

TEST_TOKENIZER_ID = "test:char-v1"


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
