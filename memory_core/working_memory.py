"""
Working Memory Module

Simulates human working memory by maintaining a bounded window of recent
interactions for prompt construction in LLM-based agents.
"""

import json
import threading
import warnings
from collections import deque
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional


class Role(str, Enum):
    """Valid message roles."""
    USER = "user"
    ASSISTANT = "assistant"


_RESERVED_KEYS = {"role", "content"}


class WorkingMemory:
    """
    Short-term conversational memory buffer.

    Thread-safe, capacity-bounded deque of chat messages.  Supports prompt
    formatting, token-budget trimming, and safe iteration.
    """

    def __init__(self, capacity: int = 10):
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer")
        self.buffer: deque = deque(maxlen=capacity)
        self._lock = threading.RLock()

    # ── Capacity ──────────────────────────────────────────────────────────────
    @property
    def capacity(self) -> int:
        """Return the current maximum capacity."""
        return self.buffer.maxlen

    # ── Add ───────────────────────────────────────────────────────────────────
    def add(self, role: Role, content: str, **metadata: Any) -> None:
        """
        Append a message to memory.

        Validation happens before the lock to reduce contention.

        Args:
            role:     Message author (Role.USER or Role.ASSISTANT).
            content:  Non-empty message text.
            **metadata: Optional extra fields (must not shadow 'role'/'content').

        Raises:
            TypeError:  If role or content have wrong types.
            ValueError: If content is blank, or metadata uses reserved keys.
        """
        if not isinstance(role, Role):
            raise TypeError(f"role must be Role enum, got {type(role).__name__}")
        if not isinstance(content, str):
            raise TypeError(f"content must be str, got {type(content).__name__}")
        if not content.strip():
            raise ValueError("content must not be empty or whitespace-only")

        reserved = _RESERVED_KEYS & metadata.keys()
        if reserved:
            raise ValueError(f"Metadata keys {reserved} are reserved")

        with self._lock:
            self.buffer.append({"role": role.value, "content": content, **metadata})

    # ── Retrieval ─────────────────────────────────────────────────────────────
    def get_all(self) -> List[Dict[str, Any]]:
        """Return safe copies of all messages."""
        with self._lock:
            return [m.copy() for m in self.buffer]

    def get_recent(self, n: int) -> List[Dict[str, Any]]:
        """
        Return the last *n* messages (or all, if fewer exist).

        Args:
            n: Number of recent messages to return (must be > 0).

        Raises:
            ValueError: If n <= 0.
        """
        if n <= 0:
            raise ValueError("n must be a positive integer")
        with self._lock:
            return [m.copy() for m in list(self.buffer)[-n:]]

    def peek(self) -> Optional[Dict[str, Any]]:
        """Return a safe copy of the most recent message, or None."""
        with self._lock:
            return dict(self.buffer[-1]) if self.buffer else None

    # ── Clear ─────────────────────────────────────────────────────────────────
    def clear(self) -> None:
        """Remove all messages from memory."""
        with self._lock:
            self.buffer.clear()

    # ── Resize ────────────────────────────────────────────────────────────────
    def resize(self, new_capacity: int) -> None:
        """
        Change the memory capacity.

        Warning: reducing capacity drops the oldest messages.

        Args:
            new_capacity: New maximum number of messages.

        Raises:
            ValueError: If new_capacity <= 0.
        """
        if new_capacity <= 0:
            raise ValueError("Capacity must be positive")

        with self._lock:
            if new_capacity < len(self.buffer):
                warnings.warn(
                    f"Dropping {len(self.buffer) - new_capacity} oldest messages",
                    UserWarning,
                    stacklevel=2,
                )
            self.buffer = deque(self.buffer, maxlen=new_capacity)

    # ── Token-budget trimming ─────────────────────────────────────────────────
    def trim_to_token_limit(
        self,
        max_tokens: int,
        chars_per_token: int = 4,
    ) -> None:
        """
        Drop oldest messages until the estimated token count is within budget.

        Uses a simple character-count heuristic (chars_per_token ≈ 4 for
        English text).  Only message content is counted; metadata is ignored.

        Args:
            max_tokens:      Maximum allowed token estimate.
            chars_per_token: Characters per token for the heuristic (default 4).

        Raises:
            ValueError: If max_tokens or chars_per_token are not positive.
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")

        with self._lock:
            while self.buffer:
                total_chars = sum(len(m["content"]) for m in self.buffer)
                if total_chars // chars_per_token <= max_tokens:
                    break
                self.buffer.popleft()

    # ── Len / Iter ────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        with self._lock:
            return len(self.buffer)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over safe copies of all messages."""
        with self._lock:
            return iter([m.copy() for m in self.buffer])

    # ── Debug ─────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        with self._lock:
            return (
                f"WorkingMemory(capacity={self.buffer.maxlen}, "
                f"size={len(self.buffer)})"
            )

    # ── Format ────────────────────────────────────────────────────────────────
    def format_for_prompt(
        self,
        mode: str = "text",
        include_metadata: bool = False,
    ) -> str:
        """
        Format buffered messages for LLM input.

        Args:
            mode:             'text' for human-readable lines, 'json' for JSON array.
            include_metadata: If True (text mode), append extra fields in brackets.

        Returns:
            Formatted string ready for injection into a prompt.

        Raises:
            ValueError: If mode is unsupported, or JSON serialisation fails.
        """
        with self._lock:
            snapshot = list(self.buffer)

        if mode == "json":
            try:
                return json.dumps(snapshot, indent=2)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Memory contains non-JSON-serializable data: {exc}"
                ) from exc

        if mode == "text":
            lines = []
            for m in snapshot:
                line = f"{m['role'].capitalize()}: {m['content']}"
                if include_metadata:
                    extras = {k: v for k, v in m.items() if k not in _RESERVED_KEYS}
                    if extras:
                        line += f" [{extras}]"
                lines.append(line)
            return "\n".join(lines)

        raise ValueError(f"Unsupported mode '{mode}'. Use 'text' or 'json'.")

    def to_prompt_messages(self) -> List[Dict[str, str]]:
        """
        Return a list of {'role': ..., 'content': ...} dicts suitable for the
        Anthropic / OpenAI messages API.
        """
        with self._lock:
            return [
                {"role": m["role"], "content": m["content"]}
                for m in self.buffer
            ]