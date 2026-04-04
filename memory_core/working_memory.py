
"""
Working Memory Module

Simulates human working memory by maintaining a bounded window of recent
interactions for prompt construction in LLM-based agents.

"""

from collections import deque
from typing import List, Dict, Any, Iterator, Optional
from enum import Enum
import threading
import json
import warnings


class Role(str, Enum):
    """Valid message roles."""
    USER = "user"
    ASSISTANT = "assistant"


_RESERVED_KEYS = {"role", "content"}


class WorkingMemory:
    """
    Short-term conversational memory buffer.
    """

    def __init__(self, capacity: int = 10):
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer")

        self.buffer: deque = deque(maxlen=capacity)
        self._lock = threading.RLock()

    #CAPACITY
    @property
    def capacity(self) -> int:
        """Return current capacity."""
        return self.buffer.maxlen

    #ADD
    def add(self, role: Role, content: str, **metadata: Any) -> None:
        """
        Add a message to memory.
        Validation is done before acquiring the lock to reduce contention.
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
            self.buffer.append({
                "role": role.value,
                "content": content,
                **metadata
            })

    #RETRIEVAL
    def get_all(self) -> List[Dict[str, Any]]:
        """Return all messages (safe copies)."""
        with self._lock:
            return [m.copy() for m in self.buffer]

    def get_recent(self, n: int) -> List[Dict[str, Any]]:
        """
        Return last n messages.

        If n exceeds buffer size, all messages are returned.
        """
        if n <= 0:
            raise ValueError("n must be a positive integer")

        with self._lock:
            return [m.copy() for m in list(self.buffer)[-n:]]

    def peek(self) -> Optional[Dict[str, Any]]:
        """Return most recent message (safe copy)."""
        with self._lock:
            return dict(self.buffer[-1]) if self.buffer else None

    #CLEAR
    def clear(self) -> None:
        """Clear all messages."""
        with self._lock:
            self.buffer.clear()

    #RESIZE
    def resize(self, new_capacity: int) -> None:
        """
        Resize memory capacity.

        WARNING:
        Reducing capacity drops oldest messages.
        """
        if new_capacity <= 0:
            raise ValueError("Capacity must be positive")

        with self._lock:
            if new_capacity < len(self.buffer):
                warnings.warn(
                    f"Dropping {len(self.buffer) - new_capacity} oldest messages",
                    UserWarning,
                    stacklevel=2
                )

            self.buffer = deque(self.buffer, maxlen=new_capacity)

    #LEN / ITER
    def __len__(self) -> int:
        with self._lock:
            return len(self.buffer)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over safe copies."""
        with self._lock:
            return iter([m.copy() for m in self.buffer])

    #DEBUG
    def __repr__(self) -> str:
        with self._lock:
            return f"WorkingMemory(capacity={self.buffer.maxlen}, size={len(self.buffer)})"


    #FORMAT
    def format_for_prompt(
        self,
        mode: str = "text",
        include_metadata: bool = False
    ) -> str:
        """
        Format memory for LLM input.
        """

        # snapshot first (minimize lock time)
        with self._lock:
            snapshot = list(self.buffer)

        if mode == "json":
            try:
                return json.dumps(snapshot, indent=2)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Memory contains non-JSON-serializable data: {e}")

        elif mode == "text":
            lines = []
            for m in snapshot:
                line = f"{m['role'].capitalize()}: {m['content']}"

                if include_metadata:
                    extras = {
                        k: v for k, v in m.items()
                        if k not in ("role", "content")
                    }
                    if extras:
                        line += f" [{extras}]"

                lines.append(line)

            return "\n".join(lines)

        else:
            raise ValueError(
                f"Unsupported mode '{mode}'. Use 'text' or 'json'."
            )
        
    def to_prompt_messages(self) -> List[Dict[str, str]]:
        with self._lock:
            return [
                {"role": m["role"], "content": m["content"]}
                for m in self.buffer
            ]