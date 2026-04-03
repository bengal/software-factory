"""Event system for the coding agent session."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class EventKind(str, Enum):
    """Kind of event emitted during a session."""
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    USER_INPUT = "user_input"
    PROCESSING_END = "processing_end"
    ASSISTANT_TEXT_END = "assistant_text_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    STEERING_INJECTED = "steering_injected"
    TURN_LIMIT = "turn_limit"
    LOOP_DETECTION = "loop_detection"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class SessionEvent:
    """An event emitted during session execution."""
    kind: EventKind
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    data: dict[str, Any] = field(default_factory=dict)


# Type alias for event callbacks
EventCallback = Callable[[SessionEvent], None]


class EventEmitter:
    """Simple callback-based event emitter.

    Subscribers register callbacks that are invoked synchronously
    whenever an event is emitted.
    """

    def __init__(self) -> None:
        self._subscribers: list[EventCallback] = []
        self._kind_subscribers: dict[EventKind, list[EventCallback]] = {}

    def subscribe(
        self,
        callback: EventCallback,
        kind: EventKind | None = None,
    ) -> None:
        """Register a callback for events.

        If *kind* is specified, the callback only fires for that event kind.
        Otherwise it fires for all events.
        """
        if kind is not None:
            self._kind_subscribers.setdefault(kind, []).append(callback)
        else:
            self._subscribers.append(callback)

    def emit(self, event: SessionEvent) -> None:
        """Emit an event, invoking all matching subscribers."""
        for cb in self._subscribers:
            try:
                cb(event)
            except Exception:
                pass  # Don't let subscriber errors break the loop

        for cb in self._kind_subscribers.get(event.kind, []):
            try:
                cb(event)
            except Exception:
                pass
