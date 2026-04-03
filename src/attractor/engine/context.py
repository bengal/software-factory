"""Context, Checkpoint, Outcome, and StageStatus definitions.

Provides a thread-safe key-value store, serializable checkpoints, and
outcome / status types used throughout the pipeline engine.
"""

from __future__ import annotations

import enum
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# StageStatus enum
# ---------------------------------------------------------------------------

class StageStatus(enum.Enum):
    """Status codes for a pipeline stage execution."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    RETRY = "retry"
    FAIL = "fail"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Outcome
# ---------------------------------------------------------------------------

@dataclass
class Outcome:
    """Result of executing a single pipeline node."""

    status: StageStatus = StageStatus.SUCCESS
    preferred_label: str = ""
    suggested_next_ids: List[str] = field(default_factory=list)
    context_updates: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    failure_reason: str = ""


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

class Context:
    """Thread-safe key-value store for pipeline state."""

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = dict(data) if data else {}

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def get_string(self, key: str, default: str = "") -> str:
        val = self.get(key, default)
        return str(val) if val is not None else default

    def snapshot(self) -> Dict[str, Any]:
        """Return a shallow copy of the internal data."""
        with self._lock:
            return dict(self._data)

    def clone(self) -> "Context":
        """Return a new Context with a copy of this context's data."""
        with self._lock:
            return Context(dict(self._data))

    def apply_updates(self, updates: Dict[str, Any]) -> None:
        """Merge *updates* into the context."""
        with self._lock:
            self._data.update(updates)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def __repr__(self) -> str:
        with self._lock:
            return f"Context({self._data!r})"


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

@dataclass
class Checkpoint:
    """Serializable snapshot of the pipeline state at a given node."""

    current_node_id: str = ""
    context_data: Dict[str, Any] = field(default_factory=dict)
    visited_nodes: List[str] = field(default_factory=list)
    retry_counts: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Serialise checkpoint to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self._to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "Checkpoint":
        """Deserialise checkpoint from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            current_node_id=data.get("current_node_id", ""),
            context_data=data.get("context_data", {}),
            visited_nodes=data.get("visited_nodes", []),
            retry_counts=data.get("retry_counts", {}),
            metadata=data.get("metadata", {}),
        )

    def _to_dict(self) -> Dict[str, Any]:
        return {
            "current_node_id": self.current_node_id,
            "context_data": self.context_data,
            "visited_nodes": self.visited_nodes,
            "retry_counts": self.retry_counts,
            "metadata": self.metadata,
        }
