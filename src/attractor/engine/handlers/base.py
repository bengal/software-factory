"""Handler abstract interface and registry.

Each pipeline node shape/type is mapped to a Handler subclass that knows
how to execute it.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Dict, Optional, Type

from ..context import Context, Outcome
from ..graph import Graph, Node


# ---------------------------------------------------------------------------
# Shape-to-type mapping
# ---------------------------------------------------------------------------

SHAPE_TO_TYPE: Dict[str, str] = {
    # Attractor spec canonical shapes (Section 2.8)
    "Mdiamond": "start",
    "Msquare": "exit",
    "box": "codergen",
    "hexagon": "wait.human",
    "diamond": "conditional",
    "component": "parallel",
    "tripleoctagon": "parallel.fan_in",
    "parallelogram": "tool",
    "house": "stack.manager_loop",
    # Common aliases
    "rect": "codergen",
    "rectangle": "codergen",
    "ellipse": "codergen",
}


# ---------------------------------------------------------------------------
# Handler abstract base class
# ---------------------------------------------------------------------------

class Handler(abc.ABC):
    """Abstract handler that executes a single pipeline node."""

    @abc.abstractmethod
    def execute(
        self,
        node: Node,
        context: Context,
        graph: Graph,
        logs_root: Optional[Path] = None,
    ) -> Outcome:
        """Execute the node and return an Outcome."""
        ...


# ---------------------------------------------------------------------------
# Handler Registry
# ---------------------------------------------------------------------------

class HandlerRegistry:
    """Maps node types to Handler instances."""

    def __init__(self) -> None:
        self._handlers: Dict[str, Handler] = {}

    def register(self, type_name: str, handler: Handler) -> None:
        """Register a handler for a given node type."""
        self._handlers[type_name] = handler

    def resolve(self, node: Node) -> Handler:
        """Resolve the handler for a node, using type then shape fallback."""
        # Explicit type takes precedence
        node_type = node.type or SHAPE_TO_TYPE.get(node.shape, "")
        handler = self._handlers.get(node_type)
        if handler:
            return handler
        # Try shape directly
        handler = self._handlers.get(node.shape)
        if handler:
            return handler
        raise KeyError(
            f"No handler registered for node '{node.id}' "
            f"(type={node.type!r}, shape={node.shape!r})"
        )
