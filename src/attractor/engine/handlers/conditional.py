"""ConditionalHandler - no-op handler for diamond decision nodes.

The actual routing is performed by the executor's edge-selection algorithm;
this handler simply returns SUCCESS so the executor proceeds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..context import Context, Outcome, StageStatus
from ..graph import Graph, Node
from .base import Handler


class ConditionalHandler(Handler):
    """No-op handler for conditional/decision nodes (shape=diamond)."""

    def execute(
        self,
        node: Node,
        context: Context,
        graph: Graph,
        logs_root: Optional[Path] = None,
    ) -> Outcome:
        return Outcome(status=StageStatus.SUCCESS)
