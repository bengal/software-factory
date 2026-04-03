"""ExitHandler - no-op handler for the pipeline exit node."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..context import Context, Outcome, StageStatus
from ..graph import Graph, Node
from .base import Handler


class ExitHandler(Handler):
    """No-op handler for exit nodes (shape=doublecircle)."""

    def execute(
        self,
        node: Node,
        context: Context,
        graph: Graph,
        logs_root: Optional[Path] = None,
    ) -> Outcome:
        return Outcome(status=StageStatus.SUCCESS)
