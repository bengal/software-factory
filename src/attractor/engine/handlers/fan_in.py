"""FanInHandler - merge results from parallel branches.

Reads ``parallel.results`` from the context, ranks branches by outcome
status, and picks the best one heuristically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..context import Context, Outcome, StageStatus
from ..graph import Graph, Node
from .base import Handler

# Ranking: lower is better
_STATUS_RANK: Dict[str, int] = {
    "success": 0,
    "partial_success": 1,
    "retry": 2,
    "skipped": 3,
    "fail": 4,
}


class FanInHandler(Handler):
    """Merge handler for fan-in nodes (shape=trapezium/invtrapezium).

    Reads ``parallel.results`` from context, ranks them by outcome status,
    and stores the best branch information for downstream use.
    """

    def execute(
        self,
        node: Node,
        context: Context,
        graph: Graph,
        logs_root: Optional[Path] = None,
    ) -> Outcome:
        raw_results: Optional[List[Dict[str, Any]]] = context.get("parallel.results")

        if not raw_results:
            return Outcome(
                status=StageStatus.SUCCESS,
                notes="No parallel results to merge.",
            )

        # Sort by status rank (best first)
        sorted_results = sorted(
            raw_results,
            key=lambda r: _STATUS_RANK.get(r.get("status", "fail"), 99),
        )

        best = sorted_results[0]
        best_status_str = best.get("status", "fail")

        # Map back to StageStatus
        try:
            best_status = StageStatus(best_status_str)
        except ValueError:
            best_status = StageStatus.FAIL

        # Store merged info in context
        context.set("fan_in.best", best)
        context.set("fan_in.all", sorted_results)

        return Outcome(
            status=best_status,
            notes=f"Best branch: {best.get('target_node_id', '?')} ({best_status_str})",
            suggested_next_ids=[best.get("target_node_id", "")]
            if best.get("target_node_id")
            else [],
        )
