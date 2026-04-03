"""ParallelHandler - fan-out execution across branches.

Runs outgoing branches concurrently using a thread pool, cloning the
context per branch and merging results back.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..context import Context, Outcome, StageStatus
from ..graph import Graph, Node
from .base import Handler


@dataclass
class BranchResult:
    """Result from a single parallel branch."""

    edge_label: str = ""
    target_node_id: str = ""
    outcome: Optional[Outcome] = None
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_label": self.edge_label,
            "target_node_id": self.target_node_id,
            "status": self.outcome.status.value if self.outcome else "fail",
            "error": self.error,
        }


class ParallelHandler(Handler):
    """Fan-out handler for parallel nodes (shape=parallelogram).

    The handler inspects outgoing edges from the node and launches one
    branch per edge using a thread pool.  Each branch receives a *clone*
    of the current context so mutations are isolated.

    Args:
        branch_executor: callable ``(node_id, context, graph) -> Outcome``
            invoked for each branch target.  When ``None`` branches return
            SUCCESS immediately (useful for testing).
        max_workers: thread-pool size (defaults to branch count).
    """

    def __init__(
        self,
        branch_executor: Any = None,
        max_workers: Optional[int] = None,
    ) -> None:
        self.branch_executor = branch_executor
        self.max_workers = max_workers

    def execute(
        self,
        node: Node,
        context: Context,
        graph: Graph,
        logs_root: Optional[Path] = None,
    ) -> Outcome:
        outgoing = graph.outgoing_edges(node.id)
        if not outgoing:
            return Outcome(status=StageStatus.SUCCESS)

        join_policy = node.attrs.get("join_policy", "wait_all")
        workers = self.max_workers or len(outgoing)
        results: List[BranchResult] = []

        def _run_branch(edge) -> BranchResult:
            branch_ctx = context.clone()
            br = BranchResult(
                edge_label=edge.label,
                target_node_id=edge.to_node,
            )
            try:
                if self.branch_executor:
                    br.outcome = self.branch_executor(edge.to_node, branch_ctx, graph)
                else:
                    br.outcome = Outcome(status=StageStatus.SUCCESS)
            except Exception as exc:
                br.outcome = Outcome(
                    status=StageStatus.FAIL, failure_reason=str(exc)
                )
                br.error = str(exc)
            return br

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_branch, e): e for e in outgoing}

            if join_policy == "first_success":
                for future in concurrent.futures.as_completed(futures):
                    br = future.result()
                    results.append(br)
                    if br.outcome and br.outcome.status == StageStatus.SUCCESS:
                        # Cancel remaining
                        for f in futures:
                            f.cancel()
                        break
            else:
                # wait_all (default)
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())

        # Store results in context
        context.set("parallel.results", [r.to_dict() for r in results])

        # Determine aggregate status
        statuses = [r.outcome.status for r in results if r.outcome]
        if all(s == StageStatus.SUCCESS for s in statuses):
            agg_status = StageStatus.SUCCESS
        elif any(s == StageStatus.SUCCESS for s in statuses):
            agg_status = StageStatus.PARTIAL_SUCCESS
        else:
            agg_status = StageStatus.FAIL

        return Outcome(status=agg_status)
