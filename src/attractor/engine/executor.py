"""Pipeline execution engine.

Implements the full execution loop: start node traversal, handler dispatch
with retry, edge selection, goal-gate enforcement, checkpointing, and
failure routing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .conditions import evaluate_condition
from .context import Checkpoint, Context, Outcome, StageStatus
from .graph import Edge, Graph, Node
from .handlers.base import Handler, HandlerRegistry, SHAPE_TO_TYPE
from .handlers.codergen import CodergenHandler
from .handlers.conditional import ConditionalHandler
from .handlers.exit import ExitHandler
from .handlers.fan_in import FanInHandler
from .handlers.parallel import ParallelHandler
from .handlers.start import StartHandler
from .handlers.tool_handler import ToolHandler
from .handlers.wait_human import WaitForHumanHandler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BackoffConfig:
    """Exponential back-off parameters for retry."""

    initial_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    jitter: float = 0.0


@dataclass
class PipelineConfig:
    """Top-level configuration for a pipeline run."""

    logs_root: Optional[Path] = None
    checkpoint_dir: Optional[Path] = None
    registry: Optional[HandlerRegistry] = None
    backoff: BackoffConfig = field(default_factory=BackoffConfig)
    max_nodes: int = 500  # circuit-breaker
    resume_checkpoint: Optional[Checkpoint] = None
    context: Optional[Context] = None
    goal_gate_evaluator: Optional[Callable[[str, Context], bool]] = None
    on_node_start: Optional[Callable[[Node, Context], None]] = None
    on_node_end: Optional[Callable[[Node, Outcome, Context], None]] = None


# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------

def _default_registry() -> HandlerRegistry:
    reg = HandlerRegistry()
    reg.register("start", StartHandler())
    reg.register("exit", ExitHandler())
    reg.register("codergen", CodergenHandler())
    reg.register("conditional", ConditionalHandler())
    reg.register("parallel", ParallelHandler())
    reg.register("fan_in", FanInHandler())
    reg.register("tool", ToolHandler())
    reg.register("wait_human", WaitForHumanHandler())
    return reg


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def execute_with_retry(
    handler: Handler,
    node: Node,
    context: Context,
    graph: Graph,
    backoff: BackoffConfig,
    logs_root: Optional[Path] = None,
) -> Outcome:
    """Execute a handler with up to ``node.max_retries`` retry attempts."""
    max_attempts = max(1, node.max_retries + 1)
    delay = backoff.initial_delay
    last_outcome: Optional[Outcome] = None

    for attempt in range(max_attempts):
        try:
            outcome = handler.execute(node, context, graph, logs_root)
        except Exception as exc:
            outcome = Outcome(
                status=StageStatus.FAIL,
                failure_reason=f"Handler exception: {exc}",
            )

        last_outcome = outcome

        if outcome.status in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS, StageStatus.SKIPPED):
            return outcome

        if outcome.status == StageStatus.RETRY and attempt < max_attempts - 1:
            logger.info(
                "Node %s: retry %d/%d after %.1fs",
                node.id, attempt + 1, node.max_retries, delay,
            )
            time.sleep(delay)
            delay = min(delay * backoff.multiplier, backoff.max_delay)
            continue

        if outcome.status == StageStatus.FAIL and attempt < max_attempts - 1:
            logger.info(
                "Node %s: retry after FAIL %d/%d after %.1fs",
                node.id, attempt + 1, node.max_retries, delay,
            )
            time.sleep(delay)
            delay = min(delay * backoff.multiplier, backoff.max_delay)
            continue

    return last_outcome  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Edge selection
# ---------------------------------------------------------------------------

def select_edge(
    edges: List[Edge],
    outcome: Outcome,
    context: Context,
) -> Optional[Edge]:
    """Five-step edge selection algorithm.

    1. Condition match - edges whose condition evaluates to True
    2. Preferred label - match outcome.preferred_label
    3. Suggested IDs - match outcome.suggested_next_ids
    4. Weight - highest weight
    5. Lexical - alphabetical by to_node (deterministic fallback)

    Returns the single best edge, or None if no edges exist.
    """
    if not edges:
        return None

    # Step 1: filter by condition (edges with no condition always pass)
    candidates = []
    for e in edges:
        if not e.condition or evaluate_condition(e.condition, outcome, context):
            candidates.append(e)

    if not candidates:
        # If no conditions matched, fall back to unconditional edges
        candidates = [e for e in edges if not e.condition]

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Step 2: preferred label
    if outcome.preferred_label:
        for e in candidates:
            if e.label and e.label.lower() == outcome.preferred_label.lower():
                return e

    # Step 3: suggested IDs
    if outcome.suggested_next_ids:
        for nid in outcome.suggested_next_ids:
            for e in candidates:
                if e.to_node == nid:
                    return e

    # Step 4: weight (highest wins)
    max_weight = max(e.weight for e in candidates)
    by_weight = [e for e in candidates if e.weight == max_weight]
    if len(by_weight) == 1:
        return by_weight[0]

    # Step 5: lexical
    return sorted(by_weight, key=lambda e: e.to_node)[0]


def _find_fail_edge(edges: List[Edge]) -> Optional[Edge]:
    """Find an edge labelled 'fail' among outgoing edges."""
    for e in edges:
        if e.label and e.label.lower() == "fail":
            return e
    return None


# ---------------------------------------------------------------------------
# Main execution loop
# ---------------------------------------------------------------------------

def run(graph: Graph, config: Optional[PipelineConfig] = None) -> Outcome:
    """Execute the pipeline defined by *graph*.

    Args:
        graph: The parsed pipeline graph.
        config: Execution configuration; uses defaults if ``None``.

    Returns:
        The final Outcome (from the exit node, or the failing node).
    """
    cfg = config or PipelineConfig()
    registry = cfg.registry or _default_registry()
    context = cfg.context or Context()
    backoff = cfg.backoff
    logs_root = cfg.logs_root

    # Populate context from graph attrs
    for k, v in graph.attrs.items():
        context.set(f"graph.{k}", v)

    # Resume or start fresh
    visited: List[str] = []
    retry_counts: Dict[str, int] = {}

    if cfg.resume_checkpoint:
        cp = cfg.resume_checkpoint
        context.apply_updates(cp.context_data)
        visited = list(cp.visited_nodes)
        retry_counts = dict(cp.retry_counts)
        current_node = graph.nodes.get(cp.current_node_id)
        if not current_node:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=f"Checkpoint node '{cp.current_node_id}' not in graph",
            )
    else:
        current_node = graph.find_start_node()
        if not current_node:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason="No start node found in graph",
            )

    exit_node = graph.find_exit_node()
    step_count = 0

    while current_node and step_count < cfg.max_nodes:
        step_count += 1
        node = current_node
        logger.info("\033[31mExecuting node: %s (%s)\033[0m", node.id, node.label)

        if cfg.on_node_start:
            cfg.on_node_start(node, context)

        # Resolve handler
        try:
            handler = registry.resolve(node)
        except KeyError as exc:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=str(exc),
            )

        # Execute with retry
        outcome = execute_with_retry(handler, node, context, graph, backoff, logs_root)

        # Track retries exhausted: when a node with retry_target fails
        # after all attempts, set a context flag so the graph can route
        # to quarantine instead of looping forever.
        if outcome.status == StageStatus.FAIL and node.retry_target:
            count_key = f"_retry_count.{node.id}"
            count = int(context.get(count_key) or 0) + 1
            context.set(count_key, str(count))
            if count >= node.max_retries + 1:
                context.set("retries_exhausted", "true")
                logger.info("Node %s: retries exhausted (%d attempts)", node.id, count)
            else:
                context.set("retries_exhausted", "false")

        logger.info(
            "\033[31mNode %s finished: status=%s%s\033[0m",
            node.id,
            outcome.status.value,
            f" reason={outcome.failure_reason}" if outcome.failure_reason else "",
        )

        # Apply context updates from outcome
        if outcome.context_updates:
            context.apply_updates(outcome.context_updates)

        if cfg.on_node_end:
            cfg.on_node_end(node, outcome, context)

        visited.append(node.id)

        # Checkpoint
        if cfg.checkpoint_dir:
            cp = Checkpoint(
                current_node_id=node.id,
                context_data=context.snapshot(),
                visited_nodes=list(visited),
                retry_counts=dict(retry_counts),
            )
            cp.save(cfg.checkpoint_dir / "checkpoint.json")

        # Exit node reached
        if exit_node and node.id == exit_node.id:
            return outcome

        # Edge selection
        outgoing = graph.outgoing_edges(node.id)

        if outcome.status == StageStatus.FAIL:
            # Failure routing: fail edge -> retry_target -> fallback_retry_target
            fail_edge = _find_fail_edge(outgoing)
            if fail_edge:
                next_node = graph.nodes.get(fail_edge.to_node)
            elif node.retry_target:
                next_node = graph.nodes.get(node.retry_target)
            elif node.fallback_retry_target:
                next_node = graph.nodes.get(node.fallback_retry_target)
            else:
                # Pipeline fails
                return outcome
            current_node = next_node
            continue

        # Goal gate enforcement
        if node.goal_gate and cfg.goal_gate_evaluator:
            if not cfg.goal_gate_evaluator(node.goal_gate, context):
                logger.warning("Goal gate failed for node %s: %s", node.id, node.goal_gate)
                # Treat as failure
                fail_edge = _find_fail_edge(outgoing)
                if fail_edge:
                    current_node = graph.nodes.get(fail_edge.to_node)
                elif node.retry_target:
                    current_node = graph.nodes.get(node.retry_target)
                else:
                    return Outcome(
                        status=StageStatus.FAIL,
                        failure_reason=f"Goal gate '{node.goal_gate}' not met at node '{node.id}'",
                    )
                continue

        # Normal edge selection
        if not outgoing:
            # Terminal node (no exit shape) - pipeline done
            return outcome

        selected = select_edge(outgoing, outcome, context)
        if selected is None:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=f"No viable edge from node '{node.id}'",
            )

        current_node = graph.nodes.get(selected.to_node)
        if current_node is None:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=f"Edge target '{selected.to_node}' not found in graph",
            )
        edge_label = f" [{selected.label}]" if selected.label else ""
        logger.info("Edge: %s -> %s%s", node.id, selected.to_node, edge_label)

    if step_count >= cfg.max_nodes:
        return Outcome(
            status=StageStatus.FAIL,
            failure_reason=f"Circuit breaker: exceeded {cfg.max_nodes} steps",
        )

    return Outcome(
        status=StageStatus.FAIL,
        failure_reason="Execution ended unexpectedly (current_node is None)",
    )
