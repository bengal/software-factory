"""Graph model for attractor pipelines.

Defines Node, Edge, and Graph dataclasses that represent a parsed DOT pipeline.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Node:
    """A node in the pipeline graph."""

    id: str
    label: str = ""
    shape: str = "box"
    type: str = ""
    prompt: str = ""
    max_retries: int = 0
    goal_gate: str = ""
    retry_target: str = ""
    fallback_retry_target: str = ""
    fidelity: str = ""
    thread_id: str = ""
    class_: str = ""
    timeout: Optional[float] = None
    llm_model: str = ""
    llm_provider: str = ""
    reasoning_effort: str = ""
    auto_status: str = ""
    allow_partial: bool = False
    output_key: str = ""
    attrs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.id


@dataclass
class Edge:
    """A directed edge in the pipeline graph."""

    from_node: str
    to_node: str
    label: str = ""
    condition: str = ""
    weight: float = 0.0
    fidelity: str = ""
    thread_id: str = ""
    loop_restart: bool = False


@dataclass
class Graph:
    """A directed graph representing a pipeline."""

    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)

    def outgoing_edges(self, node_id: str) -> List[Edge]:
        """Return all edges originating from *node_id*."""
        return [e for e in self.edges if e.from_node == node_id]

    def incoming_edges(self, node_id: str) -> List[Edge]:
        """Return all edges terminating at *node_id*."""
        return [e for e in self.edges if e.to_node == node_id]

    def find_start_node(self) -> Optional[Node]:
        """Find the start node (shape=Mdiamond, type=start, or id=start)."""
        for node in self.nodes.values():
            if node.shape == "Mdiamond" or node.type == "start" or node.id in ("start", "Start"):
                return node
        # Fallback: node with no incoming edges
        all_targets = {e.to_node for e in self.edges}
        for node in self.nodes.values():
            if node.id not in all_targets:
                return node
        return None

    def find_exit_node(self) -> Optional[Node]:
        """Find the exit node (shape=Msquare, type=exit, or id=exit/end)."""
        for node in self.nodes.values():
            if node.shape == "Msquare" or node.type == "exit" or node.id in ("exit", "end"):
                return node
        # Fallback: node with no outgoing edges
        all_sources = {e.from_node for e in self.edges}
        for node in self.nodes.values():
            if node.id not in all_sources:
                return node
        return None
