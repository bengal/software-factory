"""Lint rules for pipeline graphs.

Validates structural invariants and reports diagnostics.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Set

from .graph import Graph


# ---------------------------------------------------------------------------
# Severity & Diagnostic
# ---------------------------------------------------------------------------

class Severity(enum.Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Diagnostic:
    """A single validation finding."""

    rule: str
    severity: Severity
    message: str
    node_id: str = ""


# ---------------------------------------------------------------------------
# Built-in rules
# ---------------------------------------------------------------------------

def _rule_start_node(graph: Graph) -> List[Diagnostic]:
    """Graph must have exactly one start node."""
    starts = [n for n in graph.nodes.values()
             if n.shape == "Mdiamond" or n.type == "start" or n.id in ("start", "Start")]
    if len(starts) == 0:
        return [Diagnostic("start_node", Severity.ERROR, "No start node found (shape=Mdiamond or id=start)")]
    if len(starts) > 1:
        ids = ", ".join(n.id for n in starts)
        return [Diagnostic("start_node", Severity.ERROR, f"Multiple start nodes found: {ids}")]
    return []


def _rule_terminal_node(graph: Graph) -> List[Diagnostic]:
    """Graph must have at least one exit/terminal node."""
    exits = [n for n in graph.nodes.values()
            if n.shape == "Msquare" or n.type == "exit" or n.id in ("exit", "end")]
    if len(exits) == 0:
        return [Diagnostic("terminal_node", Severity.ERROR, "No exit node found (shape=Msquare or id=exit)")]
    return []


def _rule_edge_target_exists(graph: Graph) -> List[Diagnostic]:
    """Every edge must reference nodes that exist in the graph."""
    diags: List[Diagnostic] = []
    for e in graph.edges:
        if e.from_node not in graph.nodes:
            diags.append(Diagnostic(
                "edge_target_exists", Severity.ERROR,
                f"Edge source '{e.from_node}' does not exist in graph",
            ))
        if e.to_node not in graph.nodes:
            diags.append(Diagnostic(
                "edge_target_exists", Severity.ERROR,
                f"Edge target '{e.to_node}' does not exist in graph",
            ))
    return diags


def _rule_start_no_incoming(graph: Graph) -> List[Diagnostic]:
    """Start nodes should have no incoming edges."""
    diags: List[Diagnostic] = []
    starts = [n for n in graph.nodes.values()
             if n.shape == "Mdiamond" or n.type == "start" or n.id in ("start", "Start")]
    incoming_targets = {e.to_node for e in graph.edges}
    for s in starts:
        if s.id in incoming_targets:
            diags.append(Diagnostic(
                "start_no_incoming", Severity.WARNING,
                f"Start node '{s.id}' has incoming edges",
                node_id=s.id,
            ))
    return diags


def _rule_exit_no_outgoing(graph: Graph) -> List[Diagnostic]:
    """Exit nodes should have no outgoing edges."""
    diags: List[Diagnostic] = []
    exits = [n for n in graph.nodes.values()
            if n.shape == "Msquare" or n.type == "exit" or n.id in ("exit", "end")]
    outgoing_sources = {e.from_node for e in graph.edges}
    for x in exits:
        if x.id in outgoing_sources:
            diags.append(Diagnostic(
                "exit_no_outgoing", Severity.WARNING,
                f"Exit node '{x.id}' has outgoing edges",
                node_id=x.id,
            ))
    return diags


def _rule_reachability(graph: Graph) -> List[Diagnostic]:
    """All nodes should be reachable from the start node."""
    start = graph.find_start_node()
    if not start:
        return []  # covered by start_node rule

    reachable: Set[str] = set()
    stack = [start.id]
    while stack:
        nid = stack.pop()
        if nid in reachable:
            continue
        reachable.add(nid)
        for e in graph.outgoing_edges(nid):
            if e.to_node not in reachable:
                stack.append(e.to_node)

    diags: List[Diagnostic] = []
    for nid in graph.nodes:
        if nid not in reachable:
            diags.append(Diagnostic(
                "reachability", Severity.WARNING,
                f"Node '{nid}' is not reachable from start",
                node_id=nid,
            ))
    return diags


def _rule_condition_syntax(graph: Graph) -> List[Diagnostic]:
    """Edge conditions should parse without error."""
    from .conditions import _tokenize, _CondParser

    diags: List[Diagnostic] = []
    for e in graph.edges:
        if not e.condition:
            continue
        try:
            tokens = _tokenize(e.condition)
            parser = _CondParser(tokens)
            parser.parse()
        except Exception as exc:
            diags.append(Diagnostic(
                "condition_syntax", Severity.ERROR,
                f"Invalid condition on edge {e.from_node}->{e.to_node}: {exc}",
            ))
    return diags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_RULES = [
    _rule_start_node,
    _rule_terminal_node,
    _rule_edge_target_exists,
    _rule_start_no_incoming,
    _rule_exit_no_outgoing,
    _rule_reachability,
    _rule_condition_syntax,
]


def validate(graph: Graph) -> List[Diagnostic]:
    """Run all built-in lint rules against *graph*.

    Returns a list of Diagnostic objects (may be empty if valid).
    """
    diags: List[Diagnostic] = []
    for rule in _RULES:
        diags.extend(rule(graph))
    return diags


class ValidationError(Exception):
    """Raised when validation finds errors."""

    def __init__(self, diagnostics: List[Diagnostic]):
        self.diagnostics = diagnostics
        msgs = "; ".join(d.message for d in diagnostics if d.severity == Severity.ERROR)
        super().__init__(f"Validation failed: {msgs}")


def validate_or_raise(graph: Graph) -> None:
    """Run validation and raise ``ValidationError`` if any ERROR-level diagnostics exist."""
    diags = validate(graph)
    errors = [d for d in diags if d.severity == Severity.ERROR]
    if errors:
        raise ValidationError(errors)
