"""Model stylesheet - CSS-like rules for node styling/configuration.

Parses a ``model_stylesheet`` graph attribute and applies matching rules
to nodes based on class selectors.

Syntax example::

    .llm { llm_model = "claude-3-opus"; reasoning_effort = "high" }
    .fast { timeout = 30; max_retries = 1 }
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .graph import Graph, Node
from .parser import _coerce_value


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class StyleRule:
    """A single CSS-like rule."""

    selector: str  # e.g. ".llm", "*"
    properties: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_RULE_RE = re.compile(
    r"""
    (?P<selector>[.*]?[A-Za-z_][A-Za-z0-9_-]* | \*)  # selector
    \s*\{                                              # open brace
    (?P<body>[^}]*)                                    # property declarations
    \}                                                 # close brace
    """,
    re.VERBOSE,
)

_PROP_RE = re.compile(
    r"""
    (?P<key>[A-Za-z_][A-Za-z0-9_]*)   # property name
    \s*=\s*                            # equals
    (?P<value>"[^"]*" | [^;}\s]+)      # value (quoted string or bare)
    """,
    re.VERBOSE,
)


def parse_stylesheet(source: str) -> List[StyleRule]:
    """Parse a CSS-like stylesheet string into StyleRule objects."""
    rules: List[StyleRule] = []
    for m in _RULE_RE.finditer(source):
        selector = m.group("selector").strip()
        body = m.group("body")
        props: Dict[str, Any] = {}
        for pm in _PROP_RE.finditer(body):
            key = pm.group("key")
            raw_val = pm.group("value")
            props[key] = _coerce_value(raw_val)
        rules.append(StyleRule(selector=selector, properties=props))
    return rules


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def _selector_matches(selector: str, node: Node) -> bool:
    """Check if a selector matches a node."""
    if selector == "*":
        return True
    if selector.startswith("."):
        cls_name = selector[1:]
        node_classes = {c.strip() for c in node.class_.split(",") if c.strip()}
        return cls_name in node_classes
    # Type selector
    return node.type == selector or node.shape == selector


_NODE_ATTRS = {
    "label", "shape", "type", "prompt", "max_retries", "goal_gate",
    "retry_target", "fallback_retry_target", "fidelity", "thread_id",
    "class_", "timeout", "llm_model", "llm_provider", "reasoning_effort",
    "auto_status", "allow_partial",
}


def apply_stylesheet(graph: Graph) -> None:
    """Parse and apply the ``model_stylesheet`` graph attribute to all nodes.

    Mutates nodes in place.  Rules are applied in order; later rules
    override earlier ones.
    """
    raw = graph.attrs.get("model_stylesheet", "")
    if not raw:
        return

    rules = parse_stylesheet(str(raw))

    for node in graph.nodes.values():
        for rule in rules:
            if not _selector_matches(rule.selector, node):
                continue
            for key, val in rule.properties.items():
                attr_name = "class_" if key == "class" else key
                if attr_name in _NODE_ATTRS:
                    setattr(node, attr_name, val)
                else:
                    node.attrs[key] = val
