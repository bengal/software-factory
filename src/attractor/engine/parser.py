"""Hand-written recursive descent parser for the constrained DOT subset.

Grammar (simplified):
    graph       := 'digraph' ID? '{' stmt_list '}'
    stmt_list   := (stmt ';'?)*
    stmt        := attr_stmt | edge_stmt | node_stmt | subgraph
    attr_stmt   := ('graph' | 'node' | 'edge') attr_list
    edge_stmt   := node_id ('->' node_id)+ attr_list?
    node_stmt   := node_id attr_list?
    attr_list   := '[' (attr (',' | ';')?)* ']'
    attr        := ID '=' value
    subgraph    := 'subgraph' ID? '{' stmt_list '}'
    value       := STRING | NUMBER | BOOL | DURATION | BARE_ID
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .graph import Edge, Graph, Node

# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

_KEYWORDS = {"digraph", "graph", "node", "edge", "subgraph", "strict"}


@dataclass
class Token:
    kind: str  # ID, STRING, NUMBER, ARROW, LBRACE, RBRACE, LBRACK, RBRACK, EQ, SEMI, COMMA, EOF
    value: str
    line: int = 0
    col: int = 0


_PATTERNS: List[Tuple[str, str]] = [
    ("COMMENT_LINE", r"//[^\n]*"),
    ("COMMENT_BLOCK", r"/\*[\s\S]*?\*/"),
    ("COMMENT_HASH", r"#[^\n]*"),
    ("WS", r"[ \t\r\n]+"),
    ("ARROW", r"->"),
    ("STRING", r'"(?:[^"\\]|\\.)*"'),
    ("NUMBER", r"-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?"),
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("LBRACK", r"\["),
    ("RBRACK", r"\]"),
    ("EQ", r"="),
    ("SEMI", r";"),
    ("COMMA", r","),
    ("ID", r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"),
]

_TOKEN_RE = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in _PATTERNS))


def _tokenize(source: str) -> List[Token]:
    tokens: List[Token] = []
    line = 1
    col = 1
    for m in _TOKEN_RE.finditer(source):
        kind = m.lastgroup
        value = m.group()
        if kind in ("WS", "COMMENT_LINE", "COMMENT_BLOCK", "COMMENT_HASH"):
            line += value.count("\n")
            if "\n" in value:
                col = len(value) - value.rfind("\n")
            else:
                col += len(value)
            continue
        tokens.append(Token(kind=kind, value=value, line=line, col=col))  # type: ignore[arg-type]
        col += len(value)
    tokens.append(Token(kind="EOF", value="", line=line, col=col))
    return tokens


# ---------------------------------------------------------------------------
# Value coercion
# ---------------------------------------------------------------------------

_DURATION_RE = re.compile(r"^(\d+(?:\.\d+)?)(ms|s|m|h)$")
_BOOL_TRUE = {"true", "yes", "on"}
_BOOL_FALSE = {"false", "no", "off"}


def _coerce_value(raw: str) -> Any:
    """Convert a raw string token value to a Python type."""
    # Quoted string – strip quotes and unescape
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1].replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")

    low = raw.lower()

    # Boolean
    if low in _BOOL_TRUE:
        return True
    if low in _BOOL_FALSE:
        return False

    # Duration (e.g. "30s", "500ms")
    dm = _DURATION_RE.match(raw)
    if dm:
        num, unit = float(dm.group(1)), dm.group(2)
        multiplier = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
        return num * multiplier[unit]

    # Integer / Float
    try:
        if "." in raw or "e" in low:
            return float(raw)
        return int(raw)
    except ValueError:
        pass

    return raw


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class ParseError(Exception):
    def __init__(self, msg: str, token: Optional[Token] = None):
        loc = f" at line {token.line}, col {token.col}" if token else ""
        super().__init__(f"{msg}{loc}")


class _Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    # -- helpers --------------------------------------------------------

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def _expect(self, kind: str, value: Optional[str] = None) -> Token:
        t = self._advance()
        if t.kind != kind or (value is not None and t.value != value):
            expected = f"{kind}" + (f" '{value}'" if value else "")
            raise ParseError(f"Expected {expected}, got {t.kind} '{t.value}'", t)
        return t

    def _match(self, kind: str, value: Optional[str] = None) -> Optional[Token]:
        t = self._peek()
        if t.kind == kind and (value is None or t.value == value):
            return self._advance()
        return None

    def _at(self, kind: str, value: Optional[str] = None) -> bool:
        t = self._peek()
        return t.kind == kind and (value is None or t.value == value)

    # -- grammar --------------------------------------------------------

    def parse_graph(self) -> Graph:
        self._match("ID", "strict")
        self._expect("ID", "digraph")
        # Optional graph name
        graph_name = ""
        if self._at("ID") or self._at("STRING"):
            t = self._advance()
            graph_name = _coerce_value(t.value) if t.kind == "STRING" else t.value
        self._expect("LBRACE")

        graph = Graph()
        if graph_name:
            graph.attrs["name"] = graph_name

        node_defaults: Dict[str, Any] = {}
        edge_defaults: Dict[str, Any] = {}

        self._parse_stmt_list(graph, node_defaults, edge_defaults)

        self._expect("RBRACE")
        return graph

    def _parse_stmt_list(
        self,
        graph: Graph,
        node_defaults: Dict[str, Any],
        edge_defaults: Dict[str, Any],
    ) -> None:
        while not self._at("RBRACE") and not self._at("EOF"):
            self._parse_stmt(graph, node_defaults, edge_defaults)
            self._match("SEMI")

    def _parse_stmt(
        self,
        graph: Graph,
        node_defaults: Dict[str, Any],
        edge_defaults: Dict[str, Any],
    ) -> None:
        # subgraph
        if self._at("ID", "subgraph"):
            self._parse_subgraph(graph, node_defaults, edge_defaults)
            return

        # graph/node/edge defaults
        if self._at("ID", "graph") and self._lookahead_is("LBRACK"):
            self._advance()
            attrs = self._parse_attr_list()
            graph.attrs.update(attrs)
            return
        if self._at("ID", "node") and self._lookahead_is("LBRACK"):
            self._advance()
            attrs = self._parse_attr_list()
            node_defaults.update(attrs)
            return
        if self._at("ID", "edge") and self._lookahead_is("LBRACK"):
            self._advance()
            attrs = self._parse_attr_list()
            edge_defaults.update(attrs)
            return

        # Top-level key=value (graph attribute declaration like rankdir=LR)
        if self._at("ID") and self._lookahead_is("EQ"):
            # Check it's not a node with attr list: node [attr=val]
            # Graph attr decls are: ID = VALUE (no brackets)
            if not self._lookahead_n_is(2, "LBRACK"):
                key_tok = self._advance()
                self._expect("EQ")
                val_tok = self._advance()
                if val_tok.kind in ("STRING", "NUMBER", "ID"):
                    graph.attrs[key_tok.value] = _coerce_value(val_tok.value)
                else:
                    raise ParseError(f"Expected value, got {val_tok.kind} '{val_tok.value}'", val_tok)
                return

        # Could be node_stmt or edge_stmt – read first ID then look ahead
        if self._at("ID") or self._at("STRING") or self._at("NUMBER"):
            first_id = self._parse_node_id()
            if self._at("ARROW"):
                # edge statement (possibly chained)
                chain = [first_id]
                while self._match("ARROW"):
                    chain.append(self._parse_node_id())
                attrs = {}
                if self._at("LBRACK"):
                    attrs = self._parse_attr_list()
                merged = {**edge_defaults, **attrs}
                for i in range(len(chain) - 1):
                    edge = _make_edge(chain[i], chain[i + 1], merged)
                    graph.edges.append(edge)
                    # ensure nodes exist
                    for nid in (chain[i], chain[i + 1]):
                        if nid not in graph.nodes:
                            graph.nodes[nid] = _make_node(nid, node_defaults)
            else:
                # node statement
                attrs = {}
                if self._at("LBRACK"):
                    attrs = self._parse_attr_list()
                merged = {**node_defaults, **attrs}
                if first_id in graph.nodes:
                    _update_node(graph.nodes[first_id], merged)
                else:
                    graph.nodes[first_id] = _make_node(first_id, merged)

    def _parse_subgraph(
        self,
        graph: Graph,
        node_defaults: Dict[str, Any],
        edge_defaults: Dict[str, Any],
    ) -> None:
        self._expect("ID", "subgraph")
        sub_name = ""
        if self._at("ID") or self._at("STRING"):
            t = self._advance()
            sub_name = _coerce_value(t.value) if t.kind == "STRING" else t.value
        self._expect("LBRACE")
        # Inherit defaults
        sub_node_defaults = dict(node_defaults)
        sub_edge_defaults = dict(edge_defaults)
        self._parse_stmt_list(graph, sub_node_defaults, sub_edge_defaults)
        self._expect("RBRACE")

    def _parse_node_id(self) -> str:
        t = self._advance()
        if t.kind == "STRING":
            return _coerce_value(t.value)  # type: ignore[return-value]
        if t.kind in ("ID", "NUMBER"):
            return t.value
        raise ParseError(f"Expected node ID, got {t.kind} '{t.value}'", t)

    def _parse_attr_list(self) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {}
        self._expect("LBRACK")
        while not self._at("RBRACK") and not self._at("EOF"):
            key = self._advance()
            if key.kind not in ("ID", "STRING"):
                raise ParseError(f"Expected attribute name, got {key.kind} '{key.value}'", key)
            key_name = key.value
            self._expect("EQ")
            val_tok = self._advance()
            if val_tok.kind in ("STRING", "NUMBER", "ID"):
                val = _coerce_value(val_tok.value)
            else:
                raise ParseError(f"Expected value, got {val_tok.kind} '{val_tok.value}'", val_tok)
            attrs[key_name] = val
            self._match("COMMA")
            self._match("SEMI")
        self._expect("RBRACK")
        return attrs

    def _lookahead_is(self, kind: str) -> bool:
        if self.pos + 1 < len(self.tokens):
            return self.tokens[self.pos + 1].kind == kind
        return False

    def _lookahead_n_is(self, n: int, kind: str) -> bool:
        idx = self.pos + n
        if idx < len(self.tokens):
            return self.tokens[idx].kind == kind
        return False


# ---------------------------------------------------------------------------
# Node / Edge construction helpers
# ---------------------------------------------------------------------------

_NODE_FIELDS = {
    "label", "shape", "type", "prompt", "max_retries", "goal_gate",
    "retry_target", "fallback_retry_target", "fidelity", "thread_id",
    "class", "timeout", "llm_model", "llm_provider", "reasoning_effort",
    "auto_status", "allow_partial", "output_key",
}

_EDGE_FIELDS = {
    "label", "condition", "weight", "fidelity", "thread_id", "loop_restart",
}


def _make_node(node_id: str, attrs: Dict[str, Any]) -> Node:
    kwargs: Dict[str, Any] = {"id": node_id}
    extra: Dict[str, Any] = {}
    for k, v in attrs.items():
        mapped = "class_" if k == "class" else k
        if k in _NODE_FIELDS:
            kwargs[mapped] = v
        else:
            extra[k] = v
    kwargs["attrs"] = extra
    return Node(**kwargs)


def _update_node(node: Node, attrs: Dict[str, Any]) -> None:
    for k, v in attrs.items():
        mapped = "class_" if k == "class" else k
        if k in _NODE_FIELDS:
            setattr(node, mapped, v)
        else:
            node.attrs[k] = v


def _make_edge(from_id: str, to_id: str, attrs: Dict[str, Any]) -> Edge:
    kwargs: Dict[str, Any] = {"from_node": from_id, "to_node": to_id}
    for k, v in attrs.items():
        if k in _EDGE_FIELDS:
            kwargs[k] = v
    return Edge(**kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_dot(source: str) -> Graph:
    """Parse a DOT-format string and return a Graph."""
    tokens = _tokenize(source)
    parser = _Parser(tokens)
    return parser.parse_graph()
