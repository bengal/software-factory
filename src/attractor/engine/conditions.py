"""Condition expression evaluator for edge routing.

Supports:
    - Variable references: $var, $context.key
    - Comparisons: =, !=, >, <, >=, <=
    - Logical operators: and, or, not
    - Status predicates: outcome=success
    - String/numeric/boolean literals
    - Parenthesised sub-expressions

Grammar:
    expr     := or_expr
    or_expr  := and_expr ('or' and_expr)*
    and_expr := not_expr ('and' not_expr)*
    not_expr := 'not' not_expr | cmp_expr
    cmp_expr := primary (cmp_op primary)?
    primary  := '(' expr ')' | variable | literal
    cmp_op   := '=' | '!=' | '>' | '<' | '>=' | '<='
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional

from .context import Outcome


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOK_PATTERNS = [
    ("WS", r"[ \t]+"),
    ("CMP", r"!=|>=|<=|>|<|="),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("VAR", r"\$[A-Za-z_][A-Za-z0-9_.]*"),
    ("NUMBER", r"-?(?:\d+\.\d*|\.\d+|\d+)"),
    ("STRING", r'"(?:[^"\\]|\\.)*"'),
    ("STRING_SQ", r"'(?:[^'\\]|\\.)*'"),
    ("WORD", r"[A-Za-z_][A-Za-z0-9_]*"),
]

_TOK_RE = re.compile("|".join(f"(?P<{n}>{p})" for n, p in _TOK_PATTERNS))


@dataclass
class _Tok:
    kind: str
    value: str


def _tokenize(expr: str) -> List[_Tok]:
    tokens: List[_Tok] = []
    for m in _TOK_RE.finditer(expr):
        kind = m.lastgroup
        if kind == "WS":
            continue
        tokens.append(_Tok(kind=kind, value=m.group()))  # type: ignore[arg-type]
    tokens.append(_Tok(kind="EOF", value=""))
    return tokens


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------

@dataclass
class _BinOp:
    op: str
    left: Any
    right: Any


@dataclass
class _UnaryOp:
    op: str
    operand: Any


@dataclass
class _Var:
    name: str  # without leading $


@dataclass
class _Literal:
    value: Any


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class _CondParser:
    def __init__(self, tokens: List[_Tok]):
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> _Tok:
        return self.tokens[self.pos]

    def _advance(self) -> _Tok:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def _match_word(self, val: str) -> bool:
        t = self._peek()
        if t.kind == "WORD" and t.value.lower() == val:
            self._advance()
            return True
        return False

    def parse(self) -> Any:
        node = self._or_expr()
        return node

    def _or_expr(self) -> Any:
        left = self._and_expr()
        while self._match_word("or"):
            right = self._and_expr()
            left = _BinOp("or", left, right)
        return left

    def _and_expr(self) -> Any:
        left = self._not_expr()
        while self._match_word("and"):
            right = self._not_expr()
            left = _BinOp("and", left, right)
        return left

    def _not_expr(self) -> Any:
        if self._match_word("not"):
            return _UnaryOp("not", self._not_expr())
        return self._cmp_expr()

    def _cmp_expr(self) -> Any:
        left = self._primary()
        t = self._peek()
        if t.kind == "CMP":
            op = self._advance().value
            right = self._primary()
            return _BinOp(op, left, right)
        return left

    def _primary(self) -> Any:
        t = self._peek()
        if t.kind == "LPAREN":
            self._advance()
            node = self._or_expr()
            if self._peek().kind == "RPAREN":
                self._advance()
            return node
        if t.kind == "VAR":
            self._advance()
            return _Var(t.value[1:])  # strip $
        if t.kind == "NUMBER":
            self._advance()
            if "." in t.value:
                return _Literal(float(t.value))
            return _Literal(int(t.value))
        if t.kind in ("STRING", "STRING_SQ"):
            self._advance()
            return _Literal(t.value[1:-1])
        if t.kind == "WORD":
            self._advance()
            low = t.value.lower()
            if low == "true":
                return _Literal(True)
            if low == "false":
                return _Literal(False)
            return _Literal(t.value)
        # fallback
        self._advance()
        return _Literal(t.value)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def _resolve_var(name: str, outcome: Optional[Outcome], context: Any) -> Any:
    """Resolve a variable reference."""
    # outcome.* shortcuts
    if name == "outcome":
        return outcome.status.value if outcome else ""
    if name.startswith("outcome."):
        field = name[len("outcome."):]
        if outcome and hasattr(outcome, field):
            val = getattr(outcome, field)
            if hasattr(val, "value"):
                return val.value
            return val
        return ""

    # context.* or plain context ref
    if name.startswith("context."):
        key = name[len("context."):]
        if context and hasattr(context, "get"):
            return context.get(key, "")
        return ""

    # Direct context lookup
    if context and hasattr(context, "get"):
        val = context.get(name, None)
        if val is not None:
            return val

    return ""


def _compare(left: Any, op: str, right: Any) -> bool:
    # Coerce types for comparison.
    # Handle bools specially: compare as lowercase strings ("true"/"false")
    # because bool is a subclass of int and bool("false") is True.
    if isinstance(left, bool) or isinstance(right, bool):
        left = str(left).lower()
        right = str(right).lower()
    elif isinstance(left, str) and isinstance(right, (int, float)):
        try:
            left = type(right)(left)
        except (ValueError, TypeError):
            pass
    elif isinstance(right, str) and isinstance(left, (int, float)):
        try:
            right = type(left)(right)
        except (ValueError, TypeError):
            pass

    if op == "=":
        if isinstance(left, str) and isinstance(right, str):
            return left.lower() == right.lower()
        return left == right
    if op == "!=":
        if isinstance(left, str) and isinstance(right, str):
            return left.lower() != right.lower()
        return left != right
    if op == ">":
        return left > right
    if op == "<":
        return left < right
    if op == ">=":
        return left >= right
    if op == "<=":
        return left <= right
    return False


def _eval_node(node: Any, outcome: Optional[Outcome], context: Any) -> Any:
    if isinstance(node, _Literal):
        return node.value
    if isinstance(node, _Var):
        return _resolve_var(node.name, outcome, context)
    if isinstance(node, _UnaryOp):
        if node.op == "not":
            return not _eval_node(node.operand, outcome, context)
    if isinstance(node, _BinOp):
        left = _eval_node(node.left, outcome, context)
        right = _eval_node(node.right, outcome, context)
        if node.op == "and":
            return bool(left) and bool(right)
        if node.op == "or":
            return bool(left) or bool(right)
        return _compare(left, node.op, right)
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_condition(expr: str, outcome: Optional[Outcome] = None, context: Any = None) -> bool:
    """Parse and evaluate a condition expression.

    Args:
        expr: The condition string (e.g. ``"$outcome = success"``).
        outcome: The current Outcome from the previous node execution.
        context: The pipeline Context (key-value store).

    Returns:
        True if the condition is satisfied, False otherwise.
    """
    if not expr or not expr.strip():
        return True
    tokens = _tokenize(expr)
    parser = _CondParser(tokens)
    tree = parser.parse()
    result = _eval_node(tree, outcome, context)
    return bool(result)
