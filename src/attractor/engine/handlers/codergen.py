"""CodergenHandler - executes LLM-backed code generation nodes.

Supports pluggable backends via the CodergenBackend abstract interface.
"""

from __future__ import annotations

import abc
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..context import Context, Outcome, StageStatus
from ..graph import Graph, Node
from .base import Handler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------

class CodergenBackend(abc.ABC):
    """Abstract interface for LLM backends used by CodergenHandler."""

    @abc.abstractmethod
    def run(self, node: Node, prompt: str, context: Context) -> Union[str, Outcome]:
        """Execute a prompt and return the raw response text or a full Outcome."""
        ...


# ---------------------------------------------------------------------------
# Variable expansion
# ---------------------------------------------------------------------------

_VAR_RE = re.compile(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)")


def expand_variables(template: str, node: Node, context: Context, graph: Graph) -> str:
    """Expand ``$var`` and ``${var}`` references in *template*.

    Special variables:
        $goal    - the graph-level ``goal`` attribute
        $prompt  - the node's own prompt
        $label   - the node's label
    Other names are looked up in *context*.
    """

    def _replace(m: re.Match) -> str:
        name = m.group(1) or m.group(2)
        # Built-in specials
        if name == "goal":
            return str(graph.attrs.get("goal", ""))
        if name == "prompt":
            return node.prompt
        if name == "label":
            return node.label
        # Dotted context lookup
        if name.startswith("context."):
            key = name[len("context."):]
            return str(context.get(key, ""))
        # Plain context lookup
        val = context.get(name)
        if val is not None:
            return str(val)
        # Node attr fallback
        if hasattr(node, name):
            return str(getattr(node, name))
        return m.group(0)  # leave unresolved

    return _VAR_RE.sub(_replace, template)


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

_VERDICT_RE = re.compile(r"verdict\s*:\s*(pass|fail)", re.IGNORECASE)


def _parse_verdict(text: str) -> bool:
    """Parse a PASS/FAIL verdict from verification output.

    Looks for the last occurrence of "VERDICT: PASS" or "VERDICT: FAIL"
    (case-insensitive). Using the *last* match avoids false positives from
    test output earlier in the text that might contain those words.

    Falls back to False (fail) if no verdict line is found.
    """
    matches = list(_VERDICT_RE.finditer(text))
    if matches:
        return matches[-1].group(1).lower() == "pass"
    # No explicit verdict found — default to fail
    logger.warning("No 'VERDICT: PASS/FAIL' found in verification output; defaulting to FAIL")
    return False


# ---------------------------------------------------------------------------
# CodergenHandler
# ---------------------------------------------------------------------------

class CodergenHandler(Handler):
    """Handler for LLM code-generation nodes (shape=box/rect/ellipse)."""

    def __init__(self, backend: Optional[CodergenBackend] = None) -> None:
        self.backend = backend

    def execute(
        self,
        node: Node,
        context: Context,
        graph: Graph,
        logs_root: Optional[Path] = None,
    ) -> Outcome:
        prompt = expand_variables(node.prompt, node, context, graph)

        preview = prompt[:200] + ("…" if len(prompt) > 200 else "")
        logger.info("Prompt for %s (%d chars): \033[34m%s\033[0m", node.id, len(prompt), preview)

        # Write prompt to logs
        if logs_root:
            log_dir = logs_root / node.id
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "prompt.md").write_text(prompt, encoding="utf-8")

        # Execute backend
        if self.backend is None:
            outcome = Outcome(
                status=StageStatus.FAIL,
                failure_reason="No CodergenBackend configured",
            )
        else:
            try:
                result = self.backend.run(node, prompt, context)
                if isinstance(result, Outcome):
                    outcome = result
                else:
                    # Raw text response -> success
                    outcome = Outcome(status=StageStatus.SUCCESS, notes=str(result))
                    # Store response in context if node declares an output_key
                    if node.output_key:
                        context.set(node.output_key, str(result))
                    # If this is a verify node, parse the verdict to set verification_passed
                    if node.output_key == "verification_result":
                        context.set("verification_passed",
                                    "true" if _parse_verdict(str(result)) else "false")
                    # Write response
                    if logs_root:
                        log_dir = logs_root / node.id
                        log_dir.mkdir(parents=True, exist_ok=True)
                        (log_dir / "response.md").write_text(str(result), encoding="utf-8")
            except Exception as exc:
                outcome = Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=str(exc),
                )

        # Write status.json
        if logs_root:
            log_dir = logs_root / node.id
            log_dir.mkdir(parents=True, exist_ok=True)
            status_data = {
                "node_id": node.id,
                "status": outcome.status.value,
                "failure_reason": outcome.failure_reason,
            }
            (log_dir / "status.json").write_text(
                json.dumps(status_data, indent=2), encoding="utf-8"
            )

        return outcome
