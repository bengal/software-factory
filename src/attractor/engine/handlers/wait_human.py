"""WaitForHumanHandler - pause pipeline for human input.

Presents choices derived from outgoing edges and waits for a selection
via an Interviewer interface.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..context import Context, Outcome, StageStatus
from ..graph import Graph, Node
from .base import Handler


# ---------------------------------------------------------------------------
# Interviewer interfaces
# ---------------------------------------------------------------------------

@dataclass
class Choice:
    """A single choice derived from an outgoing edge."""

    label: str
    target_node_id: str


class Interviewer(abc.ABC):
    """Abstract interface for collecting human input."""

    @abc.abstractmethod
    def ask(self, prompt: str, choices: List[Choice]) -> Choice:
        """Present *choices* to a human and return the selected one."""
        ...


class AutoApproveInterviewer(Interviewer):
    """Automatically selects the first choice (for testing / CI)."""

    def ask(self, prompt: str, choices: List[Choice]) -> Choice:
        if not choices:
            return Choice(label="(none)", target_node_id="")
        return choices[0]


class ConsoleInterviewer(Interviewer):
    """Interactive console-based interviewer."""

    def ask(self, prompt: str, choices: List[Choice]) -> Choice:
        print(f"\n{'=' * 60}")
        print(f"  {prompt}")
        print(f"{'=' * 60}")
        if not choices:
            print("  No choices available. Press Enter to continue.")
            input()
            return Choice(label="(none)", target_node_id="")

        for i, c in enumerate(choices, 1):
            print(f"  [{i}] {c.label}  -> {c.target_node_id}")
        print()

        while True:
            try:
                raw = input("  Select [1]: ").strip()
                if not raw:
                    idx = 0
                else:
                    idx = int(raw) - 1
                if 0 <= idx < len(choices):
                    return choices[idx]
            except (ValueError, EOFError):
                pass
            print("  Invalid selection, try again.")


# ---------------------------------------------------------------------------
# WaitForHumanHandler
# ---------------------------------------------------------------------------

class WaitForHumanHandler(Handler):
    """Handler for human-gate nodes (shape=octagon).

    Derives a list of choices from outgoing edges and asks the configured
    Interviewer to select one.  The selected choice's label becomes the
    ``preferred_label`` on the Outcome so the executor routes accordingly.
    """

    def __init__(self, interviewer: Optional[Interviewer] = None) -> None:
        self.interviewer = interviewer or AutoApproveInterviewer()

    def execute(
        self,
        node: Node,
        context: Context,
        graph: Graph,
        logs_root: Optional[Path] = None,
    ) -> Outcome:
        outgoing = graph.outgoing_edges(node.id)
        choices = [
            Choice(label=e.label or e.to_node, target_node_id=e.to_node)
            for e in outgoing
        ]

        prompt_text = node.prompt or node.label or f"Choose next step from {node.id}"
        selected = self.interviewer.ask(prompt_text, choices)

        return Outcome(
            status=StageStatus.SUCCESS,
            preferred_label=selected.label,
            suggested_next_ids=[selected.target_node_id]
            if selected.target_node_id
            else [],
        )
