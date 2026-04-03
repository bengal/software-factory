"""ToolHandler - execute shell commands as pipeline nodes.

Runs a command via subprocess, captures stdout/stderr, and maps the
exit code to a StageStatus.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

from ..context import Context, Outcome, StageStatus
from ..graph import Graph, Node
from .base import Handler


class ToolHandler(Handler):
    """Handler for tool/command nodes (shape=hexagon).

    The command to execute is taken from ``node.prompt``.  Variable
    expansion is applied before execution.
    """

    def execute(
        self,
        node: Node,
        context: Context,
        graph: Graph,
        logs_root: Optional[Path] = None,
    ) -> Outcome:
        from .codergen import expand_variables

        command = expand_variables(node.prompt, node, context, graph)
        if not command.strip():
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason="No command specified (node.prompt is empty)",
            )

        timeout = node.timeout
        log_dir: Optional[Path] = None
        if logs_root:
            log_dir = logs_root / node.id
            log_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout = result.stdout
            stderr = result.stderr
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            stdout = ""
            stderr = f"Command timed out after {timeout}s"
            returncode = -1
        except Exception as exc:
            stdout = ""
            stderr = str(exc)
            returncode = -1

        # Store outputs in context
        context.set(f"{node.id}.stdout", stdout)
        context.set(f"{node.id}.stderr", stderr)
        context.set(f"{node.id}.returncode", returncode)

        # Write logs
        if log_dir:
            (log_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
            (log_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
            status_data = {
                "node_id": node.id,
                "command": command,
                "returncode": returncode,
                "status": "success" if returncode == 0 else "fail",
            }
            (log_dir / "status.json").write_text(
                json.dumps(status_data, indent=2), encoding="utf-8"
            )

        if returncode == 0:
            return Outcome(status=StageStatus.SUCCESS, notes=stdout)
        else:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=f"Exit code {returncode}: {stderr}",
            )
