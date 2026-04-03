"""Core tools for the coding agent.

Each tool is a function(arguments: dict, env: ExecutionEnvironment) -> str
with an associated JSON Schema definition.
"""

from __future__ import annotations

import difflib
import json
from typing import Any

from attractor.agent.environment import ExecutionEnvironment
from attractor.agent.tools.registry import RegisteredTool, ToolDefinition


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

_READ_FILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": "Absolute path to the file to read.",
        },
        "offset": {
            "type": "integer",
            "description": "Line number to start reading from (1-based). Optional.",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of lines to read. Optional.",
        },
    },
    "required": ["file_path"],
}


def read_file(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
    """Read a file and return its contents with line numbers."""
    path = arguments["file_path"]
    try:
        content = env.read_file(path)
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except Exception as exc:
        return f"Error reading file: {exc}"

    lines = content.splitlines(keepends=True)
    offset = arguments.get("offset", 1)
    limit = arguments.get("limit")

    start = max(0, offset - 1)
    end = start + limit if limit else len(lines)
    selected = lines[start:end]

    numbered = []
    for i, line in enumerate(selected, start=start + 1):
        numbered.append(f"{i:>6}\t{line.rstrip()}")
    return "\n".join(numbered)


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

_WRITE_FILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": "Absolute path to the file to write.",
        },
        "content": {
            "type": "string",
            "description": "The content to write to the file.",
        },
    },
    "required": ["file_path", "content"],
}


def write_file(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
    """Write content to a file."""
    path = arguments["file_path"]
    content = arguments["content"]
    try:
        env.write_file(path, content)
        lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return f"Successfully wrote {lines} lines to {path}"
    except Exception as exc:
        return f"Error writing file: {exc}"


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------

_EDIT_FILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": "Absolute path to the file to edit.",
        },
        "old_string": {
            "type": "string",
            "description": "The exact text to find and replace.",
        },
        "new_string": {
            "type": "string",
            "description": "The replacement text.",
        },
        "replace_all": {
            "type": "boolean",
            "description": "If true, replace all occurrences. Default false.",
        },
    },
    "required": ["file_path", "old_string", "new_string"],
}


def edit_file(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
    """Edit a file by replacing exact string matches."""
    path = arguments["file_path"]
    old_string = arguments["old_string"]
    new_string = arguments["new_string"]
    replace_all = arguments.get("replace_all", False)

    try:
        content = env.read_file(path)
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except Exception as exc:
        return f"Error reading file: {exc}"

    if old_string == new_string:
        return "Error: old_string and new_string are identical."

    count = content.count(old_string)
    if count == 0:
        return f"Error: old_string not found in {path}"
    if count > 1 and not replace_all:
        return (
            f"Error: old_string appears {count} times in {path}. "
            f"Provide more context to make it unique, or set replace_all=true."
        )

    if replace_all:
        new_content = content.replace(old_string, new_string)
    else:
        new_content = content.replace(old_string, new_string, 1)

    try:
        env.write_file(path, new_content)
    except Exception as exc:
        return f"Error writing file: {exc}"

    replacements = count if replace_all else 1
    return f"Successfully edited {path} ({replacements} replacement{'s' if replacements > 1 else ''})"


# ---------------------------------------------------------------------------
# shell
# ---------------------------------------------------------------------------

_SHELL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "The shell command to execute.",
        },
        "timeout": {
            "type": "integer",
            "description": "Timeout in milliseconds (default 120000).",
        },
        "working_dir": {
            "type": "string",
            "description": "Working directory for the command. Optional.",
        },
    },
    "required": ["command"],
}


def shell(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
    """Execute a shell command."""
    command = arguments["command"]
    timeout = arguments.get("timeout", 120_000)
    working_dir = arguments.get("working_dir")

    result = env.exec_command(command, timeout_ms=timeout, working_dir=working_dir)

    parts: list[str] = []
    if result.stdout:
        parts.append(result.stdout)
    if result.stderr:
        parts.append(f"STDERR:\n{result.stderr}")
    if result.timed_out:
        parts.append(f"[Command timed out after {timeout}ms]")

    output = "\n".join(parts) if parts else "(no output)"
    exit_info = f"[exit code: {result.exit_code}]"
    return f"{output}\n{exit_info}"


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------

_GREP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Regex pattern to search for.",
        },
        "path": {
            "type": "string",
            "description": "File or directory to search in. Defaults to working directory.",
        },
        "include": {
            "type": "string",
            "description": "Glob pattern to filter files (e.g., '*.py').",
        },
    },
    "required": ["pattern"],
}


def grep(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
    """Search file contents for a regex pattern."""
    pattern = arguments["pattern"]
    path = arguments.get("path")
    include = arguments.get("include")

    try:
        result = env.grep(pattern, path, include=include)
        return result if result else "No matches found."
    except Exception as exc:
        return f"Error during search: {exc}"


# ---------------------------------------------------------------------------
# glob
# ---------------------------------------------------------------------------

_GLOB_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Glob pattern to match files (e.g., '**/*.py').",
        },
        "path": {
            "type": "string",
            "description": "Directory to search in. Defaults to working directory.",
        },
    },
    "required": ["pattern"],
}


def glob_tool(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
    """Find files matching a glob pattern."""
    pattern = arguments["pattern"]
    path = arguments.get("path")

    try:
        results = env.glob(pattern, path)
        if not results:
            return "No files found."
        return "\n".join(results)
    except Exception as exc:
        return f"Error during glob: {exc}"


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_core_tools(env: ExecutionEnvironment) -> list[RegisteredTool]:
    """Create the core set of tools for a coding agent session.

    Each tool's executor is partially applied with the given environment.
    """
    tools = [
        RegisteredTool(
            definition=ToolDefinition(
                name="read_file",
                description="Read a file from the filesystem. Returns contents with line numbers.",
                parameters=_READ_FILE_SCHEMA,
            ),
            executor=lambda args, _env=env: read_file(args, _env),
        ),
        RegisteredTool(
            definition=ToolDefinition(
                name="write_file",
                description="Write content to a file. Creates parent directories as needed.",
                parameters=_WRITE_FILE_SCHEMA,
            ),
            executor=lambda args, _env=env: write_file(args, _env),
        ),
        RegisteredTool(
            definition=ToolDefinition(
                name="edit_file",
                description="Edit a file by finding and replacing an exact string.",
                parameters=_EDIT_FILE_SCHEMA,
            ),
            executor=lambda args, _env=env: edit_file(args, _env),
        ),
        RegisteredTool(
            definition=ToolDefinition(
                name="shell",
                description="Execute a shell command and return stdout/stderr.",
                parameters=_SHELL_SCHEMA,
            ),
            executor=lambda args, _env=env: shell(args, _env),
        ),
        RegisteredTool(
            definition=ToolDefinition(
                name="grep",
                description="Search file contents for a regex pattern using ripgrep.",
                parameters=_GREP_SCHEMA,
            ),
            executor=lambda args, _env=env: grep(args, _env),
        ),
        RegisteredTool(
            definition=ToolDefinition(
                name="glob",
                description="Find files matching a glob pattern.",
                parameters=_GLOB_SCHEMA,
            ),
            executor=lambda args, _env=env: glob_tool(args, _env),
        ),
    ]
    return tools
