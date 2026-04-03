"""Tool registry for the coding agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolDefinition:
    """Definition of a tool available to the agent.

    Contains the name, description, and JSON Schema for the parameters.
    """
    name: str
    description: str
    parameters: dict[str, Any]


# A tool executor receives (arguments_dict, env) and returns a string result.
ToolExecutor = Callable[..., str]


@dataclass
class RegisteredTool:
    """A tool definition paired with its executor function."""
    definition: ToolDefinition
    executor: ToolExecutor


class ToolRegistry:
    """Registry of available tools.

    Maps tool names to their definitions and executors.
    """

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        """Register a tool."""
        self._tools[tool.definition.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> RegisteredTool | None:
        """Look up a registered tool by name."""
        return self._tools.get(name)

    @property
    def definitions(self) -> list[ToolDefinition]:
        """Return all tool definitions."""
        return [t.definition for t in self._tools.values()]

    @property
    def names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
