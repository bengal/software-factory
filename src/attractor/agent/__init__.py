"""Coding agent loop: session, environment, events, and tools."""

from attractor.agent.environment import (
    ExecutionEnvironment,
    LocalExecutionEnvironment,
    ExecResult,
    DirEntry,
)
from attractor.agent.events import (
    EventKind,
    SessionEvent,
    EventEmitter,
)
from attractor.agent.session import (
    Session,
    SessionConfig,
    SessionState,
    UserTurn,
    AssistantTurn,
    ToolResultsTurn,
    SteeringTurn,
    detect_loop,
    truncate_tool_output,
)
from attractor.agent.tools.registry import ToolRegistry, RegisteredTool, ToolDefinition
from attractor.agent.tools.profiles import create_profile, ProviderProfile

__all__ = [
    # Environment
    "ExecutionEnvironment",
    "LocalExecutionEnvironment",
    "ExecResult",
    "DirEntry",
    # Events
    "EventKind",
    "SessionEvent",
    "EventEmitter",
    # Session
    "Session",
    "SessionConfig",
    "SessionState",
    "UserTurn",
    "AssistantTurn",
    "ToolResultsTurn",
    "SteeringTurn",
    "detect_loop",
    "truncate_tool_output",
    # Tools
    "ToolRegistry",
    "RegisteredTool",
    "ToolDefinition",
    "create_profile",
    "ProviderProfile",
]
