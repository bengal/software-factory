"""Coding agent session and agentic loop.

The Session drives a multi-turn conversation with an LLM, automatically
executing tool calls and feeding results back until the model stops
requesting tools or limits are reached.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from attractor.agent.events import EventEmitter, EventKind, SessionEvent
from attractor.agent.tools.registry import ToolRegistry
from attractor.llm.client import Client
from attractor.llm.types import (
    ContentKind,
    ContentPart,
    FinishReason,
    Message,
    Request,
    Role,
    ToolCallData,
    ToolDefinition as LLMToolDefinition,
    ToolResult,
    ToolResultData,
    Usage,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SessionConfig:
    """Configuration for an agent session."""
    max_turns: int = 50
    max_tool_rounds_per_input: int = 30
    default_command_timeout_ms: int = 120_000
    reasoning_effort: str | None = None  # "low" | "medium" | "high"
    enable_loop_detection: bool = True
    loop_detection_window: int = 4
    max_output_chars: int = 100_000
    truncation_head_ratio: float = 0.3


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

class SessionState(str, Enum):
    """High-level session state."""
    IDLE = "idle"
    PROCESSING = "processing"
    AWAITING_INPUT = "awaiting_input"
    CLOSED = "closed"


# ---------------------------------------------------------------------------
# Turn types
# ---------------------------------------------------------------------------

@dataclass
class UserTurn:
    """A user input turn."""
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AssistantTurn:
    """An assistant response turn."""
    message: Message
    usage: Usage = field(default_factory=Usage)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolResultsTurn:
    """A turn containing tool execution results."""
    results: list[ToolResult]
    timestamp: float = field(default_factory=time.time)


@dataclass
class SteeringTurn:
    """A steering injection turn (system-level guidance)."""
    content: str
    timestamp: float = field(default_factory=time.time)


Turn = UserTurn | AssistantTurn | ToolResultsTurn | SteeringTurn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_loop(history: list[Turn], window_size: int = 4) -> bool:
    """Detect if the agent is stuck in a loop.

    Checks whether the last `window_size` assistant turns produced
    identical or near-identical tool call sequences.
    """
    # Collect recent assistant turns
    recent_assistant: list[AssistantTurn] = []
    for turn in reversed(history):
        if isinstance(turn, AssistantTurn):
            recent_assistant.append(turn)
            if len(recent_assistant) >= window_size:
                break

    if len(recent_assistant) < window_size:
        return False

    # Hash the tool calls from each turn
    hashes: list[str] = []
    for turn in recent_assistant:
        if isinstance(turn.message.content, str):
            h = hashlib.md5(turn.message.content.encode()).hexdigest()
        else:
            # Hash the sequence of tool call names + args
            parts = []
            for part in turn.message.content:
                if part.kind == ContentKind.TOOL_CALL and part.tool_call:
                    parts.append(f"{part.tool_call.name}:{part.tool_call.arguments}")
            h = hashlib.md5("|".join(parts).encode()).hexdigest()
        hashes.append(h)

    # All the same hash -> loop
    return len(set(hashes)) == 1


def truncate_tool_output(
    output: str,
    tool_name: str,
    config: SessionConfig,
) -> str:
    """Truncate tool output if it exceeds the maximum.

    Keeps a head portion and a tail portion with a truncation notice.
    """
    if len(output) <= config.max_output_chars:
        return output

    head_chars = int(config.max_output_chars * config.truncation_head_ratio)
    tail_chars = config.max_output_chars - head_chars - 200  # room for notice

    head = output[:head_chars]
    tail = output[-tail_chars:] if tail_chars > 0 else ""
    omitted = len(output) - head_chars - max(tail_chars, 0)

    return (
        f"{head}\n\n"
        f"[... {omitted} characters omitted from {tool_name} output ...]\n\n"
        f"{tail}"
    )


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class Session:
    """Coding agent session managing the agentic loop.

    Maintains conversation history, executes tools, and drives
    the LLM through multi-turn interactions.
    """

    def __init__(
        self,
        client: Client,
        tool_registry: ToolRegistry,
        system_prompt: str = "",
        config: SessionConfig | None = None,
        llm_model: str | None = None,
        llm_tools: list[LLMToolDefinition] | None = None,
        provider_options: dict[str, Any] | None = None,
    ):
        self.id = str(uuid.uuid4())
        self.history: list[Turn] = []
        self.event_emitter = EventEmitter()
        self.config = config or SessionConfig()
        self.state = SessionState.IDLE

        self._client = client
        self._tool_registry = tool_registry
        self._system_prompt = system_prompt
        self._llm_model = llm_model or ""
        self._llm_tools = llm_tools
        self._provider_options = provider_options or {}

        self._steering_queue: deque[str] = deque()
        self._followup_queue: deque[str] = deque()
        self._total_usage = Usage()
        self._tool_rounds_used = 0

    # -- public API --------------------------------------------------------

    def steer(self, message: str) -> None:
        """Inject a steering message to be included in the next LLM call."""
        self._steering_queue.append(message)

    def follow_up(self, message: str) -> None:
        """Queue a follow-up user message to be processed after the current one."""
        self._followup_queue.append(message)

    def process_input(self, user_input: str) -> str:
        """Process a user input through the agentic loop.

        This is the core loop:
        1. Add user message to history
        2. Build LLM request from history
        3. Call LLM
        4. If tool calls: execute tools, add results, go to 2
        5. If no tool calls: return assistant text

        Returns the final assistant text response.
        """
        if self.state == SessionState.CLOSED:
            return "Session is closed."

        self.state = SessionState.PROCESSING
        self._emit(EventKind.USER_INPUT, {"input": user_input})

        # Add user turn
        self.history.append(UserTurn(content=user_input))

        try:
            result = self._run_loop()
        finally:
            self.state = SessionState.AWAITING_INPUT
            self._emit(EventKind.PROCESSING_END)

        # Process follow-up queue
        while self._followup_queue:
            followup = self._followup_queue.popleft()
            result = self.process_input(followup)

        return result

    def close(self) -> None:
        """Close the session."""
        self.state = SessionState.CLOSED
        self._emit(EventKind.SESSION_END)

    @property
    def total_usage(self) -> Usage:
        return self._total_usage

    @property
    def tool_rounds_used(self) -> int:
        """Number of tool rounds executed in the most recent process_input call."""
        return self._tool_rounds_used

    # -- internal loop -----------------------------------------------------

    def _run_loop(self) -> str:
        """Inner agentic loop: call LLM, execute tools, repeat."""
        rounds = 0
        self._tool_rounds_used = 0

        while rounds < self.config.max_tool_rounds_per_input:
            # Check total turn limit
            turn_count = sum(1 for t in self.history if isinstance(t, AssistantTurn))
            if turn_count >= self.config.max_turns:
                self._emit(EventKind.TURN_LIMIT, {"turns": turn_count})
                return "[Turn limit reached]"

            # Drain steering queue
            while self._steering_queue:
                steering_msg = self._steering_queue.popleft()
                self.history.append(SteeringTurn(content=steering_msg))
                self._emit(EventKind.STEERING_INJECTED, {"message": steering_msg})

            # Build and send LLM request
            request = self._build_request()
            response = self._client.complete(request)
            self._total_usage = self._total_usage + response.usage

            # Record assistant turn
            assistant_turn = AssistantTurn(
                message=response.message,
                usage=response.usage,
            )
            self.history.append(assistant_turn)

            cached = response.usage.cache_read_tokens
            cache_write = response.usage.cache_write_tokens
            # Vertex reports input_tokens as the non-cached/non-written portion;
            # the true total is input + cache_read + cache_write.
            total_input = response.usage.input_tokens + cached + cache_write
            if cached or cache_write:
                parts = []
                if cached:
                    parts.append("cached=%d" % cached)
                if cache_write:
                    parts.append("cache_write=%d" % cache_write)
                input_str = "input_tokens=%d (%s)" % (total_input, ", ".join(parts))
            else:
                input_str = "input_tokens=%d" % total_input
            logger.info(
                "LLM call: round=%d/%d \033[33m%s output_tokens=%d\033[0m",
                rounds + 1, self.config.max_tool_rounds_per_input,
                input_str, response.usage.output_tokens,
            )
            if response.raw:
                raw_usage = getattr(response.raw, "usage", None)
                if raw_usage:
                    logger.debug("  raw usage: %s", raw_usage)

            # Loop detection
            if self.config.enable_loop_detection:
                if detect_loop(self.history, self.config.loop_detection_window):
                    self._emit(EventKind.LOOP_DETECTION)
                    return "[Loop detected - stopping]"

            # Log response text preview
            if response.text:
                preview = response.text[:200].replace("\n", " ")
                if len(response.text) > 200:
                    preview += "…"
                logger.info("  response: \033[34m%s\033[0m", preview)

            # Check for tool calls
            if not response.has_tool_calls:
                self._emit(EventKind.ASSISTANT_TEXT_END, {"text": response.text})
                return response.text or ""

            # Log tool calls for this round
            for tc in response.tool_calls:
                try:
                    args = json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
                except Exception:
                    args = {}
                if tc.name == "shell" and "command" in args:
                    detail = args["command"][:120]
                elif tc.name == "read_file" and "path" in args:
                    detail = args["path"]
                elif tc.name == "write_file" and "path" in args:
                    detail = args["path"]
                else:
                    detail = str(args)[:120]
                logger.info("  \033[90mtool: %s — %s\033[0m", tc.name, detail)

            # Execute tool calls
            tool_results = self._execute_tool_calls(response.tool_calls)
            self.history.append(ToolResultsTurn(results=tool_results))

            rounds += 1
            self._tool_rounds_used = rounds

        return "[Tool round limit reached]"

    def _build_request(self) -> Request:
        """Build an LLM request from current history."""
        messages: list[Message] = []

        # System prompt
        if self._system_prompt:
            messages.append(Message.system(self._system_prompt))

        # Convert history to messages
        for turn in self.history:
            if isinstance(turn, UserTurn):
                messages.append(Message.user(turn.content))
            elif isinstance(turn, AssistantTurn):
                messages.append(turn.message)
            elif isinstance(turn, ToolResultsTurn):
                parts = [
                    ContentPart(
                        kind=ContentKind.TOOL_RESULT,
                        tool_result=ToolResultData(
                            tool_call_id=r.tool_call_id,
                            content=r.content,
                            is_error=r.is_error,
                        ),
                    )
                    for r in turn.results
                ]
                messages.append(Message(role=Role.USER, content=parts))
            elif isinstance(turn, SteeringTurn):
                messages.append(Message.user(f"[System guidance]: {turn.content}"))

        # Build tool definitions
        tools: list[LLMToolDefinition] | None = self._llm_tools
        if tools is None:
            tools = [
                LLMToolDefinition(
                    name=td.name,
                    description=td.description,
                    parameters=td.parameters,
                )
                for td in self._tool_registry.definitions
            ]

        return Request(
            model=self._llm_model,
            messages=messages,
            tools=tools if tools else None,
            reasoning_effort=self.config.reasoning_effort,
            provider_options=self._provider_options,
        )

    def _execute_tool_calls(
        self, tool_calls: list[Any],
    ) -> list[ToolResult]:
        """Execute a batch of tool calls and return results."""
        results: list[ToolResult] = []

        for tc in tool_calls:
            self._emit(EventKind.TOOL_CALL_START, {
                "tool": tc.name,
                "id": tc.id,
            })

            registered = self._tool_registry.get(tc.name)
            if registered is None:
                result = ToolResult(
                    tool_call_id=tc.id,
                    content=f"Error: Unknown tool '{tc.name}'",
                    is_error=True,
                )
            else:
                try:
                    # Parse arguments
                    args = tc.parsed_arguments if hasattr(tc, "parsed_arguments") else (
                        json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
                    )
                    output = registered.executor(args)
                    output = truncate_tool_output(output, tc.name, self.config)
                    result = ToolResult(
                        tool_call_id=tc.id,
                        content=output,
                    )
                except Exception as exc:
                    result = ToolResult(
                        tool_call_id=tc.id,
                        content=f"Error executing {tc.name}: {exc}",
                        is_error=True,
                    )

            self._emit(EventKind.TOOL_CALL_END, {
                "tool": tc.name,
                "id": tc.id,
                "is_error": result.is_error,
            })
            results.append(result)

        return results

    # -- events ------------------------------------------------------------

    def _emit(self, kind: EventKind, data: dict[str, Any] | None = None) -> None:
        self.event_emitter.emit(SessionEvent(
            kind=kind,
            session_id=self.id,
            data=data or {},
        ))
