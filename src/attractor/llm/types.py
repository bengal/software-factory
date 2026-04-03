"""Unified LLM Client types."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Role(str, Enum):
    """Message role."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    DEVELOPER = "developer"


class ContentKind(str, Enum):
    """Kind of content within a message part."""
    TEXT = "text"
    IMAGE = "image"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    REDACTED_THINKING = "redacted_thinking"


class FinishReason(str, Enum):
    """Why the model stopped generating."""
    STOP = "stop"
    LENGTH = "length"
    TOOL_USE = "tool_use"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    UNKNOWN = "unknown"


class StreamEventType(str, Enum):
    """Type of event in a streaming response."""
    CONTENT_START = "content_start"
    CONTENT_DELTA = "content_delta"
    CONTENT_STOP = "content_stop"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_STOP = "tool_call_stop"
    THINKING_START = "thinking_start"
    THINKING_DELTA = "thinking_delta"
    THINKING_STOP = "thinking_stop"
    MESSAGE_START = "message_start"
    MESSAGE_STOP = "message_stop"
    ERROR = "error"
    USAGE = "usage"


# ---------------------------------------------------------------------------
# Supporting data records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageData:
    """Image content, either base64-encoded or a URL."""
    media_type: str
    data: str | None = None
    url: str | None = None


@dataclass(frozen=True)
class ToolCallData:
    """Data embedded in a tool_call content part."""
    id: str
    name: str
    arguments: str  # JSON string


@dataclass(frozen=True)
class ToolResultData:
    """Data embedded in a tool_result content part."""
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class ThinkingData:
    """Data for a thinking content part."""
    text: str
    redacted: bool = False


@dataclass(frozen=True)
class ContentPart:
    """A single content part within a message."""
    kind: ContentKind
    text: str | None = None
    image: ImageData | None = None
    tool_call: ToolCallData | None = None
    tool_result: ToolResultData | None = None
    thinking: ThinkingData | None = None
    cache_control: dict[str, str] | None = None


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single message in a conversation."""
    role: Role
    content: list[ContentPart] | str
    name: str | None = None

    @staticmethod
    def system(text: str) -> Message:
        """Create a system message."""
        return Message(role=Role.SYSTEM, content=text)

    @staticmethod
    def user(text: str) -> Message:
        """Create a user message."""
        return Message(role=Role.USER, content=text)

    @staticmethod
    def assistant(text: str) -> Message:
        """Create an assistant message."""
        return Message(role=Role.ASSISTANT, content=text)

    @staticmethod
    def tool_result(tool_call_id: str, content: str, is_error: bool = False) -> Message:
        """Create a tool-result message."""
        return Message(
            role=Role.TOOL,
            content=[
                ContentPart(
                    kind=ContentKind.TOOL_RESULT,
                    tool_result=ToolResultData(
                        tool_call_id=tool_call_id,
                        content=content,
                        is_error=is_error,
                    ),
                )
            ],
        )

    @property
    def text(self) -> str | None:
        """Concatenated text of all TEXT content parts."""
        if isinstance(self.content, str):
            return self.content
        parts = [p.text for p in self.content if p.kind == ContentKind.TEXT and p.text]
        return "".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Tool-related records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolDefinition:
    """Schema definition for a tool the model can call."""
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation from the model."""
    id: str
    name: str
    arguments: str  # JSON string
    type: str = "function"

    @property
    def parsed_arguments(self) -> dict[str, Any]:
        """Parse the JSON arguments string."""
        return json.loads(self.arguments) if self.arguments else {}


@dataclass(frozen=True)
class ToolResult:
    """Result to feed back for a tool call."""
    tool_call_id: str
    content: str
    is_error: bool = False


# ---------------------------------------------------------------------------
# Response format
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResponseFormat:
    """Desired response format."""
    type: str = "text"  # "text" | "json_object" | "json_schema"
    json_schema: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

@dataclass
class Request:
    """Unified request to any LLM provider."""
    model: str
    messages: list[Message]
    provider: str | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: ResponseFormat | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop_sequences: list[str] | None = None
    reasoning_effort: str | None = None  # "low" | "medium" | "high"
    metadata: dict[str, Any] = field(default_factory=dict)
    provider_options: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

@dataclass
class Usage:
    """Token usage stats."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


# ---------------------------------------------------------------------------
# Rate limit & warnings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RateLimitInfo:
    """Rate-limit metadata from a provider response."""
    requests_remaining: int | None = None
    tokens_remaining: int | None = None
    reset_at: str | None = None  # ISO-8601


@dataclass(frozen=True)
class Warning:
    """A warning returned alongside a response."""
    code: str
    message: str


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

@dataclass
class Response:
    """Unified response from any LLM provider."""
    id: str
    model: str
    provider: str
    message: Message
    finish_reason: FinishReason
    usage: Usage
    raw: Any = None
    rate_limit: RateLimitInfo | None = None
    warnings: list[Warning] = field(default_factory=list)

    @property
    def text(self) -> str | None:
        """Extract the plain-text response."""
        return self.message.text

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Extract tool calls from the assistant message."""
        if isinstance(self.message.content, str):
            return []
        calls: list[ToolCall] = []
        for part in self.message.content:
            if part.kind == ContentKind.TOOL_CALL and part.tool_call:
                calls.append(
                    ToolCall(
                        id=part.tool_call.id,
                        name=part.tool_call.name,
                        arguments=part.tool_call.arguments,
                    )
                )
        return calls

    @property
    def has_tool_calls(self) -> bool:
        """Whether the response contains any tool calls."""
        return len(self.tool_calls) > 0

    @property
    def reasoning(self) -> str | None:
        """Extract thinking/reasoning text."""
        if isinstance(self.message.content, str):
            return None
        parts = [
            p.thinking.text
            for p in self.message.content
            if p.kind == ContentKind.THINKING
            and p.thinking is not None
            and not p.thinking.redacted
        ]
        return "".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Stream events
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    """A single event from a streaming response."""
    type: StreamEventType
    text: str | None = None
    tool_call: ToolCall | None = None
    thinking: str | None = None
    usage: Usage | None = None
    finish_reason: FinishReason | None = None
    response: Response | None = None
    error: str | None = None
