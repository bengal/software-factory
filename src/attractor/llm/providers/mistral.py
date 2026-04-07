"""Mistral provider adapter."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, AsyncIterator

from attractor.llm.providers.base import ProviderAdapter
from attractor.llm.types import (
    ContentKind,
    ContentPart,
    FinishReason,
    Message,
    Request,
    Response,
    Role,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolCallData,
    ToolDefinition,
    Usage,
)

try:
    from mistralai.client import Mistral
    HAS_MISTRAL = True
except ImportError:
    try:
        from mistralai import Mistral  # type: ignore[no-redef]
        HAS_MISTRAL = True
    except ImportError:
        Mistral = None  # type: ignore[assignment,misc]
        HAS_MISTRAL = False

PROVIDER_NAME = "mistral"
DEFAULT_MODEL = "mistral-large-latest"


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

_FINISH_MAP: dict[str, FinishReason] = {
    "stop": FinishReason.STOP,
    "tool_calls": FinishReason.TOOL_USE,
    "length": FinishReason.LENGTH,
    "model_length": FinishReason.LENGTH,
    "error": FinishReason.ERROR,
}


def _to_str(value: Any) -> str:
    """Convert a value that may be a string or an enum to a plain string."""
    if value is None:
        return ""
    return value.value if hasattr(value, "value") else str(value)


def _message_to_mistral(msg: Message) -> list[dict[str, Any]]:
    """Convert a unified Message into one or more Mistral message dicts."""
    if isinstance(msg.content, str):
        role = msg.role.value
        # Mistral has no "developer" role; map to "system".
        if msg.role == Role.DEVELOPER:
            role = "system"
        return [{"role": role, "content": msg.content}]

    result: list[dict[str, Any]] = []
    text_parts: list[str] = []
    tool_calls_list: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []

    for part in msg.content:
        if part.kind == ContentKind.TEXT and part.text:
            text_parts.append(part.text)
        elif part.kind == ContentKind.TOOL_CALL and part.tool_call:
            tc = part.tool_call
            tool_calls_list.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments,
                },
            })
        elif part.kind == ContentKind.TOOL_RESULT and part.tool_result:
            tr = part.tool_result
            tool_results.append({
                "role": "tool",
                "tool_call_id": tr.tool_call_id,
                "content": tr.content,
            })

    # Emit assistant message with optional text + tool_calls
    if text_parts or tool_calls_list:
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        assistant_msg["content"] = "".join(text_parts) if text_parts else ""
        if tool_calls_list:
            assistant_msg["tool_calls"] = tool_calls_list
        result.append(assistant_msg)

    # Emit tool results
    result.extend(tool_results)
    return result


def _tool_def_to_mistral(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _ensure_str_args(arguments: Any) -> str:
    """Ensure tool call arguments are a JSON string."""
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments) if arguments else "{}"


def _parse_response_message(resp: Any) -> tuple[list[ContentPart], FinishReason]:
    """Parse a Mistral ChatCompletion response into ContentParts."""
    choice = resp.choices[0]
    msg = choice.message
    parts: list[ContentPart] = []

    if msg.content:
        parts.append(ContentPart(kind=ContentKind.TEXT, text=msg.content))

    if msg.tool_calls:
        for tc in msg.tool_calls:
            parts.append(ContentPart(
                kind=ContentKind.TOOL_CALL,
                tool_call=ToolCallData(
                    id=tc.id or str(uuid.uuid4()),
                    name=tc.function.name,
                    arguments=_ensure_str_args(tc.function.arguments),
                ),
            ))

    finish = _FINISH_MAP.get(_to_str(choice.finish_reason), FinishReason.UNKNOWN)
    return parts, finish


def _parse_usage(resp: Any) -> Usage:
    if not resp.usage:
        return Usage()
    u = resp.usage
    inp = getattr(u, "prompt_tokens", 0) or 0
    out = getattr(u, "completion_tokens", 0) or 0
    total = getattr(u, "total_tokens", 0) or (inp + out)
    return Usage(
        input_tokens=inp,
        output_tokens=out,
        total_tokens=total,
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class MistralAdapter(ProviderAdapter):
    """Adapter for the Mistral Chat Completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        if not HAS_MISTRAL:
            raise ImportError(
                "The 'mistralai' package is required. "
                "Install it with: pip install mistralai"
            )
        self._api_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        self._default_model = model or DEFAULT_MODEL
        self._client = Mistral(api_key=self._api_key, timeout_ms=600_000)

    # -- sync --------------------------------------------------------------

    def complete(self, request: Request) -> Response:
        model = request.model or self._default_model
        kwargs = self._build_kwargs(request, model)
        raw = self._client.chat.complete(**kwargs)
        return self._parse_response(raw, model)

    # -- async stream ------------------------------------------------------

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        model = request.model or self._default_model
        kwargs = self._build_kwargs(request, model)

        stream = await self._client.chat.stream_async(**kwargs)

        current_tool_call_id = ""
        current_tool_name = ""
        text_started = False

        async for event in stream:
            chunk = event.data
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # Text content
            if delta.content:
                if not text_started:
                    yield StreamEvent(type=StreamEventType.CONTENT_START)
                    text_started = True
                yield StreamEvent(
                    type=StreamEventType.CONTENT_DELTA,
                    text=delta.content,
                )

            # Tool calls
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    if tc_delta.id:
                        current_tool_call_id = tc_delta.id
                        current_tool_name = (
                            tc_delta.function.name if tc_delta.function else ""
                        )
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL_START,
                            tool_call=ToolCall(
                                id=current_tool_call_id,
                                name=current_tool_name,
                                arguments="",
                            ),
                        )
                    if tc_delta.function and tc_delta.function.arguments:
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL_DELTA,
                            text=_ensure_str_args(tc_delta.function.arguments),
                        )

            # Finish
            finish_reason = chunk.choices[0].finish_reason
            if finish_reason is not None:
                if text_started:
                    yield StreamEvent(type=StreamEventType.CONTENT_STOP)
                finish = _FINISH_MAP.get(
                    _to_str(finish_reason), FinishReason.UNKNOWN
                )
                yield StreamEvent(
                    type=StreamEventType.MESSAGE_STOP,
                    finish_reason=finish,
                )

    # -- internals ---------------------------------------------------------

    def _build_kwargs(self, request: Request, model: str) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        for msg in request.messages:
            messages.extend(_message_to_mistral(msg))

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.stop_sequences:
            kwargs["stop"] = request.stop_sequences

        # Tools
        if request.tools:
            kwargs["tools"] = [_tool_def_to_mistral(t) for t in request.tools]
        if request.tool_choice is not None:
            if isinstance(request.tool_choice, str):
                kwargs["tool_choice"] = request.tool_choice
            else:
                kwargs["tool_choice"] = request.tool_choice

        # Response format
        if request.response_format:
            if request.response_format.type == "json_object":
                kwargs["response_format"] = {"type": "json_object"}

        # Provider-specific pass-through
        for key, val in request.provider_options.items():
            kwargs[key] = val

        return kwargs

    def _parse_response(self, raw: Any, model: str) -> Response:
        parts, finish = _parse_response_message(raw)
        usage = _parse_usage(raw)
        return Response(
            id=raw.id or str(uuid.uuid4()),
            model=raw.model or model,
            provider=PROVIDER_NAME,
            message=Message(role=Role.ASSISTANT, content=parts),
            finish_reason=finish,
            usage=usage,
            raw=raw,
        )
