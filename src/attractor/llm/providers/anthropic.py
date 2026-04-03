"""Anthropic provider adapter using the Messages API."""

from __future__ import annotations

import json
import os
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
    ThinkingData,
    ToolCall,
    ToolCallData,
    ToolDefinition,
    Usage,
)

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    anthropic = None  # type: ignore[assignment]
    HAS_ANTHROPIC = False

PROVIDER_NAME = "anthropic"
DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_MAX_TOKENS = 16384


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

_FINISH_MAP: dict[str, FinishReason] = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "max_tokens": FinishReason.LENGTH,
    "tool_use": FinishReason.TOOL_USE,
}


def _map_finish_reason(stop_reason: str | None) -> FinishReason:
    return _FINISH_MAP.get(stop_reason or "", FinishReason.UNKNOWN)


def _build_system(messages: list[Message]) -> str | list[dict[str, Any]]:
    """Extract system messages into the Anthropic ``system`` parameter."""
    parts: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role != Role.SYSTEM:
            continue
        if isinstance(msg.content, str):
            parts.append({"type": "text", "text": msg.content})
        else:
            for cp in msg.content:
                if cp.kind == ContentKind.TEXT and cp.text:
                    p: dict[str, Any] = {"type": "text", "text": cp.text}
                    if cp.cache_control:
                        p["cache_control"] = cp.cache_control
                    parts.append(p)
    if not parts:
        return ""
    # Simple case: single text-only system prompt -> plain string.
    if len(parts) == 1 and "cache_control" not in parts[0]:
        return parts[0]["text"]
    return parts


def _message_to_anthropic(msg: Message) -> dict[str, Any]:
    """Convert a unified Message to Anthropic wire format."""
    # Anthropic requires tool results under the "user" role.
    role = "user" if msg.role == Role.TOOL else msg.role.value

    if isinstance(msg.content, str):
        return {"role": role, "content": msg.content}

    blocks: list[dict[str, Any]] = []
    for part in msg.content:
        block: dict[str, Any]
        if part.kind == ContentKind.TEXT:
            block = {"type": "text", "text": part.text or ""}
        elif part.kind == ContentKind.IMAGE and part.image:
            if part.image.data:
                block = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": part.image.media_type,
                        "data": part.image.data,
                    },
                }
            else:
                block = {
                    "type": "image",
                    "source": {"type": "url", "url": part.image.url},
                }
        elif part.kind == ContentKind.TOOL_CALL and part.tool_call:
            raw_args = part.tool_call.arguments
            parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            block = {
                "type": "tool_use",
                "id": part.tool_call.id,
                "name": part.tool_call.name,
                "input": parsed,
            }
        elif part.kind == ContentKind.TOOL_RESULT and part.tool_result:
            block = {
                "type": "tool_result",
                "tool_use_id": part.tool_result.tool_call_id,
                "content": part.tool_result.content,
            }
            if part.tool_result.is_error:
                block["is_error"] = True
        elif part.kind == ContentKind.THINKING and part.thinking:
            block = {"type": "thinking", "thinking": part.thinking.text}
        elif part.kind == ContentKind.REDACTED_THINKING:
            block = {"type": "redacted_thinking"}
        else:
            continue

        if part.cache_control:
            block["cache_control"] = part.cache_control
        blocks.append(block)

    return {"role": role, "content": blocks}


def _tool_def_to_anthropic(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    }


def _parse_content_blocks(blocks: list[Any]) -> list[ContentPart]:
    """Parse Anthropic SDK content blocks into unified ContentParts."""
    parts: list[ContentPart] = []
    for block in blocks:
        btype = getattr(block, "type", None)
        if isinstance(block, dict):
            btype = block.get("type")

        if btype == "text":
            text = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else "")
            parts.append(ContentPart(kind=ContentKind.TEXT, text=text))
        elif btype == "tool_use":
            bid = getattr(block, "id", "") or (block.get("id", "") if isinstance(block, dict) else "")
            bname = getattr(block, "name", "") or (block.get("name", "") if isinstance(block, dict) else "")
            binput = getattr(block, "input", {}) or (block.get("input", {}) if isinstance(block, dict) else {})
            args = json.dumps(binput) if not isinstance(binput, str) else binput
            parts.append(ContentPart(
                kind=ContentKind.TOOL_CALL,
                tool_call=ToolCallData(id=bid, name=bname, arguments=args),
            ))
        elif btype == "thinking":
            text = getattr(block, "thinking", "") or (block.get("thinking", "") if isinstance(block, dict) else "")
            parts.append(ContentPart(
                kind=ContentKind.THINKING,
                thinking=ThinkingData(text=text),
            ))
        elif btype == "redacted_thinking":
            parts.append(ContentPart(
                kind=ContentKind.REDACTED_THINKING,
                thinking=ThinkingData(text="", redacted=True),
            ))
    return parts


def _parse_usage(raw_usage: Any) -> Usage:
    inp = getattr(raw_usage, "input_tokens", 0) or 0
    out = getattr(raw_usage, "output_tokens", 0) or 0
    return Usage(
        input_tokens=inp,
        output_tokens=out,
        total_tokens=inp + out,
        cache_read_tokens=getattr(raw_usage, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(raw_usage, "cache_creation_input_tokens", 0) or 0,
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class AnthropicAdapter(ProviderAdapter):
    """Adapter for the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ):
        if not HAS_ANTHROPIC:
            raise ImportError(
                "The 'anthropic' package is required. Install it with: pip install anthropic"
            )
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._default_model = model or DEFAULT_MODEL
        self._default_max_tokens = max_tokens or DEFAULT_MAX_TOKENS
        self._client = anthropic.Anthropic(api_key=self._api_key)
        self._async_client = anthropic.AsyncAnthropic(api_key=self._api_key)

    # -- sync --------------------------------------------------------------

    def complete(self, request: Request) -> Response:
        model = request.model or self._default_model
        kwargs = self._build_kwargs(request, model)
        raw = self._client.messages.create(**kwargs)
        return self._parse_response(raw, model)

    # -- async stream ------------------------------------------------------

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        model = request.model or self._default_model
        kwargs = self._build_kwargs(request, model)

        async with self._async_client.messages.stream(**kwargs) as stream:
            async for event in stream:
                mapped = self._map_stream_event(event)
                if mapped is not None:
                    yield mapped

            final = await stream.get_final_message()
            response = self._parse_response(final, model)
            yield StreamEvent(
                type=StreamEventType.MESSAGE_STOP,
                response=response,
                usage=response.usage,
                finish_reason=response.finish_reason,
            )

    # -- internals ---------------------------------------------------------

    def _build_kwargs(self, request: Request, model: str) -> dict[str, Any]:
        non_system = [m for m in request.messages if m.role != Role.SYSTEM]
        messages = [_message_to_anthropic(m) for m in non_system]
        system = _build_system(request.messages)

        # Mark cache breakpoints for prompt caching (up to 4 allowed):
        # 1. System prompt (stable across all rounds)
        if isinstance(system, str) and system:
            system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        elif isinstance(system, list) and system:
            system[-1] = {**system[-1], "cache_control": {"type": "ephemeral"}}

        user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]

        # 2. First user message (contains the initial prompt with the full spec;
        #    never changes between rounds, so caching it avoids re-tokenizing
        #    the spec on every LLM call)
        if user_indices:
            idx = user_indices[0]
            msg = messages[idx]
            if isinstance(msg["content"], list) and msg["content"]:
                msg["content"][-1] = {**msg["content"][-1], "cache_control": {"type": "ephemeral"}}
            elif isinstance(msg["content"], str):
                messages[idx] = {
                    **msg,
                    "content": [{"type": "text", "text": msg["content"], "cache_control": {"type": "ephemeral"}}],
                }

        # 3. Second-to-last user/tool message (the conversation prefix that
        #    doesn't change between rounds). The last message is the new one.
        if len(user_indices) >= 3:
            idx = user_indices[-2]
            msg = messages[idx]
            if isinstance(msg["content"], list) and msg["content"]:
                msg["content"][-1] = {**msg["content"][-1], "cache_control": {"type": "ephemeral"}}
            elif isinstance(msg["content"], str):
                messages[idx] = {
                    **msg,
                    "content": [{"type": "text", "text": msg["content"], "cache_control": {"type": "ephemeral"}}],
                }

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens or self._default_max_tokens,
        }
        if system:
            kwargs["system"] = system
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.stop_sequences:
            kwargs["stop_sequences"] = request.stop_sequences

        # Tools
        if request.tools:
            kwargs["tools"] = [_tool_def_to_anthropic(t) for t in request.tools]
        if request.tool_choice is not None:
            if isinstance(request.tool_choice, str):
                kwargs["tool_choice"] = {"type": request.tool_choice}
            else:
                kwargs["tool_choice"] = request.tool_choice

        # Extended thinking
        if request.reasoning_effort:
            effort_map = {"low": 1024, "medium": 8192, "high": 32768}
            budget = effort_map.get(request.reasoning_effort, 8192)
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}

        # Provider-specific pass-through options
        for key in ("metadata", "betas"):
            val = request.provider_options.get(key)
            if val is not None:
                kwargs[key] = val

        return kwargs

    def _parse_response(self, raw: Any, model: str) -> Response:
        content_parts = _parse_content_blocks(raw.content)
        usage = _parse_usage(raw.usage)
        return Response(
            id=raw.id,
            model=raw.model or model,
            provider=PROVIDER_NAME,
            message=Message(role=Role.ASSISTANT, content=content_parts),
            finish_reason=_map_finish_reason(raw.stop_reason),
            usage=usage,
            raw=raw,
        )

    @staticmethod
    def _map_stream_event(event: Any) -> StreamEvent | None:
        etype = getattr(event, "type", "")
        if etype == "content_block_start":
            block = getattr(event, "content_block", None)
            if block and getattr(block, "type", "") == "tool_use":
                return StreamEvent(
                    type=StreamEventType.TOOL_CALL_START,
                    tool_call=ToolCall(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        arguments="",
                    ),
                )
            if block and getattr(block, "type", "") == "thinking":
                return StreamEvent(type=StreamEventType.THINKING_START)
            return StreamEvent(type=StreamEventType.CONTENT_START)
        elif etype == "content_block_delta":
            delta = getattr(event, "delta", None)
            if delta:
                dtype = getattr(delta, "type", "")
                if dtype == "text_delta":
                    return StreamEvent(
                        type=StreamEventType.CONTENT_DELTA,
                        text=getattr(delta, "text", ""),
                    )
                elif dtype == "input_json_delta":
                    return StreamEvent(
                        type=StreamEventType.TOOL_CALL_DELTA,
                        text=getattr(delta, "partial_json", ""),
                    )
                elif dtype == "thinking_delta":
                    return StreamEvent(
                        type=StreamEventType.THINKING_DELTA,
                        thinking=getattr(delta, "thinking", ""),
                    )
        elif etype == "content_block_stop":
            return StreamEvent(type=StreamEventType.CONTENT_STOP)
        elif etype == "message_start":
            return StreamEvent(type=StreamEventType.MESSAGE_START)
        return None


# Backward-compatible alias
AnthropicProvider = AnthropicAdapter
