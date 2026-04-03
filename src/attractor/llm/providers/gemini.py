"""Google Gemini provider adapter using the native Gemini API."""

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
    from google import genai
    from google.genai import types as genai_types
    HAS_GEMINI = True
except ImportError:
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]
    HAS_GEMINI = False

PROVIDER_NAME = "gemini"
DEFAULT_MODEL = "gemini-2.5-pro"


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

def _extract_system(messages: list[Message]) -> str:
    """Join all system-message text into a single string."""
    parts: list[str] = []
    for msg in messages:
        if msg.role != Role.SYSTEM:
            continue
        if isinstance(msg.content, str):
            parts.append(msg.content)
        else:
            for p in msg.content:
                if p.kind == ContentKind.TEXT and p.text:
                    parts.append(p.text)
    return "\n\n".join(parts)


def _message_to_gemini(msg: Message) -> dict[str, Any] | None:
    """Convert a unified Message to a Gemini content dict."""
    if msg.role == Role.SYSTEM:
        return None

    role = "user" if msg.role in (Role.USER, Role.TOOL) else "model"

    if isinstance(msg.content, str):
        return {"role": role, "parts": [{"text": msg.content}]}

    parts: list[dict[str, Any]] = []
    for p in msg.content:
        if p.kind == ContentKind.TEXT and p.text:
            parts.append({"text": p.text})
        elif p.kind == ContentKind.TOOL_CALL and p.tool_call:
            tc = p.tool_call
            args = json.loads(tc.arguments) if isinstance(tc.arguments, str) and tc.arguments else {}
            parts.append({"function_call": {"name": tc.name, "args": args}})
        elif p.kind == ContentKind.TOOL_RESULT and p.tool_result:
            tr = p.tool_result
            parts.append({
                "function_response": {
                    "name": tr.tool_call_id,
                    "response": {"result": tr.content},
                },
            })
        elif p.kind == ContentKind.IMAGE and p.image and p.image.data:
            parts.append({
                "inline_data": {
                    "mime_type": p.image.media_type,
                    "data": p.image.data,
                },
            })

    if not parts:
        return None
    return {"role": role, "parts": parts}


def _tool_def_to_gemini(tool: ToolDefinition) -> Any:
    """Convert a ToolDefinition into a Gemini FunctionDeclaration."""
    return genai_types.FunctionDeclaration(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
    )


def _parse_candidate(candidate: Any) -> tuple[list[ContentPart], FinishReason]:
    """Parse a Gemini candidate into ContentParts and a FinishReason."""
    parts: list[ContentPart] = []
    finish = FinishReason.STOP

    if candidate.content and candidate.content.parts:
        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                parts.append(ContentPart(kind=ContentKind.TEXT, text=part.text))
            elif hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                args = dict(fc.args) if fc.args else {}
                parts.append(ContentPart(
                    kind=ContentKind.TOOL_CALL,
                    tool_call=ToolCallData(
                        id=str(uuid.uuid4()),
                        name=fc.name,
                        arguments=json.dumps(args),
                    ),
                ))
                finish = FinishReason.TOOL_USE

    # Refine from candidate's own finish_reason enum when available.
    fr = getattr(candidate, "finish_reason", None)
    if fr is not None:
        fr_str = str(fr).lower()
        if "max_tokens" in fr_str or "length" in fr_str:
            finish = FinishReason.LENGTH
        elif "safety" in fr_str:
            finish = FinishReason.CONTENT_FILTER

    return parts, finish


def _parse_usage_metadata(resp: Any) -> Usage:
    """Parse Gemini usage metadata into a Usage record."""
    if not hasattr(resp, "usage_metadata") or not resp.usage_metadata:
        return Usage()
    um = resp.usage_metadata
    inp = getattr(um, "prompt_token_count", 0) or 0
    out = getattr(um, "candidates_token_count", 0) or 0
    thinking = getattr(um, "thinking_token_count", 0) or 0
    return Usage(
        input_tokens=inp,
        output_tokens=out,
        total_tokens=inp + out,
        reasoning_tokens=thinking,
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class GeminiAdapter(ProviderAdapter):
    """Adapter for the Google Gemini API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        if not HAS_GEMINI:
            raise ImportError(
                "The 'google-genai' package is required. Install it with: pip install google-genai"
            )
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._default_model = model or DEFAULT_MODEL
        self._client = genai.Client(api_key=self._api_key)

    # -- sync --------------------------------------------------------------

    def complete(self, request: Request) -> Response:
        model = request.model or self._default_model
        kwargs = self._build_config_kwargs(request)
        contents = self._build_contents(request)
        config = genai_types.GenerateContentConfig(**kwargs)

        raw = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        return self._parse_response(raw, model)

    # -- async stream ------------------------------------------------------

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        model = request.model or self._default_model
        kwargs = self._build_config_kwargs(request)
        contents = self._build_contents(request)
        config = genai_types.GenerateContentConfig(**kwargs)

        yield StreamEvent(type=StreamEventType.MESSAGE_START)

        for chunk in self._client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        ):
            if not chunk.candidates:
                continue
            candidate = chunk.candidates[0]
            if not (candidate.content and candidate.content.parts):
                continue
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    yield StreamEvent(
                        type=StreamEventType.CONTENT_DELTA,
                        text=part.text,
                    )
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    args = dict(fc.args) if fc.args else {}
                    yield StreamEvent(
                        type=StreamEventType.TOOL_CALL_START,
                        tool_call=ToolCall(
                            id=str(uuid.uuid4()),
                            name=fc.name,
                            arguments=json.dumps(args),
                        ),
                    )

        yield StreamEvent(type=StreamEventType.MESSAGE_STOP)

    # -- internals ---------------------------------------------------------

    def _build_config_kwargs(self, request: Request) -> dict[str, Any]:
        system_text = _extract_system(request.messages)

        tools = None
        if request.tools:
            declarations = [_tool_def_to_gemini(t) for t in request.tools]
            tools = [genai_types.Tool(function_declarations=declarations)]

        kwargs: dict[str, Any] = {}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.max_tokens is not None:
            kwargs["max_output_tokens"] = request.max_tokens
        if request.stop_sequences:
            kwargs["stop_sequences"] = request.stop_sequences
        if tools:
            kwargs["tools"] = tools
        if system_text:
            kwargs["system_instruction"] = system_text
        if request.reasoning_effort:
            kwargs["thinking_config"] = genai_types.ThinkingConfig(thinking_budget=-1)

        return kwargs

    def _build_contents(self, request: Request) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for msg in request.messages:
            converted = _message_to_gemini(msg)
            if converted:
                contents.append(converted)
        return contents

    def _parse_response(self, raw: Any, model: str) -> Response:
        parts: list[ContentPart] = []
        finish = FinishReason.STOP

        if raw.candidates:
            parts, finish = _parse_candidate(raw.candidates[0])

        usage = _parse_usage_metadata(raw)

        return Response(
            id=str(uuid.uuid4()),
            model=model,
            provider=PROVIDER_NAME,
            message=Message(role=Role.ASSISTANT, content=parts),
            finish_reason=finish,
            usage=usage,
            raw=raw,
        )


# Backward-compatible aliases
GeminiProvider = GeminiAdapter
