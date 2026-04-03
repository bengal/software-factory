"""Unified LLM client abstraction layer."""

from attractor.llm.catalog import ModelInfo, get_latest_model, get_model_info, list_models
from attractor.llm.client import Client, Middleware
from attractor.llm.providers.base import ProviderAdapter
from attractor.llm.types import (
    ContentKind,
    ContentPart,
    FinishReason,
    ImageData,
    Message,
    RateLimitInfo,
    Request,
    Response,
    ResponseFormat,
    Role,
    StreamEvent,
    StreamEventType,
    ThinkingData,
    ToolCall,
    ToolCallData,
    ToolDefinition,
    ToolResult,
    ToolResultData,
    Usage,
    Warning,
)

__all__ = [
    # Enums
    "Role",
    "ContentKind",
    "FinishReason",
    "StreamEventType",
    # Data records
    "ImageData",
    "ToolCallData",
    "ToolResultData",
    "ThinkingData",
    "ContentPart",
    "Message",
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
    "ResponseFormat",
    "Request",
    "Usage",
    "RateLimitInfo",
    "Warning",
    "Response",
    "StreamEvent",
    # Client & providers
    "Client",
    "Middleware",
    "ProviderAdapter",
    # Catalog
    "ModelInfo",
    "get_model_info",
    "list_models",
    "get_latest_model",
]
