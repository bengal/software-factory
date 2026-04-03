"""Anthropic-on-Vertex provider adapter."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

from attractor.llm.providers.anthropic import (
    AnthropicAdapter,
    HAS_ANTHROPIC,
    DEFAULT_MAX_TOKENS,
)

try:
    from anthropic import AnthropicVertex, AsyncAnthropicVertex
    HAS_VERTEX = True
except ImportError:
    HAS_VERTEX = False

PROVIDER_NAME = "vertex"
DEFAULT_MODEL = "claude-opus-4-6"


class VertexAdapter(AnthropicAdapter):
    """Adapter for Claude models hosted on Google Cloud Vertex AI.

    Reuses all Anthropic message formatting from AnthropicAdapter,
    but swaps the client for AnthropicVertex.
    """

    def __init__(
        self,
        project_id: str | None = None,
        region: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ):
        if not HAS_ANTHROPIC or not HAS_VERTEX:
            raise ImportError(
                "The 'anthropic' package with Vertex support is required. "
                "Install it with: pip install 'anthropic[vertex]'"
            )
        self._default_model = model or DEFAULT_MODEL
        self._default_max_tokens = max_tokens or DEFAULT_MAX_TOKENS

        self._project_id = project_id or os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", "")
        self._region = region or os.environ.get("CLOUD_ML_REGION", "us-east5")

        self._client = AnthropicVertex(
            project_id=self._project_id,
            region=self._region,
        )
        self._async_client = AsyncAnthropicVertex(
            project_id=self._project_id,
            region=self._region,
        )
