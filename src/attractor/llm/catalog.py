"""Model catalog: metadata about known LLM models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for a known model."""
    id: str
    provider: str
    display_name: str
    context_window: int
    max_output_tokens: int
    supports_tools: bool = True
    supports_vision: bool = False
    supports_thinking: bool = False
    is_latest: bool = False


# ---------------------------------------------------------------------------
# Hard-coded catalog
# ---------------------------------------------------------------------------

_CATALOG: dict[str, ModelInfo] = {}


def _register(*models: ModelInfo) -> None:
    for m in models:
        _CATALOG[m.id] = m


# -- Anthropic ---------------------------------------------------------------

_register(
    ModelInfo(
        id="claude-opus-4-20250514",
        provider="anthropic",
        display_name="Claude Opus 4",
        context_window=200_000,
        max_output_tokens=32_768,
        supports_vision=True,
        supports_thinking=True,
        is_latest=False,
    ),
    ModelInfo(
        id="claude-sonnet-4-20250514",
        provider="anthropic",
        display_name="Claude Sonnet 4",
        context_window=200_000,
        max_output_tokens=16_384,
        supports_vision=True,
        supports_thinking=True,
        is_latest=True,
    ),
    ModelInfo(
        id="claude-3-5-haiku-20241022",
        provider="anthropic",
        display_name="Claude 3.5 Haiku",
        context_window=200_000,
        max_output_tokens=8_192,
        supports_vision=True,
        supports_thinking=False,
    ),
    ModelInfo(
        id="claude-3-5-sonnet-20241022",
        provider="anthropic",
        display_name="Claude 3.5 Sonnet v2",
        context_window=200_000,
        max_output_tokens=8_192,
        supports_vision=True,
        supports_thinking=False,
    ),
)

# -- OpenAI ------------------------------------------------------------------

_register(
    ModelInfo(
        id="gpt-4o",
        provider="openai",
        display_name="GPT-4o",
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        is_latest=True,
    ),
    ModelInfo(
        id="gpt-4o-mini",
        provider="openai",
        display_name="GPT-4o Mini",
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
    ),
    ModelInfo(
        id="o3",
        provider="openai",
        display_name="o3",
        context_window=200_000,
        max_output_tokens=100_000,
        supports_vision=True,
        supports_thinking=True,
    ),
    ModelInfo(
        id="o3-mini",
        provider="openai",
        display_name="o3-mini",
        context_window=200_000,
        max_output_tokens=100_000,
        supports_thinking=True,
    ),
    ModelInfo(
        id="o4-mini",
        provider="openai",
        display_name="o4-mini",
        context_window=200_000,
        max_output_tokens=100_000,
        supports_vision=True,
        supports_thinking=True,
    ),
)

# -- Gemini ------------------------------------------------------------------

_register(
    ModelInfo(
        id="gemini-2.5-pro",
        provider="gemini",
        display_name="Gemini 2.5 Pro",
        context_window=1_048_576,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_thinking=True,
        is_latest=True,
    ),
    ModelInfo(
        id="gemini-2.5-flash",
        provider="gemini",
        display_name="Gemini 2.5 Flash",
        context_window=1_048_576,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_thinking=True,
    ),
    ModelInfo(
        id="gemini-2.0-flash",
        provider="gemini",
        display_name="Gemini 2.0 Flash",
        context_window=1_048_576,
        max_output_tokens=8_192,
        supports_vision=True,
    ),
)

# -- Mistral -----------------------------------------------------------------

_register(
    ModelInfo(
        id="mistral-large-latest",
        provider="mistral",
        display_name="Mistral Large",
        context_window=128_000,
        max_output_tokens=8_192,
        supports_vision=False,
        is_latest=True,
    ),
    ModelInfo(
        id="mistral-small-latest",
        provider="mistral",
        display_name="Mistral Small",
        context_window=128_000,
        max_output_tokens=8_192,
        supports_vision=False,
    ),
    ModelInfo(
        id="codestral-latest",
        provider="mistral",
        display_name="Codestral",
        context_window=256_000,
        max_output_tokens=8_192,
        supports_vision=False,
    ),
    ModelInfo(
        id="devstral-latest",
        provider="mistral",
        display_name="Devstral",
        context_window=256_000,
        max_output_tokens=8_192,
        supports_vision=False,
    ),
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_model_info(model_id: str) -> ModelInfo | None:
    """Look up metadata for a model by its ID. Returns None if unknown."""
    return _CATALOG.get(model_id)


def list_models(provider: str | None = None) -> list[ModelInfo]:
    """List known models, optionally filtered by provider."""
    models = list(_CATALOG.values())
    if provider:
        models = [m for m in models if m.provider == provider]
    return models


def get_latest_model(provider: str) -> ModelInfo | None:
    """Return the model marked ``is_latest`` for a provider, or None."""
    for m in _CATALOG.values():
        if m.provider == provider and m.is_latest:
            return m
    return None
