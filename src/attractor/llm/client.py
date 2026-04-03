"""Unified LLM client with provider registry and middleware support."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Callable

from attractor.llm.providers.base import ProviderAdapter
from attractor.llm.types import Request, Response, StreamEvent

# Middleware: a callable that wraps a provider's complete() call.
# It receives (request, next_fn) where next_fn calls the actual provider.
Middleware = Callable[[Request, Callable[[Request], Response]], Response]


class Client:
    """Unified LLM client that delegates to registered provider adapters.

    Supports a provider registry, a configurable default provider, and a
    middleware stack that wraps every ``complete()`` call.
    """

    def __init__(
        self,
        default_provider: str | None = None,
        model: str | None = None,
    ):
        self._providers: dict[str, ProviderAdapter] = {}
        self._default_provider = default_provider or os.environ.get("LLM_PROVIDER", "anthropic")
        self._default_model = model
        self._middleware: list[Middleware] = []

    # -- registry ----------------------------------------------------------

    def register_provider(self, name: str, adapter: ProviderAdapter) -> None:
        """Register a provider adapter under *name*."""
        self._providers[name] = adapter

    def get_provider(self, name: str | None = None) -> ProviderAdapter:
        """Return the adapter for *name*, lazily creating built-in ones."""
        name = name or self._default_provider
        if name not in self._providers:
            self._providers[name] = _create_builtin_provider(name, self._default_model)
        return self._providers[name]

    # -- middleware ---------------------------------------------------------

    def use(self, mw: Middleware) -> None:
        """Push a middleware onto the stack."""
        self._middleware.append(mw)

    # -- main API ----------------------------------------------------------

    def complete(self, request: Request) -> Response:
        """Send a completion request, running it through middleware."""
        provider_name = request.provider or self._default_provider
        adapter = self.get_provider(provider_name)

        def call_provider(req: Request) -> Response:
            return adapter.complete(req)

        # Build the middleware chain (outermost first).
        fn = call_provider
        for mw in reversed(self._middleware):
            outer_fn = fn
            # Capture mw and outer_fn in closure defaults to avoid late-binding.
            def make_wrapped(m: Middleware, nxt: Callable[[Request], Response]) -> Callable[[Request], Response]:
                return lambda req: m(req, nxt)
            fn = make_wrapped(mw, outer_fn)

        return fn(request)

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Stream a completion request, yielding events."""
        provider_name = request.provider or self._default_provider
        adapter = self.get_provider(provider_name)
        async for event in adapter.stream(request):
            yield event

    # -- convenience constructors ------------------------------------------

    @classmethod
    def from_env(
        cls,
        provider: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> Client:
        """Create a Client configured from environment variables.

        Reads ``LLM_PROVIDER`` (default ``"anthropic"``) and delegates
        API-key lookup to each provider adapter.
        """
        return cls(default_provider=provider, model=model)


def _create_builtin_provider(name: str, model: str | None = None) -> ProviderAdapter:
    """Lazily import and instantiate a built-in provider adapter."""
    if name == "anthropic":
        from attractor.llm.providers.anthropic import AnthropicAdapter
        return AnthropicAdapter(model=model)
    elif name == "openai":
        from attractor.llm.providers.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(model=model)
    elif name == "gemini":
        from attractor.llm.providers.gemini import GeminiAdapter
        return GeminiAdapter(model=model)
    elif name == "vertex":
        from attractor.llm.providers.vertex import VertexAdapter
        return VertexAdapter(model=model)
    else:
        raise ValueError(f"Unknown LLM provider: {name!r}")
