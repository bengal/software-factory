"""Abstract base class for LLM provider adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from attractor.llm.types import Request, Response, StreamEvent


class ProviderAdapter(ABC):
    """Interface that every provider adapter must implement."""

    @abstractmethod
    def complete(self, request: Request) -> Response:
        """Send a synchronous completion request and return a Response."""
        ...

    @abstractmethod
    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Stream a completion request, yielding StreamEvents."""
        ...
        # Make this an async generator to satisfy the type checker.
        yield  # type: ignore[misc]  # pragma: no cover
