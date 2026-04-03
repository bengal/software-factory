"""Backward-compatible re-export: use openai_adapter instead."""

from attractor.llm.providers.openai_adapter import OpenAIAdapter, OpenAIProvider

__all__ = ["OpenAIAdapter", "OpenAIProvider"]
