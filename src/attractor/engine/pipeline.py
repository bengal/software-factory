"""Backward-compatibility shim.

The pipeline execution logic has been refactored into:
- ``attractor.engine.executor`` (run, PipelineConfig, etc.)
- ``attractor.engine.handlers.base`` (Handler, HandlerRegistry)
- ``attractor.engine.context`` (Context, Outcome, StageStatus)

This module re-exports key names for any code that imported from here.
"""

from attractor.engine.context import Context, Outcome, StageStatus
from attractor.engine.executor import PipelineConfig, BackoffConfig, run, select_edge, execute_with_retry
from attractor.engine.handlers.base import Handler, HandlerRegistry
from attractor.engine.handlers.codergen import CodergenBackend

__all__ = [
    "Context",
    "Outcome",
    "StageStatus",
    "PipelineConfig",
    "BackoffConfig",
    "run",
    "select_edge",
    "execute_with_retry",
    "Handler",
    "HandlerRegistry",
    "CodergenBackend",
]
