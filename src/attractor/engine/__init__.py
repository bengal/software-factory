"""Attractor pipeline engine - DOT-based workflow execution."""

from .graph import Graph, Node, Edge
from .parser import parse_dot, ParseError
from .context import Context, Checkpoint, Outcome, StageStatus
from .conditions import evaluate_condition
from .executor import run, PipelineConfig, BackoffConfig, select_edge, execute_with_retry
from .validation import validate, validate_or_raise, Diagnostic, Severity, ValidationError
from .stylesheet import apply_stylesheet, parse_stylesheet, StyleRule
from .handlers.base import Handler, HandlerRegistry, SHAPE_TO_TYPE
from .handlers.codergen import CodergenHandler, CodergenBackend
from .handlers.start import StartHandler
from .handlers.exit import ExitHandler
from .handlers.conditional import ConditionalHandler
from .handlers.parallel import ParallelHandler
from .handlers.fan_in import FanInHandler
from .handlers.tool_handler import ToolHandler
from .handlers.wait_human import (
    WaitForHumanHandler,
    Interviewer,
    AutoApproveInterviewer,
    ConsoleInterviewer,
)

__all__ = [
    # Graph model
    "Graph", "Node", "Edge",
    # Parser
    "parse_dot", "ParseError",
    # Context / Outcome
    "Context", "Checkpoint", "Outcome", "StageStatus",
    # Conditions
    "evaluate_condition",
    # Executor
    "run", "PipelineConfig", "BackoffConfig", "select_edge", "execute_with_retry",
    # Validation
    "validate", "validate_or_raise", "Diagnostic", "Severity", "ValidationError",
    # Stylesheet
    "apply_stylesheet", "parse_stylesheet", "StyleRule",
    # Handlers
    "Handler", "HandlerRegistry", "SHAPE_TO_TYPE",
    "CodergenHandler", "CodergenBackend",
    "StartHandler", "ExitHandler",
    "ConditionalHandler",
    "ParallelHandler", "FanInHandler",
    "ToolHandler",
    "WaitForHumanHandler", "Interviewer", "AutoApproveInterviewer", "ConsoleInterviewer",
]
