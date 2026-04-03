"""Command-line interface for the Attractor software factory.

Entry point registered as ``factory`` in pyproject.toml.

Subcommands:
    factory run                     Run the dark factory pipeline
    factory validate <dotfile>      Validate a DOT pipeline file
    factory run-pipeline <dotfile>  Run an arbitrary DOT pipeline
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose or os.environ.get("LOG_LEVEL", "").upper() == "DEBUG" else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def _cmd_run(args: argparse.Namespace) -> int:
    """Run the dark factory pipeline."""
    from attractor.factory.config import FactoryConfig
    from attractor.factory.pipeline import run_factory

    config_path = args.config or "factory-config.json"
    config = FactoryConfig.load(config_path)

    if args.specs_dir:
        config.specs_dir = args.specs_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model = args.model
    if args.pipeline:
        config.dotfile = args.pipeline

    outcome = run_factory(config=config)

    print(f"\nPipeline finished: {outcome.status.value}")
    if outcome.notes:
        print(f"Notes: {outcome.notes}")
    if outcome.failure_reason:
        print(f"Error: {outcome.failure_reason}")

    return 0 if outcome.status.value == "success" else 1


def _cmd_validate(args: argparse.Namespace) -> int:
    """Validate a DOT pipeline file."""
    from attractor.engine.parser import parse_dot

    dotfile = args.dotfile
    if not os.path.exists(dotfile):
        print(f"Error: File not found: {dotfile}", file=sys.stderr)
        return 1

    try:
        with open(dotfile, "r") as f:
            dot_source = f.read()
        graph = parse_dot(dot_source)
    except Exception as exc:
        print(f"Error parsing {dotfile}: {exc}", file=sys.stderr)
        return 1

    print(f"Graph: {graph.attrs.get('name', '(unnamed)')}")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")

    start = graph.find_start_node()
    exit_node = graph.find_exit_node()

    if start:
        print(f"Start node: {start.id}")
    else:
        print("Warning: No start node found", file=sys.stderr)

    if exit_node:
        print(f"Exit node: {exit_node.id}")
    else:
        print("Warning: No exit node found", file=sys.stderr)

    # Optional: use the validation module if available
    try:
        from attractor.engine.validation import validate
        diagnostics = validate(graph)
        if diagnostics:
            print(f"\nDiagnostics ({len(diagnostics)}):")
            for diag in diagnostics:
                print(f"  [{diag.severity.value}] {diag.message}")
        else:
            print("\nNo validation issues found.")
    except ImportError:
        pass

    print("\nNodes:")
    for node in graph.nodes.values():
        print(f"  {node.id} [shape={node.shape}, label={node.label!r}]")

    print("\nEdges:")
    for edge in graph.edges:
        label = f" ({edge.label})" if edge.label else ""
        condition = f" [condition={edge.condition}]" if edge.condition else ""
        print(f"  {edge.from_node} -> {edge.to_node}{label}{condition}")

    print("\nValidation passed.")
    return 0


def _cmd_run_pipeline(args: argparse.Namespace) -> int:
    """Run an arbitrary DOT pipeline."""
    from attractor.engine.executor import PipelineConfig, run
    from attractor.engine.parser import parse_dot

    dotfile = args.dotfile
    if not os.path.exists(dotfile):
        print(f"Error: File not found: {dotfile}", file=sys.stderr)
        return 1

    try:
        with open(dotfile, "r") as f:
            dot_source = f.read()
        graph = parse_dot(dot_source)
    except Exception as exc:
        print(f"Error parsing {dotfile}: {exc}", file=sys.stderr)
        return 1

    config = PipelineConfig(
        max_nodes=args.max_iterations or 500,
    )

    outcome = run(graph, config)

    print(f"\nPipeline finished: {outcome.status.value}")
    if outcome.notes:
        print(f"Notes: {outcome.notes}")
    if outcome.failure_reason:
        print(f"Error: {outcome.failure_reason}")

    return 0 if outcome.status.value == "success" else 1


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="factory",
        description="Attractor Software Factory CLI",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # factory run
    run_parser = subparsers.add_parser("run", help="Run the dark factory pipeline")
    run_parser.add_argument(
        "-c", "--config",
        default=None,
        help="Path to factory-config.json (default: ./factory-config.json)",
    )
    run_parser.add_argument(
        "--specs-dir",
        default=None,
        help="Override specs directory",
    )
    run_parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory",
    )
    run_parser.add_argument(
        "--provider",
        default=None,
        help="LLM provider (anthropic, openai, gemini)",
    )
    run_parser.add_argument(
        "--model",
        default=None,
        help="LLM model override",
    )
    run_parser.add_argument(
        "--pipeline",
        default=None,
        help="Path to a DOT pipeline file (default: from config or built-in)",
    )

    # factory validate
    validate_parser = subparsers.add_parser("validate", help="Validate a DOT pipeline file")
    validate_parser.add_argument(
        "dotfile",
        help="Path to the DOT file to validate",
    )

    # factory run-pipeline
    rp_parser = subparsers.add_parser("run-pipeline", help="Run an arbitrary DOT pipeline")
    rp_parser.add_argument(
        "dotfile",
        help="Path to the DOT file to run",
    )
    rp_parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum pipeline iterations (default: 500)",
    )
    rp_parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Total timeout in seconds (default: 3600)",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    _setup_logging(verbose=args.verbose)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "run": _cmd_run,
        "validate": _cmd_validate,
        "run-pipeline": _cmd_run_pipeline,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    exit_code = handler(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
