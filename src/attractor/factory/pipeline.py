"""Dark factory pipeline: wires the coding agent into the pipeline engine.

This module implements the FactoryBackend (a CodergenBackend) that uses
agent Sessions for code generation and verification, plus custom Handler
subclasses for each node type in the dark factory graph.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

from attractor.agent.environment import LocalExecutionEnvironment
from attractor.agent.session import Session, SessionConfig
from attractor.agent.tools.profiles import create_profile
from attractor.engine.context import Context, Outcome, StageStatus
from attractor.engine.executor import PipelineConfig, run
from attractor.engine.graph import Graph, Node
from attractor.engine.handlers.base import Handler, HandlerRegistry
from attractor.engine.handlers.codergen import CodergenBackend
from attractor.engine.parser import parse_dot
from attractor.factory.config import FactoryConfig
from attractor.llm.client import Client
from attractor.llm.types import Usage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DOT graph for the dark factory pipeline
# ---------------------------------------------------------------------------

DARK_FACTORY_DOT = r"""digraph DarkFactorySequential {
    graph [
        goal="Process all specs sequentially, each building on the previous result",
        label="Dark Factory (Sequential)",
        default_max_retries=5
    ]
    rankdir=TB

    // -- boundary nodes --
    start [shape=Mdiamond, label="Start"]
    exit  [shape=Msquare,  label="Exit"]

    // -- ingestion --
    ingest [shape=parallelogram, type="tool", label="Ingest Specs", timeout=120]

    // -- triage (orders work items by dependency) --
    triage [
        shape=box,
        label="Triage & Order",
        prompt="Analyze the work items in $work_items. For each item, identify which files and modules in the codebase it will likely touch. Determine dependencies between work items. Order them so that each item builds on the results of previous items.\n\nAfter your analysis, output the final ordered list of work item IDs as a JSON array on a line by itself. Example: [\"spec-a\", \"spec-b\", \"spec-c\"]",
        timeout=300,
        reasoning_effort="high"
    ]

    // -- sequential loop --
    next_item [shape=diamond, label="More work items?"]

    // -- per-work-item stages --
    understand [
        shape=box,
        label="Understand",
        output_key="understanding",
        prompt="You are processing work item $current_item.\n\nHere is the full spec (do NOT re-read it from disk — it is provided here in full):\n\n$current_spec\n\nIMPORTANT RULES:\n- This is an ANALYSIS step only. Do NOT create, modify, or write any files. Do NOT run mkdir, write_file, or any commands that change the filesystem.\n- Only analyze THIS spec. Do not read other specs in the specs/ directory.\n- Do not explore /opt/software-factory/ — that is the factory's own code, not the project.\n- Do NOT install packages, compilers, or toolchains. Do NOT run apt-get, curl, rustup, pip install, or similar.\n- Focus ONLY on reading and understanding existing source code in /workspace/output/.\n\nExamine the existing codebase in /workspace/output/ to understand the current state of the files and modules this spec affects. Produce a structured understanding that includes: (1) what exists today, (2) what needs to change, (3) risks and edge cases, (4) acceptance criteria restated as testable assertions.",
        timeout=600,
        fidelity="full",
        reasoning_effort="high"
    ]

    plan [
        shape=box,
        label="Plan",
        output_key="plan",
        prompt="You are processing work item $current_item.\n\nHere is the full spec (already provided — do NOT re-read from disk):\n\n$current_spec\n\nHere is the understanding of the current codebase state:\n\n$understanding\n\nBased on the spec and understanding above, create a detailed implementation plan for THIS spec only. List every file to create or modify, describe the changes for each file, specify the order of changes, and identify how to verify each change.\n\nDo NOT install packages, compilers, or toolchains in this step.",
        timeout=300,
        fidelity="full",
        reasoning_effort="high"
    ]

    implement [
        shape=box,
        label="Implement",
        prompt="You are processing work item $current_item.\n\nHere is the full spec (already provided — do NOT re-read from disk):\n\n$current_spec\n\nIMPORTANT RULES:\n- Only implement THIS spec. Do not implement other specs.\n- Do not explore /opt/software-factory/.\n- All code MUST be written under /workspace/output/. Do NOT write files to /workspace/ directly.\n- Do NOT re-read spec files from disk.\n- Be efficient with file reads: use grep and glob to locate relevant code instead of reading every file. Only read files you need to modify or that are direct dependencies of your changes. Do NOT read the entire codebase.\n\nExecute the implementation plan below. Make the code changes described. Follow existing code patterns and conventions.\n\n$plan",
        timeout=1800,
        fidelity="full",
        max_retries=0
    ]

    verify [
        shape=box,
        label="Verify",
        output_key="verification_result",
        prompt="You are verifying work item $current_item.\n\nHere is the full spec (do NOT re-read from disk):\n\n$current_spec\n\nThe implementation is complete. Your job is to verify it meets ALL acceptance criteria in the spec.\n\nSTEPS:\n1. Run the project build: $build_command\n2. Run the project tests: $test_command\n3. Run the project linter: $lint_command\n4. Check EACH acceptance criterion in the spec. If the spec requires running a script (e.g. an integration test), run it.\n5. All commands must be run from /workspace/output/ or use paths relative to it.\n\nAfter all checks, end your response with a summary:\n- For each acceptance criterion: PASS or FAIL with evidence.\n- End with exactly VERDICT: PASS (all criteria met) or VERDICT: FAIL (with list of failures).\n\nIMPORTANT RULES:\n- Do NOT modify any code or files. Only read files and run commands.\n- Do NOT install packages (apt-get, dnf, pip, etc.). All required tools are already installed.\n- Do NOT repeat checks you have already done. Be efficient.\n- If a test fails, report it and move on. Do not retry or investigate endlessly.",
        timeout=600,
        fidelity="full"
    ]

    // -- routing --
    check_result [shape=diamond, label="Verification passed?"]

    // -- failure path --
    diagnose [
        shape=box,
        label="Diagnose",
        output_key="diagnosis",
        prompt="You are processing work item $current_item.\n\nHere is the full spec (already provided — do NOT re-read from disk):\n\n$current_spec\n\nVerification failed with the following output:\n\n$verification_result\n\nIdentify the root cause of each failure. Determine whether the failures are fixable. Produce a diagnosis with a clear fix strategy for each fixable issue.\n\nIMPORTANT RULES:\n- Only fix issues related to THIS spec. Do not explore /opt/software-factory/.\n- Be efficient: use grep to locate the relevant code from the error messages. Do NOT read files that are unrelated to the failures.",
        timeout=300,
        fidelity="full",
        reasoning_effort="high"
    ]

    fix [
        shape=box,
        label="Fix",
        prompt="You are processing work item $current_item.\n\nHere is the full spec (already provided — do NOT re-read from disk):\n\n$current_spec\n\nApply the fixes described in the following diagnosis:\n\n$diagnosis\n\nMake only the changes needed to fix the identified issues.\n\nIMPORTANT RULES:\n- Do not explore /opt/software-factory/.\n- Be efficient with file reads: only read the specific files mentioned in the diagnosis. Use grep to locate code if needed. Do NOT re-read files you have already seen.",
        timeout=900,
        fidelity="full",
        max_retries=5,
        retry_target="diagnose"
    ]

    // -- quarantine, package & commit --
    quarantine [shape=parallelogram, type="tool", label="Quarantine", timeout=120]
    package    [shape=parallelogram, type="tool", label="Package",    timeout=120]
    commit     [shape=parallelogram, type="tool", label="Commit",     timeout=120]

    // -- reporting --
    report [
        shape=box,
        label="Report",
        prompt="Generate the morning report. Include: (1) Executive summary: how many items succeeded, failed, have conflicts. (2) For each successful item: title, summary, files modified. (3) For each quarantined item: title, what was tried, why it failed. (4) Recommendations. Write the report to output/report.md.",
        timeout=300,
        fidelity="compact"
    ]

    // -- edges --
    start -> ingest -> triage -> next_item

    // loop: pick next work item or finish
    next_item -> understand [label="yes", condition="$has_next_item=true"]
    next_item -> report     [label="no",  condition="$has_next_item!=true"]

    // per-item pipeline
    understand -> plan -> implement -> verify -> check_result
    check_result -> package    [label="pass", condition="$verification_passed=true"]
    check_result -> diagnose   [label="fail", condition="$verification_passed!=true"]
    diagnose -> fix
    fix -> verify              [label="retry"]
    fix -> quarantine          [label="retries exhausted", condition="$retries_exhausted=true"]

    // after package, commit then loop back; quarantine skips commit
    package    -> commit -> next_item
    quarantine -> next_item

    report -> exit
}
"""


# ---------------------------------------------------------------------------
# Factory Backend (CodergenBackend)
# ---------------------------------------------------------------------------

class TokenLimitExceeded(Exception):
    """Raised when the total token budget for a factory run is exhausted."""


class FactoryBackend(CodergenBackend):
    """Backend that uses the coding agent Session for code generation.

    Used by codergen nodes: triage, understand, plan, implement, diagnose,
    fix, integrate, report.
    """

    # Pricing per million tokens (USD).  Keyed by model prefix.
    _PRICING: dict[str, dict[str, float]] = {
        "claude-sonnet-4": {
            "input": 3.0, "output": 15.0,
            "cache_read": 0.30, "cache_write": 3.75,
        },
        "claude-opus-4": {
            "input": 15.0, "output": 75.0,
            "cache_read": 1.50, "cache_write": 18.75,
        },
        "claude-3-5-sonnet": {
            "input": 3.0, "output": 15.0,
            "cache_read": 0.30, "cache_write": 3.75,
        },
        "claude-3-5-haiku": {
            "input": 0.80, "output": 4.0,
            "cache_read": 0.08, "cache_write": 1.0,
        },
    }

    def __init__(self, config: FactoryConfig, working_dir: str):
        self.config = config
        self.working_dir = working_dir
        self._env = LocalExecutionEnvironment(working_dir=working_dir)
        self._client: Client | None = None
        self._total_usage = Usage()

    @property
    def total_tokens(self) -> int:
        return self._total_usage.input_tokens + self._total_usage.output_tokens

    @property
    def total_cost(self) -> float:
        return self._estimate_cost(self._total_usage)

    def _get_pricing(self) -> dict[str, float]:
        """Return pricing dict for the configured model."""
        model = self.config.model or ""
        for prefix, pricing in self._PRICING.items():
            if model.startswith(prefix):
                return pricing
        # Fallback to Sonnet pricing
        return self._PRICING["claude-sonnet-4"]

    def _estimate_cost(self, usage: Usage) -> float:
        """Estimate cost in USD for a Usage.

        Note: Vertex reports ``input_tokens`` as non-cached only, with
        ``cache_read_tokens`` and ``cache_write_tokens`` reported separately.
        So ``input_tokens`` already represents the uncached portion.
        """
        p = self._get_pricing()
        return (
            usage.input_tokens * p["input"]
            + usage.output_tokens * p["output"]
            + usage.cache_read_tokens * p["cache_read"]
            + usage.cache_write_tokens * p["cache_write"]
        ) / 1_000_000

    def _get_client(self) -> Client:
        if self._client is None:
            self._client = Client.from_env(
                provider=self.config.provider,
                model=self.config.model,
            )
        return self._client

    # Per-node tool round limits to control token usage.
    # Each tool round re-sends the full conversation history, so more rounds
    # means quadratically more input tokens.
    _NODE_TOOL_ROUNDS: dict[str, int] = {
        "triage": 20,
        "understand": 50,
        "plan": 1,
        "implement": 200,
        "verify": 50,
        "diagnose": 30,
        "fix": 120,
        "report": 100,
    }

    def run(self, node: Node, prompt: str, context: Context) -> Union[str, Outcome]:
        """Run an LLM task via the coding agent session."""
        import json as _json

        token_limit = self.config.limits.max_tokens
        if token_limit and self.total_tokens >= token_limit:
            raise TokenLimitExceeded(
                f"Token limit exceeded: {self.total_tokens:,} >= {token_limit:,}"
            )

        cost_limit = self.config.limits.max_cost_usd
        if cost_limit and self.total_cost >= cost_limit:
            raise TokenLimitExceeded(
                f"Cost limit exceeded: ${self.total_cost:.4f} >= ${cost_limit:.2f}"
            )

        max_rounds = self._NODE_TOOL_ROUNDS.get(node.id, 20)

        # Log which work item this node is processing and relevant context
        self._log_node_context(node, context)

        client = self._get_client()
        profile = create_profile(
            provider=self.config.provider,
            model=self.config.model,
            env=self._env,
        )

        # Block reads from the specs directory and the factory's own code.
        # The spec content is already provided in the prompt; re-reading it
        # wastes tokens. The factory code is irrelevant to the project.
        blocked = [
            os.path.abspath(self.config.specs_dir),
            "/opt/software-factory",
        ]
        registry = profile.tool_registry
        read_tool = registry.get("read_file")
        if read_tool:
            original_executor = read_tool.executor
            def _guarded_read(args, _orig=original_executor, _blocked=blocked):
                path = os.path.abspath(args.get("file_path", ""))
                for prefix in _blocked:
                    if path.startswith(prefix):
                        return (f"Error: Reading from {prefix} is not allowed. "
                                "The spec content is already in your prompt.")
                return _orig(args)
            read_tool.executor = _guarded_read

        # When max_rounds is 1 or less, don't send tools so the LLM
        # produces a single text response without attempting tool calls.
        no_tools = max_rounds <= 1
        session = Session(
            client=client,
            tool_registry=profile.tool_registry,
            system_prompt=profile.build_system_prompt(),
            config=SessionConfig(
                max_turns=100,
                max_tool_rounds_per_input=max_rounds,
                default_command_timeout_ms=120_000,
            ),
            llm_model=profile.model,
            llm_tools=[] if no_tools else profile.tools(),
        )
        result = session.process_input(prompt)

        usage = session.total_usage
        self._total_usage = self._total_usage + usage
        node_cost = self._estimate_cost(usage)
        logger.info(
            "Token usage: node=%s \033[33minput=%d output=%d cost=$%.4f\033[0m"
            " tool_rounds=%d cumulative_cost=\033[33m$%.4f\033[0m",
            node.id, usage.input_tokens, usage.output_tokens, node_cost,
            session.tool_rounds_used, self.total_cost,
        )

        # Warn when approaching cost limit
        cost_limit = self.config.limits.max_cost_usd
        if cost_limit and self.total_cost >= cost_limit * 0.8:
            logger.warning(
                "\033[31mCost warning: $%.4f spent of $%.2f budget (%.0f%%)\033[0m",
                self.total_cost, cost_limit, (self.total_cost / cost_limit) * 100,
            )

        session.close()
        return result

    @staticmethod
    def _log_node_context(node: Node, context: Context) -> None:
        """Log the work item and relevant context for the current node."""
        import json as _json

        # Identify which work item is being processed
        raw = context.get("current_item")
        if raw:
            try:
                item = _json.loads(raw) if isinstance(raw, str) else raw
                item_id = item.get("id", "?") if isinstance(item, dict) else str(item)
                item_title = item.get("title", "") if isinstance(item, dict) else ""
            except Exception:
                item_id = str(raw)[:80]
                item_title = ""
            label = f"{item_id} ({item_title})" if item_title else item_id
            logger.info("  work item: %s", label)

        # Log node-specific context summaries
        _CONTEXT_KEYS: dict[str, list[str]] = {
            "diagnose": ["verification_result"],
            "fix": ["diagnosis"],
        }
        for key in _CONTEXT_KEYS.get(node.id, []):
            val = context.get(key)
            if val:
                first_line = str(val).strip().split("\n", 1)[0][:200]
                logger.info("  %s: %s", key, first_line)


# ---------------------------------------------------------------------------
# Node handlers
# ---------------------------------------------------------------------------

def _manifest_path(output_dir: str) -> Path:
    """Return the path to the spec manifest file."""
    return Path(output_dir) / ".factory" / "manifest.json"


def _load_manifest(output_dir: str) -> dict[str, str]:
    """Load the manifest mapping spec IDs to their SHA-256 checksums."""
    import json
    path = _manifest_path(output_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(
            "Corrupt manifest at %s (%s) — all specs will be reprocessed. "
            "Delete the file to silence this warning.",
            path, exc,
        )
        return {}


def _save_manifest(output_dir: str, manifest: dict[str, str]) -> None:
    """Save the manifest to disk."""
    import json
    path = _manifest_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _file_checksum(path: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class IngestHandler(Handler):
    """Reads spec files from the specs directory into $work_items.

    Skips specs whose content has not changed since the last successful run
    (tracked via a SHA-256 manifest in the output directory).
    """

    def __init__(self, config: FactoryConfig):
        self.config = config

    def execute(self, node: Node, context: Context, graph: Graph, logs_root=None) -> Outcome:
        import json
        specs_dir = os.path.abspath(self.config.specs_dir)
        if not os.path.isdir(specs_dir):
            logger.warning("Specs directory not found: %s — producing empty work_items", specs_dir)
            context.set("work_items", "[]")
            return Outcome(status=StageStatus.SUCCESS, notes="No specs directory; empty run")

        manifest = _load_manifest(self.config.output_dir)
        work_items: list[dict] = []
        skipped: list[str] = []
        pending_checksums: dict[str, str] = {}

        for fname in sorted(os.listdir(specs_dir)):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(specs_dir, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                content = Path(fpath).read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning("Failed to read spec %s: %s", fpath, exc)
                continue

            item_id = fname.removesuffix(".md")
            checksum = _file_checksum(fpath)
            pending_checksums[item_id] = checksum

            if manifest.get(item_id) == checksum:
                logger.info("Skipping unchanged spec: %s", item_id)
                skipped.append(item_id)
                continue

            work_items.append({
                "id": item_id,
                "title": item_id.replace("-", " ").replace("_", " ").title(),
                "source_file": fpath,
                "content": content,
            })

        # Store pending checksums so CommitHandler can update the manifest
        context.set("pending_checksums", json.dumps(pending_checksums))

        context.set("work_items", json.dumps(work_items))
        skipped_msg = f", skipped {len(skipped)} unchanged" if skipped else ""
        logger.info(
            "Ingested %d spec file(s) from %s%s",
            len(work_items), specs_dir, skipped_msg,
        )
        return Outcome(
            status=StageStatus.SUCCESS,
            notes=f"Ingested {len(work_items)} spec file(s){skipped_msg}",
            context_updates={"work_items_count": len(work_items)},
        )


class VerifyHandler(Handler):
    """Runs the project's test suite, linter, type checker, and build."""

    def __init__(self, config: FactoryConfig, working_dir: str):
        self.config = config
        self._env = LocalExecutionEnvironment(working_dir=working_dir)

    def execute(self, node: Node, context: Context, graph: Graph, logs_root=None) -> Outcome:
        verify_cfg = self.config.verify
        commands: list[tuple[str, str]] = []
        if verify_cfg.test_command:
            commands.append(("test", verify_cfg.test_command))
        if verify_cfg.lint_command:
            commands.append(("lint", verify_cfg.lint_command))
        if verify_cfg.typecheck_command:
            commands.append(("typecheck", verify_cfg.typecheck_command))
        if verify_cfg.build_command:
            commands.append(("build", verify_cfg.build_command))

        if not commands:
            logger.warning("No verification commands configured — auto-pass "
                           "(test_command=%r, build_command=%r)",
                           verify_cfg.test_command, verify_cfg.build_command)
            context.set("verification_result", "No verification commands configured — auto-pass")
            return Outcome(status=StageStatus.SUCCESS, notes="No verification commands configured")

        results: list[str] = []
        failures: list[str] = []
        timeout_ms = int(self.config.limits.verify_timeout * 1000)

        for name, cmd in commands:
            logger.info("Verify %s: %s", name, cmd)
            result = self._env.exec_command(cmd, timeout_ms=timeout_ms)
            output = (result.stdout or "") + (result.stderr or "")
            results.append(f"=== {name} (exit {result.exit_code}) ===\n{output}")
            if result.exit_code != 0:
                logger.warning("Verify %s FAILED (exit %d)", name, result.exit_code)
                failures.append(f"{name}: exit code {result.exit_code}")
            else:
                logger.info("Verify %s passed", name)

        full_output = "\n\n".join(results)
        context.set("verification_result", full_output)

        if failures:
            context.set("verification_passed", "false")
            return Outcome(
                status=StageStatus.SUCCESS,
                notes=f"Verification failed: {'; '.join(failures)}",
                context_updates={
                    "verification_result": full_output,
                    "verification_passed": "false",
                },
            )
        context.set("verification_passed", "true")
        return Outcome(
            status=StageStatus.SUCCESS,
            context_updates={
                "verification_result": full_output,
                "verification_passed": "true",
            },
        )


class PackageHandler(Handler):
    """Packages successful results: generates diff.patch and copies artifacts."""

    def __init__(self, config: FactoryConfig, working_dir: str):
        self.config = config
        self.working_dir = working_dir

    def execute(self, node: Node, context: Context, graph: Graph, logs_root=None) -> Outcome:
        import json
        import subprocess

        current_item = context.get("current_item")
        if isinstance(current_item, str):
            try:
                current_item = json.loads(current_item)
            except Exception:
                current_item = {"id": "unknown"}

        item_id = current_item.get("id", "unknown") if isinstance(current_item, dict) else "unknown"
        item_dir = Path(self.config.output_dir) / "work-items" / item_id
        item_dir.mkdir(parents=True, exist_ok=True)

        # Write status
        (item_dir / "status.json").write_text(json.dumps({"status": "SUCCESS", "item_id": item_id}))

        # Copy artifacts from context
        for key in ("understanding", "plan", "verification_result"):
            val = context.get(key)
            if val:
                (item_dir / f"{key}.md").write_text(str(val))

        # Generate diff.patch via git diff (best effort)
        try:
            result = subprocess.run(
                ["git", "diff", "--cached", "--diff-filter=ACMR"],
                capture_output=True, text=True, cwd=self.working_dir, timeout=30,
            )
            if result.stdout:
                (item_dir / "diff.patch").write_text(result.stdout)
        except Exception:
            pass

        logger.info("Packaged work item %s to %s", item_id, item_dir)
        return Outcome(status=StageStatus.SUCCESS, notes=f"Packaged {item_id}")


class CommitHandler(Handler):
    """Commits the current work item's changes to git.

    After a successful commit, updates the spec manifest so the item
    is skipped on the next run (unless the spec file changes).
    """

    def __init__(self, config: FactoryConfig, working_dir: str):
        self.config = config
        self.working_dir = working_dir

    def execute(self, node: Node, context: Context, graph: Graph, logs_root=None) -> Outcome:
        import json
        import subprocess

        current_item = context.get("current_item")
        if isinstance(current_item, str):
            try:
                current_item = json.loads(current_item)
            except Exception:
                current_item = {"id": "unknown", "title": "unknown"}

        item_id = current_item.get("id", "unknown") if isinstance(current_item, dict) else "unknown"
        title = current_item.get("title", item_id) if isinstance(current_item, dict) else item_id

        try:
            # Ensure git repo is initialized
            self._ensure_git_repo()

            # Ensure .gitignore exists so build artifacts are not committed
            self._ensure_gitignore()

            # Stage all changes
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.working_dir, timeout=30, check=True,
                capture_output=True, text=True,
            )
            # Check if there is anything to commit
            result = subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=self.working_dir, timeout=30,
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                logger.info("No changes to commit for %s", item_id)
                self._update_manifest(item_id, context)
                return Outcome(status=StageStatus.SUCCESS, notes="Nothing to commit")

            # Commit
            msg = f"{title}\n\nWork item: {item_id}"
            subprocess.run(
                ["git", "commit", "-m", msg],
                cwd=self.working_dir, timeout=30, check=True,
                capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as exc:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=f"git commit failed: {exc.stderr or exc.stdout}",
            )

        self._update_manifest(item_id, context)
        logger.info("Committed work item %s", item_id)
        return Outcome(status=StageStatus.SUCCESS, notes=f"Committed {item_id}")

    def _ensure_git_repo(self) -> None:
        """Initialize a git repo in working_dir if one doesn't exist."""
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=self.working_dir, capture_output=True, text=True,
        )
        if result.returncode != 0:
            subprocess.run(
                ["git", "init"],
                cwd=self.working_dir, timeout=10, check=True,
                capture_output=True, text=True,
            )
            logger.info("Initialized git repository in %s", self.working_dir)

    def _ensure_gitignore(self) -> None:
        """Create .gitignore with build artifact patterns if one doesn't exist."""
        gitignore = Path(self.working_dir) / ".gitignore"
        if gitignore.exists():
            return

        patterns: list[str] = []
        verify = self.config.verify
        commands = " ".join(filter(None, [
            verify.test_command, verify.lint_command,
            verify.typecheck_command, verify.build_command,
        ]))

        if "cargo" in commands:
            patterns.append("target/")
        if "pytest" in commands or "python" in commands:
            patterns.extend(["__pycache__/", "*.pyc", ".eggs/", "*.egg-info/"])
        if "npm" in commands or "node" in commands:
            patterns.append("node_modules/")
        if "go " in commands or "go.mod" in commands:
            patterns.append("vendor/")
        if "mvn" in commands or "gradle" in commands:
            patterns.extend(["target/", "build/", "*.class"])

        if patterns:
            gitignore.write_text("\n".join(patterns) + "\n", encoding="utf-8")
            logger.info("Created .gitignore with %d patterns", len(patterns))

    def _update_manifest(self, item_id: str, context: Context) -> None:
        """Record the spec's checksum in the manifest so it's skipped next run."""
        import json
        pending_raw = context.get("pending_checksums")
        if not pending_raw:
            return
        try:
            pending = json.loads(pending_raw) if isinstance(pending_raw, str) else pending_raw
        except Exception as exc:
            logger.warning("Failed to parse pending_checksums: %s", exc)
            return
        checksum = pending.get(item_id)
        if not checksum:
            return
        manifest = _load_manifest(self.config.output_dir)
        manifest[item_id] = checksum
        _save_manifest(self.config.output_dir, manifest)


class QuarantineHandler(Handler):
    """Saves all state for a failed work item and marks it for human attention."""

    def __init__(self, config: FactoryConfig):
        self.config = config

    def execute(self, node: Node, context: Context, graph: Graph, logs_root=None) -> Outcome:
        import json

        current_item = context.get("current_item")
        if isinstance(current_item, str):
            try:
                current_item = json.loads(current_item)
            except Exception:
                current_item = {"id": "unknown"}

        item_id = current_item.get("id", "unknown") if isinstance(current_item, dict) else "unknown"
        item_dir = Path(self.config.output_dir) / "work-items" / item_id
        item_dir.mkdir(parents=True, exist_ok=True)

        # Write status
        status_data = {
            "status": "QUARANTINED",
            "item_id": item_id,
            "failure_reason": context.get("last_failure_reason", "Unknown"),
        }
        (item_dir / "status.json").write_text(json.dumps(status_data, indent=2))

        # Save all available artifacts
        for key in ("understanding", "plan", "diagnosis", "verification_result"):
            val = context.get(key)
            if val:
                (item_dir / f"{key}.md").write_text(str(val))

        logger.warning("Quarantined work item %s", item_id)
        return Outcome(
            status=StageStatus.SUCCESS,  # quarantine itself succeeds so pipeline continues
            notes=f"Quarantined {item_id}",
        )


class TriageHandler(Handler):
    """Runs the triage LLM and captures ordered work items in context.

    The triage codergen node analyses work items and their dependencies,
    then returns them in dependency order.  This handler parses the ordered
    result from the LLM response and stores it as ``triaged_items`` in
    context so that :class:`NextItemHandler` consumes items in the correct
    order.
    """

    def __init__(self, backend: FactoryBackend):
        self.backend = backend

    def execute(self, node: Node, context: Context, graph: Graph, logs_root=None) -> Outcome:
        import json
        import re
        from attractor.engine.handlers.codergen import expand_variables

        prompt = expand_variables(node.prompt, node, context, graph)

        preview = prompt[:200] + ("…" if len(prompt) > 200 else "")
        logger.info("Prompt for %s (%d chars): \033[34m%s\033[0m", node.id, len(prompt), preview)

        # Write prompt to logs
        if logs_root:
            log_dir = Path(logs_root) / node.id
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "prompt.md").write_text(prompt, encoding="utf-8")

        try:
            result = self.backend.run(node, prompt, context)
            response = result.notes if isinstance(result, Outcome) else str(result)
        except Exception as exc:
            return Outcome(status=StageStatus.FAIL, failure_reason=str(exc))

        # Write response to logs
        if logs_root:
            log_dir = Path(logs_root) / node.id
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "response.md").write_text(response, encoding="utf-8")

        # Extract ordered item IDs from the LLM response (a JSON array of strings).
        ordered_ids = self._extract_ordered_ids(response)

        # Reorder work_items according to the triage output
        items_json = context.get("work_items") or "[]"
        try:
            items = json.loads(items_json)
        except Exception:
            items = []

        if ordered_ids:
            items_by_id = {item["id"]: item for item in items if isinstance(item, dict)}
            ordered_items = []
            for item_id in ordered_ids:
                if item_id in items_by_id:
                    ordered_items.append(items_by_id.pop(item_id))
            # Append any items not mentioned in the triage output
            ordered_items.extend(items_by_id.values())
            items = ordered_items

        context.set("triaged_items", json.dumps(items))

        logger.info(
            "Triage complete: %d items ordered (%d IDs extracted from response)",
            len(items), len(ordered_ids),
        )
        return Outcome(
            status=StageStatus.SUCCESS,
            notes=response,
            context_updates={"triaged_items_count": len(items)},
        )

    @staticmethod
    def _extract_ordered_ids(response: str) -> list[str]:
        """Extract ordered item IDs from the LLM response.

        Looks for a JSON array of strings in the response text.
        """
        import json
        import re

        for match in re.finditer(r'\[(?:\s*"[^"]*"\s*,?\s*)+\]', response):
            try:
                result = json.loads(match.group())
                if isinstance(result, list) and all(isinstance(x, str) for x in result):
                    return result
            except json.JSONDecodeError:
                continue

        return []


class NextItemHandler(Handler):
    """Manages the sequential work item queue.

    Each time this handler is invoked (at the ``next_item`` diamond node),
    it pops the next work item from the queue stored in context and sets
    ``current_item`` and ``has_next_item`` accordingly.
    """

    def execute(self, node: Node, context: Context, graph: Graph, logs_root=None) -> Outcome:
        import json

        # Initialise the queue on first visit from the triaged work items list
        queue_json = context.get("_work_item_queue")
        if queue_json is None:
            items_json = context.get("triaged_items") or context.get("work_items") or "[]"
            try:
                items = json.loads(items_json)
            except Exception:
                items = []
            context.set("_work_item_queue", json.dumps(items))
            queue = list(items)
        else:
            try:
                queue = json.loads(queue_json)
            except Exception:
                queue = []

        if queue:
            item = queue.pop(0)
            context.set("_work_item_queue", json.dumps(queue))
            # Clear per-item state from previous iteration
            for key in ("understanding", "plan", "changed_files", "verification_result",
                        "verification_passed", "diagnosis", "retries_exhausted"):
                context.set(key, "")
            # Store spec content separately so prompts can reference $current_spec;
            # strip it from current_item to keep that variable lightweight.
            if isinstance(item, dict):
                context.set("current_spec", item.get("content", ""))
                item_ref = {k: v for k, v in item.items() if k != "content"}
                context.set("current_item", json.dumps(item_ref))
            else:
                context.set("current_item", str(item))
            context.set("has_next_item", "true")
            item_id = item.get("id", "?") if isinstance(item, dict) else str(item)
            logger.info("Next work item: %s (%d remaining)", item_id, len(queue))
            return Outcome(
                status=StageStatus.SUCCESS,
                preferred_label="yes",
                notes=f"Next item: {item_id}",
            )
        else:
            context.set("has_next_item", "false")
            context.set("current_item", "")
            logger.info("All work items processed")
            return Outcome(
                status=StageStatus.SUCCESS,
                preferred_label="no",
                notes="No more items",
            )


# ---------------------------------------------------------------------------
# Factory setup
# ---------------------------------------------------------------------------

def create_dark_factory(
    config: FactoryConfig,
    working_dir: str | None = None,
) -> tuple[Graph, PipelineConfig, HandlerRegistry, "FactoryBackend"]:
    """Create and wire the dark factory pipeline.

    Returns (Graph, PipelineConfig, HandlerRegistry) ready for engine.run().
    """
    from attractor.engine.handlers.codergen import CodergenHandler
    from attractor.engine.handlers.start import StartHandler
    from attractor.engine.handlers.exit import ExitHandler
    from attractor.engine.handlers.conditional import ConditionalHandler
    from attractor.engine.handlers.parallel import ParallelHandler
    from attractor.engine.handlers.fan_in import FanInHandler

    work_dir = working_dir or os.getcwd()

    # Parse the DOT graph — from file if configured, else use bundled default
    if config.dotfile and os.path.isfile(config.dotfile):
        logger.info("Loading pipeline from %s", config.dotfile)
        dot_source = Path(config.dotfile).read_text(encoding="utf-8")
    else:
        bundled = Path(__file__).resolve().parent.parent.parent / "pipelines" / "dark_factory_sequential.dot"
        if bundled.is_file():
            logger.info("Loading bundled pipeline from %s", bundled)
            dot_source = bundled.read_text(encoding="utf-8")
        else:
            logger.info("Using built-in pipeline definition")
            dot_source = DARK_FACTORY_DOT
    graph = parse_dot(dot_source)

    # Create the LLM backend for codergen nodes (exposed on registry for cost tracking)
    backend = FactoryBackend(config, working_dir=work_dir)

    # Build handler registry
    registry = HandlerRegistry()
    registry.register("start", StartHandler())
    registry.register("exit", ExitHandler())
    registry.register("codergen", CodergenHandler(backend=backend))
    registry.register("conditional", ConditionalHandler())
    registry.register("parallel", ParallelHandler())
    registry.register("parallel.fan_in", FanInHandler())

    # Register dark-factory-specific handlers for tool nodes
    # These are resolved by setting node.type in the graph
    ingest_handler = IngestHandler(config)
    package_handler = PackageHandler(config, working_dir=work_dir)
    commit_handler = CommitHandler(config, working_dir=work_dir)
    quarantine_handler = QuarantineHandler(config)

    triage_handler = TriageHandler(backend)
    next_item_handler = NextItemHandler()

    registry.register("ingest_tool", ingest_handler)
    registry.register("triage_tool", triage_handler)
    registry.register("package_tool", package_handler)
    registry.register("commit_tool", commit_handler)
    registry.register("quarantine_tool", quarantine_handler)
    registry.register("next_item_tool", next_item_handler)

    # Override node types so the registry resolves custom handlers
    for nid, handler_type in [
        ("ingest", "ingest_tool"),
        ("triage", "triage_tool"),
        ("package", "package_tool"),
        ("commit", "commit_tool"),
        ("quarantine", "quarantine_tool"),
        ("next_item", "next_item_tool"),
    ]:
        if nid in graph.nodes:
            graph.nodes[nid].type = handler_type

    # Seed the context with verify commands so $build_command etc. expand in prompts
    context = Context()
    context.set("build_command", config.verify.build_command or "(no build command configured)")
    context.set("test_command", config.verify.test_command or "(no test command configured)")
    context.set("lint_command", config.verify.lint_command or "(no lint command configured)")

    pipeline_config = PipelineConfig(registry=registry, context=context)

    return graph, pipeline_config, registry, backend


def run_factory(
    specs_dir: str | None = None,
    output_dir: str | None = None,
    working_dir: str | None = None,
    config: FactoryConfig | None = None,
) -> Outcome:
    """Wire everything together and execute the dark factory pipeline."""
    if config is None:
        config = FactoryConfig.load()

    if specs_dir:
        config.specs_dir = specs_dir
    if output_dir:
        config.output_dir = output_dir

    work_dir = working_dir or os.getcwd()

    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)

    logger.info(
        "Starting dark factory: specs=%s output=%s provider=%s model=%s",
        config.specs_dir, config.output_dir, config.provider, config.model,
    )
    logger.info(
        "Verify commands: build=%r test=%r lint=%r",
        config.verify.build_command, config.verify.test_command, config.verify.lint_command,
    )

    graph, pipeline_config, _, backend = create_dark_factory(config, working_dir=work_dir)
    outcome = run(graph, pipeline_config)

    usage = backend._total_usage
    logger.info(
        "Factory finished: status=%s \033[33mtotal_cost=$%.4f\033[0m"
        " (input=%d output=%d cache_read=%d cache_write=%d)",
        outcome.status.value, backend.total_cost,
        usage.input_tokens, usage.output_tokens,
        usage.cache_read_tokens, usage.cache_write_tokens,
    )
    return outcome
