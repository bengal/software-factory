"""Provider-specific tool profiles for the coding agent.

Each profile customizes the tool set and system prompt for a
particular LLM provider and model family.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from attractor.agent.environment import ExecutionEnvironment
from attractor.agent.tools.core import create_core_tools
from attractor.agent.tools.registry import RegisteredTool, ToolDefinition, ToolRegistry
from attractor.llm import types as llm_types


class ProviderProfile(ABC):
    """Abstract profile defining tools and prompts for an LLM provider."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Provider identifier (e.g. 'anthropic', 'openai', 'gemini')."""
        ...

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier."""
        ...

    @property
    @abstractmethod
    def tool_registry(self) -> ToolRegistry:
        """Return the tool registry with all tools for this profile."""
        ...

    @abstractmethod
    def build_system_prompt(self) -> str:
        """Build the system prompt for this profile."""
        ...

    def tools(self) -> list[llm_types.ToolDefinition]:
        """Return tool definitions formatted for the LLM API."""
        return [
            llm_types.ToolDefinition(
                name=td.name,
                description=td.description,
                parameters=td.parameters,
            )
            for td in self.tool_registry.definitions
        ]

    def provider_options(self) -> dict[str, Any]:
        """Return any provider-specific options for requests."""
        return {}


# ---------------------------------------------------------------------------
# Anthropic / Claude profile
# ---------------------------------------------------------------------------

_ANTHROPIC_SYSTEM_PROMPT = """\
You are an expert software engineer acting as a coding agent. You have access \
to tools for reading, writing, and editing files, running shell commands, and \
searching codebases. Use these tools to accomplish the user's task.

Guidelines:
- Always read files before editing them.
- Prefer editing existing files over creating new ones.
- Use grep and glob to explore unfamiliar codebases.
- Run tests after making changes when possible.
- Be precise with file paths - use absolute paths.
- Keep changes minimal and focused on the task.
- Only create files that are necessary for the code to compile and pass tests.
- Do NOT create documentation, summaries, checklists, or status reports \
(e.g. README, QUICKSTART, INDEX, IMPLEMENTATION_SUMMARY, VERIFICATION_CHECKLIST, \
FIXES_APPLIED, START_HERE, PROJECT_STRUCTURE, FILES_SUMMARY, changed_files). \
Only create such files if the task explicitly asks for them.
- Verify your work by running the actual test and build commands, not by \
writing markdown checklists.
"""


class AnthropicProfile(ProviderProfile):
    """Claude Code-aligned profile with standard coding tools."""

    def __init__(self, model_name: str, env: ExecutionEnvironment):
        self._model = model_name
        self._env = env
        self._registry = ToolRegistry()
        for tool in create_core_tools(env):
            self._registry.register(tool)

    @property
    def id(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._model

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._registry

    def build_system_prompt(self) -> str:
        return _ANTHROPIC_SYSTEM_PROMPT + (
            f"\nPlatform: {self._env.platform}\n"
            f"Working directory: {self._env.working_directory}\n"
        )


# ---------------------------------------------------------------------------
# OpenAI / Codex profile
# ---------------------------------------------------------------------------

_OPENAI_SYSTEM_PROMPT = """\
You are a coding agent with tools for file I/O, shell commands, and code search. \
Complete the user's task by using the available tools. Be precise, test your \
changes, and explain your reasoning. \
Only create files necessary for the code to compile and pass tests. \
Do NOT create documentation, summaries, checklists, or status reports unless \
the task explicitly asks for them. Verify by running actual commands, not by \
writing markdown checklists.
"""

_APPLY_PATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "patch": {
            "type": "string",
            "description": "A unified diff patch to apply to files.",
        },
    },
    "required": ["patch"],
}


def _apply_patch(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
    """Apply a unified diff patch via the patch command."""
    patch_content = arguments["patch"]
    # Write patch to a temp file and apply
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".patch", delete=False, dir=env.working_directory
    ) as f:
        f.write(patch_content)
        patch_path = f.name

    try:
        result = env.exec_command(
            f"patch -p1 < {patch_path}",
            timeout_ms=30_000,
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR: {result.stderr}"
        if result.exit_code != 0:
            output += f"\n[patch failed with exit code {result.exit_code}]"
        return output
    finally:
        try:
            os.unlink(patch_path)
        except OSError:
            pass


class OpenAIProfile(ProviderProfile):
    """Codex-aligned profile with apply_patch tool."""

    def __init__(self, model_name: str, env: ExecutionEnvironment):
        self._model = model_name
        self._env = env
        self._registry = ToolRegistry()
        for tool in create_core_tools(env):
            self._registry.register(tool)
        # Add apply_patch tool
        self._registry.register(RegisteredTool(
            definition=ToolDefinition(
                name="apply_patch",
                description="Apply a unified diff patch to files in the working directory.",
                parameters=_APPLY_PATCH_SCHEMA,
            ),
            executor=lambda args, _env=env: _apply_patch(args, _env),
        ))

    @property
    def id(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._registry

    def build_system_prompt(self) -> str:
        return _OPENAI_SYSTEM_PROMPT + (
            f"\nPlatform: {self._env.platform}\n"
            f"Working directory: {self._env.working_directory}\n"
        )


# ---------------------------------------------------------------------------
# Gemini profile
# ---------------------------------------------------------------------------

_GEMINI_SYSTEM_PROMPT = """\
You are a coding agent with access to file and shell tools. \
Use them to complete the user's task. Read files before editing. \
Use search tools to explore the codebase. Run tests when possible. \
Only create files necessary for the code to compile and pass tests. \
Do NOT create documentation, summaries, checklists, or status reports unless \
the task explicitly asks for them. Verify by running actual commands, not by \
writing markdown checklists.
"""


class GeminiProfile(ProviderProfile):
    """Gemini CLI-aligned profile."""

    def __init__(self, model_name: str, env: ExecutionEnvironment):
        self._model = model_name
        self._env = env
        self._registry = ToolRegistry()
        for tool in create_core_tools(env):
            self._registry.register(tool)

    @property
    def id(self) -> str:
        return "gemini"

    @property
    def model(self) -> str:
        return self._model

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._registry

    def build_system_prompt(self) -> str:
        return _GEMINI_SYSTEM_PROMPT + (
            f"\nPlatform: {self._env.platform}\n"
            f"Working directory: {self._env.working_directory}\n"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "vertex": "claude-opus-4-6",
    "openai": "gpt-4o",
    "gemini": "gemini-2.5-pro",
}


def create_profile(
    provider: str,
    model: str | None = None,
    env: ExecutionEnvironment | None = None,
) -> ProviderProfile:
    """Create a provider profile.

    Args:
        provider: Provider name ('anthropic', 'openai', 'gemini').
        model: Model name override. Uses provider default if not specified.
        env: Execution environment. Creates a LocalExecutionEnvironment if not provided.
    """
    if env is None:
        from attractor.agent.environment import LocalExecutionEnvironment
        env = LocalExecutionEnvironment()

    model_name = model or _DEFAULT_MODELS.get(provider, "")

    if provider in ("anthropic", "vertex"):
        return AnthropicProfile(model_name, env)
    elif provider == "openai":
        return OpenAIProfile(model_name, env)
    elif provider == "gemini":
        return GeminiProfile(model_name, env)
    else:
        raise ValueError(f"Unknown provider: {provider!r}")
