"""Execution environment abstraction for the coding agent."""

from __future__ import annotations

import fnmatch
import os
import pathlib
import re
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ExecResult:
    """Result of executing a command."""
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False
    duration_ms: float = 0.0


@dataclass
class DirEntry:
    """Entry in a directory listing."""
    name: str
    is_dir: bool
    size: int = 0


# ---------------------------------------------------------------------------
# Environment variable filtering
# ---------------------------------------------------------------------------

_SENSITIVE_PATTERNS = [
    "*_API_KEY",
    "*_SECRET",
    "*_TOKEN",
    "*_PASSWORD",
    "*_CREDENTIAL*",
    "*_PRIVATE_KEY",
    "AWS_ACCESS_KEY*",
    "AWS_SECRET*",
    "GITHUB_TOKEN",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
]


def _is_sensitive_env_var(name: str) -> bool:
    """Check if an environment variable name matches a sensitive pattern."""
    upper = name.upper()
    return any(fnmatch.fnmatch(upper, pat) for pat in _SENSITIVE_PATTERNS)


def _filter_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return a copy of the environment with sensitive values removed."""
    source = env if env is not None else dict(os.environ)
    return {k: v for k, v in source.items() if not _is_sensitive_env_var(k)}


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class ExecutionEnvironment(ABC):
    """Abstract interface for the execution environment.

    Provides file I/O, command execution, search, and platform info.
    """

    @abstractmethod
    def read_file(self, path: str) -> str:
        """Read a file and return its contents."""
        ...

    @abstractmethod
    def write_file(self, path: str, content: str) -> None:
        """Write content to a file, creating parent dirs as needed."""
        ...

    @abstractmethod
    def file_exists(self, path: str) -> bool:
        """Check if a file or directory exists."""
        ...

    @abstractmethod
    def list_directory(self, path: str) -> list[DirEntry]:
        """List entries in a directory."""
        ...

    @abstractmethod
    def exec_command(
        self,
        command: str,
        *,
        timeout_ms: int = 120_000,
        working_dir: str | None = None,
    ) -> ExecResult:
        """Execute a shell command."""
        ...

    @abstractmethod
    def grep(
        self,
        pattern: str,
        path: str | None = None,
        *,
        include: str | None = None,
        max_results: int = 100,
    ) -> str:
        """Search file contents for a pattern."""
        ...

    @abstractmethod
    def glob(self, pattern: str, path: str | None = None) -> list[str]:
        """Find files matching a glob pattern."""
        ...

    @property
    @abstractmethod
    def working_directory(self) -> str:
        """Return the current working directory."""
        ...

    @property
    @abstractmethod
    def platform(self) -> str:
        """Return the platform identifier (e.g., 'linux', 'darwin')."""
        ...


# ---------------------------------------------------------------------------
# Local implementation
# ---------------------------------------------------------------------------

class LocalExecutionEnvironment(ExecutionEnvironment):
    """Execution environment backed by the local filesystem and shell."""

    def __init__(self, working_dir: str | None = None):
        self._working_dir = working_dir or os.getcwd()

    def _resolve(self, path: str) -> str:
        """Resolve a path relative to the working directory."""
        p = pathlib.Path(path)
        if not p.is_absolute():
            p = pathlib.Path(self._working_dir) / p
        return str(p.resolve())

    # -- File operations ---------------------------------------------------

    def read_file(self, path: str) -> str:
        resolved = self._resolve(path)
        with open(resolved, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def write_file(self, path: str, content: str) -> None:
        resolved = self._resolve(path)
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(content)

    def file_exists(self, path: str) -> bool:
        return os.path.exists(self._resolve(path))

    def list_directory(self, path: str) -> list[DirEntry]:
        resolved = self._resolve(path)
        entries: list[DirEntry] = []
        try:
            for entry in os.scandir(resolved):
                try:
                    size = entry.stat().st_size if entry.is_file() else 0
                except OSError:
                    size = 0
                entries.append(DirEntry(
                    name=entry.name,
                    is_dir=entry.is_dir(),
                    size=size,
                ))
        except OSError:
            pass
        return sorted(entries, key=lambda e: (not e.is_dir, e.name))

    # -- Command execution -------------------------------------------------

    def exec_command(
        self,
        command: str,
        *,
        timeout_ms: int = 120_000,
        working_dir: str | None = None,
    ) -> ExecResult:
        cwd = self._resolve(working_dir) if working_dir else self._working_dir
        timeout_s = timeout_ms / 1000.0
        timed_out = False
        start = time.monotonic()

        env = _filter_env()
        # Preserve PATH and HOME
        env["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")
        env["HOME"] = os.environ.get("HOME", "")

        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                env=env,
                start_new_session=True,
            )
            try:
                stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                # Try SIGTERM first
                pgid = os.getpgid(proc.pid)
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except OSError:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except OSError:
                        pass
                    proc.wait(timeout=5)
                stdout_bytes = proc.stdout.read() if proc.stdout else b""
                stderr_bytes = proc.stderr.read() if proc.stderr else b""

            duration_ms = (time.monotonic() - start) * 1000
            return ExecResult(
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                exit_code=proc.returncode or 0,
                timed_out=timed_out,
                duration_ms=duration_ms,
            )

        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            return ExecResult(
                stdout="",
                stderr=str(exc),
                exit_code=1,
                timed_out=False,
                duration_ms=duration_ms,
            )

    # -- Search ------------------------------------------------------------

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        *,
        include: str | None = None,
        max_results: int = 100,
    ) -> str:
        search_path = self._resolve(path) if path else self._working_dir
        # Try ripgrep first
        try:
            return self._grep_rg(pattern, search_path, include, max_results)
        except (FileNotFoundError, OSError):
            pass
        # Fallback to Python re
        return self._grep_python(pattern, search_path, include, max_results)

    def _grep_rg(
        self, pattern: str, path: str, include: str | None, max_results: int,
    ) -> str:
        cmd = ["rg", "--no-heading", "--line-number", "--max-count", str(max_results)]
        if include:
            cmd.extend(["--glob", include])
        cmd.extend([pattern, path])
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout

    def _grep_python(
        self, pattern: str, path: str, include: str | None, max_results: int,
    ) -> str:
        compiled = re.compile(pattern)
        matches: list[str] = []
        search_root = pathlib.Path(path)

        if search_root.is_file():
            files = [search_root]
        else:
            glob_pattern = include or "**/*"
            files = list(search_root.glob(glob_pattern))

        for file_path in files:
            if not file_path.is_file():
                continue
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    for line_num, line in enumerate(f, 1):
                        if compiled.search(line):
                            rel = file_path.relative_to(search_root) if search_root.is_dir() else file_path
                            matches.append(f"{rel}:{line_num}:{line.rstrip()}")
                            if len(matches) >= max_results:
                                return "\n".join(matches)
            except (OSError, UnicodeDecodeError):
                continue

        return "\n".join(matches)

    # -- Glob --------------------------------------------------------------

    def glob(self, pattern: str, path: str | None = None) -> list[str]:
        root = pathlib.Path(self._resolve(path) if path else self._working_dir)
        results: list[str] = []
        for match in root.glob(pattern):
            try:
                results.append(str(match.relative_to(root)))
            except ValueError:
                results.append(str(match))
        return sorted(results)

    # -- Properties --------------------------------------------------------

    @property
    def working_directory(self) -> str:
        return self._working_dir

    @property
    def platform(self) -> str:
        import sys
        return sys.platform
