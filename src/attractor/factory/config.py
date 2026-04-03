"""Factory configuration loaded from factory-config.json."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VerifyConfig:
    """Verification commands for the factory pipeline."""
    test_command: str = "python -m pytest"
    lint_command: str = ""
    typecheck_command: str = ""
    build_command: str = ""


@dataclass
class LimitsConfig:
    """Resource limits for the factory pipeline."""
    max_retries: int = 3
    implement_timeout: int = 300  # seconds
    verify_timeout: int = 120    # seconds
    total_timeout: int = 1800    # seconds
    max_tokens: int = 0          # 0 = unlimited


@dataclass
class FactoryConfig:
    """Top-level configuration for the dark factory.

    Loaded from factory-config.json, with sensible defaults.
    """
    specs_dir: str = "specs"
    output_dir: str = "output"
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    provider: str = "anthropic"
    model: str = ""
    dotfile: str = "pipelines/dark_factory_sequential.dot"

    @classmethod
    def load(cls, path: str = "factory-config.json") -> FactoryConfig:
        """Load configuration from a JSON file.

        Falls back to defaults if the file doesn't exist.
        """
        if not os.path.exists(path):
            return cls()

        with open(path, "r") as f:
            data = json.load(f)

        verify_data = data.get("verify", {})
        verify = VerifyConfig(
            test_command=verify_data.get("test_command", "python -m pytest"),
            lint_command=verify_data.get("lint_command", ""),
            typecheck_command=verify_data.get("typecheck_command", ""),
            build_command=verify_data.get("build_command", ""),
        )

        limits_data = data.get("limits", {})
        limits = LimitsConfig(
            max_retries=limits_data.get("max_retries", 3),
            implement_timeout=limits_data.get("implement_timeout", 300),
            verify_timeout=limits_data.get("verify_timeout", 120),
            total_timeout=limits_data.get("total_timeout", 1800),
            max_tokens=limits_data.get("max_tokens", 0),
        )

        return cls(
            specs_dir=data.get("specs_dir", "specs"),
            output_dir=data.get("output_dir", "output"),
            verify=verify,
            limits=limits,
            provider=data.get("provider", "anthropic"),
            model=data.get("model", ""),
            dotfile=data.get("dotfile", "pipelines/dark_factory_sequential.dot"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "specs_dir": self.specs_dir,
            "output_dir": self.output_dir,
            "verify": {
                "test_command": self.verify.test_command,
                "lint_command": self.verify.lint_command,
                "typecheck_command": self.verify.typecheck_command,
                "build_command": self.verify.build_command,
            },
            "limits": {
                "max_retries": self.limits.max_retries,
                "implement_timeout": self.limits.implement_timeout,
                "verify_timeout": self.limits.verify_timeout,
                "total_timeout": self.limits.total_timeout,
                "max_tokens": self.limits.max_tokens,
            },
            "provider": self.provider,
            "model": self.model,
            "dotfile": self.dotfile,
        }
