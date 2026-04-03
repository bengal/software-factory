"""Dark factory: automated code generation pipeline."""

from attractor.factory.config import FactoryConfig, VerifyConfig, LimitsConfig
from attractor.factory.pipeline import (
    DARK_FACTORY_DOT,
    FactoryBackend,
    IngestHandler,
    VerifyHandler,
    PackageHandler,
    QuarantineHandler,
    create_dark_factory,
    run_factory,
)

__all__ = [
    "FactoryConfig",
    "VerifyConfig",
    "LimitsConfig",
    "DARK_FACTORY_DOT",
    "FactoryBackend",
    "IngestHandler",
    "VerifyHandler",
    "PackageHandler",
    "QuarantineHandler",
    "create_dark_factory",
    "run_factory",
]
