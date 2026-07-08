"""Captain Claw: self-hosted framework for orchestrating fleets of specialist AI agents — ensemble reasoning and a full agentic coding pipeline, model-agnostic and local-friendly."""

__version__ = "0.7.3"
__build_date__ = "2026-07-07"
__author__ = "Stevica Kuharski"

from captain_claw.config import Config

__all__ = ["Config", "main", "__version__", "__build_date__"]


def __getattr__(name: str):
    """Lazy import for heavy modules to avoid pulling in the full agent stack on package init."""
    if name == "main":
        from captain_claw.main import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
