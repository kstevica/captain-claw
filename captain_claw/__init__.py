"""Captain Claw - A powerful console-based AI agent."""

__version__ = "0.4.13"
__build_date__ = "2026-04-04"
__author__ = "Stevica Kuharski"

from captain_claw.config import Config

__all__ = ["Config", "main", "__version__", "__build_date__"]


def __getattr__(name: str):
    """Lazy import for heavy modules to avoid pulling in the full agent stack on package init."""
    if name == "main":
        from captain_claw.main import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
