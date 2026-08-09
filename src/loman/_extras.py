"""Helpers for optional Loman features."""

from importlib import import_module
from types import ModuleType


def require(module: str, extra: str) -> ModuleType:
    """Import an optional dependency or explain how to install its extra."""
    try:
        return import_module(module)
    except ImportError as exc:
        msg = f"'{module}' is required for loman's '{extra}' extra.\nInstall it with:  pip install 'loman[{extra}]'"
        raise ImportError(msg) from exc
