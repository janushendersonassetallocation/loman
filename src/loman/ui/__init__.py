"""Interactive notebook UI for Loman computations.

Importing this package checks for the ``ui`` extra up front, so a missing
dependency fails once with a clear message rather than at some arbitrary later
call. Nothing in :mod:`loman` imports this package at load time.
"""

from loman._extras import require

require("anywidget", "ui")
require("traitlets", "ui")

from .widget import ComputationWidget  # noqa: E402

__all__ = ["ComputationWidget"]
