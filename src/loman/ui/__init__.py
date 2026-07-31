"""Interactive notebook UI for Loman computations."""

from loman._extras import require

require("anywidget", "ui")

from .widget import ComputationWidget  # noqa: E402

__all__ = ["ComputationWidget"]
