"""Tests for optional dependency loading."""

import importlib
import subprocess  # nosec B404
import sys
from unittest.mock import patch

import pytest

from loman._extras import require


def test_require_returns_imported_module():
    """Available optional modules are returned normally."""
    assert require("json", "example") is importlib.import_module("json")


def test_require_explains_missing_extra():
    """A missing dependency names both its module and install command."""
    with (
        patch("loman._extras.import_module", side_effect=ImportError("missing")),
        pytest.raises(ImportError) as exc_info,
    ):
        require("anywidget", "ui")

    assert "'anywidget' is required" in str(exc_info.value)
    assert "pip install 'loman[ui]'" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ImportError)


def test_bare_import_does_not_load_ui_dependencies():
    """The base package remains independent from the optional widget stack."""
    code = "import sys, loman; assert 'anywidget' not in sys.modules; assert 'traitlets' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
