"""Tests for the compat module.

This module tests:
- _Signature dataclass
- get_signature function for extracting function signature information
"""

from loman.compat import _Signature, get_signature


class TestSignature:
    """Test _Signature dataclass."""

    def test_signature_creation(self):
        """Test that _Signature can be created with expected fields."""
        sig = _Signature(
            kwd_params=["a", "b"],
            default_params=["b"],
            has_var_args=False,
            has_var_kwds=True,
        )

        assert sig.kwd_params == ["a", "b"]
        assert sig.default_params == ["b"]
        assert sig.has_var_args is False
        assert sig.has_var_kwds is True


class TestGetSignature:
    """Test get_signature function."""

    def test_get_signature_simple_function(self):
        """Test get_signature with a simple function."""

        def func(a, b):
            pass

        sig = get_signature(func)

        assert sig.kwd_params == ["a", "b"]
        assert sig.default_params == []
        assert sig.has_var_args is False
        assert sig.has_var_kwds is False

    def test_get_signature_with_defaults(self):
        """Test get_signature with default parameter values."""

        def func(a, b=10, c=20):
            pass

        sig = get_signature(func)

        assert sig.kwd_params == ["a", "b", "c"]
        assert sig.default_params == ["b", "c"]
        assert sig.has_var_args is False
        assert sig.has_var_kwds is False

    def test_get_signature_with_var_args(self):
        """Test get_signature with *args."""

        def func(a, *args):
            pass

        sig = get_signature(func)

        assert sig.kwd_params == ["a"]
        assert sig.default_params == []
        assert sig.has_var_args is True
        assert sig.has_var_kwds is False

    def test_get_signature_with_var_kwds(self):
        """Test get_signature with **kwargs."""

        def func(a, **kwargs):
            pass

        sig = get_signature(func)

        assert sig.kwd_params == ["a"]
        assert sig.default_params == []
        assert sig.has_var_args is False
        assert sig.has_var_kwds is True

    def test_get_signature_with_var_args_and_kwds(self):
        """Test get_signature with both *args and **kwargs."""

        def func(a, *args, **kwargs):
            pass

        sig = get_signature(func)

        assert sig.kwd_params == ["a"]
        assert sig.default_params == []
        assert sig.has_var_args is True
        assert sig.has_var_kwds is True

    def test_get_signature_keyword_only_params(self):
        """Test get_signature with keyword-only parameters."""

        def func(a, *, b, c=10):
            pass

        sig = get_signature(func)

        assert sig.kwd_params == ["a", "b", "c"]
        assert sig.default_params == ["c"]
        assert sig.has_var_args is False
        assert sig.has_var_kwds is False

    def test_get_signature_lambda(self):
        """Test get_signature with a lambda function."""

        def func(x, y=5):
            return x + y

        sig = get_signature(func)

        assert sig.kwd_params == ["x", "y"]
        assert sig.default_params == ["y"]
        assert sig.has_var_args is False
        assert sig.has_var_kwds is False

    def test_get_signature_no_params(self):
        """Test get_signature with a function with no parameters."""

        def func():
            pass

        sig = get_signature(func)

        assert sig.kwd_params == []
        assert sig.default_params == []
        assert sig.has_var_args is False
        assert sig.has_var_kwds is False

    def test_get_signature_complex_function(self):
        """Test get_signature with a complex function signature."""

        def func(a, b, c=1, *args, d, e=2, **kwargs):
            pass

        sig = get_signature(func)

        assert sig.kwd_params == ["a", "b", "c", "d", "e"]
        assert sig.default_params == ["c", "e"]
        assert sig.has_var_args is True
        assert sig.has_var_kwds is True

    def test_get_signature_positional_only(self):
        """Test get_signature with positional-only parameters raises NotImplementedError."""
        import pytest

        # Create a function with positional-only parameters (Python 3.8+)
        # We need to create a function with POSITIONAL_ONLY parameter kind
        exec_globals = {}
        exec("def func_with_pos_only(x, /): pass", exec_globals)
        func = exec_globals["func_with_pos_only"]

        with pytest.raises(NotImplementedError, match="Unexpected param kind"):
            get_signature(func)
