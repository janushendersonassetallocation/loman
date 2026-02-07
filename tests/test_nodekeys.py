"""Tests for node key functionality and pattern matching in Loman."""

import pytest

from loman.nodekey import (
    NodeKey,
    PathNotFoundError,
    is_pattern,
    match_pattern,
    nodekey_join,
    quote_part,
    to_nodekey,
)

TEST_DATA = [
    ("/A", NodeKey(("A",))),
    ("A", NodeKey(("A",))),
    ("/foo/bar", NodeKey(("foo", "bar"))),
    ("foo/bar", NodeKey(("foo", "bar"))),
    ('/foo/"bar"', NodeKey(("foo", "bar"))),
    ('foo/"bar"', NodeKey(("foo", "bar"))),
    ('/foo/"bar"/baz', NodeKey(("foo", "bar", "baz"))),
    ('foo/"bar"/baz', NodeKey(("foo", "bar", "baz"))),
]


@pytest.mark.parametrize(("test_str", "expected_path"), TEST_DATA)
def test_simple_nodekey_parser(test_str, expected_path):
    """Test simple nodekey parser."""
    assert to_nodekey(test_str) == expected_path


TEST_JOIN_DATA = [
    (to_nodekey("/A"), ["B"], to_nodekey("/A/B")),
    (to_nodekey("/A"), ["B", "C"], to_nodekey("/A/B/C")),
    (to_nodekey("/A"), [to_nodekey("B/C")], to_nodekey("/A/B/C")),
]


@pytest.mark.parametrize(("base_path", "join_parts", "expected_path"), TEST_JOIN_DATA)
def test_join_nodekeys(base_path, join_parts, expected_path):
    """Test join nodekeys."""
    result = base_path.join(*join_parts)
    assert result == expected_path


TEST_ADD_DATA = [
    (to_nodekey("/A"), "B", to_nodekey("/A/B")),
    (to_nodekey("/A"), to_nodekey("B/C"), to_nodekey("/A/B/C")),
]


@pytest.mark.parametrize(("this", "other", "path_expected"), TEST_ADD_DATA)
def test_div_op(this, other, path_expected):
    """Test div op."""
    assert this / other == path_expected


TEST_JOIN_DATA_2 = [
    (["A", "B"], "A/B"),
    (["A", "B", "C"], "A/B/C"),
    (["A", "B/C"], "A/B/C"),
    (["/A", "B"], "/A/B"),
    (["/A", "B", "C"], "/A/B/C"),
    (["/A", "B/C"], "/A/B/C"),
    (["A", None, "B"], "A/B"),
]


@pytest.mark.parametrize(("paths", "expected_path"), TEST_JOIN_DATA_2)
def test_join_nodekeys_2(paths, expected_path):
    """Test join nodekeys 2."""
    result = nodekey_join(*paths)
    assert result == to_nodekey(expected_path)


TEST_COMMON_PARENT_DATA = [
    ("A", "B", ""),
    ("/A", "/B", "/"),
    ("/A/X", "/A/Y", "/A"),
]


@pytest.mark.parametrize(("path1", "path2", "expected_path"), TEST_COMMON_PARENT_DATA)
def test_common_parent(path1, path2, expected_path):
    """Test common parent."""
    result = NodeKey.common_parent(path1, path2)
    assert result == to_nodekey(expected_path)


TEST_PATTERN_MATCH_DATA = [
    (("a", "*", "c"), ("a", "b", "c"), True),
    (("a", "**", "d"), ("a", "b", "c", "d"), True),
    (("a", "**", "d"), ("a", "d"), True),
    (("**",), ("a", "b", "c"), True),
    (("a", "**"), ("a",), True),
    (("a", "*", "**"), ("a", "x", "y", "z"), True),
    (("a", "*", "c"), ("a", "b", "d"), False),
    (("a", "**", "d"), ("a", "b", "c"), False),
    (("a", "*"), ("a", "b", "c"), False),
]


@pytest.mark.parametrize(("pattern", "target", "expected"), TEST_PATTERN_MATCH_DATA)
def test_pattern_matching(pattern, target, expected):
    """Test pattern matching."""
    pattern_key = NodeKey(pattern)
    target_key = NodeKey(target)
    assert match_pattern(pattern_key, target_key) == expected


def test_is_pattern():
    """Test is pattern."""
    # Test single asterisk patterns
    assert is_pattern(NodeKey(("*",)))
    assert is_pattern(NodeKey(("abc", "*")))
    assert is_pattern(NodeKey(("*", "def")))

    # Test double asterisk patterns
    assert is_pattern(NodeKey(("**",)))
    assert is_pattern(NodeKey(("abc", "**")))
    assert is_pattern(NodeKey(("**", "def")))

    # Test non-patterns
    assert not is_pattern(NodeKey(()))
    assert not is_pattern(NodeKey(("abc",)))
    assert not is_pattern(NodeKey(("abc", "def")))

    # Test complex patterns
    assert is_pattern(NodeKey(("abc", "*", "**", "def")))
    assert is_pattern(NodeKey(("**", "*", "def")))
    assert is_pattern(NodeKey(("abc", "**", "*")))


def test_nodekey_ancestors():
    """Test NodeKey ancestors method."""
    nk = to_nodekey("foo/bar/baz")
    result = {x.name for x in nk.ancestors()}
    assert result == {"foo/bar/baz", "foo/bar", "foo", ""}


class TestNodeKeyCoverage:
    """Additional tests for nodekey.py coverage."""

    def test_quote_part_with_slash(self):
        """Test quote_part with a string containing slash."""
        result = quote_part("foo/bar")
        assert result == '"foo/bar"'

    def test_nodekey_label_empty(self):
        """Test NodeKey label property with empty parts."""
        nk = NodeKey(())
        assert nk.label == ""

    def test_nodekey_name_with_mixed_parts(self):
        """Test NodeKey name property with non-string parts."""
        obj = object()
        nk = NodeKey(("a", obj))
        # When parts contain non-strings, name returns self
        assert nk.name is nk

    def test_nodekey_parent_of_root(self):
        """Test getting parent of root NodeKey raises PathNotFoundError."""
        nk = NodeKey(())
        with pytest.raises(PathNotFoundError):
            _ = nk.parent

    def test_nodekey_eq_with_none(self):
        """Test NodeKey equality with None."""
        nk = NodeKey(("a",))
        assert nk != None  # noqa: E711

    def test_nodekey_eq_with_non_nodekey(self):
        """Test NodeKey equality with non-NodeKey returns NotImplemented."""
        nk = NodeKey(("a",))
        result = nk.__eq__("not a nodekey")
        assert result is NotImplemented

    def test_nodekey_root_singleton(self):
        """Test NodeKey.root() returns singleton."""
        r1 = NodeKey.root()
        r2 = NodeKey.root()
        assert r1 is r2

    def test_to_nodekey_value_error_unreachable(self):
        """Test to_nodekey - the else branch is unreachable but test the object path."""
        # This tests line 216 - the object path that's always taken for non-str/non-NodeKey
        obj = object()
        nk = to_nodekey(obj)
        assert nk.parts == (obj,)

    def test_drop_root_not_descendant(self):
        """Test drop_root returns None when not a descendant."""
        nk = NodeKey(("a", "b"))
        root = NodeKey(("x", "y"))

        result = nk.drop_root(root)
        assert result is None

    def test_is_descendent_of_true(self):
        """Test is_descendent_of returns True for descendant."""
        parent = NodeKey(("a",))
        child = NodeKey(("a", "b"))

        assert child.is_descendent_of(parent)

    def test_is_descendent_of_false_equal(self):
        """Test is_descendent_of returns False for equal keys."""
        nk1 = NodeKey(("a",))
        nk2 = NodeKey(("a",))

        assert not nk1.is_descendent_of(nk2)

    def test_join_parts_empty(self):
        """Test join_parts with no parts returns self."""
        nk = NodeKey(("a", "b"))
        result = nk.join_parts()
        assert result is nk

    def test_join_parts_with_parts(self):
        """Test join_parts adds parts."""
        nk = NodeKey(("a",))
        result = nk.join_parts("b", "c")
        assert result.parts == ("a", "b", "c")

    def test_truediv(self):
        """Test / operator for joining node keys."""
        nk1 = NodeKey(("a",))
        nk2 = NodeKey(("b",))

        result = nk1 / nk2
        assert result.parts == ("a", "b")

    def test_drop_root_not_descendent(self):
        """Test drop_root returns None when not descendant."""
        nk = NodeKey(("a", "b"))
        root = NodeKey(("c",))

        result = nk.drop_root(root)
        assert result is None
