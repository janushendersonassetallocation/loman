"""Tests for node key functionality and pattern matching in Loman."""

import pytest

from loman.nodekey import NodeKey, is_pattern, match_pattern, nodekey_join, to_nodekey

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


@pytest.mark.parametrize("test_str,expected_path", TEST_DATA)
def test_simple_nodekey_parser(test_str, expected_path):
    assert to_nodekey(test_str) == expected_path


TEST_JOIN_DATA = [
    (to_nodekey("/A"), ["B"], to_nodekey("/A/B")),
    (to_nodekey("/A"), ["B", "C"], to_nodekey("/A/B/C")),
    (to_nodekey("/A"), [to_nodekey("B/C")], to_nodekey("/A/B/C")),
]


@pytest.mark.parametrize("base_path,join_parts,expected_path", TEST_JOIN_DATA)
def test_join_nodekeys(base_path, join_parts, expected_path):
    result = base_path.join(*join_parts)
    assert result == expected_path


def test_add_op():
    res = to_nodekey("/A")
    res1 = to_nodekey("/B/C")
    assert (res+res1) == to_nodekey("/A/B/C")


TEST_JOIN_DATA_2 = [
    (["A", "B"], "A/B"),
    (["A", "B", "C"], "A/B/C"),
    (["A", "B/C"], "A/B/C"),
    (["/A", "B"], "/A/B"),
    (["/A", "B", "C"], "/A/B/C"),
    (["/A", "B/C"], "/A/B/C"),
    (["A", None, "B"], "A/B"),
]


@pytest.mark.parametrize("paths,expected_path", TEST_JOIN_DATA_2)
def test_join_nodekeys_2(paths, expected_path):
    result = nodekey_join(*paths)
    assert result == to_nodekey(expected_path)


TEST_COMMON_PARENT_DATA = [
    ("A", "B", ""),
    ("/A", "/B", "/"),
    ("/A/X", "/A/Y", "/A"),
]


@pytest.mark.parametrize("path1,path2,expected_path", TEST_COMMON_PARENT_DATA)
def test_common_parent(path1, path2, expected_path):
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


@pytest.mark.parametrize("pattern,target,expected", TEST_PATTERN_MATCH_DATA)
def test_pattern_matching(pattern, target, expected):
    pattern_key = NodeKey(pattern)
    target_key = NodeKey(target)
    assert match_pattern(pattern_key, target_key) == expected


def test_is_pattern():
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
