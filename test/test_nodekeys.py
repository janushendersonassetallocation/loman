import pytest
from loman.nodekey import to_nodekey, NodeKey, nodekey_join

TEST_DATA = [
    ('/A', NodeKey(('A', ))),
    ('A', NodeKey(('A', ))),
    ('/foo/bar', NodeKey(('foo', 'bar'))),
    ('foo/bar', NodeKey(('foo', 'bar'))),
    ('/foo/"bar"', NodeKey(('foo', 'bar'))),
    ('foo/"bar"', NodeKey(('foo', 'bar'))),
    ('/foo/"bar"/baz', NodeKey(('foo', 'bar', 'baz'))),
    ('foo/"bar"/baz', NodeKey(('foo', 'bar', 'baz'))),
]


@pytest.mark.parametrize("test_str,expected_path", TEST_DATA)
def test_simple_nodekey_parser(test_str, expected_path):
    assert to_nodekey(test_str) == expected_path


TEST_JOIN_DATA = [
    (to_nodekey('/A'), ['B'], to_nodekey('/A/B')),
    (to_nodekey('/A'), ['B', 'C'], to_nodekey('/A/B/C')),
    (to_nodekey('/A'), [to_nodekey('B/C')], to_nodekey('/A/B/C')),
]

@pytest.mark.parametrize("base_path,join_parts,expected_path", TEST_JOIN_DATA)
def test_join_nodekeys(base_path, join_parts, expected_path):
    result = base_path.join(*join_parts)
    assert result == expected_path


TEST_JOIN_DATA_2 = [
    (['A', 'B'], 'A/B'),
    (['A', 'B', 'C'], 'A/B/C'),
    (['A', 'B/C'], 'A/B/C'),
    (['/A', 'B'], '/A/B'),
    (['/A', 'B', 'C'], '/A/B/C'),
    (['/A', 'B/C'], '/A/B/C'),
    (['A', None, 'B'], 'A/B'),
]
@pytest.mark.parametrize("paths,expected_path", TEST_JOIN_DATA_2)
def test_join_nodekeys_2(paths, expected_path):
    result = nodekey_join(*paths)
    assert result == to_nodekey(expected_path)

TEST_COMMON_PARENT_DATA = [
    ('A', 'B', ''),
    ('/A', '/B', '/'),
    ('/A/X', '/A/Y', '/A'),
]
@pytest.mark.parametrize("path1,path2,expected_path", TEST_COMMON_PARENT_DATA)
def test_common_parent(path1, path2, expected_path):
    result = NodeKey.common_parent(path1, path2)
    assert result == to_nodekey(expected_path)