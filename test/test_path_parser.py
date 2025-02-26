import pytest
from loman.path_parser import to_path, Path, path_join, path_common_parent

TEST_DATA = [
    ('/A', Path(('A', ), is_absolute_path=True)),
    ('A', Path(('A', ), is_absolute_path=False)),
    ('/foo/bar', Path(('foo', 'bar'), is_absolute_path=True)),
    ('foo/bar', Path(('foo', 'bar'), is_absolute_path=False)),
    ('/foo/"bar"', Path(('foo', 'bar'), is_absolute_path=True)),
    ('foo/"bar"', Path(('foo', 'bar'), is_absolute_path=False)),
    ('/foo/"bar"/baz', Path(('foo', 'bar', 'baz'), is_absolute_path=True)),
    ('foo/"bar"/baz', Path(('foo', 'bar', 'baz'), is_absolute_path=False)),
]


@pytest.mark.parametrize("test_str,expected_path", TEST_DATA)
def test_simple_path_parser(test_str, expected_path):
    assert to_path(test_str) == expected_path


TEST_JOIN_DATA = [
    (to_path('/A'), ['B'], to_path('/A/B')),
    (to_path('/A'), ['B', 'C'], to_path('/A/B/C')),
    (to_path('/A'), [to_path('B/C')], to_path('/A/B/C')),
]

@pytest.mark.parametrize("base_path,join_parts,expected_path", TEST_JOIN_DATA)
def test_join_paths(base_path, join_parts, expected_path):
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
def test_join_paths_2(paths, expected_path):
    result = path_join(*paths)
    assert result == to_path(expected_path)

TEST_COMMON_PARENT_DATA = [
    ('A', 'B', ''),
    ('/A', '/B', '/'),
    ('/A/X', '/A/Y', '/A'),
]
@pytest.mark.parametrize("path1,path2,expected_path", TEST_COMMON_PARENT_DATA)
def test_common_parent(path1, path2, expected_path):
    result = path_common_parent(path1, path2)
    assert result == to_path(expected_path)