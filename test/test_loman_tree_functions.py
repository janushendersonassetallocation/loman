import loman as lm
from loman.nodekey import to_nodekey


def test_list_children():
    comp = lm.Computation()
    comp.add_node("foo1/bar1/baz1/a", value=1)
    comp.add_node("foo1/bar1/baz2/a", value=1)

    assert comp.get_tree_list_children("foo1/bar1") == {"baz1", "baz2"}


def test_has_path_has_node():
    comp = lm.Computation()
    comp.add_node("foo1/bar1/baz1/a", value=1)

    assert comp.has_node("foo1/bar1/baz1/a")
    assert comp.tree_has_path("foo1/bar1/baz1/a")
    assert not comp.has_node("foo1/bar1/baz1")
    assert comp.tree_has_path("foo1/bar1/baz1")
    assert not comp.has_node("foo1/bar1")
    assert comp.tree_has_path("foo1/bar1")


def test_nodekey_ancestors():
    nk = to_nodekey("foo/bar/baz")
    result = set(x.name for x in nk.ancestors())
    assert result == {"foo/bar/baz", "foo/bar", "foo", ""}


def test_tree_descendents():
    comp = lm.Computation()
    comp.add_node("foo/bar/baz")
    comp.add_node("foo/bar2")
    comp.add_node("beef/bar")

    assert comp.get_tree_descendents() == {"foo", "foo/bar", "foo/bar/baz", "foo/bar2", "beef", "beef/bar"}
