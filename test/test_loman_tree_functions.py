import loman as lm


def test_list_children():
    comp = lm.Computation()
    comp.add_node('foo1/bar1/baz1/a', value=1)
    comp.add_node('foo1/bar1/baz2/a', value=1)

    assert comp.list_children('foo1/bar1') == {'baz1', 'baz2'}


def test_has_path_has_node():
    comp = lm.Computation()
    comp.add_node('foo1/bar1/baz1/a', value=1)

    assert comp.has_node('foo1/bar1/baz1/a')
    assert comp.has_path('foo1/bar1/baz1/a')
    assert not comp.has_node('foo1/bar1/baz1')
    assert comp.has_path('foo1/bar1/baz1')
    assert not comp.has_node('foo1/bar1')
    assert comp.has_path('foo1/bar1')
