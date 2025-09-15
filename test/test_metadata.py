import loman as lm


def test_simple_node_metadata():
    comp = lm.Computation()
    comp.add_node("foo", metadata={"test": "working"})
    assert comp.metadata("foo")["test"] == "working"


def test_simple_computation_metadata():
    comp = lm.Computation(metadata={"test": "working"})
    assert comp.metadata("")["test"] == "working"


def test_setting_node_metadata():
    comp = lm.Computation()
    comp.add_node("foo")
    comp.metadata("foo")["test"] = "working"
    assert comp.metadata("foo")["test"] == "working"


def test_setting_block_metadata():
    comp = lm.Computation()
    comp.add_node("foo/bar")
    comp.metadata("foo")["test"] = "working"
    assert comp.metadata("foo")["test"] == "working"


def test_setting_computation_block_metadata():
    """Test setting metadata on computation blocks."""
    comp_inner = lm.Computation()
    comp_inner.add_node("bar")

    comp = lm.Computation()
    comp.add_block("foo", comp_inner, metadata={"test": "working"})
    assert comp.metadata("foo")["test"] == "working"
