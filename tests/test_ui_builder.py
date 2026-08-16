"""Tests for turning a browser node definition into a real Loman node.

This module tests:
- Reading names relative to the block the widget is rooted on
- Parsing input declarations into a parameter mapping
- Compiling an expression into a function with recoverable source
- Describing an existing node so the form can be reopened on it
"""

import numpy as np
import pytest

from loman import Computation
from loman.computeengine import C
from loman.nodekey import to_nodekey
from loman.ui.builder import (
    MAX_EXPRESSION_LENGTH,
    MAX_INPUTS,
    GraphBuildError,
    build_definition,
    compile_expression,
    default_parameter,
    describe_definition,
    format_inputs,
    parse_inputs,
    relative_name,
    resolve_name,
)
from loman.ui.value import ValueWireError


class TestResolvingNames:
    """A name typed in the browser is read against the view's root."""

    def test_a_name_is_relative_to_the_root(self):
        """Typing a name while inside a block puts the node in that block."""
        assert resolve_name("spot", to_nodekey("market")) == to_nodekey("market/spot")

    def test_a_leading_slash_escapes_the_root(self):
        """Otherwise a node inside a block could never depend on one outside it."""
        assert resolve_name("/rates/curve", to_nodekey("market")) == to_nodekey("rates/curve")

    def test_without_a_root_a_name_is_taken_whole(self):
        """The unrooted widget is the ordinary case, and needs no rewriting."""
        assert resolve_name("market/spot") == to_nodekey("market/spot")

    def test_a_blank_name_is_refused(self):
        """A node with no name is the most likely thing to submit by accident."""
        with pytest.raises(GraphBuildError, match="needs a name"):
            resolve_name("   ", None)

    @pytest.mark.parametrize(
        ("node", "root", "expected"),
        [
            ("market/spot", "market", "spot"),
            ("rates/curve", "market", "/rates/curve"),
            ("market", "market", "/market"),
            ("market/spot", None, "market/spot"),
        ],
    )
    def test_names_render_the_way_they_are_read_back(self, node, root, expected):
        """Whatever the panel shows has to be something the form would accept."""
        root_key = None if root is None else to_nodekey(root)
        assert relative_name(to_nodekey(node), root_key) == expected

    def test_rendering_and_resolving_are_inverses(self):
        """Round-tripping is the property the two functions exist to keep."""
        root = to_nodekey("market")
        for node in ("market/spot", "rates/curve", "top"):
            node_key = to_nodekey(node)
            assert resolve_name(relative_name(node_key, root), root) == node_key


class TestParsingInputs:
    """Input declarations become the function's parameters and the graph's edges."""

    def test_a_bare_name_is_named_after_its_node(self):
        """Which is also how Loman resolves a parameter to a sibling node."""
        assert parse_inputs(["price", "quantity"]) == {
            "price": to_nodekey("price"),
            "quantity": to_nodekey("quantity"),
        }

    def test_a_parameter_can_be_named_explicitly(self):
        """A path's last part is rarely what the expression wants to call it."""
        assert parse_inputs(["spot=market/spot"]) == {"spot": to_nodekey("market/spot")}

    def test_a_deep_name_lends_its_last_part(self):
        """``market/spot`` arrives as ``spot`` without having to be told."""
        assert parse_inputs(["market/spot"]) == {"spot": to_nodekey("market/spot")}

    def test_blank_entries_are_skipped(self):
        """The browser sends its rows as they stand, including the empty one."""
        assert parse_inputs(["price", "", "   "]) == {"price": to_nodekey("price")}

    def test_entries_are_relative_to_the_root(self):
        """The same rule as the node's own name, or the two would disagree."""
        assert parse_inputs(["spot"], to_nodekey("market")) == {"spot": to_nodekey("market/spot")}

    def test_a_name_that_cannot_be_a_parameter_must_be_given_one(self):
        """A numeric last part is a real Loman name and an impossible parameter."""
        with pytest.raises(GraphBuildError, match="cannot be a parameter name"):
            parse_inputs(["series/1"])

    def test_an_unusable_parameter_name_is_refused(self):
        """Including Python keywords, which would compile to a syntax error."""
        with pytest.raises(GraphBuildError, match="not a usable parameter name"):
            parse_inputs(["class=market/spot"])

    def test_two_inputs_cannot_share_a_parameter(self):
        """Silently keeping the last would drop an edge the user asked for."""
        with pytest.raises(GraphBuildError, match="both arrive as spot"):
            parse_inputs(["market/spot", "futures/spot"])

    def test_there_is_a_ceiling_on_inputs(self):
        """A node with this many parameters is not being written in a text box."""
        with pytest.raises(GraphBuildError, match=f"at most {MAX_INPUTS}"):
            parse_inputs([f"input_{i}" for i in range(MAX_INPUTS + 1)])

    @pytest.mark.parametrize(("name", "expected"), [("spot", "spot"), ("1", ""), ("two words", "")])
    def test_only_an_identifier_can_be_an_implied_parameter(self, name, expected):
        """The check the explicit form exists to work around."""
        assert default_parameter(to_nodekey(f"market/{name}")) == expected


class TestCompilingAnExpression:
    """The expression becomes the body of a real function."""

    def test_the_function_computes_what_was_typed(self):
        """The whole point, and the only thing a user actually judges it on."""
        func = compile_expression("price * quantity", ["price", "quantity"], node_key=to_nodekey("value"))
        assert func(price=10.0, quantity=3) == 30.0

    def test_the_source_can_be_recovered(self):
        """A UI-built node that cannot show its own source is a black box.

        ``inspect`` reads through ``linecache``, so the source is registered
        there under a filename with no file behind it.
        """
        comp = Computation()
        func = compile_expression("price * 2", ["price"], node_key=to_nodekey("double"))
        comp.add_node("price", value=4.0)
        comp.add_node("double", func, kwds={"price": "price"})
        assert "price * 2" in comp.get_source("double")

    def test_a_multi_line_expression_stays_one_expression(self):
        """The form is a textarea, so this is what a long definition looks like."""
        func = compile_expression("(\n  a\n  + b\n)", ["a", "b"], node_key=to_nodekey("total"))
        assert func(a=1, b=2) == 3

    def test_the_notebook_namespace_is_available(self):
        """Otherwise every expression is limited to arithmetic on builtins."""
        func = compile_expression("np.sqrt(x)", ["x"], node_key=to_nodekey("root"), namespace={"np": np})
        assert func(x=16.0) == 4.0

    def test_the_namespace_is_not_polluted(self):
        """A widget that quietly rebinds a notebook name would be a menace."""
        namespace = {"np": np}
        compile_expression("1", [], node_key=to_nodekey("value"), namespace=namespace)
        assert set(namespace) == {"np", "__builtins__"}

    def test_an_import_made_later_is_still_visible(self):
        """Defining a node before importing what it needs must keep working."""
        namespace = {}
        func = compile_expression("np.pi", [], node_key=to_nodekey("value"), namespace=namespace)
        namespace["np"] = np
        assert func() == np.pi

    def test_a_name_that_is_not_an_identifier_still_compiles(self):
        """``market/1`` is a real Loman name and an impossible function name."""
        func = compile_expression("2", [], node_key=to_nodekey("market/1"))
        assert func() == 2

    def test_a_blank_expression_is_refused(self):
        """The likeliest thing to submit by accident on a calculation node."""
        with pytest.raises(GraphBuildError, match="needs an expression"):
            compile_expression("  ", [], node_key=to_nodekey("value"))

    def test_a_statement_is_refused_as_an_expression(self):
        """Pasting a whole function in is the obvious thing to try."""
        with pytest.raises(GraphBuildError, match="does not parse"):
            compile_expression("x = 1", [], node_key=to_nodekey("value"))

    def test_a_syntax_error_is_reported_rather_than_raised(self):
        """Half-typed expressions are the normal state of a form being filled."""
        with pytest.raises(GraphBuildError, match="does not parse"):
            compile_expression("price *", ["price"], node_key=to_nodekey("value"))

    def test_an_oversized_expression_is_refused(self):
        """A runaway paste should be turned away rather than compiled."""
        with pytest.raises(GraphBuildError, match="over the limit"):
            compile_expression("1 + " * MAX_EXPRESSION_LENGTH + "1", [], node_key=to_nodekey("value"))


class TestBuildingADefinition:
    """One browser payload becomes one node ready to be added."""

    def test_an_input_node_can_start_with_a_value(self):
        """The common case: declare an input and seed it in one go."""
        definition = build_definition(
            {"name": "price", "kind": "input", "value": {"kind": "scalar", "type": "float", "value": 1.5}}
        )
        assert (definition.key, definition.value, definition.has_value) == (to_nodekey("price"), 1.5, True)

    def test_an_input_node_can_start_with_nothing(self):
        """Which is what UNINITIALIZED is for."""
        definition = build_definition({"name": "price", "kind": "input"})
        assert definition.has_value is False
        assert definition.func is None

    def test_a_calculation_node_carries_its_function_and_edges(self):
        """The parameter mapping is what Loman turns into edges."""
        definition = build_definition({"name": "value", "kind": "calc", "inputs": ["price"], "expression": "price * 2"})
        assert definition.kwds == {"price": to_nodekey("price")}
        assert definition.func(price=2) == 4

    def test_an_unknown_kind_is_refused(self):
        """The browser is untrusted, so the kind is checked rather than assumed."""
        with pytest.raises(GraphBuildError, match="not 'block'"):
            build_definition({"name": "value", "kind": "block"})

    def test_a_value_that_is_not_a_scalar_is_refused(self):
        """The wire format's own rule, which the form does not get to bypass."""
        with pytest.raises(ValueWireError):
            build_definition({"name": "price", "kind": "input", "value": {"kind": "table"}})

    def test_a_definition_adds_itself_to_a_computation(self):
        """The last step, and the one the widget's request handler leans on."""
        comp = Computation()
        comp.add_node("price", value=3.0)
        build_definition({"name": "double", "kind": "calc", "inputs": ["price"], "expression": "price * 2"}).apply(comp)
        comp.compute_all()
        assert comp.v["double"] == 6.0


class TestDescribingADefinition:
    """Reopening the form on a node means describing how it is defined."""

    def test_an_input_node_describes_itself_as_one(self):
        """There is nothing to reproduce, so it is always editable."""
        comp = Computation()
        comp.add_node("price", value=1.0)
        described = describe_definition(comp, to_nodekey("price"))
        assert described["kind"] == "input"
        assert described["editable"] is True

    def test_a_node_built_here_round_trips(self):
        """What was typed comes back, which is what makes editing editing."""
        comp = Computation()
        comp.add_node("price", value=2.0)
        build_definition({"name": "double", "kind": "calc", "inputs": ["price"], "expression": "price * 2"}).apply(comp)
        described = describe_definition(comp, to_nodekey("double"))
        assert described["expression"] == "price * 2"
        assert described["inputs"] == ["price"]
        assert described["editable"] is True

    def test_a_function_written_in_python_is_not_offered_for_editing(self):
        """Its body is not an expression this form could put back unchanged."""
        comp = Computation()
        comp.add_node("price", value=2.0)
        comp.add_node("double", lambda price: price * 2)
        described = describe_definition(comp, to_nodekey("double"))
        assert described["kind"] == "calc"
        assert described["expression"] is None
        assert described["editable"] is False

    def test_positional_arguments_are_not_offered_for_editing(self):
        """The form has no field for them, so it cannot promise to keep them."""
        comp = Computation()
        comp.add_node("price", value=2.0)
        comp.add_node("double", lambda x: x * 2, args=["price"])
        assert describe_definition(comp, to_nodekey("double"))["editable"] is False

    def test_constant_arguments_are_not_offered_for_editing(self):
        """A constant is not a node, and the inputs field only names nodes."""
        comp = Computation()
        comp.add_node("price", value=2.0)
        comp.add_node("scaled", lambda price, factor: price * factor, kwds={"factor": C(3)})
        described = describe_definition(comp, to_nodekey("scaled"))
        assert described["inputs"] == ["price"]
        assert described["editable"] is False

    def test_inputs_are_described_relative_to_the_root(self):
        """So the description can be handed straight back as a definition."""
        comp = Computation()
        comp.add_node("market/spot", value=1.0)
        comp.add_node("rates/curve", value=2.0)
        build_definition(
            {"name": "value", "kind": "calc", "inputs": ["spot", "/rates/curve"], "expression": "spot + curve"},
            root=to_nodekey("market"),
        ).apply(comp)
        described = describe_definition(comp, to_nodekey("market/value"), to_nodekey("market"))
        assert described["name"] == "value"
        assert sorted(described["inputs"]) == ["/rates/curve", "spot"]

    def test_a_description_parses_back_into_the_same_mapping(self):
        """The round trip the form depends on, stated as a property."""
        kwds = {"spot": to_nodekey("market/spot"), "curve": to_nodekey("rates/curve")}
        assert parse_inputs(format_inputs(kwds)) == kwds
