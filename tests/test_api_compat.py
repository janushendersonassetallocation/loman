"""Backwards compatibility of the public API and of files written by old releases.

The serialization rework added a lot. This module exists to check it *only*
added: that every name and parameter a released version of loman offered is
still there and still means the same thing, and that files written by earlier
format versions still load.

Two separate promises, worth keeping apart:

*Source compatibility* --- existing code keeps importing, calling and running.
Locked by the inventory below, which is not generated: adding to it is how a new
name is blessed, and removing a line from it is how a removal becomes a decision
someone made rather than a side effect.

*File compatibility* --- old files still load under new code. Version 1 and 2
fixtures live in ``test_serialization.py``; this module covers the API side and
the shape of the JSON that ``write_json`` still produces.

Note the direction that is *not* promised: a file written by this version is not
readable by a release that predates format version 3.
"""

import inspect
import io

import pytest

from loman import Computation

# Every public name released before the serialization rework, with the callables
# that could be reached on it. New names are deliberately absent: this is a
# floor, not an inventory of everything that exists now.
RELEASED_COMPUTATION_METHODS = [
    "add_node",
    "compute",
    "compute_all",
    "copy",
    "delete_node",
    "get_timing",
    "insert",
    "insert_from",
    "insert_many",
    "pin",
    "print_errors",
    "read_dill",
    "read_json",
    "rename_node",
    "restrict",
    "set_tag",
    "clear_tag",
    "state",
    "to_dict",
    "unpin",
    "value",
    "write_dill",
    "write_dill_old",
    "write_json",
]

RELEASED_TOP_LEVEL_NAMES = [
    "C",
    "CannotInsertToPlaceholderNodeError",
    "Computation",
    "ComputationFactory",
    "ComputationSerializer",
    "MapError",
    "NodeKey",
    "SerializationError",
    "States",
    "ValidationError",
    "calc_node",
    "computation_factory",
    "input_node",
    "node",
]

RELEASED_SERIALIZATION_NAMES = [
    "ComputationSerializer",
    "CustomTransformer",
    "DillFunctionTransformer",
    "MissingObject",
    "NdArrayTransformer",
    "Transformable",
    "Transformer",
    "UnrecognizedTypeException",
    "UntransformableTypeException",
    "default_transformer",
]

# Parameters each released method accepted. A new keyword-only parameter with a
# default is additive and fine; losing one of these, or making one required, is
# not.
RELEASED_SIGNATURES = {
    "add_node": [
        "name",
        "func",
        "args",
        "kwds",
        "value",
        "converter",
        "serialize",
        "inspect",
        "group",
        "tags",
        "style",
        "executor",
        "metadata",
    ],
    "write_json": ["file_", "serializer"],
    "read_json": ["file_", "serializer"],
    "write_dill": ["file_"],
    "read_dill": ["file_"],
    "write_dill_old": ["file_"],
}


class TestPublicNamesSurvive:
    """Nothing a released version exported has gone away."""

    @pytest.mark.parametrize("name", RELEASED_TOP_LEVEL_NAMES)
    def test_top_level_name(self, name):
        """Each released top-level name is still importable from loman."""
        import loman

        assert hasattr(loman, name), f"loman.{name} was removed"

    @pytest.mark.parametrize("name", RELEASED_COMPUTATION_METHODS)
    def test_computation_method(self, name):
        """Each released Computation method still exists and is callable."""
        assert callable(getattr(Computation, name, None)), f"Computation.{name} was removed"

    @pytest.mark.parametrize("name", RELEASED_SERIALIZATION_NAMES)
    def test_serialization_name(self, name):
        """Each released loman.serialization name is still exported."""
        import loman.serialization as ser

        assert hasattr(ser, name), f"loman.serialization.{name} was removed"
        assert name in ser.__all__, f"loman.serialization.{name} dropped out of __all__"


class TestSignaturesOnlyGrew:
    """Released parameters are still accepted, and still optional."""

    @pytest.mark.parametrize("method", sorted(RELEASED_SIGNATURES))
    def test_parameters_survive(self, method):
        """Every parameter a released version took is still accepted."""
        parameters = inspect.signature(getattr(Computation, method)).parameters

        for name in RELEASED_SIGNATURES[method]:
            assert name in parameters, f"Computation.{method} no longer accepts {name!r}"

    @pytest.mark.parametrize("method", sorted(RELEASED_SIGNATURES))
    def test_no_new_required_parameters(self, method):
        """A new parameter must have a default, or existing calls break."""
        parameters = inspect.signature(getattr(Computation, method)).parameters
        released = set(RELEASED_SIGNATURES[method])

        for name, parameter in parameters.items():
            if name in {"self", "cls"} or name in released:
                continue
            assert parameter.default is not inspect.Parameter.empty, (
                f"Computation.{method} gained a required parameter {name!r}"
            )


class TestReleasedCallPatternsStillWork:
    """The ways people actually call this, run unchanged."""

    def test_write_and_read_json_by_path(self, tmp_path):
        """The documented file-path round-trip."""
        comp = Computation()
        comp.add_node("a", value=1)

        path = str(tmp_path / "comp.json")
        comp.write_json(path)

        assert Computation.read_json(path).v.a == 1

    def test_write_and_read_json_by_file_object(self):
        """The documented text-buffer round-trip."""
        comp = Computation()
        comp.add_node("a", value=1)

        buf = io.StringIO()
        comp.write_json(buf)
        buf.seek(0)

        assert Computation.read_json(buf).v.a == 1

    def test_dumps_and_loads(self):
        """The string round-trip on the serializer itself."""
        from loman import ComputationSerializer

        comp = Computation()
        comp.add_node("a", value=1)

        serializer = ComputationSerializer()
        assert serializer.loads(serializer.dumps(comp)).v.a == 1

    def test_serialize_false(self, tmp_path):
        """The documented way to exclude a node."""
        from loman import States

        comp = Computation()
        comp.add_node("skipped", value=object(), serialize=False)
        comp.add_node("kept", value=42)

        path = str(tmp_path / "comp.json")
        comp.write_json(path)
        restored = Computation.read_json(path)

        assert restored.state("skipped") == States.UNINITIALIZED
        assert restored.v.kept == 42

    def test_use_dill_for_functions(self):
        """The documented escape hatch for lambdas."""
        from loman import ComputationSerializer

        comp = Computation()
        comp.add_node("a", value=3)
        comp.add_node("b", lambda a: a * 2)
        comp.compute_all()

        serializer = ComputationSerializer(use_dill_for_functions=True)
        assert serializer.loads(serializer.dumps(comp)).v.b == 6

    def test_write_dill_still_works(self, tmp_path):
        """The deprecated dill path still functions, warning as before."""
        comp = Computation()
        comp.add_node("a", value=42)
        path = str(tmp_path / "comp.dill")

        with pytest.warns(DeprecationWarning, match="write_dill"):
            comp.write_dill(path)
        with pytest.warns(DeprecationWarning, match="read_dill"):
            assert Computation.read_dill(path).v.a == 42

    def test_write_dill_old_still_works(self, tmp_path):
        """The doubly-deprecated dill path still functions.

        Removing it was tempting --- it is unsafe under concurrency --- but
        deleting a public method without a release that says so breaks callers
        silently.
        """
        import dill  # nosec B403

        comp = Computation()
        comp.add_node("a", value=42)
        path = tmp_path / "comp.dill"

        with pytest.warns(DeprecationWarning, match="write_dill_old"):
            comp.write_dill_old(str(path))

        with path.open("rb") as f:
            loaded = dill.load(f)  # nosec B301  # noqa: S301
        assert loaded.v.a == 42

    def test_write_dill_old_restores_the_class(self, tmp_path):
        """It puts __getstate__ and __setstate__ back, as it always did."""
        comp = Computation()
        comp.add_node("a", value=1)

        before = (Computation.__getstate__, Computation.__setstate__)
        with pytest.warns(DeprecationWarning, match="write_dill_old"):
            comp.write_dill_old(str(tmp_path / "comp.dill"))

        assert (Computation.__getstate__, Computation.__setstate__) == before


class TestWriteJsonOutputShape:
    """write_json still produces the document shape people parse."""

    def test_top_level_keys_are_still_present(self):
        """version, nodes and edges have not moved or been renamed."""
        import json

        comp = Computation()
        comp.add_node("a", value=1)

        document = json.loads(ComputationSerializerFactory().dumps(comp))

        assert {"version", "nodes", "edges"} <= set(document)
        assert isinstance(document["nodes"], list)
        assert isinstance(document["edges"], list)

    def test_node_fields_are_still_present(self):
        """The node fields documented in the format reference still exist."""
        import json

        def add_one(a):
            return a + 1

        comp = Computation()
        comp.add_node("a", value=1)
        comp.add_node("b", add_one)
        comp.compute_all()

        document = json.loads(ComputationSerializerFactory().dumps(comp))
        node = document["nodes"][0]

        assert {"key", "state", "value", "has_value", "func", "args", "kwds", "serialize", "tags"} <= set(node)

    def test_edge_fields_are_still_present(self):
        """The edge fields documented in the format reference still exist."""
        import json

        def add_one(a):
            return a + 1

        comp = Computation()
        comp.add_node("a", value=1)
        comp.add_node("b", add_one)

        document = json.loads(ComputationSerializerFactory().dumps(comp))
        edge = document["edges"][0]

        assert {"src", "dst", "param_type", "param"} <= set(edge)


def ComputationSerializerFactory():  # noqa: N802 - reads as a constructor at the call sites
    """Return a default serializer, imported lazily to keep the module light."""
    from loman import ComputationSerializer

    return ComputationSerializer()
