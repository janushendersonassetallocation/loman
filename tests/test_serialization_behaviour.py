"""Behavioural and integration tests for saving and loading.

The rest of the serialization suite is largely unit-level: encode a value,
decode it, compare. That leaves whole categories unexercised, and two real bugs
were found by the tests in this module before it was written --- a shared
serializer corrupting concurrent saves, and a failed directory save destroying
the previous good container.

What is covered here that is not covered elsewhere:

*Cross-process.* Every other round-trip happens inside one interpreter, so
nothing distinguishes "the file is correct" from "the object was still in
memory". These tests write in one process and read in another.

*Invariants over operations.* Saving must not change the computation. Loading
then saving must produce the same file. A failed save must leave what was there
before untouched.

*Concurrency.* The design claims a serializer can be shared. That claim is only
worth making if something checks it.

*Realistic graphs.* A computation built the way the library's own documentation
builds one --- decorators, tags, groups, errors, stale nodes --- rather than
values poked into a bare graph.
"""

import json
import subprocess
import sys
import textwrap
import threading
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from loman import Computation, States
from loman.consts import NodeAttributes
from loman.nodekey import parse_nodekey
from loman.serialization import ComputationSerializer, SerializationProfile
from loman.serialization.computation import UnserializableFunctionWarning
from tests.fixtures.factory_pipeline import Portfolio, RequiresArguments, StatefulPortfolio

REPO_ROOT = Path(__file__).parent.parent
SRC = REPO_ROOT / "src"


# ---------------------------------------------------------------------------
# Smoke tests: a fresh interpreter, a real file, no shared memory.
# ---------------------------------------------------------------------------


def _run_python(code: str) -> subprocess.CompletedProcess:
    """Run *code* in a separate interpreter with loman importable."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REPO_ROOT),
        env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"},
        check=False,
    )


WRITER = """
    import numpy as np, pandas as pd
    from loman import Computation
    from tests.fixtures.pipeline import build_pipeline

    comp = build_pipeline()
    comp.compute_all()
    comp.save({path!r}{extra})
    print("WROTE", comp.v.total)
"""

READER = """
    import numpy as np, pandas as pd
    from loman import Computation

    comp = Computation.load({path!r})
    print("TOTAL", comp.v.total)
    print("ROWS", len(comp.v.prices))
    print("TZ", comp.v.prices.index.tz)
    print("STATE", comp.state("total").name)
    comp.insert("multiplier", 10)
    comp.compute_all()
    print("RECOMPUTED", comp.v.total)
"""


class TestCrossProcessSmoke:
    """A file written by one interpreter is fully usable in another.

    This is the difference between "the format works" and "the objects happened
    to still be in memory". Everything the reader touches comes off disk.
    """

    @pytest.mark.parametrize(
        ("name", "extra"),
        [
            ("run.loman", ""),
            ("run.json", ""),
            ("run_dir", ", container='dir'"),
            ("readable.loman", ", profile='readable'"),
        ],
        ids=["zip", "json", "dir", "readable-zip"],
    )
    def test_write_in_one_process_read_in_another(self, tmp_path, name, extra):
        """Every container survives leaving the process that wrote it."""
        path = str(tmp_path / name)

        written = _run_python(WRITER.format(path=path, extra=extra))
        assert written.returncode == 0, written.stderr
        assert "WROTE" in written.stdout

        read = _run_python(READER.format(path=path))
        assert read.returncode == 0, read.stderr

        output = dict(line.split(" ", 1) for line in read.stdout.strip().splitlines())
        assert output["ROWS"] == "500"
        assert output["TZ"] == "UTC"
        assert output["STATE"] == "UPTODATE"
        assert output["TOTAL"] == written.stdout.split("WROTE ")[1].split("\n")[0]
        # The graph is not just readable, it still works.
        assert float(output["RECOMPUTED"]) != float(output["TOTAL"])

    def test_reader_needs_nothing_from_the_writer(self, tmp_path):
        """Loading works with no knowledge of how the file was produced."""
        path = str(tmp_path / "run.loman")
        assert _run_python(WRITER.format(path=path, extra="")).returncode == 0

        read = _run_python(f"""
            from loman import Computation
            comp = Computation.load({path!r})
            print(sorted(str(k) for k in comp.dag.nodes()))
        """)

        assert read.returncode == 0, read.stderr
        assert "prices" in read.stdout


# ---------------------------------------------------------------------------
# Behavioural invariants.
# ---------------------------------------------------------------------------


def _rich_computation():
    """A computation with most of the awkward corners in it at once."""
    comp = Computation()
    comp.add_node("multiplier", value=3, group="inputs", tags=["cfg"])
    comp.add_node(
        "prices",
        value=pd.DataFrame(
            {"px": np.arange(500, dtype="float64")},
            index=pd.date_range("2020-01-01", periods=500, freq="min", tz="UTC"),
        ),
    )
    comp.add_node("weights", value=np.arange(500, dtype="float64"))
    comp.add_node("notes", value={1: "int key", "type": "reserved"})
    comp.compute_all()
    return comp


class TestSaveDoesNotMutate:
    """Saving is a read of the computation, not a write to it."""

    def test_states_are_unchanged(self, tmp_path):
        """No node changes state as a side effect of being saved."""
        comp = _rich_computation()
        before = {str(k): comp.state(k) for k in comp.dag.nodes()}

        comp.save(str(tmp_path / "c.loman"))

        assert {str(k): comp.state(k) for k in comp.dag.nodes()} == before

    def test_values_are_the_same_objects(self, tmp_path):
        """Values are not copied, replaced or coerced by saving."""
        comp = _rich_computation()
        frame_id = id(comp.v.prices)
        weights_id = id(comp.v.weights)

        comp.save(str(tmp_path / "c.loman"))

        assert id(comp.v.prices) == frame_id
        assert id(comp.v.weights) == weights_id

    def test_graph_shape_is_unchanged(self, tmp_path):
        """Saving adds no nodes, edges or tags."""
        comp = _rich_computation()
        nodes = set(comp.dag.nodes())
        edges = set(comp.dag.edges())
        tags = set(comp.t.multiplier)

        comp.save(str(tmp_path / "c.loman"))

        assert set(comp.dag.nodes()) == nodes
        assert set(comp.dag.edges()) == edges
        assert set(comp.t.multiplier) == tags

    def test_computation_still_computes_afterwards(self, tmp_path):
        """A saved computation is still usable, not left in a spent state."""
        comp = _rich_computation()
        comp.save(str(tmp_path / "c.loman"))

        comp.insert("multiplier", 7)
        comp.compute_all()

        assert comp.v.multiplier == 7


class TestRoundTripIsStable:
    """Loading and re-saving converges rather than drifting."""

    @pytest.mark.parametrize("container", ["zip", "dir", "json"])
    def test_load_then_save_reproduces_the_file(self, tmp_path, container):
        """A file, loaded and written straight back, is byte-identical.

        Drift here would mean each save/load cycle changed something --- a dtype
        widening, an index losing its resolution --- which accumulates silently
        over a checkpoint loop.
        """
        suffix = {"zip": ".loman", "dir": "", "json": ".json"}[container]
        first = tmp_path / f"first{suffix}"
        second = tmp_path / f"second{suffix}"

        _rich_computation().save(str(first), container=container)
        Computation.load(str(first)).save(str(second), container=container)

        assert _digest(first) == _digest(second)

    def test_values_survive_repeated_cycles(self, tmp_path):
        """Ten save/load cycles leave the values exactly as they started."""
        original = _rich_computation()
        expected_frame = original.v.prices.copy()

        comp = original
        for i in range(10):
            path = tmp_path / f"cycle{i}.loman"
            comp.save(str(path))
            comp = Computation.load(str(path))

        assert comp.v.prices.equals(expected_frame)
        assert comp.v.prices.index.tz is not None
        assert np.array_equal(comp.v.weights, original.v.weights)
        assert comp.v.notes == {1: "int key", "type": "reserved"}


def _digest(path: Path) -> object:
    """Return a comparable representation of a container's contents."""
    if path.is_dir():
        return {str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}
    if path.suffix == ".json":
        return path.read_bytes()
    with zipfile.ZipFile(path) as zf:
        return {name: zf.read(name) for name in sorted(zf.namelist())}


class TestFailedSaveIsHarmless:
    """A save that fails leaves what was there before it intact."""

    @staticmethod
    def _unsaveable():
        """Return a computation whose first node cannot be encoded."""
        comp = Computation()
        # 'aaa' sorts before 'zzz', so the failure happens after the writer has
        # started but before the good node is written --- the worst ordering.
        comp.add_node("aaa_bad", value=object())
        comp.add_node("zzz_good", value=np.arange(20_000, dtype="float64"))
        return comp

    def test_existing_zip_survives(self, tmp_path):
        """A failed save does not damage the previous archive."""
        path = tmp_path / "c.loman"
        good = Computation()
        good.add_node("zzz_good", value=np.arange(20_000, dtype="float64"))
        good.save(str(path))
        before = path.read_bytes()

        with pytest.raises(Exception, match=r"[Ss]erializ"):
            self._unsaveable().save(str(path))

        assert path.read_bytes() == before
        assert len(Computation.load(str(path)).v.zzz_good) == 20_000

    def test_existing_directory_survives(self, tmp_path):
        """The directory container is protected the same way.

        Regression: the previous implementation cleared the blobs directory
        before writing, so a failure part way through destroyed the last good
        checkpoint --- losing data because of an operation that did not succeed.
        """
        path = tmp_path / "c_dir"
        good = Computation()
        good.add_node("zzz_good", value=np.arange(20_000, dtype="float64"))
        good.save(str(path), container="dir")
        before = _digest(path)

        with pytest.raises(Exception, match=r"[Ss]erializ"):
            self._unsaveable().save(str(path), container="dir")

        assert _digest(path) == before
        assert len(Computation.load(str(path)).v.zzz_good) == 20_000

    def test_no_staging_files_are_left_behind(self, tmp_path):
        """A failed save leaves no half-written temporaries lying about."""
        path = tmp_path / "c_dir"
        good = Computation()
        good.add_node("zzz_good", value=np.arange(1000, dtype="float64"))
        good.save(str(path), container="dir")

        with pytest.raises(Exception, match=r"[Ss]erializ"):
            self._unsaveable().save(str(path), container="dir")

        leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith((".tmp", ".previous"))]
        assert leftovers == []

    def test_failing_into_a_fresh_path_leaves_nothing(self, tmp_path):
        """A first save that fails does not leave a broken container behind."""
        path = tmp_path / "new.loman"

        with pytest.raises(Exception, match=r"[Ss]erializ"):
            self._unsaveable().save(str(path))

        assert not path.exists()


# ---------------------------------------------------------------------------
# Concurrency.
# ---------------------------------------------------------------------------


def _slow_serializer():
    """A serializer whose array encoding is slow, to widen the race window."""
    from loman.serialization.computation import default_computation_transformer
    from loman.serialization.transformer import NdArrayTransformer

    class SlowNdArray(NdArrayTransformer):
        def to_dict(self, transformer, o):
            import time

            time.sleep(0.005)
            return super().to_dict(transformer, o)

    transformer = default_computation_transformer()
    transformer._direct_type_map[np.ndarray] = SlowNdArray()
    return ComputationSerializer(transformer=transformer)


class TestConcurrentSaves:
    """One serializer, many threads.

    Regression: per-save state used to live on the serializer instance, so
    concurrent saves wrote into each other's archive. Eleven of twelve threads
    failed with "Can't write to [a closed zipfile]". The deliberately slow
    transformer is what makes the race reproducible rather than occasional.
    """

    def test_shared_serializer_survives_parallel_saves(self, tmp_path):
        """Twelve threads sharing a serializer each produce a correct file."""
        serializer = _slow_serializer()

        def save_and_check(i):
            comp = Computation()
            for j in range(3):
                comp.add_node(f"n{j}", value=np.arange(5_000, dtype="float64") + i * 100 + j)
            path = tmp_path / f"t{i}.loman"
            serializer.save(comp, str(path))

            restored = Computation.load(str(path))
            return all(
                np.array_equal(getattr(restored.v, f"n{j}"), np.arange(5_000, dtype="float64") + i * 100 + j)
                for j in range(3)
            )

        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(save_and_check, range(12)))

        assert all(results), f"{results.count(False)} of {len(results)} saves were corrupted"

    def test_parallel_saves_do_not_mix_stores(self, tmp_path):
        """Values never land in another thread's blob store."""
        from loman.serialization.blobs import BlobStore

        class TaggedStore(BlobStore):
            def __init__(self, tag):
                self.tag, self.blobs = tag, {}

            def write_blob(self, key, data):
                self.blobs[key] = data

            def read_blob(self, key):
                return self.blobs[key]

        serializer = _slow_serializer()
        stores = {i: TaggedStore(i) for i in range(8)}

        def save(i):
            comp = Computation()
            comp.add_node("v", value=np.arange(5_000, dtype="float64") + i, store="mine")
            serializer.save(comp, str(tmp_path / f"s{i}.loman"), stores={"mine": stores[i]})

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(save, range(8)))

        for i, store in stores.items():
            assert store.blobs, f"store {i} received nothing"
            restored = Computation.load(str(tmp_path / f"s{i}.loman"), stores={"mine": store})
            assert np.array_equal(restored.v.v, np.arange(5_000, dtype="float64") + i)

    def test_concurrent_loads(self, tmp_path):
        """Reading the same file from many threads is safe."""
        comp = _rich_computation()
        path = str(tmp_path / "c.loman")
        comp.save(path)

        def load_and_check(_):
            return Computation.load(path).v.prices.equals(comp.v.prices)

        with ThreadPoolExecutor(max_workers=8) as pool:
            assert all(pool.map(load_and_check, range(16)))

    def test_saves_from_threads_with_different_profiles(self, tmp_path):
        """Each thread's profile applies to its own save only."""
        serializer = _slow_serializer()
        profiles = {
            "readable": SerializationProfile("readable", inline_max_bytes=None),
            "efficient": SerializationProfile("efficient", inline_max_bytes=1024, compression="auto"),
        }
        barrier = threading.Barrier(2)

        def save(name):
            comp = Computation()
            comp.add_node("v", value=np.arange(5_000, dtype="float64"))
            barrier.wait()
            serializer.save(comp, str(tmp_path / f"{name}.loman"), profile=profiles[name])

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(save, profiles))

        with zipfile.ZipFile(tmp_path / "readable.loman") as zf:
            assert zf.namelist() == ["manifest.json"], "readable save picked up the other thread's profile"
        with zipfile.ZipFile(tmp_path / "efficient.loman") as zf:
            assert any(n.startswith("blobs/") for n in zf.namelist())


# ---------------------------------------------------------------------------
# Realistic end-to-end graphs.
# ---------------------------------------------------------------------------


def compute_returns(prices):
    """Percentage change of a price frame."""
    return prices.pct_change().fillna(0.0)


def compute_signal(returns, multiplier):
    """A trivial signal, scaled."""
    return returns * multiplier


def failing_check(signal):
    """Always raises, to put a node into ERROR state."""
    msg = "risk limit breached"
    raise ValueError(msg)


class TestRealisticPipeline:
    """A graph built with the library's own idioms, taken through save/load."""

    @staticmethod
    def _portfolio():
        """Return a computed Portfolio with a real frame in it."""
        comp = Portfolio()
        comp.insert(
            "prices",
            pd.DataFrame(
                {"a": np.linspace(100, 110, 250), "b": np.linspace(50, 55, 250)},
                index=pd.date_range("2020-01-01", periods=250, freq="D", tz="UTC"),
            ),
        )
        comp.insert("multiplier", 2.0)
        comp.compute_all()
        return comp

    def test_factory_built_graph_roundtrips_and_recomputes(self, tmp_path):
        """A decorator-declared computation survives fully, functions included.

        Regression: a calc node declared on a factory class is a method bound to
        the definition object, and a bound method has no importable path of its
        own --- after decoration the class's name refers to the factory function.
        The function used to be dropped silently, so the graph reloaded looking
        complete and then never updated again.
        """
        comp = self._portfolio()
        path = str(tmp_path / "pf.loman")

        comp.save(path)
        restored = Computation.load(path)

        assert restored.v.signal.equals(comp.v.signal)
        assert set(restored.dag.edges()) == set(comp.dag.edges())

        restored.insert("multiplier", 4.0)
        restored.compute_all()

        assert restored.state("signal") == States.UPTODATE
        assert restored.v.signal.equals(comp.v.prices * 4.0)

    def test_factory_save_emits_no_warning(self, tmp_path):
        """The ordinary factory case is fully supported, so it warns about nothing."""
        comp = self._portfolio()

        with warnings.catch_warnings():
            warnings.simplefilter("error", UnserializableFunctionWarning)
            comp.save(str(tmp_path / "pf.loman"))

    def test_definition_state_is_rebuilt_not_carried(self, tmp_path):
        """State set in ``__init__`` is reconstructed by building a fresh object.

        The restored method binds to a new definition object, so anything
        ``__init__`` computes comes back. Anything mutated on ``self`` at run
        time does not --- which is why the transformer documents the distinction
        rather than implying the original object is restored.
        """
        comp = StatefulPortfolio()
        comp.insert("prices", pd.DataFrame({"a": [1.0, 2.0]}))
        comp.compute_all()

        path = str(tmp_path / "sp.loman")
        comp.save(path)
        restored = Computation.load(path)

        restored.insert("prices", pd.DataFrame({"a": [3.0, 4.0]}))
        restored.compute_all()

        assert list(restored.v.scaled["a"]) == [30.0, 40.0]

    def test_uninstantiable_definition_class_falls_back(self, tmp_path):
        """A class needing constructor arguments cannot be rebuilt, and says so.

        The value is still saved. What is lost is the ability to recompute, and
        the warning is what stops that being discovered much later.
        """
        instance = RequiresArguments(factor=3.0)
        comp = Computation()
        comp.add_node("prices", value=pd.DataFrame({"a": [1.0, 2.0]}))
        comp.add_node("scaled", instance.scaled)
        comp.compute_all()

        path = str(tmp_path / "req.loman")
        with pytest.warns(UnserializableFunctionWarning, match="scaled"):
            comp.save(path)

        restored = Computation.load(path)
        assert restored.v.scaled.equals(comp.v.scaled)
        assert restored.dag.nodes[parse_nodekey("scaled")][NodeAttributes.FUNC] is None

    def test_module_level_functions_do_recompute(self, tmp_path):
        """The same graph built from importable functions recomputes normally."""
        comp = Computation()
        comp.add_node("prices", value=pd.DataFrame({"a": np.linspace(100, 110, 10)}))
        comp.add_node("multiplier", value=2.0)
        comp.add_node("returns", compute_returns)
        comp.add_node("signal", compute_signal)
        comp.compute_all()

        path = str(tmp_path / "plain.loman")
        comp.save(path)
        restored = Computation.load(path)

        restored.insert("multiplier", 4.0)
        restored.compute_all()

        assert restored.state("signal") == States.UPTODATE
        assert restored.v.signal.equals(comp.v.returns * 4.0)

    def test_error_and_stale_nodes_together(self, tmp_path):
        """A part-failed, part-stale graph is preserved as it stands."""
        comp = self._portfolio()
        comp.add_node("check", failing_check)
        comp.compute_all()
        assert comp.state("check") == States.ERROR

        comp.insert("multiplier", 9.0)  # signal is now out of date, check is stale

        path = str(tmp_path / "pf.loman")
        comp.save(path)
        restored = Computation.load(path)

        # Replacing an input moved the failed node from ERROR to STALE. It still
        # holds the Error it produced, and that is what a post-mortem needs.
        assert comp.state("check") == States.STALE
        assert restored.state("check") == States.STALE

        failure = restored["check"].value
        assert isinstance(failure.exception, ValueError)
        assert "risk limit breached" in str(failure.exception)
        assert "failing_check" in failure.traceback

        # The out-of-date node kept the value it last computed.
        assert restored.v.signal.equals(comp.v.signal)

    def test_error_value_survives_after_going_stale(self, tmp_path):
        """A failed node that has since gone stale can still be saved.

        Regression: the error encoding keyed off the node's *state*, so once a
        replaced input moved a failed node from ERROR to STALE the value took the
        generic path instead --- and an exception object has no encoding, so the
        whole save failed. Retaining values for stale nodes is what exposed it.
        """
        comp = Computation()
        comp.add_node("multiplier", value=1.0)
        comp.add_node("signal", value=pd.DataFrame({"a": [1.0]}))
        comp.add_node("check", failing_check)
        comp.compute_all()
        assert comp.state("check") == States.ERROR

        comp.insert("signal", pd.DataFrame({"a": [2.0]}))
        # The node has left ERROR state but still holds the Error it produced:
        # that combination is the one the encoding used to choke on.
        assert comp.state("check") != States.ERROR
        assert comp["check"].value.exception is not None

        path = str(tmp_path / "stale_error.loman")
        comp.save(path)  # must not raise

        restored = Computation.load(path)
        assert restored.state("check") == comp.state("check")
        assert isinstance(restored["check"].value.exception, ValueError)

    def test_presentation_attributes_survive(self, tmp_path):
        """Groups, styles and tags come back, so the graph still renders alike."""
        comp = self._portfolio()
        comp.add_node("annotated", value=1, group="reporting", style="small", tags=["published"])

        path = str(tmp_path / "pf.loman")
        comp.save(path)
        restored = Computation.load(path)

        from loman.consts import NodeAttributes
        from loman.nodekey import parse_nodekey

        node = restored.dag.nodes[parse_nodekey("annotated")]
        assert node[NodeAttributes.GROUP] == "reporting"
        assert node[NodeAttributes.STYLE] == "small"
        assert "published" in restored.t.annotated

    def test_partial_computation_roundtrips(self, tmp_path):
        """An incomplete graph saves and resumes from where it was."""
        comp = Computation()
        comp.add_node("prices")  # deliberately left uninitialized
        comp.add_node("multiplier", value=2.0)
        comp.add_node("returns", compute_returns)
        comp.add_node("signal", compute_signal)

        path = str(tmp_path / "partial.loman")
        comp.save(path)
        restored = Computation.load(path)

        assert restored.state("prices") == States.UNINITIALIZED
        restored.insert(
            "prices",
            pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=pd.date_range("2020", periods=3)),
        )
        restored.compute_all()
        assert restored.state("signal") == States.UPTODATE


class TestManifestIsSelfDescribing:
    """The promise that a saved file can be understood without decoding it."""

    def test_shapes_and_types_are_readable_without_blobs(self, tmp_path):
        """Everything needed to describe the values sits in the manifest."""
        comp = _rich_computation()
        path = tmp_path / "c.loman"
        comp.save(str(path))

        with zipfile.ZipFile(path) as zf:
            manifest = json.loads(zf.read("manifest.json"))

        nodes = {node["key"]: node for node in manifest["nodes"]}
        frame = nodes["prices"]["value"]

        assert frame["index"]["kind"] == "datetime"
        assert frame["index"]["tz"] == "UTC"
        assert frame["columns"]["values"] == ["px"]
        assert nodes["weights"]["value"]["shape"] == [500]
        assert nodes["weights"]["value"]["dtype"] == "<f8"

    def test_manifest_does_not_grow_with_data(self, tmp_path):
        """A hundredfold more data does not make the manifest bigger."""
        sizes = []
        for n in (500, 50_000):
            comp = Computation()
            comp.add_node("v", value=np.arange(n, dtype="float64"))
            path = tmp_path / f"c{n}.loman"
            comp.save(str(path))
            with zipfile.ZipFile(path) as zf:
                sizes.append(len(zf.read("manifest.json")))

        assert sizes[1] < sizes[0] * 1.5, f"manifest grew from {sizes[0]} to {sizes[1]} bytes"
