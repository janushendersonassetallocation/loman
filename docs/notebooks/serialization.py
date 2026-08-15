"""Example: Serializing and Reloading Loman Computations.

This notebook walks through write_json / read_json, serialize=False, the
dill fallback for lambdas, and post-mortem inspection of ERROR nodes.
"""

# ruff: noqa: E501

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.17.6",
#     "numpy",
#     "pandas",
#     "loman",
# ]
#
# [tool.uv.sources]
# loman = { path = "../..", editable = true }
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Serializing and Reloading Loman Computations

    Loman computations can be saved to a JSON file and reloaded later — useful for:

    - **Post-mortem debugging**: save a batch run in full detail so you can inspect every intermediate value if something goes wrong.
    - **Checkpoint / resume**: persist a partially-completed computation and pick up where you left off.
    - **Reproducibility**: store the exact inputs and results alongside the code that produced them.

    This notebook walks through the key features of `write_json` / `read_json`:

    1. Basic round-trip
    2. Excluding nodes with `serialize=False`
    3. Handling lambdas with `ComputationSerializer(use_dill_for_functions=True)`
    4. Preserving `PINNED` state
    5. Post-mortem inspection of `ERROR` nodes
    6. Pandas DataFrames as node values
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Basic round-trip

    We start with a small computation: three nodes where `c` depends on `a` and `b`.
    All three nodes use importable module-level functions, so they serialise cleanly.
    """)
    return


@app.cell
def _():
    import io
    import json
    import math

    from loman import Computation, ComputationSerializer, States

    def square(x):
        return x**2

    def hypotenuse(a, b):
        return math.sqrt(a + b)

    comp = Computation()
    comp.add_node("a", value=3.0)
    comp.add_node("b", value=4.0)
    comp.add_node("a_sq", square, kwds={"x": "a"})
    comp.add_node("b_sq", square, kwds={"x": "b"})
    comp.add_node("c", hypotenuse, kwds={"a": "a_sq", "b": "b_sq"})
    comp.compute_all()
    comp.to_dict()
    return Computation, ComputationSerializer, States, comp, io, json


@app.cell
def _(comp):
    comp
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Save to an in-memory buffer and reload:
    """)
    return


@app.cell
def _(Computation, comp, io):
    buf = io.StringIO()
    comp.write_json(buf)
    buf.seek(0)
    comp_loaded = Computation.read_json(buf)
    comp_loaded.to_dict()
    return (comp_loaded,)


@app.cell
def _(comp_loaded):
    comp_loaded.src["c"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The output is plain JSON text — open it in any text editor to inspect it.
    Here is what the file actually looks like for this computation:
    """)
    return


@app.cell
def _(comp, io, json):
    _buf = io.StringIO()
    comp.write_json(_buf)
    print(json.dumps(json.loads(_buf.getvalue()), indent=2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each node carries its `state`, encoded `value`, and (where applicable) a `func`
    reference stored as `{"type": "func_ref", "module": "...", "qualname": "..."}`.
    Edges record the dependency wiring including parameter names.

    The reloaded computation has the same values and states. The function references are
    also preserved — we can update an input and recompute:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The reloaded computation has the same values and states. The function references are
    also preserved — we can update an input and recompute:
    """)
    return


@app.cell
def _(comp_loaded):
    comp_loaded.insert("a", 5.0)
    comp_loaded.compute_all()
    comp_loaded.to_dict()
    return


@app.cell
def _(comp_loaded):
    comp_loaded
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Excluding nodes with `serialize=False`

    Some nodes hold values that should not be saved — database connections, licensed
    data, or objects that cannot be serialised. Pass `serialize=False` when adding the
    node: it will be stored as `UNINITIALIZED` in the file with no value.
    """)
    return


@app.cell
def _(Computation, io):
    def _expensive_db_fetch():
        # Pretend this returns data from a live database.
        return {"price": 42.0}

    comp_skip = Computation()
    comp_skip.add_node("db_conn", value=object(), serialize=False)  # not saved
    comp_skip.add_node("raw_data", value={"price": 42.0})  # saved
    comp_skip.add_node("result", value=42.0 * 1.1)  # saved

    buf_skip = io.StringIO()
    comp_skip.write_json(buf_skip)
    buf_skip.seek(0)

    comp_skip2 = Computation.read_json(buf_skip)
    {
        "db_conn": comp_skip2.state("db_conn"),  # UNINITIALIZED — not restored
        "raw_data": comp_skip2.value("raw_data"),
        "result": comp_skip2.value("result"),
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `db_conn` comes back as `UNINITIALIZED` — exactly as if the node had never been
    given a value — while `raw_data` and `result` round-trip perfectly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Lambdas and closures

    By default, `write_json` raises a `SerializationError` when a node's function is a
    lambda, because lambdas have no importable module path:
    """)
    return


@app.cell
def _(Computation, io):
    from loman import SerializationError as _SerializationError

    comp_lambda = Computation()
    comp_lambda.add_node("x", value=5)
    comp_lambda.add_node("y", lambda x: x**2)
    comp_lambda.compute_all()

    try:
        comp_lambda.write_json(io.StringIO())
        error_msg = None
    except _SerializationError as e:
        error_msg = str(e)

    error_msg
    return (comp_lambda,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The error message points to the fix: pass `use_dill_for_functions=True` to
    `ComputationSerializer`. This encodes the callable as a base64 [dill](https://github.com/uqfoundation/dill)
    blob inside the JSON, so lambdas and closures — including ones that capture local
    variables — round-trip intact:
    """)
    return


@app.cell
def _(Computation, ComputationSerializer, comp_lambda, io):
    s_dill = ComputationSerializer(use_dill_for_functions=True)

    buf_lambda = io.StringIO()
    comp_lambda.write_json(buf_lambda, serializer=s_dill)
    buf_lambda.seek(0)
    comp_lambda2 = Computation.read_json(buf_lambda, serializer=s_dill)

    # Value is restored …
    print("Loaded value of y:", comp_lambda2.value("y"))

    # … and the function is live — we can recompute after changing x.
    comp_lambda2.insert("x", 12)
    comp_lambda2.compute_all()
    print("Recomputed y after x=12:", comp_lambda2.value("y"))
    return (buf_lambda,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The lambda is stored as a `"dill_func"` object in the JSON. The `blob` field is a
    base64-encoded dill byte string — here is what the `func` field looks like
    (blob truncated for readability):
    """)
    return


@app.cell
def _(buf_lambda, json):
    from pprint import pprint

    _raw = json.loads(buf_lambda.getvalue())

    pprint(_raw)
    return (pprint,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Closures that capture variables from an enclosing scope also work:
    """)
    return


@app.cell
def _(Computation, ComputationSerializer, io, json, pprint):
    scale = 2.5

    def scale_up(x):
        return x * scale  # captures `scale` from the enclosing scope

    comp_closure = Computation()
    comp_closure.add_node("x", value=4)
    comp_closure.add_node("y", scale_up)
    comp_closure.compute_all()

    s2 = ComputationSerializer(use_dill_for_functions=True)
    buf_closure = io.StringIO()
    comp_closure.write_json(buf_closure, serializer=s2)
    buf_closure.seek(0)
    comp_closure2 = Computation.read_json(buf_closure, serializer=s2)

    comp_closure2.insert("x", 10)
    comp_closure2.compute_all()
    print("y after reload with x=10:", comp_closure2.value("y"))  # 25.0
    print("========================================")
    pprint(json.loads(buf_closure.getvalue()))
    comp_closure2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Note:** The dill blob is not portable across Python versions. Prefer named
    > module-level functions when portability matters; use `use_dill_for_functions=True`
    > when convenience is more important.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Preserving PINNED state

    A `PINNED` node's value is locked — downstream recalculations use it but it is
    never overwritten by `compute_all`. The `PINNED` state survives a round-trip:
    """)
    return


@app.cell
def _(Computation, States, io):
    comp_pin = Computation()
    comp_pin.add_node("rate", value=0.05)
    comp_pin.add_node("principal", value=1000.0)

    def calc_interest(rate, principal):
        return rate * principal

    comp_pin.add_node("interest", calc_interest)
    comp_pin.compute_all()

    # Pin the rate so that even if we reload and change inputs, it stays fixed.
    comp_pin.pin("rate")
    print("State of rate before save:", comp_pin.state("rate"))

    buf_pin = io.StringIO()
    comp_pin.write_json(buf_pin)
    buf_pin.seek(0)
    comp_pin2 = Computation.read_json(buf_pin)

    print("State of rate after reload:", comp_pin2.state("rate"))
    assert comp_pin2.state("rate") == States.PINNED
    print("Value of rate after reload:", comp_pin2.value("rate"))
    return (comp_pin,)


@app.cell
def _(comp_pin):
    comp_pin
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Post-mortem inspection of ERROR nodes

    When a node raises an exception during `compute_all`, it enters `ERROR` state.
    The exception's type, message and traceback are all preserved. A builtin
    exception type is rebuilt as itself, so `except ValueError` still matches after
    a reload; anything else becomes a `DeserializedError` carrying the original type
    name, because rebuilding it for real would mean importing whatever module the
    file names:
    """)
    return


@app.cell
def _(Computation, States, io):
    def bad_calc(x):
        msg = f"unexpected value: {x!r}"
        raise ValueError(msg)

    comp_err = Computation()
    comp_err.add_node("x", value=-1)
    comp_err.add_node("result", bad_calc)
    comp_err.compute_all()

    print("State of result:", comp_err.state("result"))

    buf_err = io.StringIO()
    comp_err.write_json(buf_err)
    buf_err.seek(0)
    comp_err2 = Computation.read_json(buf_err)

    print("State after reload:", comp_err2.state("result"))
    assert comp_err2.state("result") == States.ERROR

    err_val = comp_err2.value("result")
    print("Exception message:", err_val.exception)
    print("Traceback (first line):", err_val.traceback.splitlines()[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Pandas DataFrames as node values

    DataFrames and Series are serialised automatically using a JSON split-orientation
    format. No extra configuration needed:
    """)
    return


@app.cell
def _(Computation, io):
    import pandas as pd

    def enrich(raw):
        df = raw.copy()
        df["value_eur"] = df["qty"] * df["price_usd"] * 0.92
        return df

    prices = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "GOOG"],
            "qty": [10, 20, 5],
            "price_usd": [182.5, 375.2, 140.8],
        }
    )

    comp_df = Computation()
    comp_df.add_node("raw", value=prices)
    comp_df.add_node("enriched", enrich)
    comp_df.compute_all()

    buf_df = io.StringIO()
    comp_df.write_json(buf_df)
    buf_df.seek(0)
    comp_df2 = Computation.read_json(buf_df)
    comp_df2.value("enriched")
    return buf_df, pd


@app.cell
def _(buf_df, json, pprint):
    pprint(json.loads(buf_df.getvalue()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. `save` and `load`: the `.loman` container

    Everything above uses `write_json`, which writes one JSON document. That is
    readable and diffable, but every number is stored as text: a 100k-row frame
    becomes tens of megabytes of decimal digits.

    `save` writes a `.loman` file instead — a zip holding the same manifest, with
    large values stored beside it as compressed binary. The manifest still records
    each value's shape, dtype and index, so a saved run can be inspected without
    decoding any data.

    Both remain available. `write_json` is the right choice when you want a text
    file to diff; `save` is the right choice for everything else.
    """)
    return


@app.cell
def _(Computation, json, pd):
    import tempfile
    import zipfile
    from pathlib import Path

    import numpy as np

    tmp = Path(tempfile.mkdtemp())

    def make_prices(n=100_000, rounded=True):
        """A price series: a random walk, rounded as real prices are."""
        rng = np.random.default_rng(0)
        walk = 100 + np.cumsum(rng.standard_normal(n) * 0.01)
        return pd.DataFrame(
            {"px": np.round(walk, 2) if rounded else walk},
            index=pd.date_range("2020-01-01", periods=n, freq="min"),
        )

    def manifest_of(path):
        """Read a container's manifest without decoding any of its data."""
        path = Path(path)
        if path.is_dir():
            return json.loads((path / "manifest.json").read_text())
        if path.suffix == ".json":
            return json.loads(path.read_text())
        return json.loads(zipfile.ZipFile(path).read("manifest.json"))

    def size_of(path):
        """Total bytes on disk, whether one file or a directory."""
        path = Path(path)
        if path.is_dir():
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return path.stat().st_size

    px_frame = make_prices()
    comp_big = Computation()
    comp_big.add_node("prices", value=px_frame)

    comp_big.save(str(tmp / "run.loman"))
    comp_big.save(str(tmp / "run.json"))

    print(f"save('run.loman'): {size_of(tmp / 'run.loman'):>12,} bytes")
    print(f"save('run.json') : {size_of(tmp / 'run.json'):>12,} bytes")
    print(f"exact round-trip : {Computation.load(str(tmp / 'run.loman')).v.prices.equals(px_frame)}")
    return (
        Path,
        comp_big,
        make_prices,
        manifest_of,
        np,
        pd,
        px_frame,
        size_of,
        tmp,
        zipfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The manifest still describes everything

    Moving the data out of line does not make the file opaque. Shape, dtype,
    column names and index type stay inline — only the bulk numbers move — so you
    can see what a saved run contains without reading a single blob.
    """)
    return


@app.cell
def _(json, manifest_of, tmp):
    _manifest = manifest_of(tmp / "run.loman")
    _value = _manifest["nodes"][0]["value"]

    print(f"manifest size : {len(json.dumps(_manifest)):,} bytes (for a 100k-row frame)")
    print(f"index         : {json.dumps({k: v for k, v in _value['index'].items() if k != 'values'})}")
    print(f"columns       : {_value['columns']['values']}")
    print("blob table    :")
    for _entry in _manifest["blobs"]:
        print(f"  {_entry}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Profiles and containers

    Two independent choices, and keeping them independent is deliberate.

    The **profile** decides how a value is encoded — `"readable"` keeps everything
    inline as JSON, `"efficient"` (the default) writes large values out of line.

    The **container** decides where the bytes land — `"zip"` (a `.loman` file,
    the default), `"dir"` (the same layout unzipped), or `"json"` (one document).

    Collapsing them would rule out combinations people actually want, such as a
    readable manifest inside a zip. Only one pairing is impossible: `efficient`
    with a single JSON document, which has nowhere to put the bytes.
    """)
    return


@app.cell
def _(Computation, comp_big, px_frame, size_of, tmp):
    _rows = []
    for _label, _name, _kwargs in [
        ("efficient + zip (default)", "m_ez.loman", {}),
        ("readable  + zip", "m_rz.loman", {"profile": "readable"}),
        ("efficient + dir", "m_ed", {"container": "dir"}),
        ("readable  + json", "m_rj.json", {}),
    ]:
        _path = tmp / _name
        comp_big.save(str(_path), **_kwargs)
        _ok = Computation.load(str(_path)).v.prices.equals(px_frame)
        _rows.append((_label, size_of(_path), _ok))

    print(f"{'':28s} {'bytes':>12s}  exact")
    for _label, _bytes, _ok in _rows:
        print(f"{_label:28s} {_bytes:>12,}  {_ok}")

    print("\nThe one impossible combination:")
    try:
        comp_big.save(str(tmp / "nope.json"), profile="efficient")
    except ValueError as _exc:
        print(f"  {str(_exc)[:150]}...")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Which container?

    `.loman` is one file, which is what you want for handing a run to a colleague
    or uploading it.

    Use `container="dir"` when you save **repeatedly** — a checkpointing loop.
    Updating one value in a zip rewrites the whole archive at a cost that grows
    with its size, while a directory rewrites only the file that changed. The
    layout is identical either way; a `.loman` is simply that directory zipped.
    """)
    return


@app.cell
def _(Path, tmp):
    _dir = Path(tmp / "m_ed")
    print("directory container layout:")
    for _f in sorted(_dir.rglob("*")):
        print(f"  {_f.relative_to(_dir)}  ({_f.stat().st_size:,} bytes)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Compression is decided from the data

    Blanket compression is wrong in both directions, and the dtype cannot tell you
    which case you are in — both examples below are `float64` arrays.

    A rounded price series compresses roughly eight times in about ten
    milliseconds. Raw random floats compress a few percent and cost seconds. So
    each blob is sampled: 256 KiB is compressed, the result extrapolated, and the
    compression kept only if it saves more than 10%. The probe costs about a
    millisecond.
    """)
    return


@app.cell
def _(Computation, make_prices, manifest_of, np, size_of, tmp):
    _realistic = make_prices(200_000, rounded=True)["px"].to_numpy()
    _random = np.random.default_rng(0).standard_normal(200_000)

    for _label, _values in [("rounded px_frame", _realistic), ("random floats", _random)]:
        _comp = Computation()
        _comp.add_node("v", value=_values)
        _path = tmp / f"cmp_{_label.split()[0]}.loman"
        _comp.save(str(_path))

        _entry = manifest_of(_path)["blobs"][0]
        _chose = _entry["compression"]
        _ratio = _entry["size"] / _entry.get("stored_size", _entry["size"])
        print(f"{_label:16s} auto chose {_chose:8s}  {size_of(_path):>10,} bytes  ({_ratio:.1f}x)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To override the decision, pass a profile. `compression` accepts `"auto"` (the
    default), `"none"`, `"zlib:1"`–`"zlib:9"`, and `"zstd:N"` with the
    `loman[efficient]` extra installed.
    """)
    return


@app.cell
def _(Computation, make_prices, size_of, tmp):
    from loman import SerializationProfile

    _values = make_prices(200_000)["px"].to_numpy()
    _comp = Computation()
    _comp.add_node("v", value=_values)

    for _spec in ["none", "auto", "zlib:9"]:
        _profile = SerializationProfile("demo", inline_max_bytes=8192, compression=_spec)
        _path = tmp / f"lvl_{_spec.replace(':', '')}.loman"
        _comp.save(str(_path), profile=_profile)
        print(f"compression={_spec:8s} {size_of(_path):>10,} bytes")
    return (SerializationProfile,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Bring your own storage

    Sometimes a value does not belong in the saved file at all — a frame that
    should land in a bucket, or a result that belongs in a warehouse table.

    Mark the node with a store name, and supply the implementation when you save
    and load. A store is **two methods**: compression, deduplication, checksums,
    blob ids and the blob table are all handled for you.
    """)
    return


@app.cell
def _():
    from loman.serialization import BlobStore

    class BucketStore(BlobStore):
        """Stands in for S3 or a database. A real one would talk to a client."""

        def __init__(self, bucket):
            self.bucket = bucket
            self.objects = {}

        def write_blob(self, key, data):
            self.objects[f"s3://{self.bucket}/{key}"] = data

        def read_blob(self, key):
            return self.objects[f"s3://{self.bucket}/{key}"]

    return (BucketStore,)


@app.cell
def _(BucketStore, Computation, manifest_of, px_frame, size_of, tmp, zipfile):
    comp_remote = Computation()
    comp_remote.add_node("prices", value=px_frame, store="warehouse")
    comp_remote.add_node("summary", value={"rows": len(px_frame)})

    bucket = BucketStore("quant-data")
    comp_remote.save(str(tmp / "remote.loman"), stores={"warehouse": bucket})

    print(f"file on disk    : {size_of(tmp / 'remote.loman'):,} bytes")
    print(f"archive members : {zipfile.ZipFile(tmp / 'remote.loman').namelist()}")
    print(f"in the bucket   : {list(bucket.objects)}")
    print(f"                  {sum(len(v) for v in bucket.objects.values()):,} bytes")

    _back = Computation.load(str(tmp / "remote.loman"), stores={"warehouse": bucket})
    print(f"reload exact    : {_back.v.prices.equals(px_frame)}")
    print(f"blob entry      : {manifest_of(tmp / 'remote.loman')['blobs'][0]}")
    return (comp_remote,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The file never holds your credentials

    A manifest records a store's **name**, never its configuration. No bucket, no
    endpoint, no key reaches the file.

    The consequence is deliberate: a file with external values cannot resolve them
    on its own, so whoever loads it supplies the matching store. If they don't,
    the error says which store is missing and for which node.
    """)
    return


@app.cell
def _(Computation, comp_remote, json, manifest_of, tmp):
    print("bucket name anywhere in the file:", "quant-data" in json.dumps(manifest_of(tmp / "remote.loman")))

    print("\nForgetting the store, at each end:")
    try:
        comp_remote.save(str(tmp / "oops.loman"))
    except Exception as _exc:
        print(f"  save -> {type(_exc).__name__}: {str(_exc)[:110]}")

    try:
        Computation.load(str(tmp / "remote.loman"))
    except Exception as _exc:
        print(f"  load -> {type(_exc).__name__}: {str(_exc)[:110]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that a missing store is an **error**, not a quiet fallback to writing the
    data into the file. Believing your data went to a bucket when it is actually
    sitting in the archive is a worse outcome than a failed save.

    ### Routing is a default, not a fixture

    The store named on a node is what that node *is*; a profile override says
    where that kind of thing goes *today*. So the same computation can go to a
    bucket in production and to a plain container in a test, with no edit to the
    graph.
    """)
    return


@app.cell
def _(Computation, SerializationProfile, comp_remote, px_frame, size_of, tmp):
    _local = SerializationProfile(
        "local",
        inline_max_bytes=8192,
        compression="auto",
        overrides={"prices": {"store": None}},
    )
    comp_remote.save(str(tmp / "local.loman"), profile=_local)

    print(f"same computation, no bucket: {size_of(tmp / 'local.loman'):,} bytes")
    print(f"loads with no stores       : {Computation.load(str(tmp / 'local.loman')).v.prices.equals(px_frame)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Selectors match node-key globs (`"market/*"`) or tags (`"tag:bulky"`), so you
    can tag nodes by *what they are* and let each save decide where that kind of
    thing goes.
    """)
    return


@app.cell
def _(
    BucketStore,
    Computation,
    SerializationProfile,
    manifest_of,
    px_frame,
    tmp,
):
    _comp = Computation()
    _comp.add_node("prices", value=px_frame, tags=["bulky"])
    _comp.add_node("note", value="small enough to stay inline")

    _tagged = SerializationProfile(
        "tagged",
        inline_max_bytes=8192,
        compression="auto",
        overrides={"tag:bulky": {"store": "warehouse"}},
    )
    _bucket = BucketStore("tagged-data")
    _comp.save(str(tmp / "tagged.loman"), profile=_tagged, stores={"warehouse": _bucket})

    print("routed by tag, not by name:")
    for _entry in manifest_of(tmp / "tagged.loman")["blobs"]:
        print(f"  node {_entry['node']!r} -> store {_entry.get('store', 'container')!r}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A readable manifest with the data elsewhere

    Because an external store is independent of the container, the two combine:
    a manifest you can open in a text editor, describing values held in a bucket.
    """)
    return


@app.cell
def _(BucketStore, Computation, SerializationProfile, px_frame, size_of, tmp):
    _bucket = BucketStore("hybrid")
    _profile = SerializationProfile("hybrid", inline_max_bytes=8192, compression="auto")
    _comp = Computation()
    _comp.add_node("prices", value=px_frame, store="warehouse")
    _comp.save(str(tmp / "hybrid.json"), profile=_profile, stores={"warehouse": _bucket})

    print(f"manifest is plain JSON : {size_of(tmp / 'hybrid.json'):,} bytes")
    print(f"data in the bucket     : {sum(len(v) for v in _bucket.objects.values()):,} bytes")
    _back = Computation.load(str(tmp / "hybrid.json"), stores={"warehouse": _bucket})
    print(f"reload exact           : {_back.v.prices.equals(px_frame)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Loading files you do not trust

    Loading restores node functions, which means importing the modules the file
    names — or unpickling a dill blob out of it. **Both run code the file chose.**
    This format is not safe against a hostile file and never was.

    `allow_code=False` skips resolving callables entirely. Values, structure,
    states and tags still load, so a graph can be inspected but not recalculated.
    Treat it as a mitigation, not a security boundary.
    """)
    return


@app.function
def increment_for_demo(a):
    """Module-level so it is importable, and so the demo is not vacuous."""
    return a + 1


@app.cell
def _(Computation, tmp):
    from loman.consts import NodeAttributes
    from loman.nodekey import parse_nodekey

    _comp = Computation()
    _comp.add_node("a", value=1)
    _comp.add_node("b", increment_for_demo)
    _comp.compute_all()
    _comp.save(str(tmp / "untrusted.loman"))

    safe = Computation.load(str(tmp / "untrusted.loman"), allow_code=False)
    print(f"values still load     : a={safe.v.a}, b={safe.v.b}")
    print(f"structure still loads : {len(safe.dag.edges())} edge(s)")
    normal = Computation.load(str(tmp / "untrusted.loman"))
    print(f"function, normal load : {normal.dag.nodes[parse_nodekey('b')][NodeAttributes.FUNC]}")
    print(f"function, allow_code=False: {safe.dag.nodes[parse_nodekey('b')][NodeAttributes.FUNC]}")
    return normal, safe


@app.cell
def _(mo, normal, safe):
    mo.vstack([safe, normal])
    return


@app.cell
def _(safe):
    safe.src["b"]
    return


@app.cell
def _(normal):
    normal.src["b"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Durability, and pandas 2 versus 3

    **Values are durable.** The format carries a version, changes to it are
    additive, and files written by every earlier version are held as fixtures in
    the test suite and asserted to still load *and still recompute*.

    **Functions are only as durable as your code.** A node's function is stored as
    a module path and a qualified name, so it resolves only while that module
    still exports that name. A dill-serialized one is worse: not portable across
    Python versions.

    So a saved computation is a durable record of *what* was computed, and a
    best-effort record of *how*.

    On pandas: version 2 defaults datetimes to nanoseconds and version 3 to
    microseconds. Each value's resolution is therefore **recorded** rather than
    assumed, so a file written under one loads correctly under the other. That is
    why loman requires pandas 2.0 or later — the APIs that read and restore
    resolution arrived there, and below that the values would be silently wrong
    rather than raising.
    """)
    return


@app.cell
def _(Computation, pd, tmp):
    print(f"running pandas {pd.__version__}\n")
    for _unit in ["s", "ms", "us", "ns"]:
        _index = pd.date_range("2020-01-01", periods=3, freq="min").as_unit(_unit)
        _frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=_index)

        _comp = Computation()
        _comp.add_node("frame", value=_frame)
        _comp.save(str(tmp / f"unit_{_unit}.loman"))

        _back = Computation.load(str(tmp / f"unit_{_unit}.loman")).v.frame
        print(f"  resolution {_unit:>2s} -> {_back.index.unit:>2s}  exact: {_back.equals(_frame)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    | Scenario | How |
    |---|---|
    | Basic round-trip | `comp.save(path)` / `Computation.load(path)` |
    | A readable text file | `comp.write_json(path)` / `Computation.read_json(path)` |
    | Exclude a node | `add_node(..., serialize=False)` |
    | Lambda / closure | `ComputationSerializer(use_dill_for_functions=True)` |
    | PINNED state | Preserved automatically |
    | ERROR state | Builtin types rebuilt; others become `DeserializedError` |
    | Pandas / NumPy | Handled automatically, at any resolution |
    | Large values | Out of line and compressed, decided per value |
    | Repeated checkpoints | `container="dir"` |
    | Values in a bucket or database | `add_node(store=...)` + `save(stores={...})` |
    | Untrusted file | `load(path, allow_code=False)` |
    | Custom type | Register a `CustomTransformer` |
    | Custom destination | Subclass `BlobStore` — two methods |

    Nothing here removed or changed an existing API: `write_json`, `read_json`,
    `write_dill` and `serialize=False` all behave as they did, and every new
    parameter is keyword-only with a default.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
