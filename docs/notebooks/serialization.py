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

__generated_with = "0.23.0"
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
    _raw = json.loads(buf_lambda.getvalue())
    # show only the lambda node's func field, blob truncated
    _lambda_node = next(n for n in _raw["nodes"] if n["func"] is not None and n["func"].get("type") == "dill_func")
    _func = dict(_lambda_node["func"])
    _func["blob"] = _func["blob"][:40] + "..."
    {
        "key": _lambda_node["key"],
        "func": _func,
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Closures that capture variables from an enclosing scope also work:
    """)
    return


@app.cell
def _(Computation, ComputationSerializer, io):
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
    `write_json` preserves the exception type, message, and traceback as strings —
    so you can reload a failed computation and inspect what went wrong, even without
    the original exception class available:
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
    comp_df.add_node("prices", value=prices)
    comp_df.add_node("enriched", enrich)
    comp_df.compute_all()

    buf_df = io.StringIO()
    comp_df.write_json(buf_df)
    buf_df.seek(0)
    comp_df2 = Computation.read_json(buf_df)
    comp_df2.value("enriched")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    | Scenario | How |
    |---|---|
    | Basic round-trip | `comp.write_json(path)` / `Computation.read_json(path)` |
    | Exclude a node | `add_node(..., serialize=False)` |
    | Lambda / closure | `ComputationSerializer(use_dill_for_functions=True)` |
    | PINNED state | Preserved automatically |
    | ERROR state | Exception + traceback stored as strings |
    | Pandas / NumPy | Handled automatically |

    The JSON format is designed for short-term inspection and post-mortem debugging,
    not long-term archival. The format may change between releases.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
