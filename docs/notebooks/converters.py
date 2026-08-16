"""Examples of converting and validating node values in Loman computations."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.23.13",
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
    # Converters: coercing and validating node values

    A node can be given a `converter`: a callable Loman applies to a value on its way
    **into** the node. The node stores what the converter returns, not what was supplied.

    That one hook covers two jobs:

    - **Coercion** — force a value into the type the rest of the graph expects
      (`converter=float`).
    - **Validation** — check a value and raise if it is unacceptable. Loman has no
      separate validator hook; a converter that raises is how you get one.

    Both push the check to the graph's boundary, so downstream nodes can trust their
    inputs instead of re-checking them.

    This notebook has a few interactive controls. Run it top to bottom, then edit the
    inputs to see how each node reacts.
    """)
    return


@app.cell
def _():
    import marimo as mo

    from loman import Computation

    def node_report(comp, *names):
        """Summarise state, stored value and type for the given nodes."""
        rows = []
        for name in names:
            state = comp.s[name]
            value = comp.v[name]
            if hasattr(value, "exception"):
                shown, type_name = f"{type(value.exception).__name__}: {value.exception}", "—"
            else:
                shown, type_name = repr(value), type(value).__name__
            rows.append({"node": str(name), "state": str(state), "value": shown, "type": type_name})
        return rows

    return Computation, mo, node_report


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Coercing a value at the boundary

    Input often arrives as text — from a CSV, a form, a config file. Attaching
    `converter=float` to the input node means the conversion happens once, at the point
    of entry, rather than in every calculation that reads it.

    Type a value below. It is inserted as a **string**, but the node stores a `float`.
    """)
    return


@app.cell
def _(mo):
    raw_input_box = mo.ui.text(value="3.5", label="Raw text value to insert:")
    raw_input_box
    return (raw_input_box,)


@app.cell
def _(Computation, node_report, raw_input_box):
    coerce = Computation()
    coerce.add_node("quantity", converter=float)
    coerce.add_node("doubled", lambda quantity: quantity * 2)

    try:
        coerce.insert("quantity", raw_input_box.value)
        coerce.compute_all()
        coerce_outcome = "inserted"
    except ValueError as exc:
        coerce_outcome = f"insert raised {type(exc).__name__}: {exc}"

    {"outcome": coerce_outcome, "nodes": node_report(coerce, "quantity", "doubled")}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The string never reaches the node, so `doubled` multiplies a number rather than
    accidentally repeating a string. Try entering `abc`: the conversion fails, the insert
    raises, and `quantity` is left in `ERROR` — the bad value is not stored.

    ## 2. Converters apply to computed values too

    This is the part that surprises people. A converter belongs to the **node**, not to
    the insert path, so it also runs on the value the node calculates for itself.

    Below, the lambda returns an `int`. The node holds a `float`.
    """)
    return


@app.cell
def _(Computation, node_report):
    computed = Computation()
    computed.add_node("a", value=1)
    computed.add_node("b", lambda a: a + 1, converter=float)
    computed.compute_all()
    node_report(computed, "a", "b")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `b` computed `2` and stored `2.0`. This makes a converter a useful guarantee about a
    node's contents no matter how the value got there — which is exactly what you want
    when the node is a published output that other systems read.

    ## 3. A converter that raises is a validator

    A converter is free to inspect its argument and refuse it. Raising leaves the node in
    `ERROR` state and the value unstored.

    The important discipline: **return the value when it passes**. A validator that
    forgets to return silently replaces the node's value with `None`.
    """)
    return


@app.cell
def _(mo):
    size_slider = mo.ui.slider(-5, 10, value=4, label="Value to insert into `size`:")
    size_slider
    return (size_slider,)


@app.cell
def _(Computation, node_report, size_slider):
    def positive(x):
        if x <= 0:
            msg = f"must be positive, got {x}"
            raise ValueError(msg)
        return x  # a validator must return the value it accepts

    validated = Computation()
    validated.add_node("size", converter=positive)
    validated.add_node("area", lambda size: size**2)

    try:
        validated.insert("size", size_slider.value)
        validated.compute_all()
        validate_outcome = "accepted"
    except ValueError as exc:
        validate_outcome = f"rejected — insert raised {type(exc).__name__}: {exc}"

    {"outcome": validate_outcome, "nodes": node_report(validated, "size", "area")}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Drag the slider to zero or below. The insert is rejected, `size` goes to `ERROR`, and
    `area` never computes off a nonsensical input — the failure surfaces at the boundary
    instead of several nodes downstream.

    ## 4. Coercing and validating together

    In practice the two jobs combine: coerce the value into the right type, then assert
    the result is usable. A trade ticket arriving as text is a good example — each field
    is parsed and checked as it enters the graph.
    """)
    return


@app.cell
def _(mo):
    ticket_notional = mo.ui.text(value="1000000", label="notional")
    ticket_currency = mo.ui.text(value="usd", label="currency")
    mo.vstack([ticket_notional, ticket_currency])
    return ticket_currency, ticket_notional


@app.cell
def _(Computation, node_report, ticket_currency, ticket_notional):
    known_currencies = {"USD", "EUR", "GBP", "JPY"}

    def positive_amount(x):
        amount = float(x)
        if amount <= 0:
            msg = f"notional must be positive, got {amount}"
            raise ValueError(msg)
        return amount

    def currency_code(x):
        code = str(x).strip().upper()
        if code not in known_currencies:
            msg = f"unknown currency {code!r}, expected one of {sorted(known_currencies)}"
            raise ValueError(msg)
        return code

    ticket = Computation()
    ticket.add_node("notional", converter=positive_amount)
    ticket.add_node("currency", converter=currency_code)
    ticket.add_node("description", lambda notional, currency: f"{notional:,.2f} {currency}")

    ticket_errors = []
    for _field, _element in (("notional", ticket_notional), ("currency", ticket_currency)):
        try:
            ticket.insert(_field, _element.value)
        except ValueError as exc:
            ticket_errors.append(f"{_field}: {exc}")
    ticket.compute_all()

    {
        "rejected": ticket_errors or "nothing rejected",
        "nodes": node_report(ticket, "notional", "currency", "description"),
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that `currency` is normalized as well as checked: `usd` is stored as `USD`, so
    downstream nodes can compare against uppercase codes without defensive `.upper()`
    calls. Try `chf` to see a rejection, and note that `description` is then unable to
    compute while the other field remains fine.

    ## 5. How failures reach your code

    A failed conversion always leaves the node in `ERROR`, with the exception available at
    `comp.v[node].exception`. Whether it *also* propagates to the caller depends on how
    the value arrived:

    | Value arrives via | Node state | Exception raised to caller |
    | --- | --- | --- |
    | `add_node(value=...)` | `ERROR` | Yes |
    | `insert` / `insert_many` | `ERROR` | Yes |
    | A calculation (`compute`, `compute_all`) | `ERROR` | No |

    Insertion raises because supplying a value is something your code just did and can
    handle on the spot. A conversion failure during computation is treated like any other
    node failure: the node is marked `ERROR`, its descendants cannot proceed, and the run
    continues so unrelated branches still make progress.

    Below, the same `positive` validator fails on a *computed* value. `compute_all()`
    returns normally, and the healthy branch still finishes.
    """)
    return


@app.cell
def _(Computation, node_report):
    def positive_check(x):
        if x <= 0:
            msg = f"must be positive, got {x}"
            raise ValueError(msg)
        return x

    compute_fail = Computation()
    compute_fail.add_node("a", value=1)
    compute_fail.add_node("b", lambda a: a - 10, converter=positive_check)
    compute_fail.add_node("c", lambda b: b * 2)
    compute_fail.add_node("unrelated", lambda a: a * 100)
    compute_fail.compute_all()  # note: does not raise
    node_report(compute_fail, "a", "b", "c", "unrelated")
    return (compute_fail,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `b` is `ERROR`, `c` is left `STALE` behind it, and `unrelated` is up to date. A node
    left in `ERROR` by a failed conversion is reported by `validate()` exactly like any
    other failed node, so the usual diagnostics apply.
    """)
    return


@app.cell
def _(compute_fail):
    compute_fail.validate().to_df()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Three things worth knowing

    **Redefining a node drops its converter.** `add_node` replaces the whole node
    definition, so calling it again without `converter=` leaves the node unconverted. Pass
    the converter every time you redefine the node.

    **Converters survive saving, but must be importable.** `write_json`/`read_json` and
    `save`/`load` store a converter the way they store a node's function: by reference. A
    module-level function or a builtin such as `float` comes back intact, so a reloaded
    graph still coerces and still validates. A `lambda` has no importable path and raises
    `SerializationError` naming the node, so define converters at module level rather than
    inline.

    **`add_block` keeps converters.** A validated block template stays validated wherever
    it is used.

    The cell below demonstrates all three.
    """)
    return


@app.cell
def _(Computation):
    import io

    # Redefinition drops the converter
    redefine = Computation()
    redefine.add_node("x", converter=float)
    redefine.add_node("x")  # no converter= this time
    redefine.insert("x", 5)

    # An importable converter survives a JSON round trip
    saved = Computation()
    saved.add_node("y", value=1, converter=float)
    _buffer = io.StringIO()
    saved.write_json(_buffer)
    _buffer.seek(0)
    reloaded = Computation.read_json(_buffer)
    reloaded.insert("y", 7, force=True)

    # A lambda converter has no importable path
    inline = Computation()
    inline.add_node("w", value=1, converter=lambda v: float(v))
    try:
        inline.write_json(io.StringIO())
        lambda_outcome = "saved"
    except Exception as exc:
        lambda_outcome = type(exc).__name__

    # add_block keeps it
    template = Computation()
    template.add_node("z", converter=float)
    host = Computation()
    host.add_block("blk", template)
    host.insert("blk/z", 2)

    {
        "after redefining without converter=": type(redefine.v["x"]).__name__,
        "after write_json / read_json": type(reloaded.v["y"]).__name__,
        "saving a lambda converter": lambda_outcome,
        "inside a block added with add_block": type(host.v["blk/z"]).__name__,
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Only the first is `int` — redefining the node dropped its converter. The round-tripped
    node is still a `float`, the lambda is refused with `SerializationError`, and the block
    carries its converter along.

    ## Summary

    | Goal | Converter |
    | --- | --- |
    | Accept text input as a number | `converter=float` |
    | Normalize a code or label | `lambda x: str(x).strip().upper()` |
    | Reject an out-of-range value | raise in the converter, return `x` when it passes |
    | Coerce then check | convert first, validate the result, return it |
    | Guarantee a computed output's type | attach the converter to the calculation node |

    Keep converters cheap and free of side effects: one runs on every set of the node,
    including repeat inserts, and its return value is what the whole graph downstream
    sees.

    See also the [Converters and Validation](../user/features/creating/converters_and_validation/)
    page in the user guide.
    """)
    return


if __name__ == "__main__":
    app.run()
