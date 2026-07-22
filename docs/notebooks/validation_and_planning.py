"""Examples for validating and planning Loman computations."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.17.6",
#     "loman",
# ]
#
# [tool.uv.sources]
# loman = { path = "../..", editable = true }
# ///

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Validation and execution planning

    Loman can inspect a computation before running it:

    - `validate()` checks the **whole graph** for cycles, missing inputs, placeholders,
      failed nodes, and unknown executors.
    - `plan(target)` examines only the dependencies needed for a **specific result**.
      It shows what would run, in dependency order, without executing or changing anything.
    - `plan()` examines the work remaining across the **whole graph**.

    This notebook develops those ideas through a series of small use cases. It has no
    controls: run it from top to bottom, then edit values or graph definitions to explore.
    """)
    return


@app.cell
def _():
    import marimo as mo

    from loman import Computation

    def names(nodes):
        return [str(node) for node in nodes]

    def validation_summary(report):
        return {
            "is_valid": report.is_valid,
            "is_ready": report.is_ready,
            "cycles": [names(component) for component in report.cycles],
            "placeholders": names(report.placeholders),
            "uninitialized_inputs": names(report.uninitialized_inputs),
            "error_nodes": names(report.error_nodes),
            "missing_executors": [(str(node), executor) for node, executor in report.missing_executors],
        }

    def plan_summary(plan):
        return {
            "is_feasible": plan.is_feasible,
            "targets": None if plan.targets is None else names(plan.targets),
            "execution_order": names(plan.execution_order),
            "current_nodes": names(plan.current_nodes),
            "blocked_nodes": names(plan.blocked_nodes),
            "executor_assignments": [(str(node), executor) for node, executor in plan.executor_assignments],
        }

    return Computation, mo, plan_summary, validation_summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Preview work before execution

    Consider an order total with three supplied inputs and three calculations. The
    `discount` and `total` nodes share a named executor; here it points to the computation's
    normal executor, but in a real application it could identify a process or thread pool.

    Calling `plan("total")` should identify the supplied inputs as current and list the
    calculations in the exact dependency order required to produce `total`.
    """)
    return


@app.cell
def _(Computation):
    order = Computation()
    order.executor_map["pricing"] = order.default_executor
    order.add_node("quantity", value=3)
    order.add_node("unit_price", value=12.5)
    order.add_node("discount_rate", value=0.1)
    order.add_node("subtotal", lambda quantity, unit_price: quantity * unit_price)
    order.add_node(
        "discount",
        lambda subtotal, discount_rate: subtotal * discount_rate,
        executor="pricing",
    )
    order.add_node("total", lambda subtotal, discount: subtotal - discount, executor="pricing")
    order
    return (order,)


@app.cell
def _(order, plan_summary, validation_summary):
    order_validation_before = order.validate()
    order_plan_before = order.plan("total")
    {
        "validation": validation_summary(order_validation_before),
        "plan": plan_summary(order_plan_before),
    }
    return (order_plan_before,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The graph is both structurally valid and ready. Planning did not calculate anything:
    the three calculation nodes are still merely pending. Now execute the target and plan
    it again.
    """)
    return


@app.cell
def _(order, order_plan_before):
    # Depending on the first plan makes the before/after sequence explicit to Marimo.
    _ = order_plan_before
    order.compute("total")
    order
    return


@app.cell
def _(order):
    order.to_df()
    return


@app.cell
def _(order, plan_summary):
    order_plan_after = order.plan("total")
    {
        "values": order.to_dict(),
        "next_plan": plan_summary(order_plan_after),
    }
    return (order_plan_after,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The second plan has an empty `execution_order`: `total` is already current. This makes
    `plan()` useful as a dry-run check before an expensive calculation, and as confirmation
    that a requested result needs no work.

    ## 2. Understand selective recalculation

    Changing `discount_rate` should invalidate `discount` and `total`, but not `subtotal`.
    The next plan makes that selective recalculation visible before it happens.
    """)
    return


@app.cell
def _(order, order_plan_after):
    _ = order_plan_after
    order.insert("discount_rate", 0.2, force=True)
    order
    return


@app.cell
def _(order, plan_summary):
    recalculation_plan = order.plan("total")
    plan_summary(recalculation_plan)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `subtotal` now appears in `current_nodes`, while only `discount` and `total` appear in
    `execution_order`. This is the same incremental behavior that `compute("total")` will
    follow, exposed without doing the work.

    ## 3. Diagnose a missing input

    A graph can be structurally valid but not ready. Here, `raw_data` is a legitimate input
    node whose value has not yet been supplied. Validation identifies that root cause;
    planning shows every requested node that it blocks.
    """)
    return


@app.cell
def _(Computation):
    import statistics

    pipeline = Computation()
    pipeline.add_node("raw_data")
    pipeline.add_node("clean_data", lambda raw_data: [value for value in raw_data if value is not None])
    pipeline.add_node("mean", lambda clean_data: statistics.mean(clean_data))
    pipeline
    return (pipeline,)


@app.cell
def _(pipeline, plan_summary, validation_summary):
    pipeline_validation_missing = pipeline.validate()
    pipeline_plan_blocked = pipeline.plan("mean")
    {
        "validation": validation_summary(pipeline_validation_missing),
        "plan": plan_summary(pipeline_plan_blocked),
    }
    return (pipeline_plan_blocked,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice the distinction:

    - `is_valid` is `True`: the graph has a coherent structure.
    - `is_ready` is `False`: an external value is still required.
    - `blocked_nodes` includes `raw_data`, `clean_data`, and `mean`: none of that branch can
      complete yet.

    Supplying the input removes the blocker and produces a feasible plan.
    """)
    return


@app.cell
def _(pipeline, pipeline_plan_blocked):
    _ = pipeline_plan_blocked
    pipeline.insert("raw_data", [10, None, 20, 30])
    pipeline
    return


@app.cell
def _(pipeline, plan_summary):
    pipeline_plan_ready = pipeline.plan("mean")
    plan_summary(pipeline_plan_ready)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Plan useful work in a partially broken graph

    `validate()` intentionally checks the entire graph. A production computation may contain
    an unfinished or broken branch that is unrelated to today's requested output. A targeted
    plan answers the more practical question: **can this particular result be calculated?**

    The following graph has a healthy reporting branch and an unrelated export branch with
    a placeholder dependency.
    """)
    return


@app.cell
def _(Computation):
    mixed = Computation()
    mixed.add_node("sales", value=[100, 125, 150])
    mixed.add_node("sales_total", lambda sales: sum(sales))
    mixed.add_node("sales_report", lambda sales_total: f"Sales: {sales_total}")
    mixed.add_node("export_result", lambda database_connection, sales_report: (database_connection, sales_report))
    mixed
    return (mixed,)


@app.cell
def _(mixed, plan_summary, validation_summary):
    mixed_validation = mixed.validate()
    report_plan = mixed.plan("sales_report")
    whole_graph_plan = mixed.plan()
    {
        "whole_graph_validation": validation_summary(mixed_validation),
        "sales_report_plan": plan_summary(report_plan),
        "whole_graph_plan": plan_summary(whole_graph_plan),
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The whole graph is invalid because `database_connection` is a placeholder, and the whole
    graph plan marks the export branch as blocked. The targeted `sales_report` plan remains
    feasible because that placeholder is not one of its dependencies.

    This distinction is useful for large computations containing optional outputs, staged
    development, or environment-specific integrations.

    ## 5. Detect dependency cycles

    Cycles are structural errors: no member can be evaluated first. Validation reports each
    cyclic strongly connected group, while planning marks the cycle and its required
    descendants as blocked.

    Computing the target would raise an error. Validation makes the cycle visible before any
    execution is attempted.
    """)
    return


@app.cell
def _(Computation):
    cyclic = Computation()
    cyclic.add_node("positions", lambda risk: risk + 1)
    cyclic.add_node("value", lambda positions: positions * 2)
    cyclic.add_node("risk", lambda value: value / 10)
    cyclic.add_node("report", lambda value: f"Value: {value}")
    cyclic
    return (cyclic,)


@app.cell
def _(cyclic, plan_summary, validation_summary):
    cycle_validation = cyclic.validate()
    cycle_plan = cyclic.plan("report")
    {
        "validation": validation_summary(cycle_validation),
        "plan": plan_summary(cycle_plan),
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Use a current value as a calculation boundary

    Sometimes an intermediate result is restored from a checkpoint, supplied from a cache,
    or deliberately pinned for an experiment. Its original inputs may be unavailable, but
    downstream work can still proceed from the current value.

    Here `calibrated_model` is pinned. Planning `forecast` stops there and does not require
    the unavailable training data behind it.
    """)
    return


@app.cell
def _(Computation):
    checkpointed = Computation()
    checkpointed.add_node("training_data")
    checkpointed.add_node("calibrated_model", lambda training_data: {"mean": sum(training_data) / len(training_data)})
    checkpointed.insert("calibrated_model", {"mean": 42.0})
    checkpointed.pin("calibrated_model")
    checkpointed.add_node("scenario", value=1.1)
    checkpointed.add_node("forecast", lambda calibrated_model, scenario: calibrated_model["mean"] * scenario)
    checkpointed
    return (checkpointed,)


@app.cell
def _(checkpointed, plan_summary, validation_summary):
    checkpoint_validation = checkpointed.validate()
    forecast_plan = checkpointed.plan("forecast")
    {
        "whole_graph_validation": validation_summary(checkpoint_validation),
        "forecast_plan": plan_summary(forecast_plan),
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Whole-graph validation still tells us that `training_data` is unavailable. The targeted
    plan is nevertheless feasible: `calibrated_model` and `scenario` are current boundaries,
    so only `forecast` needs to run.

    ## 7. Diagnose a large computation with nested blocks

    Small examples make each failure obvious. The real value of planning appears when a graph
    is large enough that manually following edges is tedious.

    The portfolio pipeline below uses reusable blocks under nested paths:

    - three feed blocks normalize positions, prices, and FX data;
    - a valuation block combines those feeds and schedules stress calculation on a named
      executor;
    - a reporting branch consumes the valuation result.

    It also contains four realistic configuration mistakes:

    1. The prices feed was added without linking its `raw` input.
    2. The FX feed links to `inputs/fx_spots`, but the supplied node is `inputs/fx_spot`.
    3. The valuation block requests a `quant_pool` executor that was never configured.
    4. `report/draft` and `report/signoff` depend on one another.
    """)
    return


@app.cell
def _(Computation):
    def make_feed_block():
        block = Computation()
        block.add_node("raw")
        block.add_node("records", lambda raw: raw)
        block.add_node("count", lambda records: len(records))
        return block

    def make_valuation_block():
        block = Computation()
        block.add_node("positions")
        block.add_node("prices")
        block.add_node("fx")
        block.add_node("market_value", lambda positions, prices, fx: (positions, prices, fx))
        block.add_node("stress_loss", lambda market_value: market_value, executor="quant_pool")
        return block

    portfolio = Computation()
    portfolio.add_node("inputs/positions", value=[{"symbol": "ABC", "quantity": 100}])
    portfolio.add_node("inputs/prices", value={"ABC": 25.0})
    portfolio.add_node("inputs/fx_spot", value=1.25)

    portfolio.add_block(
        "feeds/positions",
        make_feed_block(),
        keep_values=False,
        links={"raw": "inputs/positions"},
    )
    portfolio.add_block("feeds/prices", make_feed_block(), keep_values=False)
    portfolio.add_block(
        "feeds/fx",
        make_feed_block(),
        keep_values=False,
        links={"raw": "inputs/fx_spots"},
    )
    portfolio.add_block(
        "risk/equity",
        make_valuation_block(),
        keep_values=False,
        links={
            "positions": "feeds/positions/records",
            "prices": "feeds/prices/records",
            "fx": "feeds/fx/records",
        },
    )

    portfolio.add_node(
        "report/draft",
        lambda stress_loss, signoff: (stress_loss, signoff),
        kwds={"stress_loss": "risk/equity/stress_loss", "signoff": "report/signoff"},
    )
    portfolio.add_node("report/signoff", lambda draft: draft, kwds={"draft": "report/draft"})
    portfolio.add_node("report/final", lambda draft: draft, kwds={"draft": "report/draft"})
    portfolio
    return (portfolio,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The complete issue list takes one line. Instead of searching through every block, it
    identifies the precise paths containing the root problems.
    """)
    return


@app.cell
def _(portfolio):
    portfolio.validate().to_df()
    return


@app.cell
def _(portfolio):
    # see the cycle
    portfolio.draw("report")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Validation answers **what is wrong anywhere in the graph**. Planning the desired final
    report answers **how those problems affect this output**. `blocked_by` carries each root
    cause through the dependency graph, so the final row explains all reasons the report
    cannot run without manually tracing an edge.

    The table keeps two related concepts separate:

    - `state` is Loman's existing node state (`UPTODATE`, `COMPUTABLE`, `STALE`, and so on).
    - `plan_status` is contextual to this target: `current`, `pending`, or `blocked`.

    A node can therefore be `STALE` in the graph while being `blocked` in this plan because
    an upstream input or executor is unavailable.
    """)
    return


@app.cell
def _(portfolio):
    portfolio.plan("report/final").to_df()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two tables separate diagnosis from impact:

    - `feeds/prices/raw` reveals the omitted block link.
    - `inputs/fx_spots` exposes the typo next to the existing `inputs/fx_spot` node.
    - `risk/equity/stress_loss` names the missing executor and every downstream node it blocks.
    - the cycle is shown once as a connected group rather than as a confusing compute-time
      traceback.
    - current and independently runnable nodes remain visible, making clear how far the graph
      can proceed after each fix.

    This is the intended investigation loop for a large DAG:

    ```python
    comp.validate().to_df()
    comp.plan("desired/output").to_df()
    ```

    ## Where to use each API

    | Question | API |
    | --- | --- |
    | Is the complete graph well formed and fully supplied? | `comp.validate()` |
    | What would run to produce one result? | `comp.plan("result")` |
    | Can several outputs be produced together? | `comp.plan(["a", "b"])` |
    | What work and blockers remain anywhere in the graph? | `comp.plan()` |
    | Has a target become current after execution? | `comp.plan("result").execution_order == ()` |
    | Which root issue blocks each downstream node? | `comp.plan("result").to_df()` |

    Try changing supplied values, inserting missing inputs, removing a cyclic dependency, or
    unpinning the checkpointed model. Marimo will rerun the dependent cells and expose how
    each graph change affects validation and planning.
    """)
    return


if __name__ == "__main__":
    app.run()
