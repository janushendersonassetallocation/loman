"""Example: Building a large computation with repeated pipelines."""

# ruff: noqa: TRY003

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.23.0",
#     "numpy",
#     "pandas",
#     "loman",
# ]
#
# [tool.uv.sources]
# loman = { path = "../..", editable = true }
# ///

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Large Repeated Pipelines

    This notebook builds a portfolio computation with one reusable block per
    instrument. It demonstrates every computation utility in `loman.util`:

    - `add_repeated_blocks` creates keyed copies of a calculation block.
    - `add_fan_out` slices shared market data and broadcasts shared settings.
    - `add_fan_in` concatenates instrument reports and calculates totals.
    - `add_repeated_pipeline` builds a second stress pipeline in one call.

    The example uses 24 instruments and finishes with more than 220 ordinary
    Loman nodes. The utilities only build the graph: values are still evaluated
    lazily by the standard Loman scheduler.
    """)
    return


@app.cell
def _():
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import pandas as pd

    from loman import Computation, States, util

    def select_instrument(frame, instrument_id):
        """Select one instrument while retaining a dataframe result."""
        return frame.loc[[instrument_id]].copy()

    def concat_instruments(frames):
        """Concatenate keyed instrument dataframes."""
        return pd.concat(frames, names=["instrument_id"])

    def aggregate_portfolio(frames):
        """Aggregate selected numeric columns from keyed reports."""
        report = pd.concat(frames, names=["instrument_id"])
        return report[["market_value", "daily_pnl", "var_95"]].sum()

    return (
        Computation,
        States,
        ThreadPoolExecutor,
        aggregate_portfolio,
        concat_instruments,
        np,
        pd,
        select_instrument,
        util,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Define one reusable instrument block

    The block has two inputs and five calculations. `instrument_data` receives
    one row sliced from the shared market dataframe. `risk_settings` receives
    the same settings object for every block.

    `audit` deliberately sits outside the path to `report`. It makes the graph
    larger while demonstrating that unused branches are not computed.
    """)
    return


@app.cell
def _(Computation):
    instrument_block = Computation()
    instrument_block.add_node("instrument_data")
    instrument_block.add_node("risk_settings")

    def validate_instrument(instrument_data):
        required = {"quantity", "price", "previous_close", "volatility", "beta"}
        missing = required - set(instrument_data.columns)
        if missing:
            raise ValueError(f"Missing instrument fields: {sorted(missing)}")
        return instrument_data.copy()

    def value_instrument(validate_instrument):
        result = validate_instrument.copy()
        result["market_value"] = result["quantity"] * result["price"]
        result["daily_pnl"] = result["quantity"] * (result["price"] - result["previous_close"])
        return result

    def calculate_risk(value_instrument, risk_settings):
        result = value_instrument.copy()
        horizon = risk_settings["horizon_days"] ** 0.5
        result["var_95"] = (
            result["market_value"].abs() * result["volatility"] * risk_settings["confidence_multiplier"] * horizon
        )
        result["beta_exposure"] = result["market_value"] * result["beta"]
        return result

    def format_report(calculate_risk):
        columns = [
            "sector",
            "quantity",
            "price",
            "market_value",
            "daily_pnl",
            "var_95",
            "beta_exposure",
        ]
        return calculate_risk[columns]

    def build_audit(calculate_risk):
        return {
            "rows": len(calculate_risk),
            "columns": tuple(calculate_risk.columns),
            "checked": True,
        }

    instrument_block.add_node("validated", validate_instrument, kwds={"instrument_data": "instrument_data"})
    instrument_block.add_node("valued", value_instrument, kwds={"validate_instrument": "validated"})
    instrument_block.add_node(
        "risk",
        calculate_risk,
        kwds={"value_instrument": "valued", "risk_settings": "risk_settings"},
    )
    instrument_block.add_node("report", format_report, kwds={"calculate_risk": "risk"})
    instrument_block.add_node("audit", build_audit, kwds={"calculate_risk": "risk"})
    instrument_block.draw(collapse_all=False, graph_attr={"rankdir": "LR"})
    return (instrument_block,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Create deterministic portfolio data

    The portfolio has 24 instruments spread across four sectors. A thread pool
    lets independent instrument blocks run concurrently; the graph would work
    identically with Loman's default executor.
    """)
    return


@app.cell
def _(Computation, ThreadPoolExecutor, np, pd):
    instrument_ids = tuple(f"INST-{number:03d}" for number in range(1, 25))
    sectors = ("Technology", "Financials", "Healthcare", "Industrials")
    positions = np.arange(1, len(instrument_ids) + 1)

    market_data = pd.DataFrame(
        {
            "sector": [sectors[index % len(sectors)] for index in range(len(instrument_ids))],
            "quantity": np.where(positions % 5 == 0, -positions * 7, positions * 10),
            "previous_close": 80.0 + positions * 2.25,
            "price": 80.0 + positions * 2.25 + np.sin(positions) * 1.5,
            "volatility": 0.12 + (positions % 8) * 0.015,
            "beta": 0.75 + (positions % 7) * 0.08,
        },
        index=pd.Index(instrument_ids, name="instrument_id"),
    )
    risk_settings = {"confidence_multiplier": 1.65, "horizon_days": 10}

    executor = ThreadPoolExecutor(max_workers=8)
    comp = Computation(default_executor=executor)
    comp.add_node("market_data", value=market_data)
    comp.add_node("risk_settings", value=risk_settings)
    market_data
    return comp, instrument_ids, market_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Compose repeated blocks, fan-out, and fan-in

    The three small helpers can be combined explicitly when a block has several
    inputs or the graph needs several aggregate outputs.

    - The first fan-out calls `select_instrument(market_data, instrument_id)`.
    - The second broadcasts `risk_settings` unchanged.
    - Two fan-ins consume the same per-instrument reports: one concatenates all
      rows, while the other returns portfolio totals.
    """)
    return


@app.cell
def _(
    aggregate_portfolio,
    comp,
    concat_instruments,
    instrument_block,
    instrument_ids,
    select_instrument,
    util,
):
    blocks = util.add_repeated_blocks(
        comp,
        instrument_block,
        instrument_ids,
        base_path="instruments",
    )

    instrument_inputs = {key: path / "instrument_data" for key, path in blocks.items()}
    util.add_fan_out(comp, "market_data", instrument_inputs, transform=select_instrument)

    settings_inputs = {key: path / "risk_settings" for key, path in blocks.items()}
    util.add_fan_out(comp, "risk_settings", settings_inputs)

    report_sources = {key: path / "report" for key, path in blocks.items()}
    util.add_fan_in(comp, "portfolio_report", report_sources, combine=concat_instruments)
    util.add_fan_in(comp, "portfolio_totals", report_sources, combine=aggregate_portfolio)
    comp
    return (blocks,)


@app.cell
def _(comp, pd):
    graph_size = pd.Series(
        {
            "nodes": len(comp.nodes()),
            "dependency_edges": comp.dag.number_of_edges(),
            "instrument_blocks": 24,
        },
        name="main portfolio graph",
    )
    graph_size
    return (graph_size,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The detailed graph has more than 170 nodes. Loman's block-aware view keeps
    it readable by collapsing each repeated namespace to one block node.
    """)
    return


@app.cell
def _(comp):
    comp.draw(collapse_all=True, graph_attr={"rankdir": "LR", "size": "14"})
    return


@app.cell
def _(comp):
    comp.draw(
        "instruments",
        node_transformations={"instruments/INST-001": "expand"},
        graph_attr={"rankdir": "LR", "size": "20"},
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Compute only the requested outputs

    Computing the two portfolio outputs evaluates their shared ancestors once.
    The 24 `audit` nodes become computable when their shared `risk` predecessors
    finish, but they are not executed because neither output depends on them.
    """)
    return


@app.cell
def _(comp):
    comp.compute(["portfolio_report", "portfolio_totals"])
    portfolio_report = comp.v.portfolio_report
    portfolio_totals = comp.v.portfolio_totals
    portfolio_totals.to_frame("value")
    return portfolio_report, portfolio_totals


@app.cell
def _(portfolio_report):
    portfolio_report
    return


@app.cell
def _(States, blocks, comp, pd):
    audit_states = pd.Series(
        {str(key): comp.state(path / "audit").name for key, path in blocks.items()},
        name="audit state",
    )
    audit_summary = audit_states.value_counts().rename_axis("state").to_frame("nodes")
    assert set(audit_states) == {States.COMPUTABLE.name}
    audit_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Add a second pipeline with one helper

    `add_repeated_pipeline` is the concise form for the common one-input,
    one-output pattern. Here it creates another 24 blocks for a downside stress
    scenario and combines their outputs into `stress_report`.
    """)
    return


@app.cell
def _(Computation):
    stress_block = Computation()
    stress_block.add_node("data")

    def apply_stress(data):
        result = data.copy()
        result["stressed_price"] = result["price"] * (1.0 - result["shock"])
        result["stress_pnl"] = result["quantity"] * (result["stressed_price"] - result["price"])
        return result[["sector", "price", "stressed_price", "stress_pnl"]]

    stress_block.add_node("scenario", apply_stress)
    return (stress_block,)


@app.cell
def _(
    comp,
    concat_instruments,
    instrument_ids,
    market_data,
    select_instrument,
    stress_block,
    util,
):
    sector_shocks = {
        "Technology": 0.12,
        "Financials": 0.09,
        "Healthcare": 0.07,
        "Industrials": 0.10,
    }
    stress_market_data = market_data.assign(shock=market_data["sector"].map(sector_shocks))
    comp.add_node("stress_market_data", value=stress_market_data)

    stress_pipeline = util.add_repeated_pipeline(
        comp,
        stress_block,
        instrument_ids,
        base_path="stress/instruments",
        source="stress_market_data",
        block_input="data",
        block_output="scenario",
        result="stress_report",
        transform=select_instrument,
        combine=concat_instruments,
    )
    stress_pipeline
    return (stress_pipeline,)


@app.cell
def _(comp, graph_size, pd):
    expanded_graph_size = pd.DataFrame(
        {
            "before stress pipeline": graph_size,
            "after stress pipeline": pd.Series(
                {
                    "nodes": len(comp.nodes()),
                    "dependency_edges": comp.dag.number_of_edges(),
                    "instrument_blocks": 48,
                }
            ),
        }
    )
    expanded_graph_size
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Even though the whole stress pipeline exists, we can request one scenario
    node. Only that block's required calculations become up to date.
    """)
    return


@app.cell
def _(States, comp, pd, stress_pipeline):
    selected_stress_node = stress_pipeline.blocks["INST-007"] / "scenario"
    comp.compute(selected_stress_node)
    stress_block_states = pd.Series(
        {key: comp.state(path / "scenario").name for key, path in stress_pipeline.blocks.items()},
        name="scenario state",
    )
    assert stress_block_states["INST-007"] == States.UPTODATE.name
    stress_block_states.value_counts().rename_axis("state").to_frame("nodes")
    return


@app.cell
def _(comp):
    # single computed row
    comp.v["stress/instruments/INST-001/scenario"]
    return


@app.cell
def _(comp):
    # concatentated
    comp.compute("stress_report")
    stress_report = comp.v.stress_report
    stress_report
    return


@app.cell
def _(comp):
    comp.draw("")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Selective recalculation after an update

    A shallow copy is useful for experimenting without changing the original
    computed graph. Updating the shared market dataframe invalidates the main
    portfolio descendants, but asking for one instrument report recalculates
    only that required branch. The aggregate stays stale until requested.
    """)
    return


@app.cell
def _(States, blocks, comp, market_data, pd):
    updated_comp = comp.copy()
    updated_market_data = market_data.copy()
    updated_market_data.loc["INST-012", "price"] += 8.0
    updated_comp.insert("market_data", updated_market_data)

    changed_report_node = blocks["INST-012"] / "report"
    updated_comp.compute(changed_report_node)

    recalculation_states = pd.Series(
        {key: updated_comp.state(path / "report").name for key, path in blocks.items()},
        name="report state",
    )
    assert updated_comp.state(changed_report_node) == States.UPTODATE
    assert updated_comp.state("portfolio_report") == States.STALE
    recalculation_states.value_counts().rename_axis("state").to_frame("nodes")
    return changed_report_node, updated_comp


@app.cell
def _(changed_report_node, updated_comp):
    updated_comp.value(changed_report_node)
    return


@app.cell
def _(portfolio_totals, updated_comp):
    updated_comp.compute("portfolio_totals")
    updated_totals = updated_comp.v.portfolio_totals
    totals_comparison = (
        portfolio_totals.rename("before")
        .to_frame()
        .join(updated_totals.rename("after"))
        .assign(change=lambda frame: frame["after"] - frame["before"])
    )
    totals_comparison
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    The utilities create normal computation nodes rather than introducing a
    separate execution model. Consequently, large repeated graphs retain
    Loman's core properties:

    - keyed blocks remain inspectable and visualizable;
    - fan-out transforms and fan-in combiners execute lazily;
    - shared ancestors are calculated once;
    - unused branches stay uninitialized;
    - target computation limits work to required ancestors;
    - independent branches use the configured executor.

    For JSON persistence, rebuild utility-generated graphs from their Python
    definition before recalculation; the current serializer is best used for
    snapshots of their computed values.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
