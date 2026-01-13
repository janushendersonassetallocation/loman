"""Example: Using Loman to price Interest Rate Swaps.

This notebook demonstrates calibrating interest rate curves to market prices
and using them to price portfolios of swaps.
"""
# ruff: noqa: E501, N806

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.17.6",
#     "numpy",
#     "scipy",
#     "matplotlib",
#     "loman",
# ]
#
# [tool.uv.sources]
# loman = { path = "../../..", editable = true }
# ///

import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import warnings

    import marimo as mo

    # Suppress RuntimeWarnings for division operations globally
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Example: Using Loman to price Interest Rate Swaps

    In this example, we'll look at calibrating interest rate curves to market prices, and then using them to price portfolios of swaps.

    ## Curve Classes

    Because the focus will be on using Loman, we'll adopt deliberately simplifying assumptions; time is a float, day count and business day conventions are ignored, quarters are exactly 0.25 years long and so on.

    Our interest rate curves can be used for two things: discounting and projecting rates. To do this, we define a continuously-compounded forward rate $r(t)$, so that the discount rate from a payment at time $t$ to a time $s$ is

    $$df(s,t) = \exp\left[-\int_s^t r(\tau) d\tau\right]$$

    and zero/FRA rates are defined by

    $$df(s,t) = \frac{1}{1+\text{FRA}(s,t)(t-s)}.$$

    Our BaseIRCurve class leaves the definition of $r$ blank, but otherwise fleshes out the methods we'll need, including methods to PV a set of cashflows, and also to plot the continuously-compounded forward rate, 3M FRA rates, and spot swap rates (once we define swap_rate, further below).
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import quad

    class BaseIRCurve:
        def r_quad(self, s, t):
            b = np.broadcast(s, t)
            out = np.empty(b.shape)
            out.flat = [quad(self.r, s0, t0)[0] for s0, t0 in b]
            return out

        def r(self, t):
            pass

        def df(self, s, t):
            return np.exp(-self.r_quad(s, t))

        def fra(self, s, t):
            return (1 / self.df(s, t) - 1.0) / (t - s)

        def pv(self, amts, ts, s=0):
            dfs = self.df(s, ts)
            return np.sum(amts * dfs)

        def plot_basic(self):
            """Plot r and 3M forward rates (swap rate plotting done separately to avoid cycles)."""
            ts = np.linspace(0.0, 29.75, 360)
            plt.plot(ts, self.r(ts), label="r")
            ts = np.linspace(0.0, 29.75, 120)
            plt.plot(ts, self.fra(ts, ts + 0.25), label="3M fwd")
            plt.legend()

    return BaseIRCurve, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The FlatIRCurve class fills out the definitions of r and r_quad. $r(t)$ is piece-wise flat, and hence its integral is piecewise linear. Together with the methods defined by BaseIRCurve we will now have a basic but functional interest rate curve.
    """)
    return


@app.cell
def _(BaseIRCurve, np):
    class FlatIRCurve(BaseIRCurve):
        def __init__(self, ts, rates):
            self.ts = ts
            self.rates = rates
            self.ts0 = np.zeros(len(self.ts) + 1)
            self.ts0[1:] = self.ts
            self.rquads = np.zeros_like(self.ts0)
            self.rquads[1:] = np.cumsum(np.diff(self.ts0) * self.rates)

        def r(self, t):
            idx = np.minimum(np.searchsorted(self.ts, t, "right"), len(self.ts) - 1)
            return self.rates[idx]

        def r_quad(self, s, t):
            return np.interp(t, self.ts0, self.rquads) - np.interp(s, self.ts0, self.rquads)

    return (FlatIRCurve,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, we'll need to set up some functions to calculate quarterly FRAs and discount them to calculate PVs of the fixed and floating legs of swaps, as well as the swap rate.
    """)
    return


@app.cell
def _(np):
    def sched(a, b, p):
        n = int((b - a) / p)
        if b - n * p > a + p / 2.0:
            n = n + 1
        ts = np.linspace(b - n * p, b, n + 1)
        ts[0] = a
        return ts

    def swap_leg_pvs(a, b, p, projection_curve, discount_curve):
        ts = sched(a, b, p)
        pers = np.diff(ts)
        fixed_pv = discount_curve.pv(pers, ts[1:])
        fras = projection_curve.fra(ts[:-1], ts[1:])
        float_pv = discount_curve.pv(pers * fras, ts[1:])
        return fixed_pv, float_pv

    def swap_rate(a, b, p, projection_curve, discount_curve):
        fixed_pv, float_pv = swap_leg_pvs(a, b, p, projection_curve, discount_curve)
        return float_pv / fixed_pv

    return swap_leg_pvs, swap_rate


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Calibrating a LIBOR curve

    Up until 2008, it was standard to use the LIBOR curve for both projection of LIBOR swap cashflows *and* for discounting those cashflows. We set up a Loman computation with inputs **usd_libor_ts**, a set of swap maturities, and **usd_libor_c_rate** a set of continuous compounding rates for each period. From those, we create a curve object **usd_libor_curve**, and then calculate swap rates using that curve for both projection and discounting, in **usd_libor_swap_rates**.
    """)
    return


@app.cell
def _(FlatIRCurve, np, swap_rate):
    import loman

    comp = loman.Computation()
    comp.add_node("usd_libor_ts", value=np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]))
    comp.add_node("usd_libor_c_rates", value=0.03 * np.ones(9))
    comp.add_node(
        "usd_libor_curve", lambda usd_libor_ts, usd_libor_c_rates: FlatIRCurve(usd_libor_ts, usd_libor_c_rates)
    )
    comp.add_node(
        "usd_libor_swap_rates",
        lambda usd_libor_curve, usd_libor_ts: np.vectorize(swap_rate)(
            0, usd_libor_ts, 0.25, usd_libor_curve, usd_libor_curve
        ),
    )
    comp.draw(graph_attr={"size": "12"})
    return (comp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we calculate all the nodes in Loman, we can plot the curve. Because we set all the continuously compounding rates to 3%, it's not a very interesting curve yet.
    """)
    return


@app.cell
def _(comp, np, plt, swap_rate):
    comp.compute_all()
    curve = comp.value("usd_libor_curve")
    curve.plot_basic()
    # Add swap rate to plot
    ts_plot = np.linspace(0.0, 29.75, 120)
    plt.plot(ts_plot, np.vectorize(swap_rate)(0, ts_plot, 0.25, curve, curve), label="Swap Rate")
    plt.legend()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, to help calibrate our curve, we add a node that is market swap rates (**usd_libor_mkt_swap_rates**), and calculate the difference between the rates our curve is producing, and the market rates we are trying to fit to, in **usd_libor_fitting_error**.
    """)
    return


@app.cell
def _(comp, np):
    comp.add_node(
        "usd_libor_mkt_swap_rates",
        value=np.array([0.01364, 0.01593, 0.01776, 0.02023, 0.02181, 0.02343, 0.02499, 0.02566, 0.02593]),
    )
    comp.add_node(
        "usd_libor_fitting_error",
        lambda usd_libor_swap_rates, usd_libor_mkt_swap_rates: usd_libor_swap_rates - usd_libor_mkt_swap_rates,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To calibrate our curve, we use a minimizer from scipy. Our objective function is to insert a trial set of continuously-compounded rates into the input **usd_libor_c_rates**, calculate the error vector **usd_libor_fitting_error** using Loman, and then return the sum of the squares (scaled appropriately for the solver). Our initial guess is taken from the current set of rates in **usd_libor_c_rates**.
    """)
    return


@app.cell
def _(comp, np):
    from scipy.optimize import minimize

    def error(xs):
        comp.insert("usd_libor_c_rates", xs)
        comp.compute("usd_libor_fitting_error")
        return 10000.0 * np.sum(comp.value("usd_libor_fitting_error") ** 2)

    res = minimize(error, comp.value("usd_libor_c_rates"))
    return minimize, res


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The solver indicates that it rans successfully with 275 evaluations, so we insert its solution set of input rates into the computation (solvers aren't required that their last evaluation be the solution value), re-compute everything, and plot the resulting curve. We also show the market swap rates so we can see our calibration was successful.
    """)
    return


@app.cell
def _(comp, np, plt, res, swap_rate):
    comp.insert("usd_libor_c_rates", res.x)
    comp.compute_all()
    curve_calibrated = comp.value("usd_libor_curve")
    curve_calibrated.plot_basic()
    # Add swap rate to plot
    ts_calib = np.linspace(0.0, 29.75, 120)
    plt.plot(
        ts_calib, np.vectorize(swap_rate)(0, ts_calib, 0.25, curve_calibrated, curve_calibrated), label="Swap Rate"
    )
    plt.scatter(comp.value("usd_libor_ts"), comp.value("usd_libor_mkt_swap_rates"), label="Mkt Swap Rates")
    plt.legend()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Valuing a Portfolio of Interest Rate Swaps

    Now that we have a valid interest rate curve, we can use it to value interest rate swaps. For now, our swaps will all be valued with using the same USD LIBOR curve.

    We define a swap as a collection of named parameters, create a function to value them, and add a node **portfolio** to our computation, which is a set of swaps
    """)
    return


@app.cell
def _(comp, swap_leg_pvs):
    from collections import namedtuple

    _Swap = namedtuple("Swap", ["notional", "start", "end", "rate", "freq"])

    def swap_pv(swap, projection_curve, discount_curve):
        fixed_pv, float_pv = swap_leg_pvs(swap.start, swap.end, swap.freq, projection_curve, discount_curve)
        return swap.notional * (float_pv - swap.rate * fixed_pv)

    comp.add_node("portfolio", value=[_Swap(10000000, 5, 10, 0.025, 0.25), _Swap(-5000000, 2.5, 12.5, 0.02, 0.25)])
    return namedtuple, swap_pv


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To value our portfolio, we simply apply our valuation function to each position in our portfolio, and Loman gives us an array back, with the value of each position
    """)
    return


@app.cell
def _(comp, swap_pv):
    comp.add_node(
        "portfolio_val",
        lambda portfolio, usd_libor_curve: [swap_pv(swap, usd_libor_curve, usd_libor_curve) for swap in portfolio],
    )
    comp.draw(graph_attr={"size": "12"})
    return


@app.cell
def _(comp):
    comp.compute_all()
    comp.value("portfolio_val")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dual Bootstrap Curves

    Since 2008, the spread between LIBOR and OIS has been material, and it is standard practice to calibrate curves to LIBOR swaps and LIBOR-OIS basis swaps, say. To start, we'll need some inputs, a curve, and the market rates.
    """)
    return


@app.cell
def _(FlatIRCurve, comp, np):
    comp.add_node("usd_ois_ts", value=np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]))
    comp.add_node("usd_ois_c_rates", value=0.03 * np.ones(9))
    comp.add_node("usd_ois_curve", lambda usd_ois_ts, usd_ois_c_rates: FlatIRCurve(usd_ois_ts, usd_ois_c_rates))
    comp.add_node(
        "usd_libor_ois_mkt_spreads", value=np.array([24.2, 26, 27.2, 29, 30.9, 33.4, 36.4, 38, 39.8]) / 10000.0
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we need a function to calculate LIBOR-OIS spreads from our two curves using

    $$PV_\text{LIBOR float leg} = PV_\text{OIS float leg} + s \times PV_\text{OIS 1bp fixed leg}$$
    """)
    return


@app.cell
def _(swap_leg_pvs):
    def swap_spread(a, b, p, projection_curve1, projection_curve2, discount_curve):
        fixed_pv1, float_pv1 = swap_leg_pvs(a, b, p, projection_curve1, discount_curve)
        fixed_pv2, float_pv2 = swap_leg_pvs(a, b, p, projection_curve2, discount_curve)
        return (float_pv1 - float_pv2) / fixed_pv2

    return (swap_spread,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We use that function to calculate LIBOR-OIS spreads using our two curves, as well as another fitting error vector that we will use in calibration.
    """)
    return


@app.cell
def _(comp, np, swap_spread):
    comp.add_node(
        "usd_libor_ois_spreads",
        lambda usd_libor_curve, usd_ois_curve, usd_ois_ts: np.vectorize(swap_spread)(
            0, usd_ois_ts, 0.25, usd_libor_curve, usd_ois_curve, usd_ois_curve
        ),
    )

    comp.add_node(
        "usd_libor_ois_fitting_error",
        lambda usd_libor_ois_spreads, usd_libor_ois_mkt_spreads: usd_libor_ois_spreads - usd_libor_ois_mkt_spreads,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We should also update our LIBOR swap rate calculation to use our OIS curve for discounting:
    """)
    return


@app.cell
def _(comp, np, swap_rate):
    comp.add_node(
        "usd_libor_swap_rates",
        lambda usd_libor_curve, usd_ois_curve, usd_libor_ts: np.vectorize(swap_rate)(
            0, usd_libor_ts, 0.25, usd_libor_curve, usd_ois_curve
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, for our calibration, it would be more convenient if we could just insert one vector, containing both LIBOR and OIS inputs, and have those feed through into **usd_libor_c_rates** and **usd_ois_c_rates**. Thanks to Loman, we can do this by giving them a common parent node. If we later want to insert values directly into **usd_libor_c_rates** and **usd_ois_c_rates**, then we can do that too, even though they are calculation nodes. We also define a node  **usd_fitting_error** to collect the fitting error vectors.
    """)
    return


@app.cell
def _(comp, np):
    comp.add_node("usd_c_rates", value=0.03 * np.ones(18))
    comp.add_node("usd_libor_c_rates", lambda usd_c_rates: usd_c_rates[0:9])
    comp.add_node("usd_ois_c_rates", lambda usd_c_rates: usd_c_rates[9:18])
    comp.add_node(
        "usd_fitting_error",
        lambda usd_libor_fitting_error, usd_libor_ois_fitting_error: np.concatenate(
            [usd_libor_fitting_error, usd_libor_ois_fitting_error]
        ),
    )
    comp.draw(graph_attr={"size": "12"})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Complicated.

    But it's ok. Loman can take care of this.

    We can just apply the same method as before. Having the solver insert sets of inputs, and iterate until it has minimized the error. This time, the set of inputs that it is adjusting flow into the construction of two separate curves, **usd_ois_curve**, and **usd_libor_curve**.
    """)
    return


@app.cell
def _(comp, minimize, np):
    def error_dual(xs):
        comp.insert("usd_c_rates", xs)
        comp.compute("usd_fitting_error")
        return 10000.0 * np.sum(comp.value("usd_fitting_error") ** 2)

    res_dual = minimize(error_dual, comp.value("usd_c_rates"))
    res_dual.success, res_dual.nfev
    return (res_dual,)


@app.cell
def _(comp, res_dual):
    comp.insert("usd_c_rates", res_dual.x)
    comp.compute_all()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, we can plot LIBOR-OIS spreads from our dual boot-strapped curve, as well as LIBOR and OIS swap rates, to check the calibration.
    """)
    return


@app.cell
def _(comp, np, plt, swap_spread):
    ts = np.linspace(0.0, 29.75, 360)
    spreads = np.vectorize(swap_spread)(
        0.0, ts, 0.25, comp.value("usd_libor_curve"), comp.value("usd_ois_curve"), comp.value("usd_ois_curve")
    )
    plt.plot(ts, spreads, label="Model LIBOR-OIS Spreads")
    plt.scatter(comp.value("usd_ois_ts"), comp.value("usd_libor_ois_mkt_spreads"), label="Market LIBOR-OIS Spreads")
    plt.legend()
    plt.gca()
    return


@app.cell
def _(comp, np, plt, swap_rate):
    libor_curve_dual = comp.value("usd_libor_curve")
    ois_curve_dual = comp.value("usd_ois_curve")
    libor_curve_dual.plot_basic()
    # Add swap rate to plot
    ts_dual = np.linspace(0.0, 29.75, 120)
    plt.plot(ts_dual, np.vectorize(swap_rate)(0, ts_dual, 0.25, libor_curve_dual, ois_curve_dual), label="Swap Rate")
    plt.scatter(comp.value("usd_libor_ts"), comp.value("usd_libor_mkt_swap_rates"), label="Mkt Swap Rates")
    plt.legend()
    plt.gca()
    return


@app.cell
def _(comp, np, plt, swap_rate):
    ois_curve_plot = comp.value("usd_ois_curve")
    ois_curve_plot.plot_basic()
    # Add swap rate to plot for OIS
    ts_ois = np.linspace(0.0, 29.75, 120)
    plt.plot(ts_ois, np.vectorize(swap_rate)(0, ts_ois, 0.25, ois_curve_plot, ois_curve_plot), label="Swap Rate")
    plt.legend()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Revisiting Portfolio Valuation

    Now that we have two curves, and our swaps could potentially be projected and discounted with different curves, it's appropriate to introduce a new node, **curveset** and have the valuation of our portfolio depend on that curveset, rather than pointing directly at specific curves. When we create **curveset**, we can also put in a check that our fit was good.

    We update the design of our Swap instrument, and recreate the portfolio with the additional information.

    By design, Loman lets us redefine the function that calculates portfolio_val and have it depend on **curveset** rather than **usd_libor_curve**, and the Swap instruments themselves direct curves to take from **curveset** to perform valuation.
    """)
    return


@app.cell
def _(comp, namedtuple, np, swap_pv):
    Swap = namedtuple("Swap", ["notional", "start", "end", "rate", "freq", "projection_curve", "discount_curve"])

    def create_curveset(usd_libor_curve, usd_ois_curve, usd_fitting_error):
        if np.max(np.abs(usd_fitting_error) > 0.00001):
            raise Exception("Fitting error > 0.1bps")
        return {"USD-LIBOR-3M": usd_libor_curve, "USD-OIS": usd_ois_curve}

    comp.add_node("curveset", create_curveset)

    comp.insert(
        "portfolio",
        [
            Swap(10000000, 5, 10, 0.025, 0.25, "USD-LIBOR-3M", "USD-OIS"),
            Swap(-5000000, 2.5, 12.5, 0.02, 0.25, "USD-LIBOR-3M", "USD-OIS"),
        ],
    )

    comp.add_node(
        "portfolio_val",
        lambda portfolio, curveset: [
            swap_pv(swap, curveset[swap.projection_curve], curveset[swap.discount_curve]) for swap in portfolio
        ],
    )
    comp.draw(graph_attr={"size": "12"})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, we can see that our valuation is close but materially different to the previous LIBOR-only curve's result (86,395 and -274,086), exactly as we expect.
    """)
    return


@app.cell
def _(comp):
    comp.compute("portfolio_val")
    comp.value("portfolio_val")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Risks

    There are a few ways to calculate risk for a book of interest rate derivatives:

    * Spot risks directly. We can directly perturb the inputs **usd_libor_mkt_swap_rates** or **usd_libor_ois_mkt_spreads** one element at a time, recalibrating, and repricing the portfolio each time.
    * Spot risks, using Jacobians. We can perturb the inputs **usd_libor_c_rates** and **usd_ois_c_rates**, using the changes in **usd_libor_swap_rates** and **usd_libor_ois_spreads** to calculate a Jacobian. Applying the inverse of this Jacobian matrix on the matrix of portfolio valuation changes gives us the sensitivities to market instruments as above, without the need to recalibrate.
    * Forward risk. We can directly perturb the curve using $r'(t; a,b) = r(t) + \delta I_{a \le t < b}$ to create a new curve where the continuously-compounded rate is elevated in the region $[a,b)$. This will cause the forward swap rate to change by some amount, and by scaling $\delta$, we can make that amount 1bp (or we can scale the PV changes to the same effect), so we can get the risk for our portfolio to a set of forwards, potentially with much higher granularity than our initial set of calibration instruments. Additionally this method is more numerically stable when using non-local interpolations for curve construction.

    For this exposition, we will go with the latter method, and to keep things moving along, we will only produce risks to the LIBOR curve.

    To accommodate this change, we'll create a new node called **usd_libor_curve_perturbed** which will be fed into **curveset** for valuation, and will itself be created from **usd_libor_curve** and a control input called **usd_libor_curve_perturbation**.

    First things first, we need to define how to create a perturbed curve, using the recipe above. Note that once we've defined r and r_quad, everything else will flow through
    """)
    return


@app.cell
def _(BaseIRCurve, FlatIRCurve, np):
    class SumCurve(BaseIRCurve):
        def __init__(self, *curves):
            self.curves = curves

        def r_quad(self, s, t):
            return sum(curve.r_quad(s, t) for curve in self.curves)

        def r(self, t):
            return sum(curve.r(t) for curve in self.curves)

    def create_perturbed_curve(curve, start, end, amount):
        pert_curv = FlatIRCurve(np.array([start, end, np.max(curve.ts)]), np.array([0.0, amount, 0.0]))
        return SumCurve(curve, pert_curv)

    return (create_perturbed_curve,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To test this is working, we can see what happens when we bump the continuously-compounded rate by 5% between years 5 and 10 for example.
    """)
    return


@app.cell
def _(comp, create_perturbed_curve, np, plt, swap_rate):
    perturbed_test = create_perturbed_curve(comp.value("usd_libor_curve"), 5.0, 10.0, 0.05)
    perturbed_test.plot_basic()
    # Add swap rate to plot
    ts_pert = np.linspace(0.0, 29.75, 120)
    plt.plot(ts_pert, np.vectorize(swap_rate)(0, ts_pert, 0.25, perturbed_test, perturbed_test), label="Swap Rate")
    plt.legend()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That looks as we expect, so we can go ahead and add our new nodes:
    """)
    return


@app.cell
def _(comp, create_perturbed_curve, np):
    comp.add_node("usd_libor_curve_perturbation", value=(0, 1, 0.0))
    comp.add_node(
        "usd_libor_curve_perturbed",
        lambda usd_libor_curve, usd_libor_curve_perturbation: create_perturbed_curve(
            usd_libor_curve, *usd_libor_curve_perturbation
        ),
    )

    def create_curveset_perturbed(usd_libor_curve_perturbed, usd_ois_curve, usd_fitting_error):
        if np.max(np.abs(usd_fitting_error) > 0.00001):
            raise Exception("Fitting error > 0.1bps")
        return {"USD-LIBOR-3M": usd_libor_curve_perturbed, "USD-OIS": usd_ois_curve}

    comp.add_node("curveset", create_curveset_perturbed)

    comp.draw(graph_attr={"size": "12"})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can iterate over each year, and see the sensitivity of the portfolio to a change of 1bp in the forward swap rate for that year:
    """)
    return


@app.cell
def _(comp, np, plt, swap_rate):
    comp.compute("portfolio_val")
    comp.insert("usd_libor_curve_perturbation", value=(0, 1, 0.0))
    base_value = np.array(comp.value("portfolio_val"))
    pert_values = np.empty((30, base_value.shape[0]))
    delta_value = np.empty((30, base_value.shape[0]))
    ts_risk = np.arange(30)
    for i in ts_risk:
        comp.insert("usd_libor_curve_perturbation", value=(i, i + 1, 0.0001))
        comp.compute("portfolio_val")
        pert_values[i] = comp.value("portfolio_val")
        swap_rate_base = swap_rate(i, i + 1, 0.25, comp.value("usd_libor_curve"), comp.value("usd_ois_curve"))
        swap_rate_pert = swap_rate(i, i + 1, 0.25, comp.value("usd_libor_curve_perturbed"), comp.value("usd_ois_curve"))
        swap_rate_delta = swap_rate_pert - swap_rate_base
        delta_value[i] = 0.0001 / swap_rate_delta * (pert_values[i] - base_value)

    plt.bar(ts_risk, delta_value.sum(axis=1), np.diff(np.concatenate([[0], ts_risk])))
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    This concludes our interest example. We have shown that Loman makes it easy to simultaneously calibrate interest rate curves to market quotes, to value portfolios of interest rate derivatives, and to produce bucketed sensitivities.

    The portfolio valuation also naturally extends to cover other interest rate derivatives.

    The calibration approach - an external routine driving Loman to produce calibrated inputs for the computation - easily and naturally extends to calibrating to a global set of interest rate markets, with curveset drawing from many separate independent and interdependent calibrations. It can also extend to other markets, such as FX derivatives, listed options, credit derivatives.

    Loman's ability to serialize computations allows the calibration to happen once, centrally, and be broadcast to user desktops for firm-wide consistent live valuation.

    In short, we have only scratched the surface of what is possible with Loman with this example, and we look forward to further exploration in the future.
    """)
    return


if __name__ == "__main__":
    app.run()
