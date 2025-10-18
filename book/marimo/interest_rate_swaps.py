"""Interest Rate Swaps Pricing with Loman."""

import marimo

__generated_with = "0.15.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Example: Using Loman to price Interest Rate Swaps

    This example demonstrates calibrating interest rate curves to market
    prices and using them to price portfolios of swaps using the Loman
    framework.
    """
    )
    return


@app.cell
def _():
    import numpy as np
    from scipy.integrate import quad

    import loman

    return loman, np, quad


@app.cell
def _(mo):
    mo.md(r"""## Curve Classes""")
    return


@app.cell
def _(np, quad):
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

        def plot(self):
            # This will be usable once swap_rate is defined
            pass

    return (BaseIRCurve,)


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


@app.cell
def _(mo):
    mo.md(r"""## Calibrating a LIBOR curve""")
    return


@app.cell
def _(FlatIRCurve, loman, np, swap_rate):
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
    comp.draw()
    return (comp,)


@app.cell
def _(comp):
    comp.compute_all()
    comp.value("usd_libor_curve").plot()
    return


@app.cell
def _(mo):
    mo.md(r"""## Valuing a Portfolio of Interest Rate Swaps""")
    return


@app.cell
def _(comp, swap_leg_pvs):
    from collections import namedtuple

    Swap = namedtuple("Swap", ["notional", "start", "end", "rate", "freq"])

    def swap_pv(swap, projection_curve, discount_curve):
        fixed_pv, float_pv = swap_leg_pvs(swap.start, swap.end, swap.freq, projection_curve, discount_curve)
        return swap.notional * (float_pv - swap.rate * fixed_pv)

    comp.add_node("portfolio", value=[Swap(10000000, 5, 10, 0.025, 0.25), Swap(-5000000, 2.5, 12.5, 0.02, 0.25)])
    return (swap_pv,)


@app.cell
def _(comp, swap_pv):
    comp.add_node(
        "portfolio_val",
        lambda portfolio, usd_libor_curve: [swap_pv(swap, usd_libor_curve, usd_libor_curve) for swap in portfolio],
    )
    comp.compute_all()
    print("Portfolio values:", comp.value("portfolio_val"))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Conclusion

    This demonstrates the basic functionality of using Loman for interest
    rate derivatives pricing. The full notebook would include calibration,
    dual curve bootstrapping, and risk calculations.
    """
    )
    return


if __name__ == "__main__":
    app.run()
