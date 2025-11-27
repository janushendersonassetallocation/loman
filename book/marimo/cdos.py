"""Example: Using Loman to value CDOs.

This notebook demonstrates the Andersen-Sidenius-Basu semi-analytic method
for CDO tranche valuation with loss distribution calculations.
"""
# ruff: noqa: E501, N806

import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Example - Using Loman to value CDOs

    In 2003, Andersen, Sidenius and Basu published their semi-analytic method for CDO tranche valuation in the paper ["All your hedges in one basket"](http://www.ressources-actuarielles.net/EXT/ISFA/1226.nsf/0/bf571acf7dbca8cbc12577b4001e3664/$FILE/JP%20Laurent%20Gr%C3%A9gory.pdf). For homogenous baskets, this gave sub-second valuations, and stable sensitivities. In this example, we go through calculating the value of a CDO tranche using this method.

    First, a quick recap of the core of the method. Given a normally distributed random variable $M$, representing the state of the broad economy, and normally distributed random variable $Z_i$, representing the idiosyncratic performance of a set of indexed obligors $i=1, \ldots, N$, with $M$ and each $Z_i$ independent, we construct $X_i$:

    $$ X_i = \sqrt{\rho} M + \sqrt{1-\rho} Z_i .$$

    We then have that each $X_i$ is normally distributed, and $\text{Corr}(X_i, X_j)=\rho$ where $i \ne j$.

    In a simulation context we would simulate $M$ and each $Z_i$, and then take obligor $i$ as defaulted if $X_i$ fell below some threshold. However, conditional on $M$, the $X_i$s are independent, so given the conditional probability of default, calculating the loss distribution directly is a simple convolution and we can directly calculate the conditional probability by
    $$P_\text{def}(i | M)
    = P\left( X_i < \Phi^{-1}(P_\text{def}(i)) | M \right)
    = \Phi\left( \frac{\Phi^{-1}(P_\text{def}(i)) - \sqrt{\rho}M}{\sqrt{1-\rho}} \right).$$

    We now start our implementation by importing the usual modules, and creating an empty computation. Our goal will be to calculate the PVs of the default leg and the coupon leg (and hence the fair spread).
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import scipy as sp
    import scipy.stats

    import loman

    norm = sp.stats.norm
    import matplotlib.pyplot as plt

    comp = loman.Computation()
    return comp, loman, mo, norm, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first thing to do is to work out the set of coupon times, so we create a node "ts" for this purpose. In this example, we simplify, by assuming that times can be represented as a number of years, and we ignore day count conventions, IMM payment dates and other details. To do this, we need the maturity and coupon frequency of the CDO. We then count back from maturity in steps of 1/freq. We remove the first coupon if the first coupon period would be too short, and we also add zero, although a coupon will not be paid at this time.
    """)
    return


@app.cell
def _(comp, loman, np):
    comp.add_node("maturity", value=5.0)
    comp.add_node("freq", value=4)

    @loman.node(comp)
    def ts(maturity, freq, long_stub=True):
        per = 1.0 / freq
        ts = list(np.r_[maturity:0:-per])
        if (long_stub and ts[-1] < per - 0.0001) or ts[-1] < 0.0001:
            ts = ts[:-1]
        ts.append(0.0)
        ts.reverse()
        return np.array(ts)

    comp.compute_all()
    comp.draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And we can see that the vector of times is as we require:
    """)
    return


@app.cell
def _(comp):
    comp.v.ts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, as we will be calculating PVs, we need the discount factor at each of these times. In this simple example, we take a fixed interest rate, applied continuously, to calculate discount factors. We might also choose to use a discount curve calibrated in the Loman Interest Rate Curve calibration example.
    """)
    return


@app.cell
def _(comp, loman, np):
    comp.add_node("r", value=0.04)

    @loman.node(comp)
    def dfs(ts, r):
        return np.exp(-r * ts)

    comp.compute_all()
    comp.draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also plot our wildly uninteresting discount factors to check they are reasonable:
    """)
    return


@app.cell
def _(comp, plt):
    plt.plot(comp.v.ts, comp.v.dfs)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we will need default probabilities for the obligors. In a more industrial-strength implementation, these would be calibrated to CDS and CDS index markets, but for our purposes, it is sufficient to use a simple approximation to determine a hazard rate that we will apply to all obligors.
    """)
    return


@app.cell
def _(comp, loman, np):
    comp.add_node("spread", value=0.0060)
    comp.add_node("rr", value=0.40)

    @loman.node(comp)
    def hazard_rate(spread, rr):
        return spread / (1.0 - rr)

    @loman.node(comp)
    def p_defs(hazard_rate, ts):
        return 1 - np.exp(-hazard_rate * ts)

    comp.compute_all()
    comp.draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And we plot the default probability for each obligor as a function of time, to check for reasonableness:
    """)
    return


@app.cell
def _(comp, plt):
    plt.plot(comp.v.ts, comp.v.p_defs)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this model, we will be calculating the probability of a set of evenly spaced discrete levels of loss. Here, we create a node losses where each entry is the amount of loss at the CDO level associated with each state. In this example, each obligor is $\frac{1}{125}$th of the portfolio of \$10MM, with a recovery rate of 40%, so each loss is $48k. We create a column vector for convenience, as our loss distributions will have a row for each loss level, and a column for each coupon time.
    """)
    return


@app.cell
def _(comp, loman, np):
    comp.add_node("notional", value=10000000)
    comp.add_node("n_obligors", value=125)
    comp.add_node("n_losses", lambda n_obligors: n_obligors)

    @loman.node(comp)
    def lgd(rr):
        return 1.0 - rr

    @loman.node(comp)
    def losses(notional, n_losses, n_obligors, lgd):
        loss_amount = notional * 1.0 / n_obligors * lgd
        return (np.arange(n_losses) * loss_amount)[:, np.newaxis]

    comp.compute_all()
    comp.v.losses.T
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will also need the amount of the tranche remaining in each state, with the attachment and detachment points provided as a percentage of portfolio notional, again as a column vector:
    """)
    return


@app.cell
def _(comp, loman, np):
    comp.add_node("ap", value=0.03)
    comp.add_node("dp", value=0.06)

    @loman.node(comp)
    def tranche_widths(notional, losses, ap, dp):
        return np.maximum(losses, dp * notional) - np.maximum(losses, ap * notional)

    comp.compute_all()
    comp.v.tranche_widths.T
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ok, now we get to calculating leg PVs conditional on $M$. We make a node `M`, which is an input which we will vary exogenously. The first thing to calculate is the condition default probabilities of each obligor. However, in this example, every obligor has the same default probability and hence the same conditional default probability:
    """)
    return


@app.cell
def _(comp, loman, norm, np):
    comp.add_node("corr", value=0.35)
    comp.add_node("M", value=0)

    @loman.node(comp)
    def p_def_conds(M, p_defs, corr):
        return norm.cdf((norm.ppf(p_defs) - np.sqrt(corr) * M) / np.sqrt(1 - corr))

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we calculate the loss distribution, in a node `ps`. Rather than the recursion given by Andersen et al, we use Parcell's [Loss unit interpolation](http://www.edparcell.com/lossdistint.pdf) method, which copes better with fractional losses. The node `ps` is a vector with dimensions $\text{number of loss levels} \times \text{number of coupon times}$. We first initialize it so that the zero loss level has probability 1 at each coupon time, and every other level has probability 0. We then update the loss distribution in a loop, adding each obligor in turn, and updating the loss distribution accordingly:
    """)
    return


@app.cell
def _(comp, loman, np):
    def init_loss_distribution(n_losses, n_ts):
        ps = np.zeros((n_losses, n_ts))
        ps[0, :] = 1.0
        return ps

    def update_loss_distribution(ps, p_default, loss):
        """Update loss distribution by adding an obligor.

        ps: float[n_losses, n_ts]
        p_default: float | float[n_ts]
        loss: float.
        """
        ps1 = ps.copy()
        loss_lower = int(loss)
        loss_upper = loss_lower + 1
        loss_frac = loss - loss_lower
        ps *= 1 - p_default
        ps[loss_lower:] += p_default * (1 - loss_frac) * ps1[:-loss_lower]
        ps[loss_upper:] += p_default * loss_frac * ps1[:-loss_upper]

    @loman.node(comp)
    def ps(n_losses, ts, n_obligors, p_def_conds):
        ps_dist = init_loss_distribution(n_losses, len(ts))
        for i in range(n_obligors):
            update_loss_distribution(ps_dist, p_def_conds, 1)
        return ps_dist

    comp.compute_all()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Having calculated the conditional loss distribution, we can see how it evolves over time:
    """)
    return


@app.cell
def _(comp, plt):
    plt.plot(comp.v.losses[:, 0], comp.v.ps[:, 4], label=f"t={comp.v.ts[4]}")
    plt.plot(comp.v.losses[:, 0], comp.v.ps[:, 8], label=f"t={comp.v.ts[8]}")
    plt.plot(comp.v.losses[:, 0], comp.v.ps[:, 20], label=f"t={comp.v.ts[20]}")
    plt.xlim(0, 600000)
    plt.legend()
    plt.gca()
    return


@app.cell
def _(comp):
    comp.draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now have the probability of being in each loss level at each time, and the associated tranche width in that loss state, so we can calculate and plot the conditional expected tranche width over time:
    """)
    return


@app.cell
def _(comp, loman, np, plt):
    @loman.node(comp)
    def expected_tranche_widths(ps, tranche_widths):
        return np.sum(ps * tranche_widths, axis=0)

    comp.compute_all()
    plt.plot(comp.v.ts, comp.v.expected_tranche_widths)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The conditional expected default leg is calculated as the change in conditional expected tranche width with discounting applied:
    """)
    return


@app.cell
def _(comp, loman, np):
    @loman.node(comp)
    def default_amounts(expected_tranche_widths):
        return -np.diff(expected_tranche_widths)

    @loman.node(comp)
    def default_leg_pv(default_amounts, dfs):
        return np.sum(default_amounts * dfs[1:])

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We make the approximation that coupon is paid on a notional that is the average of tranche width at the start and end of the coupon peried. Discounting is applied, and we calculate the PV of paying a 100% running coupon (!) for convenience calculating the fair spread later, and also the coupon paying a fixed level coupon, which is an input
    """)
    return


@app.cell
def _(comp, loman, np):
    @loman.node(comp)
    def average_tranche_widths(expected_tranche_widths):
        return (expected_tranche_widths[:-1] + expected_tranche_widths[1:]) / 2.0

    @loman.node(comp)
    def coupon_leg_1_pv(average_tranche_widths, dfs, ts):
        return np.sum(average_tranche_widths * dfs[1:] * np.diff(ts))

    comp.add_node("coupon", value=0.05)

    @loman.node(comp)
    def coupon_leg_pv(coupon, coupon_leg_1_pv):
        return coupon * coupon_leg_1_pv

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now have a reasonably complex graph, which represents calculating conditional default probability:
    """)
    return


@app.cell
def _(comp):
    comp.draw(show_expansion=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To calculate the (unconditional) leg PVs, we loop over various values of $M$, summing weighted conditional leg PVs. Here we use Gauss-Hermite, although other numerical integration schemes, from the trapezoid rule to adaptive methods are also possible. We would caution against using adaptive methods when calculating sensitivities however, as then perturbed results will incorporate changes in decisions made by the adaptive methods, as well as those resulting directly from changes to the inputs.

    Note how Loman is convenient in a couple of ways here. First, we do not have to worry about which things need to be recalculated and which don't as we change $M$. In fact, with extensions like stochastic recovery rates, it can become unclear which things even need to be recalculated, but that is automatically tracked. Second, Loman makes it easy to pull out multiple end or intermediate results. Here we extract three leg PVs.
    """)
    return


@app.cell
def _(comp, np):
    n_abscissas = 40
    Ms, ws = np.polynomial.hermite.hermgauss(n_abscissas)
    Ms *= np.sqrt(2)
    ws /= np.sqrt(np.pi)
    default_leg_pv_uncond = 0
    coupon_leg_1_pv_uncond = 0
    coupon_leg_pv_uncond = 0
    for M, w in zip(Ms, ws):
        comp.insert("M", M)
        comp.compute_all()
        default_leg_pv_uncond += w * comp.v.default_leg_pv
        coupon_leg_1_pv_uncond += w * comp.v.coupon_leg_1_pv
        coupon_leg_pv_uncond += w * comp.v.coupon_leg_pv
    return coupon_leg_1_pv_uncond, coupon_leg_pv_uncond, default_leg_pv_uncond


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And below are the unconditional leg PVs, and fair spread for this tranche. We note that using Loman did not significantly effect the computation time, and the method remains efficient.
    """)
    return


@app.cell
def _(default_leg_pv_uncond):
    default_leg_pv_uncond
    return


@app.cell
def _(coupon_leg_pv_uncond):
    coupon_leg_pv_uncond
    return


@app.cell
def _(coupon_leg_1_pv_uncond, default_leg_pv_uncond):
    default_leg_pv_uncond / coupon_leg_1_pv_uncond
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, for fun, we use Loman's flexibility in exposing intermediate results to calculate an unconditional loss distribution, and plot how that evolves over time:
    """)
    return


@app.cell
def _(comp, np, plt):
    n_abscissas_dist = 40
    Ms_dist, ws_dist = np.polynomial.hermite.hermgauss(n_abscissas_dist)
    Ms_dist *= np.sqrt(2)
    ws_dist /= np.sqrt(np.pi)
    ps_uncond = None
    for M_dist, w_dist in zip(Ms_dist, ws_dist):
        comp.insert("M", M_dist)
        comp.compute_all()
        if ps_uncond is None:
            ps_uncond = np.zeros_like(comp.v.ps)
        ps_uncond += comp.v.ps * w_dist

    plt.plot(comp.v.losses[:, 0], ps_uncond[:, 4], label=f"t={comp.v.ts[4]}")
    plt.plot(comp.v.losses[:, 0], ps_uncond[:, 8], label=f"t={comp.v.ts[8]}")
    plt.plot(comp.v.losses[:, 0], ps_uncond[:, 20], label=f"t={comp.v.ts[20]}")
    plt.xlim(0, 600000)
    plt.legend()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
