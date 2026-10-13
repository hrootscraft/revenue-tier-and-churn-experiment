"""Figures for the README and notebooks.

Conventions (kept deliberately plain so every figure reads the same way):

* One y-axis per panel, never two scales on one plot.
* Thin marks, hairline solid gridlines, no chart borders.
* Each learner owns a fixed colour slot, so it is the same colour in every figure.
* Holdout is always the muted grey; the incentive cell is always blue.
* Gains are blue, costs are red, everything else is grey (the diverging pair).
* Every panel has a legend when it shows two or more series, plus one or two
  direct labels where they help; values are never printed on every point.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
LEARNER_COLORS = {
    "CausalForestDML": BLUE,
    "S-learner": ORANGE,
    "T-learner": AQUA,
    "X-learner": YELLOW,
    "R-learner": MAGENTA,
    "DR-learner": GREEN,
}
CELL_COLORS = {"holdout": MUTED, "nudge": ORANGE, "incentive": BLUE}
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "font.size": 9.5,
        "lines.linewidth": 2,
        "axes.titlelocation": "left",
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "axes.axisbelow": True,
    }
)


def _fmt_pct(ax, axis="y"):
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(
        matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    )


def _fmt_money(ax, axis="y"):
    f = matplotlib.ticker.FuncFormatter(
        lambda v, _: f"${v / 1e6:.1f}M" if abs(v) >= 1e6 else f"${v / 1e3:.0f}K"
    )
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def _save(fig, path: Path | None):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
    return fig


# ---------------------------------------------------------------- ATE layer


def fig_ate_estimators(ate_tables: dict[str, pd.DataFrame], truth_rel: dict[str, float], path=None):
    """Relative effect with 95% CI per estimator, one panel per cell; the planted truth as a dotted line."""
    cells = list(ate_tables)
    fig, axes = plt.subplots(1, len(cells), figsize=(4.2 * len(cells), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cell in zip(axes, cells, strict=True):
        t = ate_tables[cell]
        y = np.arange(len(t))[::-1]
        ax.errorbar(
            t["rel_effect"],
            y,
            xerr=1.96 * t["rel_se"],
            fmt="o",
            color=CELL_COLORS[cell],
            ecolor=CELL_COLORS[cell],
            elinewidth=1.4,
            capsize=3,
            markersize=6,
        )
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.axvline(truth_rel[cell], color=INK2, linewidth=1, linestyle=":", label="planted truth")
        ax.set_yticks(y, t.index)
        ax.set_title(f"{cell} vs holdout")
        _fmt_pct(ax, "x")
        ax.set_xlabel("effect on quarterly revenue (relative)")
        ax.legend(loc="lower right")
    fig.suptitle(
        "Four unbiased estimators, four different precisions",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    return _save(fig, path)


def fig_variance_reduction(ate_table: pd.DataFrame, path=None):
    t = ate_table
    fig, ax = plt.subplots(figsize=(6, 2.8))
    y = np.arange(len(t))[::-1]
    ax.barh(y, t["variance_vs_dim"], color=[MUTED] + [BLUE] * (len(t) - 1), height=0.55)
    for yi, v in zip(y, t["variance_vs_dim"], strict=True):
        ax.text(v + 0.02, yi, f"{v:.0%}", va="center", color=INK2, fontsize=9)
    ax.set_yticks(y, t.index)
    ax.set_xlim(0, 1.15)
    _fmt_pct(ax, "x")
    ax.set_xlabel("variance relative to difference in means")
    ax.set_title("CUPED halves the variance; covariate-rich AIPW does better")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def fig_cuped_scatter(pre: np.ndarray, post: np.ndarray, rho: float, path=None):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.scatter(pre, post, s=7, color=BLUE, alpha=0.35, linewidths=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("pre-period revenue (holdout clients)")
    ax.set_ylabel("campaign-quarter revenue")
    ax.set_title(
        f"Pre-period predicts the outcome: rho = {rho:.2f}, so CUPED removes {rho**2:.0%} of the variance"
    )
    return _save(fig, path)


def fig_observational(obs: pd.DataFrame, path=None):
    fig, ax = plt.subplots(figsize=(6.4, 3))
    y = np.arange(len(obs))[::-1]
    colors = [RED if not c else BLUE for c in obs["covers_truth"]]
    ax.errorbar(
        obs["effect"],
        y,
        xerr=1.96 * obs["se"],
        fmt="o",
        color=INK,
        ecolor=INK,
        elinewidth=0,
        capsize=0,
        markersize=0,
    )
    for yi, (e, s, c) in enumerate(zip(obs["effect"], obs["se"], colors, strict=True)):
        ax.plot([e - 1.96 * s, e + 1.96 * s], [y[yi], y[yi]], color=c, linewidth=1.6)
        ax.plot(e, y[yi], "o", color=c, markersize=6)
    ax.axvline(obs["true_ate"].iloc[0], color=INK2, linestyle=":", linewidth=1, label="true ATE")
    ax.set_yticks(y, obs.index)
    _fmt_money(ax, "x")
    ax.set_xlabel("estimated effect per client (quarterly revenue)")
    ax.set_title("Without randomisation, adjustment removes most of the bias, not all of it")
    ax.legend(loc="lower right")
    return _save(fig, path)


# ------------------------------------------------------------- uplift layer


def fig_qini(curves: dict[str, pd.DataFrame], perfect: pd.DataFrame | None, path=None):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    if perfect is not None:
        ax.plot(
            perfect["fraction_targeted"],
            perfect["qini"],
            color=INK2,
            linestyle="--",
            linewidth=1.2,
            label="perfect ranking (oracle)",
        )
    for name, c in curves.items():
        ax.plot(
            c["fraction_targeted"],
            c["qini"],
            color=LEARNER_COLORS.get(name, MUTED),
            label=name,
            linewidth=2.2 if name == "CausalForestDML" else 1.4,
        )
    any_c = next(iter(curves.values()))
    ax.plot(
        any_c["fraction_targeted"],
        any_c["random"],
        color=AXIS,
        linewidth=1,
        label="random targeting",
    )
    ax.set_xlabel("share of clients targeted (ranked by predicted uplift)")
    ax.set_ylabel("cumulative incremental revenue index per client")
    _fmt_pct(ax, "x")
    ax.set_title("Qini curves of cross-fitted scores against the oracle ranking")
    ax.legend(loc="lower left", ncol=2)
    return _save(fig, path)


def fig_benchmark(bench: pd.DataFrame, path=None):
    b = bench.sort_values("auuc")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for ax, col, title, ceiling in zip(
        axes,
        ["auuc", "rate_autoc", "spearman_vs_truth"],
        [
            "AUUC (observed, cross-fitted)",
            "RATE / AUTOC (observed)",
            "Rank correlation with true CATE (oracle)",
        ],
        [bench.attrs.get("perfect_auuc"), None, 1.0],
        strict=True,
    ):
        y = np.arange(len(b))
        ax.barh(y, b[col], color=[LEARNER_COLORS.get(m, MUTED) for m in b.index], height=0.6)
        if col == "rate_autoc":
            ax.errorbar(
                b[col],
                y,
                xerr=1.96 * b["rate_se"],
                fmt="none",
                ecolor=INK2,
                elinewidth=1,
                capsize=2,
            )
        if ceiling is not None and col == "auuc":
            ax.axvline(ceiling, color=INK2, linestyle="--", linewidth=1)
            ax.text(ceiling, len(b) - 0.4, " oracle ceiling", color=INK2, fontsize=8, va="center")
        ax.set_yticks(y, b.index)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=AXIS, linewidth=1)
    fig.suptitle(
        "Uplift model benchmark: observed ranking metrics next to the oracle",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, path)


def fig_cate_vs_distance(
    df: pd.DataFrame, score_col: str, truth_col: str, path=None, n_bins: int = 12
):
    """Binned mean predicted vs true CATE (index units) against distance to the next tier."""
    d = df[df["distance_pct"].notna() & (df["distance_pct"] <= 0.6)].copy()
    d["bin"] = pd.cut(d["distance_pct"], bins=np.linspace(0, 0.6, n_bins + 1))
    g = d.groupby("bin", observed=True).agg(
        x=("distance_pct", "mean"),
        pred=(score_col, "mean"),
        true=(truth_col, "mean"),
        n=("distance_pct", "size"),
    )
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axvspan(0.10, 0.20, color=BLUES[0], alpha=0.8, label="10-20% band", zorder=0)
    ax.plot(
        g["x"], g["true"], color=INK2, linestyle="--", linewidth=1.4, label="true CATE (planted)"
    )
    ax.plot(
        g["x"],
        g["pred"],
        color=BLUE,
        marker="o",
        markersize=5,
        label="CausalForestDML (cross-fitted)",
    )
    ax.axhline(0, color=AXIS, linewidth=1)
    _fmt_pct(ax, "x")
    _fmt_pct(ax, "y")
    ax.set_xlabel("distance to next tier threshold (share of threshold)")
    ax.set_ylabel("effect on revenue index")
    ax.set_title("The forest recovers the goal-gradient shape: peak lift 10-20% from the boundary")
    ax.legend(loc="upper right")
    return _save(fig, path)


def fig_policy_tree_values(pol: pd.DataFrame, path=None):
    p = pol.sort_values("value_per_client")
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    y = np.arange(len(p))
    colors = [RED if v < 0 else BLUE for v in p["value_per_client"]]
    ax.barh(
        y,
        p["value_per_client"],
        xerr=1.96 * p["se"],
        color=colors,
        height=0.6,
        error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2},
    )
    ax.set_yticks(y, p.index)
    ax.axvline(0, color=AXIS, linewidth=1)
    _fmt_money(ax, "x")
    ax.set_xlabel("doubly-robust value per client in the population (quarterly net revenue)")
    ax.set_title(
        "Who to target: blanket rollout loses money; the band rule and the best learners make it"
    )
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


# ---------------------------------------------------------- economics layer


def fig_decomposition(dec: pd.DataFrame, path=None, order=("within_10", "10_to_20", "20_plus")):
    d = dec.loc[[o for o in order if o in dec.index]]
    labels = {
        "within_10": "within 10%\nof next tier",
        "10_to_20": "10-20% from\nnext tier",
        "20_plus": "20%+ from\nnext tier",
    }
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(d))
    w = 0.2
    ax.bar(
        x - 1.5 * w,
        d["volume_effect"],
        w,
        color=BLUE,
        label="volume effect (more trading at the old rate)",
    )
    ax.bar(
        x - 0.5 * w,
        d["rate_effect"],
        w,
        color=RED,
        label="rate effect (old volume at the cheaper rate)",
    )
    ax.bar(x + 0.5 * w, d["interaction"], w, color=MUTED, label="interaction")
    ax.bar(x + 1.5 * w, d["total"], w, color=INK, label="net revenue change")
    for xi, (tot, pct) in enumerate(zip(d["total"], d["net_revenue_pct"], strict=True)):
        ax.text(
            xi + 1.5 * w,
            tot + (800 if tot >= 0 else -800),
            f"{pct:+.0%}",
            ha="center",
            va="bottom" if tot >= 0 else "top",
            fontsize=9,
            color=INK2,
        )
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xticks(x, [labels[i] for i in d.index])
    _fmt_money(ax, "y")
    ax.set_ylabel("per client, campaign quarter")
    ax.set_title(
        "Revenue = volume x rate: the incentive only pays off 10-20% from the boundary (AIPW-adjusted arm means)",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def fig_rollout(roll: pd.DataFrame, band_grid: pd.DataFrame, path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.4]})
    y = np.arange(len(roll))[::-1]
    colors = [RED if v < 0 else BLUE for v in roll["est_net_revenue_pct"]]
    ax1.barh(y, roll["est_net_revenue_pct"], color=colors, height=0.5)
    ax1.errorbar(
        roll["est_net_revenue_pct"],
        y,
        xerr=[
            roll["est_net_revenue_pct"] - roll["ci_low_pct"],
            roll["ci_high_pct"] - roll["est_net_revenue_pct"],
        ],
        fmt="none",
        ecolor=INK2,
        elinewidth=1,
        capsize=2,
    )
    ax1.plot(roll["true_net_revenue_pct"], y, "D", color=INK, markersize=5, label="planted truth")
    ax1.set_yticks(y, roll.index)
    ax1.axvline(0, color=AXIS, linewidth=1)
    _fmt_pct(ax1, "x")
    ax1.set_xlabel("net revenue change of the targeted clients")
    ax1.set_title("Blanket vs targeted rollout")
    ax1.legend(loc="lower left")
    ax1.grid(axis="y", visible=False)

    x = np.arange(len(band_grid))
    colors = [BLUE if i == "10%-20%" else MUTED for i in band_grid.index]
    ax2.bar(x, band_grid["net_revenue_pct"], color=colors, width=0.55)
    ax2.errorbar(
        x,
        band_grid["net_revenue_pct"],
        yerr=[
            band_grid["net_revenue_pct"] - band_grid["ci_low"],
            band_grid["ci_high"] - band_grid["net_revenue_pct"],
        ],
        fmt="none",
        ecolor=INK2,
        elinewidth=1,
        capsize=2,
    )
    ax2.set_xticks(x, band_grid.index)
    ax2.axhline(0, color=AXIS, linewidth=1)
    _fmt_pct(ax2, "y")
    ax2.set_xlabel("distance to next tier")
    ax2.set_ylabel("net revenue change")
    ax2.set_title("Net revenue effect by distance band (95% CI)")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------- durability layer


def fig_weekly_effect(curve: pd.DataFrame, fit: dict, experiment_weeks: int, path=None):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = curve.index.to_numpy()
    ax.fill_between(w, curve["ci_low"], curve["ci_high"], color=BLUES[0], label="95% CI")
    ax.plot(
        w,
        curve["effect_pct"],
        color=BLUE,
        marker="o",
        markersize=3.5,
        linewidth=1.4,
        label="weekly treatment effect",
    )
    ax.plot(
        w,
        fit["initial_effect_pct"] * (1 - fit["weekly_decay"]) ** (w - 1),
        color=INK2,
        linestyle="--",
        linewidth=1.3,
        label=f"exponential fit: {fit['weekly_decay']:.1%}/week decay",
    )
    ax.axvspan(0.5, experiment_weeks + 0.5, color=GRID, alpha=0.5, zorder=0)
    ax.text(1, ax.get_ylim()[1] * 0.95, "campaign quarter", color=MUTED, fontsize=8, va="top")
    ax.axhline(0, color=AXIS, linewidth=1)
    _fmt_pct(ax, "y")
    ax.set_xlabel("week since launch")
    ax.set_ylabel("incentive minus holdout, volume index")
    ax.set_title("The effect fades: weekly treatment-vs-holdout gap over 39 weeks")
    ax.legend(loc="upper right")
    return _save(fig, path)


def fig_survival(km: pd.DataFrame, hazard: pd.DataFrame, true_hazard: float | None, path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax1.step(
        km["week"], km["survival"], where="post", color=BLUE, label="Kaplan-Meier: still engaged"
    )
    ax1.fill_between(
        km["week"],
        km["survival"] - 1.96 * km["se"],
        km["survival"] + 1.96 * km["se"],
        step="post",
        color=BLUES[0],
    )
    h = hazard.attrs["pooled_hazard"]
    ax1.plot(
        km["week"],
        (1 - h) ** km["week"],
        color=INK2,
        linestyle="--",
        linewidth=1.2,
        label=f"constant hazard {h:.1%}/week",
    )
    ax1.set_ylim(0, 1.02)
    _fmt_pct(ax1, "y")
    ax1.set_xlabel("weeks since launch")
    ax1.set_title("Engagement survival curve")
    ax1.legend(loc="lower left")

    ax2.bar(hazard.index, hazard["hazard"], color=BLUES[2], width=0.7)
    ax2.axhline(
        h,
        color=INK,
        linewidth=1.2,
        label=f"pooled estimate {h:.2%} ± {1.96 * hazard.attrs['pooled_hazard_se']:.2%}",
    )
    if true_hazard is not None:
        ax2.axhline(
            true_hazard,
            color=INK2,
            linestyle=":",
            linewidth=1.2,
            label=f"planted {true_hazard:.1%}",
        )
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax2.set_ylim(0, float(hazard["hazard"].max()) * 1.7)
    ax2.set_xlabel("week")
    ax2.set_title("Weekly lapse hazard (events / at risk)")
    ax2.legend(loc="upper right")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, path)


def fig_roi(roi: pd.DataFrame, path=None):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(roi))
    w = 0.3
    ax.bar(x - w / 2, roi["volume_gain"], w, color=BLUE, label="volume gain")
    ax.bar(x + w / 2, roi["rate_cost"], w, color=RED, label="rate-compression cost")
    ax.plot(x, roi["net"], "D", color=INK, markersize=6, label="net")
    for xi, n in zip(x, roi["net"], strict=True):
        ax.text(xi + 0.08, n, f" ${n / 1e6:.1f}M net", va="center", fontsize=9, color=INK2)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xticks(x, roi.index)
    _fmt_money(ax, "y")
    ax.set_title(
        f"Annualising the quarter overstates the benefit by {roi['benefit_overstatement_pct'].iloc[0]:.0%}"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def fig_cox(cox: pd.DataFrame, path=None):
    c = cox.sort_values("hazard_ratio")
    fig, ax = plt.subplots(figsize=(6, 2.8))
    y = np.arange(len(c))
    lo = np.exp(c["log_hr"] - 1.96 * c["se"])
    hi = np.exp(c["log_hr"] + 1.96 * c["se"])
    ax.errorbar(
        c["hazard_ratio"],
        y,
        xerr=[c["hazard_ratio"] - lo, hi - c["hazard_ratio"]],
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        capsize=3,
    )
    ax.axvline(1, color=AXIS, linewidth=1)
    ax.set_yticks(y, c.index)
    ax.set_xlabel("hazard ratio per 1 sd (>1 = lapses sooner)")
    ax.set_title("Cox model: who disengages fastest")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)
