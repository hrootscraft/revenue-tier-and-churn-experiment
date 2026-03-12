"""
Generate static figures for the leadership report.

Usage:
    cd coinbase-fee-tier-experiment
    python reports/generate_figures.py

Requires: pip install kaleido  (for plotly static image export)
Outputs PNG files to reports/figures/
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy import stats as _stats
import sys

src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))

from analysis.statistical_tests import cuped_comparison
from analysis.revenue_decomposition import summarize_by_group
from analysis.durability import compute_weekly_effects, estimate_decay_rate
from analysis.churn_analysis import survival_analysis, prepare_survival_data

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

data_dir = Path(__file__).resolve().parent.parent / "data"


def save(fig, name, width=900, height=500):
    path = FIGURES_DIR / f"{name}.png"
    fig.write_image(str(path), width=width, height=height, scale=2)
    print(f"  Saved {path.name}")


def fig_cuped(df):
    """CUPED variance reduction — before/after CI comparison."""
    t_vol = df[df["group"] == "treatment"]["post_volume"].values
    c_vol = df[df["group"] == "control"]["post_volume"].values
    t_pre = df[df["group"] == "treatment"]["pre_volume"].values
    c_pre = df[df["group"] == "control"]["pre_volume"].values

    cuped = cuped_comparison(t_vol, c_vol, t_pre, c_pre)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[cuped["raw_ci"][0], cuped["raw_ate"], cuped["raw_ci"][1]],
        y=["Raw", "Raw", "Raw"],
        mode="lines+markers",
        marker=dict(size=[8, 14, 8], color="#EF553B",
                    symbol=["line-ew", "diamond", "line-ew"]),
        line=dict(color="#EF553B", width=3),
        name="Raw ATE",
    ))
    fig.add_trace(go.Scatter(
        x=[cuped["cuped_ci"][0], cuped["cuped_ate"], cuped["cuped_ci"][1]],
        y=["CUPED", "CUPED", "CUPED"],
        mode="lines+markers",
        marker=dict(size=[8, 14, 8], color="#636EFA",
                    symbol=["line-ew", "diamond", "line-ew"]),
        line=dict(color="#636EFA", width=3),
        name="CUPED ATE",
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

    fig.add_annotation(x=cuped["raw_ate"], y="Raw",
                       text=f"${cuped['raw_ate']:,.0f}",
                       showarrow=False, yshift=22,
                       font=dict(color="#EF553B", size=13))
    fig.add_annotation(x=cuped["cuped_ate"], y="CUPED",
                       text=f"${cuped['cuped_ate']:,.0f}",
                       showarrow=False, yshift=22,
                       font=dict(color="#636EFA", size=13))

    raw_w = cuped["raw_ci"][1] - cuped["raw_ci"][0]
    cuped_w = cuped["cuped_ci"][1] - cuped["cuped_ci"][0]

    fig.update_layout(
        title=f"CUPED Shrinks the Confidence Interval by {(1 - cuped_w / raw_w) * 100:.0f}%",
        xaxis_title="Average Treatment Effect ($)",
        xaxis=dict(tickformat="$,.0f"),
        template="plotly_white",
        showlegend=False,
        margin=dict(t=60, l=80),
    )
    save(fig, "01_cuped_comparison", width=850, height=300)


def fig_revenue_waterfall(df):
    """Revenue decomposition waterfall — the hero visual."""
    t = df[df["group"] == "treatment"]
    vol = t["volume_effect"].sum()
    rate = t["rate_effect"].sum()
    inter = t["interaction_effect"].sum()
    net = t["revenue_change"].sum()

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["Volume Effect<br>(more trading)", "Rate Effect<br>(cheaper tier)",
           "Interaction<br>(new vol @ new rate)", "Net Revenue<br>Change"],
        y=[vol, rate, inter, net],
        text=[f"${vol:,.0f}", f"${rate:,.0f}", f"${inter:,.0f}", f"${net:,.0f}"],
        textposition="outside",
        connector=dict(line=dict(color="rgb(63, 63, 63)", width=1)),
        increasing=dict(marker_color="#2CA02C"),
        decreasing=dict(marker_color="#EF553B"),
        totals=dict(marker_color="#636EFA"),
    ))
    fig.update_layout(
        title="Revenue Decomposition: Treatment Group Total",
        yaxis_title="Revenue Impact ($)",
        yaxis=dict(tickformat="$,.0f"),
        template="plotly_white",
        margin=dict(t=60),
    )
    save(fig, "02_revenue_waterfall", width=800, height=480)


def fig_waterfall_by_bucket(df):
    """Revenue waterfall for each distance bucket — reveals the HTE story."""
    rev = summarize_by_group(df, "distance_bucket")
    labels_map = {
        "within_10": "Within 10%",
        "10_to_20": "10–20%",
        "20_plus": ">20%",
    }
    order = ["within_10", "10_to_20", "20_plus"]
    rev = rev.set_index("distance_bucket").loc[order].reset_index()

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            f"{labels_map[row['distance_bucket']]} (n={row['n_clients']})"
            for _, row in rev.iterrows()
        ],
        horizontal_spacing=0.08,
    )
    for i, (_, row) in enumerate(rev.iterrows(), 1):
        vals = [row["mean_volume_effect"], row["mean_rate_effect"],
                row["mean_interaction"], row["mean_net_revenue_change"]]
        fig.add_trace(go.Waterfall(
            measure=["relative", "relative", "relative", "total"],
            x=["Volume", "Rate", "Interact.", "Net"],
            y=vals,
            text=[f"${v:,.0f}" for v in vals],
            textposition="outside", textfont=dict(size=10),
            connector=dict(line=dict(width=0.5)),
            increasing=dict(marker_color="#2CA02C"),
            decreasing=dict(marker_color="#EF553B"),
            totals=dict(marker_color="#636EFA"),
            showlegend=False,
        ), row=1, col=i)

    fig.update_layout(
        title="Revenue Decomposition by Distance to Next Tier (per client avg)",
        template="plotly_white",
        margin=dict(t=80),
    )
    for i in range(1, 4):
        fig.update_yaxes(tickformat="$,.0f", row=1, col=i)

    save(fig, "03_waterfall_by_bucket", width=1050, height=420)


def fig_forest_plot(df):
    """Forest plot — HTE by distance bucket and segment."""
    from analysis.hte import compute_hte

    hte_dist = compute_hte(df, "distance_bucket", outcome_col="post_volume")
    hte_seg = compute_hte(df, "segment", outcome_col="post_volume")

    hte_dist["cut"] = "Distance"
    hte_seg["cut"] = "Segment"
    forest = pd.concat([hte_dist, hte_seg], ignore_index=True)
    forest["ci_lower"] = forest["raw_effect"] - 1.96 * forest["se"]
    forest["ci_upper"] = forest["raw_effect"] + 1.96 * forest["se"]
    forest = forest.sort_values("cut", ascending=False)

    fig = go.Figure()
    colors = {"Distance": "#636EFA", "Segment": "#00CC96"}

    for cut in ["Distance", "Segment"]:
        sub = forest[forest["cut"] == cut]
        fig.add_trace(go.Scatter(
            x=sub["raw_effect"], y=sub["group"],
            error_x=dict(
                type="data", symmetric=False,
                array=(sub["ci_upper"] - sub["raw_effect"]).values,
                arrayminus=(sub["raw_effect"] - sub["ci_lower"]).values,
            ),
            mode="markers",
            marker=dict(size=10, color=colors[cut]),
            name=cut,
        ))

    if "shrunk_effect" in forest.columns:
        fig.add_trace(go.Scatter(
            x=forest["shrunk_effect"], y=forest["group"],
            mode="markers",
            marker=dict(size=7, color="rgba(0,0,0,0)",
                        line=dict(color="black", width=1.5)),
            name="Shrunk (EB)",
        ))

    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
    fig.update_layout(
        title="Treatment Effect by Subgroup (with 95% CI + Empirical Bayes Shrinkage)",
        xaxis_title="Treatment Effect on Volume ($)",
        xaxis=dict(tickformat="$,.0f"),
        template="plotly_white",
        legend=dict(x=0.72, y=0.05),
        margin=dict(t=60, l=160),
    )
    save(fig, "04_forest_plot", width=900, height=480)


def fig_durability(post_activity, assignment):
    """Weekly treatment effect over time — does the lift persist?"""
    weekly = compute_weekly_effects(post_activity, assignment)
    decay = estimate_decay_rate(weekly)

    fig = go.Figure()

    # CI band
    fig.add_trace(go.Scatter(
        x=pd.concat([weekly["week"], weekly["week"][::-1]]),
        y=pd.concat([weekly["ci_upper"] / weekly["control_mean"] * 100,
                      (weekly["ci_lower"] / weekly["control_mean"] * 100).iloc[::-1]]),
        fill="toself",
        fillcolor="rgba(99, 110, 250, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% CI",
        hoverinfo="skip",
    ))

    # Effect line
    fig.add_trace(go.Scatter(
        x=weekly["week"], y=weekly["effect_pct"],
        mode="lines+markers",
        name="Weekly effect",
        line=dict(color="#636EFA", width=2.5),
        marker=dict(size=7),
    ))

    # Trend line
    slope, intercept, _, _, _ = _stats.linregress(weekly["week"], weekly["effect_pct"])
    trend_y = intercept + slope * weekly["week"]
    fig.add_trace(go.Scatter(
        x=weekly["week"], y=trend_y,
        mode="lines",
        name=f"Trend ({slope:+.2f}%/week)",
        line=dict(color="#EF553B", width=2, dash="dash"),
    ))

    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
    fig.update_layout(
        title="Treatment Effect Over Time: Does It Persist or Fade?",
        xaxis_title="Experiment Week",
        yaxis_title="Treatment Effect (% lift over control)",
        yaxis=dict(ticksuffix="%"),
        template="plotly_white",
        legend=dict(x=0.7, y=0.95),
        margin=dict(t=60),
    )
    save(fig, "05_durability", width=850, height=450)


def fig_kaplan_meier(df):
    """Kaplan-Meier survival curves — treatment vs control."""
    surv = survival_analysis(df)
    km_t = surv["km_treatment"]
    km_c = surv["km_control"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=km_t.survival_function_.index,
        y=km_t.survival_function_["Treatment"],
        mode="lines", name="Treatment",
        line=dict(color="#2563eb", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=km_c.survival_function_.index,
        y=km_c.survival_function_["Control"],
        mode="lines", name="Control",
        line=dict(color="#dc2626", width=2.5),
    ))
    fig.update_layout(
        title="Client Retention: Treatment vs Control",
        xaxis_title="Weeks After Experiment Start",
        yaxis_title="Retention Probability",
        yaxis_range=[0.5, 1.02],
        template="plotly_white",
        margin=dict(t=60),
    )
    save(fig, "06_kaplan_meier", width=800, height=450)


def fig_golden_handcuff(df):
    """Kaplan-Meier: crossed tier vs did not cross (within treatment)."""
    from lifelines import KaplanMeierFitter

    surv = prepare_survival_data(df)
    t_surv = surv[surv["is_treatment"] == 1]

    km_c = KaplanMeierFitter()
    km_nc = KaplanMeierFitter()

    mask = t_surv["crossed_tier"] == True  # noqa: E712
    km_c.fit(t_surv.loc[mask, "duration"], t_surv.loc[mask, "event"],
             label="Crossed Tier")
    km_nc.fit(t_surv.loc[~mask, "duration"], t_surv.loc[~mask, "event"],
              label="Did Not Cross")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=km_c.survival_function_.index,
        y=km_c.survival_function_["Crossed Tier"],
        mode="lines", name="Crossed Tier",
        line=dict(color="#16a34a", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=km_nc.survival_function_.index,
        y=km_nc.survival_function_["Did Not Cross"],
        mode="lines", name="Did Not Cross",
        line=dict(color="#f97316", width=2.5),
    ))
    fig.update_layout(
        title="The Golden Handcuff: Tier Crossers Stay Longer (Treatment Group Only)",
        xaxis_title="Weeks After Experiment Start",
        yaxis_title="Retention Probability",
        yaxis_range=[0.5, 1.02],
        template="plotly_white",
        annotations=[dict(
            x=0.5, y=0.02, xref="paper", yref="paper", showarrow=False,
            text="Observational — tier crossing was not randomized",
            font=dict(size=10, color="gray"),
        )],
        margin=dict(t=60),
    )
    save(fig, "07_golden_handcuff", width=800, height=450)


def main():
    print("Loading data...")
    df = pd.read_csv(data_dir / "processed" / "experiment_summary.csv")
    post_activity = pd.read_csv(data_dir / "raw" / "post_period_activity.csv")
    assignment = pd.read_csv(data_dir / "raw" / "assignment.csv")

    print("Generating figures...")
    fig_cuped(df)
    fig_revenue_waterfall(df)
    fig_waterfall_by_bucket(df)
    fig_forest_plot(df)
    fig_durability(post_activity, assignment)
    fig_kaplan_meier(df)
    fig_golden_handcuff(df)

    print(f"\nDone! {len(list(FIGURES_DIR.glob('*.png')))} figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
