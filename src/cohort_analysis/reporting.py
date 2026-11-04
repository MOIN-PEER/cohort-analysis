"""Run every analysis and write the outputs: CSV/JSON tables, charts, an interactive HTML report and insights."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import cohorts as co, experiments, ltv, sbg, sql_runner  # noqa: E402
from .config import CHANNEL_COLORS, COLORS, FIGURES, INTERACTIVE, PRICE_RISE, RESULTS  # noqa: E402
from .generate_data import load  # noqa: E402

BACKTEST_COHORTS = [f"2023-0{i}" for i in range(1, 7)]


def run_analysis() -> dict:
    d = load()
    c, o = co.prepare(d["customers"], d["orders"])
    sizes = co.cohort_sizes(c)
    act, surv = co.activity_retention(c, o), co.survival(c)
    rev = co.revenue_per_customer(c, o)
    nrr = co.net_revenue_retention(c, o)
    con = sql_runner.connect(d)
    tables = {
        "cohort_sizes": sizes.rename_axis("cohort").reset_index().assign(cohort=lambda x: x["cohort"].astype(str)),
        "activity_retention_matrix": act.reset_index(names="cohort"),
        "subscription_survival_matrix": surv.reset_index(names="cohort"),
        "revenue_per_customer_matrix": rev.reset_index(names="cohort"),
        "net_revenue_retention_matrix": nrr.reset_index(names="cohort"),
        "average_curves": pd.DataFrame({"activity_retention_pct": co.weighted_curve(act, sizes),
                                        "subscription_survival_pct": co.weighted_curve(surv, sizes),
                                        "net_revenue_retention_pct": co.weighted_curve(nrr, sizes),
                                        "revenue_per_customer": co.weighted_curve(rev, sizes)}).rename_axis("month").reset_index(),
        "survival_by_offer": co.segment_survival(c, "first_box_offer"),
        "survival_by_channel": co.segment_survival(c, "channel"),
        "survival_by_plan": co.segment_survival(c, "plan"),
        "curves_by_offer": co.segment_curves(c, "first_box_offer").rename_axis("month").reset_index(),
        "curves_by_channel": co.segment_curves(c, "channel").rename_axis("month").reset_index(),
        "calendar_churn": co.calendar_churn(c),
        "cac_by_channel": ltv.cac_by_channel(d["marketing_spend"]),
        "ltv_cac_by_channel": ltv.ltv_cac_table(c, o, d["marketing_spend"]),
        "cumulative_contribution_by_channel": ltv.cumulative_value(c, o, "channel").rename_axis("month").reset_index(),
        "offer_economics": ltv.offer_economics(c, o),
        "sbg_by_offer": sbg.fit_by_segment(c, "first_box_offer"),
        "sbg_by_channel": sbg.fit_by_segment(c, "channel"),
        "sql_ltv_by_channel": sql_runner.run(con, "ltv_by_channel"),
    }
    bt = sbg.backtest(c, BACKTEST_COHORTS)
    tables["sbg_backtest"] = bt["curve"]
    onboarding = experiments.onboarding_effect(c)
    cal = tables["calendar_churn"].set_index("month")
    avg = tables["average_curves"].set_index("month")
    others = cal.drop(index=str(PRICE_RISE))["tenured_churn_rate_pct"].dropna()
    others = others[others.index >= "2023-07"]          # once there is an established subscriber base
    summary = {
        "customers": int(len(c)), "orders": int(len(o)), "revenue": round(float(o["revenue"].sum()), 2),
        "cohorts": int(c["cohort"].nunique()),
        "activity_retention": {f"month_{m}": float(avg.loc[m, "activity_retention_pct"]) for m in (1, 3, 6, 12, 24)},
        "subscription_survival": {f"month_{m}": float(avg.loc[m, "subscription_survival_pct"]) for m in (1, 3, 6, 12, 24)},
        "net_revenue_retention": {f"month_{m}": float(avg.loc[m, "net_revenue_retention_pct"]) for m in (3, 6, 12, 24)},
        "price_rise": {"month": str(PRICE_RISE), "tenured_churn_pct": float(cal.loc[str(PRICE_RISE), "tenured_churn_rate_pct"]),
                       "typical_tenured_churn_pct": round(float(others.median()), 2),
                       "extra_cancellations": int(round((cal.loc[str(PRICE_RISE), "tenured_churn_rate_pct"] - others.median())
                                                        / 100 * cal.loc[str(PRICE_RISE), "subscribers_start"]))},
        "sbg_backtest": {"params": bt["params"], "train_months": bt["train_months"], "cohorts": bt["cohorts"],
                         "sbg_mape_pct": bt["sbg_mape_pct"], "constant_churn_mape_pct": bt["constant_churn_mape_pct"]},
        "onboarding": onboarding,
        "sql_matches_pandas": bool(_sql_matches(con, act)),
    }
    return {"tables": tables, "summary": summary, "act": act, "surv": surv, "sizes": sizes}


def _sql_matches(con, act: pd.DataFrame) -> bool:
    sql = sql_runner.retention_matrix_sql(con)
    a = act.copy()
    sql = sql.reindex(index=a.index, columns=a.columns)
    mask = a.notna()
    return bool(np.allclose(a[mask].fillna(0), sql[mask].fillna(0)))


# ---------------------------------------------------------------------------------------------
def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=160)
    plt.close(fig)


def make_figures(res: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    t, s, act = res["tables"], res["summary"], res["act"]

    # 1. Full heatmap
    m = act.drop(columns=0)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(m.to_numpy(dtype=float), aspect="auto", cmap="Blues", vmin=0, vmax=80)
    ax.set_yticks(range(len(m)), m.index, fontsize=8)
    ax.set_xticks(range(0, m.shape[1], 3), [str(c) for c in m.columns[::3]])
    ax.set_xlabel("Months since signup")
    ax.set_ylabel("Signup cohort")
    rise = list(m.index).index(str(PRICE_RISE)) if str(PRICE_RISE) in m.index else None
    # Price rise: the cell for (cohort, age) where cohort + age = PRICE_RISE lies on a diagonal
    xs, ys = [], []
    for i, coh in enumerate(m.index):
        age = (PRICE_RISE - pd.Period(coh, "M")).n
        if 1 <= age <= m.columns.max():
            xs.append(age - 1)
            ys.append(i)
    ax.plot(xs, ys, color=COLORS["orange"], lw=2, ls="--", label="April 2025 price rise")
    ax.legend(loc="lower left", frameon=True)
    fig.colorbar(im, ax=ax, label="% of cohort receiving a box")
    _style(ax, "Activity retention by signup cohort (month 1 onwards)")
    del rise
    _save(fig, "01_retention_heatmap.png")

    # 2. Annotated heatmap: 2024-2025 cohorts, first 12 months
    sub = act.loc[[i for i in act.index if i >= "2024-01"], list(range(0, 13))]
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.imshow(sub.to_numpy(dtype=float), aspect="auto", cmap="Blues", vmin=0, vmax=100)
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            v = sub.iat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5, color="white" if v > 55 else "#1d2733")
    ax.set_yticks(range(len(sub)), sub.index, fontsize=8)
    ax.set_xticks(range(13), range(13))
    ax.set_xlabel("Months since signup")
    _style(ax, "Activity retention %, 2024-2025 cohorts, first 12 months")
    _save(fig, "02_retention_heatmap_recent.png")

    # 3. Average curves
    avg = t["average_curves"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(avg["month"], avg["subscription_survival_pct"], color=COLORS["blue"], lw=2.5, label="Still subscribed")
    ax.plot(avg["month"], avg["activity_retention_pct"], color=COLORS["orange"], lw=2.5, label="Received a box that month")
    ax.plot(avg["month"], avg["net_revenue_retention_pct"], color=COLORS["green"], lw=2, ls="--",
            label="Revenue vs month 1 (NRR)")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Months since signup")
    ax.set_ylabel("%")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    _style(ax, "Average retention curves (weighted by cohort size)")
    _save(fig, "03_average_curves.png")

    # 4. Curves by offer and by channel
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    off = t["curves_by_offer"]
    for name, col in (("No offer", COLORS["green"]), ("50% off first box", COLORS["blue"]), ("First box free", COLORS["red"])):
        a1.plot(off["month"], off[name], lw=2.3, color=col, label=name)
    _style(a1, "Still subscribed, by first-box offer")
    chn = t["curves_by_channel"]
    for name in CHANNEL_COLORS:
        a2.plot(chn["month"], chn[name], lw=2.3, color=CHANNEL_COLORS[name], label=name)
    _style(a2, "Still subscribed, by acquisition channel")
    for a in (a1, a2):
        a.set_ylim(0, 102)
        a.set_xlabel("Months since signup")
        a.grid(alpha=0.3)
        a.legend(frameon=False, fontsize=9)
    a1.set_ylabel("% of customers")
    _save(fig, "04_curves_by_segment.png")

    # 5. Calendar churn
    # From mid-2023: before then nearly every subscriber is new, so "established" churn is not comparable
    cal = t["calendar_churn"].dropna()
    cal = cal[cal["month"] >= "2023-07"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cal["month"], cal["tenured_churn_rate_pct"], color=COLORS["blue"], marker="o", ms=3)
    i = list(cal["month"]).index(str(PRICE_RISE))
    ax.annotate(f"Price rise\n{cal['tenured_churn_rate_pct'].iloc[i]}%", (i, cal["tenured_churn_rate_pct"].iloc[i]),
                xytext=(i - 7, cal["tenured_churn_rate_pct"].iloc[i] - 1), color=COLORS["red"],
                arrowprops=dict(arrowstyle="->", color=COLORS["red"]))
    ax.set_xticks(range(0, len(cal), 3), cal["month"].iloc[::3], rotation=45)
    ax.set_ylabel("% of subscribers cancelling")
    ax.set_ylim(0, cal["tenured_churn_rate_pct"].max() * 1.2)
    ax.grid(alpha=0.3)
    _style(ax, "Monthly churn of established subscribers (2+ months)")
    _save(fig, "05_calendar_churn.png")

    # 6. Cumulative contribution vs CAC
    cum = t["cumulative_contribution_by_channel"]
    cac = t["cac_by_channel"].set_index("channel")["cac"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, col in CHANNEL_COLORS.items():
        ax.plot(cum["month"], cum[name], color=col, lw=2.3, label=f"{name} (CAC £{cac[name]:.0f})")
        ax.axhline(cac[name], color=col, lw=1.6, ls=":", alpha=0.9)
    ax.set_xlabel("Months since signup")
    ax.set_ylabel("Cumulative contribution per customer (£)")
    ax.axhline(0, color="#555", lw=0.8)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax, "Payback: cumulative contribution vs acquisition cost (dotted)")
    _save(fig, "06_payback_by_channel.png")

    # 7. LTV:CAC
    lc = t["ltv_cac_by_channel"].sort_values("ltv_12m_to_cac")
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.barh(lc["channel"], lc["ltv_12m_to_cac"], color=[CHANNEL_COLORS[c] for c in lc["channel"]])
    ax.axvline(3, color="#555", ls="--", lw=1)
    ax.text(3.1, len(lc) - 0.45, "3:1 rule of thumb", fontsize=9, color="#555")
    ax.set_ylim(-0.6, len(lc) - 0.1)
    for y, (v, p) in enumerate(zip(lc["ltv_12m_to_cac"], lc["payback_month"])):
        ax.text(v + 0.2, y, f"{v:.1f}x   payback month {p}", va="center", fontsize=9)
    ax.set_xlim(0, lc["ltv_12m_to_cac"].max() * 1.35)
    ax.set_xlabel("12-month contribution ÷ acquisition cost")
    _style(ax, "LTV:CAC by channel (12-month contribution)")
    _save(fig, "07_ltv_cac.png")

    # 8. sBG backtest
    bt = t["sbg_backtest"]
    tm = s["sbg_backtest"]["train_months"]
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.axvspan(0, tm, color="#eef3f8")
    ax.text(tm / 2, 8, "training data", ha="center", fontsize=9, color="#555")
    ax.plot(bt["month"], bt["actual"], color="#1d2733", lw=2.5, label="Actual")
    ax.plot(bt["month"], bt["sbg_forecast"], color=COLORS["green"], lw=2, ls="--",
            label=f"sBG forecast (error {s['sbg_backtest']['sbg_mape_pct']}%)")
    ax.plot(bt["month"], bt["constant_churn_forecast"], color=COLORS["red"], lw=1.8, ls=":",
            label=f"Constant churn forecast (error {s['sbg_backtest']['constant_churn_mape_pct']}%)")
    ax.set_ylim(0, 102)
    ax.set_xlabel("Months since signup")
    ax.set_ylabel("% still subscribed")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    _style(ax, "Backtest: fit on 6 months of Jan-Jun 2023 cohorts, forecast 30")
    _save(fig, "08_sbg_backtest.png")

    # 9. Onboarding
    ob = s["onboarding"]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    labels = ["Naive before/after", "Adjusted for customer mix"]
    vals = [ob["naive"]["difference_pts"], ob["adjusted"]["adjusted_difference_pts"]]
    ax.bar(labels, vals, color=[COLORS["grey"], COLORS["green"]], width=0.5)
    ax.errorbar([0], [vals[0]], yerr=[[vals[0] - ob["naive"]["ci_low_pts"]], [ob["naive"]["ci_high_pts"] - vals[0]]],
                fmt="none", ecolor="#333", capsize=6)
    tops = [ob["naive"]["ci_high_pts"], vals[1]]
    for i, v in enumerate(vals):
        ax.text(i, tops[i] + 0.4, f"+{v} pts", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.6)
    ax.set_ylabel("Change in month-3 retention (pts)")
    _style(ax, "New onboarding (Sep 2024): month-3 retention")
    _save(fig, "09_onboarding_effect.png")


def make_interactive(res: dict) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    INTERACTIVE.mkdir(parents=True, exist_ok=True)
    act, t = res["act"], res["tables"]
    fig = make_subplots(rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]], row_heights=[0.6, 0.4],
                        subplot_titles=("Activity retention by cohort (%)", "Still subscribed, by channel",
                                        "Cumulative contribution per customer (£)"), vertical_spacing=0.1)
    fig.add_trace(go.Heatmap(z=act.to_numpy(dtype=float), x=list(act.columns), y=list(act.index), colorscale="Blues",
                             zmin=0, zmax=100, colorbar=dict(title="%", len=0.55, y=0.78),
                             hovertemplate="Cohort %{y}<br>Month %{x}<br>%{z:.1f}%<extra></extra>"), row=1, col=1)
    chn, cum = t["curves_by_channel"], t["cumulative_contribution_by_channel"]
    for name, col in CHANNEL_COLORS.items():
        fig.add_trace(go.Scatter(x=chn["month"], y=chn[name], name=name, line=dict(color=col), legendgroup=name),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=cum["month"], y=cum[name], name=name, line=dict(color=col), legendgroup=name,
                                 showlegend=False), row=2, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="Months since signup", row=2)
    fig.update_layout(height=1000, title="Crate & Co - cohort retention and lifetime value (synthetic data)",
                      template="plotly_white", font=dict(family="Segoe UI, Arial"))
    fig.write_html(INTERACTIVE / "cohort_dashboard.html", include_plotlyjs="cdn")


# ---------------------------------------------------------------------------------------------
def write_insights(res: dict) -> str:
    t, s = res["tables"], res["summary"]
    sv, ob = s["subscription_survival"], s["onboarding"]
    off = t["survival_by_offer"].set_index("first_box_offer")
    oe = t["offer_economics"].set_index("first_box_offer")
    lc = t["ltv_cac_by_channel"].set_index("channel")
    pr, bt = s["price_rise"], s["sbg_backtest"]
    lines = [
        "# Key insights (generated from the data by reporting.py)",
        "",
        f"1. **Retention curve.** {sv['month_1']}% of customers are still subscribed after month 1, "
        f"{sv['month_3']}% after 3 months, {sv['month_12']}% after 12 and {sv['month_24']}% after 24. The curve "
        f"flattens: those who stay past the first few months are much less likely to leave.",
        f"2. **Revenue holds up better than customers.** Revenue at month 12 is {s['net_revenue_retention']['month_12']}% "
        f"of month-1 revenue, while subscribers are fewer, because loyal customers buy more add-ons.",
        f"3. **'First box free' attracts customers who leave.** Only {off.loc['First box free', 'month_1_pct']}% stay "
        f"past month 1 (vs {off.loc['No offer', 'month_1_pct']}% with no offer). After 12 months a free-box customer has "
        f"contributed £{oe.loc['First box free', 'contribution_12m']:.0f} vs £{oe.loc['No offer', 'contribution_12m']:.0f} "
        f"with no offer.",
        f"4. **Channels.** Referral and organic customers retain best and pay back fastest. Influencer customers have the "
        f"highest acquisition cost (£{lc.loc['Influencer', 'cac']:.0f}) and the lowest 12-month LTV:CAC "
        f"({lc.loc['Influencer', 'ltv_12m_to_cac']}x, payback month {lc.loc['Influencer', 'payback_month']}), below "
        f"the common 3:1 rule of thumb.",
        f"5. **Price rise.** In {pr['month']}, churn among established subscribers jumped to {pr['tenured_churn_pct']}% "
        f"from a typical {pr['typical_tenured_churn_pct']}%: about {pr['extra_cancellations']} extra cancellations. "
        f"It appears as a diagonal in the cohort heatmap because it hit every cohort in the same calendar month.",
        f"6. **Onboarding worked.** Month-3 retention rose {ob['naive']['difference_pts']} points before/after "
        f"September 2024; controlling for customer mix the estimate is {ob['adjusted']['adjusted_difference_pts']} points "
        f"(odds ratio {ob['adjusted']['odds_ratio']}, 95% CI {ob['adjusted']['ci_low']}-{ob['adjusted']['ci_high']}). "
        f"It is a before/after study, not an A/B test, so other changes at the same time cannot be ruled out.",
        f"7. **Forecasting.** A shifted-beta-geometric model fitted on just {bt['train_months']} months of the "
        f"Jan-Jun 2023 cohorts forecast the next two years with {bt['sbg_mape_pct']}% average error, against "
        f"{bt['constant_churn_mape_pct']}% for a constant-churn model, so young cohorts' LTV can be projected early.",
        f"8. **Check.** The SQL retention query gives exactly the same matrix as pandas: {s['sql_matches_pandas']}.",
        "",
        "_All data is synthetic, generated by src/cohort_analysis/generate_data.py._",
    ]
    text = "\n".join(lines) + "\n"
    (RESULTS / "insights.md").write_text(text)
    return text


def run_all() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    res = run_analysis()
    for name, df in res["tables"].items():
        df.to_csv(RESULTS / f"{name}.csv", index=False)
    (RESULTS / "summary.json").write_text(json.dumps(res["summary"], indent=2, default=str))
    make_figures(res)
    make_interactive(res)
    write_insights(res)
    return res
