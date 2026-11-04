"""Cohort tables: who is still active, and how much they spend, N months after signing up.

Two views of retention are used:
  * activity retention   - % of a cohort that received a box in month N (pauses count as inactive)
  * subscription survival - % of a cohort that had not cancelled by month N
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GROSS_MARGIN, LAST_MONTH


def prepare(customers: pd.DataFrame, orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = customers.copy()
    c["cohort"] = pd.PeriodIndex(c["signup_month"], freq="M")
    c["cancelled"] = pd.PeriodIndex(c["cancelled_month"], freq="M")
    end = LAST_MONTH + 1
    c["lifetime_months"] = [(x if not pd.isna(x) else end) - s for x, s in zip(c["cancelled"], c["cohort"])]
    c["lifetime_months"] = c["lifetime_months"].map(lambda d: d.n)
    c["is_cancelled"] = c["cancelled"].notna()
    c["observed_months"] = c["cohort"].map(lambda s: (LAST_MONTH - s).n)   # oldest age we can see
    o = orders.merge(c[["customer_id", "cohort", "channel", "first_box_offer", "plan"]], on="customer_id")
    o["order_period"] = pd.PeriodIndex(o["order_month"], freq="M")
    o["age"] = (o["order_period"] - o["cohort"]).map(lambda d: d.n)
    # Contribution = revenue minus cost of goods; a free box still costs money to make and deliver.
    o["contribution"] = (o["box_revenue"] + o["addon_revenue"]
                         - (1 - GROSS_MARGIN) * (o["box_list_price"] + o["addon_revenue"])).round(2)
    return c, o


def cohort_sizes(c: pd.DataFrame) -> pd.Series:
    return c.groupby("cohort").size().rename("customers")


def activity_retention(c: pd.DataFrame, o: pd.DataFrame) -> pd.DataFrame:
    """% of each cohort with a box in month N (rows: cohort, columns: months since signup)."""
    active = o.groupby(["cohort", "age"])["customer_id"].nunique().unstack(fill_value=0)
    m = active.div(cohort_sizes(c), axis=0) * 100
    return _mask_unobserved(m.round(1))


def survival(c: pd.DataFrame, max_age: int = 35) -> pd.DataFrame:
    """% of each cohort still subscribed (not cancelled) at month N."""
    rows = {}
    for coh, g in c.groupby("cohort"):
        rows[coh] = [100 * (g["lifetime_months"] > n).mean() for n in range(max_age + 1)]
    m = pd.DataFrame(rows).T
    m.columns.name = "age"
    return _mask_unobserved(m.round(1))


def revenue_per_customer(c: pd.DataFrame, o: pd.DataFrame, value: str = "revenue") -> pd.DataFrame:
    """Revenue (or contribution) in month N divided by the cohort's original size."""
    r = o.groupby(["cohort", "age"])[value].sum().unstack(fill_value=0.0)
    return _mask_unobserved(r.div(cohort_sizes(c), axis=0).round(2))


def net_revenue_retention(c: pd.DataFrame, o: pd.DataFrame) -> pd.DataFrame:
    """Revenue in month N as % of month-1 revenue (month 0 is distorted by first-box discounts)."""
    r = o.groupby(["cohort", "age"])["revenue"].sum().unstack(fill_value=0.0)
    return _mask_unobserved((r.div(r[1], axis=0) * 100).round(1))


def _mask_unobserved(m: pd.DataFrame) -> pd.DataFrame:
    """Blank out cells after the end of the data (e.g. 2025-11 cohort has no month-3 yet)."""
    m = m.copy()
    for coh in m.index:
        limit = (LAST_MONTH - coh).n
        m.loc[coh, [a for a in m.columns if a > limit]] = np.nan
    m.index = m.index.astype(str)
    return m


def weighted_curve(matrix: pd.DataFrame, sizes: pd.Series, min_cohorts: int = 3) -> pd.Series:
    """Average curve across cohorts, weighted by cohort size, using only cohorts observed at each age."""
    sizes = sizes.copy()
    sizes.index = sizes.index.astype(str)
    out = {}
    for age in matrix.columns:
        col = matrix[age].dropna()
        if len(col) >= min_cohorts:
            w = sizes.loc[col.index]
            out[age] = round(float((col * w).sum() / w.sum()), 1)
    return pd.Series(out, name="value")


def segment_survival(c: pd.DataFrame, by: str, ages=(1, 3, 6, 12)) -> pd.DataFrame:
    """Survival at selected ages by segment, using only customers observed for at least that long."""
    rows = []
    for seg, g in c.groupby(by):
        row = {by: seg, "customers": len(g)}
        for a in ages:
            eligible = g[g["observed_months"] >= a]
            row[f"month_{a}_pct"] = round(100 * (eligible["lifetime_months"] > a).mean(), 1) if len(eligible) else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values(f"month_{ages[1]}_pct", ascending=False)


def segment_curves(c: pd.DataFrame, by: str, max_age: int = 24) -> pd.DataFrame:
    """Survival curve per segment (each point uses customers observed at that age)."""
    out = {}
    for seg, g in c.groupby(by):
        out[seg] = [100 * (g.loc[g["observed_months"] >= a, "lifetime_months"] > a).mean() for a in range(max_age + 1)]
    return pd.DataFrame(out).round(1)


def calendar_churn(c: pd.DataFrame) -> pd.DataFrame:
    """Monthly churn rate by calendar month = cancellations taking effect / subscribers the month before."""
    rows = []
    for m in pd.period_range(c["cohort"].min() + 1, LAST_MONTH, freq="M"):
        base = ((c["cohort"] <= m - 1) & (c["cancelled"].isna() | (c["cancelled"] > m - 1))).sum()
        lost = (c["cancelled"] == m).sum()
        # exclude month-1 cancellations so first-box promotions do not dominate the calendar view
        lost_tenured = ((c["cancelled"] == m) & (c["cohort"] <= m - 2)).sum()
        base_tenured = ((c["cohort"] <= m - 2) & (c["cancelled"].isna() | (c["cancelled"] > m - 1))).sum()
        rows.append({"month": str(m), "subscribers_start": int(base), "cancellations": int(lost),
                     "churn_rate_pct": round(100 * lost / base, 2),
                     "tenured_churn_rate_pct": round(100 * lost_tenured / base_tenured, 2) if base_tenured else None})
    return pd.DataFrame(rows)
