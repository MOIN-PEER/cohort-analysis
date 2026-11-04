"""Customer lifetime value (contribution), acquisition cost and payback by channel."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cac_by_channel(spend: pd.DataFrame) -> pd.DataFrame:
    s = spend.groupby("channel")[["spend", "new_customers"]].sum()
    s["cac"] = (s["spend"] / s["new_customers"]).round(2)
    return s.reset_index()


def cumulative_value(c: pd.DataFrame, o: pd.DataFrame, by: str | None = None, months: int = 24) -> pd.DataFrame:
    """Average cumulative contribution per acquired customer at each age.

    Only customers observed for the full window up to each age are included, so young cohorts
    do not drag the average down.
    """
    groups = [("All", c)] if by is None else list(c.groupby(by))
    out = {}
    for seg, g in groups:
        vals = []
        for a in range(months + 1):
            ids = g.loc[g["observed_months"] >= a, "customer_id"]
            if len(ids) < 100:
                vals.append(np.nan)
                continue
            oo = o[o["customer_id"].isin(ids) & (o["age"] <= a)]
            vals.append(round(float(oo["contribution"].sum() / len(ids)), 2))
        out[seg] = vals
    return pd.DataFrame(out)


def payback_month(curve: pd.Series, cac: float):
    hit = curve[curve >= cac]
    return int(hit.index[0]) if len(hit) else None


def ltv_cac_table(c: pd.DataFrame, o: pd.DataFrame, spend: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    cac = cac_by_channel(spend).set_index("channel")["cac"]
    cum = cumulative_value(c, o, "channel", months=24)
    rows = []
    for ch in cum.columns:
        ltv = cum[ch].iloc[horizon]
        rows.append({"channel": ch, "cac": cac[ch], f"ltv_{horizon}m": ltv,
                     f"ltv_{horizon}m_to_cac": round(ltv / cac[ch], 2),
                     "ltv_24m": cum[ch].iloc[24], "payback_month": payback_month(cum[ch], cac[ch])})
    return pd.DataFrame(rows).sort_values(f"ltv_{horizon}m_to_cac", ascending=False)


def offer_economics(c: pd.DataFrame, o: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    """Contribution per customer after N months for each first-box offer."""
    cum = cumulative_value(c, o, "first_box_offer", months=horizon)
    first = o[o["age"] == 0].groupby("first_box_offer")["contribution"].mean()
    rows = [{"first_box_offer": k, "first_box_contribution": round(float(first[k]), 2),
             "contribution_3m": cum[k].iloc[3], f"contribution_{horizon}m": cum[k].iloc[horizon]} for k in cum.columns]
    return pd.DataFrame(rows).sort_values(f"contribution_{horizon}m", ascending=False)
