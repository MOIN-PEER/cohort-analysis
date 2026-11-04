"""Did the new onboarding flow (September 2024) improve retention?

A naive before/after comparison can be confounded: the customer mix changes over time (for
example, more influencer customers, who churn more, joined after the change). So the comparison is
repeated with a logistic regression that controls for channel, offer, plan and signup month-of-year.
This is still an observational before/after study, not a randomised A/B test: anything else that
changed in September 2024 could also explain the difference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm

from .config import ONBOARDING_CHANGE

WINDOW = 12         # months of cohorts either side (12, so every calendar month appears before and after)
OUTCOME_MONTH = 3   # retention measured at month 3


def _sample(c: pd.DataFrame) -> pd.DataFrame:
    lo, hi = ONBOARDING_CHANGE - WINDOW, ONBOARDING_CHANGE + WINDOW - 1
    s = c[(c["cohort"] >= lo) & (c["cohort"] <= hi)].copy()
    s = s[s["observed_months"] >= OUTCOME_MONTH]
    s["after"] = (s["cohort"] >= ONBOARDING_CHANGE).astype(int)
    s["retained"] = (s["lifetime_months"] > OUTCOME_MONTH).astype(int)
    s["signup_moy"] = s["cohort"].map(lambda p: p.month)
    return s


def two_proportion_test(x1: int, n1: int, x2: int, n2: int) -> dict:
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se
    se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p2 - p1
    return {"before_pct": round(100 * p1, 1), "after_pct": round(100 * p2, 1), "difference_pts": round(100 * d, 1),
            "ci_low_pts": round(100 * (d - 1.96 * se_diff), 1), "ci_high_pts": round(100 * (d + 1.96 * se_diff), 1),
            "z": round(float(z), 2), "p_value": round(float(2 * (1 - norm.cdf(abs(z)))), 4)}


def onboarding_effect(c: pd.DataFrame) -> dict:
    s = _sample(c)
    b, a = s[s["after"] == 0], s[s["after"] == 1]
    naive = two_proportion_test(int(b["retained"].sum()), len(b), int(a["retained"].sum()), len(a))
    mix = s.groupby("after")[["channel", "first_box_offer"]].agg(
        lambda x: round(100 * ((x == "Influencer") | (x == "First box free")).mean(), 1))
    model = smf.logit("retained ~ after + C(channel) + C(first_box_offer) + C(plan) + C(signup_moy)", data=s).fit(disp=False)
    lo, hi = model.conf_int().loc["after"]
    # Average marginal effect: predicted retention if everyone were "after" minus if everyone were "before"
    p_after = model.predict(s.assign(after=1)).mean()
    p_before = model.predict(s.assign(after=0)).mean()
    return {
        "window": f"{ONBOARDING_CHANGE - WINDOW} to {ONBOARDING_CHANGE + WINDOW - 1}",
        "outcome": f"still subscribed at month {OUTCOME_MONTH}",
        "n_before": int(len(b)), "n_after": int(len(a)),
        "naive": naive,
        "mix_shift": {"influencer_pct_before": float(mix.loc[0, "channel"]), "influencer_pct_after": float(mix.loc[1, "channel"]),
                      "free_box_pct_before": float(mix.loc[0, "first_box_offer"]),
                      "free_box_pct_after": float(mix.loc[1, "first_box_offer"])},
        "adjusted": {"odds_ratio": round(float(np.exp(model.params["after"])), 3),
                     "ci_low": round(float(np.exp(lo)), 3), "ci_high": round(float(np.exp(hi)), 3),
                     "p_value": round(float(model.pvalues["after"]), 4),
                     "adjusted_difference_pts": round(float(100 * (p_after - p_before)), 1)},
    }
