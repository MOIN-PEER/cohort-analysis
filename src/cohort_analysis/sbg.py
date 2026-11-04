"""Shifted-beta-geometric (sBG) model for subscription retention (Fader & Hardie, 2007).

Each customer cancels each month with their own probability p, and p varies across customers
following a Beta(a, b) distribution. High-risk customers leave early, so the survivors are
increasingly loyal and the retention curve flattens - which a simple constant-churn
(geometric) model cannot capture.

  P(cancel in month t) = B(a + 1, b + t - 1) / B(a, b)       t = 1, 2, ...
  S(t) = P(still subscribed after t months) = B(a, b + t) / B(a, b)

Parameters are estimated by maximum likelihood from cohort survival counts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln


def survival_curve(a: float, b: float, months: int) -> np.ndarray:
    t = np.arange(months + 1)
    return np.exp(betaln(a, b + t) - betaln(a, b))


def _neg_log_lik(log_params, lifetimes: np.ndarray, censored: np.ndarray) -> float:
    a, b = np.exp(log_params)
    ll_churn = betaln(a + 1, b + lifetimes - 1) - betaln(a, b)      # cancelled at month t
    ll_alive = betaln(a, b + lifetimes) - betaln(a, b)              # still subscribed after t months
    return -float(np.where(censored, ll_alive, ll_churn).sum())


def fit(lifetimes: np.ndarray, censored: np.ndarray) -> dict:
    """lifetimes: months until cancellation (or months observed, if still subscribed)."""
    res = minimize(_neg_log_lik, x0=np.log([1.0, 3.0]), args=(lifetimes, censored), method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
    a, b = np.exp(res.x)
    return {"a": round(float(a), 4), "b": round(float(b), 4), "mean_churn_month1": round(float(a / (a + b)), 4),
            "neg_log_lik": round(float(res.fun), 2), "converged": bool(res.success)}


def lifetimes_upto(c: pd.DataFrame, horizon: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Lifetimes and censoring flags, optionally pretending we only observed `horizon` months."""
    life = c["lifetime_months"].to_numpy()
    cens = ~c["is_cancelled"].to_numpy()
    obs = c["observed_months"].to_numpy() + 1
    if horizon is not None:
        obs = np.minimum(obs, horizon)
    cens = cens | (life > obs)
    life = np.minimum(life, obs)
    return life, cens


def backtest(c: pd.DataFrame, cohorts: list[str], train_months: int = 6, horizon: int = 30) -> dict:
    """Fit on the first `train_months` of some cohorts, then compare forecast with what really happened."""
    g = c[c["cohort"].astype(str).isin(cohorts)]
    life, cens = lifetimes_upto(g, train_months)
    params = fit(life, cens)
    forecast = 100 * survival_curve(params["a"], params["b"], horizon)
    geo_p = 1 - (np.exp(np.log((g["lifetime_months"] > train_months).mean()) / train_months))
    geometric = 100 * (1 - geo_p) ** np.arange(horizon + 1)
    actual = np.array([100 * (g["lifetime_months"] > t).mean() for t in range(horizon + 1)])
    test = slice(train_months + 1, horizon + 1)
    mape = lambda f: round(float(np.mean(np.abs(f[test] - actual[test]) / actual[test]) * 100), 1)  # noqa: E731
    return {"params": params, "train_months": train_months, "cohorts": cohorts,
            "curve": pd.DataFrame({"month": range(horizon + 1), "actual": actual.round(1),
                                   "sbg_forecast": forecast.round(1), "constant_churn_forecast": geometric.round(1)}),
            "sbg_mape_pct": mape(forecast), "constant_churn_mape_pct": mape(geometric)}


def fit_by_segment(c: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for seg, g in c.groupby(by):
        life, cens = lifetimes_upto(g)
        p = fit(life, cens)
        s = survival_curve(p["a"], p["b"], 24)
        rows.append({by: seg, **p, "projected_month_12_pct": round(100 * s[12], 1),
                     "projected_month_24_pct": round(100 * s[24], 1),
                     "expected_months_in_first_24": round(float(s[:24].sum()), 2)})
    return pd.DataFrame(rows)
