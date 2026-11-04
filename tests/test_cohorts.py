"""Cohort metrics: definitions, SQL/pandas agreement and the sBG model."""

import numpy as np
import pandas as pd
import pytest

from cohort_analysis import cohorts, experiments, ltv, sbg, sql_runner


def test_month_zero_retention_is_100(prepared):
    c, o = prepared
    act = cohorts.activity_retention(c, o)
    assert (act[0] == 100).all()


def test_unobserved_cells_are_blank(prepared):
    c, o = prepared
    act = cohorts.activity_retention(c, o)
    assert np.isnan(act.loc["2025-12", 1]) and not np.isnan(act.loc["2025-11", 1])


def test_survival_never_increases(prepared):
    c, _ = prepared
    s = cohorts.survival(c)
    diffs = s.diff(axis=1).iloc[:, 1:]
    assert (diffs.fillna(0) <= 1e-9).all().all()


def test_sql_matches_pandas(data, prepared):
    c, o = prepared
    act = cohorts.activity_retention(c, o)
    sql = sql_runner.retention_matrix_sql(sql_runner.connect(data)).reindex(index=act.index, columns=act.columns)
    mask = act.notna()
    assert np.allclose(act[mask].fillna(0), sql[mask].fillna(0))


def test_sbg_recovers_known_parameters():
    # Simulate customers with Beta(0.8, 4) churn probabilities and check the fit finds them.
    rng = np.random.default_rng(1)
    p = rng.beta(0.8, 4.0, 20000)
    life = rng.geometric(p)
    cens = life > 24
    life = np.minimum(life, 24)
    est = sbg.fit(life, cens)
    assert est["a"] == pytest.approx(0.8, rel=0.15) and est["b"] == pytest.approx(4.0, rel=0.2)


def test_sbg_survival_curve_properties():
    s = sbg.survival_curve(0.5, 2.0, 24)
    assert s[0] == 1.0 and np.all(np.diff(s) < 0)
    assert s[1] == pytest.approx(2.0 / 2.5)          # S(1) = 1 - a / (a + b)


def test_sbg_beats_constant_churn_in_backtest(prepared):
    c, _ = prepared
    bt = sbg.backtest(c, [f"2023-0{i}" for i in range(1, 7)])
    assert bt["sbg_mape_pct"] < bt["constant_churn_mape_pct"] and bt["sbg_mape_pct"] < 15


def test_free_box_customers_churn_most(prepared):
    c, _ = prepared
    s = cohorts.segment_survival(c, "first_box_offer").set_index("first_box_offer")
    assert s.loc["First box free", "month_1_pct"] < s.loc["No offer", "month_1_pct"] - 15


def test_payback_month():
    curve = pd.Series([-5.0, 10.0, 25.0, 40.0])
    assert ltv.payback_month(curve, 20.0) == 2
    assert ltv.payback_month(curve, 100.0) is None


def test_two_proportion_test_known_case():
    r = experiments.two_proportion_test(50, 100, 60, 100)
    assert r["difference_pts"] == 10.0
    assert r["p_value"] == pytest.approx(0.1556, abs=0.002)


def test_onboarding_effect_is_positive(prepared):
    c, _ = prepared
    r = experiments.onboarding_effect(c)
    assert r["adjusted"]["odds_ratio"] > 1 and r["adjusted"]["p_value"] < 0.05
