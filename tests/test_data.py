"""Integrity of the generated subscription data."""

import pandas as pd


def test_offer_labels_survive_csv_round_trip(data):
    # Regression test: a category called "None" is read back by pandas as missing.
    assert data["customers"]["first_box_offer"].notna().all()
    assert set(data["customers"]["first_box_offer"]) == {"No offer", "50% off first box", "First box free"}


def test_every_customer_has_a_first_order_in_signup_month(data):
    first = data["orders"].groupby("customer_id")["order_month"].min()
    c = data["customers"].set_index("customer_id")
    assert (first.reindex(c.index) == c["signup_month"]).all()


def test_no_orders_after_cancellation(data):
    o = data["orders"].merge(data["customers"][["customer_id", "cancelled_month"]], on="customer_id")
    o = o[o["cancelled_month"].notna()]
    assert (pd.PeriodIndex(o["order_month"], freq="M") < pd.PeriodIndex(o["cancelled_month"], freq="M")).all()


def test_revenue_adds_up(data):
    o = data["orders"]
    assert ((o["box_revenue"] + o["addon_revenue"] - o["revenue"]).abs() < 0.011).all()
    assert (o["box_revenue"] <= o["box_list_price"] + 1e-9).all()


def test_spend_matches_new_customers(data):
    s = data["marketing_spend"].groupby("channel")["new_customers"].sum()
    c = data["customers"].groupby("channel").size()
    assert (s.sort_index() == c.sort_index()).all()
