"""Generate a realistic *synthetic* dataset for "Crate & Co", a fictional UK meal-kit subscription.

Writes three CSV files to data/raw/:

  customers.csv        one row per subscriber: signup month, channel, plan, first-box offer, cancellation month
  orders.csv           one row per box delivered (plus any add-ons)
  marketing_spend.csv  acquisition spend per channel per month

Built-in effects (so the analysis has a known truth to find):
  * customers differ in how likely they are to cancel (heterogeneous churn), so retention
    curves flatten over time - the pattern the shifted-beta-geometric model describes;
  * "first box free" customers often cancel straight after the free box;
  * referral and organic customers stay longest; influencer and paid-social customers churn most;
  * a new onboarding flow from September 2024 lowers churn for later cohorts;
  * a £3 price rise in April 2025 causes a one-off spike in cancellations (a calendar effect,
    visible as a diagonal in the cohort heatmap);
  * loyal customers buy more add-ons over time (revenue expansion).
All customers are fictional.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FIRST_COHORT, LAST_MONTH, ONBOARDING_CHANGE, PRICE_RISE, RAW, SEED

CHANNELS = {  # base share, churn multiplier, CAC (£)
    "Organic":     (0.20, 0.80, 6.0),
    "Search":      (0.25, 1.00, 38.0),
    "Paid social": (0.27, 1.20, 46.0),
    "Referral":    (0.13, 0.65, 18.0),
    "Influencer":  (0.15, 1.35, 58.0),
}
PLANS = {"Classic (2 people)": (34.99, 0.75), "Family (4 people)": (54.99, 0.25)}
OFFERS = {"No offer": (0.0, 0.40, 1.00), "50% off first box": (0.5, 0.40, 1.20), "First box free": (1.0, 0.20, 1.35)}
REGIONS = {"London": 0.22, "South East": 0.17, "North West": 0.13, "Midlands": 0.14, "Scotland": 0.10,
           "Northern Ireland": 0.05, "Wales": 0.06, "South West": 0.13}
SEASON = {1: 1.45, 2: 1.10, 3: 1.0, 4: 0.95, 5: 0.9, 6: 0.85, 7: 0.8, 8: 0.85, 9: 1.15, 10: 1.05, 11: 0.95, 12: 0.7}


def generate(seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    months = pd.period_range(FIRST_COHORT, LAST_MONTH, freq="M")
    ch_names = list(CHANNELS)
    cust_rows, order_rows, spend_rows = [], [], []
    cid = 0
    for mi, m in enumerate(months):
        n_new = rng.poisson(320 * SEASON[m.month] * (1.025 ** mi))
        share = np.array([CHANNELS[c][0] for c in ch_names])
        share[ch_names.index("Influencer")] *= 1 + 0.03 * mi        # influencer marketing grows over time
        share /= share.sum()
        channel = rng.choice(ch_names, n_new, p=share)
        # Offer mix: heavy "first box free" pushes in January and September campaigns
        offer_p = np.array([OFFERS[o][1] for o in OFFERS])
        if m.month in (1, 9):
            offer_p = np.array([0.25, 0.35, 0.40])
        offer = rng.choice(list(OFFERS), n_new, p=offer_p / offer_p.sum())
        plan = rng.choice(list(PLANS), n_new, p=[PLANS[p][1] for p in PLANS])
        region = rng.choice(list(REGIONS), n_new, p=list(REGIONS.values()))
        for c in ch_names:
            k = int((channel == c).sum())
            spend_rows.append((str(m), c, k, round(k * CHANNELS[c][2] * rng.uniform(0.9, 1.1), 2)))

        for i in range(n_new):
            cid += 1
            mean_churn = 0.16 * CHANNELS[channel[i]][1] * OFFERS[offer[i]][2]
            mean_churn *= 0.85 if plan[i].startswith("Family") else 1.0
            mean_churn *= 0.78 if m >= ONBOARDING_CHANGE else 1.0
            k = 2.2                                                  # lower = more varied customers
            p = rng.beta(mean_churn * k, (1 - mean_churn) * k)       # this customer's monthly churn probability
            base_price = PLANS[plan[i]][0]
            addon_rate = rng.beta(2, 6)                              # appetite for add-ons
            day = int(rng.integers(1, 29))
            age, cancelled = 0, None
            cur = m
            while cur <= LAST_MONTH:
                price = base_price + (3.0 if cur >= PRICE_RISE else 0.0)
                if age == 0:
                    order_rows.append((cid, str(cur), f"{cur}-{day:02d}", round(price, 2),
                                       round(price * (1 - OFFERS[offer[i]][0]), 2), 0.0))
                elif rng.random() > 0.06:                            # 6% of months skipped (paused)
                    addon = 0.0
                    if rng.random() < min(0.9, addon_rate * (1 + 0.04 * age)):
                        addon = round(float(rng.uniform(6, 22)), 2)
                    order_rows.append((cid, str(cur), f"{cur}-{day:02d}", round(price, 2), round(price, 2), addon))
                # Decide whether they cancel before next month's box
                hazard = p
                if age == 0 and offer[i] == "First box free":
                    hazard = 1 - (1 - p) * (1 - 0.30)                # many leave once the free box arrives
                if cur + 1 == PRICE_RISE:
                    hazard = 1 - (1 - hazard) * (1 - 0.06)           # price-rise announcement
                if rng.random() < hazard:
                    cancelled = cur + 1
                    break
                cur += 1
                age += 1
            cust_rows.append((cid, str(m), f"{m}-{day:02d}", channel[i], plan[i], offer[i], region[i],
                              None if cancelled is None or cancelled > LAST_MONTH else str(cancelled)))

    customers = pd.DataFrame(cust_rows, columns=["customer_id", "signup_month", "signup_date", "channel", "plan",
                                                 "first_box_offer", "region", "cancelled_month"])
    customers["customer_id"] = customers["customer_id"].map(lambda x: f"CU{x:06d}")
    orders = pd.DataFrame(order_rows, columns=["customer_id", "order_month", "order_date", "box_list_price",
                                               "box_revenue", "addon_revenue"])
    orders["customer_id"] = orders["customer_id"].map(lambda x: f"CU{x:06d}")
    orders.insert(0, "order_id", [f"OR{i:07d}" for i in range(1, len(orders) + 1)])
    orders["revenue"] = (orders["box_revenue"] + orders["addon_revenue"]).round(2)
    spend = pd.DataFrame(spend_rows, columns=["month", "channel", "new_customers", "spend"])
    return {"customers": customers, "orders": orders, "marketing_spend": spend}


def save(tables: dict[str, pd.DataFrame]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(RAW / f"{name}.csv", index=False)


def load() -> dict[str, pd.DataFrame]:
    if not (RAW / "customers.csv").exists():
        save(generate())
    return {n: pd.read_csv(RAW / f"{n}.csv") for n in ("customers", "orders", "marketing_spend")}
