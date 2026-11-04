# Project summary: Cohort Analysis

**Author:** Mohamed Moin Peer  |  **Type:** Customer analytics  |  **Stack:** Python, pandas, SQL (SQLite), SciPy, statsmodels, matplotlib, Plotly, pytest

## Problem
A subscription business needs to know how long customers stay, what they are worth, which channels and offers pay
back, and how new cohorts will perform before it has months of data on them.

## What I built
* A **synthetic subscription dataset**: 17,818 customers, 134,773 boxes and monthly marketing spend for 2023-2025,
  with known effects built in.
* **Cohort retention matrices** (activity, subscription survival, revenue per customer, NRR) in pandas, with the
  same matrix in **SQL**, and a test that they match exactly.
* **Unit economics:** contribution, CAC, LTV:CAC and payback by channel; economics of first-box offers.
* A **shifted-beta-geometric model** (maximum likelihood with SciPy), backtested on held-out months.
* An **evaluation of an onboarding change**: two-proportion z-test plus a logistic regression adjusting for customer mix.
* **Outputs:** 9 charts, an interactive Plotly dashboard, CSV/JSON tables, generated insights, an executed notebook
  and 16 tests.

## Results
* 40.3% of customers are still subscribed after 12 months; revenue at month 12 is 56.6% of month 1 thanks to add-ons.
* "First box free" customers: 56.6% stay past month 1 (vs 85.9%); £37 vs £123 contribution after 12 months.
* LTV:CAC ranges from **16.6x** (organic) to **1.3x** (influencer, payback month 9).
* The price rise doubled established-subscriber churn for one month (9.85% vs 4.93%, ~293 extra cancellations).
* New onboarding: **+5.3 points** month-3 retention after adjusting for mix (OR 1.28, 95% CI 1.18-1.38).
* sBG forecast error **7.2%** vs **59.9%** for a constant-churn model.

## Skills demonstrated
Cohort and retention analysis · SQL (CTEs, joins, date arithmetic) · customer lifetime value and unit economics ·
probabilistic modelling (maximum likelihood) · statistical testing and confounding · data visualisation · testing.

## How to run
`pip install -r requirements.txt && pip install -e . && python -m cohort_analysis all`
