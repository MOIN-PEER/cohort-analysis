"""Paths, dates and business settings."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
SQL_DIR = ROOT / "sql"
RESULTS = ROOT / "outputs" / "results"
FIGURES = ROOT / "outputs" / "figures"
INTERACTIVE = ROOT / "outputs" / "interactive"

SEED = 2026
FIRST_COHORT = pd.Period("2023-01", "M")
LAST_MONTH = pd.Period("2025-12", "M")

GROSS_MARGIN = 0.35                         # contribution per £ of revenue, used for LTV
ONBOARDING_CHANGE = pd.Period("2024-09", "M")  # new onboarding flow for cohorts from this month
PRICE_RISE = pd.Period("2025-04", "M")         # +£3 per box for everyone from this month

COLORS = {"blue": "#2f6db5", "orange": "#f0a830", "green": "#2e8b57", "red": "#c8453b", "grey": "#8a96a8",
          "purple": "#6b4fa0", "soft": "#9dbbe0"}
CHANNEL_COLORS = {"Organic": "#2e8b57", "Referral": "#6b4fa0", "Search": "#2f6db5", "Paid social": "#f0a830",
                  "Influencer": "#c8453b"}
