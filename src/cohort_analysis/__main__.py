"""Command line:

    python -m cohort_analysis data      # generate data/raw/*.csv
    python -m cohort_analysis analyse   # cohorts, LTV, sBG model, onboarding test -> outputs/
    python -m cohort_analysis all       # both
"""

import sys


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd not in {"data", "analyse", "analyze", "all"}:
        sys.exit(__doc__)
    if cmd in ("data", "all"):
        from .generate_data import generate, save
        tables = generate()
        save(tables)
        print("Generated data/raw: " + ", ".join(f"{k} ({len(v):,} rows)" for k, v in tables.items()))
    if cmd in ("analyse", "analyze", "all"):
        from .reporting import run_all
        s = run_all()["summary"]
        print(f"Month-12 subscription retention {s['subscription_survival']['month_12']}% | "
              f"sBG backtest error {s['sbg_backtest']['sbg_mape_pct']}% | "
              f"onboarding +{s['onboarding']['adjusted']['adjusted_difference_pts']} pts | SQL matches pandas: "
              f"{s['sql_matches_pandas']}")
        print("Results: outputs/results, outputs/figures, outputs/interactive/cohort_dashboard.html")


if __name__ == "__main__":
    main()
