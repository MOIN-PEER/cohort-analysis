"""Run the SQL in sql/ against an in-memory SQLite copy of the CSV data."""

from __future__ import annotations

import sqlite3

import pandas as pd

from .config import SQL_DIR


def connect(tables: dict[str, pd.DataFrame]) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    for name, df in tables.items():
        df.to_sql(name, con, index=False)
    return con


def run(con: sqlite3.Connection, name: str) -> pd.DataFrame:
    return pd.read_sql_query((SQL_DIR / f"{name}.sql").read_text(), con)


def retention_matrix_sql(con: sqlite3.Connection) -> pd.DataFrame:
    r = run(con, "cohort_retention")
    m = r.pivot(index="cohort", columns="age", values="retention_pct")
    m.columns.name = None
    return m
