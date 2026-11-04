import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cohort_analysis import cohorts  # noqa: E402
from cohort_analysis.generate_data import generate  # noqa: E402


@pytest.fixture(scope="session")
def data():
    """Generate the dataset in memory (never touches data/raw)."""
    t = generate()
    # Round-trip through CSV types: the real pipeline reads CSVs, so dtypes must match.
    import io
    import pandas as pd
    return {k: pd.read_csv(io.StringIO(v.to_csv(index=False))) for k, v in t.items()}


@pytest.fixture(scope="session")
def prepared(data):
    return cohorts.prepare(data["customers"], data["orders"])
