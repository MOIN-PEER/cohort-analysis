#!/usr/bin/env bash
# macOS/Linux: generate data, run every analysis and the tests
set -e
source .venv/bin/activate
python -m cohort_analysis all
pytest
