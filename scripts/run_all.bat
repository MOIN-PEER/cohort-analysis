@echo off
REM Windows: generate data, run every analysis and the tests
call .venv\Scripts\activate
python -m cohort_analysis all
pytest
