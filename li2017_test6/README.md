# Sixth test: what the gated lengthening is, what it depends on, and whether it generalises

Analysis plan: `analysis_plan_li2017_test6.md` (committed with this code before the run).

    python run_li2017_test6.py --workers 4       # about 2-3 hours; writes li2017_test6_results.json
    python analyse_li2017_test6.py               # writes li2017_test6_tests.json

Each worker needs about 1.2 GB of memory; the number of workers does not affect the results.

Files: `run_li2017_test6.py` (simulation), `analyse_li2017_test6.py` (pre-specified analysis), `li2017_t6.py` (the
port's simulate() with an attention-layer option; identical to `li2017.py` for input increments), `rate_models.py`
(two Laing-Chow rate models), and, unchanged from earlier tests, `li2017.py`, `run_li2017_test.py`,
`run_li2017_test3.py`, `README_original_Li2017.ME` (licence of the original MATLAB code). `dry6.json`,
`dry6_tests.json`: zero-increment dry run and its analysis. `code_sha256.txt`: checksums.
