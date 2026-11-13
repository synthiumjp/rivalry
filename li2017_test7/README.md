# Seventh test: the gated lengthening where successive durations are as weakly correlated as observers'

Analysis plan: `analysis_plan_li2017_test7.md` (committed with this code before the run).

    python run_li2017_test7.py --workers 6       # several hours; writes li2017_test7_results.json
    python analyse_li2017_test7.py               # writes li2017_test7_tests.json

Configuration acceptance runs in the main process (serially) before the conditions are simulated in parallel; the
number of workers does not affect the results. A partial results file (`li2017_test7_results.json.partial.json`) is
written after each model.

Files: `run_li2017_test7.py` (simulation: masked self-replay and an acceptance rule with a serial-correlation
criterion), `analyse_li2017_test7.py` (pre-specified analysis), and, unchanged from the sixth test, `li2017.py`,
`li2017_t6.py`, `rate_models.py`, `run_li2017_test.py`, `run_li2017_test3.py`, `run_li2017_test6.py`,
`README_original_Li2017.ME` (licence of the original MATLAB code). `dry7.json`, `dry7_tests.json`: zero-increment dry
run and its analysis. `synth7.json`, `synth7_tests.json`: the analysis checked on a copy of the dry run with imposed
effects. `pilot/`: the baseline-only (zero increment) simulations that set the ranges. `precheck7.md`: the pre-run
review. `code_sha256.txt`: checksums.
