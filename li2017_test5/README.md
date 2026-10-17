# Fifth test (pre-specified analysis plan, not registered) in the attention model of binocular rivalry of Li et al. (2017)

Analysis plan: `analysis_plan_li2017_test5.md`, committed with the code before the run (not registered on OSF).

`li2017.py`, `run_li2017_test.py`, `run_li2017_test3.py`, `run_li2017_test4.py` and `sim_noise.py` are those of
`../li2017_test4/`, unchanged. The original MATLAB code of Li et al. (2017) is at
https://archive.nyu.edu/handle/2451/38721 (CC BY-SA 3.0); its README is included as `README_original_Li2017.ME`,
and this port is distributed under the same licence.

| File | Role |
|---|---|
| `run_li2017_test5.py` | the simulation: `python run_li2017_test5.py --workers 6` (more workers is faster and does not change the results) |
| `analyse_li2017_test5.py` | the pre-specified analysis: `python analyse_li2017_test5.py` |
| `dry5.json`, `dry5_tests.json` | dry run with every increment zero, seed 105, 4 configurations, 30 s runs |
| `codecheck5.json`, `codecheck5_tests.json` | the dry run with four of the fourth test's kernels substituted, to check the kernel-prediction code |
| `precheck/` | scripts of the independent check of an earlier version of the code (Section 1 of the plan) |
| `code_sha256.txt` | checksums of the code and plan at the commit that precedes the run |
