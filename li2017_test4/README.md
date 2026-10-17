# Fourth test (pre-specified analysis plan, not registered) in the attention model of binocular rivalry of Li et al. (2017)

Analysis plan: `analysis_plan_li2017_test4.md`, committed with the code before the run (not registered on OSF).

The port `li2017.py` and `run_li2017_test3.py` are those of `../li2017_test3/`, unchanged. The original
MATLAB code of Li et al. (2017) is at https://archive.nyu.edu/handle/2451/38721 and is licensed CC BY-SA
3.0; its README is included as `README_original_Li2017.ME`, and this port is distributed under the same
licence.

| File | Role |
|---|---|
| `run_li2017_test4.py` | the simulation: `python run_li2017_test4.py --workers 2` (about 5 h on 2 cores; more workers is faster and does not change the results) |
| `analyse_li2017_test4.py` | the pre-specified analysis: `python analyse_li2017_test4.py` |
| `sim_noise.py` | `li2017.simulate` without increment, recording the noise on grating 1's input (checked identical at start-up) |
| `run_li2017_test.py`, `run_li2017_test3.py` | earlier tests' scripts, used for their helpers (unchanged) |
| `dry4.json`, `dry4_tests.json` | dry run with every increment zero, seed 104, 4 configurations, 30 s runs |
| `pilot/` | the exploratory pilot described in Section 1 of the plan (run on the third test's configurations; needs `../../li2017_test3/li2017_test3_results.json`), and the analysis run on the pilot's data in this test's format |
| `code_sha256.txt` | checksums of the code and plan at the commit that precedes the run |
