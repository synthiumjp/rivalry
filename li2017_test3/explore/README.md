# Exploratory analyses of the third test (not registered)

| Script | Output | What it does |
|---|---|---|
| `followup.py` | `followup_summary.json` | H14 and H21 under the two scorings the registered script omitted; H21 against the assumed withdrawal offset (300 to 900 ms) with a constant comparator; the fraction of episodes the observer reports; decomposition of H19 |
| `followup2.py` | `followup2_summary.json` | H21 with re-anchored first-order curves; every hypothesis under the passive observer; the matched-noise comparison of adaptation |
| `finite_accounts.py` | `finite_accounts_summary_e.json`, `_u.json` | finite-effect and first-order accounts calibrated on continuous delivery, compared on the window profile, the second test's windows and delayed-withdrawal curve, and the hold curve (Table S16) |

Run from this folder: `python followup.py ../li2017_test3_results.json`, `python followup2.py ../li2017_test3_results.json ../li2017_test3_tests.json`, `python finite_accounts.py ../li2017_test3_results.json ../../li2017_test2/li2017_test2_results.json e` (and `u`).
