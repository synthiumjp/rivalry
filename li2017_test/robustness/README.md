# Robustness re-analysis of the registered Li et al. (2017) test (exploratory, not registered)

Run from this folder. The re-run reproduces the registered summaries exactly (same configuration
batches and seeds, so identical noise), then computes alternative measures.

| Script | What it does | Output |
|---|---|---|
| `robustness_li2017.py` | re-runs baseline and gated, ungated, yoked and delayed-withdrawal 0.25 T conditions (pass 1), and delayed-onset 0.25, 0.5 and 1.0 T (pass 2, `ROBUST_PASS=2`); per instance: registered measures, interval between attended episodes, merged competitor episodes, censored-inclusive means, A-to-B transition time, episode counts | `robust_results_pass1.json`, `robust_results_pass2.json` |
| `check_and_summarise.py` | checks exact reproduction of the registered summaries; summarises pass 1 | `robust_summary.json` |
| `diag.py` | counts brief excursions of the attended grating at baseline and under the gate (six configurations) | printed |
| `json_analyses.py` | bootstrap intervals, trend tests for H4 and H5, per-dose effects, H6 partial correlations (from the registered results file) | `json_analyses.json` |
| `regime.py`, `strat.py` | noise-free classification of each configuration's regime, and results by regime | `regime.json`, `strat.json` |

Reported in the supplementary material, Section S14.
