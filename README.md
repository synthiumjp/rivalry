# Delivery timing in binocular rivalry: code and data

Code and simulation data for Cacioli & Marmo, *When does a contrast increment mimic attention in binocular rivalry?
When it tracks the percept*. Registrations, analysis plans and data are also on the Open Science Framework:
https://osf.io/kqdgb/.

## What is here

| Folder or files | What it holds | Paper |
|---|---|---|
| `li2017_test/` | Python port of Li, Rankin, Rinzel, Carrasco & Heeger's (2017) model, its validation, and the first registered test | Sections 3.2, 4.1; S9, S10 |
| `li2017_test2/`, `li2017_test3/` | the second and third registered tests | Sections 4.1, 4.2, 4.5, 4.6; S11, S12, S14 |
| `li2017_test4/` | the baseline-noise kernel (pre-specified), with its pilot | Sections 4.2, 4.3; S16 |
| `li2017_test5/` | the split gate, the switch itself, brief pulses (pre-specified) | Section 4.4; S17 |
| `li2017_sens/` | the sampling check (pre-specified) | Section 4.1; S18 |
| `li2017_test6/` | distribution, self-replay, graded withdrawal, attention layer, serial correlation, two rate models (pre-specified) | Sections 4.1, 4.3 to 4.5; S19 |
| `li2017_test7/` | observer-like serial correlation, masked self-replay, the serial-correlation predictor (pre-specified) | to be added |
| top-level `*.py` | the goal-conditioned leaky competing accumulator (GC-LCA) of the original registered study, its analyses and figures | Sections 3.2, 4.1, 4.3, 4.5; S1 to S5, S8, S13; companion report |

Each test folder has its own README with the commands to run it, its analysis plan, checksums of the code as committed
before the run, a zero-increment dry run, and its results. The folder for each test is self-contained: modules shared
between tests are copied unchanged into each folder.

## Requirements

Python 3.12 or 3.13 with NumPy and SciPy (the GC-LCA scripts also need Numba and Matplotlib). The versions used are
recorded in each results file.

## Licences

- The port of Li et al.'s model (`li2017.py`, `li2017_t6.py`, and the copies of them in each test folder) is derived from
  the published MATLAB implementation (archive.nyu.edu/handle/2451/38721) and is distributed under that
  implementation's licence, CC BY-SA 3.0 (see `README_original_Li2017.ME` in each folder).
- All other code is released under the MIT licence (`LICENSE`).

## Errata

See `ERRATA.md` for dated corrections to committed files.

## Citation

Cacioli, J.-P., & Marmo, C. (2026). *When does a contrast increment mimic attention in binocular rivalry? When it
tracks the percept.* Manuscript submitted for publication. Data and registrations: https://osf.io/kqdgb/.
