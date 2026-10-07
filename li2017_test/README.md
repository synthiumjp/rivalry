# Pre-registered test in the Li et al. (2017) attention model of binocular rivalry

`li2017.py` is a Python port of the MATLAB model of Li H-H, Rankin J, Rinzel J, Carrasco M and
Heeger DJ (2017), Attention Model of Binocular Rivalry, PNAS 114, E6192-E6201. The original code is
at https://archive.nyu.edu/handle/2451/38721 and is licensed CC BY-SA 3.0; its README is included
here as `README_original_Li2017.ME`, as the licence requires, and this port is distributed under
the same licence. Please cite Li et al. (2017) when using it.

- `validate_conditions.py`  reproduces the published conditions 1-4 (no noise).
- `validate_noise.py`       dominance statistics under input noise.
- `validate_levelt.py`      modified Levelt propositions at baseline.
- `run_li2017_test.py`      the registered test (`prereg_li2017_test.md`).

    python run_li2017_test.py --dry-run --n-config 4 --chunk 2 --t-ms 30000   # zero-increment check
    python run_li2017_test.py                                                # registered run
