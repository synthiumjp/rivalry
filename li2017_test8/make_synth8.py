"""Synthetic data for checking analyse_li2017_test8.py (analysis plan, Section 7). Per configuration and size s, the
attended grating's observer-scored duration changes by k s + q s^2 (q = 0 is linear) and the competitor's by
c1 s + c2 s^2, with seed-level noise; a fraction of configurations is made observer-invalid (baseline ratio of reported
to merged competitor episodes 0.5) and a fraction of values is set to NaN. The true calibrated competitor change is the
exact (continuous) solution, so the check also measures the bias of interpolating between sizes.
Scenarios:
  lengthen   linear, competitor lengthens; all configurations valid; no NaN
  shorten    linear, competitor shortens; all valid; no NaN
  convex     convex attended response, competitor not proportional to it; 30% invalid (90% in 'dep', so that C2 and
             C3 are not tested there); about a third never reach 29%; some NaN
    python make_synth8.py   # writes synth8_<scenario>.json, synth8_<scenario>_ref7.json and prints the truth"""
import json
import numpy as np

SIZES = (0.025, 0.05, 0.075, 0.10, 0.20)
KEYS = [f'chong_{s:g}' for s in SIZES]
SC = ('mA', 'mB', 'mBabs', 'mBm', 'nBm', 'nB', 'nBabs', 'mOA', 'mOB', 'nOB', 'mOA2', 'mOB2', 'nOB2', 'rB_dom',
      'area_comp', 'dose_comp', 'dose_att')


def s_star(k, q):
    """Size at which k s + q s^2 = 0.29 (fraction), the smallest positive root."""
    if q == 0:
        return 0.29 / k
    disc = k * k + 4 * q * 0.29
    return (-k + np.sqrt(disc)) / (2 * q)


def make(scenario, seed):
    rng = np.random.default_rng(seed)
    models, ref, truth = {}, {}, {}
    for model in ('li', 'sfa', 'dep'):
        n = 60
        if scenario == 'convex':
            k = rng.uniform(0.1, 2.0, n); q = rng.uniform(2.0, 10.0, n)     # attended at 20%: 10% to 80%
            c1 = rng.normal(0.3, 0.2, n); c2 = rng.normal(-1.0, 0.5, n)
            invalid = rng.random(n) < (0.9 if model == 'dep' else 0.3)   # dep: too few valid, C2/C3 not tested
            p_nan = 0.02
        else:
            k = rng.uniform(1.0, 12.0, n); q = np.zeros(n)
            c1 = rng.normal(0.6 if scenario == 'lengthen' else -0.6, 0.3, n); c2 = np.zeros(n)
            invalid = np.zeros(n, bool)
            p_nan = 0.0
        base, cond, g7 = [], {kk: [] for kk in KEYS}, []
        nanmask = rng.random((n, len(SIZES))) < p_nan
        for i in range(n):
            T = rng.uniform(500, 5000)
            b = [{s: float(T * rng.uniform(0.95, 1.05)) for s in SC} for _ in range(32)]
            for x in b:
                x['nBm'] = 20.0; x['nOB'] = 10.0 if invalid[i] else 19.0
            base.append(b)
            for j, (size, kk) in enumerate(zip(SIZES, KEYS)):
                fa = 1 + k[i] * size + q[i] * size ** 2
                fc = 1 + c1[i] * size + c2[i] * size ** 2
                nanv = float('nan')
                cond[kk].append([dict(x, mOA=(nanv if nanmask[i, j] else x['mOA'] * fa), mA=x['mA'] * fa,
                                      mOB=x['mOB'] * fc, mBm=x['mBm'] * fc, mB=x['mB'] * fc, mBabs=x['mBabs'] * fc,
                                      mOB2=x['mOB2'] * fc, mOA2=x['mOA2'] * fa) for x in b])
            g7.append([dict(x, mBm=x['mBm'] * 1.05) for x in b])
        # truth on the valid configurations that reach 29% by 20% and have no NaN up to the crossing
        tv = []
        for i in range(n):
            if invalid[i]:
                continue
            ss = s_star(k[i], q[i])
            if not (ss <= 0.20):
                continue
            jcross = int(np.searchsorted(SIZES, ss))      # the first grid size at or beyond the crossing
            if nanmask[i, :jcross + 1].any():
                continue
            tv.append(100 * (c1[i] * ss + c2[i] * ss ** 2))
        truth[model] = dict(median_calibrated_comp_valid=float(np.median(tv)), n_valid_calibrated=len(tv),
                            n_invalid=int(invalid.sum()))
        R = dict(base=base, cond=cond, T=[1000.0] * n, duty={kk: [0.5] * n for kk in KEYS},
                 level_in_comp={kk: [0.01] * n for kk in KEYS},
                 predominance=dict(base=[0.5] * n, **{kk: [0.5] * n for kk in KEYS}))
        models[model] = dict(accepted=[{}] * n, results=R)
        ref[model] = dict(results=dict(base=base, cond=dict(gated=g7)))
    json.dump(dict(dry_run=False, models=models, truth=truth), open(f'synth8_{scenario}.json', 'w'))
    json.dump(dict(models=ref), open(f'synth8_{scenario}_ref7.json', 'w'))
    print(scenario, json.dumps(truth))


make('lengthen', 8101)
make('shorten', 8102)
make('convex', 8103)
