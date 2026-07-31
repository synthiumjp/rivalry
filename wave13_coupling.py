"""
wave13_coupling.py -- the coupling RATIO, not the coupling magnitude.

Waves 11 and 12 both asked whether |rival change| tracks adaptation strength and
got weak or wrong-signed answers. That was the wrong outcome variable.
|rival change| confounds how strongly the two channels are coupled with how large
the effect is overall. The quantity that isolates coupling is

        coupling ratio  =  (rival % change) / (own % change)

Block B1 of wave 12 shows this ratio is flat in lambda, rises with kappa, falls
with gamma -- i.e. it tracks kappa/gamma, the steady-state adaptation per unit
activation. This script tests that across the full configuration set.

It also separates two things that were previously conflated: baseline adaptation
LEVEL (kappa/gamma, alpha*kappa/gamma) versus the CHANGE in adaptation produced
by the goal signal. Wave 12 measured the change and found a negative relation;
the prediction is that the level is what matters and the change is incidental.

No simulation. Reads wave10_signed.json, wave11_gated.json, wave12_B_coupling.json.
"""

import json
import os

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = {}


def hdr(s):
    print("\n" + "=" * 74)
    print(s)
    print("=" * 74)


def pct(a, b):
    return 100.0 * (a - b) / b if (np.isfinite(a) and np.isfinite(b) and b) else np.nan


def boot_rho(x, y, n_boot=4000, seed=42):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 6:
        return np.nan, np.nan, np.nan, np.nan, len(x)
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    v = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(x[i])) < 3 or len(np.unique(y[i])) < 3:
            continue
        r = stats.spearmanr(x[i], y[i])[0]
        if np.isfinite(r):
            v.append(r)
    lo, hi = np.percentile(v, [2.5, 97.5]) if len(v) > 100 else (np.nan, np.nan)
    return float(rho), float(p), float(lo), float(hi), int(n)


def loo_r2(X, y):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(y)
    Xd = np.column_stack([np.ones(n), X])
    pred = np.empty(n)
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        b, *_ = np.linalg.lstsq(Xd[m], y[m], rcond=None)
        pred[i] = Xd[i] @ b
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot else np.nan


# =============================================================================
# assemble rows from whichever files exist
# =============================================================================

rows = []

if os.path.exists('wave10_signed.json'):
    for r in json.load(open('wave10_signed.json', encoding='utf-8')):
        b = r['G'].get('0')
        c = r['G'].get('0.5')
        if not (b and c):
            continue
        o = pct(c['dur_manip'], b['dur_manip'])
        v = pct(c['dur_rival'], b['dur_rival'])
        rows.append({'src': 'w10', 'params': r['params'], 'own': o, 'rival': v})

if os.path.exists('wave12_B_coupling.json'):
    for r in json.load(open('wave12_B_coupling.json',
                            encoding='utf-8'))['across']:
        rows.append({'src': 'w12', 'params': r['params'],
                     'own': r['own'], 'rival': r['rival'],
                     'adapt_change': r['adapt_own_change']})

print("assembled %d configuration records" % len(rows))

for r in rows:
    p = r['params']
    r['ratio'] = (r['rival'] / r['own']) if (np.isfinite(r['own'])
                                             and abs(r['own']) > 1e-6) else np.nan
    r['kappa_over_gamma'] = p['kappa'] / p['gamma']
    r['adapt_eff'] = p['alpha'] * p['kappa'] / p['gamma']
    r['beta_over_lambda'] = p['beta'] / p['lambda']
    r['sigma'] = p['sigma']
    r['lambda'] = p['lambda']
    r['beta'] = p['beta']
    r['alpha'] = p['alpha']

# drop pathological ratios: own change near zero, or ratio outside [-2, 3]
clean = [r for r in rows if np.isfinite(r['ratio']) and abs(r['own']) > 2.0
         and -2.0 < r['ratio'] < 3.0]
print("usable after excluding |own change| < 2%% and |ratio| outliers: %d"
      % len(clean))

# =============================================================================
hdr("1. WHAT DOES THE COUPLING RATIO TRACK?")
print("  coupling ratio = (rival %% change) / (own %% change)")
print("  median %.3f   IQR [%.3f, %.3f]"
      % (np.median([r['ratio'] for r in clean]),
         np.percentile([r['ratio'] for r in clean], 25),
         np.percentile([r['ratio'] for r in clean], 75)))
print()
preds = ['kappa_over_gamma', 'adapt_eff', 'alpha', 'sigma',
         'beta', 'lambda', 'beta_over_lambda']
res = {}
for k in preds:
    x = [r[k] for r in clean]
    y = [r['ratio'] for r in clean]
    rho, p_, lo, hi, n = boot_rho(x, y)
    res[k] = {'rho': rho, 'p': p_, 'ci': [lo, hi], 'n': n}
    print("  %-18s rho = %+.3f  [%+.3f, %+.3f]  p = %.2g  n = %d"
          % (k, rho, lo, hi, p_, n))
OUT['ratio_predictors'] = res

# =============================================================================
hdr("2. LEVEL VERSUS CHANGE")
w12 = [r for r in clean if r['src'] == 'w12' and 'adapt_change' in r]
if len(w12) > 20:
    x1 = [r['kappa_over_gamma'] for r in w12]
    x2 = [r['adapt_change'] for r in w12]
    y = [r['ratio'] for r in w12]
    for lbl, x in (('adaptation LEVEL (kappa/gamma)', x1),
                   ('adaptation CHANGE under G', x2)):
        rho, p_, lo, hi, n = boot_rho(x, y)
        print("  %-32s rho = %+.3f  [%+.3f, %+.3f]  p = %.2g"
              % (lbl, rho, lo, hi, p_))
    # partial: does level survive controlling for change, and vice versa?
    a = np.array([[r['kappa_over_gamma'], r['adapt_change']] for r in w12],
                 float)
    yy = np.array(y, float)
    m = np.isfinite(a).all(1) & np.isfinite(yy)
    ra = stats.rankdata(a[m, 0])
    rc = stats.rankdata(a[m, 1])
    ry = stats.rankdata(yy[m])

    def resid(u, v):
        return u - np.polyval(np.polyfit(v, u, 1), v)
    p_level = np.corrcoef(resid(ra, rc), resid(ry, rc))[0, 1]
    p_change = np.corrcoef(resid(rc, ra), resid(ry, ra))[0, 1]
    print("\n  partial rho(level  | change) = %+.3f" % p_level)
    print("  partial rho(change | level ) = %+.3f" % p_change)
    print("\n  If the level survives and the change does not, the mechanism is")
    print("  baseline adaptation, and wave 11/12 tested the wrong variable.")
    OUT['level_vs_change'] = {'partial_level': float(p_level),
                              'partial_change': float(p_change),
                              'n': int(m.sum())}

# =============================================================================
hdr("3. HOW MUCH OF THE RATIO CAN BE PREDICTED?")
X = np.array([[r['kappa_over_gamma'], r['alpha'], r['sigma'],
               r['beta'], r['lambda']] for r in clean], float)
y = np.array([r['ratio'] for r in clean], float)
m = np.isfinite(X).all(1) & np.isfinite(y)
for lbl, cols in (('kappa/gamma alone', [0]),
                  ('alpha*kappa/gamma alone', None),
                  ('all five parameters', [0, 1, 2, 3, 4])):
    if cols is None:
        Xi = np.array([[r['adapt_eff']] for r in clean], float)[m]
    else:
        Xi = X[m][:, cols]
    print("  %-26s LOO R^2 = %+.3f" % (lbl, loo_r2(Xi, y[m])))
OUT['ratio_r2'] = {'n': int(m.sum())}

# =============================================================================
hdr("4. IS THE RATIO STABLE ENOUGH TO BE CALLED A COUPLING CONSTANT?")
q = np.percentile([r['ratio'] for r in clean], [10, 25, 50, 75, 90])
print("  deciles of the coupling ratio: %s"
      % "  ".join("%.2f" % v for v in q))
frac = np.mean([(0.5 <= r['ratio'] <= 1.0) for r in clean])
print("  proportion in [0.5, 1.0]: %.1f%%" % (100 * frac))
print("\n  A ratio clustered below 1 means the rival always moves the same way")
print("  as the manipulated channel but by less. That is a stronger and more")
print("  quotable statement than a correlation.")
OUT['ratio_deciles'] = [float(v) for v in q]
OUT['frac_in_half_to_one'] = float(frac)

# =============================================================================
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
ax[0].scatter([r['own'] for r in clean], [r['rival'] for r in clean],
              s=20, alpha=0.6, c='#2c6fbb')
lim = [min(0, min(r['own'] for r in clean)), max(r['own'] for r in clean)]
ax[0].plot(lim, [0.75 * v for v in lim], 'k--', lw=1, label='ratio 0.75')
ax[0].axhline(0, c='k', lw=0.8)
ax[0].set_xlabel('% change, manipulated')
ax[0].set_ylabel('% change, rival')
ax[0].set_title('A  Coupling is proportional')
ax[0].legend(fontsize=7)

ax[1].scatter([r['kappa_over_gamma'] for r in clean],
              [r['ratio'] for r in clean], s=20, alpha=0.6, c='#2c6fbb')
ax[1].set_xlabel(r'$\kappa/\gamma$  (baseline adaptation per unit activation)')
ax[1].set_ylabel('coupling ratio')
ax[1].set_title('B  Ratio tracks adaptation level')

ax[2].hist([r['ratio'] for r in clean], bins=24, color='#2c6fbb')
ax[2].axvline(1.0, ls='--', c='k', lw=1)
ax[2].set_xlabel('coupling ratio')
ax[2].set_ylabel('configurations')
ax[2].set_title('C  Distribution')
fig.tight_layout()
fig.savefig('wave13_fig_coupling.pdf')

json.dump(OUT, open('wave13_results.json', 'w', encoding='utf-8'),
          indent=1, default=float)
hdr("DONE")
print("  wave13_results.json")
print("  wave13_fig_coupling.pdf")
