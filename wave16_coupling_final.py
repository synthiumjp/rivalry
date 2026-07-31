"""
wave16_coupling_final.py -- close the two outstanding items.

ITEM 1  THE kappa/gamma RESULT NEEDS REPORTING AND A CONFOUND CHECK
    rho = +0.907 for kappa/gamma against the coupling ratio appears in the
    abstract, contributions and Discussion, and nowhere in Results. It also has a
    confound: gamma takes three grid values and kappa is defined as a multiple of
    alpha, so kappa/gamma is partly a three-level factor and partly a restatement
    of alpha. Required: the within-gamma correlation and a partial controlling
    for alpha.

ITEM 2  DOES SUPPRESSED-PHASE ACTIVATION SUBSUME IT?
    Section 4.7.1 shows that the attended channel's activation during the RIVAL's
    dominance episodes orders five formulations at rho = -0.970. If the same
    quantity also predicts the across-configuration variation in coupling within
    a single formulation, it does both jobs and kappa/gamma is redundant. That
    would let the paper carry one variable instead of two.

ITEM 3  BUILD THE CENTRAL FIGURE
    The rho = -0.970 relationship is now the paper's headline and has no figure.

No simulation. Reads wave15_A_paired.json.
"""

import json
import os
import sys

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET_OWN = 30.0
MODES = {1: 'persistence', 2: 'increment', 3: 'response gain',
         4: 'inhibitory gain', 5: 'adaptation'}
COLS = {1: '#2c6fbb', 2: '#d1495b', 3: '#4c9f70', 4: '#e9a13b', 5: '#8a4fbd'}
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


def partial_rho(x, y, z):
    """Spearman partial correlation of x and y controlling z."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 8:
        return np.nan
    rx, ry, rz = (stats.rankdata(v[m]) for v in (x, y, z))

    def res(u, w):
        return u - np.polyval(np.polyfit(w, u, 1), w)
    return float(np.corrcoef(res(rx, rz), res(ry, rz))[0, 1])


def loo_r2(X, y):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if X.ndim == 1:
        X = X[:, None]
    m = np.isfinite(X).all(1) & np.isfinite(y)
    X, y = X[m], y[m]
    n = len(y)
    if n < X.shape[1] + 4:
        return np.nan
    Xd = np.column_stack([np.ones(n), X])
    pred = np.empty(n)
    for i in range(n):
        k = np.ones(n, bool)
        k[i] = False
        b, *_ = np.linalg.lstsq(Xd[k], y[k], rcond=None)
        pred[i] = Xd[i] @ b
    ssr = float(((y - pred) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return 1 - ssr / sst if sst else np.nan


# =============================================================================

fn = 'wave15_A_paired.json'
if not os.path.exists(fn):
    sys.exit("ERROR: %s not found." % fn)
D = json.load(open(fn, encoding='utf-8'))
print("loaded %d configurations" % len(D))


def interp(lv, b, key):
    xs = np.array([pct(c['own'], b['own']) for c in lv], float)
    ys = np.array([pct(c[key], b[key]) for c in lv], float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 3:
        return np.nan
    o = np.argsort(xs[m])
    xx, yy = xs[m][o], ys[m][o]
    if not (xx.min() <= TARGET_OWN <= xx.max()):
        return np.nan
    return float(np.interp(TARGET_OWN, xx, yy))


rows = []
for rec in D:
    b = rec['baseline']
    if not (b['own'] and b['rival'] and b.get('supp_act')):
        continue
    p = rec['params']
    for mode in MODES:
        lv = rec['modes'][str(mode)]
        rv = interp(lv, b, 'rival')
        su = interp(lv, b, 'supp_act')
        if not (np.isfinite(rv) and np.isfinite(su)):
            continue
        rows.append({'grid_index': rec['grid_index'], 'mode': mode,
                     'rival': rv, 'supp': su,
                     'ratio': rv / TARGET_OWN,
                     'kg': p['kappa'] / p['gamma'],
                     'akg': p['alpha'] * p['kappa'] / p['gamma'],
                     'alpha': p['alpha'], 'gamma': p['gamma'],
                     'kappa': p['kappa'], 'beta': p['beta'],
                     'lam': p['lambda'], 'sigma': p['sigma']})
print("usable formulation x configuration records: %d" % len(rows))

# =============================================================================
hdr("1. THE ORDERING VARIABLE, POOLED AND WITHIN FORMULATION")
x = [r['supp'] for r in rows]
y = [r['rival'] for r in rows]
rho, p_, lo, hi, n = boot_rho(x, y)
print("  pooled across all five formulations")
print("    rho(suppressed-phase activation, rival) = %+.3f [%+.3f, %+.3f]  "
      "p = %.2g  n = %d" % (rho, lo, hi, p_, n))
OUT['pooled'] = {'rho': rho, 'ci': [lo, hi], 'n': n}
print("\n  within each formulation (across configurations)")
for mode, name in MODES.items():
    sub = [r for r in rows if r['mode'] == mode]
    r2, p2, l2, h2, n2 = boot_rho([r['supp'] for r in sub],
                                  [r['rival'] for r in sub])
    print("    %-18s rho = %+.3f [%+.3f, %+.3f]  p = %.2g  n = %d"
          % (name, r2, l2, h2, p2, n2))
    OUT.setdefault('within_mode', {})[name] = {'rho': r2, 'ci': [l2, h2],
                                               'n': n2}
print("\n  A strong pooled correlation with weak within-formulation ones would")
print("  mean the variable only distinguishes mechanisms, not configurations.")

# =============================================================================
hdr("2. THE kappa/gamma RESULT, WITH THE CONFOUND CHECKED")
pers = [r for r in rows if r['mode'] == 1]
print("  persistence modulation only, n = %d" % len(pers))
for lbl, k in (('kappa/gamma', 'kg'), ('alpha*kappa/gamma', 'akg'),
               ('alpha', 'alpha'), ('gamma', 'gamma'), ('kappa', 'kappa')):
    r2, p2, l2, h2, n2 = boot_rho([r[k] for r in pers],
                                  [r['ratio'] for r in pers])
    print("    rho(%-18s, coupling ratio) = %+.3f [%+.3f, %+.3f]  p = %.2g"
          % (lbl, r2, l2, h2, p2))
    OUT.setdefault('kg', {})[lbl] = {'rho': r2, 'ci': [l2, h2], 'p': p2}

print("\n  partial correlations")
pr_a = partial_rho([r['kg'] for r in pers], [r['ratio'] for r in pers],
                   [r['alpha'] for r in pers])
pr_s = partial_rho([r['kg'] for r in pers], [r['ratio'] for r in pers],
                   [r['supp'] for r in pers])
pr_k = partial_rho([r['supp'] for r in pers], [r['ratio'] for r in pers],
                   [r['kg'] for r in pers])
print("    kappa/gamma vs ratio, controlling alpha              = %+.3f" % pr_a)
print("    kappa/gamma vs ratio, controlling suppressed-phase   = %+.3f" % pr_s)
print("    suppressed-phase vs ratio, controlling kappa/gamma   = %+.3f" % pr_k)
OUT['partials'] = {'kg_given_alpha': pr_a, 'kg_given_supp': pr_s,
                   'supp_given_kg': pr_k}

print("\n  within each level of gamma (gamma is a three-level grid factor)")
for g in sorted(set(r['gamma'] for r in pers)):
    sub = [r for r in pers if r['gamma'] == g]
    if len(sub) < 8:
        print("    gamma = %.2f : n = %d, too few" % (g, len(sub)))
        continue
    r2, p2, l2, h2, n2 = boot_rho([r['kg'] for r in sub],
                                  [r['ratio'] for r in sub])
    print("    gamma = %.2f : rho = %+.3f [%+.3f, %+.3f]  p = %.2g  n = %d"
          % (g, r2, l2, h2, p2, n2))
    OUT.setdefault('within_gamma', {})[str(g)] = {'rho': r2, 'n': n2}

# =============================================================================
hdr("3. MODEL COMPARISON: WHICH VARIABLE SHOULD THE PAPER CARRY?")
yv = np.array([r['ratio'] for r in pers], float)
cands = [('suppressed-phase activation', np.array([[r['supp']] for r in pers])),
         ('kappa/gamma', np.array([[r['kg']] for r in pers])),
         ('alpha*kappa/gamma', np.array([[r['akg']] for r in pers])),
         ('both', np.array([[r['supp'], r['kg']] for r in pers])),
         ('six parameters', np.array([[r['lam'], r['beta'], r['alpha'],
                                       r['sigma'], r['gamma'], r['kappa']]
                                      for r in pers]))]
for lbl, X in cands:
    print("  %-30s LOO R^2 = %+.3f" % (lbl, loo_r2(X, yv)))
    OUT.setdefault('model_comparison', {})[lbl] = float(loo_r2(X, yv))
print("\n  If 'suppressed-phase activation' alone matches or beats 'both', the")
print("  paper should carry one variable and drop kappa/gamma.")

# =============================================================================
hdr("4. FIGURE")
fig = plt.figure(figsize=(13, 4.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1])

ax = fig.add_subplot(gs[0])
for mode, name in MODES.items():
    sub = [r for r in rows if r['mode'] == mode]
    ax.scatter([r['supp'] for r in sub], [r['rival'] for r in sub],
               s=16, alpha=0.55, c=COLS[mode], label=name, edgecolors='none')
xs = np.array([r['supp'] for r in rows], float)
ys = np.array([r['rival'] for r in rows], float)
m = np.isfinite(xs) & np.isfinite(ys)
b = np.polyfit(xs[m], ys[m], 1)
xx = np.linspace(xs[m].min(), xs[m].max(), 50)
ax.plot(xx, np.polyval(b, xx), 'k--', lw=1)
ax.axhline(0, c='k', lw=0.7)
ax.axvline(0, c='k', lw=0.7)
ax.set_xlabel('% change in attended-channel activation\nduring the competitor'
              "'s dominance episodes", fontsize=8)
ax.set_ylabel("% change in competitor's mean dominance duration", fontsize=8)
ax.set_title(r'A   $\rho$ = %.3f, $n$ = %d' % (rho, n), fontsize=10)
ax.legend(fontsize=6.5, loc='upper right', frameon=False)

ax = fig.add_subplot(gs[1])
order = sorted(MODES, key=lambda k: np.nanmedian(
    [r['supp'] for r in rows if r['mode'] == k]))
for i, mode in enumerate(order):
    sub = [r for r in rows if r['mode'] == mode]
    v = np.array([r['rival'] for r in sub], float)
    q = np.nanpercentile(v, [25, 50, 75])
    ax.barh(i, q[1], 0.62, xerr=[[q[1] - q[0]], [q[2] - q[1]]],
            color=COLS[mode], capsize=3)
ax.axvline(0, c='k', lw=1)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([MODES[m] for m in order], fontsize=7.5)
ax.set_xlabel("% change, competitor's duration", fontsize=8)
ax.set_title('B   Ordered by suppressed-phase activation', fontsize=10)

ax = fig.add_subplot(gs[2])
for mode, name in MODES.items():
    sub = [r for r in rows if r['mode'] == mode]
    if len(sub) < 6:
        continue
    xv = np.array([r['supp'] for r in sub], float)
    yv2 = np.array([r['rival'] for r in sub], float)
    mk = np.isfinite(xv) & np.isfinite(yv2)
    if mk.sum() < 6:
        continue
    bb = np.polyfit(xv[mk], yv2[mk], 1)
    xr = np.linspace(xv[mk].min(), xv[mk].max(), 30)
    rr = stats.spearmanr(xv[mk], yv2[mk])[0]
    ax.plot(xr, np.polyval(bb, xr), '-', lw=1.8, c=COLS[mode],
            label='%s  $\\rho$=%.2f' % (name, rr))
ax.axhline(0, c='k', lw=0.7)
ax.axvline(0, c='k', lw=0.7)
ax.set_xlabel('% change in suppressed-phase activation', fontsize=8)
ax.set_ylabel("% change, competitor's duration", fontsize=8)
ax.set_title('C   Within each formulation', fontsize=10)
ax.legend(fontsize=6, frameon=False)

fig.tight_layout()
fig.savefig('wave16_fig5_coupling.pdf')
fig.savefig('wave16_fig5_coupling.png', dpi=200)
print("  wave16_fig5_coupling.pdf")
print("  wave16_fig5_coupling.png")

json.dump(OUT, open('wave16_results.json', 'w', encoding='utf-8'),
          indent=1, default=float)
hdr("DONE")
print("  wave16_results.json")
