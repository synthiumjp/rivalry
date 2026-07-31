"""
wave7_gate_gradient.py -- is floor occupancy a gate or a gradient?

Wave 6 produced an apparent contradiction:
    AUC (floor occupancy)  = 0.750   -- good binary classifier
    LOO R^2 (floor alone)  = +0.006  -- useless linear predictor
    LOO R^2 (six params)   = +0.112

That pattern is the signature of a STEP function with within-group variance:
floor occupancy says whether control is possible, something else says how much.
This script tests that directly.

  1. Per-regime held-out correlations and AUC (the pooled figures treat 100
     paired configurations as 200 independent ones).
  2. Among configurations that switch at all, does floor occupancy predict the
     rate? If not, the gate/gradient split is clean.
  3. Model comparison: step vs linear vs parameters vs step+parameters, all by
     leave-one-out.
  4. If floor gates, what governs the gradient? Univariate LOO R^2 for each
     parameter within the switching subset.

Reads wave6_H_mediator.json. No simulation. Run from C:\\crewther.
Writes wave7_results.json, wave7_fig_gate.pdf
"""

import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_BOOT = 10000
SEED = 42
SWITCH_THRESH = 0.05
PARAMS = ['lambda', 'beta', 'alpha', 'sigma', 'gamma', 'kappa']

OUT = {}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def ci_str(lo, hi, fmt="%+.3f"):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "[  --  ,   --  ]"
    return "[" + fmt % lo + ", " + fmt % hi + "]"


def boot_spearman(x, y, n_boot=N_BOOT, seed=SEED):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 4:
        return np.nan, np.nan, np.nan, np.nan, len(x)
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    rs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(x[i])) < 3 or len(np.unique(y[i])) < 3:
            continue
        r = stats.spearmanr(x[i], y[i])[0]
        if np.isfinite(r):
            rs.append(r)
    lo, hi = (np.percentile(rs, [2.5, 97.5]) if len(rs) > 100
              else (np.nan, np.nan))
    return float(rho), float(p), float(lo), float(hi), int(n)


def auc_ci(scores, labels, n_boot=N_BOOT, seed=SEED):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, bool)
    m = np.isfinite(scores)
    scores, labels = scores[m], labels[m]

    def _a(s, l):
        pos, neg = s[l], s[~l]
        if len(pos) == 0 or len(neg) == 0:
            return np.nan
        r = stats.rankdata(np.concatenate([pos, neg]))
        return ((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                / (len(pos) * len(neg)))

    a = _a(scores, labels)
    rng = np.random.default_rng(seed)
    n = len(scores)
    v = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        x = _a(scores[i], labels[i])
        if np.isfinite(x):
            v.append(x)
    lo, hi = np.percentile(v, [2.5, 97.5]) if v else (np.nan, np.nan)
    return float(a), float(lo), float(hi)


def loo_r2(X, y):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(y)
    if n < X.shape[1] + 3:
        return np.nan
    Xd = np.column_stack([np.ones(n), X])
    pred = np.empty(n)
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        try:
            beta, *_ = np.linalg.lstsq(Xd[m], y[m], rcond=None)
        except np.linalg.LinAlgError:
            return np.nan
        pred[i] = Xd[i] @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot else np.nan


def hdr(s):
    print("\n" + "=" * 72)
    print(s)
    print("=" * 72)


# =============================================================================

H = json.load(open('wave6_H_mediator.json', encoding='utf-8'))
print("loaded %d held-out configurations" % len(H))

rows = []
for rec in H:
    for rn in ('dorsal', 'ventral'):
        R = rec['regimes'][rn]
        d = {'regime': rn,
             'floor': R['baseline_floor_frac'],
             'switch': R['max_switch_rate'],
             'aol': R['alpha_over_lambda'],
             'ceil': R.get('baseline_ceil_frac', np.nan)}
        for k in PARAMS:
            d[k] = rec['params'][k]
        rows.append(d)

floor = np.array([r['floor'] for r in rows])
switch = np.array([r['switch'] for r in rows])
aol = np.array([r['aol'] for r in rows])
regime = np.array([r['regime'] for r in rows])
P = np.array([[np.log(r[k]) if r[k] > 0 else np.log(1e-6) for k in PARAMS]
              for r in rows])
lab = switch > SWITCH_THRESH

# -----------------------------------------------------------------------------
hdr("1. PER-REGIME (pooled figures treat 100 paired configs as 200)")
for rn in ('dorsal', 'ventral', 'POOLED'):
    m = np.ones(len(rows), bool) if rn == 'POOLED' else (regime == rn)
    n = int(m.sum())
    rho, p, lo, hi, _ = boot_spearman(floor[m], switch[m])
    a, alo, ahi = auc_ci(-floor[m], lab[m])
    a2, a2lo, a2hi = auc_ci(-aol[m], lab[m])
    frac = float(lab[m].mean())
    print("  %-8s n=%3d  switching %4.1f%%" % (rn, n, 100 * frac))
    print("      rho(floor, switch) = %+.3f %s p=%.2g"
          % (rho, ci_str(lo, hi), p))
    print("      AUC floor          = %.3f [%.3f, %.3f]" % (a, alo, ahi))
    print("      AUC alpha/lambda   = %.3f [%.3f, %.3f]" % (a2, a2lo, a2hi))
    OUT.setdefault('per_regime', {})[rn] = {
        'n': n, 'frac_switching': frac,
        'rho_floor': rho, 'rho_ci': [lo, hi], 'p': p,
        'auc_floor': [a, alo, ahi], 'auc_aol': [a2, a2lo, a2hi]}

# -----------------------------------------------------------------------------
hdr("2. GATE OR GRADIENT? Among configurations that switch at all")
sw = lab
print("  switching subset: n = %d of %d" % (int(sw.sum()), len(rows)))
if sw.sum() >= 10:
    rho, p, lo, hi, n = boot_spearman(floor[sw], switch[sw])
    print("  rho(floor, switch rate | switching) = %+.3f %s p=%.2g  n=%d"
          % (rho, ci_str(lo, hi), p, n))
    print("  switch rate among switchers: median %.3f  IQR [%.3f, %.3f]"
          % (np.median(switch[sw]), np.percentile(switch[sw], 25),
             np.percentile(switch[sw], 75)))
    print("  floor among switchers      : median %.3f  IQR [%.3f, %.3f]"
          % (np.median(floor[sw]), np.percentile(floor[sw], 25),
             np.percentile(floor[sw], 75)))
    print("  floor among non-switchers  : median %.3f  IQR [%.3f, %.3f]"
          % (np.median(floor[~sw]), np.percentile(floor[~sw], 25),
             np.percentile(floor[~sw], 75)))
    OUT['gradient_within_switchers'] = {
        'n': int(sw.sum()), 'rho': rho, 'ci': [lo, hi], 'p': p}

# -----------------------------------------------------------------------------
hdr("3. MODEL COMPARISON (leave-one-out R^2 on switch rate)")
thr_grid = np.unique(np.round(floor, 3))
best_thr, best_acc = None, -1
for t in thr_grid:
    acc = np.mean((floor < t) == lab)
    if acc > best_acc:
        best_thr, best_acc = t, acc
step = (floor < best_thr).astype(float)

models = [
    ('floor, linear', floor[:, None]),
    ('floor, step (thr %.3f)' % best_thr, step[:, None]),
    ('six log-parameters', P),
    ('step + parameters', np.column_stack([step, P])),
    ('floor + step + params', np.column_stack([floor, step, P])),
]
for name, X in models:
    print("  %-26s LOO R^2 = %+.4f" % (name, loo_r2(X, switch)))
OUT['model_comparison'] = {name: float(loo_r2(X, switch))
                           for name, X in models}
print("\n  A step model beating a linear one on the same variable is the")
print("  signature of a threshold: the variable says WHETHER, not HOW MUCH.")

# -----------------------------------------------------------------------------
hdr("4. WHAT GOVERNS THE GRADIENT, among switchers?")
if sw.sum() >= 15:
    print("  univariate LOO R^2 predicting switch rate within the switching subset")
    uni = {}
    for j, k in enumerate(PARAMS):
        r2 = loo_r2(P[sw, j][:, None], switch[sw])
        rho, p, lo, hi, _ = boot_spearman(P[sw, j], switch[sw])
        uni[k] = {'loo_r2': float(r2), 'rho': rho, 'ci': [lo, hi], 'p': p}
        print("    log %-7s R^2 %+.4f   rho %+.3f %s p=%.2g"
              % (k, r2, rho, ci_str(lo, hi), p))
    r2_all = loo_r2(P[sw], switch[sw])
    r2_fl = loo_r2(floor[sw][:, None], switch[sw])
    print("    %-11s R^2 %+.4f" % ("all six", r2_all))
    print("    %-11s R^2 %+.4f" % ("floor", r2_fl))
    OUT['gradient_drivers'] = {'univariate': uni,
                               'all_six': float(r2_all),
                               'floor': float(r2_fl)}

# -----------------------------------------------------------------------------
hdr("5. CLASSIFICATION with intervals, per regime")
for rn in ('dorsal', 'ventral', 'POOLED'):
    m = np.ones(len(rows), bool) if rn == 'POOLED' else (regime == rn)
    k = int(np.sum((floor[m] < best_thr) == lab[m]))
    n = int(m.sum())
    lo, hi = wilson(k, n)
    base = max(lab[m].mean(), 1 - lab[m].mean())
    print("  %-8s %3d/%3d = %.1f%%  95%% CI [%.1f%%, %.1f%%]   "
          "majority-class baseline %.1f%%"
          % (rn, k, n, 100 * k / n, 100 * lo, 100 * hi, 100 * base))
    OUT.setdefault('classification', {})[rn] = {
        'correct': k, 'n': n, 'ci': [float(lo), float(hi)],
        'baseline': float(base)}

# -----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
cols = {'dorsal': '#2c6fbb', 'ventral': '#d1495b'}
for rn in ('dorsal', 'ventral'):
    m = regime == rn
    ax[0].scatter(floor[m], switch[m], s=22, alpha=0.6, c=cols[rn], label=rn)
ax[0].axvline(best_thr, ls='--', c='k', lw=1)
ax[0].set_xlabel('floor occupancy')
ax[0].set_ylabel('max switch rate')
ax[0].set_title('A  Gate, not gradient')
ax[0].legend(fontsize=7)

ax[1].hist([floor[lab], floor[~lab]], bins=16,
           label=['switches', 'no switches'], color=['#2c6fbb', '#cccccc'])
ax[1].axvline(best_thr, ls='--', c='k', lw=1)
ax[1].set_xlabel('floor occupancy')
ax[1].set_ylabel('configurations')
ax[1].set_title('B  Separation')
ax[1].legend(fontsize=7)

names = [m[0] for m in models]
vals = [loo_r2(X, switch) for _, X in models]
ax[2].barh(range(len(names)), vals, color='#2c6fbb')
ax[2].set_yticks(range(len(names)))
ax[2].set_yticklabels(names, fontsize=6)
ax[2].axvline(0, c='k', lw=1)
ax[2].set_xlabel(r'LOO $R^2$')
ax[2].set_title('C  Model comparison')
fig.tight_layout()
fig.savefig('wave7_fig_gate.pdf')

OUT['best_threshold'] = float(best_thr)
with open('wave7_results.json', 'w', encoding='utf-8') as f:
    json.dump(OUT, f, indent=1, default=float)
hdr("DONE")
print("  wave7_results.json")
print("  wave7_fig_gate.pdf")
