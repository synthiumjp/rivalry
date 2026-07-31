"""
wave21_ratio.py -- is the Chong agreement informative?

The model's gated condition gives a competitor-to-attended ratio near 0.31, which
is what Chong et al. (2005) observed. That is evidence only if the model COULD
have produced something else. Two ways it might not be:

  (a) the ratio may be tightly constrained across the parameter space, so that
      any gated increment gives ~0.3 regardless of configuration;
  (b) it may be insensitive to the parameters, so that no configuration
      predicts a different value.

If the ratio is wide and parameter-dependent, 0.31 is a specific value the model
had to land on. If it is narrow and parameter-independent, the agreement carries
little information and the manuscript should say so.

Also reports the ungated ratio for contrast, and whether the gated and ungated
distributions overlap.

Reads wave18_yoked.json. No simulation.
"""

import json
import os
import sys

import numpy as np
from scipy import stats

if not os.path.exists('wave18_yoked.json'):
    sys.exit("wave18_yoked.json not found")
D = json.load(open('wave18_yoked.json', encoding='utf-8'))
OUT = {}
MAGS = [0.125, 0.25, 0.5, 1.0, 2.0]
PARAMS = ['lambda', 'beta', 'alpha', 'sigma', 'gamma', 'kappa']


def hdr(s):
    print("\n" + "=" * 74)
    print(s)
    print("=" * 74)


def pct(a, b):
    return 100.0 * (a - b) / b if (np.isfinite(a) and np.isfinite(b) and b) else np.nan


def collect(mode, mag):
    r, P = [], []
    for rec in D:
        b = rec['baseline']
        c = rec['cond'].get('%d_%g' % (mode, mag))
        if not c or not (b['own'] and b['rival']):
            continue
        x = pct(c['own'], b['own'])
        y = pct(c['rival'], b['rival'])
        if np.isfinite(x) and abs(x) > 2.0 and np.isfinite(y):
            r.append(y / x)
            P.append([rec['params'][k] for k in PARAMS])
    return np.array(r), np.array(P)


hdr("1. HOW WIDE IS THE GATED RATIO ACROSS CONFIGURATIONS?")
print("  Chong et al. (2005) Exp 3 observed 0.310\n")
print("  %6s %5s %8s %8s %8s %20s %10s"
      % ("mag", "n", "min", "median", "max", "IQR", "% in .25-.37"))
for mag in MAGS:
    r, _ = collect(1, mag)
    if len(r) < 10:
        continue
    q = np.percentile(r, [25, 50, 75])
    frac = float(np.mean((r >= 0.25) & (r <= 0.37)))
    print("  %6.3f %5d %8.3f %8.3f %8.3f  [%6.3f, %6.3f] %9.0f%%"
          % (mag, len(r), r.min(), q[1], r.max(), q[0], q[2], 100 * frac))
    OUT.setdefault('gated', {})['%g' % mag] = {
        'n': len(r), 'min': float(r.min()), 'median': float(q[1]),
        'max': float(r.max()), 'iqr': [float(q[0]), float(q[2])],
        'frac_near_chong': frac}

hdr("2. UNGATED, FOR CONTRAST")
print("  %6s %5s %8s %8s %8s %20s" % ("mag", "n", "min", "median", "max", "IQR"))
for mag in MAGS:
    r, _ = collect(0, mag)
    if len(r) < 10:
        continue
    q = np.percentile(r, [25, 50, 75])
    print("  %6.3f %5d %8.3f %8.3f %8.3f  [%6.3f, %6.3f]"
          % (mag, len(r), r.min(), q[1], r.max(), q[0], q[2]))
    OUT.setdefault('ungated', {})['%g' % mag] = {
        'median': float(q[1]), 'iqr': [float(q[0]), float(q[2])]}

hdr("3. IS THE RATIO PARAMETER-DEPENDENT?")
r, P = collect(1, 1.0)
if len(r) > 20:
    print("  gated 1x, n = %d\n" % len(r))
    for j, k in enumerate(PARAMS):
        rho, p = stats.spearmanr(P[:, j], r)
        flag = "  <--" if p < 0.01 else ""
        print("    rho(%-7s, ratio) = %+.3f  p = %.2g%s" % (k, rho, p, flag))
        OUT.setdefault('param_dep', {})[k] = {'rho': float(rho), 'p': float(p)}
    # how much of the ratio is predictable from parameters?
    X = np.column_stack([np.ones(len(r)), np.log(P)])
    n = len(r)
    pred = np.empty(n)
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        bta, *_ = np.linalg.lstsq(X[m], r[m], rcond=None)
        pred[i] = X[i] @ bta
    ss = 1 - ((r - pred) ** 2).sum() / ((r - r.mean()) ** 2).sum()
    print("\n    leave-one-out R^2 from the six log-parameters: %+.3f" % ss)
    OUT['param_r2'] = float(ss)

hdr("4. VERDICT CRITERIA")
g = OUT.get('gated', {}).get('1', {})
if g:
    span = g['max'] - g['min']
    iqr = g['iqr'][1] - g['iqr'][0]
    print("  gated 1x ratio spans %.3f (full range) and %.3f (IQR)"
          % (span, iqr))
    print("  %.0f%% of configurations fall within +/-0.06 of Chong's 0.310"
          % (100 * g['frac_near_chong']))
    print()
    print("  The agreement is INFORMATIVE if the ratio is wide and")
    print("  parameter-dependent: the model could have produced other values and")
    print("  did not. It is UNINFORMATIVE if most configurations cluster near")
    print("  0.31 regardless of parameters, in which case the manuscript should")
    print("  report the agreement as consistency rather than as a prediction.")

json.dump(OUT, open('wave21_results.json', 'w', encoding='utf-8'),
          indent=1, default=float)
hdr("DONE")
print("  wave21_results.json")
