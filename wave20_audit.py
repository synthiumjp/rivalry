"""
wave20_audit.py -- two questions the manuscript cannot currently answer.

Q1  THE CHONG COMPARISON, FROM THE CANONICAL RUN
    Section 4.7.5 quotes ratios of 0.276 and 0.278 from a pre-canonical
    campaign. Section 4.7.2's canonical run contains gated conditions at the same
    magnitudes and implies a different ratio. One number should appear in the
    paper, and it should be the canonical one. This recomputes the
    competitor-to-attended ratio for every gated magnitude in the canonical run,
    with a bootstrap interval, so the comparison with Chong et al.'s 0.310 rests
    on one campaign.

Q2  ARE THE TWO rho = +0.774 RESULTS INDEPENDENT?
    The manuscript reports rho = +0.774 for lambda/(lambda+beta+alpha*kappa/gamma)
    against the maximum achievable attentional gain, and rho = +0.774 for
    1/(lambda+beta+alpha*kappa/gamma) against the attended-channel change under an
    input increment. Identical coefficients to three decimals across what are
    presented as two distinct findings.

    The two predictors differ only by a factor of lambda, which takes four grid
    values, so they may be nearly collinear. The two outcomes are both measures
    of how strongly a configuration responds to a manipulation, so they may be
    nearly the same variable. If both correlations are high, "one quantity
    governs both arms" reduces to "one quantity indexes how responsive a
    configuration is", which is a weaker and different claim.

Q3  IS THE YOKED CONTRAST EXPLAINED BY ITS DOSE DIFFERENCE?
    Live gate 2x delivers 1.08 and yoked gate 2x delivers 0.98. The dose-response
    slope from the ungated conditions gives the expected effect of a 10%
    difference, which can be compared against the observed 20.4 pp.

Reads wave18_yoked.json, wave15_A_paired.json.
"""

import json
import os

import numpy as np
from scipy import stats

OUT = {}


def hdr(s):
    print("\n" + "=" * 74)
    print(s)
    print("=" * 74)


def pct(a, b):
    return 100.0 * (a - b) / b if (np.isfinite(a) and np.isfinite(b) and b) else np.nan


def boot_med(v, n_boot=4000, seed=42):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) < 6:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)]
    return float(np.median(v)), float(np.percentile(m, 2.5)), \
        float(np.percentile(m, 97.5))


MAGS = [0.125, 0.25, 0.5, 1.0, 2.0]

# =============================================================================
hdr("Q1. CHONG COMPARISON FROM THE CANONICAL RUN")
fn = 'wave18_yoked.json'
if not os.path.exists(fn):
    print("  %s not found" % fn)
else:
    D = json.load(open(fn, encoding='utf-8'))
    print("  Chong et al. (2005) Exp 3 observed 9/29 = 0.310")
    print("  gated condition (increment delivered only while attended dominant)\n")
    print("  %6s %11s %11s %11s %22s"
          % ("mag", "attended", "competitor", "ratio", "95% CI on ratio"))
    rows = {}
    for mag in MAGS:
        o, r, ratio = [], [], []
        for rec in D:
            b = rec['baseline']
            c = rec['cond'].get('1_%g' % mag)
            if not c or not (b['own'] and b['rival']):
                continue
            x = pct(c['own'], b['own'])
            y = pct(c['rival'], b['rival'])
            o.append(x)
            r.append(y)
            if np.isfinite(x) and abs(x) > 1.0 and np.isfinite(y):
                ratio.append(y / x)
        med, lo, hi = boot_med(ratio)
        print("  %6.3f %+10.1f%% %+10.1f%% %11.3f    [%.3f, %.3f]"
              % (mag, np.nanmedian(o), np.nanmedian(r), med, lo, hi))
        rows['%g' % mag] = {'attended': float(np.nanmedian(o)),
                            'competitor': float(np.nanmedian(r)),
                            'ratio': med, 'ci': [lo, hi], 'n': len(ratio)}
    OUT['chong_canonical'] = rows
    vals = [v['ratio'] for v in rows.values() if np.isfinite(v['ratio'])]
    if vals:
        print("\n  spread of ratio across a %gx range of magnitude: %.3f to %.3f"
              % (MAGS[-1] / MAGS[0], min(vals), max(vals)))
        inside = [k for k, v in rows.items()
                  if np.isfinite(v['ci'][0]) and v['ci'][0] <= 0.310 <= v['ci'][1]]
        print("  magnitudes whose 95%% CI contains Chong's 0.310: %s"
              % (", ".join(inside) if inside else "none"))
    print("\n  These are the numbers the manuscript should quote. Section 4.7.5")
    print("  currently quotes 0.276 and 0.278 from an earlier campaign.")

# =============================================================================
hdr("Q2. ARE THE TWO rho = +0.774 RESULTS INDEPENDENT?")
fn2 = 'wave15_A_paired.json'
if not os.path.exists(fn2):
    print("  %s not found" % fn2)
else:
    P = json.load(open(fn2, encoding='utf-8'))
    lam_d, inv_d, max_g, inc_o = [], [], [], []
    for rec in P:
        b = rec['baseline']
        if not (b['own'] and b['rival']):
            continue
        p = rec['params']
        den = p['lambda'] + p['beta'] + p['alpha'] * p['kappa'] / p['gamma']
        # outcome 1: maximum achievable attended-channel gain under persistence
        best = -1e9
        for c in rec['modes']['1']:
            v = pct(c['own'], b['own'])
            if np.isfinite(v):
                best = max(best, v)
        # outcome 2: attended-channel change under the increment nearest +30%
        near, nb = None, np.nan
        for c in rec['modes']['2']:
            v = pct(c['own'], b['own'])
            if np.isfinite(v) and (near is None or abs(v - 30) < abs(near - 30)):
                near, nb = v, v
        if best < -1e8 or not np.isfinite(nb):
            continue
        lam_d.append(p['lambda'] / den)
        inv_d.append(1.0 / den)
        max_g.append(best)
        inc_o.append(nb)
    lam_d, inv_d = np.array(lam_d), np.array(inv_d)
    max_g, inc_o = np.array(max_g), np.array(inc_o)
    print("  n = %d configurations\n" % len(lam_d))
    r_pred, p_pred = stats.spearmanr(lam_d, inv_d)
    r_out, p_out = stats.spearmanr(max_g, inc_o)
    print("  rho(lambda/denom, 1/denom)                  = %+.3f  p = %.2g"
          % (r_pred, p_pred))
    print("  rho(max attentional gain, increment gain)   = %+.3f  p = %.2g"
          % (r_out, p_out))
    print()
    r1, p1 = stats.spearmanr(lam_d, max_g)
    r2, p2 = stats.spearmanr(inv_d, inc_o)
    r3, _ = stats.spearmanr(inv_d, max_g)
    r4, _ = stats.spearmanr(lam_d, inc_o)
    print("  reported result 1: rho(lambda/denom, max gain)      = %+.3f" % r1)
    print("  reported result 2: rho(1/denom, increment gain)     = %+.3f" % r2)
    print("  cross:             rho(1/denom, max gain)           = %+.3f" % r3)
    print("  cross:             rho(lambda/denom, increment gain)= %+.3f" % r4)
    OUT['independence'] = {'rho_predictors': float(r_pred),
                           'rho_outcomes': float(r_out),
                           'r1': float(r1), 'r2': float(r2),
                           'r3': float(r3), 'r4': float(r4),
                           'n': int(len(lam_d))}
    print("\n  If the two predictors correlate above about 0.9 AND the two")
    print("  outcomes do too, the claim that one quantity governs both arms")
    print("  reduces to one quantity indexing overall responsiveness.")

# =============================================================================
hdr("Q3. IS THE YOKED CONTRAST EXPLAINED BY ITS DOSE DIFFERENCE?")
if os.path.exists(fn):
    D = json.load(open(fn, encoding='utf-8'))

    def cell(mode, mag, key='rival_abs'):
        v, d = [], []
        for rec in D:
            b = rec['baseline']
            c = rec['cond'].get('%d_%g' % (mode, mag))
            if not c or not b[key]:
                continue
            v.append(pct(c[key], b[key]))
            if rec['b_unit']:
                d.append(c['dose'] / rec['b_unit'])
        return float(np.nanmedian(v)), float(np.nanmedian(d))

    u1, du1 = cell(0, 1.0)
    u2, du2 = cell(0, 2.0)
    lg, dlg = cell(1, 2.0)
    yg, dyg = cell(3, 2.0)
    slope = (u2 - u1) / (du2 - du1) if (du2 - du1) else np.nan
    dose_gap = dlg - dyg
    expected = slope * dose_gap
    observed = lg - yg
    print("  ungated dose-response: %+.2f%% at dose %.2f, %+.2f%% at dose %.2f"
          % (u1, du1, u2, du2))
    print("  implied slope: %+.2f percentage points per unit dose" % slope)
    print("  live gate 2x dose %.2f, yoked gate 2x dose %.2f, gap %.2f"
          % (dlg, dyg, dose_gap))
    print("  effect attributable to the dose gap : %+.2f pp" % expected)
    print("  observed live-minus-yoked difference: %+.2f pp" % observed)
    if np.isfinite(expected) and expected:
        print("  observed exceeds dose expectation by %.0fx"
              % abs(observed / expected))
    OUT['dose_check'] = {'slope': float(slope), 'dose_gap': float(dose_gap),
                         'expected': float(expected), 'observed': float(observed)}

json.dump(OUT, open('wave20_results.json', 'w', encoding='utf-8'),
          indent=1, default=float)
hdr("DONE")
print("  wave20_results.json")
