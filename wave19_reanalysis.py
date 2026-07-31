"""
wave19_reanalysis.py -- three questions, no new simulation.

Q1  IS THE GATED EFFECT MORE THAN DUTY CYCLE?
    The gated condition lengthens the attended channel's episodes by 61%, which
    by itself gives the competitor more recovery time. The claim that gate
    timing matters requires the live gate to exceed a delivery schedule with the
    same duty cycle but no contingency with the current trial. Wave 18 ran that
    comparison and reported it only under the difference-based criterion. The
    absolute-criterion values were computed and not printed. They are the
    decisive numbers: if live and yoked coincide under the absolute criterion,
    the +3.6% is a duty-cycle effect and the gate-timing claim is not
    established.

Q2  IS THE CHONG RATIO DIAGNOSTIC?
    The model reproduces Chong et al.'s competitor-to-attended ratio (0.276 and
    0.278 against 0.310). That is only evidence if other formulations give
    different ratios at matched attended-channel change. If every mechanism
    lands near 0.28, the agreement is uninformative.

Q3  WHY DOES THE INCREMENT ARM FAIL LEVELT II?
    No configuration reproduces own ~0% with the competitor at -30%. At steady
    state an increment raises the attended channel by roughly
    dS / (lambda + beta + alpha*kappa/gamma), and adaptation scales with
    activation rather than cancelling it. If the observed attended-channel change
    tracks that expression, the failure is derived rather than merely reported.

Reads wave18_yoked.json and wave15_A_paired.json.
"""

import json
import os
import sys

import numpy as np
from scipy import stats

OUT = {}
NAMES = {0: 'ungated', 1: 'live gate', 2: 'live anti-gate',
         3: 'YOKED gate', 4: 'YOKED anti-gate'}
MAGS = [0.125, 0.25, 0.5, 1.0, 2.0]
MODES = {1: 'persistence', 2: 'increment', 3: 'response gain',
         4: 'inhibitory gain', 5: 'adaptation'}
TARGET_OWN = 30.0


def hdr(s):
    print("\n" + "=" * 74)
    print(s)
    print("=" * 74)


def pct(a, b):
    return 100.0 * (a - b) / b if (np.isfinite(a) and np.isfinite(b) and b) else np.nan


def boot_ci(v, n_boot=4000, seed=42):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) < 6:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# =============================================================================
# Q1
# =============================================================================

def q1():
    hdr("Q1. LIVE VERSUS YOKED, UNDER BOTH CRITERIA")
    fn = 'wave18_yoked.json'
    if not os.path.exists(fn):
        print("  %s not found" % fn)
        return
    D = json.load(open(fn, encoding='utf-8'))
    print("  %-18s %6s %11s %11s %11s %11s"
          % ("condition", "mag", "own", "comp (rel)", "comp (abs)", "dose"))
    store = {}
    for mode in (1, 3, 0, 4, 2):
        for mag in (1.0, 2.0):
            o, r, a, d = [], [], [], []
            for rec in D:
                b = rec['baseline']
                c = rec['cond'].get('%d_%g' % (mode, mag))
                if not c:
                    continue
                o.append(pct(c['own'], b['own']))
                r.append(pct(c['rival'], b['rival']))
                a.append(pct(c['rival_abs'], b['rival_abs']))
                if rec['b_unit']:
                    d.append(c['dose'] / rec['b_unit'])
            f = lambda v: float(np.nanmedian(v)) if v else np.nan
            print("  %-18s %6.1f %+10.1f%% %+10.1f%% %+10.1f%% %10.2f"
                  % (NAMES[mode], mag, f(o), f(r), f(a), f(d)))
            store['%d_%g' % (mode, mag)] = {'own': o, 'rel': r, 'abs': a}

    print("\n  DECISIVE CONTRAST: live gate versus yoked gate, matched dose")
    for mag in (1.0, 2.0):
        lv = store.get('1_%g' % mag)
        yk = store.get('3_%g' % mag)
        if not (lv and yk):
            continue
        for key, lbl in (('rel', 'difference criterion'),
                         ('abs', 'absolute criterion')):
            x = np.array(lv[key], float)
            y = np.array(yk[key], float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 10:
                continue
            diff = x[m] - y[m]
            d = diff.mean() / diff.std(ddof=1)
            t, p = stats.ttest_rel(x[m], y[m])
            lo, hi = boot_ci(diff)
            print("    %gx  %-22s live %+7.2f%%  yoked %+7.2f%%  "
                  "diff %+6.2f pp [%+.2f, %+.2f]  d = %+.2f  p = %.2g"
                  % (mag, lbl, np.median(x[m]), np.median(y[m]),
                     np.median(diff), lo, hi, d, p))
            OUT.setdefault('q1', {})['%g_%s' % (mag, key)] = {
                'live': float(np.median(x[m])), 'yoked': float(np.median(y[m])),
                'diff': float(np.median(diff)), 'ci': [lo, hi],
                'd': float(d), 'p': float(p), 'n': int(m.sum())}
    print("\n  If the live-yoked difference under the ABSOLUTE criterion is")
    print("  small or spans zero, the gated effect is a duty-cycle consequence")
    print("  and the gate-timing claim is not established.")


# =============================================================================
# Q2
# =============================================================================

def q2():
    hdr("Q2. IS THE COMPETITOR-TO-ATTENDED RATIO DIAGNOSTIC?")
    fn = 'wave15_A_paired.json'
    if not os.path.exists(fn):
        print("  %s not found" % fn)
        return
    D = json.load(open(fn, encoding='utf-8'))
    print("  ratio at matched attended-channel change of +%.0f%%" % TARGET_OWN)
    print("  Chong et al. (2005) Exp 3 observed 9/29 = 0.310\n")
    print("  %-18s %8s %10s %20s" % ("formulation", "n", "median", "IQR"))
    res = {}
    for mode, name in MODES.items():
        rs = []
        for rec in D:
            b = rec['baseline']
            if not (b['own'] and b['rival']):
                continue
            lv = rec['modes'][str(mode)]
            xs = np.array([pct(c['own'], b['own']) for c in lv], float)
            ys = np.array([pct(c['rival'], b['rival']) for c in lv], float)
            m = np.isfinite(xs) & np.isfinite(ys)
            if m.sum() < 3:
                continue
            o = np.argsort(xs[m])
            xx, yy = xs[m][o], ys[m][o]
            if not (xx.min() <= TARGET_OWN <= xx.max()):
                continue
            rs.append(float(np.interp(TARGET_OWN, xx, yy)) / TARGET_OWN)
        if len(rs) < 6:
            print("  %-18s %8d  (too few)" % (name, len(rs)))
            continue
        q = np.percentile(rs, [25, 50, 75])
        print("  %-18s %8d %+9.3f  [%+.3f, %+.3f]" % (name, len(rs), q[1],
                                                      q[0], q[2]))
        res[name] = {'n': len(rs), 'median': float(q[1]),
                     'iqr': [float(q[0]), float(q[2])]}
    OUT['q2'] = res
    vals = [v['median'] for v in res.values()]
    if len(vals) >= 3:
        print("\n  spread of medians across formulations: %.3f to %.3f"
              % (min(vals), max(vals)))
        print("  A narrow spread means the ratio does not discriminate")
        print("  mechanisms and the agreement with Chong is weak evidence.")


# =============================================================================
# Q3
# =============================================================================

def q3():
    hdr("Q3. IS THE LEVELT II FAILURE DERIVABLE?")
    fn = 'wave15_A_paired.json'
    if not os.path.exists(fn):
        return
    D = json.load(open(fn, encoding='utf-8'))
    print("  At steady state an increment dS raises the attended channel by")
    print("  roughly dS / (lambda + beta + alpha*kappa/gamma). If the attended-")
    print("  channel change under the increment tracks that, the failure to")
    print("  reproduce Levelt II is derived rather than merely reported.\n")
    gain, own, comp, ratio = [], [], [], []
    for rec in D:
        b = rec['baseline']
        if not (b['own'] and b['rival']):
            continue
        p = rec['params']
        denom = p['lambda'] + p['beta'] + p['alpha'] * p['kappa'] / p['gamma']
        lv = rec['modes']['2']
        # take the level closest to a +30% attended change
        best, bo, bc = None, np.nan, np.nan
        for c in lv:
            o = pct(c['own'], b['own'])
            if np.isfinite(o):
                if best is None or abs(o - TARGET_OWN) < abs(best - TARGET_OWN):
                    best, bo, bc = o, o, pct(c['rival'], b['rival'])
        if best is None or not np.isfinite(bc):
            continue
        gain.append(1.0 / denom)
        own.append(bo)
        comp.append(bc)
        ratio.append(bc / bo if abs(bo) > 1 else np.nan)
    gain, own, comp = np.array(gain), np.array(own), np.array(comp)
    r1, p1 = stats.spearmanr(gain, own)
    r2, p2 = stats.spearmanr(gain, comp)
    print("  n = %d configurations" % len(gain))
    print("  rho(1/(lambda+beta+alpha*kappa/gamma), attended change)   = %+.3f  p = %.2g"
          % (r1, p1))
    print("  rho(same, competitor change)                              = %+.3f  p = %.2g"
          % (r2, p2))
    rr = np.array([x for x in ratio if np.isfinite(x)])
    print("\n  competitor-to-attended ratio under the ungated increment:")
    print("    median %+.3f  IQR [%+.3f, %+.3f]  n = %d"
          % (np.median(rr), np.percentile(rr, 25), np.percentile(rr, 75),
             len(rr)))
    print("    Levelt II requires this ratio to diverge (attended change ~0")
    print("    with the competitor compressed). Observed values near -1 mean")
    print("    the two channels trade off roughly symmetrically, which is")
    print("    exactly what the proposition denies.")
    n_ext = int(np.sum(np.abs(rr) > 5))
    print("    configurations with |ratio| > 5 (approaching Levelt II): %d/%d"
          % (n_ext, len(rr)))
    OUT['q3'] = {'rho_gain_own': float(r1), 'rho_gain_comp': float(r2),
                 'median_ratio': float(np.median(rr)), 'n': len(rr),
                 'n_extreme': n_ext}


if __name__ == '__main__':
    q1()
    q2()
    q3()
    json.dump(OUT, open('wave19_results.json', 'w', encoding='utf-8'),
              indent=1, default=float)
    hdr("DONE")
    print("  wave19_results.json")
