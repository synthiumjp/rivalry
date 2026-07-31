"""
wave14_formulations.py -- the three items the reviewer identified as decisive.

BLOCK A  IS THE DICHOTOMY DEFINITIONAL?
    The manuscript contrasts a state-dependent goal signal with an additive input
    increment and claims the literature treats them as interchangeable. No
    citation supports that, and the field's leading formalisation -- normalisation
    (Reynolds & Heeger, 2009), on which Li et al. (2017) build -- is neither of
    the two options compared. So the contrast may be partly definitional.

    We implement five formulations of attentional modulation in the SAME
    architecture and compare them OUTCOME-MATCHED, tuning each so the manipulated
    channel's dominance duration rises by the same amount, then reading off the
    rival's response:

      1 persistence      G*x on the recurrent term          (the paper's account)
      2 increment        additive constant on the input     (stimulus strength)
      3 response gain    whole update scaled by (1+m)       (multiplicative)
      4 inhibitory gain  attended channel inhibits more     (biased competition)
      5 adaptation       attended channel adapts less       (kappa reduced)

    If several formulations produce same-sign coupling, the finding is about
    state-dependence generally rather than about this one term. If only
    persistence does, the specificity is a result. Either way the dichotomy stops
    being an assertion about the literature.

BLOCK B  IS THE AUC REVERSAL MECHANISM OR RANGE RESTRICTION?
    Regimes are constructed by multiplying alpha, beta and kappa, so within-regime
    AUC for alpha/lambda is computed on a truncated range of that same predictor.
    Range-restricted AUCs are unstable and can flip sign for reasons unrelated to
    mechanism. Here we reassign strata by MEASURED floor occupancy and by a
    median split on sigma (orthogonal to the multipliers) and check whether the
    reversal survives. Reanalysis of wave6_H_mediator.json -- no simulation.

BLOCK C  CEILING AUDIT
    Section 2.1 claims x_max = 5.0 gives roughly threefold headroom. At the
    low-adaptation corner the steady-state relation gives x = 2.65 at
    equidominance and 5.62 during dominance, above the ceiling. This computes the
    headroom analytically across the whole grid and separates configurations that
    fail to reach a 50% increase because of the dissipativity ceiling from those
    limited by saturation.

Run:  python wave14_formulations.py   (~5 min)  |  --analyse  |  --quick
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from numba import njit
except ImportError:
    def njit(*a, **k):
        def w(f):
            return f
        return w if not (a and callable(a[0])) else a[0]

from wave2_campaign import (
    run_trace, extract_durations, occupancy_stats, _rectify,
    BURN_IN, MARGIN, X_MAX, G_SAFETY, SIGNAL_NEUTRAL, load_configs,
)
import wave6_robustness as w6

TARGET_OWN = 30.0          # outcome-matching criterion, % change
MODES = {1: 'persistence', 2: 'increment', 3: 'response gain',
         4: 'inhibitory gain', 5: 'adaptation'}
OUT = {}


def hdr(s):
    print("\n" + "=" * 74)
    print(s)
    print("=" * 74)


def pct(a, b):
    return 100.0 * (a - b) / b if (np.isfinite(a) and np.isfinite(b) and b) else np.nan


# =============================================================================
# KERNEL
# =============================================================================

@njit(cache=True)
def run_mode(lam, beta, alpha, sigma, gamma_adapt, kappa,
             sig_a, sig_b, mode, m, n_steps, seed, x_max):
    """
    mode 0 baseline
         1 persistence      : recurrent coefficient (1 - lam + m)
         2 increment        : additive m on channel A's input
         3 response gain    : channel A's whole update scaled by (1 + m)
         4 inhibitory gain  : inhibition A exerts on B scaled by (1 + m)
         5 adaptation       : channel A's kappa scaled by (1 - m)
    """
    np.random.seed(seed)
    ta = np.empty(n_steps, dtype=np.float64)
    tb = np.empty(n_steps, dtype=np.float64)
    x_a = 0.1
    x_b = 0.1
    a_a = 0.0
    a_b = 0.0
    g_a = m if mode == 1 else 0.0
    b_a = m if mode == 2 else 0.0
    rg = m if mode == 3 else 0.0
    ig = m if mode == 4 else 0.0
    kr = (1.0 - m) if mode == 5 else 1.0
    n_ceil = 0
    n_cnt = 0

    for t in range(n_steps):
        ta[t] = x_a
        tb[t] = x_b
        ea = np.random.normal(0.0, sigma)
        eb = np.random.normal(0.0, sigma)
        na = (1.0 - lam + g_a) * x_a + sig_a + b_a - beta * x_b \
            - alpha * a_a + ea
        if rg > 0.0:
            na = (1.0 + rg) * na
        nb = (1.0 - lam) * x_b + sig_b - beta * (1.0 + ig) * x_a \
            - alpha * a_b + eb
        x_a = _rectify(na, -1.0, x_max)
        x_b = _rectify(nb, -1.0, x_max)
        a_a = (1.0 - gamma_adapt) * a_a + kappa * kr * x_a
        a_b = (1.0 - gamma_adapt) * a_b + kappa * x_b
        if t >= BURN_IN:
            hi = x_a if x_a > x_b else x_b
            if hi >= x_max - 1e-9:
                n_ceil += 1
            n_cnt += 1
    return ta, tb, (n_ceil / n_cnt if n_cnt else 0.0)


def summarise(ta, tb):
    ch, dur = extract_durations(ta, tb, BURN_IN, MARGIN)
    if len(ch) > 2:
        ch, dur = ch[1:-1], dur[1:-1].astype(float)
    else:
        ch, dur = np.empty(0, np.int32), np.empty(0, float)
    da, db = dur[ch == 0], dur[ch == 1]
    nsw = int(np.sum(ch[1:] != ch[:-1])) if len(ch) > 1 else 0
    tot = da.sum() + db.sum()
    return {'own': float(da.mean()) if len(da) else np.nan,
            'rival': float(db.mean()) if len(db) else np.nan,
            'pred': float(da.sum() / tot) if tot else np.nan,
            'alt': nsw / (len(ta) - BURN_IN)}


def levels_for(mode, lam, x_base):
    if mode == 1:
        return [f * lam for f in (0.1, 0.25, 0.5, 0.75, 0.9)]
    if mode == 2:
        return [f * lam * x_base for f in (0.1, 0.25, 0.5, 1.0, 2.0)]
    if mode == 3:
        return [0.005, 0.01, 0.02, 0.05, 0.10]
    if mode == 4:
        return [0.1, 0.25, 0.5, 1.0, 2.0]
    if mode == 5:
        return [0.1, 0.25, 0.5, 0.75, 0.9]
    return [0.0]


def cond(p, mode, m, seedkey, n_seeds, n_steps):
    rows, ce = [], []
    for s in range(n_seeds):
        ta, tb, c = run_mode(p['lambda'], p['beta'], p['alpha'], p['sigma'],
                             p['gamma'], p['kappa'], SIGNAL_NEUTRAL,
                             SIGNAL_NEUTRAL, mode, m, n_steps,
                             w6.make_seed(*seedkey, s), X_MAX)
        rows.append(summarise(ta, tb))
        ce.append(c)

    def f(k):
        v = [r[k] for r in rows if np.isfinite(r[k])]
        return float(np.mean(v)) if v else np.nan
    return {'own': f('own'), 'rival': f('rival'), 'pred': f('pred'),
            'alt': f('alt'), 'ceil': float(np.mean(ce))}


def _save(tag, obj):
    fn = "wave14_%s.json" % tag
    json.dump(obj, open(fn, 'w', encoding='utf-8'), indent=1, default=float)
    print("    -> %s" % fn)


# =============================================================================
# BLOCK A
# =============================================================================

def block_A(pool, n_seeds, n_steps):
    hdr("BLOCK A: five attention formulations, outcome-matched")
    out = []
    t0 = time.time()
    for n, ent in enumerate(pool):
        p = ent['params']
        ci = ent['grid_index'] % 1000
        acts = []
        for s in range(20):
            t_a, _, _ = run_mode(p['lambda'], p['beta'], p['alpha'], p['sigma'],
                                 p['gamma'], p['kappa'], SIGNAL_NEUTRAL,
                                 SIGNAL_NEUTRAL, 0, 0.0, n_steps,
                                 w6.make_seed(0, ci, 0, 0, s), X_MAX)
            acts.append(float(np.mean(t_a[BURN_IN:])))
        x_base = float(np.mean(acts))
        rec = {'grid_index': ent['grid_index'], 'params': p,
               'x_baseline': x_base, 'modes': {}}
        rec['baseline'] = cond(p, 0, 0.0, (1, ci, 0, 0), n_seeds, n_steps)
        li = 1
        for mode in MODES:
            lv = []
            for m in levels_for(mode, p['lambda'], x_base):
                c = cond(p, mode, m, (2, ci, mode, li), n_seeds, n_steps)
                c['m'] = m
                lv.append(c)
                li += 1
            rec['modes'][str(mode)] = lv
        out.append(rec)
        if (n + 1) % 20 == 0:
            print("    %3d/%d  (%.0fs)" % (n + 1, len(pool), time.time() - t0))
    _save('A_modes', out)
    return out


def analyse_A(D):
    hdr("A. RIVAL RESPONSE AT MATCHED OWN-DURATION CHANGE (+%.0f%%)" % TARGET_OWN)
    print("  %-18s %6s %12s %20s %12s"
          % ("formulation", "n", "rival med", "rival IQR", "alt rate"))
    res = {}
    for mode, name in MODES.items():
        rv, al, pr = [], [], []
        for rec in D:
            b = rec['baseline']
            if not (b['own'] and b['rival'] and b['alt']):
                continue
            lv = rec['modes'][str(mode)]
            xs = np.array([pct(c['own'], b['own']) for c in lv], float)
            yr = np.array([pct(c['rival'], b['rival']) for c in lv], float)
            ya = np.array([pct(c['alt'], b['alt']) for c in lv], float)
            yp = np.array([pct(c['pred'], b['pred']) for c in lv], float)
            m = np.isfinite(xs) & np.isfinite(yr)
            if m.sum() < 3:
                continue
            o = np.argsort(xs[m])
            xx = xs[m][o]
            if not (xx.min() <= TARGET_OWN <= xx.max()):
                continue
            rv.append(float(np.interp(TARGET_OWN, xx, yr[m][o])))
            al.append(float(np.interp(TARGET_OWN, xx, ya[m][o])))
            pr.append(float(np.interp(TARGET_OWN, xx, yp[m][o])))
        if len(rv) < 5:
            print("  %-18s %6d  (too few reach the criterion)" % (name, len(rv)))
            continue
        q = np.percentile(rv, [25, 50, 75])
        n_pos = int(np.sum(np.array(rv) > 0))
        lo, hi = w6.wilson(n_pos, len(rv))
        print("  %-18s %6d %+11.1f%% [%+7.1f%%, %+7.1f%%] %+11.1f%%"
              % (name, len(rv), q[1], q[0], q[2], np.median(al)))
        print("  %-18s        same sign as own: %d/%d  [%.0f%%, %.0f%%]"
              % ("", n_pos, len(rv), 100 * lo, 100 * hi))
        res[name] = {'n': len(rv), 'rival_median': float(q[1]),
                     'rival_iqr': [float(q[0]), float(q[2])],
                     'alt_median': float(np.median(al)),
                     'pred_median': float(np.median(pr)),
                     'n_same_sign': n_pos,
                     'same_sign_ci': [float(lo), float(hi)]}
    OUT['formulations'] = res
    print("\n  If only persistence gives same-sign coupling, the specificity is")
    print("  a result. If several do, the finding is about state-dependence")
    print("  generally, and the paper should say so.")


# =============================================================================
# BLOCK B  (reanalysis)
# =============================================================================

def _auc(sc, lab):
    sc, lab = np.asarray(sc, float), np.asarray(lab, bool)
    m = np.isfinite(sc)
    sc, lab = sc[m], lab[m]
    if lab.sum() < 3 or (~lab).sum() < 3:
        return np.nan
    r = stats.rankdata(np.concatenate([sc[lab], sc[~lab]]))
    npos = int(lab.sum())
    return (r[:npos].sum() - npos * (npos + 1) / 2) / (npos * (~lab).sum())


def block_B():
    hdr("B. IS THE AUC REVERSAL MECHANISM OR RANGE RESTRICTION?")
    fn = 'wave6_H_mediator.json'
    if not os.path.exists(fn):
        print("  %s not found; skipping" % fn)
        return
    H = json.load(open(fn, encoding='utf-8'))
    rows = []
    for rec in H:
        for rn in ('dorsal', 'ventral'):
            R = rec['regimes'][rn]
            rows.append({'regime': rn, 'floor': R['baseline_floor_frac'],
                         'switch': R['max_switch_rate'],
                         'aol': R['alpha_over_lambda'],
                         'sigma': rec['params']['sigma'],
                         'beta': rec['params']['beta']})
    fl = np.array([r['floor'] for r in rows])
    sw = np.array([r['switch'] for r in rows])
    ao = np.array([r['aol'] for r in rows])
    sg = np.array([r['sigma'] for r in rows])
    rg = np.array([r['regime'] for r in rows])
    lab = sw > 0.05

    print("  alpha/lambda range within each stratification\n")
    strata = [
        ('regime label (as published)', [('dorsal', rg == 'dorsal'),
                                         ('ventral', rg == 'ventral')]),
        ('MEASURED floor occupancy', [('low floor', fl < np.median(fl)),
                                      ('high floor', fl >= np.median(fl))]),
        ('sigma median split', [('low sigma', sg < np.median(sg)),
                                ('high sigma', sg >= np.median(sg))]),
    ]
    res = {}
    for sname, groups in strata:
        print("  %s" % sname)
        vals = []
        for gname, m in groups:
            if m.sum() < 10:
                continue
            a_aol = _auc(-ao[m], lab[m])
            a_fl = _auc(-fl[m], lab[m])
            rng = (float(np.min(ao[m])), float(np.max(ao[m])))
            print("    %-12s n=%3d  a/l range [%.2f, %.2f]  "
                  "AUC a/l = %.3f   AUC floor = %.3f"
                  % (gname, m.sum(), rng[0], rng[1], a_aol, a_fl))
            vals.append(a_aol)
            res.setdefault(sname, {})[gname] = {
                'n': int(m.sum()), 'aol_range': rng,
                'auc_aol': float(a_aol), 'auc_floor': float(a_fl)}
        if len(vals) == 2 and all(np.isfinite(vals)):
            flip = (vals[0] - 0.5) * (vals[1] - 0.5) < 0
            print("    -> AUC reverses across strata: %s\n" % ("YES" if flip else "no"))
            res[sname]['reverses'] = bool(flip)
        else:
            print()
    print("  The reversal is evidence of mechanism only if it survives a")
    print("  stratification that does not truncate the predictor's own range.")
    OUT['regime_control'] = res


# =============================================================================
# BLOCK C
# =============================================================================

def block_C():
    hdr("C. CEILING AUDIT")
    fn = 'phase1_grid_results.json'
    if not os.path.exists(fn):
        print("  %s not found; skipping" % fn)
        return
    grid = json.load(open(fn, encoding='utf-8'))
    eq, dom, prm = [], [], []
    for c in grid['results']:
        try:
            lam, beta = c['lambda'], c['beta']
            al, kp, gm = c['alpha'], c['kappa'], c['gamma']
        except KeyError:
            continue
        ad = al * kp / gm
        x_eq = 0.5 / (lam + beta + ad)          # equidominance
        x_dom = 0.5 / (lam + ad)                # rival suppressed, beta drops out
        eq.append(x_eq)
        dom.append(x_dom)
        prm.append((lam, beta, al, kp, gm))
    eq, dom = np.array(eq), np.array(dom)
    print("  steady-state activation across the %d-point grid" % len(eq))
    print("    at equidominance : median %.2f   max %.2f   headroom to 5.0 = %.1fx"
          % (np.median(eq), eq.max(), X_MAX / eq.max()))
    print("    during dominance : median %.2f   max %.2f   headroom to 5.0 = %.1fx"
          % (np.median(dom), dom.max(), X_MAX / dom.max()))
    n_over = int(np.sum(dom > X_MAX))
    print("    configurations whose dominance-phase steady state EXCEEDS x_max:"
          " %d/%d (%.1f%%)" % (n_over, len(dom), 100 * n_over / len(dom)))
    i = int(np.argmax(dom))
    print("    worst case: lam=%.2f beta=%.2f alpha=%.2f kappa=%.4f gamma=%.2f"
          "  -> x_dom = %.2f" % (prm[i] + (dom[i],)))
    print("\n  Section 2.1 claims roughly threefold headroom. The correct")
    print("  statement is whatever this prints, and the dominance-phase value")
    print("  is the one that matters because that is when the ceiling binds.")
    OUT['ceiling'] = {'median_eq': float(np.median(eq)),
                      'max_eq': float(eq.max()),
                      'median_dom': float(np.median(dom)),
                      'max_dom': float(dom.max()),
                      'n_over_xmax': n_over, 'n_total': len(dom)}


def figure(D):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    res = OUT.get('formulations', {})
    names = list(res.keys())
    med = [res[k]['rival_median'] for k in names]
    lo = [res[k]['rival_median'] - res[k]['rival_iqr'][0] for k in names]
    hi = [res[k]['rival_iqr'][1] - res[k]['rival_median'] for k in names]
    x = np.arange(len(names))
    cols = ['#2c6fbb' if m > 0 else '#d1495b' for m in med]
    ax[0].bar(x, med, 0.6, yerr=[lo, hi], capsize=3, color=cols)
    ax[0].axhline(0, c='k', lw=1)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(names, rotation=30, ha='right', fontsize=7)
    ax[0].set_ylabel('% change, rival channel')
    ax[0].set_title('A  Rival response at matched own change')

    alt = [res[k]['alt_median'] for k in names]
    prd = [res[k]['pred_median'] for k in names]
    ax[1].scatter(prd, alt, s=70, c=cols, zorder=3)
    for i, k in enumerate(names):
        ax[1].annotate(k, (prd[i], alt[i]), fontsize=6,
                       xytext=(4, 3), textcoords='offset points')
    ax[1].axhline(0, ls='--', c='k', lw=1)
    ax[1].axvline(0, ls='--', c='k', lw=1)
    ax[1].set_xlabel('% change, predominance')
    ax[1].set_ylabel('% change, alternation rate')
    ax[1].set_title('B  Which measure each formulation acts on')
    fig.tight_layout()
    fig.savefig('wave14_fig.pdf')
    print("\n  wave14_fig.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analyse', action='store_true')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    n_cfg, n_seed, n_steps = (8, 8, 4000) if a.quick else (100, 40, 20000)

    cfgs = load_configs()
    if a.analyse:
        D = json.load(open('wave14_A_modes.json', encoding='utf-8'))
    else:
        pool, _ = w6.eligible_pool(list(cfgs.values()), n_cfg)
        D = block_A(pool, n_seed, n_steps)
    analyse_A(D)
    block_B()
    block_C()
    figure(D)
    json.dump(OUT, open('wave14_results.json', 'w', encoding='utf-8'),
              indent=1, default=float)
    hdr("DONE")
    print("  wave14_results.json")


if __name__ == '__main__':
    main()
