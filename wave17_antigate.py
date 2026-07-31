"""
wave17_antigate.py -- promote the coupling variable from correlate to cause.

THE PROBLEM
    Dominance is defined as x_A - x_B > theta. A rival episode ends when x_A
    rises relative to x_B. The predictor in Section 4.7.1 is x_A averaged over
    exactly those episodes, and the averaging window IS the outcome. Predictor
    and outcome are two readings of the same threshold crossing, so rho = -0.970
    is close to what the extraction rule guarantees rather than evidence for a
    mechanism.

BLOCK A  THE ANTI-GATE
    The fix is to set the predictor by DESIGN rather than measure it. Section
    4.7.4 already delivers an increment only while the attended channel is
    DOMINANT (the gate) and obtains same-sign coupling. The complement is to
    deliver the same increment only while the attended channel is SUPPRESSED.

    That raises suppressed-phase activation without raising dominance-phase
    activation at all, so the account predicts strongly NEGATIVE coupling, more
    negative than an ungated increment of the same time-averaged magnitude.
    Nothing else in the paper predicts this: a state-dependence account predicts
    nothing, because the anti-gated increment is state-gated too.

    Three conditions on one physical manipulation -- gated, ungated, anti-gated
    -- give a causal series with gate timing as the independent variable.

BLOCK B  IS THE 45% BOUND DERIVABLE?
    Persistence modulation removes at most lambda from the steady-state
    denominator, so the maximum fractional activation gain is bounded by
    lambda / (lambda + beta + alpha*kappa/gamma). If that is the mechanism, the
    configurations that cannot reach a +50% duration increase should be those
    with the smallest values of it -- and the bound is independent of both x_max
    and the 0.95*lambda constraint, which is why relaxing them changed nothing.

BLOCK C  FIXED-WINDOW PREDICTOR
    Partial fix for the window-boundary coupling: measure the attended channel's
    activation over a FIXED window after each rival-dominance onset, constant
    across conditions, rather than over the whole episode. Breaks the
    window-length dependence though not the definitional one.

BLOCK D  CLUSTERED STATISTICS
    n = 220 is 44 configurations x 5 formulations. Report the within-
    configuration correlation and the distribution of per-configuration slopes
    rather than a raw Spearman on clustered data.

Run:  python wave17_antigate.py   (~4 min)  |  --analyse  |  --quick
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
    run_trace, extract_durations, _rectify,
    BURN_IN, MARGIN, X_MAX, G_SAFETY, SIGNAL_NEUTRAL, load_configs,
)
import wave6_robustness as w6

FIXED_W = 50          # fixed window after rival-dominance onset
GATES = {0: 'ungated', 1: 'gated (attended dominant)',
         2: 'anti-gated (attended suppressed)'}
MAGS = [0.5, 1.0, 2.0]
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
def run_gate(lam, beta, alpha, sigma, gamma_adapt, kappa, sig_a, sig_b,
             boost, gate, n_steps, seed, x_max, margin, fixed_w):
    """
    gate 0 : increment applied throughout
         1 : only while the attended channel A is DOMINANT
         2 : only while the attended channel A is SUPPRESSED  (the anti-gate)

    Returns traces, mean x_A over whole rival-dominance episodes, mean x_A over a
    FIXED window after each rival-dominance onset, and the time-averaged
    increment actually delivered.
    """
    np.random.seed(seed)
    ta = np.empty(n_steps, dtype=np.float64)
    tb = np.empty(n_steps, dtype=np.float64)
    x_a = 0.1
    x_b = 0.1
    a_a = 0.0
    a_b = 0.0
    supp_sum = 0.0
    n_supp = 0
    win_sum = 0.0
    n_win = 0
    since_onset = -1
    prev_rival = 0
    dose = 0.0
    n_cnt = 0

    for t in range(n_steps):
        ta[t] = x_a
        tb[t] = x_b
        d = x_a - x_b
        if gate == 0:
            b_cur = boost
        elif gate == 1:
            b_cur = boost if d > margin else 0.0
        else:
            b_cur = boost if d < -margin else 0.0

        ea = np.random.normal(0.0, sigma)
        eb = np.random.normal(0.0, sigma)
        na = (1.0 - lam) * x_a + sig_a + b_cur - beta * x_b - alpha * a_a + ea
        nb = (1.0 - lam) * x_b + sig_b - beta * x_a - alpha * a_b + eb
        x_a = _rectify(na, -1.0, x_max)
        x_b = _rectify(nb, -1.0, x_max)
        a_a = (1.0 - gamma_adapt) * a_a + kappa * x_a
        a_b = (1.0 - gamma_adapt) * a_b + kappa * x_b

        if t >= BURN_IN:
            dose += b_cur
            n_cnt += 1
            rival = 1 if (x_b - x_a) > margin else 0
            if rival == 1:
                supp_sum += x_a
                n_supp += 1
                if prev_rival == 0:
                    since_onset = 0
                if 0 <= since_onset < fixed_w:
                    win_sum += x_a
                    n_win += 1
                    since_onset += 1
            else:
                since_onset = -1
            prev_rival = rival

    return (ta, tb,
            supp_sum / n_supp if n_supp else 0.0,
            win_sum / n_win if n_win else 0.0,
            dose / n_cnt if n_cnt else 0.0)


def summarise(ta, tb):
    ch, dur = extract_durations(ta, tb, BURN_IN, MARGIN)
    if len(ch) > 2:
        ch, dur = ch[1:-1], dur[1:-1].astype(float)
    else:
        ch, dur = np.empty(0, np.int32), np.empty(0, float)
    da, db = dur[ch == 0], dur[ch == 1]
    return {'own': float(da.mean()) if len(da) else np.nan,
            'rival': float(db.mean()) if len(db) else np.nan}


def cond(p, boost, gate, key, n_seeds, n_steps):
    rows, sp, wn, ds = [], [], [], []
    for s in range(n_seeds):
        ta, tb, su, wi, do = run_gate(
            p['lambda'], p['beta'], p['alpha'], p['sigma'], p['gamma'],
            p['kappa'], SIGNAL_NEUTRAL, SIGNAL_NEUTRAL, boost, gate,
            n_steps, w6.make_seed(*key, s), X_MAX, MARGIN, FIXED_W)
        rows.append(summarise(ta, tb))
        sp.append(su)
        wn.append(wi)
        ds.append(do)

    def f(k):
        v = [r[k] for r in rows if np.isfinite(r[k])]
        return float(np.mean(v)) if v else np.nan
    return {'own': f('own'), 'rival': f('rival'),
            'supp': float(np.mean(sp)), 'win': float(np.mean(wn)),
            'dose': float(np.mean(ds))}


def _save(t, o):
    fn = "wave17_%s.json" % t
    json.dump(o, open(fn, 'w', encoding='utf-8'), indent=1, default=float)
    print("    -> %s" % fn)


# =============================================================================
# BLOCK A
# =============================================================================

def block_A(pool, n_seeds, n_steps):
    hdr("BLOCK A: gated / ungated / anti-gated")
    out = []
    t0 = time.time()
    for n, ent in enumerate(pool):
        p = ent['params']
        ci = ent['grid_index'] % 1000
        acts = []
        for s in range(20):
            t_a, _, _, _, _ = run_gate(
                p['lambda'], p['beta'], p['alpha'], p['sigma'], p['gamma'],
                p['kappa'], SIGNAL_NEUTRAL, SIGNAL_NEUTRAL, 0.0, 0,
                n_steps, w6.make_seed(0, ci, 0, 0, s), X_MAX, MARGIN, FIXED_W)
            acts.append(float(np.mean(t_a[BURN_IN:])))
        x_base = float(np.mean(acts))
        b_unit = 0.5 * p['lambda'] * x_base
        rec = {'grid_index': ent['grid_index'], 'params': p,
               'x_baseline': x_base, 'b_unit': b_unit, 'cond': {}}
        rec['baseline'] = cond(p, 0.0, 0, (1, ci, 0, 0), n_seeds, n_steps)
        li = 1
        for gate in GATES:
            for mag in MAGS:
                rec['cond']['%d_%g' % (gate, mag)] = cond(
                    p, mag * b_unit, gate, (2, ci, gate, li), n_seeds, n_steps)
                li += 1
        out.append(rec)
        if (n + 1) % 20 == 0:
            print("    %3d/%d  (%.0fs)" % (n + 1, len(pool), time.time() - t0))
    _save('A_gates', out)
    return out


def analyse_A(D):
    hdr("A. THE CAUSAL SERIES")
    print("  The account predicts: raising the attended channel's activation")
    print("  while it is SUPPRESSED shortens the competitor. The anti-gate does")
    print("  exactly that by construction, so its coupling should be the most")
    print("  negative of the three. The predictor is set by design, not measured.\n")
    print("  %-34s %9s %9s %9s %9s %8s"
          % ("condition", "own", "rival", "supp act", "fixed win", "dose"))
    res = {}
    for gate, gname in GATES.items():
        for mag in MAGS:
            o, r, su, wi, ds = [], [], [], [], []
            for rec in D:
                b = rec['baseline']
                c = rec['cond'].get('%d_%g' % (gate, mag))
                if not (b['own'] and b['rival'] and c):
                    continue
                o.append(pct(c['own'], b['own']))
                r.append(pct(c['rival'], b['rival']))
                su.append(pct(c['supp'], b['supp']))
                wi.append(pct(c['win'], b['win']))
                ds.append(c['dose'] / rec['b_unit'] if rec['b_unit'] else np.nan)
            f = lambda v: float(np.nanmedian(v)) if v else np.nan
            npos = int(np.nansum(np.array(r) > 0))
            print("  %-34s %+8.1f%% %+8.1f%% %+8.1f%% %+8.1f%% %7.2f"
                  % ("%s  %gx" % (gname, mag), f(o), f(r), f(su), f(wi), f(ds)))
            res['%d_%g' % (gate, mag)] = {
                'gate': gname, 'mag': mag, 'n': len(r), 'own': f(o),
                'rival': f(r), 'supp': f(su), 'win': f(wi),
                'n_same_sign': npos}
    OUT['gates'] = res

    print("\n  ordering test at matched time-averaged dose")
    print("  (gated and anti-gated deliver roughly half the ungated dose,")
    print("   so compare gated 2x / ungated 1x / anti-gated 2x)")
    trio = [('1_2', 'gated 2x'), ('0_1', 'ungated 1x'), ('2_2', 'anti-gated 2x')]
    xs, ys = [], []
    for k, lbl in trio:
        v = res.get(k)
        if v:
            print("    %-16s supp act %+7.1f%%   rival %+7.1f%%"
                  % (lbl, v['supp'], v['rival']))
            xs.append(v['supp'])
            ys.append(v['rival'])
    if len(xs) == 3:
        mono = (xs[0] < xs[1] < xs[2]) and (ys[0] > ys[1] > ys[2])
        print("\n    monotone in the predicted direction: %s" % ("YES" if mono else "no"))
        OUT['series_monotone'] = bool(mono)

    print("\n  per-configuration test: does the anti-gate give more negative")
    print("  coupling than the ungated increment in the same configuration?")
    pa, pb = [], []
    for rec in D:
        b = rec['baseline']
        ca = rec['cond'].get('2_2')
        cb = rec['cond'].get('0_1')
        if not (b['rival'] and ca and cb):
            continue
        pa.append(pct(ca['rival'], b['rival']))
        pb.append(pct(cb['rival'], b['rival']))
    pa, pb = np.array(pa, float), np.array(pb, float)
    m = np.isfinite(pa) & np.isfinite(pb)
    if m.sum() > 10:
        t, pv = stats.ttest_rel(pa[m], pb[m])
        d = (pa[m] - pb[m]).mean() / (pa[m] - pb[m]).std(ddof=1)
        n_lower = int(np.sum(pa[m] < pb[m]))
        lo, hi = w6.wilson(n_lower, int(m.sum()))
        print("    anti-gate %+.1f%%  vs ungated %+.1f%%   d = %+.2f  p = %.2g"
              % (np.median(pa[m]), np.median(pb[m]), d, pv))
        print("    anti-gate more negative in %d/%d  [%.0f%%, %.0f%%]"
              % (n_lower, m.sum(), 100 * lo, 100 * hi))
        OUT['antigate_vs_ungated'] = {'d': float(d), 'p': float(pv),
                                      'n_lower': n_lower, 'n': int(m.sum())}


# =============================================================================
# BLOCK B
# =============================================================================

def block_B(D):
    hdr("B. IS THE 45% BOUND DERIVABLE?")
    print("  Persistence modulation removes at most lambda from the steady-state")
    print("  denominator, so the maximum fractional activation gain is bounded by")
    print("  lambda / (lambda + beta + alpha*kappa/gamma).\n")
    fn = 'wave15_A_paired.json'
    if not os.path.exists(fn):
        print("  wave15_A_paired.json not found; skipping")
        return
    P = json.load(open(fn, encoding='utf-8'))
    rows = []
    for rec in P:
        b = rec['baseline']
        if not b['own']:
            continue
        p = rec['params']
        bnd = p['lambda'] / (p['lambda'] + p['beta']
                             + p['alpha'] * p['kappa'] / p['gamma'])
        best = -1e9
        for c in rec['modes']['1']:
            v = pct(c['own'], b['own'])
            if np.isfinite(v):
                best = max(best, v)
        if best > -1e8:
            rows.append({'bound': bnd, 'max_own': best,
                         'reaches50': best >= 50.0})
    bd = np.array([r['bound'] for r in rows])
    mx = np.array([r['max_own'] for r in rows])
    rc = np.array([r['reaches50'] for r in rows])
    rho, pv = stats.spearmanr(bd, mx)
    print("  n = %d configurations" % len(rows))
    print("  rho(analytic bound, maximum achievable own-duration gain) = %+.3f"
          "  p = %.2g" % (rho, pv))
    print("  reaches +50%%: %d/%d" % (int(rc.sum()), len(rc)))
    if rc.sum() >= 5 and (~rc).sum() >= 5:
        print("    bound among reaching     : median %.3f  IQR [%.3f, %.3f]"
              % (np.median(bd[rc]), np.percentile(bd[rc], 25),
                 np.percentile(bd[rc], 75)))
        print("    bound among not reaching : median %.3f  IQR [%.3f, %.3f]"
              % (np.median(bd[~rc]), np.percentile(bd[~rc], 25),
                 np.percentile(bd[~rc], 75)))
        t, p2 = stats.mannwhitneyu(bd[rc], bd[~rc], alternative='greater')
        print("    Mann-Whitney (reaching > not reaching): p = %.2g" % p2)
    OUT['bound'] = {'rho': float(rho), 'p': float(pv), 'n': len(rows),
                    'n_reach': int(rc.sum())}
    print("\n  A strong positive correlation converts an admitted unexplained")
    print("  failure into a derived architectural bound, and predicts that large")
    print("  attentional effects require high-leak, low-inhibition regimes.")


# =============================================================================
# BLOCK C / D
# =============================================================================

def block_CD(D):
    hdr("C. FIXED-WINDOW PREDICTOR")
    print("  Whole-episode averaging couples the predictor to the outcome through")
    print("  the window boundaries. A fixed %d-step window after rival onset"
          % FIXED_W)
    print("  breaks that dependence, though not the definitional one.\n")
    for lbl, k in (('whole episode', 'supp'), ('fixed %d-step window' % FIXED_W, 'win')):
        xs, ys = [], []
        for rec in D:
            b = rec['baseline']
            if not (b['rival'] and b[k]):
                continue
            for key, c in rec['cond'].items():
                x = pct(c[k], b[k])
                y = pct(c['rival'], b['rival'])
                if np.isfinite(x) and np.isfinite(y):
                    xs.append(x)
                    ys.append(y)
        rho, pv = stats.spearmanr(xs, ys)
        print("    %-26s rho = %+.3f  p = %.2g  n = %d" % (lbl, rho, pv, len(xs)))
        OUT.setdefault('window', {})[lbl] = {'rho': float(rho), 'n': len(xs)}

    hdr("D. CLUSTERED STATISTICS")
    print("  per-configuration slopes of rival response on suppressed-phase")
    print("  activation, across the nine gate x magnitude conditions\n")
    slopes, rhos = [], []
    for rec in D:
        b = rec['baseline']
        if not (b['rival'] and b['supp']):
            continue
        xs = [pct(c['supp'], b['supp']) for c in rec['cond'].values()]
        ys = [pct(c['rival'], b['rival']) for c in rec['cond'].values()]
        xs, ys = np.array(xs, float), np.array(ys, float)
        m = np.isfinite(xs) & np.isfinite(ys)
        if m.sum() < 5 or len(np.unique(xs[m])) < 4:
            continue
        slopes.append(float(np.polyfit(xs[m], ys[m], 1)[0]))
        rhos.append(float(stats.spearmanr(xs[m], ys[m])[0]))
    slopes, rhos = np.array(slopes), np.array(rhos)
    n_neg = int(np.sum(slopes < 0))
    lo, hi = w6.wilson(n_neg, len(slopes))
    print("    n = %d configurations with usable slopes" % len(slopes))
    print("    slope   : median %+.3f  IQR [%+.3f, %+.3f]  negative in %d/%d"
          " [%.0f%%, %.0f%%]"
          % (np.median(slopes), np.percentile(slopes, 25),
             np.percentile(slopes, 75), n_neg, len(slopes),
             100 * lo, 100 * hi))
    print("    within-config rho: median %+.3f  IQR [%+.3f, %+.3f]"
          % (np.median(rhos), np.percentile(rhos, 25),
             np.percentile(rhos, 75)))
    t, pv = stats.wilcoxon(slopes)
    print("    Wilcoxon on slopes vs zero: p = %.2g" % pv)
    OUT['clustered'] = {'n': len(slopes), 'median_slope': float(np.median(slopes)),
                        'n_negative': n_neg,
                        'median_rho': float(np.median(rhos)),
                        'wilcoxon_p': float(pv)}


def figure(D):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    cols = {0: '#d1495b', 1: '#2c6fbb', 2: '#4c9f70'}
    for gate, gname in GATES.items():
        xs, ys = [], []
        for rec in D:
            b = rec['baseline']
            if not (b['rival'] and b['supp']):
                continue
            for mag in MAGS:
                c = rec['cond'].get('%d_%g' % (gate, mag))
                if not c:
                    continue
                x = pct(c['supp'], b['supp'])
                y = pct(c['rival'], b['rival'])
                if np.isfinite(x) and np.isfinite(y):
                    xs.append(x)
                    ys.append(y)
        ax[0].scatter(xs, ys, s=15, alpha=0.5, c=cols[gate],
                      label=gname, edgecolors='none')
    ax[0].axhline(0, c='k', lw=0.7)
    ax[0].axvline(0, c='k', lw=0.7)
    ax[0].set_xlabel('% change in attended-channel activation while suppressed',
                     fontsize=8)
    ax[0].set_ylabel("% change, competitor's duration", fontsize=8)
    ax[0].set_title('A  Gate timing sets the predictor by design', fontsize=10)
    ax[0].legend(fontsize=6.5, frameon=False)

    g = OUT.get('gates', {})
    order = [('1_2', 'gated'), ('0_1', 'ungated'), ('2_2', 'anti-gated')]
    xs = [g[k]['supp'] for k, _ in order if k in g]
    ys = [g[k]['rival'] for k, _ in order if k in g]
    ax[1].plot(xs, ys, 'o-', ms=10, lw=2, c='#2c6fbb')
    for (k, lbl), x, y in zip(order, xs, ys):
        ax[1].annotate(lbl, (x, y), fontsize=8, xytext=(6, 5),
                       textcoords='offset points')
    ax[1].axhline(0, c='k', lw=0.7)
    ax[1].set_xlabel('% change in suppressed-phase activation', fontsize=8)
    ax[1].set_ylabel("% change, competitor's duration", fontsize=8)
    ax[1].set_title('B  One manipulation, three gate timings', fontsize=10)
    fig.tight_layout()
    fig.savefig('wave17_fig_antigate.pdf')
    fig.savefig('wave17_fig_antigate.png', dpi=200)
    print("\n  wave17_fig_antigate.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analyse', action='store_true')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    n_cfg, n_seed, n_steps = (8, 8, 4000) if a.quick else (100, 40, 20000)
    cfgs = load_configs()
    if a.analyse:
        D = json.load(open('wave17_A_gates.json', encoding='utf-8'))
    else:
        pool, _ = w6.eligible_pool(list(cfgs.values()), n_cfg)
        D = block_A(pool, n_seed, n_steps)
    analyse_A(D)
    block_B(D)
    block_CD(D)
    figure(D)
    json.dump(OUT, open('wave17_results.json', 'w', encoding='utf-8'),
              indent=1, default=float)
    hdr("DONE")
    print("  wave17_results.json")


if __name__ == '__main__':
    main()
