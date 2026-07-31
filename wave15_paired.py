"""
wave15_paired.py -- the objections that could sink the central claim.

BLOCK A  PAIRED FORMULATION COMPARISON, AND THE DUTY-CYCLE PREDICTOR
    Section 4.7.1 compared five formulations at denominators of 62, 60, 84, 92
    and 67 out of 100. Reachability of the outcome target correlates with the
    parameters that carry the coupling, so the class contrast compares different
    subsets of parameter space rather than different mechanisms. Here the same
    five are run on every configuration and compared on the INTERSECTION where
    all five reach the target, as a within-configuration paired analysis.

    We also test a better abstraction than "state-dependence". A binary gate on
    an ordinary increment reproduces the goal-signal signature (Section 4.7.3),
    so the operative property may be that the modulation is OFF during the
    rival's dominance rather than that it scales with x. The candidate predictor
    is the attended channel's mean activation during rival-dominance episodes,
    relative to baseline. If that orders all five formulations including the
    adaptation exception, it supersedes both "state-dependence" and kappa/gamma.

BLOCK B  DOES THE CONTRAST SURVIVE WHERE THE INCREMENT ARM IS CORRECT?
    The model mis-predicts Levelt II by the full size of the effect: fitted to a
    -30% rival compression it gives +39% on the manipulated channel where ~0% is
    observed. The adaptation term responsible is the same term that carries
    same-sign coupling. This searches for configurations where an ungated
    increment reproduces approximately 0% / -30%, then applies the goal signal in
    exactly those configurations. If same-sign coupling survives there the result
    is real; if it does not, the two signatures are an artefact of the model's
    failure to reproduce stimulus-strength effects.

BLOCK C  UNFILTERED SAMPLE
    The principal sample is drawn from the 8.5% of the grid passing a CV filter,
    and CV is set by the adaptation/noise balance that carries the coupling.
    Repeat the signature contrast on a random draw from all 6,814
    rivalry-producing configurations with no CV or rho filter.

BLOCK D  CEILING VERSUS DISSIPATIVITY
    Rerun the goal-signal sweep with x_max = 20 and G up to 0.99 lambda, and
    recount how many configurations cannot reach a 50% increase.

Alternation rate is logged both unfiltered and with the registered 5-timestep
minimum throughout, since the transitional population of margin crossings may be
carrying the alternation-rate result.

Run:  python wave15_paired.py   (~10 min)  |  --analyse  |  --quick
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

TARGET_OWN = 30.0
MIN_DUR = 5
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
def run_m(lam, beta, alpha, sigma, gamma_adapt, kappa, sig_a, sig_b,
          mode, m, n_steps, seed, x_max, margin):
    """
    Returns traces plus the attended channel's mean activation during
    RIVAL-dominance episodes (the duty-cycle quantity) and ceiling occupancy.
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
    supp_sum = 0.0
    n_supp = 0
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
            if (x_b - x_a) > margin:          # rival dominant
                supp_sum += x_a
                n_supp += 1
            hi = x_a if x_a > x_b else x_b
            if hi >= x_max - 1e-9:
                n_ceil += 1
            n_cnt += 1
    return (ta, tb,
            supp_sum / n_supp if n_supp else 0.0,
            n_ceil / n_cnt if n_cnt else 0.0)


def summarise(ta, tb):
    ch, dur = extract_durations(ta, tb, BURN_IN, MARGIN)
    n = len(ta) - BURN_IN
    if len(ch) > 2:
        ch, dur = ch[1:-1], dur[1:-1].astype(float)
    else:
        ch, dur = np.empty(0, np.int32), np.empty(0, float)
    da, db = dur[ch == 0], dur[ch == 1]
    nsw_raw = int(np.sum(ch[1:] != ch[:-1])) if len(ch) > 1 else 0
    k = dur >= MIN_DUR
    cf = ch[k]
    nsw_f = int(np.sum(cf[1:] != cf[:-1])) if len(cf) > 1 else 0
    tot = da.sum() + db.sum()
    return {'own': float(da.mean()) if len(da) else np.nan,
            'rival': float(db.mean()) if len(db) else np.nan,
            'pred': float(da.sum() / tot) if tot else np.nan,
            'alt': nsw_raw / n, 'alt_f': nsw_f / n}


def cond(p, mode, m, key, n_seeds, n_steps, x_max=X_MAX):
    rows, sp, ce = [], [], []
    for s in range(n_seeds):
        ta, tb, su, c = run_m(p['lambda'], p['beta'], p['alpha'], p['sigma'],
                              p['gamma'], p['kappa'], SIGNAL_NEUTRAL,
                              SIGNAL_NEUTRAL, mode, m, n_steps,
                              w6.make_seed(*key, s), x_max, MARGIN)
        rows.append(summarise(ta, tb))
        sp.append(su)
        ce.append(c)

    def f(k):
        v = [r[k] for r in rows if np.isfinite(r[k])]
        return float(np.mean(v)) if v else np.nan
    return {'own': f('own'), 'rival': f('rival'), 'pred': f('pred'),
            'alt': f('alt'), 'alt_f': f('alt_f'),
            'supp_act': float(np.mean(sp)), 'ceil': float(np.mean(ce))}


def levels_for(mode, lam, x_base):
    if mode == 1:
        return [f * lam for f in (0.1, 0.25, 0.5, 0.75, 0.9)]
    if mode == 2:
        return [f * lam * x_base for f in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)]
    if mode == 3:
        return [0.005, 0.01, 0.02, 0.05, 0.10]
    if mode == 4:
        return [0.1, 0.25, 0.5, 1.0, 2.0]
    if mode == 5:
        return [0.1, 0.25, 0.5, 0.75, 0.9]
    return [0.0]


def _save(tag, obj):
    fn = "wave15_%s.json" % tag
    json.dump(obj, open(fn, 'w', encoding='utf-8'), indent=1, default=float)
    print("    -> %s" % fn)


def _load(tag):
    fn = "wave15_%s.json" % tag
    if not os.path.exists(fn):
        sys.exit("ERROR: %s missing." % fn)
    return json.load(open(fn, encoding='utf-8'))


def baseline_activation(p, key, n_steps, n=20):
    a = []
    for s in range(n):
        t_a, _, _, _ = run_m(p['lambda'], p['beta'], p['alpha'], p['sigma'],
                             p['gamma'], p['kappa'], SIGNAL_NEUTRAL,
                             SIGNAL_NEUTRAL, 0, 0.0, n_steps,
                             w6.make_seed(*key, s), X_MAX, MARGIN)
        a.append(float(np.mean(t_a[BURN_IN:])))
    return float(np.mean(a))


# =============================================================================
# BLOCK A
# =============================================================================

def block_A(pool, n_seeds, n_steps):
    hdr("BLOCK A: five formulations on every configuration")
    out = []
    t0 = time.time()
    for n, ent in enumerate(pool):
        p = ent['params']
        ci = ent['grid_index'] % 1000
        x_base = baseline_activation(p, (0, ci, 0, 0), n_steps)
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
    _save('A_paired', out)
    return out


def interp_at(lv, b, key):
    xs = np.array([pct(c['own'], b['own']) for c in lv], float)
    ys = np.array([pct(c[key], b[key]) if key in ('rival', 'alt', 'alt_f',
                                                  'pred')
                   else pct(c[key], b[key]) for c in lv], float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 3:
        return np.nan
    o = np.argsort(xs[m])
    xx, yy = xs[m][o], ys[m][o]
    if not (xx.min() <= TARGET_OWN <= xx.max()):
        return np.nan
    return float(np.interp(TARGET_OWN, xx, yy))


def analyse_A(D):
    hdr("A1. PAIRED COMPARISON ON THE INTERSECTION")
    per = {}
    for mode in MODES:
        per[mode] = {}
        for rec in D:
            b = rec['baseline']
            if not (b['own'] and b['rival'] and b['alt'] and b['alt_f']):
                continue
            lv = rec['modes'][str(mode)]
            per[mode][rec['grid_index']] = {
                'rival': interp_at(lv, b, 'rival'),
                'alt': interp_at(lv, b, 'alt'),
                'alt_f': interp_at(lv, b, 'alt_f'),
                'pred': interp_at(lv, b, 'pred'),
                'supp': interp_at(lv, b, 'supp_act')}
    inter = None
    for mode in MODES:
        ok = {g for g, v in per[mode].items() if np.isfinite(v['rival'])}
        inter = ok if inter is None else (inter & ok)
        print("  %-18s reaches target in %3d configurations"
              % (MODES[mode], len(ok)))
    inter = sorted(inter)
    print("\n  INTERSECTION: %d configurations reach the target in all five\n"
          % len(inter))
    if len(inter) < 10:
        print("  too few for a paired analysis")
        return per, inter

    print("  %-18s %11s %20s %12s %12s"
          % ("formulation", "rival med", "rival IQR", "alt (raw)", "alt (min 5)"))
    res = {}
    for mode in MODES:
        rv = np.array([per[mode][g]['rival'] for g in inter], float)
        al = np.array([per[mode][g]['alt'] for g in inter], float)
        af = np.array([per[mode][g]['alt_f'] for g in inter], float)
        q = np.nanpercentile(rv, [25, 50, 75])
        npos = int(np.nansum(rv > 0))
        lo, hi = w6.wilson(npos, int(np.sum(np.isfinite(rv))))
        print("  %-18s %+10.1f%% [%+7.1f%%, %+7.1f%%] %+11.1f%% %+11.1f%%"
              % (MODES[mode], q[1], q[0], q[2], np.nanmedian(al),
                 np.nanmedian(af)))
        print("  %-18s        same sign %d/%d [%.0f%%, %.0f%%]"
              % ("", npos, int(np.sum(np.isfinite(rv))), 100 * lo, 100 * hi))
        res[MODES[mode]] = {'n': len(inter), 'rival_median': float(q[1]),
                            'rival_iqr': [float(q[0]), float(q[2])],
                            'alt_raw': float(np.nanmedian(al)),
                            'alt_filtered': float(np.nanmedian(af)),
                            'n_same_sign': npos}
    # paired contrasts against the increment
    print("\n  paired contrasts against the input increment (same configurations)")
    base = np.array([per[2][g]['rival'] for g in inter], float)
    for mode in (1, 3, 4, 5):
        v = np.array([per[mode][g]['rival'] for g in inter], float)
        m = np.isfinite(v) & np.isfinite(base)
        if m.sum() < 6:
            continue
        t, pv = stats.ttest_rel(v[m], base[m])
        d = (v[m] - base[m]).mean() / (v[m] - base[m]).std(ddof=1)
        print("    %-18s d = %+.2f  p = %.2g  higher in %d/%d"
              % (MODES[mode], d, pv, int(np.sum(v[m] > base[m])), m.sum()))
    OUT['paired'] = res

    hdr("A2. DOES DUTY CYCLE ORDER THE FORMULATIONS?")
    print("  candidate predictor: attended channel's mean activation during")
    print("  RIVAL-dominance episodes, % change from baseline\n")
    print("  %-18s %14s %14s" % ("formulation", "supp activation", "rival"))
    xs, ys = [], []
    for mode in MODES:
        su = np.nanmedian([per[mode][g]['supp'] for g in inter])
        rv = np.nanmedian([per[mode][g]['rival'] for g in inter])
        print("  %-18s %+13.1f%% %+13.1f%%" % (MODES[mode], su, rv))
        for g in inter:
            a, b_ = per[mode][g]['supp'], per[mode][g]['rival']
            if np.isfinite(a) and np.isfinite(b_):
                xs.append(a)
                ys.append(b_)
    if len(xs) > 20:
        rho, pv = stats.spearmanr(xs, ys)
        print("\n  across all formulations and configurations:")
        print("    rho(suppressed-phase activation, rival response) = %+.3f  p=%.2g  n=%d"
              % (rho, pv, len(xs)))
        OUT['duty_cycle'] = {'rho': float(rho), 'p': float(pv), 'n': len(xs)}
    print("\n  A strong negative correlation supports the duty-cycle account:")
    print("  raising the attended channel's activation while the rival is")
    print("  dominant makes it compete harder there and shortens the rival.")
    return per, inter


# =============================================================================
# BLOCK B  -- the decisive one
# =============================================================================

def block_B(D, n_seeds, n_steps):
    hdr("B. DOES THE CONTRAST SURVIVE WHERE THE INCREMENT ARM IS CORRECT?")
    print("  Target: an ungated increment giving own ~0%% and rival ~-30%%,")
    print("  which is the canonical Levelt II phenomenology.\n")
    hits = []
    for rec in D:
        b = rec['baseline']
        if not (b['own'] and b['rival']):
            continue
        lv = rec['modes']['2']
        for c in lv:
            o = pct(c['own'], b['own'])
            r = pct(c['rival'], b['rival'])
            if np.isfinite(o) and np.isfinite(r):
                if abs(o) <= 10.0 and -45.0 <= r <= -15.0:
                    hits.append({'grid_index': rec['grid_index'],
                                 'params': rec['params'], 'm': c['m'],
                                 'own': o, 'rival': r})
                    break
    print("  configurations where the increment reproduces Levelt II: %d/%d"
          % (len(hits), len(D)))
    if not hits:
        print("\n  NONE. The model cannot reproduce Levelt II at any increment")
        print("  magnitude in this sample, so the increment arm is not merely")
        print("  mis-scaled but structurally wrong. That belongs in the")
        print("  Discussion beside the increasing-duration problem.")
        OUT['levelt_valid'] = {'n': 0}
        return []
    print("    own change: median %+.1f%%   rival change: median %+.1f%%"
          % (np.median([h['own'] for h in hits]),
             np.median([h['rival'] for h in hits])))
    print("\n  applying the goal signal in exactly those configurations:")
    res = []
    for i, h in enumerate(hits):
        p = h['params']
        ci = h['grid_index'] % 1000
        b = cond(p, 0, 0.0, (5, ci, 0, 0), n_seeds, n_steps)
        if not (b['own'] and b['rival']):
            continue
        for gf in (0.25, 0.5, 0.75, 0.9):
            g = cond(p, 1, min(gf * p['lambda'], G_SAFETY * p['lambda']),
                     (6, ci, 0, int(gf * 100)), n_seeds, n_steps)
            o, r = pct(g['own'], b['own']), pct(g['rival'], b['rival'])
            if np.isfinite(o) and o > 5.0:
                res.append({'grid_index': h['grid_index'], 'g_frac': gf,
                            'own': o, 'rival': r})
                break
    if res:
        rv = np.array([x['rival'] for x in res], float)
        npos = int(np.sum(rv > 0))
        lo, hi = w6.wilson(npos, len(rv))
        print("    n = %d   rival median %+.1f%%   IQR [%+.1f%%, %+.1f%%]"
              % (len(rv), np.median(rv), np.percentile(rv, 25),
                 np.percentile(rv, 75)))
        print("    same sign as own: %d/%d  [%.0f%%, %.0f%%]"
              % (npos, len(rv), 100 * lo, 100 * hi))
        print("\n  If same-sign coupling survives here, the contrast is real.")
        print("  If it does not, the two signatures are an artefact of the")
        print("  model's failure to reproduce stimulus-strength effects.")
        OUT['levelt_valid'] = {'n_configs': len(hits), 'n_tested': len(rv),
                               'rival_median': float(np.median(rv)),
                               'n_same_sign': npos}
    _save('B_levelt_valid', {'hits': hits, 'goal': res})
    return res


# =============================================================================
# BLOCK C / D
# =============================================================================

def block_CD(cfgs, n_seeds, n_steps, n_cfg):
    hdr("C. UNFILTERED SAMPLE (no CV or rho criterion)")
    with open("phase1_grid_results.json", encoding='utf-8') as f:
        grid = json.load(f)
    keys = ['lambda', 'beta', 'alpha', 'sigma', 'gamma', 'kappa']
    pool = []
    for c in grid['results']:
        if not c.get('rivalry_producing'):
            continue
        p = {k: c[k] for k in keys if k in c}
        if len(p) == 6:
            pool.append({'params': p, 'grid_index': c.get('index', -1)})
    rng = np.random.default_rng(7)
    idx = rng.choice(len(pool), size=min(n_cfg, len(pool)), replace=False)
    sel = [pool[i] for i in sorted(idx)]
    print("  drawing %d from %d rivalry-producing configurations"
          % (len(sel), len(pool)))
    go, gr, bo, br = [], [], [], []
    for n, ent in enumerate(sel):
        p = ent['params']
        ci = ent['grid_index'] % 1000
        x_b = baseline_activation(p, (7, ci, 0, 0), n_steps)
        b = cond(p, 0, 0.0, (8, ci, 0, 0), n_seeds, n_steps)
        if not (b['own'] and b['rival']):
            continue
        g = cond(p, 1, min(0.5 * p['lambda'], G_SAFETY * p['lambda']),
                 (9, ci, 0, 1), n_seeds, n_steps)
        x = cond(p, 2, 0.5 * p['lambda'] * x_b, (9, ci, 0, 2), n_seeds, n_steps)
        go.append(pct(g['own'], b['own']))
        gr.append(pct(g['rival'], b['rival']))
        bo.append(pct(x['own'], b['own']))
        br.append(pct(x['rival'], b['rival']))
    f = lambda v: float(np.nanmedian(v))
    ng = int(np.nansum(np.array(gr) > 0))
    nb = int(np.nansum(np.array(br) > 0))
    print("    goal signal : own %+.1f%%  rival %+.1f%%   same sign %d/%d"
          % (f(go), f(gr), ng, len(gr)))
    print("    increment   : own %+.1f%%  rival %+.1f%%   same sign %d/%d"
          % (f(bo), f(br), nb, len(br)))
    print("\n  If the sign contrast holds without the CV filter, the generality")
    print("  claim in the Conclusion is earned; if not, it must be narrowed.")
    OUT['unfiltered'] = {'n': len(gr), 'goal_rival': f(gr),
                         'incr_rival': f(br), 'goal_same_sign': ng,
                         'incr_same_sign': nb}

    hdr("D. CEILING VERSUS DISSIPATIVITY")
    print("  can the model reach a +50%% own-duration increase with a raised")
    print("  ceiling and a relaxed dissipativity bound?\n")
    for lbl, xm, gmax in (('x_max = 5,  G <= 0.95L', X_MAX, 0.95),
                          ('x_max = 20, G <= 0.95L', 20.0, 0.95),
                          ('x_max = 20, G <= 0.99L', 20.0, 0.99)):
        reach = 0
        tot = 0
        for ent in sel[:min(40, len(sel))]:
            p = ent['params']
            ci = ent['grid_index'] % 1000
            b = cond(p, 0, 0.0, (10, ci, 0, 0), n_seeds, n_steps, xm)
            if not b['own']:
                continue
            tot += 1
            best = -1e9
            for gf in (0.5, 0.75, gmax):
                g = cond(p, 1, gf * p['lambda'], (11, ci, 0, int(gf * 100)),
                         n_seeds, n_steps, xm)
                v = pct(g['own'], b['own'])
                if np.isfinite(v):
                    best = max(best, v)
            if best >= 50.0:
                reach += 1
        lo, hi = w6.wilson(reach, tot) if tot else (np.nan, np.nan)
        print("    %-24s reach +50%%: %d/%d  [%.0f%%, %.0f%%]"
              % (lbl, reach, tot, 100 * lo, 100 * hi))
        OUT.setdefault('ceiling_test', {})[lbl] = {'reach': reach, 'n': tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analyse', action='store_true')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    n_cfg, n_seed, n_steps = (8, 8, 4000) if a.quick else (100, 40, 20000)

    cfgs = load_configs()
    if a.analyse:
        D = _load('A_paired')
    else:
        pool, _ = w6.eligible_pool(list(cfgs.values()), n_cfg)
        D = block_A(pool, n_seed, n_steps)
    per, inter = analyse_A(D)
    if not a.analyse:
        block_B(D, n_seed, n_steps)
        block_CD(cfgs, n_seed, n_steps, n_cfg)
    json.dump(OUT, open('wave15_results.json', 'w', encoding='utf-8'),
              indent=1, default=float)
    hdr("DONE")
    print("  wave15_results.json")


if __name__ == '__main__':
    main()
