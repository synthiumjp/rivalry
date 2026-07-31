"""
wave10_signed_G.py -- can one mechanism, sign-reversed, account for both
attentional enhancement and attentional suppression?

THE EMPIRICAL PATTERN TO REPRODUCE
  Chong et al. 2005 (enhancement, feature tracking during rivalry)
      attended  +50%   unattended  +5% ns          -> positive G
  Paffen et al. 2008, irrelevant vs NEUTRAL (suppression, training carryover)
      irrelevant OWN duration DOWN, neutral rival unchanged
                                                   -> negative G
  Levelt 1965 / Mueller & Blake 1989 (ungated contrast increment)
      manipulated ~unchanged, rival DOWN            -> positive boost

The discriminating cell is Paffen's irrelevant-vs-neutral pairing. Levelt
Proposition II cannot produce it: weakening a stimulus should lengthen its
rival's dominance and leave its own alone. It did the reverse. A negative
state-dependent term does produce it, because -G*x is zero at the floor and
therefore cannot touch suppression durations.

WHAT THIS SCRIPT TESTS
  1. Sign symmetry. Is -G the mirror of +G on the manipulated channel?
  2. Rival invariance. Does the rival stay flat across the G sweep while
     varying across the boost sweep? This is the whole claim.
  3. Alternation direction. +G should slow alternation, -G speed it up.
     Moreno-Sanchez Exp 3 reports F(1,83) = 43.59, p < .0001 for an increase.
  4. Direct comparison against the three published signatures, in % change.

Stability note: negative G increases effective leak to lambda + |G|, so the
dissipativity constraint that bounds positive G does not apply. The binding
constraint is (1 - lambda + G) > -1, comfortably satisfied over the range used.

Run:  python wave10_signed_G.py   (~3 min)  |  --analyse  |  --quick
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

from wave2_campaign import (
    run_trace, extract_durations, occupancy_stats,
    BURN_IN, MARGIN, X_MAX, G_SAFETY, SIGNAL_NEUTRAL, load_configs,
)
import wave6_robustness as w6

G_FRACS = [-0.90, -0.50, -0.25, 0.0, 0.25, 0.50, 0.90]
N_BOOT = 4000
OUT = {}


def hdr(s):
    print("\n" + "=" * 74)
    print(s)
    print("=" * 74)


def summarise(ta, tb):
    """Durations of the MANIPULATED channel (A) and its RIVAL (B)."""
    ch, dur = extract_durations(ta, tb, BURN_IN, MARGIN)
    n = len(ta) - BURN_IN
    if len(ch) > 2:
        ch, dur = ch[1:-1], dur[1:-1].astype(float)
    else:
        ch, dur = np.empty(0, np.int32), np.empty(0, float)
    da = dur[ch == 0]
    db = dur[ch == 1]
    nsw = int(np.sum(ch[1:] != ch[:-1])) if len(ch) > 1 else 0
    tot = da.sum() + db.sum()
    fl, ce, _ = occupancy_stats(ta, tb, BURN_IN, X_MAX)
    return {
        'dur_manip': float(da.mean()) if len(da) else np.nan,
        'dur_rival': float(db.mean()) if len(db) else np.nan,
        'predominance': float(da.sum() / tot) if tot else np.nan,
        'alternation': nsw / n,
        'is_wta': bool(len(da) == 0 or len(db) == 0),
        'ceil': float(ce), 'floor': float(fl),
    }


def agg(rows):
    def m(k):
        v = [r[k] for r in rows if np.isfinite(r[k])]
        return float(np.mean(v)) if v else np.nan
    return {'dur_manip': m('dur_manip'), 'dur_rival': m('dur_rival'),
            'predominance': m('predominance'), 'alternation': m('alternation'),
            'wta_rate': float(np.mean([r['is_wta'] for r in rows])),
            'ceil': m('ceil'), 'floor': m('floor')}


def campaign(pool, n_seeds, n_steps):
    hdr("SIMULATION: signed goal signal vs signed input increment")
    out = []
    t0 = time.time()
    for n, ent in enumerate(pool):
        p = ent['params']
        lam, beta, alpha = p['lambda'], p['beta'], p['alpha']
        sigma, gam, kap = p['sigma'], p['gamma'], p['kappa']
        ci = ent['grid_index'] % 1000

        acts = []
        for s in range(20):
            ta, _ = run_trace(lam, beta, alpha, sigma, gam, kap,
                              SIGNAL_NEUTRAL, SIGNAL_NEUTRAL, 0, 0, 0, 0,
                              n_steps, w6.make_seed(0, ci, 0, 0, s),
                              X_MAX, -1.0, 0.0, 0.0)
            acts.append(float(np.mean(ta[BURN_IN:])))
        x_base = float(np.mean(acts))

        rec = {'grid_index': ent['grid_index'], 'params': p,
               'x_baseline': x_base, 'G': {}, 'boost': {}}
        li = 0
        for gf in G_FRACS:
            g_val = gf * lam
            if gf > 0:
                g_val = min(g_val, G_SAFETY * lam)
            b_val = gf * lam * x_base          # matched increment, same sign
            for mode, ga, ba in (('G', g_val, 0.0), ('boost', 0.0, b_val)):
                key = '%g' % gf
                if gf == 0.0 and mode == 'boost':
                    rec['boost'][key] = rec['G'][key]   # identical condition
                    continue
                rows = []
                for s in range(n_seeds):
                    ta, tb = run_trace(lam, beta, alpha, sigma, gam, kap,
                                       SIGNAL_NEUTRAL, SIGNAL_NEUTRAL,
                                       ga, 0.0, ba, 0.0,
                                       n_steps, w6.make_seed(1, ci, 0, li, s),
                                       X_MAX, -1.0, 0.0, 0.0)
                    rows.append(summarise(ta, tb))
                rec[mode][key] = agg(rows)
                li += 1
        out.append(rec)
        if (n + 1) % 20 == 0:
            print("    %3d/%d  (%.0fs)" % (n + 1, len(pool), time.time() - t0))
    _save('signed', out)
    return out


def _save(tag, obj):
    fn = "wave10_%s.json" % tag
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)
    print("    -> %s" % fn)


def _load(tag):
    fn = "wave10_%s.json" % tag
    if not os.path.exists(fn):
        sys.exit("ERROR: %s missing." % fn)
    return json.load(open(fn, encoding="utf-8"))


def pct(a, b):
    """percent change of a relative to b"""
    return 100.0 * (a - b) / b if (np.isfinite(a) and np.isfinite(b) and b) else np.nan


def analyse(D):
    hdr("1. SIGNATURE TABLE: %% change from baseline (medians across configs)")
    print("  %-8s %-6s | %10s %10s | %10s %10s"
          % ("mode", "G/lam", "own dur", "rival dur", "predom", "alt rate"))
    tab = {}
    for mode in ('G', 'boost'):
        for gf in G_FRACS:
            k = '%g' % gf
            dm, dr, pr, al, wt = [], [], [], [], []
            for r in D:
                b = r['G']['0']
                c = r[mode].get(k)
                if not c:
                    continue
                dm.append(pct(c['dur_manip'], b['dur_manip']))
                dr.append(pct(c['dur_rival'], b['dur_rival']))
                pr.append(pct(c['predominance'], b['predominance']))
                al.append(pct(c['alternation'], b['alternation']))
                wt.append(c['wta_rate'])
            f = lambda v: float(np.nanmedian(v)) if v else np.nan
            tab.setdefault(mode, {})[k] = {
                'own': f(dm), 'rival': f(dr), 'pred': f(pr), 'alt': f(al),
                'wta': float(np.mean(wt)) if wt else np.nan}
            print("  %-8s %+6.2f | %+9.1f%% %+9.1f%% | %+9.1f%% %+9.1f%%"
                  % (mode, gf, f(dm), f(dr), f(pr), f(al)))
    OUT['signature_table'] = tab

    hdr("2. RIVAL INVARIANCE -- the central claim")
    print("  Does the rival's duration stay flat under G but move under boost?")
    for mode in ('G', 'boost'):
        vals, gfs = [], []
        for gf in G_FRACS:
            if gf == 0:
                continue
            for r in D:
                b = r['G']['0']
                c = r[mode].get('%g' % gf)
                if c:
                    v = pct(c['dur_rival'], b['dur_rival'])
                    if np.isfinite(v):
                        vals.append(v)
                        gfs.append(gf)
        vals, gfs = np.array(vals), np.array(gfs)
        rho, p = stats.spearmanr(gfs, vals)
        print("    %-6s rho(G/lam, rival %% change) = %+.3f  p=%.2g   "
              "median |change| = %.1f%%"
              % (mode, rho, p, np.median(np.abs(vals))))
        OUT.setdefault('rival_invariance', {})[mode] = {
            'rho': float(rho), 'p': float(p),
            'median_abs_change': float(np.median(np.abs(vals)))}

    hdr("2b. RIVAL INVARIANCE IS CONDITIONAL ON FLOOR OCCUPANCY")
    print("  -G*x is zero only where the suppressed channel is annihilated.")
    print("  Where residual survives, weakening A releases B and the rival")
    print("  should LENGTHEN. Prediction: invariance only in high-floor configs.")
    fl = np.array([r['G']['0'].get('floor', np.nan) for r in D], float)
    med = np.nanmedian(fl)
    print("  median baseline floor occupancy = %.3f" % med)
    for lbl, sel in (('HIGH floor', fl >= med), ('LOW floor', fl < med)):
        for mode in ('G', 'boost'):
            v = []
            for gf in G_FRACS:
                if gf == 0:
                    continue
                for i, r in enumerate(D):
                    if not sel[i]:
                        continue
                    b, c = r['G']['0'], r[mode].get('%g' % gf)
                    if c:
                        x = pct(c['dur_rival'], b['dur_rival'])
                        if np.isfinite(x):
                            v.append(x)
            print("    %-11s %-6s median |rival change| = %6.1f%%  (n=%d)"
                  % (lbl, mode, np.median(np.abs(v)) if v else np.nan, len(v)))
            OUT.setdefault('rival_by_floor', {}).setdefault(lbl, {})[mode] = (
                float(np.median(np.abs(v))) if v else None)
    print("\n  Also reported separately for negative G only, since that is the")
    print("  condition Paffen's irrelevant-vs-neutral pairing corresponds to:")
    for lbl, sel in (('HIGH floor', fl >= med), ('LOW floor', fl < med)):
        own, riv = [], []
        for gf in (-0.90, -0.50, -0.25):
            for i, r in enumerate(D):
                if not sel[i]:
                    continue
                b, c = r['G']['0'], r['G'].get('%g' % gf)
                if c:
                    own.append(pct(c['dur_manip'], b['dur_manip']))
                    riv.append(pct(c['dur_rival'], b['dur_rival']))
        print("    %-11s  -G: own %+7.1f%%   rival %+7.1f%%"
              % (lbl, np.nanmedian(own), np.nanmedian(riv)))
        OUT.setdefault('negG_by_floor', {})[lbl] = {
            'own': float(np.nanmedian(own)), 'rival': float(np.nanmedian(riv))}

    hdr("3. SIGN SYMMETRY of the goal signal")
    for gf in (0.25, 0.50, 0.90):
        pos, neg = [], []
        for r in D:
            b = r['G']['0']
            cp, cn = r['G'].get('%g' % gf), r['G'].get('%g' % -gf)
            if cp and cn:
                pos.append(pct(cp['dur_manip'], b['dur_manip']))
                neg.append(pct(cn['dur_manip'], b['dur_manip']))
        print("    |G|=%.2f lam : +G own %+7.1f%%   -G own %+7.1f%%"
              % (gf, np.nanmedian(pos), np.nanmedian(neg)))

    hdr("4. ALTERNATION RATE DIRECTION")
    print("  Model predicts +G slows alternation, -G speeds it up.")
    print("  Moreno-Sanchez Exp 3 reports an increase, F(1,83)=43.59, p<.0001.")
    for gf in (-0.90, -0.50, 0.50, 0.90):
        v = []
        for r in D:
            b = r['G']['0']
            c = r['G'].get('%g' % gf)
            if c:
                v.append(pct(c['alternation'], b['alternation']))
        n_up = int(np.sum(np.array(v) > 0))
        lo, hi = w6.wilson(n_up, len(v))
        print("    G=%+.2f lam : alternation %+7.1f%%   increased in %d/%d "
              "[%.0f%%, %.0f%%]"
              % (gf, np.nanmedian(v), n_up, len(v), 100 * lo, 100 * hi))

    hdr("5. COMPARISON WITH PUBLISHED SIGNATURES")
    print("  %-42s %10s %10s" % ("", "own dur", "rival dur"))
    emp = [
        ("Chong 2005 Exp1 (attention, tracking)", +50.0, +5.0),
        ("Chong 2005 Exp3 (contrast, GATED on dominance)", +29.0, +9.0),
        ("Levelt / Mueller-Blake (contrast, UNGATED)", 0.0, -30.0),
        ("Paffen 2008 irrelevant vs NEUTRAL", -20.0, 0.0),
    ]
    for lbl, o, rv in emp:
        print("  %-42s %+9.1f%% %+9.1f%%" % (lbl, o, rv))
    print("  " + "-" * 64)
    for mode, gf, lbl in (('G', 0.50, 'MODEL  +G  (state-dependent, positive)'),
                          ('G', -0.50, 'MODEL  -G  (state-dependent, negative)'),
                          ('boost', 0.50, 'MODEL  +boost (state-independent)'),
                          ('boost', -0.50, 'MODEL  -boost (state-independent)')):
        t = OUT['signature_table'][mode]['%g' % gf]
        print("  %-42s %+9.1f%% %+9.1f%%" % (lbl, t['own'], t['rival']))
    print("\n  The discriminating row is Paffen's irrelevant-vs-neutral:")
    print("  own duration DOWN with rival UNCHANGED. Levelt Prop II predicts")
    print("  the opposite. Compare MODEL -G against MODEL -boost.")

    hdr("6. CEILING / WTA CHECK")
    for mode in ('G', 'boost'):
        w = [OUT['signature_table'][mode]['%g' % g]['wta'] for g in G_FRACS]
        print("    %-6s max WTA rate across levels: %.3f" % (mode, np.nanmax(w)))


def figure(D):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    t = OUT['signature_table']
    gs = np.array(G_FRACS)
    for mode, c, m in (('G', '#2c6fbb', 'o'), ('boost', '#d1495b', 's')):
        ax[0].plot(gs, [t[mode]['%g' % g]['own'] for g in gs], m + '-',
                   color=c, label=mode)
        ax[1].plot(gs, [t[mode]['%g' % g]['rival'] for g in gs], m + '-',
                   color=c, label=mode)
        ax[2].plot(gs, [t[mode]['%g' % g]['alt'] for g in gs], m + '-',
                   color=c, label=mode)
    for a, ttl, yl in ((ax[0], 'A  Manipulated channel', '% change in duration'),
                       (ax[1], 'B  Rival channel', '% change in duration'),
                       (ax[2], 'C  Alternation rate', '% change')):
        a.axhline(0, ls='--', c='k', lw=1)
        a.axvline(0, ls=':', c='k', lw=0.8)
        a.set_xlabel(r'$G/\lambda$  (or matched increment)')
        a.set_ylabel(yl)
        a.set_title(ttl)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig('wave10_fig_signed.pdf')
    print("\n  wave10_fig_signed.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analyse', action='store_true')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    n_cfg, n_seed, n_steps = (8, 8, 4000) if a.quick else (100, 50, 20000)

    cfgs = load_configs()
    if a.analyse:
        D = _load('signed')
    else:
        pool, _ = w6.eligible_pool(list(cfgs.values()), n_cfg)
        D = campaign(pool, n_seed, n_steps)
    analyse(D)
    figure(D)
    json.dump(OUT, open('wave10_results.json', 'w', encoding='utf-8'),
              indent=1, default=float)
    hdr("DONE")
    print("  wave10_results.json")


if __name__ == '__main__':
    main()
