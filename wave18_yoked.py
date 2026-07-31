"""
wave18_yoked.py -- is the gate-timing effect definitional?

THE OBJECTION
    Dominance is x_A - x_B > theta. The anti-gate delivers an increment to A
    during exactly the epochs when B is dominant, and B's episode ends when x_A
    rises relative to x_B. Adding drive to A during those epochs advances the
    crossing by construction. The DIRECTION of the anti-gate result is therefore
    guaranteed by the measurement rule, and the causal series may show only that
    the extraction rule works.

    Gate timing also changes the dominance structure it is measured against: the
    gated condition lengthens the attended channel's episodes by 79.7% and the
    anti-gated shortens them by 8.9%, so "activation while the competitor is
    dominant" is computed over systematically different epochs across conditions.

TEST 1  YOKED REPLAY
    Deliver the increment on a schedule taken from a DIFFERENT run's dominance
    time-course. Delivery has the same duty cycle and temporal statistics but is
    uncorrelated with the present trial's dominance. If gated and anti-gated
    schedules still separate under yoking, the effect does not require
    contingency with the actual state and cannot be definitional. If they
    collapse toward the ungated condition, contingency is required -- which is
    consistent with both a mechanism and an artefact, and settles only half.

TEST 2  ABSOLUTE-THRESHOLD CRITERION
    Recompute the outcome with the competitor's dominance defined by its OWN
    activation crossing a fixed threshold, x_B > c, rather than by the difference
    x_B - x_A > theta. The predictor and the outcome are then no longer defined
    on the same comparison. The increment still reaches B through inhibition, but
    that is a modelled mechanism rather than an artefact of the criterion.

TEST 3  SMALL-INCREMENT LIMIT
    A purely definitional push should scale with the increment. Sweep the
    increment toward zero and track the coupling RATIO. If the ratio is stable as
    the increment shrinks, the relationship is proportional and not a threshold
    artefact; if it grows without bound, the effect is dominated by the crossing.

Run:  python wave18_yoked.py   (~5 min)  |  --analyse  |  --quick
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
    extract_durations, _rectify,
    BURN_IN, MARGIN, X_MAX, SIGNAL_NEUTRAL, load_configs,
)
import wave6_robustness as w6

MAGS = [0.125, 0.25, 0.5, 1.0, 2.0]
OUT = {}


def hdr(s):
    print("\n" + "=" * 74)
    print(s)
    print("=" * 74)


def pct(a, b):
    return 100.0 * (a - b) / b if (np.isfinite(a) and np.isfinite(b) and b) else np.nan


# =============================================================================
# KERNELS
# =============================================================================

@njit(cache=True)
def run_sched(lam, beta, alpha, sigma, gamma_adapt, kappa, sig_a, sig_b,
              boost, mode, sched, n_steps, seed, x_max, margin):
    """
    mode 0 : ungated
         1 : live gate      -- increment on while A dominant NOW
         2 : live anti-gate -- increment on while A suppressed NOW
         3 : yoked gate     -- increment on where sched == 1
         4 : yoked anti     -- increment on where sched == 0 and sched valid
    sched is a 0/1 array recorded from a different run's dominance time-course.
    """
    np.random.seed(seed)
    ta = np.empty(n_steps, dtype=np.float64)
    tb = np.empty(n_steps, dtype=np.float64)
    x_a = 0.1
    x_b = 0.1
    a_a = 0.0
    a_b = 0.0
    dose = 0.0
    n_cnt = 0
    for t in range(n_steps):
        ta[t] = x_a
        tb[t] = x_b
        d = x_a - x_b
        if mode == 0:
            b_cur = boost
        elif mode == 1:
            b_cur = boost if d > margin else 0.0
        elif mode == 2:
            b_cur = boost if d < -margin else 0.0
        elif mode == 3:
            b_cur = boost if sched[t] == 1 else 0.0
        else:
            b_cur = boost if sched[t] == 0 else 0.0
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
    return ta, tb, (dose / n_cnt if n_cnt else 0.0)


@njit(cache=True)
def schedule_from(ta, tb, margin):
    """1 where A is dominant, 0 where A is suppressed, 2 where indeterminate."""
    n = len(ta)
    out = np.empty(n, dtype=np.int8)
    for t in range(n):
        d = ta[t] - tb[t]
        if d > margin:
            out[t] = 1
        elif d < -margin:
            out[t] = 0
        else:
            out[t] = 2
    return out


@njit(cache=True)
def abs_durations(tb, burn_in, thresh):
    """Episode durations of channel B defined by x_B > thresh, an ABSOLUTE
    criterion that does not reference channel A."""
    n = len(tb)
    tot = 0
    cnt = 0
    run = 0
    for t in range(burn_in, n):
        if tb[t] > thresh:
            run += 1
        else:
            if run > 0:
                tot += run
                cnt += 1
            run = 0
    if run > 0:
        tot += run
        cnt += 1
    return (tot / cnt if cnt else 0.0), cnt


def rel_summary(ta, tb):
    ch, dur = extract_durations(ta, tb, BURN_IN, MARGIN)
    if len(ch) > 2:
        ch, dur = ch[1:-1], dur[1:-1].astype(float)
    else:
        ch, dur = np.empty(0, np.int32), np.empty(0, float)
    da, db = dur[ch == 0], dur[ch == 1]
    return (float(da.mean()) if len(da) else np.nan,
            float(db.mean()) if len(db) else np.nan)


def _save(t, o):
    fn = "wave18_%s.json" % t
    json.dump(o, open(fn, 'w', encoding='utf-8'), indent=1, default=float)
    print("    -> %s" % fn)


# =============================================================================

def campaign(pool, n_seeds, n_steps):
    hdr("SIMULATION: live gates, yoked gates, absolute-threshold outcome")
    out = []
    t0 = time.time()
    for n, ent in enumerate(pool):
        p = ent['params']
        ci = ent['grid_index'] % 1000
        L = (p['lambda'], p['beta'], p['alpha'], p['sigma'], p['gamma'],
             p['kappa'])

        # baseline: activation, absolute threshold, and yoke schedules
        acts, thr_b = [], []
        scheds = []
        for s in range(max(20, n_seeds)):
            ta, tb, _ = run_sched(*L, SIGNAL_NEUTRAL, SIGNAL_NEUTRAL, 0.0, 0,
                                  np.zeros(1, dtype=np.int8), n_steps,
                                  w6.make_seed(0, ci, 0, 0, s), X_MAX, MARGIN)
            acts.append(float(np.mean(ta[BURN_IN:])))
            thr_b.append(float(np.median(tb[BURN_IN:])))
            if s < n_seeds:
                scheds.append(schedule_from(ta, tb, MARGIN))
        x_base = float(np.mean(acts))
        thresh = float(np.mean(thr_b))
        b_unit = 0.5 * p['lambda'] * x_base

        rec = {'grid_index': ent['grid_index'], 'params': p,
               'x_baseline': x_base, 'b_unit': b_unit, 'abs_thresh': thresh,
               'cond': {}}

        def measure(mode, mag, key):
            oa, rb, ab, ds = [], [], [], []
            for s in range(n_seeds):
                # yoked conditions take the schedule from a DIFFERENT seed
                sch = scheds[(s + 1) % len(scheds)] if mode >= 3 \
                    else np.zeros(1, dtype=np.int8)
                ta, tb, do = run_sched(*L, SIGNAL_NEUTRAL, SIGNAL_NEUTRAL,
                                       mag * b_unit, mode, sch, n_steps,
                                       w6.make_seed(*key, s), X_MAX, MARGIN)
                o, r = rel_summary(ta, tb)
                aB, _ = abs_durations(tb, BURN_IN, thresh)
                oa.append(o)
                rb.append(r)
                ab.append(aB)
                ds.append(do)

            def f(v):
                v = [x for x in v if np.isfinite(x)]
                return float(np.mean(v)) if v else np.nan
            return {'own': f(oa), 'rival': f(rb), 'rival_abs': f(ab),
                    'dose': float(np.mean(ds))}

        li = 1
        rec['baseline'] = measure(0, 0.0, (1, ci, 0, 0))
        li += 1
        for mode in (0, 1, 2, 3, 4):
            for mag in MAGS:
                rec['cond']['%d_%g' % (mode, mag)] = measure(
                    mode, mag, (2, ci, mode, li))
                li += 1
        out.append(rec)
        if (n + 1) % 20 == 0:
            print("    %3d/%d  (%.0fs)" % (n + 1, len(pool), time.time() - t0))
    _save('yoked', out)
    return out


NAMES = {0: 'ungated', 1: 'live gate', 2: 'live anti-gate',
         3: 'YOKED gate', 4: 'YOKED anti-gate'}


def med(D, mode, mag, key, bkey=None):
    v = []
    for rec in D:
        b = rec['baseline']
        c = rec['cond'].get('%d_%g' % (mode, mag))
        if not c:
            continue
        base = b[bkey or key]
        x = pct(c[key], base)
        if np.isfinite(x):
            v.append(x)
    return float(np.nanmedian(v)) if v else np.nan, v


def analyse(D):
    hdr("1. YOKED REPLAY")
    print("  Yoked conditions use a delivery schedule taken from a different")
    print("  run, so duty cycle and temporal statistics match but contingency")
    print("  with the present trial's dominance is broken.\n")
    print("  %-18s %8s %10s %10s %8s" % ("condition", "mag", "own", "competitor",
                                         "dose"))
    res = {}
    for mode in (1, 2, 3, 4, 0):
        for mag in (1.0, 2.0):
            o, _ = med(D, mode, mag, 'own')
            r, rv = med(D, mode, mag, 'rival')
            d = np.nanmedian([rec['cond']['%d_%g' % (mode, mag)]['dose']
                              / rec['b_unit'] for rec in D
                              if rec['b_unit'] and
                              rec['cond'].get('%d_%g' % (mode, mag))])
            print("  %-18s %8.1f %+9.1f%% %+9.1f%% %7.2f"
                  % (NAMES[mode], mag, o, r, d))
            res['%d_%g' % (mode, mag)] = {'name': NAMES[mode], 'mag': mag,
                                          'own': o, 'rival': r, 'dose': float(d),
                                          'n': len(rv)}
    OUT['yoked'] = res

    print("\n  live separation  (gate 2x minus anti-gate 2x):  %+.1f pp"
          % (res['1_2']['rival'] - res['2_2']['rival']))
    print("  yoked separation (gate 2x minus anti-gate 2x):  %+.1f pp"
          % (res['3_2']['rival'] - res['4_2']['rival']))
    live = res['1_2']['rival'] - res['2_2']['rival']
    yok = res['3_2']['rival'] - res['4_2']['rival']
    frac = yok / live if live else np.nan
    print("  yoked retains %.0f%% of the live separation" % (100 * frac))
    OUT['separation'] = {'live': float(live), 'yoked': float(yok),
                         'fraction': float(frac)}
    print("\n  If yoked separation is near zero, the effect requires contingency")
    print("  with the actual dominance state. That is consistent with both a")
    print("  mechanism and a definitional artefact, and settles only half.")

    hdr("2. ABSOLUTE-THRESHOLD OUTCOME")
    print("  competitor dominance redefined as x_B > c, a fixed threshold that")
    print("  does not reference the attended channel.\n")
    print("  %-18s %8s %14s %14s" % ("condition", "mag", "relative", "absolute"))
    for mode in (1, 2, 0):
        for mag in (1.0, 2.0):
            r, _ = med(D, mode, mag, 'rival')
            a, _ = med(D, mode, mag, 'rival_abs')
            print("  %-18s %8.1f %+13.1f%% %+13.1f%%" % (NAMES[mode], mag, r, a))
            OUT.setdefault('absolute', {})['%d_%g' % (mode, mag)] = {
                'relative': r, 'absolute': a}
    print("\n  If the sign contrast survives the absolute criterion, the")
    print("  relationship is not an artefact of defining dominance by the")
    print("  difference between the two channels.")

    hdr("3. SMALL-INCREMENT LIMIT")
    print("  coupling ratio = competitor %% change / attended %% change\n")
    print("  %-18s %s" % ("condition", "  ".join("%6.3fx" % m for m in MAGS)))
    for mode in (1, 2, 0):
        row = []
        for mag in MAGS:
            o, _ = med(D, mode, mag, 'own')
            r, _ = med(D, mode, mag, 'rival')
            row.append(r / o if (np.isfinite(o) and abs(o) > 0.5) else np.nan)
        print("  %-18s %s" % (NAMES[mode],
                              "  ".join("%7.2f" % v for v in row)))
        OUT.setdefault('small_limit', {})[NAMES[mode]] = [float(v) for v in row]
    print("\n  A stable ratio as the increment shrinks indicates a proportional")
    print("  relationship. A ratio that grows without bound indicates the effect")
    print("  is dominated by the threshold crossing itself.")


def figure(D):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    cols = {0: '#999999', 1: '#2c6fbb', 2: '#4c9f70', 3: '#7fa8d6',
            4: '#a3cfb5'}
    for mode in (1, 2, 3, 4, 0):
        xs, ys = [], []
        for mag in MAGS:
            o, _ = med(D, mode, mag, 'own')
            r, _ = med(D, mode, mag, 'rival')
            xs.append(o)
            ys.append(r)
        ax[0].plot(xs, ys, marker='o', ms=6, c=cols[mode], label=NAMES[mode],
                   ls='--' if mode >= 3 else '-')
    ax[0].axhline(0, c='k', lw=0.7)
    ax[0].set_xlabel('% change, attended channel', fontsize=8)
    ax[0].set_ylabel('% change, competitor', fontsize=8)
    ax[0].set_title('A  Live versus yoked delivery', fontsize=10)
    ax[0].legend(fontsize=6.5, frameon=False)

    lbl, rel, ab = [], [], []
    for mode in (1, 0, 2):
        for mag in (2.0,):
            lbl.append(NAMES[mode])
            r, _ = med(D, mode, mag, 'rival')
            a, _ = med(D, mode, mag, 'rival_abs')
            rel.append(r)
            ab.append(a)
    x = np.arange(len(lbl))
    ax[1].bar(x - 0.2, rel, 0.4, label='relative criterion', color='#2c6fbb')
    ax[1].bar(x + 0.2, ab, 0.4, label='absolute criterion', color='#d1495b')
    ax[1].axhline(0, c='k', lw=1)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(lbl, fontsize=7.5)
    ax[1].set_ylabel("% change, competitor's duration", fontsize=8)
    ax[1].set_title('B  Does the criterion carry the effect?', fontsize=10)
    ax[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig('wave18_fig_yoked.pdf')
    print("\n  wave18_fig_yoked.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analyse', action='store_true')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    n_cfg, n_seed, n_steps = (8, 8, 4000) if a.quick else (100, 40, 20000)
    cfgs = load_configs()
    if a.analyse:
        D = json.load(open('wave18_yoked.json', encoding='utf-8'))
    else:
        pool, _ = w6.eligible_pool(list(cfgs.values()), n_cfg)
        D = campaign(pool, n_seed, n_steps)
    analyse(D)
    figure(D)
    json.dump(OUT, open('wave18_results.json', 'w', encoding='utf-8'),
              indent=1, default=float)
    hdr("DONE")
    print("  wave18_results.json")


if __name__ == '__main__':
    main()
