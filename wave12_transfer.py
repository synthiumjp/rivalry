"""
wave12_transfer.py -- three targeted experiments.

BLOCK A  SATURATING INPUT TRANSFER FUNCTION
    Two of the model's failures share a cause. It occupies only the
    increasing-duration regime (Section 4.2) and it overshoots the magnitude of
    stimulus-strength effects (Section 4.7.3), both because input maps linearly
    onto activation. Platonov & Goossens (2013) reconciled contrast and coherence
    rivalry data using a saturating input transfer function, and Seely & Chow
    (2011) list input nonlinearity among the modifications that resolve the
    Proposition IV problem. Neither has been implemented here.

    We add a Naka-Rushton transfer, S_eff = k * S^n / (S^n + c^n), normalised so
    that S = 0.5 maps to 0.5 and the baseline operating point is unchanged. Then
    we re-run the two failing tests and check that the signature results survive.

BLOCK B  WHAT DRIVES THE SAME-SIGN COUPLING?
    Section 4.7 reports that a goal signal shifts BOTH channels' durations in the
    same direction and states that we cannot explain it. Adaptation recovery was
    the obvious candidate and the correlation with alpha*kappa/gamma did not
    support it. Two sharper tests here:
      (i)  one-parameter-at-a-time sweeps from a fixed base configuration, to
           see which parameter the coupling actually tracks;
      (ii) direct measurement of the manipulated channel's mean adaptation under
           G, correlated with the rival's response. If the mechanism is that G
           raises the manipulated channel's adaptation and therefore lengthens
           its own recovery from suppression, that correlation should be strong
           where alpha*kappa/gamma was not.

BLOCK C  HOW FAST MUST THE GATE BE?
    Section 4.7.2 found that a slow ramp inverts the signature. Sweep tau finely,
    locate the crossover, and express it relative to mean dominance duration so
    the prediction is stated in units an experiment can use.

Run:  python wave12_transfer.py   (~5 min)  |  --analyse  |  --quick
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

SAT_N, SAT_C = 2.0, 0.30          # Naka-Rushton exponent and half-saturation
EQUAL_SWEEP = [round(0.20 + 0.05 * i, 2) for i in range(11)]
B_LEVELS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
G_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]
TAUS = [1, 5, 10, 20, 40, 80, 160]
PARAMS = ['lambda', 'beta', 'alpha', 'sigma', 'gamma', 'kappa']
GRID = {'lambda': [0.08, 0.10, 0.15, 0.20],
        'beta': [0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
        'alpha': [0.03, 0.05, 0.08, 0.10, 0.12],
        'sigma': [0.04, 0.06, 0.08, 0.10, 0.12],
        'gamma': [0.02, 0.03, 0.05]}
OUT = {}


def hdr(s):
    print("\n" + "=" * 74)
    print(s)
    print("=" * 74)


# =============================================================================
# KERNEL
# =============================================================================

@njit(cache=True)
def _transfer(S, n, c):
    """Naka-Rushton, normalised so that S = 0.5 maps to 0.5."""
    if n <= 0.0:
        return S
    if S <= 0.0:
        return 0.0
    num = S ** n / (S ** n + c ** n)
    ref = 0.5 ** n / (0.5 ** n + c ** n)
    return 0.5 * num / ref


@njit(cache=True)
def run_v12(lam, beta, alpha, sigma, gamma_adapt, kappa,
            signal_a, signal_b, g_a, boost_a, gate, tau,
            sat_n, sat_c, n_steps, seed, x_max, margin):
    """
    Adds a saturating input transfer (sat_n <= 0 disables it) and returns the
    mean adaptation variable of each channel over the post-burn-in period.
    """
    np.random.seed(seed)
    ta = np.empty(n_steps, dtype=np.float64)
    tb = np.empty(n_steps, dtype=np.float64)
    x_a = 0.1
    x_b = 0.1
    a_a = 0.0
    a_b = 0.0
    b_cur = 0.0 if gate == 1 else boost_a
    sa = _transfer(signal_a, sat_n, sat_c)
    sb = _transfer(signal_b, sat_n, sat_c)
    ad_a = 0.0
    ad_b = 0.0
    n_cnt = 0

    for t in range(n_steps):
        ta[t] = x_a
        tb[t] = x_b
        if gate == 1:
            tgt = boost_a if (x_a - x_b) > margin else 0.0
            if tau <= 1:
                b_cur = tgt
            else:
                b_cur += (tgt - b_cur) / tau
        ea = np.random.normal(0.0, sigma)
        eb = np.random.normal(0.0, sigma)
        na = (1.0 - lam + g_a) * x_a + sa + b_cur - beta * x_b - alpha * a_a + ea
        nb = (1.0 - lam) * x_b + sb - beta * x_a - alpha * a_b + eb
        x_a = _rectify(na, -1.0, x_max)
        x_b = _rectify(nb, -1.0, x_max)
        a_a = (1.0 - gamma_adapt) * a_a + kappa * x_a
        a_b = (1.0 - gamma_adapt) * a_b + kappa * x_b
        if t >= BURN_IN:
            ad_a += a_a
            ad_b += a_b
            n_cnt += 1
    return ta, tb, ad_a / n_cnt, ad_b / n_cnt


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


def run_cond(p, g_a, b_a, gate, tau, sat, sig_a, sig_b, seed_key,
             n_seeds, n_steps):
    rows, aa, ab = [], [], []
    sn, sc = (SAT_N, SAT_C) if sat else (-1.0, 0.0)
    for s in range(n_seeds):
        ta, tb, m_a, m_b = run_v12(
            p['lambda'], p['beta'], p['alpha'], p['sigma'], p['gamma'],
            p['kappa'], sig_a, sig_b, g_a, b_a, gate, tau, sn, sc,
            n_steps, w6.make_seed(*seed_key, s), X_MAX, MARGIN)
        rows.append(summarise(ta, tb))
        aa.append(m_a)
        ab.append(m_b)

    def m(k):
        v = [r[k] for r in rows if np.isfinite(r[k])]
        return float(np.mean(v)) if v else np.nan
    return {'own': m('own'), 'rival': m('rival'), 'pred': m('pred'),
            'alt': m('alt'), 'adapt_own': float(np.mean(aa)),
            'adapt_rival': float(np.mean(ab))}


def _save(tag, obj):
    fn = "wave12_%s.json" % tag
    json.dump(obj, open(fn, 'w', encoding='utf-8'), indent=1, default=float)
    print("    -> %s" % fn)


def _load(tag):
    fn = "wave12_%s.json" % tag
    if not os.path.exists(fn):
        sys.exit("ERROR: %s missing." % fn)
    return json.load(open(fn, encoding='utf-8'))


def pct(a, b):
    return 100.0 * (a - b) / b if (np.isfinite(a) and np.isfinite(b) and b) else np.nan


# =============================================================================
# BLOCK A
# =============================================================================

def block_A(pool, n_seeds, n_steps):
    hdr("BLOCK A: saturating input transfer (Naka-Rushton, n=%.1f, c=%.2f)"
        % (SAT_N, SAT_C))
    out = []
    t0 = time.time()
    for n, ent in enumerate(pool):
        p = ent['params']
        ci = ent['grid_index'] % 1000
        rec = {'grid_index': ent['grid_index'], 'params': p,
               'linear': {}, 'saturating': {}}
        for si, sat in enumerate((False, True)):  # si used in seeds
            tag = 'saturating' if sat else 'linear'
            eq = []
            for li, sg in enumerate(EQUAL_SWEEP):
                eq.append(run_cond(p, 0, 0, 0, 1, sat, sg, sg,
                                   (0, ci, si, li), max(4, n_seeds // 6),
                                   min(n_steps, 12000)))
                eq[-1]['signal'] = sg
            rec[tag]['equal_sweep'] = eq
            rec[tag]['baseline'] = run_cond(
                p, 0, 0, 0, 1, sat, SIGNAL_NEUTRAL, SIGNAL_NEUTRAL,
                (1, ci, si, 0), n_seeds, n_steps)
            bl, gl = {}, {}
            # reference increment must scale with mean ACTIVATION, not with
            # mean dominance duration. Measure it directly.
            sn, sc = (SAT_N, SAT_C) if sat else (-1.0, 0.0)
            acts = []
            for s_ in range(20):
                t_a, _, _, _ = run_v12(
                    p['lambda'], p['beta'], p['alpha'], p['sigma'],
                    p['gamma'], p['kappa'], SIGNAL_NEUTRAL, SIGNAL_NEUTRAL,
                    0.0, 0.0, 0, 1, sn, sc, n_steps,
                    w6.make_seed(4, ci, si, 0, s_), X_MAX, MARGIN)
                acts.append(float(np.mean(t_a[BURN_IN:])))
            x_act = float(np.mean(acts))
            b_unit = 0.5 * p['lambda'] * x_act
            rec[tag]['x_activation'] = x_act
            rec[tag]['b_unit'] = b_unit
            for li, bf in enumerate(B_LEVELS):
                bl['%g' % bf] = run_cond(p, 0, bf * b_unit, 0, 1, sat,
                                         SIGNAL_NEUTRAL, SIGNAL_NEUTRAL,
                                         (2, ci, si, li), n_seeds, n_steps)
            for li, gf in enumerate(G_LEVELS):
                gl['%g' % gf] = run_cond(
                    p, min(gf * p['lambda'], G_SAFETY * p['lambda']), 0, 0, 1,
                    sat, SIGNAL_NEUTRAL, SIGNAL_NEUTRAL,
                    (3, ci, si, li), n_seeds, n_steps)
            rec[tag]['boost'] = bl
            rec[tag]['G'] = gl
        out.append(rec)
        if (n + 1) % 20 == 0:
            print("    %3d/%d  (%.0fs)" % (n + 1, len(pool), time.time() - t0))
    _save('A_transfer', out)
    return out


def analyse_A(D):
    hdr("A1. MODIFIED PROPOSITION IV under each input transfer")
    for tag in ('linear', 'saturating'):
        rhos = []
        for r in D:
            eq = r[tag]['equal_sweep']
            S = np.array([e['signal'] for e in eq], float)
            A = np.array([e['alt'] for e in eq], float)
            m = np.isfinite(A) & (A > 0)
            if m.sum() >= 5:
                rhos.append(stats.spearmanr(S[m], A[m])[0])
        rhos = np.array([x for x in rhos if np.isfinite(x)])
        dec = int(np.sum(rhos < -0.7))
        inc = int(np.sum(rhos > 0.7))
        lo, hi = w6.wilson(inc, len(rhos))
        print("  %-12s median rho %+.3f | decreasing %d/%d | INCREASING %d/%d "
              "[%.0f%%, %.0f%%]"
              % (tag, np.median(rhos), dec, len(rhos), inc, len(rhos),
                 100 * lo, 100 * hi))
        OUT.setdefault('propIV', {})[tag] = {
            'median_rho': float(np.median(rhos)), 'n': int(len(rhos)),
            'n_decreasing': dec, 'n_increasing': inc}
    print("\n  Proposition IV requires alternation rate to INCREASE with")
    print("  equal-strength input. A saturating transfer should move the model")
    print("  out of the increasing-duration regime.")

    hdr("A2. LEVELT ONE-PARAMETER TEST under each input transfer")
    print("  Fit the increment so the rival falls 30%; predict the manipulated")
    print("  channel. Observed value is approximately 0%.")
    for tag in ('linear', 'saturating'):
        preds, n_ok = [], 0
        for r in D:
            b = r[tag]['baseline']
            if not b['rival']:
                continue
            xs, ys, zs = [], [], []
            for bf in B_LEVELS:
                c = r[tag]['boost']['%g' % bf]
                ys.append(pct(c['rival'], b['rival']))
                zs.append(pct(c['own'], b['own']))
                xs.append(bf)
            ys, zs = np.array(ys, float), np.array(zs, float)
            m = np.isfinite(ys) & np.isfinite(zs)
            if m.sum() < 3:
                continue
            o = np.argsort(ys[m])
            yy, zz = ys[m][o], zs[m][o]
            if yy.min() <= -30.0 <= yy.max():
                preds.append(float(np.interp(-30.0, yy, zz)))
                n_ok += 1
        if preds:
            print("  %-12s predicted own change: median %+.1f%%  "
                  "IQR [%+.1f%%, %+.1f%%]   (n=%d)"
                  % (tag, np.median(preds), np.percentile(preds, 25),
                     np.percentile(preds, 75), n_ok))
            OUT.setdefault('levelt_fit', {})[tag] = {
                'median': float(np.median(preds)), 'n': n_ok}

    hdr("A3. DOES THE SIGNATURE SURVIVE THE NEW TRANSFER?")
    print("  %-12s %10s %10s %10s %10s" % ("transfer", "G own", "G rival",
                                           "boost own", "boost rival"))
    for tag in ('linear', 'saturating'):
        go, gr, bo, br = [], [], [], []
        for r in D:
            b = r[tag]['baseline']
            g = r[tag]['G'].get('0.5')
            x = r[tag]['boost'].get('1')
            if not (b['rival'] and g and x):
                continue
            go.append(pct(g['own'], b['own']))
            gr.append(pct(g['rival'], b['rival']))
            bo.append(pct(x['own'], b['own']))
            br.append(pct(x['rival'], b['rival']))
        f = lambda v: float(np.nanmedian(v)) if v else np.nan
        print("  %-12s %+9.1f%% %+9.1f%% %+9.1f%% %+9.1f%%"
              % (tag, f(go), f(gr), f(bo), f(br)))
        OUT.setdefault('signature', {})[tag] = {
            'G_own': f(go), 'G_rival': f(gr),
            'boost_own': f(bo), 'boost_rival': f(br)}
    print("\n  The same-sign / opposite-sign contrast must survive, or the")
    print("  central result is an artifact of the linear input mapping.")


# =============================================================================
# BLOCK B
# =============================================================================

def block_B(pool, n_seeds, n_steps):
    hdr("BLOCK B: what drives the same-sign coupling?")
    # (i) one-parameter-at-a-time from a fixed base
    base = dict(pool[len(pool) // 2]['params'])
    base['kappa'] = base['alpha']
    print("  base configuration: " + ", ".join(
        "%s=%.3f" % (k, base[k]) for k in PARAMS))
    iso = []
    li = 0
    for k in PARAMS:
        vals = GRID.get(k, [0.5 * base['alpha'], 0.75 * base['alpha'],
                            base['alpha'], 1.25 * base['alpha'],
                            1.5 * base['alpha']])
        for v in vals:
            p = dict(base)
            p[k] = v
            if k == 'alpha':
                p['kappa'] = v
            b = run_cond(p, 0, 0, 0, 1, False, SIGNAL_NEUTRAL, SIGNAL_NEUTRAL,
                         (4, 0, 0, li), n_seeds, n_steps)
            li += 1
            g = run_cond(p, min(0.5 * p['lambda'], G_SAFETY * p['lambda']),
                         0, 0, 1, False, SIGNAL_NEUTRAL, SIGNAL_NEUTRAL,
                         (5, 0, 0, li), n_seeds, n_steps)
            li += 1
            iso.append({'param': k, 'value': v,
                        'own': pct(g['own'], b['own']),
                        'rival': pct(g['rival'], b['rival']),
                        'adapt_own_change': pct(g['adapt_own'], b['adapt_own']),
                        'adapt_rival_change': pct(g['adapt_rival'],
                                                  b['adapt_rival'])})
    # (ii) adaptation change vs coupling across the pool
    across = []
    for n, ent in enumerate(pool):
        p = ent['params']
        ci = ent['grid_index'] % 1000
        b = run_cond(p, 0, 0, 0, 1, False, SIGNAL_NEUTRAL, SIGNAL_NEUTRAL,
                     (6, ci, 0, 0), n_seeds, n_steps)
        g = run_cond(p, min(0.5 * p['lambda'], G_SAFETY * p['lambda']),
                     0, 0, 1, False, SIGNAL_NEUTRAL, SIGNAL_NEUTRAL,
                     (7, ci, 0, 0), n_seeds, n_steps)
        across.append({'grid_index': ent['grid_index'], 'params': p,
                       'own': pct(g['own'], b['own']),
                       'rival': pct(g['rival'], b['rival']),
                       'adapt_own_change': pct(g['adapt_own'], b['adapt_own']),
                       'adapt_rival_change': pct(g['adapt_rival'],
                                                 b['adapt_rival'])})
    _save('B_coupling', {'isolated': iso, 'across': across, 'base': base})
    return {'isolated': iso, 'across': across, 'base': base}


def analyse_B(B):
    hdr("B1. ONE PARAMETER AT A TIME: what does the coupling track?")
    print("  %-8s %8s | %9s %9s %12s" % ("param", "value", "own %", "rival %",
                                         "own adapt %"))
    for k in PARAMS:
        rows = [r for r in B['isolated'] if r['param'] == k]
        for r in rows:
            print("  %-8s %8.3f | %+8.1f%% %+8.1f%% %+11.1f%%"
                  % (k, r['value'], r['own'], r['rival'],
                     r['adapt_own_change']))
        v = np.array([r['value'] for r in rows], float)
        c = np.array([r['rival'] for r in rows], float)
        m = np.isfinite(c)
        if m.sum() >= 3:
            rho, p_ = stats.spearmanr(v[m], c[m])
            print("    -> rho(%s, rival coupling) = %+.3f  p=%.2g\n" % (k, rho, p_))
            OUT.setdefault('isolated', {})[k] = {'rho': float(rho),
                                                 'p': float(p_)}

    hdr("B2. IS THE COUPLING CARRIED BY THE MANIPULATED CHANNEL'S ADAPTATION?")
    A = B['across']
    ad = np.array([r['adapt_own_change'] for r in A], float)
    rv = np.array([r['rival'] for r in A], float)
    ow = np.array([r['own'] for r in A], float)
    m = np.isfinite(ad) & np.isfinite(rv)
    rho, p_ = stats.spearmanr(ad[m], rv[m])
    print("  rho(change in manipulated channel's adaptation, rival response)")
    print("      = %+.3f  p=%.2g  n=%d" % (rho, p_, m.sum()))
    r2, p2 = stats.spearmanr(ow[m], rv[m])
    print("  rho(own duration change, rival response) = %+.3f  p=%.2g"
          % (r2, p2))
    # partial: does adaptation explain coupling beyond own-duration change?
    def resid(y, x):
        return y - np.polyval(np.polyfit(x, y, 1), x)
    ra, rr, ro = (stats.rankdata(ad[m]), stats.rankdata(rv[m]),
                  stats.rankdata(ow[m]))
    pr = np.corrcoef(resid(ra, ro), resid(rr, ro))[0, 1]
    print("  partial rho(adaptation, rival | own duration change) = %+.3f" % pr)
    print("\n  A strong partial correlation supports the adaptation account.")
    print("  A near-zero one means the coupling is carried by the duration")
    print("  change itself and adaptation is incidental.")
    OUT['coupling'] = {'rho_adapt': float(rho), 'rho_own': float(r2),
                       'partial': float(pr), 'n': int(m.sum())}


# =============================================================================
# BLOCK C
# =============================================================================

def block_C(pool, n_seeds, n_steps):
    hdr("BLOCK C: how fast must the gate be?")
    out = []
    for n, ent in enumerate(pool[:20]):
        p = ent['params']
        ci = ent['grid_index'] % 1000
        b = run_cond(p, 0, 0, 0, 1, False, SIGNAL_NEUTRAL, SIGNAL_NEUTRAL,
                     (8, ci, 0, 0), n_seeds, n_steps)
        acts = []
        for s_ in range(20):
            t_a, _, _, _ = run_v12(
                p['lambda'], p['beta'], p['alpha'], p['sigma'], p['gamma'],
                p['kappa'], SIGNAL_NEUTRAL, SIGNAL_NEUTRAL, 0.0, 0.0, 0, 1,
                -1.0, 0.0, n_steps, w6.make_seed(10, ci, 0, 0, s_), X_MAX, MARGIN)
            acts.append(float(np.mean(t_a[BURN_IN:])))
        b_unit = 0.5 * p['lambda'] * float(np.mean(acts))
        rec = {'grid_index': ent['grid_index'], 'params': p,
               'baseline_dur': b['own'], 'taus': {}}
        for li, tau in enumerate(TAUS):
            c = run_cond(p, 0, 1.0 * b_unit, 1, tau, False,
                         SIGNAL_NEUTRAL, SIGNAL_NEUTRAL,
                         (9, ci, 0, li), n_seeds, n_steps)
            rec['taus'][str(tau)] = {
                'own': pct(c['own'], b['own']),
                'rival': pct(c['rival'], b['rival']),
                'alt': pct(c['alt'], b['alt']),
                'tau_over_dur': tau / b['own'] if b['own'] else np.nan}
        out.append(rec)
    _save('C_ramp', out)
    return out


def analyse_C(C):
    hdr("C1. RAMP SPEED AND THE SIGN OF THE RIVAL RESPONSE")
    print("  %6s %10s %10s %10s %14s" % ("tau", "own %", "rival %", "alt %",
                                         "tau/duration"))
    cross = []
    for tau in TAUS:
        o = [r['taus'][str(tau)]['own'] for r in C]
        rv = [r['taus'][str(tau)]['rival'] for r in C]
        al = [r['taus'][str(tau)]['alt'] for r in C]
        td = [r['taus'][str(tau)]['tau_over_dur'] for r in C]
        f = lambda v: float(np.nanmedian(v))
        n_same = int(np.sum(np.array(rv) > 0))
        print("  %6d %+9.1f%% %+9.1f%% %+9.1f%% %13.3f   same-sign %d/%d"
              % (tau, f(o), f(rv), f(al), f(td), n_same, len(rv)))
        cross.append((f(td), f(rv)))
    xs = np.array([c[0] for c in cross])
    ys = np.array([c[1] for c in cross])
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() >= 3 and ys[m].min() < 0 < ys[m].max():
        o = np.argsort(ys[m])
        x0 = float(np.interp(0.0, ys[m][o], xs[m][o]))
        print("\n  Sign crossover at tau / mean dominance duration ~ %.3f" % x0)
        print("  The gate must complete within roughly %.0f%% of a dominance"
              % (100 * x0))
        print("  episode for the state-dependent signature to appear.")
        OUT['ramp_crossover'] = float(x0)
    else:
        print("\n  No sign crossover within the tested range.")


def figure(D, C):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for tag, c in (('linear', '#2c6fbb'), ('saturating', '#d1495b')):
        sig = OUT.get('signature', {}).get(tag)
        if not sig:
            continue
        x = np.arange(2) + (0.2 if tag == 'saturating' else -0.2)
        ax[0].bar(x, [sig['G_rival'], sig['boost_rival']], 0.4,
                  color=c, label=tag)
    ax[0].axhline(0, c='k', lw=1)
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(['goal signal', 'increment'], fontsize=8)
    ax[0].set_ylabel('% change, rival channel')
    ax[0].set_title('A  Signature under each transfer')
    ax[0].legend(fontsize=7)

    A = json.load(open('wave12_B_coupling.json', encoding='utf-8'))['across'] \
        if os.path.exists('wave12_B_coupling.json') else []
    if A:
        ax[1].scatter([r['own'] for r in A], [r['rival'] for r in A],
                      s=20, alpha=0.6, c='#2c6fbb')
        ax[1].axhline(0, ls='--', c='k', lw=1)
        ax[1].set_xlabel('% change, manipulated')
        ax[1].set_ylabel('% change, rival')
        ax[1].set_title('B  Coupling across configurations')

    any_pos = False
    for tau in TAUS:
        v = np.array([r['taus'][str(tau)]['rival'] for r in C], float)
        t = np.array([r['taus'][str(tau)]['tau_over_dur'] for r in C], float)
        m = np.isfinite(t) & np.isfinite(v) & (t > 0)
        if m.any():
            any_pos = True
            ax[2].scatter(t[m], v[m], s=14, alpha=0.5, c='#2c6fbb')
    ax[2].axhline(0, ls='--', c='k', lw=1)
    if any_pos:
        ax[2].set_xscale('log')
    ax[2].set_xlabel(r'$\tau$ / mean dominance duration')
    ax[2].set_ylabel('% change, rival')
    ax[2].set_title('C  Gate speed inverts the signature')
    fig.tight_layout()
    fig.savefig('wave12_fig.pdf')
    print("\n  wave12_fig.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analyse', action='store_true')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    n_cfg, n_seed, n_steps = (8, 8, 4000) if a.quick else (100, 40, 20000)

    cfgs = load_configs()
    pool, _ = w6.eligible_pool(list(cfgs.values()), n_cfg)
    if a.analyse:
        D, B, C = _load('A_transfer'), _load('B_coupling'), _load('C_ramp')
    else:
        t0 = time.time()
        D = block_A(pool, n_seed, n_steps)
        B = block_B(pool, n_seed, n_steps)
        C = block_C(pool, n_seed, n_steps)
        print("\n  simulation total: %.0f s" % (time.time() - t0))
    analyse_A(D)
    analyse_B(B)
    analyse_C(C)
    figure(D, C)
    json.dump(OUT, open('wave12_results.json', 'w', encoding='utf-8'),
              indent=1, default=float)
    hdr("DONE")
    print("  wave12_results.json")


if __name__ == '__main__':
    main()
