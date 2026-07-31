"""
wave11_gated.py -- three tests that anchor the model to published data.

TEST 1  CHONG EXPERIMENT 3, IN SILICO
    Chong, Tadin & Blake (2005) doubled the contrast of the attended grating
    over 520 ms *whenever observers reported it dominant*, and returned it to
    baseline over 520 ms as soon as dominance switched away. That is a physical,
    binary implementation of state-gating. They obtained the attentional
    signature (attended +29%, unattended +9%) from what is otherwise a pure
    contrast manipulation, and concluded that attention's boost applies only
    while the stimulus is dominant whereas an ungated contrast increase can also
    affect suppression durations.

    Here we implement exactly that: an input increment applied only while the
    target channel is dominant, with a first-order lag standing in for their
    520 ms ramp. If gated boost reproduces the G*x signature while ungated boost
    reproduces the Levelt signature, then their manipulation and the GC-LCA
    mechanism are the same thing, and their result is a direct test rather than
    a convergence.

TEST 2  IS THE MECHANISM ADAPTATION RECOVERY?
    Section 4.7 attributes same-sign coupling to the rival de-adapting further
    during a lengthened dominance episode. That is asserted, not tested. It
    predicts coupling strength should scale with adaptation strength,
    alpha*kappa/gamma. Tested by reanalysis of wave10_signed.json -- no new
    simulation.

TEST 3  ONE-PARAMETER FITS
    Fit G per configuration so the manipulated channel changes by +50%, matching
    Chong Exp 1. Then read off the predicted rival change with no free
    parameters left and compare against their observed +5%. Separately, fit the
    ungated increment so the rival changes by -30%, matching Levelt / Mueller &
    Blake, and compare the predicted manipulated-channel change against ~0%.

Run:  python wave11_gated.py   (~4 min)  |  --analyse  |  --quick
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

G_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9]
B_LEVELS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
GATED = [(1.0, 1), (1.0, 50), (2.0, 1), (2.0, 50)]   # (magnitude x, tau)

# published targets
CHONG_OWN, CHONG_RIVAL = 50.0, 5.0        # Exp 1, attention
CHONG3_OWN, CHONG3_RIVAL = 29.0, 9.0      # Exp 3, GATED contrast
LEVELT_OWN, LEVELT_RIVAL = 0.0, -30.0     # ungated contrast

OUT = {}


def hdr(s):
    print("\n" + "=" * 74)
    print(s)
    print("=" * 74)


# =============================================================================
# KERNEL
# =============================================================================

@njit(cache=True)
def run_gated(lam, beta, alpha, sigma, gamma_adapt, kappa,
              signal_a, signal_b, g_a, boost_a, gate, tau,
              n_steps, seed, x_max, margin):
    """
    gate = 0 : boost_a applied throughout (ungated increment)
    gate = 1 : boost_a applied only while channel A is dominant, approached
               with a first-order lag of time constant tau. tau = 1 is
               instantaneous; larger tau approximates Chong's 520 ms ramp.

    Also returns the time-averaged increment actually delivered, so the gated
    and ungated conditions can be compared on effective dose as well as on
    instantaneous magnitude.
    """
    np.random.seed(seed)
    ta = np.empty(n_steps, dtype=np.float64)
    tb = np.empty(n_steps, dtype=np.float64)
    x_a = 0.1
    x_b = 0.1
    a_a = 0.0
    a_b = 0.0
    b_cur = 0.0 if gate == 1 else boost_a
    b_sum = 0.0
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
        if t >= BURN_IN:
            b_sum += b_cur
            n_cnt += 1

        ea = np.random.normal(0.0, sigma)
        eb = np.random.normal(0.0, sigma)
        na = (1.0 - lam + g_a) * x_a + signal_a + b_cur - beta * x_b \
            - alpha * a_a + ea
        nb = (1.0 - lam) * x_b + signal_b - beta * x_a - alpha * a_b + eb
        x_a = _rectify(na, -1.0, x_max)
        x_b = _rectify(nb, -1.0, x_max)
        a_a = (1.0 - gamma_adapt) * a_a + kappa * x_a
        a_b = (1.0 - gamma_adapt) * a_b + kappa * x_b

    return ta, tb, (b_sum / n_cnt if n_cnt else 0.0)


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
            'alt': nsw / (len(ta) - BURN_IN),
            'wta': bool(len(da) == 0 or len(db) == 0)}


def agg(rows, doses):
    def m(k):
        v = [r[k] for r in rows if np.isfinite(r[k])]
        return float(np.mean(v)) if v else np.nan
    return {'own': m('own'), 'rival': m('rival'), 'pred': m('pred'),
            'alt': m('alt'), 'wta_rate': float(np.mean([r['wta'] for r in rows])),
            'mean_dose': float(np.mean(doses))}


# =============================================================================
# CAMPAIGN
# =============================================================================

def campaign(pool, n_seeds, n_steps):
    hdr("SIMULATION: gated vs ungated increment, fine G and boost sweeps")
    out = []
    t0 = time.time()
    for n, ent in enumerate(pool):
        p = ent['params']
        lam, beta, alpha = p['lambda'], p['beta'], p['alpha']
        sigma, gam, kap = p['sigma'], p['gamma'], p['kappa']
        ci = ent['grid_index'] % 1000

        acts = []
        for s in range(20):
            t_a, _ = run_trace(lam, beta, alpha, sigma, gam, kap,
                               SIGNAL_NEUTRAL, SIGNAL_NEUTRAL, 0, 0, 0, 0,
                               n_steps, w6.make_seed(0, ci, 0, 0, s),
                               X_MAX, -1.0, 0.0, 0.0)
            acts.append(float(np.mean(t_a[BURN_IN:])))
        x_base = float(np.mean(acts))
        b_unit = 0.5 * lam * x_base           # reference increment magnitude

        rec = {'grid_index': ent['grid_index'], 'params': p,
               'x_baseline': x_base, 'b_unit': b_unit,
               'adapt_strength': alpha * kap / gam,
               'baseline': None, 'G': {}, 'boost': {}, 'gated': {}}
        li = 0

        def run_cond(g_a, b_a, gate, tau, li):
            rows, doses = [], []
            for s in range(n_seeds):
                ta, tb, dose = run_gated(lam, beta, alpha, sigma, gam, kap,
                                         SIGNAL_NEUTRAL, SIGNAL_NEUTRAL,
                                         g_a, b_a, gate, tau, n_steps,
                                         w6.make_seed(1, ci, 0, li, s),
                                         X_MAX, MARGIN)
                rows.append(summarise(ta, tb))
                doses.append(dose)
            return agg(rows, doses)

        rec['baseline'] = run_cond(0.0, 0.0, 0, 1, li); li += 1
        for gf in G_LEVELS:
            rec['G']['%g' % gf] = run_cond(min(gf * lam, G_SAFETY * lam),
                                           0.0, 0, 1, li); li += 1
        for bf in B_LEVELS:
            rec['boost']['%g' % bf] = run_cond(0.0, bf * b_unit, 0, 1, li)
            li += 1
        for mag, tau in GATED:
            rec['gated']['%g_%d' % (mag, tau)] = run_cond(
                0.0, mag * b_unit, 1, tau, li); li += 1

        out.append(rec)
        if (n + 1) % 20 == 0:
            print("    %3d/%d  (%.0fs)" % (n + 1, len(pool), time.time() - t0))
    _save('gated', out)
    return out


def _save(tag, obj):
    fn = "wave11_%s.json" % tag
    json.dump(obj, open(fn, 'w', encoding='utf-8'), indent=1, default=float)
    print("    -> %s" % fn)


def _load(tag):
    fn = "wave11_%s.json" % tag
    if not os.path.exists(fn):
        sys.exit("ERROR: %s missing." % fn)
    return json.load(open(fn, encoding='utf-8'))


def pct(a, b):
    return 100.0 * (a - b) / b if (np.isfinite(a) and np.isfinite(b) and b) else np.nan


# =============================================================================
# TEST 1
# =============================================================================

def test1(D):
    hdr("TEST 1: CHONG EXPERIMENT 3 IN SILICO -- gated vs ungated increment")
    print("  Chong Exp 3 gated a contrast increment on dominance and obtained")
    print("  the ATTENTIONAL signature from a pure contrast manipulation.")
    print("  If gating flips the signature in the model, their manipulation and")
    print("  the GC-LCA mechanism are the same thing.\n")
    print("  %-30s %9s %9s %9s %9s %8s"
          % ("condition", "own", "rival", "predom", "alt rate", "dose"))

    def row(lbl, getter):
        o, r, pr, al, ds = [], [], [], [], []
        for rec in D:
            b = rec['baseline']
            c = getter(rec)
            if not c or not b['rival']:
                continue
            o.append(pct(c['own'], b['own']))
            r.append(pct(c['rival'], b['rival']))
            pr.append(pct(c['pred'], b['pred']))
            al.append(pct(c['alt'], b['alt']))
            ds.append(c['mean_dose'] / rec['b_unit'] if rec['b_unit'] else np.nan)
        f = lambda v: float(np.nanmedian(v)) if v else np.nan
        print("  %-30s %+8.1f%% %+8.1f%% %+8.1f%% %+8.1f%% %7.2f"
              % (lbl, f(o), f(r), f(pr), f(al), f(ds)))
        return {'own': f(o), 'rival': f(r), 'pred': f(pr), 'alt': f(al),
                'dose': f(ds)}

    res = {}
    res['G_50'] = row("G*x  (state-dependent, 50%L)",
                      lambda r: r['G'].get('0.5'))
    res['ungated_1'] = row("increment, UNGATED (1x)",
                           lambda r: r['boost'].get('1'))
    res['ungated_2'] = row("increment, UNGATED (2x)",
                           lambda r: r['boost'].get('2'))
    for mag, tau in GATED:
        res['gated_%g_%d' % (mag, tau)] = row(
            "increment, GATED (%gx, tau=%d)" % (mag, tau),
            lambda r, k='%g_%d' % (mag, tau): r['gated'].get(k))

    print("\n  PUBLISHED, for comparison")
    print("  %-30s %+8.1f%% %+8.1f%%" % ("Chong Exp1 (attention)",
                                          CHONG_OWN, CHONG_RIVAL))
    print("  %-30s %+8.1f%% %+8.1f%%" % ("Chong Exp3 (GATED contrast)",
                                          CHONG3_OWN, CHONG3_RIVAL))
    print("  %-30s %+8.1f%% %+8.1f%%" % ("Levelt / Mueller-Blake (ungated)",
                                          LEVELT_OWN, LEVELT_RIVAL))

    print("\n  KEY: sign of the rival's change. Same sign as own = state-")
    print("  dependent signature. Opposite = stimulus-strength signature.")
    n_flip = 0
    for mag, tau in GATED:
        g = res['gated_%g_%d' % (mag, tau)]
        u = res['ungated_%g' % mag] if 'ungated_%g' % mag in res else res['ungated_1']
        same = (np.sign(g['own']) == np.sign(g['rival']))
        print("    gated %gx tau=%-3d : rival %+.1f%%  -> %s"
              % (mag, tau, g['rival'],
                 "SAME sign (state-dependent)" if same
                 else "OPPOSITE sign (strength-like)"))
        n_flip += int(same)
    print("    gated conditions showing the state-dependent signature: %d/%d"
          % (n_flip, len(GATED)))
    OUT['test1'] = res


# =============================================================================
# TEST 2  (reanalysis, no new simulation)
# =============================================================================

def test2():
    hdr("TEST 2: IS SAME-SIGN COUPLING DUE TO ADAPTATION RECOVERY?")
    fn = 'wave10_signed.json'
    if not os.path.exists(fn):
        print("  wave10_signed.json not found; skipping")
        return
    D = json.load(open(fn, encoding='utf-8'))
    print("  Section 4.7 attributes same-sign coupling to the rival de-adapting")
    print("  further during a lengthened dominance episode. If so, coupling")
    print("  strength should scale with adaptation strength alpha*kappa/gamma.\n")
    for mode, lbl in (('G', 'state-dependent'), ('boost', 'input increment')):
        ad, cp = [], []
        for r in D:
            p = r['params']
            a = p['alpha'] * p['kappa'] / p['gamma']
            b = r['G'].get('0')
            c = r[mode].get('0.5')
            if not (b and c):
                continue
            v = pct(c['dur_rival'], b['dur_rival'])
            if np.isfinite(v) and np.isfinite(a):
                ad.append(a)
                cp.append(abs(v))
        if len(ad) < 10:
            print("    %-18s only %d usable configurations; skipped"
                  % (lbl, len(ad)))
            continue
        rho, p_ = stats.spearmanr(ad, cp)
        r2, p2 = stats.spearmanr(np.log(ad), cp)
        print("    %-18s rho(adapt strength, |rival change|) = %+.3f  p=%.2g  n=%d"
              % (lbl, rho, p_, len(ad)))
        OUT.setdefault('test2', {})[mode] = {
            'rho': float(rho), 'p': float(p_), 'n': int(len(ad))}
    print("\n  A positive correlation for the state-dependent mode, and a weaker")
    print("  or absent one for the increment, supports the stated mechanism.")
    print("  A flat result means the finding stands but the explanation does not.")


# =============================================================================
# TEST 3
# =============================================================================

def interp_level(levels, xs, target, key_out):
    """Given (level, own%, rival%) triples, interpolate to target own%."""
    xs = np.array(xs, float)
    ys = np.array([l[0] for l in levels], float)     # criterion series
    zs = np.array([l[1] for l in levels], float)     # predicted series
    m = np.isfinite(ys) & np.isfinite(zs)
    if m.sum() < 3:
        return np.nan, np.nan, 'insufficient'
    o = np.argsort(ys[m])
    yy, zz, xx = ys[m][o], zs[m][o], xs[m][o]
    if target < yy.min():
        return np.nan, np.nan, 'below_range'
    if target > yy.max():
        return np.nan, np.nan, 'above_range'
    return float(np.interp(target, yy, xx)), float(np.interp(target, yy, zz)), 'ok'


def test3(D):
    hdr("TEST 3: ONE-PARAMETER FITS TO PUBLISHED VALUES")
    print("  Fit the free parameter to ONE published number, then predict the")
    print("  other with no degrees of freedom left.\n")

    # --- fit G to Chong own +50%, predict rival ---
    fits, preds, status = [], [], []
    for rec in D:
        b = rec['baseline']
        if not b['rival']:
            continue
        lv, xs = [], []
        for gf in G_LEVELS:
            c = rec['G'].get('%g' % gf)
            if c:
                lv.append((pct(c['own'], b['own']), pct(c['rival'], b['rival'])))
                xs.append(gf)
        f, pr, st = interp_level(lv, xs, CHONG_OWN, 'rival')
        status.append(st)
        if st == 'ok':
            fits.append(f)
            preds.append(pr)
    from collections import Counter
    print("  FIT: G such that own duration changes by %+.0f%% (Chong Exp 1)"
          % CHONG_OWN)
    for k, v in Counter(status).items():
        print("    %-16s %d/%d" % (k, v, len(status)))
    if fits:
        print("    fitted G/lambda : median %.2f  IQR [%.2f, %.2f]"
              % (np.median(fits), np.percentile(fits, 25),
                 np.percentile(fits, 75)))
        print("    PREDICTED rival change : median %+.1f%%  IQR [%+.1f%%, %+.1f%%]"
              % (np.median(preds), np.percentile(preds, 25),
                 np.percentile(preds, 75)))
        print("    OBSERVED  (Chong Exp 1) : %+.1f%%  (ns, n=4)" % CHONG_RIVAL)
        lo, hi = np.percentile(preds, [25, 75])
        print("    -> observed value %s the predicted IQR"
              % ("lies WITHIN" if lo <= CHONG_RIVAL <= hi else "lies OUTSIDE"))
        OUT.setdefault('test3', {})['chong'] = {
            'n_fitted': len(fits), 'median_G': float(np.median(fits)),
            'pred_rival_median': float(np.median(preds)),
            'pred_rival_iqr': [float(lo), float(hi)],
            'observed_rival': CHONG_RIVAL}

    # --- fit ungated increment to Levelt rival -30%, predict own ---
    fits, preds, status = [], [], []
    for rec in D:
        b = rec['baseline']
        if not b['rival']:
            continue
        lv, xs = [], []
        for bf in B_LEVELS:
            c = rec['boost'].get('%g' % bf)
            if c:
                lv.append((pct(c['rival'], b['rival']), pct(c['own'], b['own'])))
                xs.append(bf)
        f, pr, st = interp_level(lv, xs, LEVELT_RIVAL, 'own')
        status.append(st)
        if st == 'ok':
            fits.append(f)
            preds.append(pr)
    print("\n  FIT: increment such that RIVAL duration changes by %+.0f%%"
          % LEVELT_RIVAL)
    for k, v in Counter(status).items():
        print("    %-16s %d/%d" % (k, v, len(status)))
    if fits:
        print("    fitted increment : median %.2f x reference"
              % np.median(fits))
        print("    PREDICTED own change : median %+.1f%%  IQR [%+.1f%%, %+.1f%%]"
              % (np.median(preds), np.percentile(preds, 25),
                 np.percentile(preds, 75)))
        print("    OBSERVED (Levelt / Mueller-Blake) : approximately %+.0f%%"
              % LEVELT_OWN)
        OUT.setdefault('test3', {})['levelt'] = {
            'n_fitted': len(fits),
            'pred_own_median': float(np.median(preds)),
            'observed_own': LEVELT_OWN}


def figure(D):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    t = OUT.get('test1', {})
    labels, owns, rivals = [], [], []
    for k, lbl in (('G_50', 'G·x'), ('ungated_1', 'ungated 1x'),
                   ('ungated_2', 'ungated 2x'),
                   ('gated_1_1', 'gated 1x'), ('gated_2_1', 'gated 2x'),
                   ('gated_2_50', 'gated 2x lag')):
        if k in t:
            labels.append(lbl)
            owns.append(t[k]['own'])
            rivals.append(t[k]['rival'])
    x = np.arange(len(labels))
    ax[0].bar(x - 0.2, owns, 0.4, label='manipulated', color='#2c6fbb')
    ax[0].bar(x + 0.2, rivals, 0.4, label='rival', color='#d1495b')
    ax[0].axhline(0, c='k', lw=1)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
    ax[0].set_ylabel('% change in duration')
    ax[0].set_title('A  Gating flips the rival sign')
    ax[0].legend(fontsize=7)

    ax[1].scatter(owns, rivals, s=60, c='#2c6fbb', zorder=3)
    for i, l in enumerate(labels):
        ax[1].annotate(l, (owns[i], rivals[i]), fontsize=6,
                       xytext=(3, 3), textcoords='offset points')
    for lbl, o, r, c in (('Chong E1', CHONG_OWN, CHONG_RIVAL, '#2a9d8f'),
                         ('Chong E3', CHONG3_OWN, CHONG3_RIVAL, '#2a9d8f'),
                         ('Levelt', LEVELT_OWN, LEVELT_RIVAL, '#e76f51')):
        ax[1].scatter([o], [r], s=80, marker='*', c=c, zorder=4)
        ax[1].annotate(lbl, (o, r), fontsize=6, xytext=(3, -9),
                       textcoords='offset points')
    ax[1].axhline(0, ls='--', c='k', lw=1)
    ax[1].axvline(0, ls='--', c='k', lw=1)
    ax[1].set_xlabel('% change, manipulated channel')
    ax[1].set_ylabel('% change, rival channel')
    ax[1].set_title('B  Model conditions and published values')
    fig.tight_layout()
    fig.savefig('wave11_fig_gated.pdf')
    print("\n  wave11_fig_gated.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analyse', action='store_true')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    n_cfg, n_seed, n_steps = (8, 8, 4000) if a.quick else (100, 50, 20000)

    cfgs = load_configs()
    if a.analyse:
        D = _load('gated')
    else:
        pool, _ = w6.eligible_pool(list(cfgs.values()), n_cfg)
        D = campaign(pool, n_seed, n_steps)
    test1(D)
    test2()
    test3(D)
    figure(D)
    json.dump(OUT, open('wave11_results.json', 'w', encoding='utf-8'),
              indent=1, default=float)
    hdr("DONE")
    print("  wave11_results.json")


if __name__ == '__main__':
    main()
