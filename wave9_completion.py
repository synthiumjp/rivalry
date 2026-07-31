"""
wave9_completion.py -- the referee's computational items, all of them.

ESSENTIAL
  #1  Floor occupancy is a hard-rectifier proxy, not the general gate.
      Test drive-to-noise and drive-to-competition as CLASSIFIERS (not linear
      predictors -- that was the wrong test in wave 6), including on softplus
      data where floor occupancy is identically zero and control still fails.
  #5  Input gain is matched only on baseline drive. Add realised-drive matching
      and predominance (outcome) matching, plus a dose-response surface.
  #3  The ANOVA null is asserted, not demonstrated. Reconstruct it: stratify by
      the ratio, balance its distribution, add it as a covariate.
  #2  Gate/gradient conditions on the outcome. Fit a hurdle model; compare
      step, linear, spline and logistic transitions by out-of-sample log loss.
  #16 Pulse definition confounds transient capture with sustained switching.
      Decompose: during-pulse, post-pulse, sustained at window end.

DESIRABLE
  #6  Orientation-adjusted discrimination max(AUC, 1-AUC) alongside signed AUC.
  #7  Report overlap between the four draws.
  #13 Does the effect hold outside the eligible pool? Stratified broad sample.
  #14 Segmentation robustness: margin, minimum dwell, hysteresis, smoothing.
  #15 Winner-take-all as an outcome, not only a nuisance.

Run:
    python wave9_completion.py            # simulate + analyse  (~8 min)
    python wave9_completion.py --analyse  # re-analyse saved blocks
    python wave9_completion.py --quick    # smoke test
"""

import argparse
import itertools
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
    run_trace, extract_durations, dominance_summary, occupancy_stats,
    BURN_IN, MARGIN, X_MAX, G_SAFETY, SIGNAL_NEUTRAL,
    DORSAL_MULTS, VENTRAL_MULTS, N_STEPS_BASELINE, N_STEPS_TOTAL,
    N_STEPS_PULSE, SWITCH_CRITERION, LOOKBACK_WINDOW, load_configs, _rectify,
)
import wave6_robustness as w6

N_BOOT = 4000
G_LEVELS = [0.0, 0.25, 0.50, 0.75, 0.90]
K_SHARP = [2.0, 5.0, 20.0, -1.0]
BOOST_DOSES = [0.5, 1.0, 2.0, 4.0, 8.0]
SEG_RULES = [
    ('registered', 0.05, 1, 0.0, 1),
    ('min_dwell_5', 0.05, 5, 0.0, 1),
    ('min_dwell_10', 0.05, 10, 0.0, 1),
    ('wide_margin', 0.15, 1, 0.0, 1),
    ('hysteresis', 0.10, 1, 0.05, 1),
    ('smoothed', 0.05, 1, 0.0, 11),
]
OUT = {}
PLOTS = []


def hdr(s):
    print("\n" + "=" * 74)
    print(s)
    print("=" * 74)


# =============================================================================
# KERNELS
# =============================================================================

@njit(cache=True)
def pulse_trial_v3(lam, beta, alpha, sigma, gamma_adapt, kappa,
                   signal_a, signal_b, g_value,
                   n_base, n_total, n_pulse, seed, x_max,
                   lookback, criterion, margin, k_sharp):
    """
    Registered pulse protocol, fully instrumented.

    Returns 12 values:
      0 switched (any)          1 latency
      2 target                  3 floor occupancy (baseline)
      4 ceiling occupancy       5 mean residual of suppressed channel
      6 mean activation of dominant channel (baseline)
      7 switch onset occurred DURING the pulse
      8 switch onset occurred AFTER pulse offset
      9 target dominant at the final timestep
     10 fraction of response window with target dominant
     11 mean suppressed activation during the pulse itself
    """
    np.random.seed(seed)
    x_a = 0.1
    x_b = 0.1
    a_a = 0.0
    a_b = 0.0
    lb_a = np.zeros(lookback)
    lb_b = np.zeros(lookback)

    n_floor = 0
    n_ceil = 0
    resid = 0.0
    dom_sum = 0.0
    n_cnt = 0

    for t in range(n_base):
        ea = np.random.normal(0.0, sigma)
        eb = np.random.normal(0.0, sigma)
        na = (1.0 - lam) * x_a + signal_a - beta * x_b - alpha * a_a + ea
        nb = (1.0 - lam) * x_b + signal_b - beta * x_a - alpha * a_b + eb
        x_a = _rectify(na, k_sharp, x_max)
        x_b = _rectify(nb, k_sharp, x_max)
        a_a = (1.0 - gamma_adapt) * a_a + kappa * x_a
        a_b = (1.0 - gamma_adapt) * a_b + kappa * x_b
        if t >= BURN_IN:
            lo = x_a if x_a < x_b else x_b
            hi = x_a if x_a > x_b else x_b
            if lo <= 1e-12:
                n_floor += 1
            if hi >= x_max - 1e-9:
                n_ceil += 1
            resid += lo
            dom_sum += hi
            n_cnt += 1
        if t >= n_base - lookback:
            i = t - (n_base - lookback)
            lb_a[i] = x_a
            lb_b[i] = x_b

    ma = 0.0
    mb = 0.0
    for i in range(lookback):
        ma += lb_a[i]
        mb += lb_b[i]
    ma /= lookback
    mb /= lookback
    target = 1 if ma > mb else 0

    g_a = g_value if target == 0 else 0.0
    g_b = g_value if target == 1 else 0.0

    switched = 0
    latency = -1
    consec = 0
    dur_pulse = 0
    aft_pulse = 0
    n_target_dom = 0
    supp_pulse = 0.0
    n_pulse_steps = 0

    for t in range(n_base, n_total):
        step = t - n_base
        in_pulse = step < n_pulse
        if not in_pulse:
            g_a = 0.0
            g_b = 0.0
        ea = np.random.normal(0.0, sigma)
        eb = np.random.normal(0.0, sigma)
        na = (1.0 - lam + g_a) * x_a + signal_a - beta * x_b - alpha * a_a + ea
        nb = (1.0 - lam + g_b) * x_b + signal_b - beta * x_a - alpha * a_b + eb
        x_a = _rectify(na, k_sharp, x_max)
        x_b = _rectify(nb, k_sharp, x_max)
        a_a = (1.0 - gamma_adapt) * a_a + kappa * x_a
        a_b = (1.0 - gamma_adapt) * a_b + kappa * x_b

        if in_pulse:
            supp_pulse += x_b if target == 0 else x_a
            n_pulse_steps += 1

        d = x_a - x_b
        tgt_dom = (target == 0 and d > margin) or (target == 1 and d < -margin)
        if tgt_dom:
            n_target_dom += 1
            consec += 1
        else:
            consec = 0
        if consec >= criterion and switched == 0:
            switched = 1
            latency = step
            if in_pulse:
                dur_pulse = 1
            else:
                aft_pulse = 1

    d_end = x_a - x_b
    at_end = 1 if ((target == 0 and d_end > margin) or
                   (target == 1 and d_end < -margin)) else 0
    nresp = n_total - n_base
    return (switched, latency, target,
            n_floor / n_cnt, n_ceil / n_cnt, resid / n_cnt, dom_sum / n_cnt,
            dur_pulse, aft_pulse, at_end, n_target_dom / nresp,
            supp_pulse / n_pulse_steps if n_pulse_steps > 0 else 0.0)


@njit(cache=True)
def extract_flexible(ta, tb, burn_in, margin_in, margin_out, min_dur, smooth):
    """
    Segmentation with hysteresis and optional smoothing.
      margin_in  : threshold to ENTER dominance
      margin_out : threshold to LEAVE it (< margin_in gives hysteresis)
      smooth     : boxcar width on the difference trace (1 = none)
    Returns (n_switches, time_a, time_b, n_epi_a, n_epi_b, sum_a, sum_b).
    """
    n = len(ta)
    cur = -1
    dur = 0
    n_sw = 0
    t_a = 0
    t_b = 0
    ea = 0
    eb = 0
    sa = 0
    sb = 0
    last = -1
    half = smooth // 2
    for t in range(burn_in, n):
        if smooth > 1:
            lo = t - half
            hi = t + half + 1
            if lo < 0:
                lo = 0
            if hi > n:
                hi = n
            s = 0.0
            for u in range(lo, hi):
                s += ta[u] - tb[u]
            d = s / (hi - lo)
        else:
            d = ta[t] - tb[t]

        if cur == 0:
            dom = 0 if d > margin_out else (1 if d < -margin_in else -1)
        elif cur == 1:
            dom = 1 if d < -margin_out else (0 if d > margin_in else -1)
        else:
            dom = 0 if d > margin_in else (1 if d < -margin_in else -1)

        if dom == cur and dom >= 0:
            dur += 1
        else:
            if cur >= 0 and dur >= min_dur:
                if cur == 0:
                    ea += 1
                    sa += dur
                    t_a += dur
                else:
                    eb += 1
                    sb += dur
                    t_b += dur
                if last >= 0 and cur != last:
                    n_sw += 1
                last = cur
            cur = dom
            dur = 1 if dom >= 0 else 0
    if cur >= 0 and dur >= min_dur:
        if cur == 0:
            ea += 1
            sa += dur
            t_a += dur
        else:
            eb += 1
            sb += dur
            t_b += dur
        if last >= 0 and cur != last:
            n_sw += 1
    return n_sw, t_a, t_b, ea, eb, sa, sb


def multi_seg(ta, tb):
    """All segmentation rules from one trace."""
    out = {}
    n = len(ta) - BURN_IN
    for name, m_in, mdur, hyst, sm in SEG_RULES:
        m_out = m_in - hyst
        nsw, t_a, t_b, ea, eb, sa, sb = extract_flexible(
            ta, tb, BURN_IN, m_in, m_out, mdur, sm)
        tot = t_a + t_b
        out[name] = {
            'alternation_rate': nsw / n,
            'predominance': (t_a / tot) if tot else np.nan,
            'mean_dur_a': (sa / ea) if ea else np.nan,
            'mean_dur_b': (sb / eb) if eb else np.nan,
            'n_epi_a': int(ea), 'n_epi_b': int(eb),
            'is_wta': bool(ea == 0 or eb == 0 or ea + eb <= 1),
        }
    return out


# =============================================================================
# POOLS
# =============================================================================

def build_pools(cfgs, n, quick=False):
    """Eligible pool and a broader rivalry-producing pool (#13)."""
    with open("phase1_grid_results.json", "r", encoding="utf-8") as f:
        grid = json.load(f)
    keys = ['lambda', 'beta', 'alpha', 'sigma', 'gamma', 'kappa']
    excl = {tuple(round(p[k], 6) for k in keys) for p in cfgs.values()}

    elig, broad = [], []
    for c in grid['results']:
        if not c.get('rivalry_producing'):
            continue
        p = {k: c[k] for k in keys if k in c}
        if len(p) != 6:
            continue
        if tuple(round(p[k], 6) for k in keys) in excl:
            continue
        rm = c.get('rivalry_metrics') or {}
        cv = rm.get('cv')
        lr = c.get('levelt_rho')
        rec = {'params': p, 'cv': cv, 'grid_index': c.get('index', -1)}
        broad.append(rec)
        if cv is not None and 0.35 <= cv <= 0.65 and lr is not None and lr > 0.7:
            elig.append(rec)

    rng = np.random.default_rng(42)
    ei = rng.choice(len(elig), size=min(n, len(elig)), replace=False)
    bi = rng.choice(len(broad), size=min(n, len(broad)), replace=False)
    print("  eligible pool %d, broad rivalry-producing pool %d"
          % (len(elig), len(broad)))
    return ([elig[i] for i in sorted(ei)], [broad[i] for i in sorted(bi)],
            len(elig), len(broad))


def draw_overlap(cfgs, n=100):
    """#7: the four draws come from 732; report overlap."""
    with open("phase1_grid_results.json", "r", encoding="utf-8") as f:
        grid = json.load(f)
    keys = ['lambda', 'beta', 'alpha', 'sigma', 'gamma', 'kappa']
    excl = {tuple(round(p[k], 6) for k in keys) for p in cfgs.values()}
    elig = []
    for c in grid['results']:
        if not c.get('rivalry_producing'):
            continue
        lr = c.get('levelt_rho')
        rm = c.get('rivalry_metrics') or {}
        cv = rm.get('cv')
        p = {k: c[k] for k in keys if k in c}
        if (len(p) == 6 and lr is not None and lr > 0.7 and cv is not None
                and 0.35 <= cv <= 0.65
                and tuple(round(p[k], 6) for k in keys) not in excl):
            elig.append(c.get('index', -1))
    sets = []
    for s in (42, 43, 44, 45):
        rng = np.random.default_rng(s)
        idx = rng.choice(len(elig), size=n, replace=False)
        sets.append(set(elig[i] for i in idx))
    ov = [(a, b, len(sets[a] & sets[b]))
          for a, b in itertools.combinations(range(4), 2)]
    union = len(set().union(*sets))
    return {'pool_size': len(elig), 'per_draw': n,
            'pairwise_overlap': [{'draws': [int(a), int(b)], 'shared': int(k)}
                                 for a, b, k in ov],
            'mean_overlap': float(np.mean([k for _, _, k in ov])),
            'union': int(union)}


# =============================================================================
# BLOCK M -- instrumented pulse protocol, hard + softplus, both pools
# =============================================================================

def block_M(elig, broad, n_seeds):
    hdr("BLOCK M: instrumented pulse protocol")
    out = []
    t0 = time.time()
    jobs = ([('eligible', e) for e in elig] + [('broad', b) for b in broad])
    for n, (pool_tag, ent) in enumerate(jobs):
        p = ent['params']
        lam, sigma, gam = p['lambda'], p['sigma'], p['gamma']
        ci = ent['grid_index'] % 1000
        rec = {'pool': pool_tag, 'grid_index': ent['grid_index'],
               'params': p, 'cv': ent.get('cv'), 'regimes': {}}
        for ri, (rn, m) in enumerate([('adaptation', DORSAL_MULTS),
                                      ('inhibition', VENTRAL_MULTS)]):
            a_r = p['alpha'] * m['alpha']
            b_r = p['beta'] * m['beta']
            k_r = p['kappa'] * m['kappa']
            reg = {'alpha_over_lambda': a_r / lam, 'beta_regime': b_r,
                   'levels': {}, 'softplus': {}}
            for li, gf in enumerate(G_LEVELS):
                gv = min(gf * lam, G_SAFETY * lam)
                acc = np.zeros(12)
                for s in range(n_seeds):
                    r = pulse_trial_v3(lam, b_r, a_r, sigma, gam, k_r,
                                       SIGNAL_NEUTRAL, SIGNAL_NEUTRAL, gv,
                                       N_STEPS_BASELINE, N_STEPS_TOTAL,
                                       N_STEPS_PULSE,
                                       w6.make_seed(0, ci, ri, li, s),
                                       X_MAX, LOOKBACK_WINDOW,
                                       SWITCH_CRITERION, MARGIN, -1.0)
                    acc += np.array(r)
                a = acc / n_seeds
                reg['levels']["G%d" % int(gf * 100)] = {
                    'g_fraction': gf, 'g_value': gv,
                    'switch_rate': float(a[0]),
                    'floor_frac': float(a[3]), 'ceil_frac': float(a[4]),
                    'resid_supp': float(a[5]), 'mean_dom': float(a[6]),
                    'switch_during_pulse': float(a[7]),
                    'switch_after_pulse': float(a[8]),
                    'target_dom_at_end': float(a[9]),
                    'frac_window_target': float(a[10]),
                    'resid_during_pulse': float(a[11]),
                }
            # softplus variants at G = 90%, eligible pool only (#1)
            if pool_tag == 'eligible':
                for ki, k in enumerate(K_SHARP):
                    gv = min(0.9 * lam, G_SAFETY * lam)
                    acc = np.zeros(12)
                    for s in range(n_seeds):
                        r = pulse_trial_v3(lam, b_r, a_r, sigma, gam, k_r,
                                           SIGNAL_NEUTRAL, SIGNAL_NEUTRAL, gv,
                                           N_STEPS_BASELINE, N_STEPS_TOTAL,
                                           N_STEPS_PULSE,
                                           w6.make_seed(1, ci, ri, ki, s),
                                           X_MAX, LOOKBACK_WINDOW,
                                           SWITCH_CRITERION, MARGIN, k)
                        acc += np.array(r)
                    a = acc / n_seeds
                    reg['softplus']["k%s" % ('hard' if k < 0 else ('%g' % k))] = {
                        'k_sharp': k, 'g_value': gv,
                        'switch_rate': float(a[0]), 'floor_frac': float(a[3]),
                        'resid_supp': float(a[5]), 'mean_dom': float(a[6]),
                    }
            rec['regimes'][rn] = reg
        out.append(rec)
        if (n + 1) % 25 == 0:
            print("    %3d/%d  (%.0fs)" % (n + 1, len(jobs), time.time() - t0))
    _save('M_pulse', out)
    return out


# =============================================================================
# BLOCK N -- matching schemes and dose-response (#5, #14, #15)
# =============================================================================

def block_N(elig, n_seeds, n_steps):
    hdr("BLOCK N: matching schemes, dose-response, segmentation robustness")
    out = []
    t0 = time.time()
    for n, ent in enumerate(elig):
        p = ent['params']
        lam, beta, alpha = p['lambda'], p['beta'], p['alpha']
        sigma, gam, kap = p['sigma'], p['gamma'], p['kappa']
        ci = ent['grid_index'] % 1000
        gv = min(0.5 * lam, G_SAFETY * lam)

        acts = []
        for s in range(20):
            ta, _ = run_trace(lam, beta, alpha, sigma, gam, kap,
                              SIGNAL_NEUTRAL, SIGNAL_NEUTRAL, 0, 0, 0, 0,
                              n_steps, w6.make_seed(2, ci, 0, 0, s),
                              X_MAX, -1.0, 0.0, 0.0)
            acts.append(float(np.mean(ta[BURN_IN:])))
        x_base = float(np.mean(acts))
        boost_base = gv * x_base          # baseline-drive matching

        conds = {'baseline': 0.0, 'gclca': None}
        for d in BOOST_DOSES:
            conds['boost_%g' % d] = boost_base * d

        rec = {'grid_index': ent['grid_index'], 'params': p,
               'g_value': gv, 'x_baseline': x_base,
               'boost_baseline_matched': boost_base, 'conditions': {}}

        gclca_x = []
        for li, (cn, bo) in enumerate(conds.items()):
            rows = []
            for s in range(n_seeds):
                ga = gv if cn == 'gclca' else 0.0
                ba = 0.0 if bo is None else bo
                ta, tb = run_trace(lam, beta, alpha, sigma, gam, kap,
                                   SIGNAL_NEUTRAL, SIGNAL_NEUTRAL,
                                   ga, 0.0, ba, 0.0,
                                   n_steps, w6.make_seed(3, ci, 0, li, s),
                                   X_MAX, -1.0, 0.0, 0.0)
                seg = multi_seg(ta, tb)
                if cn == 'gclca':
                    gclca_x.append(float(np.mean(ta[BURN_IN:])))
                fl, ce, rs = occupancy_stats(ta, tb, BURN_IN, X_MAX)
                seg['_ceil'] = float(ce)
                rows.append(seg)
            agg = {}
            for name, _, _, _, _ in SEG_RULES:
                sub = [r[name] for r in rows]
                agg[name] = {
                    'alternation_rate': float(np.nanmean(
                        [x['alternation_rate'] for x in sub])),
                    'predominance': float(np.nanmean(
                        [x['predominance'] for x in sub])),
                    'mean_dur_a': float(np.nanmean(
                        [x['mean_dur_a'] for x in sub])),
                    'mean_dur_b': float(np.nanmean(
                        [x['mean_dur_b'] for x in sub])),
                    'wta_rate': float(np.mean([x['is_wta'] for x in sub])),
                }
            agg['ceil_frac'] = float(np.mean([r['_ceil'] for r in rows]))
            rec['conditions'][cn] = agg

        # realised-drive matching: mean(G * x_A) during intervention
        rec['x_during_gclca'] = float(np.mean(gclca_x)) if gclca_x else np.nan
        rec['boost_realised_matched'] = gv * rec['x_during_gclca']
        out.append(rec)
        if (n + 1) % 20 == 0:
            print("    %3d/%d  (%.0fs)" % (n + 1, len(elig), time.time() - t0))
    _save('N_matching', out)
    return out


def _save(tag, obj):
    fn = "wave9_%s.json" % tag
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)
    print("    -> %s" % fn)


def _load(tag):
    fn = "wave9_%s.json" % tag
    if not os.path.exists(fn):
        sys.exit("ERROR: %s missing." % fn)
    with open(fn, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# ANALYSIS
# =============================================================================

def _auc(sc, lab):
    sc, lab = np.asarray(sc, float), np.asarray(lab, bool)
    m = np.isfinite(sc)
    sc, lab = sc[m], lab[m]
    if lab.sum() == 0 or (~lab).sum() == 0:
        return np.nan
    r = stats.rankdata(np.concatenate([sc[lab], sc[~lab]]))
    npos = int(lab.sum())
    return (r[:npos].sum() - npos * (npos + 1) / 2) / (npos * (~lab).sum())


def auc_boot(sc, lab, n_boot=N_BOOT, seed=42):
    a = _auc(sc, lab)
    rng = np.random.default_rng(seed)
    sc, lab = np.asarray(sc, float), np.asarray(lab, bool)
    n = len(sc)
    v = [x for x in (_auc(sc[i], lab[i]) for i in
                     (rng.integers(0, n, n) for _ in range(n_boot)))
         if np.isfinite(x)]
    lo, hi = np.percentile(v, [2.5, 97.5]) if v else (np.nan, np.nan)
    return a, lo, hi


def A_classifiers(M):
    """#1, #6: mechanistic quantities as classifiers, hard and softplus."""
    hdr("A. MECHANISTIC CLASSIFIERS (#1) with orientation adjustment (#6)")
    rows = []
    for rec in M:
        if rec['pool'] != 'eligible':
            continue
        sig = rec['params']['sigma']
        for rn, reg in rec['regimes'].items():
            g90 = reg['levels']['G90']
            g0 = reg['levels']['G0']
            gv = g90['g_value']
            resid = g0['resid_supp']
            dom = g0['mean_dom']
            beta_r = reg['beta_regime']
            sw = max(reg['levels'][k]['switch_rate']
                     for k in reg['levels'] if k != 'G0')
            rows.append({
                'regime': rn, 'rectifier': 'hard',
                'floor': g0['floor_frac'],
                'drive_noise': gv * resid / sig,
                'drive_comp': gv * resid / (beta_r * dom) if dom > 0 else np.nan,
                'resid': resid,
                'alpha_over_lambda': reg['alpha_over_lambda'],
                'switch': sw,
            })
            for kk, sp in reg.get('softplus', {}).items():
                rows.append({
                    'regime': rn, 'rectifier': kk,
                    'floor': sp['floor_frac'],
                    'drive_noise': sp['g_value'] * sp['resid_supp'] / sig,
                    'drive_comp': (sp['g_value'] * sp['resid_supp']
                                   / (beta_r * sp['mean_dom'])
                                   if sp['mean_dom'] > 0 else np.nan),
                    'resid': sp['resid_supp'],
                    'alpha_over_lambda': reg['alpha_over_lambda'],
                    'switch': sp['switch_rate'],
                })

    def col(k, sel):
        return np.array([r[k] for r in rows if sel(r)], float)

    subsets = [
        ('hard, adaptation-dom', lambda r: r['rectifier'] == 'hard'
         and r['regime'] == 'adaptation'),
        ('hard, inhibition-dom', lambda r: r['rectifier'] == 'hard'
         and r['regime'] == 'inhibition'),
        ('hard, pooled', lambda r: r['rectifier'] == 'hard'),
        ('softplus only', lambda r: r['rectifier'] != 'hard'),
        ('ALL (hard + softplus)', lambda r: True),
    ]
    preds = ['floor', 'drive_noise', 'drive_comp', 'resid', 'alpha_over_lambda']
    res = {}
    print("  %-24s %-18s %7s %7s %16s" %
          ("subset", "predictor", "AUC", "|adj|", "95% CI"))
    for sname, sel in subsets:
        y = col('switch', sel) > 0.05
        if y.sum() < 4 or (~y).sum() < 4:
            print("  %-24s (insufficient)" % sname)
            continue
        for pname in preds:
            x = col(pname, sel)
            sgn = -1.0 if pname in ('floor', 'alpha_over_lambda') else 1.0
            a, lo, hi = auc_boot(sgn * x, y)
            adj = max(a, 1 - a) if np.isfinite(a) else np.nan
            res.setdefault(sname, {})[pname] = {
                'auc': float(a), 'adjusted': float(adj),
                'ci': [float(lo), float(hi)]}
            print("  %-24s %-18s %7.3f %7.3f  [%.3f, %.3f]"
                  % (sname, pname, a, adj, lo, hi))
    print("\n  KEY TEST: on softplus data floor occupancy is ~0 everywhere, so a")
    print("  general gating variable must still classify there. Compare the")
    print("  'softplus only' and 'ALL' rows.")
    OUT['classifiers'] = res
    OUT['classifier_rows'] = rows
    return rows


def B_matching(N):
    """#5: three matching schemes plus dose-response."""
    hdr("B. MATCHING SCHEMES (#5)")
    seg = 'registered'
    recs = []
    for r in N:
        b = r['conditions']['baseline'][seg]
        g = r['conditions']['gclca'][seg]
        if not b['mean_dur_b'] or not b['alternation_rate']:
            continue
        doses = []
        for d in BOOST_DOSES:
            c = r['conditions']['boost_%g' % d][seg]
            doses.append({'dose': d, 'pred': c['predominance'],
                          'dpr': (c['mean_dur_b'] / b['mean_dur_b'])
                          if c['mean_dur_b'] else np.nan,
                          'alt': c['alternation_rate'] / b['alternation_rate']})
        # interpolate boost dose that matches GC-LCA predominance
        dp = np.array([d['pred'] for d in doses], float)
        dd = np.array([d['dose'] for d in doses], float)
        ok = np.isfinite(dp)
        target = g['predominance']
        if ok.sum() >= 2 and dp[ok].min() <= target <= dp[ok].max():
            order = np.argsort(dp[ok])
            dose_m = float(np.interp(target, dp[ok][order], dd[ok][order]))
            dpr_m = float(np.interp(target, dp[ok][order],
                                    np.array([d['dpr'] for d in doses])[ok][order]))
            alt_m = float(np.interp(target, dp[ok][order],
                                    np.array([d['alt'] for d in doses])[ok][order]))
        else:
            dose_m = dpr_m = alt_m = np.nan
        why = 'ok'
        if ok.sum() >= 2:
            if target > dp[ok].max():
                why = 'target_above_max_dose'
            elif target < dp[ok].min():
                why = 'target_below_min_dose'
        else:
            why = 'insufficient_doses'
        recs.append({
            'match_status': why,
            'grid_index': r['grid_index'],
            'dpr_g': (g['mean_dur_b'] / b['mean_dur_b']) if g['mean_dur_b'] else np.nan,
            'alt_g': g['alternation_rate'] / b['alternation_rate'],
            'pred_g': g['predominance'],
            'dpr_b1': doses[1]['dpr'], 'alt_b1': doses[1]['alt'],
            'pred_b1': doses[1]['pred'],  # doses[1] == 1.0x baseline-matched
            'dpr_bm': dpr_m, 'alt_bm': alt_m, 'dose_matched': dose_m,
            'realised_ratio': (r['boost_realised_matched']
                               / r['boost_baseline_matched'])
            if r['boost_baseline_matched'] else np.nan,
        })

    def c(k):
        return np.array([x[k] for x in recs], float)

    print("  realised/baseline boost ratio: median %.3f  IQR [%.3f, %.3f]"
          % (np.nanmedian(c('realised_ratio')),
             np.nanpercentile(c('realised_ratio'), 25),
             np.nanpercentile(c('realised_ratio'), 75)))
    print("  predominance-matched boost dose: median %.2fx baseline-matched"
          % np.nanmedian(c('dose_matched')))
    print("  configurations where matching is interpolable: %d/%d"
          % (int(np.isfinite(c('dpr_bm')).sum()), len(recs)))
    from collections import Counter
    for k, v in Counter(x['match_status'] for x in recs).items():
        print("    %-26s %d" % (k, v))
    print("    (target_above_max_dose means persistence produced a predominance")
    print("     shift no tested input-gain dose could reach -- itself a result)")
    res = {}
    for lbl, gk, bk in (('duration (DPR), baseline-matched', 'dpr_g', 'dpr_b1'),
                        ('duration (DPR), PREDOMINANCE-matched', 'dpr_g', 'dpr_bm'),
                        ('alternation, baseline-matched', 'alt_g', 'alt_b1'),
                        ('alternation, PREDOMINANCE-matched', 'alt_g', 'alt_bm')):
        a, b_ = c(gk), c(bk)
        m = np.isfinite(a) & np.isfinite(b_)
        if m.sum() < 5:
            continue
        d = (a[m] - b_[m]).mean() / (a[m] - b_[m]).std(ddof=1)
        t, p = stats.ttest_rel(a[m], b_[m])
        res[lbl] = {'n': int(m.sum()), 'd': float(d), 't': float(t),
                    'p': float(p),
                    'n_pos': int(np.sum(a[m] > b_[m]))}
        print("  %-42s n=%3d  d=%+.2f  p=%.2g  persistence higher %d/%d"
              % (lbl, m.sum(), d, p, res[lbl]['n_pos'], m.sum()))
    print("\n  If the temporal signature survives PREDOMINANCE matching, dose")
    print("  mismatch cannot explain it.")
    OUT['matching'] = {'tests': res, 'per_config': recs}
    return recs


def C_anova(M):
    """#3: demonstrate the null rather than assert it."""
    hdr("C. RECONSTRUCTING THE PRE-REGISTERED NULL (#3)")
    rows = []
    for rec in M:
        if rec['pool'] != 'eligible':
            continue
        for rn, reg in rec['regimes'].items():
            for k, lv in reg['levels'].items():
                if lv['g_fraction'] == 0:
                    continue
                rows.append({'regime': rn, 'g': lv['g_fraction'],
                             'sw': lv['switch_rate'],
                             'aol': reg['alpha_over_lambda'],
                             'cfg': rec['grid_index']})
    reg_a = np.array([r['regime'] == 'adaptation' for r in rows])
    sw = np.array([r['sw'] for r in rows])
    aol = np.array([r['aol'] for r in rows])
    gl = np.array([r['g'] for r in rows])

    print("  marginal means:  adaptation-dom %.4f   inhibition-dom %.4f"
          % (sw[reg_a].mean(), sw[~reg_a].mean()))
    t, p = stats.ttest_ind(sw[reg_a], sw[~reg_a], equal_var=False)
    print("  marginal contrast: t=%+.2f p=%.3f   <-- the pre-registered null"
          % (t, p))

    print("\n  alpha/lambda distribution by regime:")
    for lbl, m in (('adaptation-dom', reg_a), ('inhibition-dom', ~reg_a)):
        print("    %-16s median %.3f  IQR [%.3f, %.3f]  range [%.3f, %.3f]"
              % (lbl, np.median(aol[m]), np.percentile(aol[m], 25),
                 np.percentile(aol[m], 75), aol[m].min(), aol[m].max()))
    ov = (min(aol[reg_a].max(), aol[~reg_a].max())
          - max(aol[reg_a].min(), aol[~reg_a].min()))
    print("    overlap width: %.3f" % max(ov, 0))

    print("\n  within-regime slope of switch rate on log(alpha/lambda):")
    slopes = {}
    for lbl, m in (('adaptation-dom', reg_a), ('inhibition-dom', ~reg_a)):
        b = np.polyfit(np.log(aol[m]), sw[m], 1)[0]
        r, pv = stats.spearmanr(aol[m], sw[m])
        slopes[lbl] = {'slope': float(b), 'rho': float(r), 'p': float(pv)}
        print("    %-16s slope %+.4f   rho %+.3f  p=%.2g" % (lbl, b, r, pv))

    print("\n  STRATIFIED on alpha/lambda (common support only):")
    lo = max(aol[reg_a].min(), aol[~reg_a].min())
    hi = min(aol[reg_a].max(), aol[~reg_a].max())
    ins = (aol >= lo) & (aol <= hi)
    if ins.sum() > 20 and (ins & reg_a).sum() > 5 and (ins & ~reg_a).sum() > 5:
        t2, p2 = stats.ttest_ind(sw[ins & reg_a], sw[ins & ~reg_a],
                                 equal_var=False)
        print("    n=%d in common support  adaptation %.4f  inhibition %.4f"
              % (ins.sum(), sw[ins & reg_a].mean(), sw[ins & ~reg_a].mean()))
        print("    contrast: t=%+.2f p=%.3g" % (t2, p2))
    else:
        t2 = p2 = np.nan
        print("    insufficient common support -- the regimes barely overlap,")
        print("    which is itself the explanation.")

    print("\n  regime effect WITH alpha/lambda as covariate:")
    X = np.column_stack([np.ones(len(sw)), reg_a.astype(float),
                         np.log(aol), gl])
    beta_hat, *_ = np.linalg.lstsq(X, sw, rcond=None)
    resid = sw - X @ beta_hat
    se = np.sqrt(np.sum(resid ** 2) / (len(sw) - X.shape[1])
                 * np.diag(np.linalg.pinv(X.T @ X)))
    tt = beta_hat / se
    print("    regime coefficient %+.4f  t=%+.2f" % (beta_hat[1], tt[1]))
    print("    log(a/l) coefficient %+.4f  t=%+.2f" % (beta_hat[2], tt[2]))
    print("\n  Interpretation: the marginal contrast is null while within-regime")
    print("  associations are strong and OPPOSITE in sign. Whether that")
    print("  constitutes cancellation is now shown, not asserted.")
    OUT['anova_reconstruction'] = {
        'marginal': {'t': float(t), 'p': float(p),
                     'mean_adaptation': float(sw[reg_a].mean()),
                     'mean_inhibition': float(sw[~reg_a].mean())},
        'within_regime_slopes': slopes,
        'stratified': {'t': float(t2), 'p': float(p2)},
        'covariate_model': {'regime_coef': float(beta_hat[1]),
                            'regime_t': float(tt[1]),
                            'aol_coef': float(beta_hat[2]),
                            'aol_t': float(tt[2])},
    }


def D_hurdle(rows):
    """#2: hurdle model, functional form comparison by log loss."""
    hdr("D. HURDLE MODEL AND FUNCTIONAL FORM (#2)")
    hard = [r for r in rows if r['rectifier'] == 'hard']
    x = np.array([r['floor'] for r in hard], float)
    y = np.array([r['switch'] for r in hard], float)
    lab = (y > 0.05).astype(float)

    def logloss_loo(X):
        X = np.atleast_2d(X.T).T
        n = len(lab)
        Xd = np.column_stack([np.ones(n), X])
        ll = 0.0
        for i in range(n):
            m = np.ones(n, bool)
            m[i] = False
            b = np.zeros(Xd.shape[1])
            for _ in range(50):
                eta = Xd[m] @ b
                pr = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
                W = np.clip(pr * (1 - pr), 1e-6, None)
                z = eta + (lab[m] - pr) / W
                try:
                    b = np.linalg.lstsq(Xd[m] * np.sqrt(W)[:, None],
                                        z * np.sqrt(W), rcond=None)[0]
                except np.linalg.LinAlgError:
                    break
            pi = 1 / (1 + np.exp(-np.clip(Xd[i] @ b, -30, 30)))
            pi = min(max(pi, 1e-6), 1 - 1e-6)
            ll += lab[i] * np.log(pi) + (1 - lab[i]) * np.log(1 - pi)
        return -ll / n

    thr = np.median(x)
    forms = {
        'intercept only': np.zeros((len(x), 0)),
        'linear': x[:, None],
        'step (median split)': (x < thr).astype(float)[:, None],
        'logistic transition': (1 / (1 + np.exp(20 * (x - 0.5))))[:, None],
        'spline (3 knots)': np.column_stack(
            [x] + [np.clip(x - k, 0, None) ** 3
                   for k in np.percentile(x, [25, 50, 75])]),
    }
    print("  STAGE 1  P(any switch), leave-one-out log loss (lower is better)")
    ll = {}
    for name, X in forms.items():
        try:
            ll[name] = float(logloss_loo(X)) if X.shape[1] else float(
                logloss_loo(np.zeros((len(x), 1))))
        except Exception:
            ll[name] = np.nan
        print("    %-24s %.4f" % (name, ll[name]))
    best = min((v, k) for k, v in ll.items() if np.isfinite(v))
    print("    best: %s" % best[1])

    sw = lab.astype(bool)
    print("\n  STAGE 2  switch rate among switchers only (n=%d)" % sw.sum())
    if sw.sum() >= 10:
        r, p = stats.spearmanr(x[sw], y[sw])
        print("    rho(floor, rate | switching) = %+.3f  p=%.3g" % (r, p))
        print("    NOTE: this conditions on the outcome and the predictor range")
        print("    is compressed (floor IQR among switchers %.3f-%.3f versus"
              % (np.percentile(x[sw], 25), np.percentile(x[sw], 75)))
        print("    %.3f-%.3f overall). A near-zero correlation here is weak"
              % (np.percentile(x, 25), np.percentile(x, 75)))
        print("    evidence for a pure gate.")
    OUT['hurdle'] = {'stage1_logloss': ll, 'best_form': best[1],
                     'n_switchers': int(sw.sum())}


def E_pulse_timing(M):
    """#16: transient capture versus sustained switching."""
    hdr("E. PULSE TIMING DECOMPOSITION (#16)")
    dur, aft, end, frac, sw = [], [], [], [], []
    for rec in M:
        if rec['pool'] != 'eligible':
            continue
        for reg in rec['regimes'].values():
            for k, lv in reg['levels'].items():
                if lv['g_fraction'] == 0 or lv['switch_rate'] <= 0:
                    continue
                sw.append(lv['switch_rate'])
                dur.append(lv['switch_during_pulse'])
                aft.append(lv['switch_after_pulse'])
                end.append(lv['target_dom_at_end'])
                frac.append(lv['frac_window_target'])
    sw, dur, aft = np.array(sw), np.array(dur), np.array(aft)
    end, frac = np.array(end), np.array(frac)
    print("  cells with any switching: %d" % len(sw))
    print("  of switches, fraction with onset DURING the pulse : %.3f"
          % (dur.sum() / sw.sum()))
    print("  of switches, fraction with onset AFTER the pulse  : %.3f"
          % (aft.sum() / sw.sum()))
    print("  target still dominant at end of response window   : %.3f"
          % (end.sum() / sw.sum()))
    print("  mean fraction of response window target-dominant  : %.3f"
          % frac.mean())
    print("\n  If most switch onsets fall inside the pulse and the target is")
    print("  still dominant at window end, the effect is sustained capture")
    print("  rather than transient threshold crossing.")
    OUT['pulse_timing'] = {
        'n_cells': int(len(sw)),
        'frac_during': float(dur.sum() / sw.sum()),
        'frac_after': float(aft.sum() / sw.sum()),
        'frac_dominant_at_end': float(end.sum() / sw.sum()),
        'mean_window_fraction': float(frac.mean())}


def F_segmentation(N):
    """#14: does the dissociation survive segmentation choices?"""
    hdr("F. SEGMENTATION ROBUSTNESS (#14)")
    print("  %-14s %10s %10s %10s %10s" %
          ("rule", "d(DPR)", "d(alt)", "n(DPR)", "WTA gclca"))
    res = {}
    for name, _, _, _, _ in SEG_RULES:
        dg, db, ag, ab, w = [], [], [], [], []
        for r in N:
            b = r['conditions']['baseline'][name]
            g = r['conditions']['gclca'][name]
            x = r['conditions']['boost_1'][name]
            if not b['mean_dur_b'] or not b['alternation_rate']:
                continue
            dg.append(g['mean_dur_b'] / b['mean_dur_b'] if g['mean_dur_b'] else np.nan)
            db.append(x['mean_dur_b'] / b['mean_dur_b'] if x['mean_dur_b'] else np.nan)
            ag.append(g['alternation_rate'] / b['alternation_rate'])
            ab.append(x['alternation_rate'] / b['alternation_rate'])
            w.append(g['wta_rate'])
        dg, db = np.array(dg, float), np.array(db, float)
        ag, ab = np.array(ag, float), np.array(ab, float)
        m1 = np.isfinite(dg) & np.isfinite(db)
        m2 = np.isfinite(ag) & np.isfinite(ab)
        d1 = ((dg[m1] - db[m1]).mean() / (dg[m1] - db[m1]).std(ddof=1)
              if m1.sum() > 2 else np.nan)
        d2 = ((ag[m2] - ab[m2]).mean() / (ag[m2] - ab[m2]).std(ddof=1)
              if m2.sum() > 2 else np.nan)
        res[name] = {'d_dpr': float(d1), 'd_alt': float(d2),
                     'n_dpr': int(m1.sum()), 'wta': float(np.mean(w))}
        print("  %-14s %+10.2f %+10.2f %10d %10.3f"
              % (name, d1, d2, m1.sum(), np.mean(w)))
    OUT['segmentation'] = res


def G_wta_outcome(N):
    """#15: winner-take-all as an outcome, and effects conditional on rivalry."""
    hdr("G. WINNER-TAKE-ALL AS AN OUTCOME (#15)")
    seg = 'registered'
    wg, wb, dg, db, ag, ab = [], [], [], [], [], []
    for r in N:
        b = r['conditions']['baseline'][seg]
        g = r['conditions']['gclca'][seg]
        x = r['conditions']['boost_1'][seg]
        wg.append(g['wta_rate'])
        wb.append(x['wta_rate'])
        if not b['mean_dur_b'] or not b['alternation_rate']:
            continue
        dg.append(g['mean_dur_b'] / b['mean_dur_b'] if g['mean_dur_b'] else np.nan)
        db.append(x['mean_dur_b'] / b['mean_dur_b'] if x['mean_dur_b'] else np.nan)
        ag.append(g['alternation_rate'] / b['alternation_rate'])
        ab.append(x['alternation_rate'] / b['alternation_rate'])
    wg, wb = np.array(wg), np.array(wb)
    print("  P(winner-take-all): persistence %.4f   input gain %.4f"
          % (wg.mean(), wb.mean()))
    t, p = stats.ttest_rel(wg, wb)
    print("  paired t=%+.2f p=%.3g   persistence higher in %d/%d"
          % (t, p, int(np.sum(wg > wb)), len(wg)))
    keep = wg < 0.01
    dg, db = np.array(dg, float), np.array(db, float)
    ag, ab = np.array(ag, float), np.array(ab, float)
    k = keep[:len(dg)]
    if k.sum() > 10:
        m1 = k & np.isfinite(dg) & np.isfinite(db)
        m2 = k & np.isfinite(ag) & np.isfinite(ab)
        d1 = (dg[m1] - db[m1]).mean() / (dg[m1] - db[m1]).std(ddof=1)
        d2 = (ag[m2] - ab[m2]).mean() / (ag[m2] - ab[m2]).std(ddof=1)
        print("\n  restricted to rivalry-preserving configurations (n=%d):"
              % m1.sum())
        print("    d(DPR) = %+.2f    d(alternation) = %+.2f" % (d1, d2))
        OUT['wta_outcome'] = {'p_wta_gclca': float(wg.mean()),
                              'p_wta_boost': float(wb.mean()),
                              'd_dpr_rivalry_only': float(d1),
                              'd_alt_rivalry_only': float(d2)}


def H_broad(M):
    """#13: does the classifier hold outside the eligible pool?"""
    hdr("H. BEYOND THE ELIGIBLE POOL (#13)")
    for pool in ('eligible', 'broad'):
        fl, sw, rg = [], [], []
        for rec in M:
            if rec['pool'] != pool:
                continue
            for rn, reg in rec['regimes'].items():
                fl.append(reg['levels']['G0']['floor_frac'])
                sw.append(max(reg['levels'][k]['switch_rate']
                              for k in reg['levels'] if k != 'G0'))
                rg.append(rn)
        fl, sw, rg = np.array(fl), np.array(sw), np.array(rg)
        lab = sw > 0.05
        print("  %-9s n=%3d  switching %.1f%%" % (pool, len(fl),
                                                  100 * lab.mean()))
        for rn in ('adaptation', 'inhibition'):
            m = rg == rn
            if (lab[m].sum() < 3) or ((~lab[m]).sum() < 3):
                print("      %-12s (insufficient)" % rn)
                continue
            a, lo, hi = auc_boot(-fl[m], lab[m])
            print("      %-12s AUC floor = %.3f [%.3f, %.3f]" % (rn, a, lo, hi))
        OUT.setdefault('broad_pool', {})[pool] = {
            'n': int(len(fl)), 'frac_switching': float(lab.mean())}


def I_eigen():
    """#9: report Jacobian eigenvalues already computed."""
    hdr("I. FIXED-POINT EIGENVALUES (#9)")
    fn = 'paper1_mechanistic_results.json'
    if not os.path.exists(fn):
        print("  %s not found; skipping" % fn)
        return
    d = json.load(open(fn, encoding='utf-8'))
    tbl = []
    for cfg in d.get('b11_phase_portraits', []):
        for cond, cd in cfg['conditions'].items():
            for fp in cd['fixed_points']:
                tbl.append({'config': cfg['label'], 'condition': cond,
                            'type': fp['type'], 'x_a': fp['x_a'],
                            'x_b': fp['x_b'],
                            'ev_real': fp['eigenvalues_real'],
                            'ev_imag': fp['eigenvalues_imag'],
                            'stable': fp['stable'],
                            'oscillatory': fp.get('oscillatory', False)})
    print("  %-16s %-9s %-12s %8s %8s %10s %6s"
          % ("config", "condition", "type", "Re(l1)", "Re(l2)", "Im", "stable"))
    for r in tbl:
        print("  %-16s %-9s %-12s %8.4f %8.4f %10.4f %6s"
              % (r['config'][:16], r['condition'], r['type'][:12],
                 r['ev_real'][0], r['ev_real'][1],
                 max(abs(np.array(r['ev_imag']))), r['stable']))
    nosc = sum(1 for r in tbl if max(abs(np.array(r['ev_imag']))) > 1e-9)
    print("\n  fixed points with complex eigenvalues (spiral/oscillatory): %d/%d"
          % (nosc, len(tbl)))
    print("  NOTE: adaptation was held at its steady-state value, so this is a")
    print("  reduced two-dimensional analysis of a four-dimensional system.")
    OUT['eigenvalues'] = tbl


def figures(rows, N):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    hard = [r for r in rows if r['rectifier'] == 'hard']
    soft = [r for r in rows if r['rectifier'] != 'hard']
    for grp, c, lbl in ((hard, '#2c6fbb', 'hard rectifier'),
                        (soft, '#d1495b', 'softplus')):
        ax[0].scatter([r['drive_noise'] for r in grp],
                      [r['switch'] for r in grp], s=18, alpha=0.55,
                      c=c, label=lbl)
    ax[0].set_xscale('symlog', linthresh=1e-4)
    ax[0].set_xlabel(r'$G\,\bar{x}_{supp}/\sigma$')
    ax[0].set_ylabel('max switch rate')
    ax[0].set_title('A  Drive-to-noise, both rectifiers')
    ax[0].legend(fontsize=7)

    for grp, c, lbl in ((hard, '#2c6fbb', 'hard'), (soft, '#d1495b', 'softplus')):
        ax[1].scatter([r['floor'] for r in grp],
                      [r['switch'] for r in grp], s=18, alpha=0.55,
                      c=c, label=lbl)
    ax[1].set_xlabel('floor occupancy')
    ax[1].set_ylabel('max switch rate')
    ax[1].set_title('B  Floor occupancy fails on softplus')
    ax[1].legend(fontsize=7)

    names = [n for n, *_ in SEG_RULES]
    dv = [OUT['segmentation'][n]['d_dpr'] for n in names]
    av = [OUT['segmentation'][n]['d_alt'] for n in names]
    xx = np.arange(len(names))
    ax[2].bar(xx - 0.2, dv, 0.4, label='d(DPR)', color='#2c6fbb')
    ax[2].bar(xx + 0.2, av, 0.4, label='d(alternation)', color='#d1495b')
    ax[2].axhline(0, c='k', lw=1)
    ax[2].set_xticks(xx)
    ax[2].set_xticklabels(names, rotation=35, ha='right', fontsize=6)
    ax[2].set_title('C  Segmentation robustness')
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig('wave9_fig1.pdf')
    PLOTS.append('wave9_fig1.pdf')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analyse', action='store_true')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    n_cfg, ns_m, ns_n, nsteps = (8, 8, 8, 4000) if a.quick else (100, 50, 100, 20000)

    cfgs = load_configs()
    if a.analyse:
        M, N = _load('M_pulse'), _load('N_matching')
    else:
        elig, broad, ne, nb = build_pools(cfgs, n_cfg, a.quick)
        OUT['pools'] = {'n_eligible': ne, 'n_broad': nb, 'n_sampled': n_cfg}
        t0 = time.time()
        M = block_M(elig, broad, ns_m)
        N = block_N(elig, ns_n, nsteps)
        print("\n  simulation total: %.0f s" % (time.time() - t0))

    OUT['draw_overlap'] = draw_overlap(cfgs, n_cfg)
    rows = A_classifiers(M)
    B_matching(N)
    C_anova(M)
    D_hurdle(rows)
    E_pulse_timing(M)
    F_segmentation(N)
    G_wta_outcome(N)
    H_broad(M)
    I_eigen()

    hdr("J. DRAW OVERLAP (#7)")
    o = OUT['draw_overlap']
    print("  pool %d, %d per draw, mean pairwise overlap %.1f configurations"
          % (o['pool_size'], o['per_draw'], o['mean_overlap']))
    print("  union across four draws: %d distinct configurations" % o['union'])
    print("  -> these are RESAMPLES, not independent replications.")

    figures(rows, N)
    with open('wave9_results.json', 'w', encoding='utf-8') as f:
        json.dump(OUT, f, indent=1, default=float)
    hdr("DONE")
    print("  wave9_results.json")
    for p in PLOTS:
        print("  " + p)


if __name__ == '__main__':
    main()
