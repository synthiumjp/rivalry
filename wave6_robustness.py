"""
wave6_robustness.py -- extended samples, interval estimates, held-out validation.

Addresses the objections raised against the v4 draft:

  1. Predominance "null" (p = .46, n = 10) is not a null. The per-config
     differences are bimodal and track winner-take-all rate. Replace the
     equivalence claim with the interaction that is actually there.
  2. The floor threshold 0.901 was fitted in-sample on the same 30 configs it
     was then scored on. Fit on 30, validate on 100 FRESH configurations.
  3. n = 10 carries the headline. Extend to 100 configurations drawn from the
     762 eligible pool.
  4. Section 4.3 circularity: report grid-level CV statistics, not
     selected-set ones.
  5. Pooled mediator treats 30 paired configs as 60 independent. Report
     per-regime.
  6. No interval estimates anywhere. Bootstrap CIs on every correlation and
     effect size, Wilson CIs on every proportion.
  7. Does floor occupancy beat the six raw parameters? Nested LOO comparison.
  8. Modified Prop IV shown on selected 30 only, with no winner-take-all
     logging. Extend and log.
  9. Sharpness sweep ran at G = 90%lambda only. Sweep goal level too.

Run:
    python wave6_robustness.py              # simulate + analyse
    python wave6_robustness.py --analyse    # re-analyse saved blocks only
    python wave6_robustness.py --quick      # smoke test

Writes wave6_<BLOCK>.json, wave6_results.json, wave6_fig*.pdf
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
    run_trace, extract_durations, pulse_trial, seed_summary,
    aggregate, load_configs,
    BURN_IN, MARGIN, X_MAX, G_SAFETY, SIGNAL_NEUTRAL,
    DORSAL_MULTS, VENTRAL_MULTS, STRICT_CONFIGS,
    N_STEPS_BASELINE, N_STEPS_TOTAL, N_STEPS_PULSE,
    SWITCH_CRITERION, LOOKBACK_WINDOW,
)

N_BOOT = 10000
RNG_SEED = 42


def make_seed(block, config, regime, level, seed):
    """
    Supersedes the arithmetic scheme in wave2_campaign.

    That scheme is injective BEFORE the modulus, but its maximum value is
    about 4e10 against a modulus of 2^31 - 1, so it wraps roughly nineteen
    times and distinct cells can collide after reduction. That is the same
    defect class as the additive-offset scheme it replaced, just rarer and
    unstructured rather than systematic.

    SeedSequence is built for this: it derives well-separated streams from a
    tuple of integers with no wraparound structure.
    """
    ss = np.random.SeedSequence([int(block), int(config), int(regime),
                                 int(level), int(seed)])
    return int(ss.generate_state(1, dtype=np.uint32)[0] % (2 ** 31 - 1))
EQUAL_SWEEP = [round(0.20 + 0.05 * i, 2) for i in range(11)]
K_VALUES = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, -1.0]
G_LEVELS_SHARP = [0.25, 0.50, 0.75, 0.90]

OUT = {}
PLOTS = []


# =============================================================================
# INTERVAL ESTIMATION
# =============================================================================

def ci_str(lo, hi, fmt="%+.3f"):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "[  --  ,   --  ]"
    return "[" + fmt % lo + ", " + fmt % hi + "]"


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def boot_spearman(x, y, n_boot=N_BOOT, seed=RNG_SEED):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    rs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(x[i])) < 3 or len(np.unique(y[i])) < 3:
            continue
        r = stats.spearmanr(x[i], y[i])[0]
        if np.isfinite(r):
            rs.append(r)
    lo, hi = np.percentile(rs, [2.5, 97.5]) if rs else (np.nan, np.nan)
    return float(rho), float(p), float(lo), float(hi), int(n)


def boot_paired_d(a, b, n_boot=N_BOOT, seed=RNG_SEED):
    """Cohen's d for paired differences, with bootstrap CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    diff = a - b
    sd = diff.std(ddof=1)
    d = diff.mean() / sd if sd else np.nan
    rng = np.random.default_rng(seed)
    n = len(diff)
    ds, ms = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        s = diff[i].std(ddof=1)
        ms.append(diff[i].mean())
        if s > 0:
            ds.append(diff[i].mean() / s)
    dlo, dhi = np.percentile(ds, [2.5, 97.5]) if ds else (np.nan, np.nan)
    mlo, mhi = np.percentile(ms, [2.5, 97.5])
    t, p = stats.ttest_rel(a, b)
    return {'n': int(n), 'mean_diff': float(diff.mean()),
            'mean_ci': [float(mlo), float(mhi)],
            'd': float(d), 'd_ci': [float(dlo), float(dhi)],
            't': float(t), 'p': float(p),
            'n_positive': int(np.sum(diff > 0))}


def auc_ci(scores, labels, n_boot=N_BOOT, seed=RNG_SEED):
    """AUC with bootstrap CI. labels: boolean array (True = positive)."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, bool)
    m = np.isfinite(scores)
    scores, labels = scores[m], labels[m]

    def _auc(s, l):
        pos, neg = s[l], s[~l]
        if len(pos) == 0 or len(neg) == 0:
            return np.nan
        r = stats.rankdata(np.concatenate([pos, neg]))
        return ((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                / (len(pos) * len(neg)))

    a = _auc(scores, labels)
    rng = np.random.default_rng(seed)
    n = len(scores)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        v = _auc(scores[i], labels[i])
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return float(a), float(lo), float(hi)


def loo_r2(X, y):
    """Leave-one-out cross-validated R^2 for OLS."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(y)
    Xd = np.column_stack([np.ones(n), X])
    pred = np.empty(n)
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        beta, *_ = np.linalg.lstsq(Xd[m], y[m], rcond=None)
        pred[i] = Xd[i] @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot else np.nan


# =============================================================================
# ELIGIBLE POOL
# =============================================================================

def eligible_pool(exclude_params, n_sample, seed=RNG_SEED):
    """
    762 eligible = rivalry-producing AND levelt_rho > 0.7 AND cv in [0.35,0.65].
    Note strict inequality on rho: >= gives 764, which is the discrepancy in
    the submitted manuscript.
    """
    fn = "phase1_grid_results.json"
    if not os.path.exists(fn):
        sys.exit("ERROR: %s not found." % fn)
    with open(fn, "r", encoding="utf-8") as f:
        grid = json.load(f)

    keys = ['lambda', 'beta', 'alpha', 'sigma', 'gamma', 'kappa']
    excl = {tuple(round(p[k], 6) for k in keys) for p in exclude_params}

    elig = []
    for c in grid['results']:
        if not c.get('rivalry_producing'):
            continue
        lr = c.get('levelt_rho')
        if lr is None or lr <= 0.7:
            continue
        rm = c.get('rivalry_metrics') or {}
        cv = rm.get('cv')
        if cv is None or not (0.35 <= cv <= 0.65):
            continue
        p = {k: c[k] for k in keys if k in c}
        if len(p) != 6:
            continue
        if tuple(round(p[k], 6) for k in keys) in excl:
            continue
        elig.append({'params': p, 'cv': cv, 'grid_index': c.get('index', -1)})

    print("  eligible pool (excluding Phase II configs): %d" % len(elig))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(elig), size=min(n_sample, len(elig)), replace=False)
    return [elig[i] for i in sorted(idx)], len(elig)


# =============================================================================
# BLOCK G -- extended double dissociation
# =============================================================================

def block_G(pool, n_seeds, n_steps):
    print("\n" + "=" * 70)
    print("BLOCK G: double dissociation, %d configurations" % len(pool))
    print("=" * 70)
    out = []
    t0 = time.time()
    for n, ent in enumerate(pool):
        p = ent['params']
        lam, beta, alpha = p['lambda'], p['beta'], p['alpha']
        sigma, gam, kap = p['sigma'], p['gamma'], p['kappa']
        ci = ent['grid_index'] % 1000
        g_value = min(0.5 * lam, G_SAFETY * lam)

        acts = []
        for s in range(20):
            ta, _ = run_trace(lam, beta, alpha, sigma, gam, kap,
                              SIGNAL_NEUTRAL, SIGNAL_NEUTRAL, 0, 0, 0, 0,
                              n_steps, make_seed(16, ci, 0, 0, s),
                              X_MAX, -1.0, 0.0, 0.0)
            acts.append(float(np.mean(ta[BURN_IN:])))
        mean_x = float(np.mean(acts))
        boost = g_value * mean_x

        rec = {'grid_index': ent['grid_index'], 'params': p,
               'g_value': g_value, 'boost_amount': boost,
               'mean_activation': mean_x, 'conditions': {}}
        for li, (cname, ga, ba) in enumerate(
                [('baseline', 0.0, 0.0), ('gclca', g_value, 0.0),
                 ('boost', 0.0, boost)]):
            rows = []
            for s in range(n_seeds):
                ta, tb = run_trace(lam, beta, alpha, sigma, gam, kap,
                                   SIGNAL_NEUTRAL, SIGNAL_NEUTRAL,
                                   ga, 0.0, ba, 0.0,
                                   n_steps, make_seed(17, ci, 0, li, s),
                                   X_MAX, -1.0, 0.0, 0.0)
                rows.append(seed_summary(ta, tb))
            rec['conditions'][cname] = aggregate(rows)
        out.append(rec)
        if (n + 1) % 10 == 0:
            print("    %3d/%d  (%.0fs)" % (n + 1, len(pool), time.time() - t0))
    _save('G_dissociation', out)
    return out


# =============================================================================
# BLOCK H -- extended mediator, held-out validation set
# =============================================================================

def block_H(pool, n_seeds):
    print("\n" + "=" * 70)
    print("BLOCK H: mediator on %d held-out configurations" % len(pool))
    print("=" * 70)
    out = []
    t0 = time.time()
    for n, ent in enumerate(pool):
        p = ent['params']
        lam, sigma, gam = p['lambda'], p['sigma'], p['gamma']
        ci = ent['grid_index'] % 1000
        rec = {'grid_index': ent['grid_index'], 'params': p, 'regimes': {}}
        for ri, (rn, m) in enumerate([('dorsal', DORSAL_MULTS),
                                      ('ventral', VENTRAL_MULTS)]):
            a_r = p['alpha'] * m['alpha']
            b_r = p['beta'] * m['beta']
            k_r = p['kappa'] * m['kappa']
            rates, floors, ceils = [], [], []
            for li, gf in enumerate([0.0, 0.25, 0.50, 0.75, 0.90]):
                g_value = min(gf * lam, G_SAFETY * lam)
                sw, fl, ce = [], [], []
                for s in range(n_seeds):
                    r = pulse_trial(lam, b_r, a_r, sigma, gam, k_r,
                                    SIGNAL_NEUTRAL, SIGNAL_NEUTRAL, g_value,
                                    N_STEPS_BASELINE, N_STEPS_TOTAL,
                                    N_STEPS_PULSE,
                                    make_seed(18, ci, ri, li, s),
                                    X_MAX, LOOKBACK_WINDOW,
                                    SWITCH_CRITERION, MARGIN, -1.0)
                    sw.append(r[0])
                    fl.append(r[3])
                    ce.append(r[4])
                if gf > 0:
                    rates.append(float(np.mean(sw)))
                else:
                    floors.append(float(np.mean(fl)))
                    ceils.append(float(np.mean(ce)))
            rec['regimes'][rn] = {
                'alpha_over_lambda': a_r / lam,
                'max_switch_rate': float(max(rates)),
                'baseline_floor_frac': floors[0],
                'baseline_ceil_frac': ceils[0],
            }
        out.append(rec)
        if (n + 1) % 20 == 0:
            print("    %3d/%d  (%.0fs)" % (n + 1, len(pool), time.time() - t0))
    _save('H_mediator', out)
    return out


# =============================================================================
# BLOCK I -- Modified Prop IV, extended, with WTA logging
# =============================================================================

def block_I(pool, n_seeds, n_steps):
    print("\n" + "=" * 70)
    print("BLOCK I: Modified Prop IV, %d configurations" % len(pool))
    print("=" * 70)
    out = []
    for n, ent in enumerate(pool):
        p = ent['params']
        ci = ent['grid_index'] % 1000
        levels = []
        for li, s in enumerate(EQUAL_SWEEP):
            rows = []
            for seed in range(n_seeds):
                ta, tb = run_trace(p['lambda'], p['beta'], p['alpha'],
                                   p['sigma'], p['gamma'], p['kappa'],
                                   s, s, 0, 0, 0, 0,
                                   n_steps, make_seed(19, ci, 0, li, seed),
                                   X_MAX, -1.0, 0.0, 0.0)
                rows.append(seed_summary(ta, tb))
            agg = aggregate(rows)
            agg['signal'] = s
            # alternation rate with min-duration 5 applied post hoc is not
            # available from seed_summary; alternation_rate_mean is unfiltered
            levels.append(agg)
        S = np.array([l['signal'] for l in levels])
        A = np.array([l['alternation_rate_mean'] if l['alternation_rate_mean']
                      else np.nan for l in levels], dtype=float)
        m = np.isfinite(A)
        rho, pv = stats.spearmanr(S[m], A[m]) if m.sum() >= 5 else (np.nan, np.nan)
        out.append({'grid_index': ent['grid_index'], 'params': p,
                    'levels': levels, 'rho': float(rho), 'p': float(pv),
                    'wta_max': float(max(l['wta_rate'] for l in levels))})
        if (n + 1) % 20 == 0:
            print("    %3d/%d" % (n + 1, len(pool)))
    _save('I_prop4', out)
    return out


# =============================================================================
# BLOCK J -- sharpness x goal level
# =============================================================================

def block_J(cfgs, n_seeds):
    print("\n" + "=" * 70)
    print("BLOCK J: sharpness x goal level")
    print("=" * 70)
    out = []
    for ci in STRICT_CONFIGS:
        p = cfgs[ci]
        lam, sigma, gam = p['lambda'], p['sigma'], p['gamma']
        rec = {'config_idx': ci, 'params': p, 'sweep': []}
        li = 0
        for k in K_VALUES:
            for gf in G_LEVELS_SHARP:
                g_value = min(gf * lam, G_SAFETY * lam)
                e = {'k_sharp': k, 'g_fraction': gf,
                     'f_at_zero': float(np.log(2) / k) if k > 0 else 0.0}
                for ri, (rn, m) in enumerate([('dorsal', DORSAL_MULTS),
                                              ('ventral', VENTRAL_MULTS)]):
                    sw, fl = [], []
                    for s in range(n_seeds):
                        r = pulse_trial(lam, p['beta'] * m['beta'],
                                        p['alpha'] * m['alpha'], sigma, gam,
                                        p['kappa'] * m['kappa'],
                                        SIGNAL_NEUTRAL, SIGNAL_NEUTRAL,
                                        g_value, N_STEPS_BASELINE,
                                        N_STEPS_TOTAL, N_STEPS_PULSE,
                                        make_seed(20, ci, ri, li, s),
                                        X_MAX, LOOKBACK_WINDOW,
                                        SWITCH_CRITERION, MARGIN, k)
                        sw.append(r[0])
                        fl.append(r[3])
                    e[rn + '_rate'] = float(np.mean(sw))
                    e[rn + '_floor'] = float(np.mean(fl))
                e['dissociation'] = e['dorsal_rate'] - e['ventral_rate']
                rec['sweep'].append(e)
                li += 1
        out.append(rec)
        print("    cfg %2d done" % ci)
    _save('J_sharpness', out)
    return out


def _save(tag, obj):
    fn = "wave6_%s.json" % tag
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)
    print("    -> %s" % fn)


def _load(tag):
    fn = "wave6_%s.json" % tag
    if not os.path.exists(fn):
        sys.exit("ERROR: %s missing. Run without --analyse first." % fn)
    with open(fn, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# ANALYSIS
# =============================================================================

def hdr(s):
    print("\n" + "=" * 70)
    print(s)
    print("=" * 70)


def analyse_dissociation(G):
    hdr("A1. DOUBLE DISSOCIATION, n = %d, with intervals" % len(G))
    rows = []
    for r in G:
        b, g, s = (r['conditions']['baseline'], r['conditions']['gclca'],
                   r['conditions']['boost'])
        if not b['mean_dur_b'] or not b['alternation_rate_mean']:
            continue
        rows.append({
            'grid_index': r['grid_index'],
            'dpr_g': (g['mean_dur_b'] / b['mean_dur_b']) if g['mean_dur_b'] else np.nan,
            'dpr_b': (s['mean_dur_b'] / b['mean_dur_b']) if s['mean_dur_b'] else np.nan,
            'alt_g': (g['alternation_rate_mean'] / b['alternation_rate_mean'])
                     if g['alternation_rate_mean'] is not None else np.nan,
            'alt_b': (s['alternation_rate_mean'] / b['alternation_rate_mean'])
                     if s['alternation_rate_mean'] is not None else np.nan,
            'pred_g': g['predominance_mean'], 'pred_b': s['predominance_mean'],
            'wta_g': g['wta_rate'], 'wta_b': s['wta_rate'],
            'ceil_g': g['ceil_frac_mean'],
        })
    def col(k):
        return np.array([r[k] for r in rows], float)

    print("  configurations with usable baseline: %d" % len(rows))
    print("  DPR defined for %d (gclca) / %d (boost)"
          % (int(np.isfinite(col('dpr_g')).sum()),
             int(np.isfinite(col('dpr_b')).sum())))
    res = {}
    for lbl, ka, kb in (('duration (DPR)', 'dpr_g', 'dpr_b'),
                        ('alternation rate', 'alt_g', 'alt_b'),
                        ('predominance', 'pred_g', 'pred_b')):
        r = boot_paired_d(col(ka), col(kb))
        lo, hi = wilson(r['n_positive'], r['n'])
        r['prop_positive_ci'] = [lo, hi]
        res[lbl] = r
        print("  %-18s n=%3d  mean diff %+.4f [%+.4f, %+.4f]  "
              "d %+.2f [%+.2f, %+.2f]  p=%.2g  gclca higher %d/%d [%.2f, %.2f]"
              % (lbl, r['n'], r['mean_diff'], r['mean_ci'][0], r['mean_ci'][1],
                 r['d'], r['d_ci'][0], r['d_ci'][1], r['p'],
                 r['n_positive'], r['n'], lo, hi))

    # OBJECTION 1: predominance is an interaction, not a null
    hdr("A2. PREDOMINANCE: interaction with winner-take-all, not a null")
    gap = col('pred_g') - col('pred_b')
    wta = col('wta_g')
    rho, p, lo, hi, n = boot_spearman(wta, gap)
    print("  rho(WTA rate, predominance gap) = %+.3f %s  p=%.2g  n=%d"
          % (rho, ci_str(lo, hi), p, n))
    hi_w = wta > 0.05
    if hi_w.sum() >= 3 and (~hi_w).sum() >= 3:
        t, pv = stats.ttest_ind(gap[hi_w], gap[~hi_w], equal_var=False)
        print("  WTA > 5%%   : n=%2d  mean gap %+.4f" % (hi_w.sum(), gap[hi_w].mean()))
        print("  WTA <= 5%%  : n=%2d  mean gap %+.4f" % ((~hi_w).sum(), gap[~hi_w].mean()))
        print("  Welch t=%+.2f p=%.2g" % (t, pv))
        res['predominance_interaction'] = {
            'spearman_rho': rho, 'ci': [lo, hi], 'p': p,
            'welch_t': float(t), 'welch_p': float(pv),
            'mean_gap_high_wta': float(gap[hi_w].mean()),
            'mean_gap_low_wta': float(gap[~hi_w].mean()),
        }
    res['ceiling_max'] = float(np.nanmax(col('ceil_g')))
    print("  max ceiling occupancy under gclca: %.4f" % res['ceiling_max'])
    OUT['dissociation'] = {'per_config': rows, 'tests': res}
    return rows


def analyse_mediator(H, cmed30):
    hdr("A3. MEDIATOR: held-out validation of the 0.901 threshold")
    # threshold fitted on the ORIGINAL 30
    f30, s30 = [], []
    for rec in cmed30:
        for rn in ('dorsal', 'ventral'):
            R = rec['regimes'][rn]
            f30.append(R['baseline_floor_frac'])
            s30.append(R['max_switch_rate'])
    f30, s30 = np.array(f30), np.array(s30)
    thr = max(((t, np.mean((f30 < t) == (s30 > 0.05)))
               for t in np.unique(np.round(f30, 4))), key=lambda x: x[1])
    print("  threshold fitted on original 30 configs: %.4f (in-sample %.1f%%)"
          % (thr[0], 100 * thr[1]))

    # applied to the fresh sample
    fh, sh, ah, rg = [], [], [], []
    for rec in H:
        for rn in ('dorsal', 'ventral'):
            R = rec['regimes'][rn]
            fh.append(R['baseline_floor_frac'])
            sh.append(R['max_switch_rate'])
            ah.append(R['alpha_over_lambda'])
            rg.append(rn)
    fh, sh, ah = np.array(fh), np.array(sh), np.array(ah)
    rg = np.array(rg)
    lab = sh > 0.05
    correct = int(np.sum((fh < thr[0]) == lab))
    lo, hi = wilson(correct, len(fh))
    print("  HELD-OUT: %d/%d correct = %.1f%%  95%% CI [%.1f%%, %.1f%%]"
          % (correct, len(fh), 100 * correct / len(fh), 100 * lo, 100 * hi))

    a, alo, ahi = auc_ci(-fh, lab)
    print("  AUC (floor occupancy)      = %.3f [%.3f, %.3f]" % (a, alo, ahi))
    a2, a2lo, a2hi = auc_ci(-ah, lab)
    print("  AUC (alpha/lambda)         = %.3f [%.3f, %.3f]" % (a2, a2lo, a2hi))

    for rn in ('dorsal', 'ventral'):
        m = rg == rn
        rho, p, l, h, n = boot_spearman(fh[m], sh[m])
        print("  %-8s rho(floor, switch) = %+.3f %s p=%.2g n=%d"
              % (rn, rho, ci_str(l, h), p, n))
        rho2, p2, l2, h2, _ = boot_spearman(ah[m], sh[m])
        print("           rho(a/l,   switch) = %+.3f %s p=%.2g"
              % (rho2, ci_str(l2, h2), p2))

    # OBJECTION 7: does floor beat the six raw parameters?
    hdr("A4. Does floor occupancy beat the six raw parameters?")
    P = []
    for rec in H:
        for rn in ('dorsal', 'ventral'):
            p = rec['params']
            P.append([np.log(p['lambda']), np.log(p['beta']),
                      np.log(p['alpha']), np.log(p['sigma']),
                      np.log(p['gamma']), np.log(max(p['kappa'], 1e-6))])
    P = np.array(P)
    r2_floor = loo_r2(fh[:, None], sh)
    r2_par = loo_r2(P, sh)
    r2_both = loo_r2(np.column_stack([fh, P]), sh)
    print("  LOO R^2, floor occupancy alone (1 predictor)  : %+.3f" % r2_floor)
    print("  LOO R^2, six log-parameters                   : %+.3f" % r2_par)
    print("  LOO R^2, floor + six parameters               : %+.3f" % r2_both)
    print("  -> floor alone %s the six-parameter model"
          % ("MATCHES OR BEATS" if r2_floor >= r2_par else "does NOT beat"))

    OUT['mediator'] = {
        'threshold': float(thr[0]), 'in_sample_acc': float(thr[1]),
        'heldout_correct': correct, 'heldout_n': int(len(fh)),
        'heldout_ci': [float(lo), float(hi)],
        'auc_floor': [a, alo, ahi], 'auc_alpha_lambda': [a2, a2lo, a2hi],
        'loo_r2_floor': float(r2_floor), 'loo_r2_params': float(r2_par),
        'loo_r2_both': float(r2_both),
    }
    return fh, sh, ah, rg, lab


def analyse_prop4(I):
    hdr("A5. MODIFIED PROP IV, n = %d, with winner-take-all logged" % len(I))
    rho = np.array([r['rho'] for r in I], float)
    wta = np.array([r['wta_max'] for r in I], float)
    m = np.isfinite(rho)
    dec = int(np.sum(rho[m] < -0.7))
    inc = int(np.sum(rho[m] > 0.7))
    lo, hi = wilson(dec, int(m.sum()))
    print("  decreasing (rho < -0.7): %d/%d = %.1f%%  95%% CI [%.1f%%, %.1f%%]"
          % (dec, m.sum(), 100 * dec / m.sum(), 100 * lo, 100 * hi))
    print("  increasing (rho > +0.7): %d/%d" % (inc, m.sum()))
    print("  median rho = %+.3f   IQR [%+.3f, %+.3f]"
          % (np.median(rho[m]), np.percentile(rho[m], 25),
             np.percentile(rho[m], 75)))
    print("  max winner-take-all rate across all levels: %.3f" % np.nanmax(wta))
    print("  configs with any WTA > 5%%: %d/%d"
          % (int(np.sum(wta > 0.05)), len(wta)))
    r, p, l, h, n = boot_spearman(wta, rho)
    print("  rho(WTA, prop4 rho) = %+.3f %s p=%.2g" % (r, ci_str(l, h), p))
    OUT['prop4'] = {'n': int(m.sum()), 'n_decreasing': dec,
                    'n_increasing': inc, 'ci': [float(lo), float(hi)],
                    'median_rho': float(np.median(rho[m])),
                    'max_wta': float(np.nanmax(wta))}


def analyse_sharpness(J):
    hdr("A6. SHARPNESS x GOAL LEVEL")
    print("   k     f(0)   |  G=25%   G=50%   G=75%   G=90%   (ventral rate)")
    by_k = {}
    for rec in J:
        for e in rec['sweep']:
            by_k.setdefault(e['k_sharp'], {}).setdefault(
                e['g_fraction'], []).append(e['ventral_rate'])
    for k in K_VALUES:
        f0 = np.log(2) / k if k > 0 else 0.0
        line = "  %5s  %.4f  |" % (('hard' if k < 0 else '%g' % k), f0)
        for gf in G_LEVELS_SHARP:
            v = by_k.get(k, {}).get(gf, [])
            line += "  %.2f  " % (np.mean(v) if v else np.nan)
        print(line)
    print("\n  The dissociation should be present at every goal level once k is")
    print("  large enough. If ventral rate rises with goal level at fixed k,")
    print("  the 90%% result in the draft was goal-level specific.")
    OUT['sharpness'] = {str(k): {str(g): float(np.mean(by_k.get(k, {}).get(g, [np.nan])))
                                 for g in G_LEVELS_SHARP} for k in K_VALUES}


def analyse_grid_cv():
    hdr("A7. CV STATISTICS AT GRID LEVEL (non-circular)")
    fn = "phase1_grid_results.json"
    with open(fn, "r", encoding="utf-8") as f:
        grid = json.load(f)
    riv = [c for c in grid['results'] if c.get('rivalry_producing')]
    cvs = np.array([(c.get('rivalry_metrics') or {}).get('cv', np.nan)
                    for c in riv], float)
    m = np.isfinite(cvs)
    for lo_, hi_ in [(0.35, 0.65), (0.40, 0.60), (0.25, 0.75)]:
        k = int(np.sum((cvs[m] >= lo_) & (cvs[m] <= hi_)))
        wl, wh = wilson(k, int(m.sum()))
        print("  CV in [%.2f, %.2f]: %d/%d = %.1f%%  95%% CI [%.1f%%, %.1f%%]"
              % (lo_, hi_, k, m.sum(), 100 * k / m.sum(), 100 * wl, 100 * wh))
    print("  median CV across rivalry-producing configs: %.3f" % np.median(cvs[m]))
    print("\n  These are the non-circular statistics. The 29/30 figure in the")
    print("  draft is guaranteed by a selection rule that minimised |CV - 0.5|.")
    OUT['grid_cv'] = {'n_rivalry': int(m.sum()),
                      'median_cv': float(np.median(cvs[m]))}


def figures(rows, fh, sh, ah, rg, lab):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    gap = np.array([r['pred_g'] - r['pred_b'] for r in rows], float)
    wta = np.array([r['wta_g'] for r in rows], float)
    ax[0].scatter(100 * wta, gap, s=24, alpha=0.65, c='#2c6fbb')
    ax[0].axhline(0, ls='--', c='k', lw=1)
    ax[0].set_xlabel('% seeds winner-take-all (persistence)')
    ax[0].set_ylabel('predominance: persistence − gain')
    ax[0].set_title('A  Predominance is an interaction')

    d_g = np.array([r['dpr_g'] for r in rows], float)
    d_b = np.array([r['dpr_b'] for r in rows], float)
    a_g = np.array([r['alt_g'] for r in rows], float)
    a_b = np.array([r['alt_b'] for r in rows], float)
    ax[1].scatter(d_b, d_g, s=20, alpha=0.6, c='#2c6fbb', label='duration')
    ax[1].scatter(a_b, a_g, s=20, alpha=0.6, c='#d1495b', label='alternation')
    lim = [0, max(np.nanmax(d_g), np.nanmax(d_b), 1.5)]
    ax[1].plot(lim, lim, ls='--', c='k', lw=1)
    ax[1].set_xlabel('input gain')
    ax[1].set_ylabel('persistence modulation')
    ax[1].set_title('B  Orthogonal axes')
    ax[1].legend(fontsize=7)

    for rn, c in (('dorsal', '#2c6fbb'), ('ventral', '#d1495b')):
        m = rg == rn
        ax[2].scatter(fh[m], sh[m], s=24, alpha=0.65, c=c, label=rn)
    ax[2].axvline(OUT['mediator']['threshold'], ls='--', c='k', lw=1)
    ax[2].set_xlabel('floor occupancy (held-out configs)')
    ax[2].set_ylabel('max switch rate')
    ax[2].set_title('C  Held-out validation')
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig('wave6_fig1_robustness.pdf')
    PLOTS.append('wave6_fig1_robustness.pdf')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analyse', action='store_true')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()

    n_cfg, n_seed_g, n_seed_h, n_seed_i, n_steps = (
        (12, 10, 10, 3, 5000) if args.quick else (100, 100, 50, 8, 20000))

    cfgs = load_configs()
    cmed30 = json.load(open('wave2_C_mediator.json', encoding='utf-8'))

    if args.analyse:
        G, H, I, J = (_load('G_dissociation'), _load('H_mediator'),
                      _load('I_prop4'), _load('J_sharpness'))
    else:
        pool, n_elig = eligible_pool(list(cfgs.values()), n_cfg)
        OUT['pool'] = {'n_eligible_excluding_phase2': n_elig,
                       'n_sampled': len(pool)}
        t0 = time.time()
        G = block_G(pool, n_seed_g, n_steps)
        H = block_H(pool, n_seed_h)
        I = block_I(pool, n_seed_i, min(n_steps, 12000))
        J = block_J(cfgs, n_seed_h)
        print("\n  simulation total: %.0f s" % (time.time() - t0))

    rows = analyse_dissociation(G)
    fh, sh, ah, rg, lab = analyse_mediator(H, cmed30)
    analyse_prop4(I)
    analyse_sharpness(J)
    analyse_grid_cv()
    figures(rows, fh, sh, ah, rg, lab)

    with open('wave6_results.json', 'w', encoding='utf-8') as f:
        json.dump(OUT, f, indent=1, default=float)
    hdr("DONE")
    print("  wave6_results.json")
    for p in PLOTS:
        print("  " + p)


if __name__ == '__main__':
    main()
