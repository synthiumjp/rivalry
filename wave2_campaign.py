"""
wave2_campaign.py -- Paper 1 (GC-LCA) corrected analysis campaign.

Addresses register items:
  A1  regenerate canonical H4 numbers
  B5  random ADDITIVE boost control (the correct state-independent control)
  B19 predominance + alternation rate alongside DPR
  B22 monotonicity guard
  B23 non-colliding seed scheme
  B24 survivorship: episode retention logged on every condition
  B25 kappa sweep under the REGISTERED pulse protocol
  C6  Levelt Prop III as an inverted U in predominance
  C7  Prop II/IV magnitude comparison, not rank correlation
  C11 floor occupancy as the measured mediator
  C12 ceiling (x_max) occupancy
  C13 sharpness-parameterised rectifier

Usage:
    python wave2_campaign.py                # all blocks
    python wave2_campaign.py --blocks A,C    # selected blocks
    python wave2_campaign.py --quick         # reduced seeds, for a smoke test

Requires: phase2_dissociation_results.json (for the 30 Phase II config params)
Writes:   wave2_results.json  (+ wave2_<BLOCK>.json after each block)

Kernel is byte-for-byte equivalent to the original where it matters:
    x_i(t+1) = clip[ (1 - lam + G_i) x_i + S_i + b_i - beta x_j - alpha a_i + eta_i ]
    a_i(t+1) = (1 - gamma) a_i + kappa x_i
"""

import argparse
import json
import os
import sys
import time

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def wrapper(f):
            return f
        if args and callable(args[0]):
            return args[0]
        return wrapper


# =============================================================================
# CONSTANTS -- matched to the original pipeline
# =============================================================================

BURN_IN = 500
MARGIN = 0.05
X_MAX = 5.0
G_SAFETY = 0.95
SIGNAL_NEUTRAL = 0.50

DORSAL_MULTS = {'alpha': 1.5, 'beta': 0.6, 'kappa': 1.5}
VENTRAL_MULTS = {'alpha': 0.4, 'beta': 1.5, 'kappa': 0.4}

# Pulse protocol (registered)
N_STEPS_BASELINE = 5000
N_STEPS_PULSE = 500
N_STEPS_RESPONSE = 800
N_STEPS_TOTAL = N_STEPS_BASELINE + N_STEPS_RESPONSE
SWITCH_CRITERION = 50
LOOKBACK_WINDOW = 200

G_FRACTIONS = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90]
SIGNAL_LEVELS = [round(0.25 + 0.05 * i, 2) for i in range(11)]  # 0.25 .. 0.75

# The 10 configs used in the original block5 H4 analysis
H4_CONFIGS = [1, 4, 5, 6, 7, 13, 14, 16, 22, 27]
# The 5 strict-passing configs
STRICT_CONFIGS = [1, 6, 7, 13, 22]

FLOOR_TOL = 1e-12
CEIL_TOL = 1e-9


# =============================================================================
# B23 -- non-colliding seed scheme
# =============================================================================

def make_seed(block, config, regime, level, seed):
    """
    Injective over: block<20, config<1000, regime<10, level<200, seed<1000.
    Replaces the original additive scheme, whose offsets overlapped
    (dorsal G50 collided with ventral G0, etc.).
    """
    assert 0 <= block < 20
    assert 0 <= config < 1000
    assert 0 <= regime < 10
    assert 0 <= level < 200
    assert 0 <= seed < 1000
    v = ((((block * 1000 + config) * 10 + regime) * 200 + level) * 1000 + seed)
    return int(v % (2 ** 31 - 1))


# =============================================================================
# KERNELS
# =============================================================================

@njit(cache=True)
def _rectify(x, k_sharp, x_max):
    """
    k_sharp <= 0  -> hard half-wave rectifier max(0, x)
    k_sharp  > 0  -> (1/k) log(1 + exp(k x)); f(0) = log(2)/k -> 0 as k -> inf
    """
    if k_sharp <= 0.0:
        v = x if x > 0.0 else 0.0
    else:
        z = k_sharp * x
        if z > 30.0:
            v = x
        elif z < -30.0:
            v = 0.0
        else:
            v = np.log(1.0 + np.exp(z)) / k_sharp
    if v > x_max:
        v = x_max
    return v


@njit(cache=True)
def run_trace(lam, beta, alpha, sigma, gamma_adapt, kappa,
              signal_a, signal_b, g_a, g_b, boost_a, boost_b,
              n_steps, seed, x_max, k_sharp,
              rand_g_max, rand_boost_max):
    """
    Continuous-application run. Returns full traces.

    rand_g_max     > 0 -> channel A receives G ~ U(0, rand_g_max) each step
                          (STATE-DEPENDENT: enters as G * x_a)
    rand_boost_max > 0 -> channel A receives b ~ U(0, rand_boost_max) each step
                          (STATE-INDEPENDENT: enters additively)
    """
    np.random.seed(seed)
    ta = np.empty(n_steps, dtype=np.float64)
    tb = np.empty(n_steps, dtype=np.float64)
    x_a = 0.1
    x_b = 0.1
    a_a = 0.0
    a_b = 0.0

    for t in range(n_steps):
        ta[t] = x_a
        tb[t] = x_b

        eta_a = np.random.normal(0.0, sigma)
        eta_b = np.random.normal(0.0, sigma)

        ga = g_a
        if rand_g_max > 0.0:
            ga = np.random.uniform(0.0, rand_g_max)

        ba = boost_a
        if rand_boost_max > 0.0:
            ba = np.random.uniform(0.0, rand_boost_max)

        new_a = ((1.0 - lam + ga) * x_a + signal_a + ba
                 - beta * x_b - alpha * a_a + eta_a)
        new_b = ((1.0 - lam + g_b) * x_b + signal_b + boost_b
                 - beta * x_a - alpha * a_b + eta_b)

        x_a = _rectify(new_a, k_sharp, x_max)
        x_b = _rectify(new_b, k_sharp, x_max)

        a_a = (1.0 - gamma_adapt) * a_a + kappa * x_a
        a_b = (1.0 - gamma_adapt) * a_b + kappa * x_b

    return ta, tb


@njit(cache=True)
def extract_durations(trace_a, trace_b, burn_in, margin):
    """Verbatim from the original pipeline. Do not change."""
    n = len(trace_a) - burn_in
    if n <= 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    channels = np.empty(n, dtype=np.int32)
    durations = np.empty(n, dtype=np.int32)
    current_channel = -1
    current_duration = 0
    n_episodes = 0
    for t in range(burn_in, len(trace_a)):
        diff = trace_a[t] - trace_b[t]
        if diff > margin:
            dom = 0
        elif diff < -margin:
            dom = 1
        else:
            dom = -1
        if dom == current_channel and dom >= 0:
            current_duration += 1
        else:
            if current_channel >= 0 and current_duration >= 1:
                channels[n_episodes] = current_channel
                durations[n_episodes] = current_duration
                n_episodes += 1
            current_channel = dom
            current_duration = 1 if dom >= 0 else 0
    if current_channel >= 0 and current_duration >= 1:
        channels[n_episodes] = current_channel
        durations[n_episodes] = current_duration
        n_episodes += 1
    return channels[:n_episodes], durations[:n_episodes]


@njit(cache=True)
def dominance_summary(ta, tb, burn_in, margin):
    """
    B19: predominance and alternation rate over the FULL post-burn-in trace.
    Immune to the boundary-exclusion survivorship artifact.
    """
    n = len(ta) - burn_in
    t_a = 0
    t_b = 0
    n_switch = 0
    last = -1
    for t in range(burn_in, len(ta)):
        d = ta[t] - tb[t]
        if d > margin:
            dom = 0
        elif d < -margin:
            dom = 1
        else:
            dom = -1
        if dom == 0:
            t_a += 1
        elif dom == 1:
            t_b += 1
        if dom >= 0:
            if last >= 0 and dom != last:
                n_switch += 1
            last = dom
    tot = t_a + t_b
    pred = t_a / tot if tot > 0 else np.nan
    alt = n_switch / n if n > 0 else np.nan
    return pred, alt, t_a, t_b, n_switch


@njit(cache=True)
def occupancy_stats(ta, tb, burn_in, x_max):
    """
    C11/C12. Suppressed channel = the lower-activation channel at each step.
      floor_frac : fraction of steps with suppressed channel exactly at 0
      ceil_frac  : fraction of steps with either channel at x_max
      mean_resid : mean residual activation of the suppressed channel
    """
    n = len(ta) - burn_in
    if n <= 0:
        return np.nan, np.nan, np.nan
    n_floor = 0
    n_ceil = 0
    resid = 0.0
    for t in range(burn_in, len(ta)):
        lo = ta[t] if ta[t] < tb[t] else tb[t]
        hi = ta[t] if ta[t] > tb[t] else tb[t]
        if lo <= FLOOR_TOL:
            n_floor += 1
        if hi >= x_max - CEIL_TOL:
            n_ceil += 1
        resid += lo
    return n_floor / n, n_ceil / n, resid / n


@njit(cache=True)
def pulse_trial(lam, beta, alpha, sigma, gamma_adapt, kappa,
                signal_a, signal_b, g_value,
                n_steps_baseline, n_steps_total, n_steps_pulse,
                seed, x_max, lookback, switch_criterion, margin, k_sharp):
    """
    The REGISTERED transient protocol. This is what b14_kappa_role failed to
    use: baseline at G=0, identify the SUPPRESSED channel, pulse G onto it.

    Also returns floor/ceiling occupancy measured during the baseline period,
    so the C11 mediator is computed on the very trials that produce the
    switch outcome.
    """
    np.random.seed(seed)
    x_a = 0.1
    x_b = 0.1
    a_a = 0.0
    a_b = 0.0

    lookback_a = np.zeros(lookback)
    lookback_b = np.zeros(lookback)

    n_floor = 0
    n_ceil = 0
    resid = 0.0
    n_counted = 0

    # Phase 1: baseline, G = 0
    for t in range(n_steps_baseline):
        eta_a = np.random.normal(0.0, sigma)
        eta_b = np.random.normal(0.0, sigma)
        new_a = (1.0 - lam) * x_a + signal_a - beta * x_b - alpha * a_a + eta_a
        new_b = (1.0 - lam) * x_b + signal_b - beta * x_a - alpha * a_b + eta_b
        x_a = _rectify(new_a, k_sharp, x_max)
        x_b = _rectify(new_b, k_sharp, x_max)
        a_a = (1.0 - gamma_adapt) * a_a + kappa * x_a
        a_b = (1.0 - gamma_adapt) * a_b + kappa * x_b

        if t >= BURN_IN:
            lo = x_a if x_a < x_b else x_b
            hi = x_a if x_a > x_b else x_b
            if lo <= FLOOR_TOL:
                n_floor += 1
            if hi >= x_max - CEIL_TOL:
                n_ceil += 1
            resid += lo
            n_counted += 1

        if t >= n_steps_baseline - lookback:
            idx = t - (n_steps_baseline - lookback)
            lookback_a[idx] = x_a
            lookback_b[idx] = x_b

    mean_a = 0.0
    mean_b = 0.0
    for i in range(lookback):
        mean_a += lookback_a[i]
        mean_b += lookback_b[i]
    mean_a /= lookback
    mean_b /= lookback

    if mean_a > mean_b:
        target = 1   # B is suppressed -> pulse B
    else:
        target = 0

    g_a = g_value if target == 0 else 0.0
    g_b = g_value if target == 1 else 0.0

    switched = 0
    latency = -1
    consecutive = 0

    for t in range(n_steps_baseline, n_steps_total):
        step_in_response = t - n_steps_baseline
        if step_in_response >= n_steps_pulse:
            g_a = 0.0
            g_b = 0.0

        eta_a = np.random.normal(0.0, sigma)
        eta_b = np.random.normal(0.0, sigma)
        new_a = ((1.0 - lam + g_a) * x_a + signal_a
                 - beta * x_b - alpha * a_a + eta_a)
        new_b = ((1.0 - lam + g_b) * x_b + signal_b
                 - beta * x_a - alpha * a_b + eta_b)
        x_a = _rectify(new_a, k_sharp, x_max)
        x_b = _rectify(new_b, k_sharp, x_max)
        a_a = (1.0 - gamma_adapt) * a_a + kappa * x_a
        a_b = (1.0 - gamma_adapt) * a_b + kappa * x_b

        diff = x_a - x_b
        if target == 0 and diff > margin:
            consecutive += 1
        elif target == 1 and diff < -margin:
            consecutive += 1
        else:
            consecutive = 0

        if consecutive >= switch_criterion and switched == 0:
            switched = 1
            latency = step_in_response

    fl = n_floor / n_counted if n_counted > 0 else np.nan
    ce = n_ceil / n_counted if n_counted > 0 else np.nan
    rs = resid / n_counted if n_counted > 0 else np.nan
    return switched, latency, target, fl, ce, rs


# =============================================================================
# PYTHON-SIDE HELPERS
# =============================================================================

def seed_summary(ta, tb, x_max=X_MAX):
    """Everything we log for one continuous run."""
    ch, dur = extract_durations(ta, tb, BURN_IN, MARGIN)
    n_raw_a = int(np.sum(ch == 0))
    n_raw_b = int(np.sum(ch == 1))

    # registered boundary exclusion
    if len(ch) > 2:
        ch_x = ch[1:-1]
        dur_x = dur[1:-1]
    else:
        ch_x = np.empty(0, dtype=np.int32)
        dur_x = np.empty(0, dtype=np.int32)
    d_a = dur_x[ch_x == 0].astype(float)
    d_b = dur_x[ch_x == 1].astype(float)

    pred, alt, t_a, t_b, n_sw = dominance_summary(ta, tb, BURN_IN, MARGIN)
    fl, ce, rs = occupancy_stats(ta, tb, BURN_IN, x_max)

    # B24: winner-take-all flag
    is_wta = (n_raw_a == 0) or (n_raw_b == 0) or (n_raw_a + n_raw_b <= 1)

    return {
        'n_raw_a': n_raw_a, 'n_raw_b': n_raw_b,
        'n_kept_a': int(len(d_a)), 'n_kept_b': int(len(d_b)),
        'sum_dur_a': float(d_a.sum()), 'sum_dur_b': float(d_b.sum()),
        'predominance': float(pred), 'alternation_rate': float(alt),
        'time_a': int(t_a), 'time_b': int(t_b), 'n_switches': int(n_sw),
        'floor_frac': float(fl), 'ceil_frac': float(ce),
        'mean_residual': float(rs),
        'is_wta': bool(is_wta),
    }


def aggregate(seed_rows):
    """Pool per-seed summaries. Reports BOTH DPR conventions and retention."""
    n = len(seed_rows)
    if n == 0:
        return {}
    tot_a = sum(r['sum_dur_a'] for r in seed_rows)
    tot_b = sum(r['sum_dur_b'] for r in seed_rows)
    n_a = sum(r['n_kept_a'] for r in seed_rows)
    n_b = sum(r['n_kept_b'] for r in seed_rows)
    raw_a = sum(r['n_raw_a'] for r in seed_rows)
    raw_b = sum(r['n_raw_b'] for r in seed_rows)

    preds = [r['predominance'] for r in seed_rows
             if not np.isnan(r['predominance'])]
    alts = [r['alternation_rate'] for r in seed_rows
            if not np.isnan(r['alternation_rate'])]

    return {
        'n_seeds': n,
        'n_wta_seeds': sum(1 for r in seed_rows if r['is_wta']),
        'wta_rate': sum(1 for r in seed_rows if r['is_wta']) / n,
        'mean_dur_a': tot_a / n_a if n_a else None,
        'mean_dur_b': tot_b / n_b if n_b else None,
        'n_kept_a': n_a, 'n_kept_b': n_b,
        'n_raw_a': raw_a, 'n_raw_b': raw_b,
        'retention_a': n_a / raw_a if raw_a else None,
        'retention_b': n_b / raw_b if raw_b else None,
        'predominance_mean': float(np.mean(preds)) if preds else None,
        'predominance_sd': float(np.std(preds, ddof=1)) if len(preds) > 1 else None,
        'alternation_rate_mean': float(np.mean(alts)) if alts else None,
        'alternation_rate_sd': float(np.std(alts, ddof=1)) if len(alts) > 1 else None,
        'floor_frac_mean': float(np.mean([r['floor_frac'] for r in seed_rows])),
        'ceil_frac_mean': float(np.mean([r['ceil_frac'] for r in seed_rows])),
        'mean_residual_mean': float(np.mean([r['mean_residual'] for r in seed_rows])),
    }


def load_configs():
    fn = "phase2_dissociation_results.json"
    if not os.path.exists(fn):
        sys.exit("ERROR: %s not found. Run from C:\\crewther." % fn)
    with open(fn, "r", encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for cr in d['config_results']:
        out[cr['config_idx']] = cr['base_params']
    return out


def save(tag, obj):
    fn = "wave2_%s.json" % tag
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)
    print("    -> %s" % fn)


# =============================================================================
# BLOCK A -- H4 canonical (A1, B19, B24, C12)
# =============================================================================

def block_A(cfgs, n_seeds, n_steps):
    print("\n" + "=" * 70)
    print("BLOCK A: H4 canonical regeneration")
    print("=" * 70)
    out = []
    for ci in H4_CONFIGS:
        p = cfgs[ci]
        lam, beta = p['lambda'], p['beta']
        alpha, sigma = p['alpha'], p['sigma']
        gam, kap = p['gamma'], p['kappa']
        g_value = min(0.5 * lam, G_SAFETY * lam)

        # calibrate boost on 20 baseline seeds (as originally specified)
        acts = []
        for s in range(20):
            ta, tb = run_trace(lam, beta, alpha, sigma, gam, kap,
                               SIGNAL_NEUTRAL, SIGNAL_NEUTRAL,
                               0.0, 0.0, 0.0, 0.0,
                               n_steps, make_seed(0, ci, 0, 0, s),
                               X_MAX, -1.0, 0.0, 0.0)
            acts.append(float(np.mean(ta[BURN_IN:])))
        mean_x = float(np.mean(acts))
        boost_amount = g_value * mean_x

        conds = {
            'baseline': dict(g_a=0.0, boost_a=0.0),
            'gclca':    dict(g_a=g_value, boost_a=0.0),
            'boost':    dict(g_a=0.0, boost_a=boost_amount),
        }

        rec = {'config_idx': ci, 'params': p, 'g_value': g_value,
               'mean_activation': mean_x, 'boost_amount': boost_amount,
               'conditions': {}}

        for li, (cname, cc) in enumerate(conds.items()):
            rows = []
            for s in range(n_seeds):
                ta, tb = run_trace(lam, beta, alpha, sigma, gam, kap,
                                   SIGNAL_NEUTRAL, SIGNAL_NEUTRAL,
                                   cc['g_a'], 0.0, cc['boost_a'], 0.0,
                                   n_steps, make_seed(1, ci, 0, li + 1, s),
                                   X_MAX, -1.0, 0.0, 0.0)
                rows.append(seed_summary(ta, tb))
            rec['conditions'][cname] = {
                'summary': aggregate(rows),
                'per_seed': rows,
            }
        out.append(rec)
        a = rec['conditions']
        print("  cfg %2d  pred base/gc/boost = %.3f / %.3f / %.3f   "
              "wta gc = %.0f%%  retain_b gc = %.0f%%"
              % (ci,
                 a['baseline']['summary']['predominance_mean'],
                 a['gclca']['summary']['predominance_mean'],
                 a['boost']['summary']['predominance_mean'],
                 100 * a['gclca']['summary']['wta_rate'],
                 100 * (a['gclca']['summary']['retention_b'] or 0)))
    save("A_h4", out)
    return out


# =============================================================================
# BLOCK B -- Levelt sweep retaining per-level data (C6, C7)
# =============================================================================

def block_B(cfgs, n_seeds, n_steps):
    print("\n" + "=" * 70)
    print("BLOCK B: Levelt sweep with per-signal-level retention")
    print("=" * 70)
    out = []
    for ci in sorted(cfgs.keys()):
        p = cfgs[ci]
        lam, beta = p['lambda'], p['beta']
        alpha, sigma = p['alpha'], p['sigma']
        gam, kap = p['gamma'], p['kappa']
        levels = []
        for li, sig_a in enumerate(SIGNAL_LEVELS):
            rows = []
            for s in range(n_seeds):
                ta, tb = run_trace(lam, beta, alpha, sigma, gam, kap,
                                   sig_a, SIGNAL_NEUTRAL,
                                   0.0, 0.0, 0.0, 0.0,
                                   n_steps, make_seed(2, ci, 0, li, s),
                                   X_MAX, -1.0, 0.0, 0.0)
                rows.append(seed_summary(ta, tb))
            agg = aggregate(rows)
            agg['signal_a'] = sig_a
            levels.append(agg)
        out.append({'config_idx': ci, 'params': p, 'levels': levels})
        pk = max(range(len(levels)),
                 key=lambda i: levels[i]['alternation_rate_mean'] or -1)
        print("  cfg %2d  alt-rate peak at S_A = %.2f  (pred there = %.3f)"
              % (ci, SIGNAL_LEVELS[pk], levels[pk]['predominance_mean']))
    save("B_levelt", out)
    return out


# =============================================================================
# BLOCK C -- floor-occupancy mediator + clean pulse protocol (C11, C12, B22, B23)
# =============================================================================

def block_C(cfgs, n_seeds):
    print("\n" + "=" * 70)
    print("BLOCK C: pulse protocol + floor occupancy mediator")
    print("=" * 70)
    out = []
    for ci in sorted(cfgs.keys()):
        p = cfgs[ci]
        lam, sigma, gam = p['lambda'], p['sigma'], p['gamma']
        rec = {'config_idx': ci, 'params': p, 'regimes': {}}
        for ri, (rname, m) in enumerate(
                [('dorsal', DORSAL_MULTS), ('ventral', VENTRAL_MULTS)]):
            a_r = p['alpha'] * m['alpha']
            b_r = p['beta'] * m['beta']
            k_r = p['kappa'] * m['kappa']
            g_levels = {}
            for li, gf in enumerate(G_FRACTIONS):
                g_value = min(gf * lam, G_SAFETY * lam)
                sw, lat, fl, ce, rs = [], [], [], [], []
                for s in range(n_seeds):
                    r = pulse_trial(lam, b_r, a_r, sigma, gam, k_r,
                                    SIGNAL_NEUTRAL, SIGNAL_NEUTRAL, g_value,
                                    N_STEPS_BASELINE, N_STEPS_TOTAL,
                                    N_STEPS_PULSE,
                                    make_seed(3, ci, ri, li, s),
                                    X_MAX, LOOKBACK_WINDOW,
                                    SWITCH_CRITERION, MARGIN, -1.0)
                    sw.append(r[0])
                    if r[1] >= 0:
                        lat.append(r[1])
                    fl.append(r[3])
                    ce.append(r[4])
                    rs.append(r[5])
                g_levels["G%d" % int(gf * 100)] = {
                    'g_fraction': gf, 'g_value': g_value,
                    'switch_rate': float(np.mean(sw)),
                    'n_switched': int(np.sum(sw)), 'n_seeds': len(sw),
                    'median_latency': float(np.median(lat)) if lat else None,
                    'floor_frac_mean': float(np.mean(fl)),
                    'ceil_frac_mean': float(np.mean(ce)),
                    'mean_residual_mean': float(np.mean(rs)),
                }
            # B22: monotonicity guard -- requires a real increase
            rates = [g_levels["G%d" % int(gf * 100)]['switch_rate']
                     for gf in G_FRACTIONS if gf > 0]
            nondecr = all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))
            rec['regimes'][rname] = {
                'regime_params': {'alpha': a_r, 'beta': b_r, 'kappa': k_r},
                'alpha_over_lambda': a_r / lam,
                'g_levels': g_levels,
                'dose_response_nondecreasing': bool(nondecr),
                'dose_response_monotonic_strict': bool(
                    nondecr and max(rates) > 0 and max(rates) > min(rates)),
                'max_switch_rate': float(max(rates)),
                'baseline_floor_frac': g_levels['G0']['floor_frac_mean'],
                'baseline_ceil_frac': g_levels['G0']['ceil_frac_mean'],
                'baseline_mean_residual': g_levels['G0']['mean_residual_mean'],
            }
        out.append(rec)
        d = rec['regimes']['dorsal']
        v = rec['regimes']['ventral']
        print("  cfg %2d  dorsal max %.2f (floor %.3f) | "
              "ventral max %.2f (floor %.3f)"
              % (ci, d['max_switch_rate'], d['baseline_floor_frac'],
                 v['max_switch_rate'], v['baseline_floor_frac']))
    save("C_mediator", out)
    return out


# =============================================================================
# BLOCK D -- kappa sweep under the REGISTERED protocol (B25)
# =============================================================================

def block_D(cfgs, n_seeds, n_steps):
    print("\n" + "=" * 70)
    print("BLOCK D: kappa sweep, REGISTERED pulse protocol (B25 fix)")
    print("=" * 70)
    p = cfgs[7]
    lam, beta, alpha, sigma, gam = (p['lambda'], p['beta'], p['alpha'],
                                    p['sigma'], p['gamma'])
    print("  Config 7 base: lam=%.3f beta=%.3f alpha=%.3f sigma=%.3f gamma=%.3f"
          % (lam, beta, alpha, sigma, gam))
    kappas = [round(0.01 + 0.01 * i, 4) for i in range(20)]
    out = []
    for li, kap in enumerate(kappas):
        # CV at baseline (this part of the original was valid)
        cvs = []
        for s in range(min(20, n_seeds)):
            ta, tb = run_trace(lam, beta, alpha, sigma, gam, kap,
                               SIGNAL_NEUTRAL, SIGNAL_NEUTRAL,
                               0.0, 0.0, 0.0, 0.0,
                               n_steps, make_seed(4, 7, 0, li, s),
                               X_MAX, -1.0, 0.0, 0.0)
            ch, dur = extract_durations(ta, tb, BURN_IN, MARGIN)
            if len(dur) > 4:
                d = dur[1:-1].astype(float)
                if len(d) > 2 and d.mean() > 0:
                    cvs.append(float(d.std(ddof=1) / d.mean()))
        rec = {'kappa': kap,
               'cv_mean': float(np.mean(cvs)) if cvs else None}
        g_value = min(0.5 * lam, G_SAFETY * lam)
        for ri, (rname, m) in enumerate(
                [('dorsal', DORSAL_MULTS), ('ventral', VENTRAL_MULTS)]):
            a_r = alpha * m['alpha']
            b_r = beta * m['beta']
            k_r = kap * m['kappa']
            sw, fl = [], []
            for s in range(n_seeds):
                r = pulse_trial(lam, b_r, a_r, sigma, gam, k_r,
                                SIGNAL_NEUTRAL, SIGNAL_NEUTRAL, g_value,
                                N_STEPS_BASELINE, N_STEPS_TOTAL, N_STEPS_PULSE,
                                make_seed(5, 7, ri, li, s),
                                X_MAX, LOOKBACK_WINDOW,
                                SWITCH_CRITERION, MARGIN, -1.0)
                sw.append(r[0])
                fl.append(r[3])
            rec[rname + '_rate'] = float(np.mean(sw))
            rec[rname + '_floor_frac'] = float(np.mean(fl))
        rec['dissociation'] = rec['dorsal_rate'] - rec['ventral_rate']
        out.append(rec)
        print("  k=%.3f  CV=%s  dorsal=%.2f (floor %.3f)  "
              "ventral=%.2f (floor %.3f)  dissoc=%+.2f"
              % (kap,
                 ("%.3f" % rec['cv_mean']) if rec['cv_mean'] else " N/A ",
                 rec['dorsal_rate'], rec['dorsal_floor_frac'],
                 rec['ventral_rate'], rec['ventral_floor_frac'],
                 rec['dissociation']))
    save("D_kappa", out)
    return out


# =============================================================================
# BLOCK E -- random ADDITIVE boost control (B5)
# =============================================================================

def block_E(cfgs, n_seeds, n_steps):
    print("\n" + "=" * 70)
    print("BLOCK E: random controls -- state-dependent vs state-independent")
    print("=" * 70)
    out = []
    for ci in STRICT_CONFIGS:
        p = cfgs[ci]
        lam, beta = p['lambda'], p['beta']
        alpha, sigma = p['alpha'], p['sigma']
        gam, kap = p['gamma'], p['kappa']

        acts = []
        for s in range(20):
            ta, tb = run_trace(lam, beta, alpha, sigma, gam, kap,
                               SIGNAL_NEUTRAL, SIGNAL_NEUTRAL,
                               0.0, 0.0, 0.0, 0.0,
                               n_steps, make_seed(6, ci, 0, 0, s),
                               X_MAX, -1.0, 0.0, 0.0)
            acts.append(float(np.mean(ta[BURN_IN:])))
        mean_x = float(np.mean(acts))

        g_const = min(0.5 * lam, G_SAFETY * lam)   # mean of U(0, lam) is lam/2
        boost_const = g_const * mean_x
        rand_g_max = lam                            # mean = lam/2 = g_const
        rand_boost_max = 2.0 * boost_const          # mean = boost_const

        conds = {
            'baseline':     dict(g=0.0, b=0.0, rg=0.0, rb=0.0),
            'gclca_const':  dict(g=g_const, b=0.0, rg=0.0, rb=0.0),
            'gclca_random': dict(g=0.0, b=0.0, rg=rand_g_max, rb=0.0),
            'boost_const':  dict(g=0.0, b=boost_const, rg=0.0, rb=0.0),
            'boost_random': dict(g=0.0, b=0.0, rg=0.0, rb=rand_boost_max),
        }
        rec = {'config_idx': ci, 'params': p, 'mean_activation': mean_x,
               'g_const': g_const, 'boost_const': boost_const,
               'conditions': {}}
        for li, (cname, cc) in enumerate(conds.items()):
            rows = []
            for s in range(n_seeds):
                ta, tb = run_trace(lam, beta, alpha, sigma, gam, kap,
                                   SIGNAL_NEUTRAL, SIGNAL_NEUTRAL,
                                   cc['g'], 0.0, cc['b'], 0.0,
                                   n_steps, make_seed(7, ci, 0, li, s),
                                   X_MAX, -1.0, cc['rg'], cc['rb'])
                rows.append(seed_summary(ta, tb))
            rec['conditions'][cname] = {'summary': aggregate(rows)}
        out.append(rec)
        s = rec['conditions']
        print("  cfg %2d  predominance: gc_const %.3f  gc_rand %.3f  "
              "bo_const %.3f  bo_rand %.3f"
              % (ci,
                 s['gclca_const']['summary']['predominance_mean'],
                 s['gclca_random']['summary']['predominance_mean'],
                 s['boost_const']['summary']['predominance_mean'],
                 s['boost_random']['summary']['predominance_mean']))
    save("E_random", out)
    return out


# =============================================================================
# BLOCK F -- rectifier sharpness sweep (C13)
# =============================================================================

def block_F(cfgs, n_seeds):
    print("\n" + "=" * 70)
    print("BLOCK F: rectifier sharpness sweep (C13)")
    print("=" * 70)
    k_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, -1.0]  # -1 = hard
    out = []
    for ci in STRICT_CONFIGS:
        p = cfgs[ci]
        lam, sigma, gam = p['lambda'], p['sigma'], p['gamma']
        rec = {'config_idx': ci, 'params': p, 'sweep': []}
        for li, k in enumerate(k_values):
            entry = {'k_sharp': k,
                     'f_at_zero': (float(np.log(2.0) / k) if k > 0 else 0.0),
                     'label': 'hard' if k < 0 else ('k=%g' % k)}
            g_value = min(0.9 * lam, G_SAFETY * lam)
            for ri, (rname, m) in enumerate(
                    [('dorsal', DORSAL_MULTS), ('ventral', VENTRAL_MULTS)]):
                a_r = p['alpha'] * m['alpha']
                b_r = p['beta'] * m['beta']
                k_r = p['kappa'] * m['kappa']
                sw, fl = [], []
                for s in range(n_seeds):
                    r = pulse_trial(lam, b_r, a_r, sigma, gam, k_r,
                                    SIGNAL_NEUTRAL, SIGNAL_NEUTRAL, g_value,
                                    N_STEPS_BASELINE, N_STEPS_TOTAL,
                                    N_STEPS_PULSE,
                                    make_seed(8, ci, ri, li, s),
                                    X_MAX, LOOKBACK_WINDOW,
                                    SWITCH_CRITERION, MARGIN, k)
                    sw.append(r[0])
                    fl.append(r[3])
                entry[rname + '_rate'] = float(np.mean(sw))
                entry[rname + '_floor_frac'] = float(np.mean(fl))
            entry['dissociation'] = entry['dorsal_rate'] - entry['ventral_rate']
            rec['sweep'].append(entry)
            print("  cfg %2d  %-7s f(0)=%.4f  dorsal=%.2f  ventral=%.2f  "
                  "floor_v=%.3f  dissoc=%+.2f"
                  % (ci, entry['label'], entry['f_at_zero'],
                     entry['dorsal_rate'], entry['ventral_rate'],
                     entry['ventral_floor_frac'], entry['dissociation']))
        out.append(rec)
    save("F_sharpness", out)
    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--blocks', default='A,B,C,D,E,F')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()

    if not HAVE_NUMBA:
        print("WARNING: numba not available. This will be very slow.")

    blocks = [b.strip().upper() for b in args.blocks.split(',') if b.strip()]

    if args.quick:
        n_h4, n_lev, n_pulse, n_steps = 10, 3, 10, 5000
        print("QUICK MODE -- smoke test only, results not for use")
    else:
        n_h4, n_lev, n_pulse, n_steps = 100, 8, 50, 20000

    cfgs = load_configs()
    print("Loaded %d Phase II configs" % len(cfgs))

    results = {
        'metadata': {
            'analysis': 'Paper 1 Wave 2 corrected campaign',
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'quick': args.quick,
            'blocks': blocks,
            'seed_scheme': 'make_seed(block, config, regime, level, seed) '
                           '-- injective, replaces colliding additive scheme',
            'notes': [
                'predominance and alternation_rate computed on FULL '
                'post-burn-in trace (no boundary exclusion)',
                'duration statistics use registered boundary exclusion, '
                'reported with retention rates',
                'kappa sweep uses the REGISTERED pulse protocol',
            ],
        }
    }

    t0 = time.time()
    if 'A' in blocks:
        results['A_h4'] = block_A(cfgs, n_h4, n_steps)
    if 'B' in blocks:
        results['B_levelt'] = block_B(cfgs, n_lev, min(n_steps, 12000))
    if 'C' in blocks:
        results['C_mediator'] = block_C(cfgs, n_pulse)
    if 'D' in blocks:
        results['D_kappa'] = block_D(cfgs, n_pulse, n_steps)
    if 'E' in blocks:
        results['E_random'] = block_E(cfgs, n_h4, n_steps)
    if 'F' in blocks:
        results['F_sharpness'] = block_F(cfgs, n_pulse)

    results['metadata']['total_time_seconds'] = round(time.time() - t0, 1)

    with open("wave2_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=float)

    print("\n" + "=" * 70)
    print("DONE in %.1f s -> wave2_results.json"
          % results['metadata']['total_time_seconds'])
    print("=" * 70)


if __name__ == '__main__':
    main()
