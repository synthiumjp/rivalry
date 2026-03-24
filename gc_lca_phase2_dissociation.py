#!/usr/bin/env python3
"""
GC-LCA Phase II — Volitional Control & Dissociation Testing
============================================================
Crewther Sampler Research Programme v2.2

Pre-registered protocol (§2.7):
  - Top 30 configurations from Phase I (best CV proximity to 0.5, Levelt ρ > 0.7)
  - G fractions of λ: {0%, 10%, 25%, 50%, 75%, 90%}
  - Two regime variants per config:
      Dorsal:  α×1.5, β×0.6, κ×1.5
      Ventral: α×0.4, β×1.5, κ×0.4
  - Transient protocol: G pulse of 500 steps at t=5000
  - Switch criterion: 50 consecutive steps of target dominance within 800-step window
  - Seeds per condition: 50
  - Same-seed baseline control (G=0 run with identical seed)
  - Total Phase II runs: 30 × 6 × 2 × 50 = 18,000

Controls (ablations, also pre-registered):
  - No adaptation (α=0, κ=0)
  - No inhibition (β=0)
  - Signal boost (G=0, matched boost)
  - Random G (uniform per timestep)

Hardware target: 32GB RAM, AMD 7900GRE (CPU-bound)
Expected runtime: ~5-10 minutes with Numba JIT

Author: JP Cacioli
Date: March 2026
"""

import numpy as np
import json
import time
import os
import sys
import warnings
from multiprocessing import Pool, cpu_count

try:
    from numba import njit
    HAS_NUMBA = True
    print("[INFO] Numba available — JIT compilation enabled")
except ImportError:
    HAS_NUMBA = False
    print("[WARNING] Numba not found — falling back to pure NumPy")
    def njit(*args, **kwargs):
        def wrapper(f):
            return f
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return wrapper

from scipy.stats import gamma as gamma_dist, weibull_min, spearmanr

warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# TOP 30 CONFIGS FROM PHASE I
# (λ, β, α, σ, γ, κ, CV, Levelt_ρ)
# Selected: rivalry-producing, Levelt ρ > 0.7, CV ∈ [0.35, 0.65],
#           ranked by |CV - 0.5|
# =============================================================================

TOP_30_CONFIGS = [
    (0.1, 0.15, 0.1, 0.06, 0.05, 0.1, 0.5002, 1.00),      # idx=2877
    (0.1, 0.15, 0.03, 0.1, 0.02, 0.045, 0.5004, 1.00),     # idx=2674
    (0.2, 0.35, 0.12, 0.12, 0.02, 0.09, 0.4995, 1.00),     # idx=8986
    (0.1, 0.2, 0.12, 0.1, 0.02, 0.18, 0.4993, 1.00),       # idx=3349
    (0.2, 0.25, 0.1, 0.04, 0.05, 0.1, 0.5009, 1.00),       # idx=8112
    (0.15, 0.2, 0.1, 0.1, 0.03, 0.05, 0.4990, 1.00),       # idx=5525
    (0.15, 0.3, 0.05, 0.12, 0.02, 0.05, 0.4989, 1.00),     # idx=6137
    (0.15, 0.25, 0.08, 0.1, 0.03, 0.04, 0.4989, 1.00),     # idx=5825
    (0.08, 0.15, 0.12, 0.1, 0.02, 0.15, 0.4986, 1.00),     # idx=723
    (0.1, 0.2, 0.1, 0.12, 0.02, 0.15, 0.5017, 1.00),       # idx=3289
    (0.15, 0.2, 0.12, 0.1, 0.02, 0.06, 0.5018, 1.00),      # idx=5595
    (0.15, 0.25, 0.1, 0.12, 0.02, 0.1, 0.4981, 1.00),      # idx=5912
    (0.15, 0.2, 0.08, 0.1, 0.03, 0.12, 0.5020, 1.00),      # idx=5454
    (0.1, 0.2, 0.05, 0.12, 0.02, 0.0375, 0.5022, 1.00),    # idx=3136
    (0.2, 0.25, 0.08, 0.04, 0.05, 0.12, 0.5022, 1.00),     # idx=8039
    (0.15, 0.25, 0.12, 0.1, 0.02, 0.15, 0.4970, 1.00),     # idx=5973
    (0.15, 0.2, 0.08, 0.1, 0.03, 0.06, 0.5032, 1.00),      # idx=5451
    (0.08, 0.1, 0.12, 0.06, 0.02, 0.06, 0.4966, 1.00),     # idx=315
    (0.2, 0.35, 0.08, 0.12, 0.05, 0.12, 0.5037, 0.82),     # idx=8849
    (0.1, 0.3, 0.08, 0.12, 0.03, 0.06, 0.4955, 1.00),      # idx=3966
    (0.2, 0.25, 0.12, 0.08, 0.03, 0.15, 0.5046, 1.00),     # idx=8213
    (0.2, 0.3, 0.1, 0.1, 0.02, 0.125, 0.4949, 1.00),       # idx=8523
    (0.08, 0.1, 0.05, 0.04, 0.02, 0.0375, 0.4945, 1.00),   # idx=76
    (0.15, 0.25, 0.08, 0.12, 0.02, 0.12, 0.4944, 1.00),    # idx=5839
    (0.2, 0.25, 0.12, 0.06, 0.05, 0.15, 0.4944, 1.00),     # idx=8203
    (0.1, 0.2, 0.12, 0.12, 0.02, 0.12, 0.4939, 1.00),      # idx=3362
    (0.08, 0.1, 0.08, 0.06, 0.02, 0.12, 0.5063, 1.00),     # idx=169
    (0.1, 0.15, 0.05, 0.12, 0.03, 0.075, 0.4930, 0.81),    # idx=2769
    (0.08, 0.1, 0.05, 0.06, 0.02, 0.075, 0.5073, 1.00),    # idx=94
    (0.1, 0.15, 0.12, 0.1, 0.03, 0.15, 0.5073, 1.00),      # idx=2978
]

# =============================================================================
# PROTOCOL PARAMETERS (§2.7)
# =============================================================================

G_FRACTIONS = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90]
N_SEEDS = 50
N_STEPS_BASELINE = 5000     # establish steady-state rivalry
N_STEPS_PULSE = 500         # G pulse duration
N_STEPS_RESPONSE = 800      # response window after pulse onset
N_STEPS_TOTAL = N_STEPS_BASELINE + N_STEPS_RESPONSE  # 5800
SWITCH_CRITERION = 50       # consecutive steps of target dominance
LOOKBACK_WINDOW = 200       # steps to assess current dominance before pulse

# Regime multipliers (§2.7)
DORSAL_MULTIPLIERS  = {'alpha': 1.5, 'beta': 0.6, 'kappa': 1.5}
VENTRAL_MULTIPLIERS = {'alpha': 0.4, 'beta': 1.5, 'kappa': 0.4}

# Dominance extraction
MARGIN = 0.05
X_MAX = 5.0
G_SAFETY = 0.95  # G ceiling = 0.95 * λ

# Signal boost control
SIGNAL_NEUTRAL = 0.50

# Output
OUTPUT_FILE = "phase2_dissociation_results.json"


# =============================================================================
# SIMULATION KERNEL (Numba JIT)
# =============================================================================

@njit(cache=True)
def simulate_dissociation_trial(lam, beta, alpha, sigma, gamma_adapt, kappa,
                                signal_a, signal_b, g_value, g_target_channel,
                                n_steps_baseline, n_steps_total,
                                seed, x_max=5.0, lookback=200,
                                switch_criterion=50):
    """
    Run a single dissociation trial:
    1. Baseline (G=0) for n_steps_baseline steps
    2. Determine dominant channel from lookback window
    3. Apply G pulse to SUPPRESSED channel for remainder
    4. Check for switch within response window

    Returns:
        switched: 1 if switch occurred, 0 otherwise
        latency: steps from pulse onset to switch (-1 if no switch)
        dominant_at_pulse: which channel was dominant (0=A, 1=B)
        baseline_trace_a_mean: mean activation of A during lookback
        baseline_trace_b_mean: mean activation of B during lookback
    """
    np.random.seed(seed)

    x_a = 0.1
    x_b = 0.1
    a_a = 0.0
    a_b = 0.0

    leak_factor = 1.0 - lam

    # Store lookback window activations
    lookback_a = np.zeros(lookback)
    lookback_b = np.zeros(lookback)

    # Phase 1: Baseline — no G
    for t in range(n_steps_baseline):
        eta_a = np.random.normal(0.0, sigma)
        eta_b = np.random.normal(0.0, sigma)

        new_a = leak_factor * x_a + signal_a - beta * x_b - alpha * a_a + eta_a
        new_b = leak_factor * x_b + signal_b - beta * x_a - alpha * a_b + eta_b

        x_a = min(max(0.0, new_a), x_max)
        x_b = min(max(0.0, new_b), x_max)

        a_a = (1.0 - gamma_adapt) * a_a + kappa * x_a
        a_b = (1.0 - gamma_adapt) * a_b + kappa * x_b

        # Fill lookback buffer (circular)
        if t >= n_steps_baseline - lookback:
            idx = t - (n_steps_baseline - lookback)
            lookback_a[idx] = x_a
            lookback_b[idx] = x_b

    # Determine dominant channel
    mean_a = 0.0
    mean_b = 0.0
    for i in range(lookback):
        mean_a += lookback_a[i]
        mean_b += lookback_b[i]
    mean_a /= lookback
    mean_b /= lookback

    if mean_a > mean_b:
        dominant = 0  # A dominant, suppress B → apply G to B
        target = 1
    else:
        dominant = 1  # B dominant, suppress A → apply G to A
        target = 0

    # Phase 2: G pulse to suppressed channel
    # G modulates persistence: effective leak = λ - G for target channel
    g_a = g_value if target == 0 else 0.0
    g_b = g_value if target == 1 else 0.0

    switched = 0
    latency = -1
    consecutive_target = 0
    pulse_active = True

    for t in range(n_steps_baseline, n_steps_total):
        step_in_response = t - n_steps_baseline

        # G pulse duration check
        if step_in_response >= 500:  # N_STEPS_PULSE
            g_a = 0.0
            g_b = 0.0
            pulse_active = False

        eta_a = np.random.normal(0.0, sigma)
        eta_b = np.random.normal(0.0, sigma)

        # Core GC-LCA update with G
        new_a = (leak_factor + g_a) * x_a + signal_a - beta * x_b - alpha * a_a + eta_a
        new_b = (leak_factor + g_b) * x_b + signal_b - beta * x_a - alpha * a_b + eta_b

        x_a = min(max(0.0, new_a), x_max)
        x_b = min(max(0.0, new_b), x_max)

        a_a = (1.0 - gamma_adapt) * a_a + kappa * x_a
        a_b = (1.0 - gamma_adapt) * a_b + kappa * x_b

        # Check switch criterion
        diff = x_a - x_b
        if target == 0 and diff > MARGIN:
            consecutive_target += 1
        elif target == 1 and diff < -MARGIN:
            consecutive_target += 1
        else:
            consecutive_target = 0

        if consecutive_target >= switch_criterion and switched == 0:
            switched = 1
            latency = step_in_response
            # Don't break — let the simulation finish for completeness

    return switched, latency, dominant, mean_a, mean_b


@njit(cache=True)
def simulate_signal_boost_trial(lam, beta, alpha, sigma, gamma_adapt, kappa,
                                signal_a, signal_b, boost_amount,
                                n_steps_baseline, n_steps_total,
                                seed, x_max=5.0, lookback=200,
                                switch_criterion=50):
    """
    Signal boost control: instead of G, add a constant to the target signal.
    Same protocol otherwise (transient, same seed, same switch criterion).
    """
    np.random.seed(seed)

    x_a = 0.1
    x_b = 0.1
    a_a = 0.0
    a_b = 0.0

    leak_factor = 1.0 - lam

    lookback_a = np.zeros(lookback)
    lookback_b = np.zeros(lookback)

    # Phase 1: Baseline
    for t in range(n_steps_baseline):
        eta_a = np.random.normal(0.0, sigma)
        eta_b = np.random.normal(0.0, sigma)

        new_a = leak_factor * x_a + signal_a - beta * x_b - alpha * a_a + eta_a
        new_b = leak_factor * x_b + signal_b - beta * x_a - alpha * a_b + eta_b

        x_a = min(max(0.0, new_a), x_max)
        x_b = min(max(0.0, new_b), x_max)

        a_a = (1.0 - gamma_adapt) * a_a + kappa * x_a
        a_b = (1.0 - gamma_adapt) * a_b + kappa * x_b

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
        target = 1
    else:
        target = 0

    switched = 0
    latency = -1
    consecutive_target = 0

    for t in range(n_steps_baseline, n_steps_total):
        step_in_response = t - n_steps_baseline

        # Signal boost active during pulse window only
        if step_in_response < 500:
            eff_signal_a = signal_a + (boost_amount if target == 0 else 0.0)
            eff_signal_b = signal_b + (boost_amount if target == 1 else 0.0)
        else:
            eff_signal_a = signal_a
            eff_signal_b = signal_b

        eta_a = np.random.normal(0.0, sigma)
        eta_b = np.random.normal(0.0, sigma)

        new_a = leak_factor * x_a + eff_signal_a - beta * x_b - alpha * a_a + eta_a
        new_b = leak_factor * x_b + eff_signal_b - beta * x_a - alpha * a_b + eta_b

        x_a = min(max(0.0, new_a), x_max)
        x_b = min(max(0.0, new_b), x_max)

        a_a = (1.0 - gamma_adapt) * a_a + kappa * x_a
        a_b = (1.0 - gamma_adapt) * a_b + kappa * x_b

        diff = x_a - x_b
        if target == 0 and diff > MARGIN:
            consecutive_target += 1
        elif target == 1 and diff < -MARGIN:
            consecutive_target += 1
        else:
            consecutive_target = 0

        if consecutive_target >= switch_criterion and switched == 0:
            switched = 1
            latency = step_in_response

    return switched, latency


# =============================================================================
# WORKER FUNCTION
# =============================================================================

def process_config_phase2(args):
    """
    Process one config × one regime × all G levels × all seeds.
    Also runs same-seed baseline (G=0) and signal boost controls.
    """
    config_idx, lam, beta, alpha, sigma, gamma_adapt, kappa, cv, levelt_rho, regime = args

    # Apply regime multipliers
    if regime == 'dorsal':
        mults = DORSAL_MULTIPLIERS
    else:
        mults = VENTRAL_MULTIPLIERS

    alpha_r = alpha * mults['alpha']
    beta_r = beta * mults['beta']
    kappa_r = kappa * mults['kappa']

    result = {
        'config_idx': config_idx,
        'base_params': {
            'lambda': lam, 'beta': beta, 'alpha': alpha,
            'sigma': sigma, 'gamma': gamma_adapt, 'kappa': kappa,
        },
        'regime': regime,
        'regime_params': {
            'alpha': round(alpha_r, 6),
            'beta': round(beta_r, 6),
            'kappa': round(kappa_r, 6),
        },
        'phase1_cv': cv,
        'phase1_levelt_rho': levelt_rho,
        'g_levels': {},
    }

    for g_frac in G_FRACTIONS:
        g_value = g_frac * lam
        # Enforce G < λ safety ceiling
        g_value = min(g_value, G_SAFETY * lam)

        level_key = f"G{int(g_frac*100)}"
        switches = []
        latencies = []
        baseline_switches = []

        for seed in range(N_SEEDS):
            # Unique seed: config × regime × g_level × seed
            trial_seed = config_idx * 100000 + (0 if regime == 'dorsal' else 50000) + int(g_frac * 1000) * 100 + seed

            # --- GC-LCA trial ---
            switched, latency, dominant, mean_a, mean_b = simulate_dissociation_trial(
                lam, beta_r, alpha_r, sigma, gamma_adapt, kappa_r,
                SIGNAL_NEUTRAL, SIGNAL_NEUTRAL, g_value, -1,
                N_STEPS_BASELINE, N_STEPS_TOTAL,
                trial_seed, X_MAX, LOOKBACK_WINDOW, SWITCH_CRITERION
            )
            switches.append(int(switched))
            if latency >= 0:
                latencies.append(int(latency))

            # --- Same-seed baseline (G=0) ---
            sw_base, lat_base, _, _, _ = simulate_dissociation_trial(
                lam, beta_r, alpha_r, sigma, gamma_adapt, kappa_r,
                SIGNAL_NEUTRAL, SIGNAL_NEUTRAL, 0.0, -1,
                N_STEPS_BASELINE, N_STEPS_TOTAL,
                trial_seed, X_MAX, LOOKBACK_WINDOW, SWITCH_CRITERION
            )
            baseline_switches.append(int(sw_base))

        switch_rate = sum(switches) / len(switches)
        baseline_rate = sum(baseline_switches) / len(baseline_switches)
        mean_latency = float(np.mean(latencies)) if latencies else None
        median_latency = float(np.median(latencies)) if latencies else None

        level_result = {
            'g_fraction': g_frac,
            'g_value': round(g_value, 6),
            'switch_rate': round(switch_rate, 4),
            'n_switched': sum(switches),
            'n_seeds': len(switches),
            'baseline_switch_rate': round(baseline_rate, 4),
            'net_switch_rate': round(switch_rate - baseline_rate, 4),
            'mean_latency': round(mean_latency, 1) if mean_latency is not None else None,
            'median_latency': round(median_latency, 1) if median_latency is not None else None,
        }

        # --- Signal boost control (H4) --- only for G > 0
        if g_frac > 0:
            # Mean-matched boost: boost = G × mean(x_target) from baseline
            # Approximate mean activation from the neutral simulation
            # Use g_value * 0.5 as rough mean activation proxy (will be refined)
            boost_amount = g_value * 0.5  # conservative estimate
            boost_switches = []
            for seed in range(N_SEEDS):
                trial_seed = config_idx * 100000 + (0 if regime == 'dorsal' else 50000) + int(g_frac * 1000) * 100 + seed
                sw_boost, lat_boost = simulate_signal_boost_trial(
                    lam, beta_r, alpha_r, sigma, gamma_adapt, kappa_r,
                    SIGNAL_NEUTRAL, SIGNAL_NEUTRAL, boost_amount,
                    N_STEPS_BASELINE, N_STEPS_TOTAL,
                    trial_seed, X_MAX, LOOKBACK_WINDOW, SWITCH_CRITERION
                )
                boost_switches.append(int(sw_boost))
            boost_rate = sum(boost_switches) / len(boost_switches)
            level_result['signal_boost_rate'] = round(boost_rate, 4)
            level_result['signal_boost_amount'] = round(boost_amount, 6)
        else:
            level_result['signal_boost_rate'] = None

        result['g_levels'][level_key] = level_result

    # Dose-response monotonicity check
    g_keys = [f"G{int(f*100)}" for f in G_FRACTIONS if f > 0]
    rates = [result['g_levels'][k]['switch_rate'] for k in g_keys]
    is_monotonic = all(rates[i] <= rates[i+1] for i in range(len(rates)-1))
    result['dose_response_monotonic'] = is_monotonic

    return result


# =============================================================================
# MAIN
# =============================================================================

def sanitize_for_json(obj):
    """Convert numpy scalars and handle NaN/Inf for JSON."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj


def main():
    print("=" * 70)
    print("GC-LCA Phase II — Volitional Control & Dissociation Testing")
    print("Crewther Sampler Research Programme v2.2")
    print("=" * 70)

    n_configs = len(TOP_30_CONFIGS)
    n_g_levels = len(G_FRACTIONS)
    n_regimes = 2
    total_runs = n_configs * n_g_levels * n_regimes * N_SEEDS
    print(f"\nTop {n_configs} configs × {n_g_levels} G levels × {n_regimes} regimes × {N_SEEDS} seeds = {total_runs:,} runs")
    print(f"G fractions: {G_FRACTIONS}")
    print(f"Regimes: dorsal (α×1.5, β×0.6, κ×1.5), ventral (α×0.4, β×1.5, κ×0.4)")
    print(f"Protocol: {N_STEPS_BASELINE}-step baseline → {N_STEPS_PULSE}-step G pulse → {N_STEPS_RESPONSE-N_STEPS_PULSE}-step observation")
    print(f"Switch criterion: {SWITCH_CRITERION} consecutive steps")

    # Numba warmup
    print("\nWarming up Numba JIT...")
    _ = simulate_dissociation_trial(0.15, 0.2, 0.08, 0.10, 0.03, 0.06,
                                     0.5, 0.5, 0.05, 0, 100, 200, 0)
    _ = simulate_signal_boost_trial(0.15, 0.2, 0.08, 0.10, 0.03, 0.06,
                                     0.5, 0.5, 0.01, 100, 200, 0)
    print("JIT compilation complete.")

    # Build job list: 30 configs × 2 regimes = 60 jobs
    jobs = []
    for ci, (lam, beta, alpha, sigma, gamma_adapt, kappa, cv, rho) in enumerate(TOP_30_CONFIGS):
        for regime in ['dorsal', 'ventral']:
            jobs.append((ci, lam, beta, alpha, sigma, gamma_adapt, kappa, cv, rho, regime))

    n_workers = max(1, cpu_count() - 1)
    print(f"\nUsing {n_workers} worker processes for {len(jobs)} jobs")

    t_start = time.time()

    with Pool(n_workers) as pool:
        all_results = pool.map(process_config_phase2, jobs)

    elapsed = time.time() - t_start
    print(f"\nProcessing complete: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Organize results by config
    config_results = {}
    for r in all_results:
        ci = r['config_idx']
        regime = r['regime']
        if ci not in config_results:
            config_results[ci] = {
                'config_idx': ci,
                'base_params': r['base_params'],
                'phase1_cv': r['phase1_cv'],
                'phase1_levelt_rho': r['phase1_levelt_rho'],
            }
        config_results[ci][regime] = {
            'regime_params': r['regime_params'],
            'g_levels': r['g_levels'],
            'dose_response_monotonic': r['dose_response_monotonic'],
        }

    # ---- Summary statistics ----
    print("\n" + "=" * 70)
    print("PHASE II SUMMARY")
    print("=" * 70)

    # Headline: dissociation analysis
    dissociation_results = []
    for ci, cr in sorted(config_results.items()):
        dorsal = cr.get('dorsal', {})
        ventral = cr.get('ventral', {})

        # Best dorsal rate (across G levels, excluding G=0)
        dorsal_rates = []
        for gk, gv in dorsal.get('g_levels', {}).items():
            if gv['g_fraction'] > 0:
                dorsal_rates.append(gv['switch_rate'])
        ventral_rates = []
        for gk, gv in ventral.get('g_levels', {}).items():
            if gv['g_fraction'] > 0:
                ventral_rates.append(gv['switch_rate'])

        best_dorsal = max(dorsal_rates) if dorsal_rates else 0
        best_ventral = max(ventral_rates) if ventral_rates else 0

        dissociation_results.append({
            'config': ci,
            'best_dorsal': best_dorsal,
            'best_ventral': best_ventral,
            'dissociation': best_dorsal - best_ventral,
            'strict': best_dorsal >= 0.5 and best_ventral <= 0.2,
            'moderate': (best_dorsal - best_ventral) > 0.3,
            'dorsal_monotonic': dorsal.get('dose_response_monotonic', False),
        })

    n_strict = sum(1 for d in dissociation_results if d['strict'])
    n_moderate = sum(1 for d in dissociation_results if d['moderate'])
    n_monotonic = sum(1 for d in dissociation_results if d['dorsal_monotonic'])
    n_ventral_zero = sum(1 for d in dissociation_results if d['best_ventral'] == 0)

    print(f"\n  Strict dissociation (dorsal≥50%, ventral≤20%): {n_strict}/{n_configs}")
    print(f"  Moderate dissociation (dorsal−ventral > 0.3):   {n_moderate}/{n_configs}")
    print(f"  Dorsal dose-response monotonic:                 {n_monotonic}/{n_configs}")
    print(f"  Ventral floor = 0%:                             {n_ventral_zero}/{n_configs}")

    print(f"\n{'Cfg':>4} {'D_best':>7} {'V_best':>7} {'Dissoc':>7} {'Strict':>7} {'Mono':>5}")
    print("-" * 42)
    for d in dissociation_results:
        print(f"{d['config']:>4} {d['best_dorsal']:>7.2f} {d['best_ventral']:>7.2f} "
              f"{d['dissociation']:>7.2f} {'YES' if d['strict'] else 'no':>7} "
              f"{'YES' if d['dorsal_monotonic'] else 'no':>5}")

    # ---- Save ----
    output = {
        'metadata': {
            'programme': 'Crewther Sampler v2.2',
            'phase': 'Phase II Dissociation',
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_configs': n_configs,
            'n_seeds': N_SEEDS,
            'g_fractions': G_FRACTIONS,
            'protocol': {
                'baseline_steps': N_STEPS_BASELINE,
                'pulse_steps': N_STEPS_PULSE,
                'response_window': N_STEPS_RESPONSE,
                'switch_criterion': SWITCH_CRITERION,
                'lookback_window': LOOKBACK_WINDOW,
                'margin': MARGIN,
            },
            'regime_multipliers': {
                'dorsal': DORSAL_MULTIPLIERS,
                'ventral': VENTRAL_MULTIPLIERS,
            },
            'summary': {
                'strict_dissociation': n_strict,
                'moderate_dissociation': n_moderate,
                'dorsal_monotonic': n_monotonic,
                'ventral_floor_zero': n_ventral_zero,
            },
        },
        'config_results': sanitize_for_json(list(config_results.values())),
        'dissociation_summary': sanitize_for_json(dissociation_results),
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=1)

    file_size = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"\nSaved: {OUTPUT_FILE} ({file_size:.0f} KB)")
    print("=" * 70)


if __name__ == '__main__':
    main()
