#!/usr/bin/env python3
"""
GC-LCA Phase I Confirmatory Grid — Paper 1 Pre-Registered Protocol
===================================================================
Crewther Sampler Research Programme v2.2

Matches OSF pre-registration exactly:
  - Parameter grid: 4λ × 6β × 5α × 5σ × 3γ × 5κ = 9,000 configs
  - 30 seeds × 20,000 steps per config (rivalry/CV analysis)
  - Levelt: 11 signal levels {0.25, 0.30, ..., 0.75}, 8 seeds × 12,000 steps
  - Dominance extraction per §2.9: burn-in=500, margin=0.05, boundary exclusion
  - Metrics: CV, MLE/MoM gamma shape, Weibull params, AIC, Levelt ρ

Hardware target: 32GB RAM, AMD 7900GRE (CPU-bound, no GPU)
Expected runtime: ~10-12 hours overnight with Numba JIT + multiprocessing

Author: JP Cacioli
Date: March 2026
"""

import numpy as np
import json
import time
import os
import sys
import warnings
from itertools import product
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass, asdict

# Numba JIT for inner simulation loop
try:
    from numba import njit, prange
    HAS_NUMBA = True
    print("[INFO] Numba available — JIT compilation enabled")
except ImportError:
    HAS_NUMBA = False
    print("[WARNING] Numba not found — falling back to pure NumPy (will be slower)")
    print("         Install with: pip install numba")
    # Define a no-op decorator
    def njit(*args, **kwargs):
        def wrapper(f):
            return f
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return wrapper
    prange = range

# SciPy for distribution fitting
from scipy.stats import gamma as gamma_dist, weibull_min, expon, spearmanr
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# PRE-REGISTERED PARAMETER GRID (§2.7)
# =============================================================================

LAMBDA_VALUES = [0.08, 0.10, 0.15, 0.20]
BETA_VALUES   = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
ALPHA_VALUES  = [0.03, 0.05, 0.08, 0.10, 0.12]
SIGMA_VALUES  = [0.04, 0.06, 0.08, 0.10, 0.12]
GAMMA_VALUES  = [0.02, 0.03, 0.05]
KAPPA_FRACS   = [0.50, 0.75, 1.00, 1.25, 1.50]  # κ = frac × α

# Rivalry simulation
N_SEEDS_RIVALRY = 30
N_STEPS_RIVALRY = 20_000

# Levelt testing
SIGNAL_LEVELS = np.round(np.arange(0.25, 0.76, 0.05), 2)  # {0.25, 0.30, ..., 0.75}
N_SEEDS_LEVELT = 8
N_STEPS_LEVELT = 12_000

# Dominance extraction (§2.9)
BURN_IN = 500
MARGIN = 0.05
X_MAX = 5.0       # Firing rate saturation
G_SAFETY = 0.95   # G ceiling = 0.95 * λ (5% safety margin)

# Minimum rivalry threshold
MIN_SWITCHES_PER_SEED = 10

# Output
OUTPUT_FILE = "phase1_grid_results.json"
CHECKPOINT_DIR = "phase1_checkpoints"
CHECKPOINT_INTERVAL = 500  # save every N configs


# =============================================================================
# CORE SIMULATION (Numba JIT)
# =============================================================================

@njit(cache=True)
def simulate_gclca(lam, beta, alpha, sigma, gamma_adapt, kappa,
                   signal_a, signal_b, n_steps, seed, x_max=5.0):
    """
    Run a single GC-LCA trial with G=0 (neutral conditions).

    Returns trace_a, trace_b as float64 arrays.

    Core equation (§2.3):
        x_i(t+1) = clip[0, x_max]( (1 - λ + G_i)·x_i(t) + Signal_i
                                     - β·x_j(t) - α·a_i(t) + η_i(t) )
        a_i(t+1) = (1 - γ)·a_i(t) + κ·x_i(t)

    With G=0 for Phase I neutral conditions.
    """
    np.random.seed(seed)

    trace_a = np.empty(n_steps, dtype=np.float64)
    trace_b = np.empty(n_steps, dtype=np.float64)

    # Initial conditions: small random activation
    x_a = 0.1
    x_b = 0.1
    a_a = 0.0  # adaptation state
    a_b = 0.0

    leak_factor = 1.0 - lam  # G=0, so effective leak = λ

    for t in range(n_steps):
        trace_a[t] = x_a
        trace_b[t] = x_b

        # Noise
        eta_a = np.random.normal(0.0, sigma)
        eta_b = np.random.normal(0.0, sigma)

        # Update activations
        new_a = leak_factor * x_a + signal_a - beta * x_b - alpha * a_a + eta_a
        new_b = leak_factor * x_b + signal_b - beta * x_a - alpha * a_b + eta_b

        # Rectifier + saturation
        x_a = min(max(0.0, new_a), x_max)
        x_b = min(max(0.0, new_b), x_max)

        # Update adaptation
        a_a = (1.0 - gamma_adapt) * a_a + kappa * x_a
        a_b = (1.0 - gamma_adapt) * a_b + kappa * x_b

    return trace_a, trace_b


@njit(cache=True)
def extract_dominance_episodes(trace_a, trace_b, burn_in, margin):
    """
    Extract dominance episodes per §2.9:
    1. Discard burn_in steps
    2. Channel A dominant when trace_A - trace_B > margin
    3. Channel B dominant when trace_B - trace_A > margin
    4. Otherwise undetermined

    Returns arrays of (channel, duration) pairs.
    Channel: 0=A, 1=B
    """
    n = len(trace_a) - burn_in
    if n <= 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    # Determine dominance at each timestep
    max_episodes = n  # upper bound
    channels = np.empty(max_episodes, dtype=np.int32)
    durations = np.empty(max_episodes, dtype=np.int32)

    current_channel = -1  # -1 = undetermined, 0 = A, 1 = B
    current_duration = 0
    n_episodes = 0

    for t in range(burn_in, len(trace_a)):
        diff = trace_a[t] - trace_b[t]

        if diff > margin:
            dom = 0  # A dominant
        elif diff < -margin:
            dom = 1  # B dominant
        else:
            dom = -1  # undetermined

        if dom == current_channel and dom >= 0:
            current_duration += 1
        else:
            if current_channel >= 0 and current_duration > 0:
                channels[n_episodes] = current_channel
                durations[n_episodes] = current_duration
                n_episodes += 1
            current_channel = dom
            current_duration = 1 if dom >= 0 else 0

    # Final episode
    if current_channel >= 0 and current_duration > 0:
        channels[n_episodes] = current_channel
        durations[n_episodes] = current_duration
        n_episodes += 1

    return channels[:n_episodes], durations[:n_episodes]


# =============================================================================
# DISTRIBUTION FITTING (SciPy)
# =============================================================================

def fit_distributions(durations):
    """
    Fit gamma, Weibull, and exponential distributions.
    Compute AIC for each. Return MLE & MoM gamma shape, Weibull params.

    Returns dict with all metrics or None if insufficient data.
    """
    if len(durations) < 10:
        return None

    d = durations.astype(np.float64)
    n = len(d)
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    cv = np.std(d, ddof=1) / mean_d if mean_d > 0 else np.nan

    result = {
        'n_durations': int(n),
        'mean_duration': float(mean_d),
        'cv': float(cv),
        'std_duration': float(np.std(d, ddof=1)),
    }

    # MoM gamma shape: shape = (mean/std)^2
    if var_d > 0:
        mom_shape = mean_d**2 / var_d
    else:
        mom_shape = np.nan
    result['gamma_shape_mom'] = float(mom_shape)

    # MLE fits with error handling
    try:
        # Gamma MLE
        gamma_a, gamma_loc, gamma_scale = gamma_dist.fit(d, floc=0)
        gamma_ll = np.sum(gamma_dist.logpdf(d, gamma_a, loc=0, scale=gamma_scale))
        gamma_aic = 2 * 2 - 2 * gamma_ll  # 2 params (shape, scale)
        result['gamma_shape_mle'] = float(gamma_a)
        result['gamma_scale_mle'] = float(gamma_scale)
        result['gamma_aic'] = float(gamma_aic)
    except Exception:
        result['gamma_shape_mle'] = np.nan
        result['gamma_scale_mle'] = np.nan
        result['gamma_aic'] = np.inf

    try:
        # Weibull MLE
        wb_c, wb_loc, wb_scale = weibull_min.fit(d, floc=0)
        wb_ll = np.sum(weibull_min.logpdf(d, wb_c, loc=0, scale=wb_scale))
        wb_aic = 2 * 2 - 2 * wb_ll  # 2 params (shape, scale)
        result['weibull_shape'] = float(wb_c)
        result['weibull_scale'] = float(wb_scale)
        result['weibull_aic'] = float(wb_aic)
    except Exception:
        result['weibull_shape'] = np.nan
        result['weibull_scale'] = np.nan
        result['weibull_aic'] = np.inf

    try:
        # Exponential MLE
        exp_loc, exp_scale = expon.fit(d, floc=0)
        exp_ll = np.sum(expon.logpdf(d, loc=0, scale=exp_scale))
        exp_aic = 2 * 1 - 2 * exp_ll  # 1 param (scale)
        result['exponential_aic'] = float(exp_aic)
    except Exception:
        result['exponential_aic'] = np.inf

    # MLE/MoM agreement diagnostic
    result['mle_mom_agreement'] = abs(result['gamma_shape_mle'] - mom_shape) < 2.0

    # Best distribution by AIC
    aics = {
        'gamma': result.get('gamma_aic', np.inf),
        'weibull': result.get('weibull_aic', np.inf),
        'exponential': result.get('exponential_aic', np.inf),
    }
    result['best_distribution'] = min(aics, key=aics.get)

    return result


# =============================================================================
# LEVELT TESTING
# =============================================================================

def test_levelt_prop1(lam, beta, alpha, sigma, gamma_adapt, kappa,
                      signal_levels, n_seeds, n_steps):
    """
    Levelt Proposition I: increasing strength of stimulus to one eye
    increases predominance of that eye's percept.

    For each signal level pair (s, 0.5) where s varies, compute mean
    dominance duration ratio. Then Spearman ρ between signal strength
    and target predominance.

    Returns Spearman ρ and p-value.
    """
    signal_base = 0.50  # fixed signal for channel B
    predominance_ratios = []

    for sig_a in signal_levels:
        total_dur_a = 0
        total_dur_b = 0

        for seed in range(n_seeds):
            trace_a, trace_b = simulate_gclca(
                lam, beta, alpha, sigma, gamma_adapt, kappa,
                float(sig_a), signal_base, n_steps, seed * 1000 + int(sig_a * 100)
            )

            channels, durations = extract_dominance_episodes(
                trace_a, trace_b, BURN_IN, MARGIN
            )

            if len(channels) < 4:  # need at least a few episodes
                continue

            # Boundary exclusion (§2.9 step 4)
            if len(channels) > 2:
                ch_inner = channels[1:-1]
                dur_inner = durations[1:-1]
            else:
                continue

            for i in range(len(ch_inner)):
                if ch_inner[i] == 0:
                    total_dur_a += dur_inner[i]
                else:
                    total_dur_b += dur_inner[i]

        if total_dur_a + total_dur_b > 0:
            predominance_ratios.append(total_dur_a / (total_dur_a + total_dur_b))
        else:
            predominance_ratios.append(np.nan)

    # Spearman ρ: signal strength vs predominance
    valid_mask = ~np.isnan(predominance_ratios)
    if np.sum(valid_mask) < 4:
        return np.nan, np.nan

    rho, p_val = spearmanr(
        np.array(signal_levels)[valid_mask],
        np.array(predominance_ratios)[valid_mask]
    )
    return float(rho), float(p_val)


# =============================================================================
# SINGLE CONFIGURATION WORKER
# =============================================================================

def process_config(args):
    """
    Process a single parameter configuration.
    Returns a dict with all metrics.
    """
    config_idx, lam, beta, alpha, sigma, gamma_adapt, kappa_frac = args
    kappa = kappa_frac * alpha

    config_id = {
        'index': config_idx,
        'lambda': lam,
        'beta': beta,
        'alpha': alpha,
        'sigma': sigma,
        'gamma': gamma_adapt,
        'kappa_frac': kappa_frac,
        'kappa': round(kappa, 6),
    }

    # ---- Phase I.a: Rivalry simulation (30 seeds × 20K steps) ----
    signal_a = 0.50  # equal signals for neutral conditions
    signal_b = 0.50

    all_durations = []      # pooled across seeds (after boundary exclusion)
    per_seed_cvs = []
    per_seed_n_switches = []
    rivalry_producing = True

    for seed in range(N_SEEDS_RIVALRY):
        trace_a, trace_b = simulate_gclca(
            lam, beta, alpha, sigma, gamma_adapt, kappa,
            signal_a, signal_b, N_STEPS_RIVALRY, seed
        )

        channels, durations = extract_dominance_episodes(
            trace_a, trace_b, BURN_IN, MARGIN
        )

        n_switches = max(0, len(channels) - 1)
        per_seed_n_switches.append(n_switches)

        if len(channels) < 4:
            per_seed_cvs.append(np.nan)
            continue

        # Boundary exclusion (§2.9 step 4)
        dur_inner = durations[1:-1]

        if len(dur_inner) < 3:
            per_seed_cvs.append(np.nan)
            continue

        d = dur_inner.astype(np.float64)
        m = np.mean(d)
        if m > 0:
            per_seed_cvs.append(float(np.std(d, ddof=1) / m))
        else:
            per_seed_cvs.append(np.nan)

        all_durations.extend(dur_inner.tolist())

    # Check rivalry criterion
    mean_switches = np.mean(per_seed_n_switches) if per_seed_n_switches else 0
    if mean_switches < MIN_SWITCHES_PER_SEED:
        rivalry_producing = False

    config_id['rivalry_producing'] = rivalry_producing
    config_id['mean_switches_per_seed'] = float(mean_switches)

    if not rivalry_producing or len(all_durations) < 30:
        config_id['rivalry_metrics'] = None
        config_id['levelt_rho'] = np.nan
        config_id['levelt_p'] = np.nan
        return config_id

    # ---- Distribution fitting on pooled durations ----
    durations_arr = np.array(all_durations)
    dist_results = fit_distributions(durations_arr)

    # Per-seed CV stats
    valid_cvs = [c for c in per_seed_cvs if not np.isnan(c)]
    if valid_cvs:
        dist_results['cv_per_seed_mean'] = float(np.mean(valid_cvs))
        dist_results['cv_per_seed_std'] = float(np.std(valid_cvs, ddof=1)) if len(valid_cvs) > 1 else 0.0
        dist_results['cv_per_seed_min'] = float(np.min(valid_cvs))
        dist_results['cv_per_seed_max'] = float(np.max(valid_cvs))
        dist_results['n_valid_seeds'] = len(valid_cvs)
    else:
        dist_results['cv_per_seed_mean'] = np.nan
        dist_results['cv_per_seed_std'] = np.nan
        dist_results['n_valid_seeds'] = 0

    config_id['rivalry_metrics'] = dist_results

    # ---- Phase I.b: Levelt Proposition I test ----
    levelt_rho, levelt_p = test_levelt_prop1(
        lam, beta, alpha, sigma, gamma_adapt, kappa,
        SIGNAL_LEVELS, N_SEEDS_LEVELT, N_STEPS_LEVELT
    )
    config_id['levelt_rho'] = levelt_rho
    config_id['levelt_p'] = levelt_p

    return config_id


# =============================================================================
# MAIN
# =============================================================================

def build_config_list():
    """Build the full 9,000-config parameter grid."""
    configs = []
    idx = 0
    for lam in LAMBDA_VALUES:
        for beta in BETA_VALUES:
            for alpha in ALPHA_VALUES:
                for sigma in SIGMA_VALUES:
                    for gamma_adapt in GAMMA_VALUES:
                        for kappa_frac in KAPPA_FRACS:
                            configs.append(
                                (idx, lam, beta, alpha, sigma, gamma_adapt, kappa_frac)
                            )
                            idx += 1
    return configs


def sanitize_for_json(obj):
    """Replace NaN/Inf with None; convert numpy scalars to native Python types."""
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
    print("GC-LCA Phase I Confirmatory Grid")
    print("Crewther Sampler Research Programme v2.2")
    print("=" * 70)

    configs = build_config_list()
    n_configs = len(configs)
    print(f"\nParameter grid: {n_configs} configurations")
    print(f"  λ: {LAMBDA_VALUES}")
    print(f"  β: {BETA_VALUES}")
    print(f"  α: {ALPHA_VALUES}")
    print(f"  σ: {SIGMA_VALUES}")
    print(f"  γ: {GAMMA_VALUES}")
    print(f"  κ fracs: {KAPPA_FRACS}")
    print(f"\nRivalry: {N_SEEDS_RIVALRY} seeds × {N_STEPS_RIVALRY} steps")
    print(f"Levelt: {N_SEEDS_LEVELT} seeds × {N_STEPS_LEVELT} steps × {len(SIGNAL_LEVELS)} levels")
    print(f"\nTotal simulations: {n_configs * N_SEEDS_RIVALRY + n_configs * N_SEEDS_LEVELT * len(SIGNAL_LEVELS):,}")

    # Numba warmup — compile the JIT functions with a tiny run
    print("\nWarming up Numba JIT...")
    _ = simulate_gclca(0.15, 0.20, 0.08, 0.10, 0.03, 0.06, 0.5, 0.5, 100, 0)
    _ = extract_dominance_episodes(
        np.random.randn(200), np.random.randn(200), 50, 0.05
    )
    print("JIT compilation complete.")

    # Determine parallelism
    n_workers = max(1, cpu_count() - 1)  # leave 1 core free
    print(f"\nUsing {n_workers} worker processes")

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Check for existing checkpoint to resume from
    results = []
    start_idx = 0
    checkpoint_files = sorted([
        f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("ckpt_") and f.endswith(".json")
    ])
    if checkpoint_files:
        latest_ckpt = os.path.join(CHECKPOINT_DIR, checkpoint_files[-1])
        print(f"\nFound checkpoint: {latest_ckpt}")
        try:
            with open(latest_ckpt, 'r') as f:
                results = json.load(f)
            start_idx = len(results)
            print(f"  Resuming from config {start_idx}/{n_configs}")
        except Exception as e:
            print(f"  Failed to load checkpoint: {e}. Starting fresh.")
            results = []
            start_idx = 0

    remaining_configs = configs[start_idx:]
    if not remaining_configs:
        print("\nAll configurations already processed!")
    else:
        print(f"\nProcessing {len(remaining_configs)} configurations...")
        t_start = time.time()

        # Process in batches for checkpointing
        batch_size = CHECKPOINT_INTERVAL
        for batch_start in range(0, len(remaining_configs), batch_size):
            batch = remaining_configs[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            n_batches = (len(remaining_configs) + batch_size - 1) // batch_size

            t_batch_start = time.time()

            with Pool(n_workers) as pool:
                batch_results = pool.map(process_config, batch)

            results.extend(batch_results)

            # Progress report
            elapsed = time.time() - t_start
            total_done = start_idx + batch_start + len(batch)
            rate = (batch_start + len(batch)) / elapsed if elapsed > 0 else 0
            remaining_secs = (n_configs - total_done) / rate if rate > 0 else 0

            print(f"  Batch {batch_num}/{n_batches}: "
                  f"{total_done}/{n_configs} configs "
                  f"({100*total_done/n_configs:.1f}%) | "
                  f"Elapsed: {elapsed/3600:.1f}h | "
                  f"ETA: {remaining_secs/3600:.1f}h | "
                  f"Rate: {rate:.1f} configs/s")

            # Checkpoint
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"ckpt_{total_done:06d}.json")
            with open(ckpt_path, 'w') as f:
                json.dump(sanitize_for_json(results), f)

        total_time = time.time() - t_start
        print(f"\nTotal processing time: {total_time/3600:.2f} hours")

    # ---- Save final results ----
    print(f"\nSaving results to {OUTPUT_FILE}...")

    # Summary statistics
    n_rivalry = sum(1 for r in results if r.get('rivalry_producing', False))
    n_levelt = sum(1 for r in results
                   if r.get('levelt_rho') is not None
                   and r['levelt_rho'] is not None
                   and r['levelt_rho'] > 0.7)
    n_cv_target = 0
    for r in results:
        m = r.get('rivalry_metrics')
        if m and m.get('cv') is not None:
            if 0.35 <= m['cv'] <= 0.65:
                n_cv_target += 1

    output = {
        'metadata': {
            'programme': 'Crewther Sampler v2.2',
            'phase': 'Phase I Confirmatory Grid',
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_configs': n_configs,
            'grid': {
                'lambda': LAMBDA_VALUES,
                'beta': BETA_VALUES,
                'alpha': ALPHA_VALUES,
                'sigma': SIGMA_VALUES,
                'gamma': GAMMA_VALUES,
                'kappa_fracs': KAPPA_FRACS,
            },
            'rivalry_params': {
                'n_seeds': N_SEEDS_RIVALRY,
                'n_steps': N_STEPS_RIVALRY,
                'burn_in': BURN_IN,
                'margin': MARGIN,
                'x_max': X_MAX,
            },
            'levelt_params': {
                'signal_levels': SIGNAL_LEVELS.tolist(),
                'n_seeds': N_SEEDS_LEVELT,
                'n_steps': N_STEPS_LEVELT,
            },
            'summary': {
                'rivalry_producing': n_rivalry,
                'levelt_rho_gt_0.7': n_levelt,
                'cv_in_target': n_cv_target,
            },
        },
        'results': sanitize_for_json(results),
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=1)

    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"  Saved: {OUTPUT_FILE} ({file_size_mb:.1f} MB)")

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print("PHASE I SUMMARY")
    print("=" * 70)
    print(f"  Total configurations:     {n_configs}")
    print(f"  Rivalry-producing:        {n_rivalry} ({100*n_rivalry/n_configs:.1f}%)")
    print(f"  Levelt ρ > 0.7:           {n_levelt}")
    print(f"  CV ∈ [0.35, 0.65]:        {n_cv_target}")
    print(f"\nOutput: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == '__main__':
    main()
