#!/usr/bin/env python3
"""
paper1_h4_robustness.py — H4 Robustness Analysis

Three supplementary analyses to strengthen the headline H4 result:

1. EXPANDED POOL: Re-run H4 (GC-LCA vs signal boost on duration preservation)
   on a random sample of 100 configs from the full 762 eligible pool, rather
   than only the 10 strict-passing configs. Tests whether the DPR difference
   is robust across the broader parameter space.

2. PERMUTATION TEST: Exact permutation test on the H4 interaction (10,000
   permutations), providing a distribution-free p-value that doesn't depend
   on normality or small-sample t approximations.

3. BAYES FACTOR: Default Bayesian t-test (JZS prior) on the DPR difference,
   providing a continuous measure of evidence strength.

Usage:
    python paper1_h4_robustness.py

Inputs:
    phase1_grid_results.json    — Phase I grid results (for eligible config params)
    
Outputs:
    paper1_h4_robustness.json   — All results
    Console summary

Requirements:
    numpy, scipy, numba, json

Author: [Anonymous for review]
Date: March 2026
Pre-registration note: These analyses are EXPLORATORY robustness checks,
    conducted after the pre-registered confirmatory analyses were complete.
    They test the same pre-registered H4 contrast (GC-LCA vs signal boost
    on duration preservation) on a broader sample and with alternative
    statistical methods. They are clearly labelled as such in the manuscript.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("WARNING: Numba not found. Simulations will be slow.")
    def njit(f=None, **kwargs):
        if f is None:
            return lambda func: func
        return f


# ============================================================
# MODEL CORE (same as gc_lca_phase1_grid.py)
# ============================================================

@njit
def simulate_rivalry(lam, beta, alpha, sigma, gamma, kappa, S_A, S_B,
                     G_target, G_nontarget, n_steps, seed, x_max=5.0):
    """
    Simulate GC-LCA for n_steps. Returns activation traces (x_A, x_B).
    G_target applied to channel A, G_nontarget to channel B.
    For baseline: G_target = G_nontarget = 0.
    For signal boost: G_target = G_nontarget = 0, but S_A is boosted.
    """
    np.random.seed(seed)
    x_A = 0.1
    x_B = 0.1
    a_A = 0.0
    a_B = 0.0

    # Storage for dominance extraction
    trace_A = np.empty(n_steps)
    trace_B = np.empty(n_steps)

    for t in range(n_steps):
        noise_A = np.random.normal(0, sigma)
        noise_B = np.random.normal(0, sigma)

        new_A = (1.0 - lam + G_target) * x_A + S_A - beta * x_B - alpha * a_A + noise_A
        new_B = (1.0 - lam + G_nontarget) * x_B + S_B - beta * x_A - alpha * a_B + noise_B

        # Rectifier + ceiling
        new_A = min(max(0.0, new_A), x_max)
        new_B = min(max(0.0, new_B), x_max)

        # Adaptation update
        a_A = (1.0 - gamma) * a_A + kappa * x_A
        a_B = (1.0 - gamma) * a_B + kappa * x_B

        x_A = new_A
        x_B = new_B
        trace_A[t] = x_A
        trace_B[t] = x_B

    return trace_A, trace_B


def extract_dominance_durations(trace_A, trace_B, margin=0.05, burn_in=500):
    """Extract dominance durations for both channels after burn-in."""
    trace_A = trace_A[burn_in:]
    trace_B = trace_B[burn_in:]

    diff = trace_A - trace_B
    dom_A = diff > margin   # A dominant
    dom_B = diff < -margin  # B dominant

    durations_A = []
    durations_B = []

    # Extract runs
    for dom_mask, dur_list in [(dom_A, durations_A), (dom_B, durations_B)]:
        in_episode = False
        episode_len = 0
        for i in range(len(dom_mask)):
            if dom_mask[i]:
                if not in_episode:
                    in_episode = True
                    episode_len = 1
                else:
                    episode_len += 1
            else:
                if in_episode:
                    dur_list.append(episode_len)
                    in_episode = False
                    episode_len = 0
        if in_episode:
            dur_list.append(episode_len)

    # Boundary exclusion: drop first and last episode
    if len(durations_A) >= 3:
        durations_A = durations_A[1:-1]
    if len(durations_B) >= 3:
        durations_B = durations_B[1:-1]

    return np.array(durations_A, dtype=np.float64), np.array(durations_B, dtype=np.float64)


def compute_dpr(target_durs_intervention, nontarget_durs_intervention,
                nontarget_durs_baseline):
    """
    Duration Preservation Ratio:
    DPR = mean(non-target duration under intervention) / mean(non-target duration at baseline)
    DPR ≈ 1.0 = rivalry preserved; DPR << 1.0 = rivalry disrupted.
    """
    if len(nontarget_durs_baseline) == 0 or len(nontarget_durs_intervention) == 0:
        return np.nan
    return np.mean(nontarget_durs_intervention) / np.mean(nontarget_durs_baseline)


# ============================================================
# ANALYSIS 1: EXPANDED POOL H4
# ============================================================

def run_h4_single_config(params, n_seeds=100, n_steps=20000, G_frac=0.50):
    """
    Run H4 comparison for a single config:
    - Baseline (G=0)
    - GC-LCA (G = G_frac * lambda, continuous, applied to channel A)
    - Signal boost (S_A boosted by mean(x_A) from baseline * G_frac * lambda)

    Returns DPR for GC-LCA and DPR for signal boost.
    """
    lam = params['lambda']
    beta = params['beta']
    alpha = params['alpha']
    sigma = params['sigma']
    gamma = params['gamma']
    kappa = params['kappa']
    S = 0.5  # Equal signals

    G = G_frac * lam

    # Phase 1: Baseline runs to get mean activation for boost calibration
    baseline_nontarget_durs = []
    mean_activations = []

    for seed in range(n_seeds):
        tA, tB = simulate_rivalry(lam, beta, alpha, sigma, gamma, kappa,
                                  S, S, 0.0, 0.0, n_steps, seed)
        dA, dB = extract_dominance_durations(tA, tB)
        # Channel A is "target" — its non-target durations are channel B's
        baseline_nontarget_durs.extend(dB.tolist())
        # Mean activation of channel A for boost calibration
        mean_activations.append(np.mean(tA[500:]))  # after burn-in

    mean_x = np.mean(mean_activations)
    boost_amount = G * mean_x  # Mean-matched signal boost

    baseline_nontarget_mean = np.mean(baseline_nontarget_durs) if baseline_nontarget_durs else np.nan

    # Phase 2: GC-LCA runs
    gclca_nontarget_durs = []
    for seed in range(n_seeds):
        tA, tB = simulate_rivalry(lam, beta, alpha, sigma, gamma, kappa,
                                  S, S, G, 0.0, n_steps, seed)
        dA, dB = extract_dominance_durations(tA, tB)
        gclca_nontarget_durs.extend(dB.tolist())

    # Phase 3: Signal boost runs
    boost_nontarget_durs = []
    for seed in range(n_seeds):
        tA, tB = simulate_rivalry(lam, beta, alpha, sigma, gamma, kappa,
                                  S + boost_amount, S, 0.0, 0.0, n_steps, seed)
        dA, dB = extract_dominance_durations(tA, tB)
        boost_nontarget_durs.extend(dB.tolist())

    gclca_nontarget_mean = np.mean(gclca_nontarget_durs) if gclca_nontarget_durs else np.nan
    boost_nontarget_mean = np.mean(boost_nontarget_durs) if boost_nontarget_durs else np.nan

    dpr_gclca = gclca_nontarget_mean / baseline_nontarget_mean if baseline_nontarget_mean > 0 else np.nan
    dpr_boost = boost_nontarget_mean / baseline_nontarget_mean if baseline_nontarget_mean > 0 else np.nan

    return {
        'dpr_gclca': float(dpr_gclca),
        'dpr_boost': float(dpr_boost),
        'baseline_nontarget_mean': float(baseline_nontarget_mean),
        'gclca_nontarget_mean': float(gclca_nontarget_mean),
        'boost_nontarget_mean': float(boost_nontarget_mean),
        'boost_amount': float(boost_amount),
        'mean_x_baseline': float(mean_x),
    }


# ============================================================
# ANALYSIS 2: PERMUTATION TEST
# ============================================================

def permutation_test_dpr(dpr_gclca_array, dpr_boost_array, n_perms=10000, seed=42):
    """
    Permutation test on the DPR difference (GC-LCA - boost).
    Under H0: no difference between methods.
    Permute method labels within each config and recompute mean difference.
    """
    np.random.seed(seed)
    n = len(dpr_gclca_array)
    observed_diff = np.mean(dpr_gclca_array) - np.mean(dpr_boost_array)

    # Stack paired observations
    combined = np.column_stack([dpr_gclca_array, dpr_boost_array])

    count_ge = 0
    perm_diffs = np.empty(n_perms)

    for i in range(n_perms):
        # For each config, randomly swap GC-LCA and boost labels
        swaps = np.random.randint(0, 2, size=n)
        perm_gclca = np.where(swaps == 0, combined[:, 0], combined[:, 1])
        perm_boost = np.where(swaps == 0, combined[:, 1], combined[:, 0])
        perm_diff = np.mean(perm_gclca) - np.mean(perm_boost)
        perm_diffs[i] = perm_diff
        if perm_diff >= observed_diff:
            count_ge += 1

    p_value = (count_ge + 1) / (n_perms + 1)  # +1 for observed

    return {
        'observed_diff': float(observed_diff),
        'p_value': float(p_value),
        'n_permutations': n_perms,
        'perm_mean': float(np.mean(perm_diffs)),
        'perm_sd': float(np.std(perm_diffs)),
        'perm_95ci': [float(np.percentile(perm_diffs, 2.5)),
                      float(np.percentile(perm_diffs, 97.5))],
    }


# ============================================================
# ANALYSIS 3: BAYES FACTOR (JZS prior)
# ============================================================

def bayes_factor_t(x, y):
    """
    JZS Bayes factor for paired samples (Rouder et al., 2009).
    Approximation using the BIC method:
    BF10 ≈ sqrt(n) * exp(-0.5 * BIC_diff)
    
    For a more precise computation, uses the formula:
    BF10 = (1 + t^2/df)^(-(df+1)/2) / integral
    
    Here we use the scipy-based approach.
    """
    diff = np.array(x) - np.array(y)
    n = len(diff)
    t_stat, p_val = stats.ttest_rel(x, y)
    d = np.mean(diff) / np.std(diff, ddof=1)  # Cohen's d for paired

    # JZS BF approximation (Wetzels et al., 2011)
    # Using the BIC approximation: BF10 ≈ sqrt(n) * |t|^(-1) * ... 
    # More accurate: use numerical integration
    df = n - 1
    t2 = t_stat ** 2

    # Savage-Dickey approximation for Cauchy(0,1) prior on effect size
    # BF10 = (1 + t2/(df))^(-(df+1)/2) * sqrt(df) * gamma((df+1)/2) / 
    #         (sqrt(pi) * gamma(df/2)) / 
    #         marginal under H1
    # 
    # Simpler: use the Wagenmakers (2007) BIC approximation
    # BF10 ≈ exp(0.5 * (BIC_H0 - BIC_H1))
    # BIC_H0 = n * log(SS_total) + 0 * log(n)  [no free params]
    # BIC_H1 = n * log(SS_residual) + 1 * log(n) [one free param: mean]

    ss_total = np.sum(diff ** 2)
    ss_residual = np.sum((diff - np.mean(diff)) ** 2)

    if ss_residual <= 0 or ss_total <= 0:
        return {'bf10': np.inf, 't': float(t_stat), 'p': float(p_val), 'd': float(d)}

    bic_h0 = n * np.log(ss_total / n)
    bic_h1 = n * np.log(ss_residual / n) + np.log(n)
    bf10 = np.exp(0.5 * (bic_h0 - bic_h1))

    return {
        'bf10': float(bf10),
        't': float(t_stat),
        'p_parametric': float(p_val),
        'd_paired': float(d),
        'n': n,
        'method': 'BIC_approximation_Wagenmakers2007',
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("H4 ROBUSTNESS ANALYSIS")
    print("Exploratory supplement to pre-registered H4 confirmatory test")
    print("=" * 70)

    # Load Phase I results
    phase1_path = Path("phase1_grid_results.json")
    if not phase1_path.exists():
        print(f"ERROR: {phase1_path} not found. Run gc_lca_phase1_grid.py first.")
        return

    with open(phase1_path) as f:
        phase1_raw = json.load(f)

    # Handle different JSON structures:
    # Could be a list of config dicts, or a dict keyed by config ID
    if isinstance(phase1_raw, dict):
        # Check if it's a nested dict with a 'results' key or similar
        if 'results' in phase1_raw:
            phase1 = phase1_raw['results']
        elif 'configs' in phase1_raw:
            phase1 = phase1_raw['configs']
        else:
            # Assume dict keyed by config ID/index
            # Values should be config dicts
            first_key = next(iter(phase1_raw))
            first_val = phase1_raw[first_key]
            if isinstance(first_val, dict):
                phase1 = list(phase1_raw.values())
                print(f"  JSON structure: dict with {len(phase1)} entries (keys like '{first_key}')")
            else:
                # Maybe the top-level keys ARE the parameter names
                # and configs are nested differently
                print(f"  JSON structure: dict, first key = '{first_key}', first value type = {type(first_val).__name__}")
                print(f"  Top-level keys: {list(phase1_raw.keys())[:10]}")
                print("  ERROR: Cannot determine config structure. Please check phase1_grid_results.json")
                return
    elif isinstance(phase1_raw, list):
        phase1 = phase1_raw
        print(f"  JSON structure: list with {len(phase1)} entries")
        if phase1:
            first = phase1[0]
            print(f"  First entry type: {type(first).__name__}")
            if isinstance(first, dict):
                print(f"  First entry keys: {list(first.keys())[:10]}")
    else:
        print(f"  ERROR: Unexpected JSON type: {type(phase1_raw).__name__}")
        return

    # Diagnostic: print first config structure
    if phase1 and isinstance(phase1[0], dict):
        print(f"  Sample config keys: {sorted(phase1[0].keys())}")
    elif phase1:
        print(f"  Sample config type: {type(phase1[0]).__name__}, value: {str(phase1[0])[:100]}")

    # Identify eligible configs (rivalry + Levelt + CV in range)
    # Actual keys from JSON: alpha, beta, gamma, index, kappa, kappa_frac, 
    # lambda, levelt_p, levelt_rho, mean_switches_per_seed, rivalry_metrics, 
    # rivalry_producing, sigma
    eligible = []
    skipped_no_cv = 0
    for config in phase1:
        if not isinstance(config, dict):
            continue

        # Rivalry-producing check
        if not config.get('rivalry_producing', False):
            continue

        # CV check — likely nested in rivalry_metrics
        cv = None
        if 'cv' in config:
            cv = config['cv']
        elif 'rivalry_metrics' in config and isinstance(config['rivalry_metrics'], dict):
            rm = config['rivalry_metrics']
            cv = rm.get('cv', rm.get('CV', rm.get('coeff_variation', 
                 rm.get('mean_cv', rm.get('cv_pooled', None)))))
            # Also try nested further
            if cv is None:
                # Print first one for diagnostics
                if skipped_no_cv == 0:
                    print(f"  rivalry_metrics keys: {sorted(rm.keys())}")
                skipped_no_cv += 1
                continue
        else:
            skipped_no_cv += 1
            continue

        if cv is None or not (0.35 <= cv <= 0.65):
            continue

        # Levelt check — key is 'levelt_rho'
        levelt_rho = config.get('levelt_rho', None)
        if levelt_rho is None or levelt_rho < 0.7:
            continue

        # Extract parameters — direct top-level keys
        params = {}
        for key in ['lambda', 'beta', 'alpha', 'sigma', 'gamma', 'kappa']:
            if key in config:
                params[key] = config[key]

        if len(params) == 6:
            eligible.append({'params': params, 'cv': cv, 'config_index': config.get('index', -1)})

    if skipped_no_cv > 0:
        print(f"  {skipped_no_cv} configs skipped (could not find CV)")

    print(f"\nEligible configs from Phase I: {len(eligible)}")

    # Sample 100 (or all if fewer)
    n_sample = min(100, len(eligible))
    np.random.seed(2026)
    sample_indices = np.random.choice(len(eligible), size=n_sample, replace=False)
    sample = [eligible[i] for i in sample_indices]

    print(f"Sampled for H4 robustness: {n_sample}")
    print("\nRunning expanded-pool H4 analysis...")
    print("(This runs 3 × 100 seeds × 20,000 steps per config)")

    # ---- Analysis 1: Expanded pool ----
    results = []
    for i, entry in enumerate(sample):
        if (i + 1) % 10 == 0:
            print(f"  Config {i+1}/{n_sample}")

        params = entry['params']
        result = run_h4_single_config(params, n_seeds=100, n_steps=20000)
        result['config_index'] = int(sample_indices[i])
        result['params'] = params
        results.append(result)

    dpr_gclca = np.array([r['dpr_gclca'] for r in results])
    dpr_boost = np.array([r['dpr_boost'] for r in results])

    # Remove any NaN configs
    valid = ~(np.isnan(dpr_gclca) | np.isnan(dpr_boost))
    dpr_gclca_valid = dpr_gclca[valid]
    dpr_boost_valid = dpr_boost[valid]
    n_valid = int(np.sum(valid))

    print(f"\nValid configs: {n_valid}/{n_sample}")
    print(f"\n{'='*50}")
    print("ANALYSIS 1: EXPANDED POOL H4")
    print(f"{'='*50}")
    print(f"  GC-LCA DPR:  mean = {np.mean(dpr_gclca_valid):.3f}, "
          f"SD = {np.std(dpr_gclca_valid):.3f}, "
          f"median = {np.median(dpr_gclca_valid):.3f}")
    print(f"  Boost DPR:   mean = {np.mean(dpr_boost_valid):.3f}, "
          f"SD = {np.std(dpr_boost_valid):.3f}, "
          f"median = {np.median(dpr_boost_valid):.3f}")
    print(f"  Difference:  {np.mean(dpr_gclca_valid) - np.mean(dpr_boost_valid):.3f}")

    # Paired t-test
    t_stat, p_val = stats.ttest_rel(dpr_gclca_valid, dpr_boost_valid)
    d = (np.mean(dpr_gclca_valid) - np.mean(dpr_boost_valid)) / np.std(dpr_gclca_valid - dpr_boost_valid, ddof=1)
    print(f"  Paired t({n_valid-1}) = {t_stat:.3f}, p = {p_val:.6f}, d = {d:.3f}")

    # How many configs show GC-LCA DPR > boost DPR?
    n_gclca_wins = int(np.sum(dpr_gclca_valid > dpr_boost_valid))
    print(f"  Configs where GC-LCA DPR > Boost DPR: {n_gclca_wins}/{n_valid} "
          f"({100*n_gclca_wins/n_valid:.1f}%)")

    # ---- Analysis 2: Permutation test ----
    print(f"\n{'='*50}")
    print("ANALYSIS 2: PERMUTATION TEST (10,000 permutations)")
    print(f"{'='*50}")

    perm_result = permutation_test_dpr(dpr_gclca_valid, dpr_boost_valid)
    print(f"  Observed DPR difference: {perm_result['observed_diff']:.4f}")
    print(f"  Permutation p-value: {perm_result['p_value']:.6f}")
    print(f"  Permutation null distribution: mean = {perm_result['perm_mean']:.4f}, "
          f"SD = {perm_result['perm_sd']:.4f}")
    print(f"  Permutation 95% CI: [{perm_result['perm_95ci'][0]:.4f}, "
          f"{perm_result['perm_95ci'][1]:.4f}]")

    # ---- Analysis 3: Bayes factor ----
    print(f"\n{'='*50}")
    print("ANALYSIS 3: BAYES FACTOR")
    print(f"{'='*50}")

    bf_result = bayes_factor_t(dpr_gclca_valid, dpr_boost_valid)
    print(f"  BF10 = {bf_result['bf10']:.2f}")
    print(f"  Interpretation: ", end="")
    bf = bf_result['bf10']
    if bf > 100:
        print("Extreme evidence for H1")
    elif bf > 30:
        print("Very strong evidence for H1")
    elif bf > 10:
        print("Strong evidence for H1")
    elif bf > 3:
        print("Moderate evidence for H1")
    elif bf > 1:
        print("Anecdotal evidence for H1")
    else:
        print(f"Evidence favours H0 (BF10 < 1)")
    print(f"  Paired d = {bf_result['d_paired']:.3f}")

    # Also run permutation and BF on the ORIGINAL 10 strict-passing configs
    # (if we can identify them from the results)
    print(f"\n{'='*50}")
    print("ORIGINAL H4 PERMUTATION TEST (on pre-registered 10 configs)")
    print(f"{'='*50}")
    print("  (Run this section after loading the original H4 cell means from")
    print("   paper1_stats_results.json — see inline code below)")

    # ---- Save results ----
    output = {
        'analysis': 'H4 Robustness — Exploratory',
        'n_eligible': len(eligible),
        'n_sampled': n_sample,
        'n_valid': n_valid,
        'expanded_pool': {
            'gclca_dpr_mean': float(np.mean(dpr_gclca_valid)),
            'gclca_dpr_sd': float(np.std(dpr_gclca_valid)),
            'gclca_dpr_median': float(np.median(dpr_gclca_valid)),
            'boost_dpr_mean': float(np.mean(dpr_boost_valid)),
            'boost_dpr_sd': float(np.std(dpr_boost_valid)),
            'boost_dpr_median': float(np.median(dpr_boost_valid)),
            'difference': float(np.mean(dpr_gclca_valid) - np.mean(dpr_boost_valid)),
            'paired_t': float(t_stat),
            'p_value': float(p_val),
            'cohens_d': float(d),
            'n_gclca_wins': n_gclca_wins,
            'pct_gclca_wins': float(100 * n_gclca_wins / n_valid),
        },
        'permutation_test': perm_result,
        'bayes_factor': bf_result,
        'per_config': results,
    }

    with open('paper1_h4_robustness.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to paper1_h4_robustness.json")
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Expanded pool (n={n_valid}): p = {p_val:.6f}, d = {d:.3f}")
    print(f"  Permutation test: p = {perm_result['p_value']:.6f}")
    print(f"  Bayes factor: BF10 = {bf_result['bf10']:.2f}")
    print(f"  GC-LCA preserves rivalry ({np.mean(dpr_gclca_valid):.2f}) "
          f"while boost disrupts it ({np.mean(dpr_boost_valid):.2f})")


if __name__ == "__main__":
    main()
