"""
GC-LCA Zoomed Heatmap — Local Execution Script
================================================
Run this on your PC. Pure CPU/NumPy — no GPU needed.

Zoomed parameter space:
  α (adaptation) ∈ [0.04, 0.14]  — 11 values
  β (inhibition)  ∈ [0.10, 0.35]  — 11 values  
  κ (adaptation gain) — 5 values per α (independent, not locked to 0.75α)

Fixed: λ=0.15, σ=0.10, γ=0.03
Total: 11 × 11 × 5 = 605 configurations × 20 seeds each

Estimated runtime: 20-40 minutes on a modern CPU.
Estimated RAM: < 1GB peak.

Outputs:
  zoomed_heatmap_results.json  — full results for every config
  zoomed_heatmap_summary.txt   — human-readable summary
  
Bring the JSON back to Claude for visualization and analysis.

Usage:
  python gc_lca_zoomed_heatmap.py
"""

import numpy as np
from scipy import stats
import json
import time
import sys
import os

# ═══════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════

def run_gclca(signal_A, signal_B, G_A=0.0, G_B=0.0,
              lam=0.15, beta=0.20, alpha=0.08, sigma=0.10,
              gamma_adapt=0.03, kappa=0.06, T=10000, seed=42, x_max=5.0):
    """GC-LCA with stability constraint."""
    rng = np.random.RandomState(seed)
    xA, xB = 0.01, 0.01
    aA, aB = 0.0, 0.0
    trace_A = np.empty(T)
    trace_B = np.empty(T)
    eff_GA = min(G_A, lam * 0.95)
    eff_GB = min(G_B, lam * 0.95)
    
    for t in range(T):
        eta_A = rng.normal(0, sigma)
        eta_B = rng.normal(0, sigma)
        xA_new = (1 - lam + eff_GA) * xA + signal_A - beta * xB - alpha * aA + eta_A
        xB_new = (1 - lam + eff_GB) * xB + signal_B - beta * xA - alpha * aB + eta_B
        xA_new = max(0.0, min(x_max, xA_new))
        xB_new = max(0.0, min(x_max, xB_new))
        aA = (1 - gamma_adapt) * aA + kappa * xA
        aB = (1 - gamma_adapt) * aB + kappa * xB
        xA, xB = xA_new, xB_new
        trace_A[t] = xA
        trace_B[t] = xB
    
    return trace_A, trace_B


# ═══════════════════════════════════════════════════════════════════════════
# DOMINANCE EXTRACTION (rigorous, following Levelt 1965)
# ═══════════════════════════════════════════════════════════════════════════

def extract_dominance(trace_A, trace_B, margin=0.05, min_dur=1,
                      exclude_boundaries=True, burnin=500):
    """
    Extract dominance durations with proper methodology.
    - margin: activation difference for dominance declaration
    - min_dur: minimum timesteps for a valid dominance episode
    - exclude_boundaries: drop first/last episodes (Levelt convention)
    - burnin: discard initial transient
    """
    tA = trace_A[burnin:]
    tB = trace_B[burnin:]
    
    dom = np.where(tA - tB > margin, 1,
          np.where(tB - tA > margin, -1, 0))
    
    if len(dom) == 0:
        return [], 0
    
    # Extract runs
    runs = []
    state = dom[0]
    dur = 1
    for t in range(1, len(dom)):
        if dom[t] == state:
            dur += 1
        else:
            runs.append((state, dur))
            state = dom[t]
            dur = 1
    runs.append((state, dur))
    
    # Filter: valid dominance episodes
    mixed_count = 0
    valid_runs = []
    for state, dur in runs:
        if state != 0 and dur >= min_dur:
            valid_runs.append(dur)
        elif state != 0 and dur < min_dur:
            mixed_count += 1
    
    # Exclude boundaries
    if exclude_boundaries and len(valid_runs) >= 3:
        valid_runs = valid_runs[1:-1]
    
    return valid_runs, mixed_count


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def analyse_durations(durations):
    """Compute all duration metrics for a set of dominance durations."""
    if len(durations) < 10:
        return {
            'n': len(durations),
            'valid': False
        }
    
    ad = np.array(durations, dtype=float)
    raw_mean = float(np.mean(ad))
    raw_std = float(np.std(ad))
    raw_cv = raw_std / raw_mean if raw_mean > 0 else float('nan')
    
    # MLE gamma
    try:
        sh_mle, _, sc_mle = stats.gamma.fit(ad, floc=0)
    except:
        sh_mle, sc_mle = float('nan'), float('nan')
    
    # MoM gamma
    sh_mom = (raw_mean / raw_std) ** 2 if raw_std > 0 else float('nan')
    
    # MLE/MoM agreement
    if not (np.isnan(sh_mle) or np.isnan(sh_mom)):
        mle_mom_agree = abs(sh_mle - sh_mom) < 2
    else:
        mle_mom_agree = False
    
    # AIC: gamma vs exponential vs weibull
    try:
        g_ll = float(np.sum(stats.gamma.logpdf(ad, a=sh_mle, scale=sc_mle)))
        aic_gamma = -2 * g_ll + 4
    except:
        aic_gamma = float('inf')
    
    try:
        _, esc = stats.expon.fit(ad, floc=0)
        e_ll = float(np.sum(stats.expon.logpdf(ad, loc=0, scale=esc)))
        aic_expon = -2 * e_ll + 2
    except:
        aic_expon = float('inf')
    
    try:
        wb_c, _, wb_sc = stats.weibull_min.fit(ad, floc=0)
        w_ll = float(np.sum(stats.weibull_min.logpdf(ad, c=wb_c, loc=0, scale=wb_sc)))
        aic_weibull = -2 * w_ll + 4
    except:
        aic_weibull = float('inf')
        wb_c = float('nan')
    
    return {
        'n': len(ad),
        'valid': True,
        'mean': raw_mean,
        'std': raw_std,
        'cv': raw_cv,
        'median': float(np.median(ad)),
        'p5': float(np.percentile(ad, 5)),
        'p95': float(np.percentile(ad, 95)),
        'sh_mle': float(sh_mle),
        'sc_mle': float(sc_mle),
        'sh_mom': float(sh_mom),
        'mle_mom_agree': mle_mom_agree,
        'aic_gamma': aic_gamma,
        'aic_expon': aic_expon,
        'aic_weibull': aic_weibull,
        'gamma_preferred': aic_gamma < aic_expon,
        'weibull_shape': float(wb_c) if not np.isnan(wb_c) else None,
        'best_model': ['gamma', 'exponential', 'weibull'][
            np.argmin([aic_gamma, aic_expon, aic_weibull])]
    }


def test_levelt_prop1(params, signals=None, n_seeds=10, T=15000):
    """Test Levelt Proposition I: predominance increases with signal strength."""
    if signals is None:
        signals = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    
    predominances = []
    for sig_A in signals:
        seed_predoms = []
        for seed in range(n_seeds):
            trA, trB = run_gclca(sig_A, 0.5, T=T, seed=seed, **params)
            durs = extract_dominance(trA, trB, margin=0.05, min_dur=1,
                                     exclude_boundaries=False, burnin=500)
            durations_list = durs[0]
            # Compute predominance from trace directly
            tA = trA[500:]
            tB = trB[500:]
            dom_A = np.sum(tA > tB + 0.05)
            dom_B = np.sum(tB > tA + 0.05)
            total = dom_A + dom_B
            if total > 0:
                seed_predoms.append(dom_A / total)
        
        if seed_predoms:
            predominances.append(float(np.mean(seed_predoms)))
        else:
            predominances.append(float('nan'))
    
    # Spearman correlation
    valid = [i for i, p in enumerate(predominances) if not np.isnan(p)]
    if len(valid) >= 4:
        sigs_valid = [signals[i] for i in valid]
        pred_valid = [predominances[i] for i in valid]
        rho, pval = stats.spearmanr(sigs_valid, pred_valid)
        return {
            'signals': signals,
            'predominances': predominances,
            'rho': float(rho),
            'p': float(pval),
            'holds': rho > 0.7
        }
    return {
        'signals': signals,
        'predominances': predominances,
        'rho': float('nan'),
        'p': float('nan'),
        'holds': False
    }


def test_dissociation(params, G_frac=0.5, n_seeds=20, T=10000):
    """
    Test dorsal/ventral dissociation at a single G level.
    Dorsal: α×1.5, β×0.6 (high adaptation, low inhibition)
    Ventral: α×0.4, β×1.5 (low adaptation, high inhibition)
    """
    lam = params['lam']
    G_val = G_frac * lam
    
    results = {}
    for regime_name, a_scale, b_scale in [('dorsal', 1.5, 0.6), ('ventral', 0.4, 1.5)]:
        regime_params = params.copy()
        regime_params['alpha'] = params['alpha'] * a_scale
        regime_params['kappa'] = params['kappa'] * a_scale  # scale kappa with alpha
        regime_params['beta'] = params['beta'] * b_scale
        
        switches = 0
        total = 0
        for seed in range(n_seeds):
            # Run without G first — find who's dominant at switch time
            trA, trB = run_gclca(0.5, 0.5, T=T, seed=seed, **regime_params)
            switch_time = T // 2
            
            # Who is dominant just before switch?
            window = slice(switch_time - 200, switch_time)
            A_dom_before = np.mean(trA[window]) > np.mean(trB[window])
            
            # Now run with G on the NON-dominant channel at switch_time
            if A_dom_before:
                # B is suppressed, try to switch TO B by boosting B
                trA2, trB2 = run_gclca(0.5, 0.5, G_A=0.0, G_B=G_val,
                                        T=T, seed=seed, **regime_params)
            else:
                # A is suppressed, try to switch TO A by boosting A
                trA2, trB2 = run_gclca(0.5, 0.5, G_A=G_val, G_B=0.0,
                                        T=T, seed=seed, **regime_params)
            
            # Check: did the previously suppressed channel become dominant?
            post_window = slice(switch_time + 200, switch_time + 500)
            if switch_time + 500 <= T:
                if A_dom_before:
                    switched = np.mean(trB2[post_window]) > np.mean(trA2[post_window]) + 0.05
                else:
                    switched = np.mean(trA2[post_window]) > np.mean(trB2[post_window]) + 0.05
                
                if switched:
                    switches += 1
                total += 1
        
        results[regime_name] = float(switches / total) if total > 0 else 0.0
    
    dissoc_magnitude = results.get('dorsal', 0) - results.get('ventral', 0)
    
    return {
        'dorsal_switch_rate': results.get('dorsal', 0),
        'ventral_switch_rate': results.get('ventral', 0),
        'dissociation': dissoc_magnitude,
        'good_dissoc': results.get('dorsal', 0) >= 0.5 and results.get('ventral', 0) <= 0.2
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN SWEEP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("GC-LCA ZOOMED HEATMAP — LOCAL EXECUTION")
    print("=" * 70)
    
    # Fixed parameters
    LAM = 0.15
    SIGMA = 0.10
    GAMMA = 0.03
    
    # Sweep ranges
    alphas = np.linspace(0.04, 0.14, 11)
    betas = np.linspace(0.10, 0.35, 11)
    
    # κ values: 5 independent values spanning 0.5α to 1.5α for each α
    # We define absolute κ values that cover the range
    kappa_fracs = [0.50, 0.75, 1.00, 1.25, 1.50]  # multiplier on α
    
    # Simulation params
    N_SEEDS = 20
    T_RIVALRY = 20000      # for duration analysis
    T_LEVELT = 12000       # for Levelt test  
    T_DISSOC = 8000        # for dissociation test
    N_SEEDS_LEVELT = 8
    N_SEEDS_DISSOC = 15
    
    total_configs = len(alphas) * len(betas) * len(kappa_fracs)
    print(f"\nSweep: {len(alphas)} α × {len(betas)} β × {len(kappa_fracs)} κ = {total_configs} configs")
    print(f"Seeds per config: {N_SEEDS} (rivalry), {N_SEEDS_LEVELT} (Levelt), {N_SEEDS_DISSOC} (dissoc)")
    print(f"Fixed: λ={LAM}, σ={SIGMA}, γ={GAMMA}")
    print(f"\nEstimated time: 20-45 minutes")
    print()
    
    results = []
    start_time = time.time()
    completed = 0
    
    for ai, alpha in enumerate(alphas):
        for bi, beta in enumerate(betas):
            for ki, kfrac in enumerate(kappa_fracs):
                kappa = alpha * kfrac
                
                params = {
                    'lam': LAM,
                    'beta': beta,
                    'alpha': alpha,
                    'sigma': SIGMA,
                    'gamma_adapt': GAMMA,
                    'kappa': kappa,
                }
                
                config_id = f"a{alpha:.3f}_b{beta:.3f}_k{kappa:.4f}"
                
                # ── 1. Rivalry durations ──────────────────────────
                all_durations = []
                total_mixed = 0
                total_switches = 0
                
                for seed in range(N_SEEDS):
                    trA, trB = run_gclca(0.5, 0.5, T=T_RIVALRY, seed=seed, **params)
                    durs, mixed = extract_dominance(
                        trA, trB, margin=0.05, min_dur=1,
                        exclude_boundaries=True, burnin=500
                    )
                    all_durations.extend(durs)
                    total_mixed += mixed
                    total_switches += len(durs)
                
                # Check: does this config produce rivalry?
                avg_switches = total_switches / N_SEEDS
                produces_rivalry = avg_switches >= 10
                
                # Duration metrics
                dur_metrics = analyse_durations(all_durations)
                
                # Also compute filtered metrics (min_dur=3)
                all_filtered = []
                for seed in range(N_SEEDS):
                    trA, trB = run_gclca(0.5, 0.5, T=T_RIVALRY, seed=seed, **params)
                    durs_f, _ = extract_dominance(
                        trA, trB, margin=0.05, min_dur=3,
                        exclude_boundaries=True, burnin=500
                    )
                    all_filtered.extend(durs_f)
                
                dur_filtered = analyse_durations(all_filtered)
                
                # ── 2. Levelt Prop I (only if rivalry) ────────────
                if produces_rivalry:
                    levelt = test_levelt_prop1(params, n_seeds=N_SEEDS_LEVELT, T=T_LEVELT)
                else:
                    levelt = {'rho': float('nan'), 'holds': False}
                
                # ── 3. Dissociation (only if rivalry) ─────────────
                if produces_rivalry:
                    dissoc = test_dissociation(params, G_frac=0.5,
                                               n_seeds=N_SEEDS_DISSOC, T=T_DISSOC)
                else:
                    dissoc = {'dorsal_switch_rate': 0, 'ventral_switch_rate': 0,
                              'dissociation': 0, 'good_dissoc': False}
                
                # ── Store result ──────────────────────────────────
                result = {
                    'config_id': config_id,
                    'alpha': float(alpha),
                    'beta': float(beta),
                    'kappa': float(kappa),
                    'kappa_frac': float(kfrac),
                    'lam': LAM,
                    'sigma': SIGMA,
                    'gamma': GAMMA,
                    'produces_rivalry': produces_rivalry,
                    'avg_switches_per_seed': float(avg_switches),
                    'total_mixed': int(total_mixed),
                    'durations_raw': dur_metrics,
                    'durations_filtered': dur_filtered,
                    'levelt': levelt,
                    'dissociation': dissoc,
                }
                
                # Check all-three-criteria overlap
                cv_ok = (dur_metrics.get('cv', 0) >= 0.35 and 
                         dur_metrics.get('cv', 99) <= 0.65) if dur_metrics.get('valid') else False
                cv_filtered_ok = (dur_filtered.get('cv', 0) >= 0.35 and 
                                   dur_filtered.get('cv', 99) <= 0.65) if dur_filtered.get('valid') else False
                levelt_ok = levelt.get('holds', False)
                dissoc_ok = dissoc.get('good_dissoc', False)
                
                result['cv_in_range'] = cv_ok
                result['cv_filtered_in_range'] = cv_filtered_ok
                result['all_three_raw'] = cv_ok and levelt_ok and dissoc_ok
                result['all_three_filtered'] = cv_filtered_ok and levelt_ok and dissoc_ok
                
                results.append(result)
                completed += 1
                
                # Progress
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_configs - completed) / rate if rate > 0 else 0
                
                # Print status every config
                status = []
                if produces_rivalry:
                    status.append(f"rivalry({avg_switches:.0f}sw)")
                    if dur_metrics.get('valid'):
                        status.append(f"CV={dur_metrics['cv']:.3f}")
                    if levelt_ok:
                        status.append("Levelt✓")
                    if dissoc_ok:
                        status.append("Dissoc✓")
                    if result['all_three_raw']:
                        status.append("★ALL THREE★")
                else:
                    status.append("no rivalry")
                
                print(f"[{completed:>4}/{total_configs}] "
                      f"α={alpha:.3f} β={beta:.3f} κ={kappa:.4f} | "
                      f"{' | '.join(status)}"
                      f"  [{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")
    
    # ═══════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════
    
    total_time = time.time() - start_time
    
    output = {
        'metadata': {
            'description': 'GC-LCA zoomed heatmap results',
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_configs': total_configs,
            'total_time_seconds': total_time,
            'fixed_params': {'lam': LAM, 'sigma': SIGMA, 'gamma': GAMMA},
            'sweep_params': {
                'alphas': [float(a) for a in alphas],
                'betas': [float(b) for b in betas],
                'kappa_fracs': kappa_fracs,
            },
            'sim_params': {
                'T_rivalry': T_RIVALRY,
                'T_levelt': T_LEVELT,
                'T_dissoc': T_DISSOC,
                'n_seeds': N_SEEDS,
                'n_seeds_levelt': N_SEEDS_LEVELT,
                'n_seeds_dissoc': N_SEEDS_DISSOC,
            }
        },
        'results': results
    }
    
    # Save JSON
    json_path = 'zoomed_heatmap_results.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved: {json_path} ({os.path.getsize(json_path) / 1024:.0f} KB)")
    
    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    
    n_rivalry = sum(1 for r in results if r['produces_rivalry'])
    n_levelt = sum(1 for r in results if r['levelt'].get('holds', False))
    n_cv_raw = sum(1 for r in results if r['cv_in_range'])
    n_cv_filt = sum(1 for r in results if r['cv_filtered_in_range'])
    n_dissoc = sum(1 for r in results if r['dissociation'].get('good_dissoc', False))
    n_all_raw = sum(1 for r in results if r['all_three_raw'])
    n_all_filt = sum(1 for r in results if r['all_three_filtered'])
    
    summary = f"""
{'='*70}
ZOOMED HEATMAP SUMMARY
{'='*70}

Total configurations: {total_configs}
Total time: {total_time:.0f}s ({total_time/60:.1f} min)
Time per config: {total_time/total_configs:.1f}s

RESULTS:
  Rivalry-producing (≥10 switches):  {n_rivalry:>4} / {total_configs} ({100*n_rivalry/total_configs:.1f}%)
  Levelt Prop I holds:               {n_levelt:>4} / {n_rivalry} ({100*n_levelt/n_rivalry:.1f}% of rivalry) 
  CV ∈ [0.35, 0.65] (raw):           {n_cv_raw:>4} / {n_rivalry} ({100*n_cv_raw/max(n_rivalry,1):.1f}% of rivalry)
  CV ∈ [0.35, 0.65] (filtered):      {n_cv_filt:>4} / {n_rivalry} ({100*n_cv_filt/max(n_rivalry,1):.1f}% of rivalry)
  Dissociation (dorsal≥50%,vent≤20%): {n_dissoc:>4} / {n_rivalry} ({100*n_dissoc/max(n_rivalry,1):.1f}% of rivalry)
  
  ★ ALL THREE (raw CV):               {n_all_raw:>4} / {total_configs}
  ★ ALL THREE (filtered CV):          {n_all_filt:>4} / {total_configs}
"""
    
    # List overlap configs
    overlap_raw = [r for r in results if r['all_three_raw']]
    overlap_filt = [r for r in results if r['all_three_filtered']]
    
    if overlap_raw:
        summary += "\nOVERLAP CONFIGURATIONS (raw CV):\n"
        for r in overlap_raw:
            d = r['durations_raw']
            summary += (f"  α={r['alpha']:.3f} β={r['beta']:.3f} κ={r['kappa']:.4f} "
                       f"| CV={d['cv']:.3f} MLE_sh={d['sh_mle']:.1f} MoM_sh={d['sh_mom']:.1f} "
                       f"| Levelt ρ={r['levelt']['rho']:.2f} "
                       f"| Dorsal={r['dissociation']['dorsal_switch_rate']:.0%} "
                       f"Ventral={r['dissociation']['ventral_switch_rate']:.0%}\n")
    
    if overlap_filt:
        summary += "\nOVERLAP CONFIGURATIONS (filtered CV):\n"
        for r in overlap_filt:
            d = r['durations_filtered']
            summary += (f"  α={r['alpha']:.3f} β={r['beta']:.3f} κ={r['kappa']:.4f} "
                       f"| CV={d['cv']:.3f} MLE_sh={d['sh_mle']:.1f} MoM_sh={d['sh_mom']:.1f} "
                       f"| Levelt ρ={r['levelt']['rho']:.2f} "
                       f"| Dorsal={r['dissociation']['dorsal_switch_rate']:.0%} "
                       f"Ventral={r['dissociation']['ventral_switch_rate']:.0%}\n")
    
    # Best CV configs (closest to 0.5)
    rivalry_results = [r for r in results if r['produces_rivalry'] and r['durations_raw'].get('valid')]
    if rivalry_results:
        by_cv = sorted(rivalry_results, key=lambda r: abs(r['durations_raw']['cv'] - 0.5))
        summary += "\nTOP 10 CLOSEST TO CV=0.5:\n"
        for r in by_cv[:10]:
            d = r['durations_raw']
            summary += (f"  α={r['alpha']:.3f} β={r['beta']:.3f} κ={r['kappa']:.4f} "
                       f"| CV={d['cv']:.3f} | switches={r['avg_switches_per_seed']:.0f}/seed "
                       f"| MLE_sh={d['sh_mle']:.1f} MoM_sh={d['sh_mom']:.1f} "
                       f"| agree={'Y' if d['mle_mom_agree'] else 'N'}"
                       f"| best={d['best_model']}\n")
    
    print(summary)
    
    # Save summary
    summary_path = 'zoomed_heatmap_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"Summary saved: {summary_path}")
    

if __name__ == '__main__':
    main()
