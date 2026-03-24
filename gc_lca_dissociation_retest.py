"""
GC-LCA Dissociation Retest — Transient Intervention Protocol
=============================================================
Fixes the dissociation test to match the programme's theoretical claim:
G is applied as a TRANSIENT intervention, not a sustained bias.

Protocol:
  1. Run model to steady-state rivalry (5000 steps, G=0)
  2. Identify which channel is currently dominant
  3. Apply G to the SUPPRESSED channel at t=5000
  4. Check whether a switch occurs within 500 steps
  5. Compare dorsal vs ventral regime switch success

Reads: zoomed_heatmap_results.json (for the config list)
Outputs: dissociation_retest_results.json

Usage:
  python gc_lca_dissociation_retest.py

Estimated runtime: 10-20 minutes
"""

import numpy as np
from scipy import stats
import json
import time
import os


# ═══════════════════════════════════════════════════════════════════════════
# MODEL (same as heatmap script)
# ═══════════════════════════════════════════════════════════════════════════

def run_gclca_transient_G(signal_A, signal_B, G_target, target_channel,
                           switch_time, G_duration,
                           lam=0.15, beta=0.20, alpha=0.08, sigma=0.10,
                           gamma_adapt=0.03, kappa=0.06, T=10000, seed=42,
                           x_max=5.0):
    """
    GC-LCA with transient G intervention.
    
    G is applied to target_channel ('A' or 'B') starting at switch_time
    for G_duration timesteps, then removed.
    
    Returns full traces for both channels.
    """
    rng = np.random.RandomState(seed)
    xA, xB = 0.01, 0.01
    aA, aB = 0.0, 0.0
    trace_A = np.empty(T)
    trace_B = np.empty(T)
    
    eff_G = min(G_target, lam * 0.95)
    
    for t in range(T):
        eta_A = rng.normal(0, sigma)
        eta_B = rng.normal(0, sigma)
        
        # Determine if G is active
        g_active = switch_time <= t < switch_time + G_duration
        
        if g_active and target_channel == 'A':
            gA, gB = eff_G, 0.0
        elif g_active and target_channel == 'B':
            gA, gB = 0.0, eff_G
        else:
            gA, gB = 0.0, 0.0
        
        xA_new = (1 - lam + gA) * xA + signal_A - beta * xB - alpha * aA + eta_A
        xB_new = (1 - lam + gB) * xB + signal_B - beta * xA - alpha * aB + eta_B
        
        xA_new = max(0.0, min(x_max, xA_new))
        xB_new = max(0.0, min(x_max, xB_new))
        
        aA = (1 - gamma_adapt) * aA + kappa * xA
        aB = (1 - gamma_adapt) * aB + kappa * xB
        
        xA, xB = xA_new, xB_new
        trace_A[t] = xA
        trace_B[t] = xB
    
    return trace_A, trace_B


def run_gclca_baseline(signal_A, signal_B,
                        lam=0.15, beta=0.20, alpha=0.08, sigma=0.10,
                        gamma_adapt=0.03, kappa=0.06, T=10000, seed=42,
                        x_max=5.0):
    """Standard GC-LCA with G=0 throughout."""
    rng = np.random.RandomState(seed)
    xA, xB = 0.01, 0.01
    aA, aB = 0.0, 0.0
    trace_A = np.empty(T)
    trace_B = np.empty(T)
    
    for t in range(T):
        eta_A = rng.normal(0, sigma)
        eta_B = rng.normal(0, sigma)
        xA_new = (1 - lam) * xA + signal_A - beta * xB - alpha * aA + eta_A
        xB_new = (1 - lam) * xB + signal_B - beta * xA - alpha * aB + eta_B
        xA_new = max(0.0, min(x_max, xA_new))
        xB_new = max(0.0, min(x_max, xB_new))
        aA = (1 - gamma_adapt) * aA + kappa * xA
        aB = (1 - gamma_adapt) * aB + kappa * xB
        xA, xB = xA_new, xB_new
        trace_A[t] = xA
        trace_B[t] = xB
    
    return trace_A, trace_B


# ═══════════════════════════════════════════════════════════════════════════
# TRANSIENT DISSOCIATION TEST
# ═══════════════════════════════════════════════════════════════════════════

def test_dissociation_transient(params, G_fracs=[0.25, 0.50, 0.75, 0.90],
                                 n_seeds=30, T=8000,
                                 switch_time=5000, G_duration=500,
                                 check_window=300):
    """
    Test dorsal/ventral dissociation with transient G intervention.
    
    Protocol per seed:
      1. Run baseline (G=0) for T steps with same seed
      2. At switch_time, determine who is dominant (mean over preceding 200 steps)
      3. Run again WITH transient G on the suppressed channel
      4. Check if the suppressed channel becomes dominant within check_window
         after switch_time
      5. A "switch" = target channel's mean activation exceeds other by margin
         in the check window AFTER G onset
    
    Tests multiple G levels for dose-response.
    """
    lam = params['lam']
    
    results = {}
    
    for regime_name, a_scale, b_scale in [('dorsal', 1.5, 0.6), ('ventral', 0.4, 1.5)]:
        regime_params = params.copy()
        regime_params['alpha'] = params['alpha'] * a_scale
        regime_params['kappa'] = params['kappa'] * a_scale
        regime_params['beta'] = params['beta'] * b_scale
        
        regime_results = {}
        
        for G_frac in G_fracs:
            G_val = G_frac * lam
            
            switches = 0
            valid_trials = 0
            switch_latencies = []
            
            for seed in range(n_seeds):
                # Step 1: Baseline run to establish dominance
                trA_base, trB_base = run_gclca_baseline(
                    0.5, 0.5, T=T, seed=seed, **regime_params)
                
                # Step 2: Who is dominant just before switch_time?
                pre_window = slice(switch_time - 200, switch_time)
                mean_A_pre = np.mean(trA_base[pre_window])
                mean_B_pre = np.mean(trB_base[pre_window])
                
                # Need clear dominance to run the trial
                margin = 0.03
                if abs(mean_A_pre - mean_B_pre) < margin:
                    continue  # ambiguous, skip this seed
                
                A_dominant = mean_A_pre > mean_B_pre
                target = 'B' if A_dominant else 'A'  # boost the suppressed one
                
                # Step 3: Run with transient G
                trA_int, trB_int = run_gclca_transient_G(
                    0.5, 0.5, G_target=G_val, target_channel=target,
                    switch_time=switch_time, G_duration=G_duration,
                    T=T, seed=seed, **regime_params)
                
                # Step 4: Did the target become dominant?
                # Check in window after G onset
                post_start = switch_time + 50  # small delay for response
                post_end = min(switch_time + G_duration + check_window, T)
                
                if post_end <= post_start:
                    continue
                
                post_window = slice(post_start, post_end)
                
                if target == 'A':
                    target_trace = trA_int[post_window]
                    other_trace = trB_int[post_window]
                else:
                    target_trace = trB_int[post_window]
                    other_trace = trA_int[post_window]
                
                # Switch criterion: target exceeds other by margin for ≥50 consecutive steps
                dominance = target_trace - other_trace
                switch_threshold = 0.05
                consecutive_required = 50
                
                switched = False
                switch_latency = None
                consecutive = 0
                for i in range(len(dominance)):
                    if dominance[i] > switch_threshold:
                        consecutive += 1
                        if consecutive >= consecutive_required:
                            switched = True
                            switch_latency = (post_start - switch_time) + i - consecutive_required + 1
                            break
                    else:
                        consecutive = 0
                
                valid_trials += 1
                if switched:
                    switches += 1
                    if switch_latency is not None:
                        switch_latencies.append(switch_latency)
            
            rate = switches / valid_trials if valid_trials > 0 else 0.0
            regime_results[f'G{int(G_frac*100)}'] = {
                'switch_rate': rate,
                'switches': switches,
                'valid_trials': valid_trials,
                'mean_latency': float(np.mean(switch_latencies)) if switch_latencies else None,
                'G_frac': G_frac,
                'G_val': G_val
            }
        
        results[regime_name] = regime_results
    
    # Compute dissociation at each G level
    dissociation = {}
    for G_frac in G_fracs:
        key = f'G{int(G_frac*100)}'
        d_rate = results['dorsal'][key]['switch_rate']
        v_rate = results['ventral'][key]['switch_rate']
        dissociation[key] = {
            'dorsal': d_rate,
            'ventral': v_rate,
            'difference': d_rate - v_rate,
            'good_strict': d_rate >= 0.5 and v_rate <= 0.2,
            'good_moderate': (d_rate - v_rate) >= 0.3,
        }
    
    # Best G level for dissociation
    best_key = max(dissociation.keys(), 
                   key=lambda k: dissociation[k]['difference'])
    
    return {
        'dorsal': results['dorsal'],
        'ventral': results['ventral'],
        'dissociation': dissociation,
        'best_G': best_key,
        'best_dissoc': dissociation[best_key]
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("GC-LCA DISSOCIATION RETEST — TRANSIENT INTERVENTION")
    print("=" * 70)
    
    # Load existing results for config list
    json_path = 'zoomed_heatmap_results.json'
    if not os.path.exists(json_path):
        print(f"ERROR: {json_path} not found. Run gc_lca_zoomed_heatmap.py first.")
        return
    
    with open(json_path) as f:
        heatmap_data = json.load(f)
    
    all_results = heatmap_data['results']
    rivalry_configs = [r for r in all_results if r['produces_rivalry']]
    
    # De-duplicate: unique (α, β, best-κ) combos
    # We test the κ that gave best CV for each α,β
    # Build lookup
    seen = {}
    for r in rivalry_configs:
        key = (round(r['alpha'], 4), round(r['beta'], 4))
        cv = r['durations_raw'].get('cv', 99)
        if key not in seen or abs(cv - 0.5) < abs(seen[key]['durations_raw'].get('cv', 99) - 0.5):
            seen[key] = r
    
    configs_to_test = list(seen.values())
    print(f"\nUnique (α,β) configs to retest: {len(configs_to_test)}")
    print(f"G levels: 25%, 50%, 75%, 90% of λ")
    print(f"Seeds per regime per G: 30")
    print(f"Protocol: Transient G for 500 steps at t=5000, check 300 steps after")
    print(f"\nEstimated time: 10-20 minutes")
    print()
    
    results = []
    start_time = time.time()
    
    for i, r in enumerate(configs_to_test):
        params = {
            'lam': r['lam'],
            'beta': r['beta'],
            'alpha': r['alpha'],
            'sigma': r['sigma'],
            'gamma_adapt': r['gamma'],
            'kappa': r['kappa'],
        }
        
        dissoc = test_dissociation_transient(
            params, 
            G_fracs=[0.25, 0.50, 0.75, 0.90],
            n_seeds=30,
            T=8000,
            switch_time=5000,
            G_duration=500,
            check_window=300
        )
        
        result_entry = {
            'alpha': r['alpha'],
            'beta': r['beta'],
            'kappa': r['kappa'],
            'kappa_frac': r['kappa_frac'],
            'cv_raw': r['durations_raw'].get('cv'),
            'cv_in_range': r.get('cv_in_range', False),
            'levelt_holds': r.get('levelt', {}).get('holds', False),
            'levelt_rho': r.get('levelt', {}).get('rho'),
            'dissociation_retest': dissoc
        }
        
        results.append(result_entry)
        
        # Progress
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(configs_to_test) - i - 1) / rate if rate > 0 else 0
        
        bd = dissoc['best_dissoc']
        status = (f"best_G={dissoc['best_G']} "
                  f"dorsal={bd['dorsal']:.0%} ventral={bd['ventral']:.0%} "
                  f"diff={bd['difference']:.2f}")
        
        if bd['good_strict']:
            status += " ★STRICT★"
        elif bd['good_moderate']:
            status += " ◆moderate◆"
        
        cv_str = f"CV={r['durations_raw'].get('cv', 0):.3f}" if r['durations_raw'].get('cv') else "CV=?"
        
        print(f"[{i+1:>4}/{len(configs_to_test)}] "
              f"α={r['alpha']:.3f} β={r['beta']:.3f} κ={r['kappa']:.4f} | "
              f"{cv_str} | {status}"
              f"  [{elapsed:.0f}s, ~{eta:.0f}s left]")
    
    total_time = time.time() - start_time
    
    # ═══════════════════════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════════════════════
    
    output = {
        'metadata': {
            'description': 'GC-LCA dissociation retest with transient G protocol',
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_configs': len(configs_to_test),
            'total_time_seconds': total_time,
            'protocol': {
                'T': 8000,
                'switch_time': 5000,
                'G_duration': 500,
                'check_window': 300,
                'consecutive_required': 50,
                'switch_threshold': 0.05,
                'n_seeds': 30,
                'G_fracs': [0.25, 0.50, 0.75, 0.90],
                'dorsal_scaling': {'alpha': 1.5, 'beta': 0.6, 'kappa': 1.5},
                'ventral_scaling': {'alpha': 0.4, 'beta': 1.5, 'kappa': 0.4},
            }
        },
        'results': results
    }
    
    out_path = 'dissociation_retest_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print(f"DISSOCIATION RETEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total configs tested: {len(results)}")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    
    # Count by G level
    for G_frac in [0.25, 0.50, 0.75, 0.90]:
        key = f'G{int(G_frac*100)}'
        strict = sum(1 for r in results 
                     if r['dissociation_retest']['dissociation'][key]['good_strict'])
        moderate = sum(1 for r in results 
                       if r['dissociation_retest']['dissociation'][key]['good_moderate'])
        
        # With CV and Levelt
        strict_all = sum(1 for r in results 
                         if r['dissociation_retest']['dissociation'][key]['good_strict']
                         and r.get('cv_in_range') and r.get('levelt_holds'))
        moderate_all = sum(1 for r in results 
                          if r['dissociation_retest']['dissociation'][key]['good_moderate']
                          and r.get('cv_in_range') and r.get('levelt_holds'))
        
        print(f"\n  {key} (G = {G_frac*100:.0f}%λ):")
        print(f"    Strict dissoc (dorsal≥50%, ventral≤20%): {strict}")
        print(f"    Moderate dissoc (diff > 0.3):             {moderate}")
        print(f"    ★ Strict + CV + Levelt:                   {strict_all}")
        print(f"    ◆ Moderate + CV + Levelt:                 {moderate_all}")
    
    # Best overall configs
    print(f"\n--- BEST OVERALL CONFIGS (all three criteria, strict) ---")
    best_configs = [r for r in results 
                    if r.get('cv_in_range') and r.get('levelt_holds')
                    and r['dissociation_retest']['best_dissoc']['good_strict']]
    
    if best_configs:
        for r in sorted(best_configs, key=lambda x: -x['dissociation_retest']['best_dissoc']['difference']):
            bd = r['dissociation_retest']['best_dissoc']
            print(f"  α={r['alpha']:.3f} β={r['beta']:.3f} κ={r['kappa']:.4f} "
                  f"| CV={r['cv_raw']:.3f} | ρ={r['levelt_rho']:.2f} "
                  f"| best_G={r['dissociation_retest']['best_G']} "
                  f"dorsal={bd['dorsal']:.0%} ventral={bd['ventral']:.0%}")
    else:
        print("  None found with strict criteria.")
        
        # Try moderate
        print(f"\n--- BEST OVERALL CONFIGS (moderate dissoc) ---")
        mod_configs = [r for r in results 
                       if r.get('cv_in_range') and r.get('levelt_holds')
                       and r['dissociation_retest']['best_dissoc']['good_moderate']]
        
        if mod_configs:
            for r in sorted(mod_configs, key=lambda x: -x['dissociation_retest']['best_dissoc']['difference'])[:15]:
                bd = r['dissociation_retest']['best_dissoc']
                print(f"  α={r['alpha']:.3f} β={r['beta']:.3f} κ={r['kappa']:.4f} "
                      f"| CV={r['cv_raw']:.3f} | ρ={r['levelt_rho']:.2f} "
                      f"| best_G={r['dissociation_retest']['best_G']} "
                      f"dorsal={bd['dorsal']:.0%} ventral={bd['ventral']:.0%} "
                      f"diff={bd['difference']:.2f}")
        else:
            print("  None found with moderate criteria either.")
            
            # Show best dissociation among CV-good configs regardless
            print(f"\n--- TOP 10 DISSOCIATION AMONG CV-GOOD CONFIGS ---")
            cv_good = [r for r in results if r.get('cv_in_range') and r.get('levelt_holds')]
            if cv_good:
                for r in sorted(cv_good, key=lambda x: -x['dissociation_retest']['best_dissoc']['difference'])[:10]:
                    bd = r['dissociation_retest']['best_dissoc']
                    print(f"  α={r['alpha']:.3f} β={r['beta']:.3f} κ={r['kappa']:.4f} "
                          f"| CV={r['cv_raw']:.3f} "
                          f"| best_G={r['dissociation_retest']['best_G']} "
                          f"dorsal={bd['dorsal']:.0%} ventral={bd['ventral']:.0%} "
                          f"diff={bd['difference']:.2f}")
    
    print(f"\nResults saved: {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")
    print(f"Summary saved: dissociation_retest_summary.txt")
    
    # Save summary
    # (redirect the print to file is clunky, just save key stats)
    with open('dissociation_retest_summary.txt', 'w') as f:
        f.write(f"Dissociation Retest Summary\n")
        f.write(f"Protocol: Transient G, 500 steps at t=5000, 30 seeds\n")
        f.write(f"Configs: {len(results)}\n")
        f.write(f"Time: {total_time:.0f}s\n\n")
        
        for G_frac in [0.25, 0.50, 0.75, 0.90]:
            key = f'G{int(G_frac*100)}'
            strict_all = sum(1 for r in results 
                             if r['dissociation_retest']['dissociation'][key]['good_strict']
                             and r.get('cv_in_range') and r.get('levelt_holds'))
            moderate_all = sum(1 for r in results 
                              if r['dissociation_retest']['dissociation'][key]['good_moderate']
                              and r.get('cv_in_range') and r.get('levelt_holds'))
            f.write(f"{key}: strict+CV+Levelt={strict_all}, moderate+CV+Levelt={moderate_all}\n")


if __name__ == '__main__':
    main()
