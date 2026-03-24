#!/usr/bin/env python3
"""
Paper 1 Mechanistic Analysis — Phase Portraits, Sensitivity, κ Role
====================================================================
Crewther Sampler Research Programme v2.2

B.1.1: Phase portrait / nullcline analysis (deterministic system)
B.1.2: Sensitivity analysis (criteria tightening + stability radius)
B.1.3: Transitional vs sustained episode characterisation
B.1.4: κ role analysis

Inputs: phase1_grid_results.json, phase2_dissociation_results.json
Outputs: paper1_mechanistic_results.json, fig7_phase_portrait.pdf,
         fig8_sensitivity.pdf, fig9_kappa_role.pdf

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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib import cm
from scipy.stats import weibull_min, gamma as gamma_dist
from scipy.optimize import fsolve

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})

C_BLUE = '#2171B5'
C_RED = '#CB181D'
C_GREEN = '#238B45'
C_ORANGE = '#D94801'
C_PURPLE = '#6A51A3'
C_GREY = '#737373'

BURN_IN = 500
MARGIN = 0.05
X_MAX = 5.0
G_SAFETY = 0.95

# Representative configs from the overlap region
# Low-α/high-β edge, mid/mid, high-α/low-β edge
REPRESENTATIVE_CONFIGS = [
    # (label, λ, β, α, σ, γ, κ) — from the top-30
    ('Low α, high β', 0.15, 0.30, 0.05, 0.12, 0.02, 0.05),
    ('Mid α, mid β',  0.15, 0.25, 0.08, 0.10, 0.03, 0.04),
    ('High α, low β', 0.08, 0.10, 0.05, 0.04, 0.02, 0.0375),
]

DORSAL_MULTS = {'alpha': 1.5, 'beta': 0.6, 'kappa': 1.5}
VENTRAL_MULTS = {'alpha': 0.4, 'beta': 1.5, 'kappa': 0.4}


# =============================================================================
# SIMULATION KERNEL
# =============================================================================

@njit(cache=True)
def simulate_gclca(lam, beta, alpha, sigma, gamma_adapt, kappa,
                   signal_a, signal_b, g_a, g_b, n_steps, seed, x_max=5.0):
    np.random.seed(seed)
    trace_a = np.empty(n_steps, dtype=np.float64)
    trace_b = np.empty(n_steps, dtype=np.float64)
    x_a, x_b, a_a, a_b = 0.1, 0.1, 0.0, 0.0
    for t in range(n_steps):
        trace_a[t] = x_a
        trace_b[t] = x_b
        eta_a = np.random.normal(0.0, sigma)
        eta_b = np.random.normal(0.0, sigma)
        new_a = (1.0 - lam + g_a) * x_a + signal_a - beta * x_b - alpha * a_a + eta_a
        new_b = (1.0 - lam + g_b) * x_b + signal_b - beta * x_a - alpha * a_b + eta_b
        x_a = min(max(0.0, new_a), x_max)
        x_b = min(max(0.0, new_b), x_max)
        a_a = (1.0 - gamma_adapt) * a_a + kappa * x_a
        a_b = (1.0 - gamma_adapt) * a_b + kappa * x_b
    return trace_a, trace_b


@njit(cache=True)
def extract_durations(trace_a, trace_b, burn_in, margin, min_dur=0):
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
            if current_channel >= 0 and current_duration >= max(1, min_dur):
                channels[n_episodes] = current_channel
                durations[n_episodes] = current_duration
                n_episodes += 1
            current_channel = dom
            current_duration = 1 if dom >= 0 else 0
    if current_channel >= 0 and current_duration >= max(1, min_dur):
        channels[n_episodes] = current_channel
        durations[n_episodes] = current_duration
        n_episodes += 1
    return channels[:n_episodes], durations[:n_episodes]


# =============================================================================
# B.1.1: PHASE PORTRAIT / NULLCLINE ANALYSIS
# =============================================================================

def compute_nullclines(lam, beta, alpha, gamma_adapt, kappa, signal, g_a=0.0, g_b=0.0):
    """
    Compute nullclines for the deterministic GC-LCA (σ=0).

    At steady state for adaptation: a_i = (κ/γ) * x_i

    The effective adaptation coefficient becomes α_eff = α * κ / γ

    x_A nullcline (dx_A/dt = 0):
        (1 - λ + G_A) * x_A + S_A - β * x_B - α_eff * x_A = 0
        => x_B = [(1 - λ + G_A - α_eff) * x_A + S_A] / β

    x_B nullcline (dx_B/dt = 0):
        (1 - λ + G_B) * x_B + S_B - β * x_A - α_eff * x_B = 0
        => x_A = [(1 - λ + G_B - α_eff) * x_B + S_B] / β

    But we also need to account for the rectifier: x_i = max(0, ·)
    """
    alpha_eff = alpha * kappa / gamma_adapt if gamma_adapt > 0 else 0

    x_range = np.linspace(0, 3.0, 500)

    # x_A nullcline: x_B as function of x_A
    # From dx_A/dt = 0: -(λ - G_A)x_A + S_A - β·x_B - α_eff·x_A = 0
    # Rearranging: x_B = [S_A - (λ - G_A + α_eff)·x_A] / β
    coeff_a = lam - g_a + alpha_eff  # net decay for channel A
    nullcline_a_xb = (signal - coeff_a * x_range) / beta  # x_B values on A-nullcline

    # x_B nullcline: x_A as function of x_B
    coeff_b = lam - g_b + alpha_eff
    nullcline_b_xa = (signal - coeff_b * x_range) / beta  # x_A values on B-nullcline

    # Clip to non-negative (rectifier)
    nullcline_a_xb = np.clip(nullcline_a_xb, 0, None)
    nullcline_b_xa = np.clip(nullcline_b_xa, 0, None)

    return x_range, nullcline_a_xb, nullcline_b_xa, alpha_eff


def find_fixed_points(lam, beta, alpha, gamma_adapt, kappa, signal, g_a=0.0, g_b=0.0):
    """Find fixed points of the deterministic system."""
    alpha_eff = alpha * kappa / gamma_adapt if gamma_adapt > 0 else 0

    fixed_points = []

    # Symmetric fixed point (both active): x_A = x_B = x*
    # (λ - G + α_eff + β) * x* = S
    # For symmetric case (G_A = G_B = 0): x* = S / (λ + α_eff + β)
    coeff_sym = lam + alpha_eff + beta
    if coeff_sym > 0:
        x_sym = signal / coeff_sym
        fixed_points.append(('symmetric', x_sym, x_sym))

    # Asymmetric fixed points (winner-take-all with rectifier)
    # If x_B = 0: (λ - G_A + α_eff) * x_A = S_A  =>  x_A = S / (λ - G_A + α_eff)
    coeff_a_only = lam - g_a + alpha_eff
    if coeff_a_only > 0:
        x_a_wta = signal / coeff_a_only
        # Check that x_B would be driven below 0
        x_b_test = signal - beta * x_a_wta
        if x_b_test <= 0:
            fixed_points.append(('A-dominant', x_a_wta, 0.0))

    coeff_b_only = lam - g_b + alpha_eff
    if coeff_b_only > 0:
        x_b_wta = signal / coeff_b_only
        x_a_test = signal - beta * x_b_wta
        if x_a_test <= 0:
            fixed_points.append(('B-dominant', 0.0, x_b_wta))

    # With G on channel A: asymmetric interior fixed point
    if g_a > 0 and g_b == 0:
        # x_A nullcline: x_B = [S - (λ - G_A + α_eff)·x_A] / β
        # x_B nullcline: x_A = [S - (λ + α_eff)·x_B] / β
        # Substituting:
        ca = lam - g_a + alpha_eff
        cb = lam + alpha_eff
        # β·x_B = S - ca·x_A  and  β·x_A = S - cb·x_B
        # => β²·x_A = β·S - cb·(S - ca·x_A)
        # => β²·x_A = β·S - cb·S + cb·ca·x_A
        # => (β² - cb·ca)·x_A = S·(β - cb)
        denom = beta**2 - cb * ca
        if abs(denom) > 1e-10:
            x_a_asym = signal * (beta - cb) / denom
            x_b_asym = (signal - ca * x_a_asym) / beta
            if x_a_asym >= 0 and x_b_asym >= 0:
                fixed_points.append(('G-biased', x_a_asym, x_b_asym))

    return fixed_points


def compute_jacobian_eigenvalues(lam, beta, alpha, gamma_adapt, kappa, signal,
                                 x_a, x_b, g_a=0.0, g_b=0.0):
    """
    Compute eigenvalues of the Jacobian at a fixed point.
    Using the 2D reduction (adaptation at steady state).
    """
    alpha_eff = alpha * kappa / gamma_adapt if gamma_adapt > 0 else 0

    # Jacobian of the 2D reduced system (both channels active):
    # dx_A/dt ≈ -(λ - G_A + α_eff)·x_A + S - β·x_B
    # dx_B/dt ≈ -(λ - G_B + α_eff)·x_B + S - β·x_A
    # But need to account for rectifier: if x_i = 0, the row becomes trivial

    if x_a > 0 and x_b > 0:
        # Both active
        J = np.array([
            [-(lam - g_a + alpha_eff), -beta],
            [-beta, -(lam - g_b + alpha_eff)]
        ])
    elif x_a > 0 and x_b == 0:
        # A active, B at rectifier floor
        J = np.array([
            [-(lam - g_a + alpha_eff), -beta],
            [0, 0]  # B is clamped at 0 — no dynamics
        ])
    elif x_a == 0 and x_b > 0:
        J = np.array([
            [0, 0],
            [-beta, -(lam - g_b + alpha_eff)]
        ])
    else:
        return np.array([0.0, 0.0])

    eigenvalues = np.linalg.eigvals(J)
    return eigenvalues


def compute_vector_field(lam, beta, alpha, gamma_adapt, kappa, signal,
                         g_a=0.0, g_b=0.0, x_range=(0, 2.5), n_points=20):
    """Compute the vector field for the 2D reduced system."""
    alpha_eff = alpha * kappa / gamma_adapt if gamma_adapt > 0 else 0

    xa_vals = np.linspace(x_range[0], x_range[1], n_points)
    xb_vals = np.linspace(x_range[0], x_range[1], n_points)
    XA, XB = np.meshgrid(xa_vals, xb_vals)

    # dx_A/dt = -(λ - G_A + α_eff)·x_A + S - β·x_B  (if x_A > 0)
    # dx_B/dt = -(λ - G_B + α_eff)·x_B + S - β·x_A  (if x_B > 0)
    DXA = -(lam - g_a + alpha_eff) * XA + signal - beta * XB
    DXB = -(lam - g_b + alpha_eff) * XB + signal - beta * XA

    # Apply rectifier: if x would go below 0, clip derivative
    DXA[XA <= 0.01] = np.maximum(DXA[XA <= 0.01], 0)
    DXB[XB <= 0.01] = np.maximum(DXB[XB <= 0.01], 0)

    # Normalise for plotting
    magnitude = np.sqrt(DXA**2 + DXB**2)
    magnitude[magnitude == 0] = 1
    DXA_norm = DXA / magnitude
    DXB_norm = DXB / magnitude

    return XA, XB, DXA_norm, DXB_norm, magnitude


def b11_phase_portraits():
    """
    Generate phase portrait analysis for 3 representative configs.
    Show nullclines, fixed points, vector field, and effect of G.
    """
    print("\n" + "="*70)
    print("B.1.1: Phase Portrait / Nullcline Analysis")
    print("="*70)

    results = []

    # Figure 7: Phase portraits (3 configs × 3 conditions = 9 panels)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for row, (label, lam, beta, alpha, sigma, gamma_adapt, kappa) in enumerate(REPRESENTATIVE_CONFIGS):
        alpha_eff = alpha * kappa / gamma_adapt

        config_result = {
            'label': label,
            'params': {'lambda': lam, 'beta': beta, 'alpha': alpha,
                       'gamma': gamma_adapt, 'kappa': kappa},
            'alpha_eff': round(alpha_eff, 6),
            'conditions': {}
        }

        for col, (cond_label, g_a, regime) in enumerate([
            ('Baseline (G=0)', 0.0, 'base'),
            ('Dorsal + G=50%λ', 0.5 * lam * G_SAFETY, 'dorsal'),
            ('Ventral + G=50%λ', 0.5 * lam * G_SAFETY, 'ventral'),
        ]):
            ax = axes[row, col]

            # Apply regime multipliers for dorsal/ventral
            if regime == 'dorsal':
                a_r = alpha * DORSAL_MULTS['alpha']
                b_r = beta * DORSAL_MULTS['beta']
                k_r = kappa * DORSAL_MULTS['kappa']
            elif regime == 'ventral':
                a_r = alpha * VENTRAL_MULTS['alpha']
                b_r = beta * VENTRAL_MULTS['beta']
                k_r = kappa * VENTRAL_MULTS['kappa']
            else:
                a_r, b_r, k_r = alpha, beta, kappa

            g_val = g_a if regime != 'base' else 0.0

            # Nullclines
            x_range, nc_a_xb, nc_b_xa, a_eff = compute_nullclines(
                lam, b_r, a_r, gamma_adapt, k_r, 0.5, g_a=g_val, g_b=0.0)

            ax.plot(x_range, nc_a_xb, color=C_BLUE, linewidth=1.5,
                    label='$\\dot{x}_A=0$' if row == 0 and col == 0 else '')
            ax.plot(nc_b_xa, x_range, color=C_RED, linewidth=1.5,
                    label='$\\dot{x}_B=0$' if row == 0 and col == 0 else '')

            # Also plot baseline nullclines as dotted reference for G conditions
            if regime != 'base':
                _, nc_a_base, nc_b_base, _ = compute_nullclines(
                    lam, b_r, a_r, gamma_adapt, k_r, 0.5, g_a=0.0, g_b=0.0)
                ax.plot(x_range, nc_a_base, color=C_BLUE, linewidth=0.8,
                        linestyle=':', alpha=0.4)
                ax.plot(nc_b_base, x_range, color=C_RED, linewidth=0.8,
                        linestyle=':', alpha=0.4)

            # Vector field
            XA, XB, DXA, DXB, mag = compute_vector_field(
                lam, b_r, a_r, gamma_adapt, k_r, 0.5,
                g_a=g_val, g_b=0.0, x_range=(0, 2.5), n_points=15)
            ax.quiver(XA, XB, DXA, DXB, mag, cmap='Greys', alpha=0.3,
                      scale=30, width=0.004)

            # Fixed points
            fps = find_fixed_points(lam, b_r, a_r, gamma_adapt, k_r, 0.5,
                                    g_a=g_val, g_b=0.0)
            fp_data = []
            for fp_type, xa, xb in fps:
                eigs = compute_jacobian_eigenvalues(
                    lam, b_r, a_r, gamma_adapt, k_r, 0.5, xa, xb,
                    g_a=g_val, g_b=0.0)
                # Classify: stable if all real parts negative
                real_parts = np.real(eigs)
                is_stable = np.all(real_parts < 0)
                has_imaginary = np.any(np.abs(np.imag(eigs)) > 1e-6)

                marker = 'o' if is_stable else 'x'
                color = C_GREEN if is_stable else C_ORANGE
                ax.plot(xa, xb, marker, color=color, markersize=8,
                        markeredgewidth=2, zorder=5)

                fp_data.append({
                    'type': fp_type, 'x_a': round(xa, 4), 'x_b': round(xb, 4),
                    'eigenvalues_real': [round(r, 6) for r in real_parts],
                    'eigenvalues_imag': [round(float(i), 6) for i in np.imag(eigs)],
                    'stable': bool(is_stable),
                    'oscillatory': bool(has_imaginary),
                })

            config_result['conditions'][regime] = {
                'g_value': round(g_val, 6),
                'alpha_regime': round(a_r, 6),
                'beta_regime': round(b_r, 6),
                'kappa_regime': round(k_r, 6),
                'alpha_eff': round(a_r * k_r / gamma_adapt, 6),
                'fixed_points': fp_data,
            }

            # Diagonal (symmetry line)
            ax.plot([0, 2.5], [0, 2.5], '--', color=C_GREY, linewidth=0.5, alpha=0.3)

            # Rectifier floor annotation
            ax.axhline(0, color='black', linewidth=0.3, alpha=0.3)
            ax.axvline(0, color='black', linewidth=0.3, alpha=0.3)

            ax.set_xlim(0, 2.5)
            ax.set_ylim(0, 2.5)
            ax.set_aspect('equal')

            if col == 0:
                ax.set_ylabel(f'{label}\n$x_B$', fontsize=8)
            if row == 2:
                ax.set_xlabel('$x_A$')
            if row == 0:
                ax.set_title(cond_label, fontsize=9, fontweight='bold')

            # Annotate fixed point stability
            for fp in fp_data:
                stability = 'stable' if fp['stable'] else 'unstable'
                osc = ', osc' if fp['oscillatory'] else ''
                ax.annotate(f"{fp['type']}\n({stability}{osc})",
                           (fp['x_a'], fp['x_b']),
                           textcoords="offset points", xytext=(10, 5),
                           fontsize=5, color=C_GREY)

        results.append(config_result)

        # Print summary
        print(f"\n  {label} (α_eff = {alpha_eff:.4f}):")
        for regime, cond in config_result['conditions'].items():
            n_stable = sum(1 for fp in cond['fixed_points'] if fp['stable'])
            n_fp = len(cond['fixed_points'])
            fp_types = [fp['type'] for fp in cond['fixed_points']]
            print(f"    {regime}: {n_fp} FPs ({n_stable} stable) — {fp_types}")

    # Add legend to top-left panel
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=C_BLUE, linewidth=1.5, label='$\\dot{x}_A=0$ nullcline'),
        Line2D([0], [0], color=C_RED, linewidth=1.5, label='$\\dot{x}_B=0$ nullcline'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_GREEN,
               markersize=8, label='Stable FP'),
        Line2D([0], [0], marker='x', color=C_ORANGE, markersize=8,
               markeredgewidth=2, label='Unstable FP'),
    ]
    axes[0, 0].legend(handles=legend_elements, fontsize=6, loc='upper right')

    fig.tight_layout()
    fig.savefig('fig7_phase_portrait.pdf')
    plt.close()
    print("  Saved: fig7_phase_portrait.pdf")

    # --- Bifurcation analysis: G vs fixed point position ---
    print("\n  Bifurcation analysis (G sweep)...")
    fig_bif, axes_bif = plt.subplots(1, 3, figsize=(10, 3.5))

    for idx, (label, lam, beta, alpha, sigma, gamma_adapt, kappa) in enumerate(REPRESENTATIVE_CONFIGS):
        ax = axes_bif[idx]

        for regime_name, mults, color, lstyle in [
            ('Dorsal', DORSAL_MULTS, C_BLUE, '-'),
            ('Ventral', VENTRAL_MULTS, C_RED, '--'),
        ]:
            a_r = alpha * mults['alpha']
            b_r = beta * mults['beta']
            k_r = kappa * mults['kappa']

            g_fracs = np.linspace(0, 0.9, 50)
            sym_xa = []
            dom_xa = []

            for gf in g_fracs:
                g_val = gf * lam * G_SAFETY
                fps = find_fixed_points(lam, b_r, a_r, gamma_adapt, k_r, 0.5,
                                        g_a=g_val, g_b=0.0)
                # Track the symmetric and A-dominant fixed points
                for fp_type, xa, xb in fps:
                    if fp_type == 'symmetric' or fp_type == 'G-biased':
                        sym_xa.append((gf, xa))
                    elif fp_type == 'A-dominant':
                        dom_xa.append((gf, xa))

            if sym_xa:
                gs, xs = zip(*sym_xa)
                ax.plot(gs, xs, color=color, linewidth=1.5, linestyle=lstyle,
                        label=f'{regime_name} (interior)')
            if dom_xa:
                gs, xs = zip(*dom_xa)
                ax.plot(gs, xs, color=color, linewidth=1, linestyle=':',
                        alpha=0.5, label=f'{regime_name} (WTA)')

        ax.set_xlabel('G (fraction of λ)')
        ax.set_ylabel('$x_A$ at fixed point')
        ax.set_title(label, fontsize=9, fontweight='bold')
        if idx == 0:
            ax.legend(fontsize=6)

    fig_bif.tight_layout()
    fig_bif.savefig('fig7b_bifurcation.pdf')
    plt.close()
    print("  Saved: fig7b_bifurcation.pdf")

    return results


# =============================================================================
# B.1.2: SENSITIVITY ANALYSIS
# =============================================================================

def b12_sensitivity(phase1_data, phase2_data):
    """
    Tighten criteria and check stability radius.
    """
    print("\n" + "="*70)
    print("B.1.2: Sensitivity Analysis")
    print("="*70)

    results_p1 = phase1_data['results']
    cr = phase2_data['config_results']
    ds = phase2_data['dissociation_summary']

    # --- Criteria tightening ---
    criteria_sets = [
        ('Pre-registered', 0.35, 0.65, 0.50, 0.20),
        ('Tightened CV', 0.40, 0.60, 0.50, 0.20),
        ('Tightened ventral', 0.35, 0.65, 0.50, 0.10),
        ('Tightened dorsal', 0.35, 0.65, 0.60, 0.20),
        ('All tightened', 0.40, 0.60, 0.60, 0.10),
    ]

    sensitivity_results = []
    print("\n  Criteria tightening (Phase I eligible × Phase II dissociation):")

    for label, cv_lo, cv_hi, dorsal_min, ventral_max in criteria_sets:
        # Phase I eligible
        eligible = [r for r in results_p1
                    if r.get('rivalry_producing', False)
                    and r.get('levelt_rho') is not None and r['levelt_rho'] > 0.7
                    and r.get('rivalry_metrics') and r['rivalry_metrics'].get('cv') is not None
                    and cv_lo <= r['rivalry_metrics']['cv'] <= cv_hi]

        # Phase II strict from dissociation summary
        strict = sum(1 for d in ds
                     if d['best_dorsal'] >= dorsal_min and d['best_ventral'] <= ventral_max)

        sensitivity_results.append({
            'label': label,
            'cv_range': [cv_lo, cv_hi],
            'dorsal_min': dorsal_min,
            'ventral_max': ventral_max,
            'phase1_eligible': len(eligible),
            'phase2_strict': strict,
        })
        print(f"    {label}: Phase I eligible={len(eligible)}, Phase II strict={strict}/30")

    # --- Stability radius ---
    # For each strict-passing config, check neighbouring configs in Phase I
    print("\n  Stability radius check:")
    strict_configs = [d for d in ds if d.get('strict', False)]

    stability_results = []
    for d in strict_configs:
        ci = d['config']
        c = cr[ci]
        bp = c['base_params']

        # Find Phase I configs within ±1 grid step of this config's α, β
        alpha_step = 0.02  # approximate grid step for α
        beta_step = 0.05   # approximate grid step for β

        neighbours = [r for r in results_p1
                      if abs(r['alpha'] - bp['alpha']) <= alpha_step * 1.5
                      and abs(r['beta'] - bp['beta']) <= beta_step * 1.5
                      and r.get('rivalry_producing', False)
                      and r.get('rivalry_metrics') and r['rivalry_metrics'].get('cv') is not None]

        n_in_cv = sum(1 for r in neighbours
                      if 0.35 <= r['rivalry_metrics']['cv'] <= 0.65)
        n_total = len(neighbours)

        stability_results.append({
            'config': ci,
            'base_alpha': bp['alpha'],
            'base_beta': bp['beta'],
            'n_neighbours': n_total,
            'n_in_cv_target': n_in_cv,
            'fraction_stable': round(n_in_cv / n_total, 3) if n_total > 0 else 0,
        })
        print(f"    Config {ci} (α={bp['alpha']}, β={bp['beta']}): "
              f"{n_in_cv}/{n_total} neighbours in CV target ({n_in_cv/n_total*100:.0f}%)")

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))

    # Panel A: Criteria tightening
    ax = axes[0]
    labels = [r['label'] for r in sensitivity_results]
    p1_counts = [r['phase1_eligible'] for r in sensitivity_results]
    p2_counts = [r['phase2_strict'] for r in sensitivity_results]
    x = np.arange(len(labels))
    ax.bar(x - 0.15, p1_counts, 0.3, label='Phase I eligible', color=C_BLUE, alpha=0.7)
    ax.bar(x + 0.15, p2_counts, 0.3, label='Phase II strict', color=C_RED, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
    ax.set_ylabel('Count')
    ax.set_title('A  Criteria tightening', loc='left', fontweight='bold', fontsize=9)
    ax.legend(fontsize=7)

    # Panel B: Stability radius
    ax = axes[1]
    configs = [r['config'] for r in stability_results]
    fracs = [r['fraction_stable'] for r in stability_results]
    ax.bar(range(len(configs)), fracs, color=C_GREEN, alpha=0.8)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([str(c) for c in configs], fontsize=7)
    ax.set_xlabel('Strict-passing config')
    ax.set_ylabel('Fraction of neighbours\nin CV target')
    ax.set_title('B  Stability radius', loc='left', fontweight='bold', fontsize=9)
    ax.axhline(0.5, color=C_GREY, linestyle='--', linewidth=0.8, alpha=0.5)

    fig.tight_layout()
    fig.savefig('fig8_sensitivity.pdf')
    plt.close()
    print("  Saved: fig8_sensitivity.pdf")

    return {'criteria_tightening': sensitivity_results, 'stability_radius': stability_results}


# =============================================================================
# B.1.3: TRANSITIONAL VS SUSTAINED EPISODES
# =============================================================================

def b13_transitional_states():
    """
    Compare Weibull shape for short (transitional) vs long (sustained) episodes.
    """
    print("\n" + "="*70)
    print("B.1.3: Transitional vs Sustained Episodes")
    print("="*70)

    # Use mid config
    lam, beta, alpha, sigma, gamma_adapt, kappa = 0.15, 0.25, 0.08, 0.10, 0.03, 0.04

    all_durations = []
    for seed in range(50):
        ta, tb = simulate_gclca(lam, beta, alpha, sigma, gamma_adapt, kappa,
                                0.5, 0.5, 0.0, 0.0, 20000, seed)
        ch, dur = extract_durations(ta, tb, BURN_IN, MARGIN, 0)
        if len(ch) > 2:
            all_durations.extend(dur[1:-1].tolist())  # boundary exclusion

    d = np.array(all_durations, dtype=float)
    print(f"  Total episodes: {len(d)}")

    # Split at threshold = 5 (a common short-episode boundary)
    threshold = 5
    short = d[d <= threshold]
    long = d[d > threshold]
    print(f"  Short (≤{threshold}): {len(short)} ({100*len(short)/len(d):.1f}%)")
    print(f"  Long (>{threshold}): {len(long)} ({100*len(long)/len(d):.1f}%)")

    results = {'threshold': threshold, 'n_total': len(d)}

    for label, subset in [('short', short), ('long', long), ('all', d)]:
        if len(subset) < 10:
            results[label] = None
            continue
        try:
            wc, _, ws = weibull_min.fit(subset, floc=0)
            cv = float(np.std(subset, ddof=1) / np.mean(subset))
            results[label] = {
                'n': len(subset),
                'mean': float(np.mean(subset)),
                'cv': cv,
                'weibull_shape': float(wc),
                'weibull_scale': float(ws),
            }
            print(f"  {label}: mean={np.mean(subset):.1f}, CV={cv:.3f}, "
                  f"Weibull shape={wc:.2f}")
        except Exception:
            results[label] = None

    return results


# =============================================================================
# B.1.4: κ ROLE ANALYSIS
# =============================================================================

def b14_kappa_role():
    """
    Characterise κ's functional role:
    - CV vs κ (holding α, β fixed)
    - Dissociation vs κ
    - Switch rate vs κ
    """
    print("\n" + "="*70)
    print("B.1.4: κ Role Analysis")
    print("="*70)

    # Fixed params (mid-range)
    lam = 0.15
    beta = 0.25
    alpha = 0.08
    sigma = 0.10
    gamma_adapt = 0.03

    kappa_values = np.round(np.linspace(0.01, 0.20, 20), 4)
    N_SEEDS = 20
    N_STEPS = 20000

    kappa_results = []

    for kappa in kappa_values:
        # Rivalry CV
        cvs = []
        for seed in range(N_SEEDS):
            ta, tb = simulate_gclca(lam, beta, alpha, sigma, gamma_adapt, kappa,
                                    0.5, 0.5, 0.0, 0.0, N_STEPS, seed)
            ch, dur = extract_durations(ta, tb, BURN_IN, MARGIN)
            if len(ch) > 4:
                d = dur[1:-1].astype(float)
                if len(d) > 2 and np.mean(d) > 0:
                    cvs.append(np.std(d, ddof=1) / np.mean(d))

        cv_mean = float(np.mean(cvs)) if cvs else None

        # Dissociation test (simplified: dorsal switch rate at G=50%λ)
        g_val = 0.5 * lam * G_SAFETY
        dorsal_switches = 0
        ventral_switches = 0
        n_trials = 20

        for seed in range(n_trials):
            # Dorsal regime
            a_d = alpha * DORSAL_MULTS['alpha']
            b_d = beta * DORSAL_MULTS['beta']
            k_d = kappa * DORSAL_MULTS['kappa']
            ta, tb = simulate_gclca(lam, b_d, a_d, sigma, gamma_adapt, k_d,
                                    0.5, 0.5, g_val, 0.0, 6000, seed + 50000)
            # Check switch in window 5000-5800
            consec = 0
            for t in range(5000, min(5800, len(ta))):
                if ta[t] - tb[t] > MARGIN:
                    consec += 1
                else:
                    consec = 0
                if consec >= 50:
                    dorsal_switches += 1
                    break

            # Ventral regime
            a_v = alpha * VENTRAL_MULTS['alpha']
            b_v = beta * VENTRAL_MULTS['beta']
            k_v = kappa * VENTRAL_MULTS['kappa']
            ta, tb = simulate_gclca(lam, b_v, a_v, sigma, gamma_adapt, k_v,
                                    0.5, 0.5, g_val, 0.0, 6000, seed + 50000)
            consec = 0
            for t in range(5000, min(5800, len(ta))):
                if ta[t] - tb[t] > MARGIN:
                    consec += 1
                else:
                    consec = 0
                if consec >= 50:
                    ventral_switches += 1
                    break

        kappa_results.append({
            'kappa': float(kappa),
            'cv_mean': cv_mean,
            'dorsal_rate': dorsal_switches / n_trials,
            'ventral_rate': ventral_switches / n_trials,
            'dissociation': (dorsal_switches - ventral_switches) / n_trials,
        })
        cv_str = f"{cv_mean:.3f}" if cv_mean is not None else "N/A"
        print(f"  κ={kappa:.3f}: CV={cv_str:>6}, "
              f"dorsal={dorsal_switches/n_trials:.2f}, ventral={ventral_switches/n_trials:.2f}")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

    kappas = [r['kappa'] for r in kappa_results]
    cvs = [r['cv_mean'] if r['cv_mean'] else 0 for r in kappa_results]
    dorsal = [r['dorsal_rate'] for r in kappa_results]
    ventral = [r['ventral_rate'] for r in kappa_results]
    dissoc = [r['dissociation'] for r in kappa_results]

    # Panel A: CV vs κ
    ax = axes[0]
    ax.plot(kappas, cvs, 'o-', color=C_BLUE, markersize=4, linewidth=1.2)
    ax.axhspan(0.40, 0.60, alpha=0.1, color=C_GREEN, label='Target CV [0.4, 0.6]')
    ax.axhspan(0.35, 0.65, alpha=0.05, color=C_GREEN)
    ax.set_xlabel('κ (adaptation gain)')
    ax.set_ylabel('CV')
    ax.set_title('A  CV vs κ', loc='left', fontweight='bold', fontsize=9)
    ax.legend(fontsize=7)

    # Panel B: Switch rates vs κ
    ax = axes[1]
    ax.plot(kappas, dorsal, 'o-', color=C_BLUE, markersize=4, linewidth=1.2, label='Dorsal')
    ax.plot(kappas, ventral, 's-', color=C_RED, markersize=4, linewidth=1.2, label='Ventral')
    ax.set_xlabel('κ (adaptation gain)')
    ax.set_ylabel('Switch rate (G=50%λ)')
    ax.set_title('B  Controllability vs κ', loc='left', fontweight='bold', fontsize=9)
    ax.legend(fontsize=7)

    # Panel C: Dissociation vs κ
    ax = axes[2]
    ax.plot(kappas, dissoc, 'o-', color=C_PURPLE, markersize=4, linewidth=1.2)
    ax.axhline(0, color=C_GREY, linewidth=0.5, alpha=0.3)
    ax.set_xlabel('κ (adaptation gain)')
    ax.set_ylabel('Dissociation\n(dorsal − ventral)')
    ax.set_title('C  Dissociation vs κ', loc='left', fontweight='bold', fontsize=9)

    fig.tight_layout()
    fig.savefig('fig9_kappa_role.pdf')
    plt.close()
    print("  Saved: fig9_kappa_role.pdf")

    return kappa_results


# =============================================================================
# MAIN
# =============================================================================

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
    elif isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    return obj


def main():
    print("=" * 70)
    print("Paper 1 Mechanistic Analysis")
    print("Crewther Sampler Research Programme v2.2")
    print("=" * 70)

    # Load data
    with open('phase1_grid_results.json') as f:
        phase1_data = json.load(f)
    with open('phase2_dissociation_results.json') as f:
        phase2_data = json.load(f)

    # Warmup
    if HAS_NUMBA:
        _ = simulate_gclca(0.15, 0.2, 0.08, 0.10, 0.03, 0.06, 0.5, 0.5, 0.0, 0.0, 100, 0)
        _ = extract_durations(np.random.randn(200), np.random.randn(200), 50, 0.05)

    t_start = time.time()
    output = {'metadata': {
        'programme': 'Crewther Sampler v2.2',
        'analysis': 'Paper 1 Mechanistic Analysis (B.1.1–B.1.4)',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
    }}

    output['b11_phase_portraits'] = b11_phase_portraits()
    output['b12_sensitivity'] = b12_sensitivity(phase1_data, phase2_data)
    output['b13_transitional'] = b13_transitional_states()
    output['b14_kappa_role'] = b14_kappa_role()

    elapsed = time.time() - t_start
    output['metadata']['total_time_seconds'] = round(elapsed, 1)
    print(f"\nTotal time: {elapsed:.1f}s")

    outfile = 'paper1_mechanistic_results.json'
    with open(outfile, 'w') as f:
        json.dump(sanitize(output), f, indent=1)
    print(f"Saved: {outfile}")

    print("\nFigures saved:")
    print("  fig7_phase_portrait.pdf")
    print("  fig7b_bifurcation.pdf")
    print("  fig8_sensitivity.pdf")
    print("  fig9_kappa_role.pdf")
    print("=" * 70)


if __name__ == '__main__':
    main()
