"""
wave4_levelt.py -- correct tests of the MODIFIED Levelt propositions.

Brascamp, Klink & Levelt (2015), Vision Res 109:20-37, Section 4.6:

  I.   Increasing stimulus strength for one eye increases that eye's
       perceptual predominance.                          [tautological]
  II.  Increasing the DIFFERENCE in stimulus strength between the eyes
       primarily increases the dominance duration of the STRONGER stimulus.
  III. Increasing the difference reduces the alternation rate.
       [follows from II; already tested as a peak at equidominance]
  IV.  Increasing stimulus strength in BOTH eyes, held equal, generally
       increases the alternation rate, possibly reversing near threshold.

II and IV are the two uniquely informative propositions.

The submitted manuscript tests original II and III, and its "Prop IV" is
rho(S_A, duration_A), which is not Modified Prop IV. Modified Prop IV requires
an equal-strength sweep that has never been run.

Run from C:\\crewther after wave2_campaign.py.
Writes wave4_results.json, wave4_fig_levelt.pdf
"""

import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wave2_campaign import (
    run_trace, extract_durations, make_seed,
    BURN_IN, MARGIN, X_MAX, SIGNAL_NEUTRAL, load_configs,
)

N_SEEDS = 8
N_STEPS = 12000

# Modified Prop II: asymmetric sweep, S_B fixed
SWEEP_ASYM = [round(0.25 + 0.05 * i, 2) for i in range(11)]     # 0.25 .. 0.75
# Modified Prop IV: equal-strength sweep, S_A = S_B
SWEEP_EQUAL = [round(0.20 + 0.05 * i, 2) for i in range(11)]    # 0.20 .. 0.70


def measure(lam, beta, alpha, sigma, gam, kap, s_a, s_b, block, ci, li):
    """Mean per-eye duration and alternation rate at one signal pair."""
    da, db, alt = [], [], []
    for s in range(N_SEEDS):
        ta, tb = run_trace(lam, beta, alpha, sigma, gam, kap,
                           s_a, s_b, 0.0, 0.0, 0.0, 0.0,
                           N_STEPS, make_seed(block, ci, 0, li, s),
                           X_MAX, -1.0, 0.0, 0.0)
        ch, dur = extract_durations(ta, tb, BURN_IN, MARGIN)
        if len(ch) < 3:
            continue
        ch, dur = ch[1:-1], dur[1:-1].astype(float)
        if len(ch) < 2:
            continue
        if np.any(ch == 0):
            da.append(dur[ch == 0].mean())
        if np.any(ch == 1):
            db.append(dur[ch == 1].mean())
        n_sw = int(np.sum(ch[1:] != ch[:-1]))
        alt.append(n_sw / (N_STEPS - BURN_IN))
    return (float(np.mean(da)) if da else np.nan,
            float(np.mean(db)) if db else np.nan,
            float(np.mean(alt)) if alt else np.nan)


def prop2(cfgs):
    """
    Modified Prop II, tested on each side of equidominance separately.

    Difference D = |S_A - 0.5|. On the S_A > 0.5 side channel A is stronger;
    on the S_A < 0.5 side channel B is stronger. Prediction: on each side the
    STRONGER channel's duration changes more with D than the weaker one's.

    Slopes are normalised by that channel's duration at equidominance so
    configurations with different absolute timescales are comparable.
    """
    print("\n" + "=" * 70)
    print("MODIFIED PROPOSITION II")
    print("=" * 70)
    print("  cfg | RIGHT (A stronger)      | LEFT (B stronger)       | both")
    print("      | str    weak    ratio    | str    weak    ratio    | sides")
    out = []
    for ci in sorted(cfgs.keys()):
        p = cfgs[ci]
        rows = []
        for li, s_a in enumerate(SWEEP_ASYM):
            a, b, al = measure(p['lambda'], p['beta'], p['alpha'], p['sigma'],
                               p['gamma'], p['kappa'], s_a, SIGNAL_NEUTRAL,
                               12, ci, li)
            rows.append({'s_a': s_a, 'dur_a': a, 'dur_b': b, 'alt': al})

        eq = min(rows, key=lambda r: abs(r['s_a'] - 0.5))
        ref_a, ref_b = eq['dur_a'], eq['dur_b']

        rec = {'config_idx': ci, 'params': p, 'levels': rows}
        for side, sel in (('right', [r for r in rows if r['s_a'] >= 0.5]),
                          ('left',  [r for r in rows if r['s_a'] <= 0.5])):
            D = np.array([abs(r['s_a'] - 0.5) for r in sel])
            A = np.array([r['dur_a'] for r in sel], dtype=float)
            B = np.array([r['dur_b'] for r in sel], dtype=float)
            m = np.isfinite(A) & np.isfinite(B)
            if m.sum() < 3 or not (ref_a and ref_b):
                rec[side] = None
                continue
            # normalised slope per 0.1 of difference
            sl_a = np.polyfit(D[m], A[m], 1)[0] * 0.1 / ref_a
            sl_b = np.polyfit(D[m], B[m], 1)[0] * 0.1 / ref_b
            if side == 'right':
                strong, weak = sl_a, sl_b      # A is stronger
            else:
                strong, weak = sl_b, sl_a      # B is stronger
            rec[side] = {
                'slope_strong': float(strong), 'slope_weak': float(weak),
                'ratio': float(abs(strong) / abs(weak)) if weak else None,
                'supported': bool(abs(strong) > abs(weak) and strong > 0),
            }
        both = (rec.get('right') and rec.get('left')
                and rec['right']['supported'] and rec['left']['supported'])
        rec['supported_both_sides'] = bool(both)
        out.append(rec)

        def fmt(d):
            if not d:
                return "  n/a     n/a     n/a  "
            r = d['ratio']
            return "%+6.2f %+6.2f %6s" % (
                d['slope_strong'], d['slope_weak'],
                ("%.2f" % r) if r else "inf")
        print("  %3d | %s | %s | %s"
              % (ci, fmt(rec.get('right')), fmt(rec.get('left')),
                 "YES" if both else "no"))

    n = len(out)
    nr = sum(1 for r in out if r.get('right') and r['right']['supported'])
    nl = sum(1 for r in out if r.get('left') and r['left']['supported'])
    nb = sum(1 for r in out if r['supported_both_sides'])
    print("\n  stronger-eye effect dominates:")
    print("    S_A > 0.5 side : %d/%d" % (nr, n))
    print("    S_A < 0.5 side : %d/%d" % (nl, n))
    print("    both sides     : %d/%d" % (nb, n))
    return out


def prop4(cfgs):
    """
    Modified Prop IV: S_A = S_B swept together. Alternation rate should
    generally increase with strength, possibly reversing near threshold.
    This condition has never been simulated.
    """
    print("\n" + "=" * 70)
    print("MODIFIED PROPOSITION IV  (equal-strength sweep -- NEW)")
    print("=" * 70)
    out = []
    for ci in sorted(cfgs.keys()):
        p = cfgs[ci]
        rows = []
        for li, s in enumerate(SWEEP_EQUAL):
            a, b, al = measure(p['lambda'], p['beta'], p['alpha'], p['sigma'],
                               p['gamma'], p['kappa'], s, s, 13, ci, li)
            rows.append({'signal': s, 'alt': al,
                         'dur_a': a, 'dur_b': b})
        S = np.array([r['signal'] for r in rows])
        A = np.array([r['alt'] for r in rows], dtype=float)
        m = np.isfinite(A)
        if m.sum() < 5:
            print("  cfg %2d  insufficient data" % ci)
            continue
        rho, pv = stats.spearmanr(S[m], A[m])
        # reversal at low strength: is the peak interior?
        c = np.polyfit(S[m], A[m], 2)
        vertex = -c[1] / (2 * c[0]) if c[0] else np.nan
        interior = bool(c[0] < 0 and S[m].min() < vertex < S[m].max())
        rec = {'config_idx': ci, 'params': p, 'levels': rows,
               'spearman_rho': float(rho), 'p': float(pv),
               'monotone_increasing': bool(rho > 0.7 and pv < 0.05),
               'interior_peak': interior,
               'peak_signal': float(vertex) if interior else None}
        out.append(rec)
        print("  cfg %2d  rho=%+.3f p=%.4f  %s%s"
              % (ci, rho, pv,
                 "INCREASING" if rec['monotone_increasing'] else "          ",
                 ("  peak at S=%.2f" % vertex) if interior else ""))
    n = len(out)
    ni = sum(1 for r in out if r['monotone_increasing'])
    np_ = sum(1 for r in out if r['interior_peak'])
    print("\n  monotone increasing (rho>0.7)  : %d/%d" % (ni, n))
    print("  interior peak (reversal at low): %d/%d" % (np_, n))
    return out


def figure(p2, p4):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))

    for rec in p2:
        S = [r['s_a'] for r in rec['levels']]
        A = [r['dur_a'] for r in rec['levels']]
        B = [r['dur_b'] for r in rec['levels']]
        eq = rec['levels'][len(rec['levels']) // 2]
        if not (eq['dur_a'] and eq['dur_b']):
            continue
        ax[0].plot(S, np.array(A) / eq['dur_a'], '-', lw=0.7,
                   alpha=0.4, color='#2c6fbb')
        ax[0].plot(S, np.array(B) / eq['dur_b'], '-', lw=0.7,
                   alpha=0.4, color='#d1495b')
    ax[0].axvline(0.5, ls='--', c='k', lw=1)
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$S_A$   ($S_B$ = 0.5)')
    ax[0].set_ylabel('duration / duration at equidominance')
    ax[0].set_title('A  Modified Prop II')

    ratios_r = [r['right']['ratio'] for r in p2
                if r.get('right') and r['right']['ratio']]
    ratios_l = [r['left']['ratio'] for r in p2
                if r.get('left') and r['left']['ratio']]
    ax[1].hist([np.log10(ratios_r), np.log10(ratios_l)], bins=10,
               label=[r'$S_A>0.5$', r'$S_A<0.5$'],
               color=['#2c6fbb', '#d1495b'])
    ax[1].axvline(0, ls='--', c='k', lw=1)
    ax[1].set_xlabel(r'$\log_{10}$ (stronger slope / weaker slope)')
    ax[1].set_ylabel('configs')
    ax[1].set_title('B  Stronger-eye dominance')
    ax[1].legend(fontsize=7)

    for rec in p4:
        S = [r['signal'] for r in rec['levels']]
        A = [r['alt'] for r in rec['levels']]
        ax[2].plot(S, A, '-', lw=0.7, alpha=0.45, color='#2c6fbb')
    ax[2].set_xlabel(r'$S_A = S_B$')
    ax[2].set_ylabel('alternation rate')
    ax[2].set_title('C  Modified Prop IV')

    fig.tight_layout()
    fig.savefig('wave4_fig_levelt.pdf')
    print("\n  wave4_fig_levelt.pdf")


def main():
    cfgs = load_configs()
    print("Loaded %d configs" % len(cfgs))
    p2 = prop2(cfgs)
    p4 = prop4(cfgs)
    figure(p2, p4)
    with open('wave4_results.json', 'w', encoding='utf-8') as f:
        json.dump({'modified_prop2': p2, 'modified_prop4': p4},
                  f, indent=1, default=float)
    print("  wave4_results.json")


if __name__ == '__main__':
    main()
