"""
wave8_closeout.py -- the last two open objections.

PART 1  DRAW STABILITY
    The wave 6 results rest on one random draw of 100 configurations from the
    eligible pool (seed 42). A referee can ask whether a different draw gives
    different numbers. Four independent draws, same statistics, spread reported.

PART 2  MODIFIED PROPOSITION II, LEFT SIDE
    Prop II holds in 27/30 on the stronger-stimulus side and 16/30 on the other.
    Section 4.2 attributes the asymmetry to survivorship: as S_A falls, channel A
    weakens, channel B dominates near-continuously, and boundary exclusion
    removes what remains. That explanation is untested and predicts a specific
    thing -- left-side failures should coincide with high winner-take-all rate
    and low episode retention at low S_A. Tested here on 100 configurations.

Run from C:\\crewther, after wave6_robustness.py.
Writes wave8_results.json, wave8_fig_closeout.pdf

Note on runtime: four draws at reduced seed counts. Seed counts are lower than
wave 6 because the question is config-to-config spread, not within-config
precision. Draw 42 is re-run at the reduced counts so all four are comparable,
and its agreement with the wave 6 values is itself a precision check.
"""

import json
import time

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wave6_robustness as w6
from wave2_campaign import (
    run_trace, extract_durations, seed_summary, aggregate,
    BURN_IN, MARGIN, X_MAX, SIGNAL_NEUTRAL, load_configs,
)

# suppress the per-block file writes so repeated draws do not overwrite
w6._save = lambda tag, obj: None

DRAWS = [42, 43, 44, 45]
N_CFG = 100
N_SEED_G = 50
N_SEED_H = 30
N_SEED_I = 8
N_STEPS = 20000
ASYM_SWEEP = [round(0.25 + 0.05 * i, 2) for i in range(11)]

OUT = {}


def hdr(s):
    print("\n" + "=" * 72)
    print(s)
    print("=" * 72)


# =============================================================================
# PART 1 -- draw stability
# =============================================================================

def stats_for_draw(G, H, I):
    """Reduce one draw to the statistics the manuscript quotes."""
    s = {}

    # double dissociation
    dg, db, ag, ab, pg, pb = [], [], [], [], [], []
    for r in G:
        b = r['conditions']['baseline']
        g = r['conditions']['gclca']
        x = r['conditions']['boost']
        if not b['mean_dur_b'] or not b['alternation_rate_mean']:
            continue
        dg.append(g['mean_dur_b'] / b['mean_dur_b'] if g['mean_dur_b'] else np.nan)
        db.append(x['mean_dur_b'] / b['mean_dur_b'] if x['mean_dur_b'] else np.nan)
        ag.append(g['alternation_rate_mean'] / b['alternation_rate_mean'])
        ab.append(x['alternation_rate_mean'] / b['alternation_rate_mean'])
        pg.append(g['predominance_mean'])
        pb.append(x['predominance_mean'])
    for lbl, a, b_ in (('d_duration', dg, db), ('d_alternation', ag, ab),
                       ('d_predominance', pg, pb)):
        r = w6.boot_paired_d(np.array(a, float), np.array(b_, float))
        s[lbl] = r['d']
        s[lbl + '_n'] = r['n']
        s[lbl + '_prop'] = r['n_positive'] / r['n'] if r['n'] else np.nan

    # mediator AUCs, per regime
    for rn in ('dorsal', 'ventral'):
        fl = np.array([r['regimes'][rn]['baseline_floor_frac'] for r in H])
        sw = np.array([r['regimes'][rn]['max_switch_rate'] for r in H])
        ao = np.array([r['regimes'][rn]['alpha_over_lambda'] for r in H])
        lab = sw > 0.05
        if lab.sum() < 3 or (~lab).sum() < 3:
            s['auc_floor_' + rn] = np.nan
            s['auc_aol_' + rn] = np.nan
            continue
        s['auc_floor_' + rn] = w6.auc_ci(-fl, lab, n_boot=1)[0]
        s['auc_aol_' + rn] = w6.auc_ci(-ao, lab, n_boot=1)[0]
        s['frac_switch_' + rn] = float(lab.mean())

    # Modified Prop IV
    rho = np.array([r['rho'] for r in I], float)
    m = np.isfinite(rho)
    s['prop4_frac_decreasing'] = float(np.mean(rho[m] < -0.7))
    s['prop4_median_rho'] = float(np.median(rho[m]))
    s['prop4_max_wta'] = float(np.nanmax([r['wta_max'] for r in I]))
    return s


def part1(cfgs):
    hdr("PART 1: DRAW STABILITY -- four independent samples of 100")
    excl = list(cfgs.values())
    rows = []
    t0 = time.time()
    for ds in DRAWS:
        print("\n  --- draw seed %d ---" % ds)
        pool, n_elig = w6.eligible_pool(excl, N_CFG, seed=ds)
        G = w6.block_G(pool, N_SEED_G, N_STEPS)
        H = w6.block_H(pool, N_SEED_H)
        I = w6.block_I(pool, N_SEED_I, 12000)
        s = stats_for_draw(G, H, I)
        s['draw_seed'] = ds
        rows.append(s)
        print("    d_dur %+.2f  d_alt %+.2f  d_pred %+.2f  "
              "AUC_floor d/v %.3f/%.3f  AUC_aol d/v %.3f/%.3f  prop4 %.2f"
              % (s['d_duration'], s['d_alternation'], s['d_predominance'],
                 s['auc_floor_dorsal'], s['auc_floor_ventral'],
                 s['auc_aol_dorsal'], s['auc_aol_ventral'],
                 s['prop4_frac_decreasing']))
    print("\n  (%.0f s)" % (time.time() - t0))

    hdr("PART 1 SUMMARY: spread across draws")
    keys = ['d_duration', 'd_alternation', 'd_predominance',
            'auc_floor_dorsal', 'auc_floor_ventral',
            'auc_aol_dorsal', 'auc_aol_ventral',
            'prop4_frac_decreasing', 'prop4_median_rho', 'prop4_max_wta']
    print("  %-24s %8s %8s %8s %8s   %s"
          % ("statistic", "min", "median", "max", "range", "draws"))
    summ = {}
    for k in keys:
        v = np.array([r[k] for r in rows], float)
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        summ[k] = {'min': float(v.min()), 'median': float(np.median(v)),
                   'max': float(v.max()), 'range': float(v.max() - v.min()),
                   'values': [float(x) for x in v]}
        print("  %-24s %+8.3f %+8.3f %+8.3f %8.3f   %s"
              % (k, v.min(), np.median(v), v.max(), v.max() - v.min(),
                 " ".join("%+.3f" % x for x in v)))
    OUT['draw_stability'] = {'per_draw': rows, 'summary': summ}
    return rows


# =============================================================================
# PART 2 -- Modified Proposition II, left side
# =============================================================================

def prop2_with_wta(pool, n_seeds=8, n_steps=12000):
    hdr("PART 2: MODIFIED PROP II with winner-take-all and retention logged")
    out = []
    for n, ent in enumerate(pool):
        p = ent['params']
        ci = ent['grid_index'] % 1000
        levels = []
        for li, s_a in enumerate(ASYM_SWEEP):
            rowset = []
            for seed in range(n_seeds):
                ta, tb = run_trace(p['lambda'], p['beta'], p['alpha'],
                                   p['sigma'], p['gamma'], p['kappa'],
                                   s_a, SIGNAL_NEUTRAL, 0, 0, 0, 0,
                                   n_steps, w6.make_seed(21, ci, 0, li, seed),
                                   X_MAX, -1.0, 0.0, 0.0)
                rowset.append(seed_summary(ta, tb))
            agg = aggregate(rowset)
            agg['signal_a'] = s_a
            levels.append(agg)

        eq = min(levels, key=lambda l: abs(l['signal_a'] - 0.5))
        ref_a, ref_b = eq['mean_dur_a'], eq['mean_dur_b']
        rec = {'grid_index': ent['grid_index'], 'params': p, 'levels': levels}

        for side, sel in (('right', [l for l in levels if l['signal_a'] >= 0.5]),
                          ('left', [l for l in levels if l['signal_a'] <= 0.5])):
            D = np.array([abs(l['signal_a'] - 0.5) for l in sel])
            A = np.array([l['mean_dur_a'] if l['mean_dur_a'] else np.nan
                          for l in sel], float)
            B = np.array([l['mean_dur_b'] if l['mean_dur_b'] else np.nan
                          for l in sel], float)
            m = np.isfinite(A) & np.isfinite(B)
            if m.sum() < 3 or not (ref_a and ref_b):
                rec[side] = None
                continue
            sl_a = np.polyfit(D[m], A[m], 1)[0] * 0.1 / ref_a
            sl_b = np.polyfit(D[m], B[m], 1)[0] * 0.1 / ref_b
            strong, weak = (sl_a, sl_b) if side == 'right' else (sl_b, sl_a)
            rec[side] = {'slope_strong': float(strong),
                         'slope_weak': float(weak),
                         'supported': bool(abs(strong) > abs(weak) and strong > 0)}

        # survivorship indices at the extreme of each side
        lo_end = min(levels, key=lambda l: l['signal_a'])
        hi_end = max(levels, key=lambda l: l['signal_a'])
        rec['wta_at_low_SA'] = lo_end['wta_rate']
        rec['wta_at_high_SA'] = hi_end['wta_rate']
        rec['retention_a_at_low_SA'] = lo_end['retention_a']
        rec['n_raw_a_at_low_SA'] = lo_end['n_raw_a']
        out.append(rec)
        if (n + 1) % 20 == 0:
            print("    %3d/%d" % (n + 1, len(pool)))

    nr = sum(1 for r in out if r.get('right') and r['right']['supported'])
    nl = sum(1 for r in out if r.get('left') and r['left']['supported'])
    nb = sum(1 for r in out if r.get('right') and r.get('left')
             and r['right']['supported'] and r['left']['supported'])
    n = len(out)
    for lbl, k in (("stronger-stimulus side", nr), ("weaker side", nl),
                   ("both sides", nb)):
        lo, hi = w6.wilson(k, n)
        print("  %-24s %3d/%d = %.1f%%  95%% CI [%.1f%%, %.1f%%]"
              % (lbl, k, n, 100 * k / n, 100 * lo, 100 * hi))

    hdr("PART 2: does survivorship explain the left-side failures?")
    left_ok = np.array([bool(r.get('left') and r['left']['supported'])
                        for r in out])
    wta_lo = np.array([r['wta_at_low_SA'] for r in out], float)
    ret_lo = np.array([r['retention_a_at_low_SA'] if r['retention_a_at_low_SA']
                       is not None else np.nan for r in out], float)
    nraw = np.array([r['n_raw_a_at_low_SA'] for r in out], float)

    print("  Section 4.2 predicts: left-side FAILURES should show HIGHER")
    print("  winner-take-all and LOWER retention at low S_A.\n")
    for lbl, v in (("WTA rate at low S_A", wta_lo),
                   ("retention (channel A) at low S_A", ret_lo),
                   ("raw A episodes at low S_A", nraw)):
        m = np.isfinite(v)
        a, b = v[m & left_ok], v[m & ~left_ok]
        if len(a) < 3 or len(b) < 3:
            print("  %-34s insufficient data" % lbl)
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        u, pu = stats.mannwhitneyu(a, b, alternative='two-sided')
        print("  %-34s supported %.4f (n=%d)  failed %.4f (n=%d)  "
              "Welch p=%.3g  MW p=%.3g"
              % (lbl, np.mean(a), len(a), np.mean(b), len(b), p, pu))
    rho, p, lo, hi, nn = w6.boot_spearman(wta_lo, left_ok.astype(float))
    print("\n  rho(WTA at low S_A, left-side supported) = %+.3f %s p=%.3g"
          % (rho, w6.ci_str(lo, hi), p))

    OUT['prop2'] = {
        'n': n, 'right': nr, 'left': nl, 'both': nb,
        'per_config': [{k: r[k] for k in
                        ('grid_index', 'right', 'left', 'wta_at_low_SA',
                         'retention_a_at_low_SA', 'n_raw_a_at_low_SA')}
                       for r in out],
    }
    return out


# =============================================================================

def figures(rows, prop2):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    keys = [('d_duration', 'duration'), ('d_alternation', 'alternation'),
            ('d_predominance', 'predominance')]
    for i, (k, lbl) in enumerate(keys):
        v = [r[k] for r in rows]
        ax[0].plot([i] * len(v), v, 'o', ms=7, alpha=0.7, color='#2c6fbb')
    ax[0].axhline(0, ls='--', c='k', lw=1)
    ax[0].set_xticks(range(3))
    ax[0].set_xticklabels([l for _, l in keys], fontsize=8)
    ax[0].set_ylabel("Cohen's d")
    ax[0].set_title('A  Effect sizes across four draws')

    for i, (k, lbl) in enumerate([('auc_floor_dorsal', 'floor D'),
                                  ('auc_floor_ventral', 'floor V'),
                                  ('auc_aol_dorsal', 'a/l D'),
                                  ('auc_aol_ventral', 'a/l V')]):
        v = [r[k] for r in rows]
        ax[1].plot([i] * len(v), v, 'o', ms=7, alpha=0.7,
                   color='#2c6fbb' if 'floor' in k else '#d1495b')
    ax[1].axhline(0.5, ls='--', c='k', lw=1)
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels(['floor D', 'floor V', 'a/l D', 'a/l V'], fontsize=8)
    ax[1].set_ylabel('AUC')
    ax[1].set_ylim(0, 1)
    ax[1].set_title('B  Classifier AUC across draws')

    ok = np.array([bool(r.get('left') and r['left']['supported'])
                   for r in prop2])
    wta = np.array([r['wta_at_low_SA'] for r in prop2], float)
    ax[2].hist([wta[ok], wta[~ok]], bins=12,
               label=['left side supported', 'left side failed'],
               color=['#2c6fbb', '#cccccc'])
    ax[2].set_xlabel('winner-take-all rate at low $S_A$')
    ax[2].set_ylabel('configurations')
    ax[2].set_title('C  Prop II left-side diagnostic')
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig('wave8_fig_closeout.pdf')
    print("\n  wave8_fig_closeout.pdf")


def main():
    cfgs = load_configs()
    rows = part1(cfgs)
    pool, _ = w6.eligible_pool(list(cfgs.values()), N_CFG, seed=42)
    p2 = prop2_with_wta(pool)
    figures(rows, p2)
    with open('wave8_results.json', 'w', encoding='utf-8') as f:
        json.dump(OUT, f, indent=1, default=float)
    hdr("DONE")
    print("  wave8_results.json")


if __name__ == '__main__':
    main()
