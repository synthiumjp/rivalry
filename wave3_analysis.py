"""
wave3_analysis.py -- analysis and figures from the Wave 2 campaign.

Two short reruns first (about 10 s total):
  B2  Levelt sweep WITH the registered minimum-duration filter
      (my Wave 2 block B omitted it; unfiltered alternation rate is inflated
      by 1-2 step transitional flickers at asymmetric signal levels)
  F2  sharpness sweep, additionally logging mean residual so Lambda is
      computable there

Then:
  1. Lambda collapse       G * x_supp / sigma  as the unifying mediator
  2. Prop III (C6)         quadratic fit in predominance, vertex test
  3. Prop II / IV (C7)     magnitude comparison, not rank correlation
  4. H4 canonical (A1)     replacement Table 3, both DPR conventions
  5. Mediation (C10/C11)   alpha/lambda -> floor occupancy -> switch rate

Run from C:\\crewther after wave2_campaign.py.
Writes wave3_results.json and wave3_fig*.pdf
"""

import json
import os
import sys

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wave2_campaign import (
    run_trace, extract_durations, pulse_trial, make_seed,
    BURN_IN, MARGIN, X_MAX, G_SAFETY, SIGNAL_NEUTRAL, SIGNAL_LEVELS,
    DORSAL_MULTS, VENTRAL_MULTS, STRICT_CONFIGS,
    N_STEPS_BASELINE, N_STEPS_TOTAL, N_STEPS_PULSE,
    SWITCH_CRITERION, LOOKBACK_WINDOW, load_configs,
)

OUT = {}
PLOTS = []


def load(tag):
    fn = "wave2_%s.json" % tag
    if not os.path.exists(fn):
        sys.exit("ERROR: %s missing. Run wave2_campaign.py first." % fn)
    with open(fn, "r", encoding="utf-8") as f:
        return json.load(f)


def hdr(s):
    print("\n" + "=" * 70)
    print(s)
    print("=" * 70)


# =============================================================================
# RERUN B2 -- Levelt sweep with the registered minimum-duration filter
# =============================================================================

def filtered_summary(ta, tb, min_dur):
    """Predominance and alternation rate after min-duration filtering."""
    ch, dur = extract_durations(ta, tb, BURN_IN, MARGIN)
    if len(ch) == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    keep = dur >= min_dur
    c, d = ch[keep], dur[keep]
    if len(c) < 2:
        return np.nan, np.nan, np.nan, np.nan, len(c)
    t_a = float(d[c == 0].sum())
    t_b = float(d[c == 1].sum())
    tot = t_a + t_b
    pred = t_a / tot if tot > 0 else np.nan
    n_sw = int(np.sum(c[1:] != c[:-1]))
    alt = n_sw / (len(ta) - BURN_IN)
    mda = float(d[c == 0].mean()) if np.any(c == 0) else np.nan
    mdb = float(d[c == 1].mean()) if np.any(c == 1) else np.nan
    return pred, alt, mda, mdb, len(c)


def rerun_B2(cfgs, n_seeds=8, n_steps=12000):
    hdr("RERUN B2: Levelt sweep with registered min-duration filter")
    out = []
    for ci in sorted(cfgs.keys()):
        p = cfgs[ci]
        lam, beta, alpha = p['lambda'], p['beta'], p['alpha']
        sigma, gam, kap = p['sigma'], p['gamma'], p['kappa']

        # registered filter: 5th pct of per-seed durations at neutral signal
        pool = []
        for s in range(n_seeds):
            ta, tb = run_trace(lam, beta, alpha, sigma, gam, kap,
                               SIGNAL_NEUTRAL, SIGNAL_NEUTRAL, 0, 0, 0, 0,
                               n_steps, make_seed(9, ci, 0, 0, s),
                               X_MAX, -1.0, 0.0, 0.0)
            _, d = extract_durations(ta, tb, BURN_IN, MARGIN)
            pool.extend(d.tolist())
        min_dur = int(round(np.percentile(pool, 5))) if pool else 1
        min_dur = max(1, min_dur)

        levels = []
        for li, sig_a in enumerate(SIGNAL_LEVELS):
            pr, al, da, db = [], [], [], []
            for s in range(n_seeds):
                ta, tb = run_trace(lam, beta, alpha, sigma, gam, kap,
                                   sig_a, SIGNAL_NEUTRAL, 0, 0, 0, 0,
                                   n_steps, make_seed(10, ci, 0, li, s),
                                   X_MAX, -1.0, 0.0, 0.0)
                a, b, c, d, _ = filtered_summary(ta, tb, min_dur)
                if not np.isnan(a):
                    pr.append(a)
                    al.append(b)
                if not np.isnan(c):
                    da.append(c)
                if not np.isnan(d):
                    db.append(d)
            levels.append({
                'signal_a': sig_a,
                'predominance': float(np.mean(pr)) if pr else None,
                'alternation_rate': float(np.mean(al)) if al else None,
                'mean_dur_a': float(np.mean(da)) if da else None,
                'mean_dur_b': float(np.mean(db)) if db else None,
            })
        out.append({'config_idx': ci, 'params': p,
                    'min_dur': min_dur, 'levels': levels})
        print("  cfg %2d  min_dur=%d" % (ci, min_dur))
    return out


def rerun_F2(cfgs, n_seeds=50):
    hdr("RERUN F2: sharpness sweep with residual logging")
    ks = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, -1.0]
    out = []
    for ci in STRICT_CONFIGS:
        p = cfgs[ci]
        lam, sigma, gam = p['lambda'], p['sigma'], p['gamma']
        sweep = []
        g_value = min(0.9 * lam, G_SAFETY * lam)
        for li, k in enumerate(ks):
            e = {'k_sharp': k,
                 'f_at_zero': float(np.log(2) / k) if k > 0 else 0.0}
            for ri, (rn, m) in enumerate([('dorsal', DORSAL_MULTS),
                                          ('ventral', VENTRAL_MULTS)]):
                sw, rs = [], []
                for s in range(n_seeds):
                    r = pulse_trial(lam, p['beta'] * m['beta'],
                                    p['alpha'] * m['alpha'], sigma, gam,
                                    p['kappa'] * m['kappa'],
                                    SIGNAL_NEUTRAL, SIGNAL_NEUTRAL, g_value,
                                    N_STEPS_BASELINE, N_STEPS_TOTAL,
                                    N_STEPS_PULSE,
                                    make_seed(11, ci, ri, li, s),
                                    X_MAX, LOOKBACK_WINDOW,
                                    SWITCH_CRITERION, MARGIN, k)
                    sw.append(r[0])
                    rs.append(r[5])
                e[rn + '_rate'] = float(np.mean(sw))
                e[rn + '_residual'] = float(np.mean(rs))
                e[rn + '_lambda_drive'] = float(g_value * np.mean(rs) / sigma)
            sweep.append(e)
        out.append({'config_idx': ci, 'params': p, 'g_value': g_value,
                    'sweep': sweep})
        print("  cfg %2d done" % ci)
    return out


# =============================================================================
# 1. LAMBDA COLLAPSE
# =============================================================================

def analyse_lambda(cmed, f2):
    hdr("1. LAMBDA COLLAPSE:  Lambda = G * x_supp / sigma")
    rows = []
    for rec in cmed:
        sigma = rec['params']['sigma']
        aol = {}
        for rn in ('dorsal', 'ventral'):
            R = rec['regimes'][rn]
            aol[rn] = R['alpha_over_lambda']
            for gk, gv in R['g_levels'].items():
                if gv['g_fraction'] == 0:
                    continue
                lamdrive = (gv['g_value'] * gv['mean_residual_mean'] / sigma)
                rows.append({'config': rec['config_idx'], 'regime': rn,
                             'g_fraction': gv['g_fraction'],
                             'lambda_drive': lamdrive,
                             'switch_rate': gv['switch_rate'],
                             'floor': gv['floor_frac_mean'],
                             'alpha_over_lambda': aol[rn],
                             'source': 'hard'})
    for rec in f2:
        sigma = rec['params']['sigma']
        for e in rec['sweep']:
            for rn in ('dorsal', 'ventral'):
                rows.append({'config': rec['config_idx'], 'regime': rn,
                             'g_fraction': 0.9,
                             'lambda_drive': e[rn + '_lambda_drive'],
                             'switch_rate': e[rn + '_rate'],
                             'floor': np.nan,
                             'alpha_over_lambda': np.nan,
                             'source': 'k=%g' % e['k_sharp']})

    L = np.array([r['lambda_drive'] for r in rows])
    S = np.array([r['switch_rate'] for r in rows])
    ok = np.isfinite(L) & np.isfinite(S)
    L, S = L[ok], S[ok]
    rho, p = stats.spearmanr(L, S)
    print("  n = %d  Spearman rho = %.3f  p = %.2e" % (len(L), rho, p))

    best = max(((t, np.mean((L > t) == (S > 0.05)))
                for t in np.unique(np.round(L, 4))), key=lambda x: x[1])
    print("  best Lambda threshold = %.4f -> %.1f%% classification"
          % (best[0], 100 * best[1]))

    # compare against floor and alpha/lambda on the hard-rectifier subset
    hard = [r for r in rows if r['source'] == 'hard']
    fl = np.array([r['floor'] for r in hard])
    sh = np.array([r['switch_rate'] for r in hard])
    al = np.array([r['alpha_over_lambda'] for r in hard])
    for nm, v in (('Lambda', np.array([r['lambda_drive'] for r in hard])),
                  ('floor occupancy', fl), ('alpha/lambda', al)):
        m = np.isfinite(v)
        b = max(((t, max(np.mean((v[m] > t) == (sh[m] > 0.05)),
                         np.mean((v[m] < t) == (sh[m] > 0.05))))
                 for t in np.unique(np.round(v[m], 4))), key=lambda x: x[1])
        r2, _ = stats.spearmanr(v[m], sh[m])
        print("    %-18s rho=%+.3f   best split %.1f%%" % (nm, r2, 100 * b[1]))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for src, mk, c in (('hard', 'o', '#2c6fbb'), ('soft', '^', '#d1495b')):
        sel = [r for r in rows
               if (r['source'] == 'hard') == (src == 'hard')
               and np.isfinite(r['lambda_drive'])]
        ax[0].scatter([r['lambda_drive'] for r in sel],
                      [r['switch_rate'] for r in sel],
                      s=22, marker=mk, alpha=0.6, c=c,
                      label='hard rectifier' if src == 'hard' else 'softplus sweep')
    ax[0].axvline(best[0], ls='--', c='k', lw=1,
                  label='threshold %.3f' % best[0])
    ax[0].set_xscale('symlog', linthresh=1e-3)
    ax[0].set_xlabel(r'$\Lambda = G\,\bar{x}_{supp}/\sigma$')
    ax[0].set_ylabel('switch rate')
    ax[0].set_title('A  Drive-to-noise collapse')
    ax[0].legend(fontsize=7)

    ax[1].scatter(fl, sh, s=22, alpha=0.6, c='#2c6fbb')
    ax[1].axvline(0.901, ls='--', c='k', lw=1)
    ax[1].set_xlabel('floor occupancy  P(x_supp = 0)')
    ax[1].set_ylabel('switch rate')
    ax[1].set_title('B  Floor occupancy (hard rectifier only)')
    fig.tight_layout()
    fig.savefig('wave3_fig1_lambda.pdf')
    PLOTS.append('wave3_fig1_lambda.pdf')

    OUT['lambda_collapse'] = {
        'n': int(len(L)), 'spearman_rho': float(rho), 'p': float(p),
        'best_threshold': float(best[0]),
        'classification': float(best[1]), 'rows': rows,
    }


# =============================================================================
# 2. PROPOSITION III  (C6)
# =============================================================================

def analyse_prop3(b2):
    hdr("2. PROPOSITION III: alternation rate peaks at equidominance?")
    res = []
    for rec in b2:
        pr = np.array([l['predominance'] for l in rec['levels']
                       if l['predominance'] is not None], dtype=float)
        al = np.array([l['alternation_rate'] for l in rec['levels']
                       if l['predominance'] is not None], dtype=float)
        if len(pr) < 5:
            continue
        c = np.polyfit(pr, al, 2)
        vertex = -c[1] / (2 * c[0]) if c[0] != 0 else np.nan
        yhat = np.polyval(c, pr)
        ss = 1 - np.sum((al - yhat) ** 2) / np.sum((al - al.mean()) ** 2)
        interior = bool(c[0] < 0 and pr.min() < vertex < pr.max())
        res.append({'config_idx': rec['config_idx'],
                    'quad_coef': float(c[0]), 'vertex': float(vertex),
                    'r2': float(ss), 'interior_peak': interior,
                    'near_equidominance': bool(interior and
                                               abs(vertex - 0.5) < 0.15)})
    n = len(res)
    ni = sum(r['interior_peak'] for r in res)
    ne = sum(r['near_equidominance'] for r in res)
    vs = [r['vertex'] for r in res if r['interior_peak']]
    print("  configs with concave interior peak : %d/%d" % (ni, n))
    print("  peak within 0.15 of equidominance  : %d/%d" % (ne, n))
    if vs:
        print("  vertex location: median %.3f  IQR [%.3f, %.3f]"
              % (np.median(vs), np.percentile(vs, 25), np.percentile(vs, 75)))
        t, p = stats.wilcoxon(np.array(vs) - 0.5)
        print("  Wilcoxon vertex vs 0.5: W=%.1f p=%.4f" % (t, p))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for rec in b2:
        pr = [l['predominance'] for l in rec['levels']]
        al = [l['alternation_rate'] for l in rec['levels']]
        ax[0].plot(pr, al, '-', lw=0.8, alpha=0.45)
    ax[0].axvline(0.5, ls='--', c='k', lw=1)
    ax[0].set_xlabel('predominance of percept A')
    ax[0].set_ylabel('alternation rate (filtered)')
    ax[0].set_title('A  Modified Proposition III')
    if vs:
        ax[1].hist(vs, bins=12, color='#2c6fbb', alpha=0.8)
    ax[1].axvline(0.5, ls='--', c='k', lw=1)
    ax[1].set_xlabel('fitted peak location')
    ax[1].set_ylabel('configs')
    ax[1].set_title('B  Peak locations')
    fig.tight_layout()
    fig.savefig('wave3_fig2_prop3.pdf')
    PLOTS.append('wave3_fig2_prop3.pdf')

    OUT['prop3'] = {'n': n, 'n_interior': ni, 'n_near_equi': ne,
                    'per_config': res}


# =============================================================================
# 3. PROPOSITIONS II / IV  (C7)
# =============================================================================

def analyse_prop24(b2):
    hdr("3. PROPOSITIONS II / IV: magnitude, not rank")
    res = []
    for rec in b2:
        sa = np.array([l['signal_a'] for l in rec['levels']], dtype=float)
        da = np.array([l['mean_dur_a'] if l['mean_dur_a'] else np.nan
                       for l in rec['levels']], dtype=float)
        db = np.array([l['mean_dur_b'] if l['mean_dur_b'] else np.nan
                       for l in rec['levels']], dtype=float)
        m = np.isfinite(da) & np.isfinite(db)
        if m.sum() < 5:
            continue
        ref = np.argmin(np.abs(sa[m] - 0.5))
        # % change per 0.1 signal, normalised to the neutral point
        sa_a = np.polyfit(sa[m], da[m], 1)[0] * 0.1 / da[m][ref] * 100
        sb_b = np.polyfit(sa[m], db[m], 1)[0] * 0.1 / db[m][ref] * 100
        res.append({'config_idx': rec['config_idx'],
                    'pct_change_same_eye_per_0.1': float(sa_a),
                    'pct_change_other_eye_per_0.1': float(sb_b),
                    'ratio_other_to_same': float(abs(sb_b) / abs(sa_a))
                    if sa_a else None})
    ra = [r['ratio_other_to_same'] for r in res if r['ratio_other_to_same']]
    same = [r['pct_change_same_eye_per_0.1'] for r in res]
    other = [r['pct_change_other_eye_per_0.1'] for r in res]
    print("  same-eye  (Prop IV) %%change per +0.1 S_A: median %+.1f%%"
          % np.median(same))
    print("  other-eye (Prop II) %%change per +0.1 S_A: median %+.1f%%"
          % np.median(other))
    if ra:
        print("  |other| / |same| ratio: median %.2f  (Prop II dominates "
              "if > 1)" % np.median(ra))
        print("  configs with ratio > 1: %d/%d"
              % (sum(1 for x in ra if x > 1), len(ra)))
    OUT['prop24'] = {'per_config': res,
                     'median_ratio': float(np.median(ra)) if ra else None}


# =============================================================================
# 4. H4 CANONICAL TABLE  (A1 / B19 / B24)
# =============================================================================

def analyse_h4(ah4):
    hdr("4. H4 CANONICAL -- replacement for Table 3")
    rows = []
    for rec in ah4:
        r = {'config_idx': rec['config_idx']}
        for cn in ('baseline', 'gclca', 'boost'):
            s = rec['conditions'][cn]['summary']
            r[cn] = {
                'mean_dur_target': s['mean_dur_a'],
                'mean_dur_nontarget': s['mean_dur_b'],
                'predominance': s['predominance_mean'],
                'alternation_rate': s['alternation_rate_mean'],
                'wta_rate': s['wta_rate'],
                'retention_nontarget': s['retention_b'],
                'ceil_frac': s['ceil_frac_mean'],
            }
        b = r['baseline']
        for cn in ('gclca', 'boost'):
            # every seed winner-take-all -> no surviving non-target episodes.
            # DPR is undefined there. That is the B24 artifact, not a value.
            nt = r[cn]['mean_dur_nontarget']
            bnt = b['mean_dur_nontarget']
            r[cn]['dpr'] = (nt / bnt) if (nt and bnt) else None
            ar, bar = r[cn]['alternation_rate'], b['alternation_rate']
            r[cn]['alt_ratio'] = (ar / bar) if (ar is not None and bar) else None
        rows.append(r)

    def col(cn, k):
        return np.array([r[cn][k] if r[cn][k] is not None else np.nan
                         for r in rows], dtype=float)

    def nanfmt(v, f="%.3f"):
        return "  n/a" if (v is None or not np.isfinite(v)) else f % v

    print("  cfg | pred base/gc/bo | altrate gc/bo (x base) | "
          "DPR gc/bo | WTA gc | ceil gc")
    for r in rows:
        print("  %3d | %s %s %s | %s %s | %s %s | %4.0f%% | %s"
              % (r['config_idx'], nanfmt(r['baseline']['predominance']),
                 nanfmt(r['gclca']['predominance']),
                 nanfmt(r['boost']['predominance']),
                 nanfmt(r['gclca']['alt_ratio']), nanfmt(r['boost']['alt_ratio']),
                 nanfmt(r['gclca']['dpr']), nanfmt(r['boost']['dpr']),
                 100 * r['gclca']['wta_rate'], nanfmt(r['gclca']['ceil_frac'])))

    print("\n  MEAN OF PER-CONFIG (the correct convention)")
    for k in ('predominance', 'alternation_rate'):
        print("    %-18s base %.4f  gclca %.4f  boost %.4f"
              % (k, np.nanmean(col('baseline', k)),
                 np.nanmean(col('gclca', k)), np.nanmean(col('boost', k))))
    ng = int(np.sum(np.isfinite(col('gclca', 'dpr'))))
    print("    DPR                gclca %.4f  boost %.4f   "
          "(defined for %d/%d configs)"
          % (np.nanmean(col('gclca', 'dpr')),
             np.nanmean(col('boost', 'dpr')), ng, len(rows)))

    print("\n  RATIO OF POOLED MEANS (the convention that misled us)")
    bn = np.nanmean(col('baseline', 'mean_dur_nontarget'))
    print("    DPR                gclca %.4f  boost %.4f"
          % (np.nanmean(col('gclca', 'mean_dur_nontarget')) / bn,
             np.nanmean(col('boost', 'mean_dur_nontarget')) / bn))

    for k in ('predominance', 'dpr', 'alt_ratio'):
        g, b_ = col('gclca', k), col('boost', k)
        m = np.isfinite(g) & np.isfinite(b_)
        if m.sum() < 3:
            print("    %-14s too few defined pairs (%d)" % (k, m.sum()))
            continue
        t, p = stats.ttest_rel(g[m], b_[m])
        dd = g[m] - b_[m]
        d = dd.mean() / dd.std(ddof=1) if dd.std(ddof=1) else np.nan
        print("    %-14s GC-LCA vs boost: t=%+.2f p=%.4f d=%+.2f  "
              "(GC-LCA higher in %d/%d defined)"
              % (k, t, p, d, int(np.sum(g[m] > b_[m])), int(m.sum())))

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    x = np.arange(len(rows))
    w = 0.27
    for i, (cn, c) in enumerate((('baseline', '#999999'),
                                 ('gclca', '#2c6fbb'), ('boost', '#d1495b'))):
        ax[0].bar(x + (i - 1) * w, col(cn, 'predominance'), w, color=c,
                  label=cn)
    ax[0].axhline(0.5, ls='--', c='k', lw=1)
    ax[0].set_ylabel('predominance')
    ax[0].set_title('A  Predominance')
    ax[0].legend(fontsize=7)
    for i, (cn, c) in enumerate((('gclca', '#2c6fbb'), ('boost', '#d1495b'))):
        ax[1].bar(x + (i - 0.5) * w, np.nan_to_num(col(cn, 'alt_ratio')), w, color=c,
                  label=cn)
    ax[1].axhline(1.0, ls='--', c='k', lw=1)
    ax[1].set_ylabel('alternation rate / baseline')
    ax[1].set_title('B  Alternation rate')
    ax[1].legend(fontsize=7)
    ax[2].bar(x, 100 * col('gclca', 'wta_rate'), color='#2c6fbb')
    ax[2].set_ylabel('% seeds winner-take-all')
    ax[2].set_title('C  GC-LCA WTA rate')
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels([r['config_idx'] for r in rows], fontsize=7)
        a.set_xlabel('configuration')
    fig.tight_layout()
    fig.savefig('wave3_fig3_h4.pdf')
    PLOTS.append('wave3_fig3_h4.pdf')

    OUT['h4_canonical'] = rows


# =============================================================================
# 5. MEDIATION  (C10 / C11)
# =============================================================================

def analyse_mediation(cmed):
    hdr("5. MEDIATION: alpha/lambda -> floor occupancy -> switch rate")
    A, F, S, R = [], [], [], []
    for rec in cmed:
        for rn in ('dorsal', 'ventral'):
            Rg = rec['regimes'][rn]
            A.append(Rg['alpha_over_lambda'])
            F.append(Rg['baseline_floor_frac'])
            S.append(Rg['max_switch_rate'])
            R.append(rn)
    A, F, S = np.array(A), np.array(F), np.array(S)

    def sp(x, y, lbl):
        r, p = stats.spearmanr(x, y)
        print("    %-34s rho=%+.3f  p=%.2e" % (lbl, r, p))
        return r
    print("  pooled (both regimes, n=%d)" % len(A))
    sp(A, S, "alpha/lambda -> switch rate (c)")
    sp(A, F, "alpha/lambda -> floor (a)")
    sp(F, S, "floor -> switch rate (b)")

    # partial correlation of A with S controlling F (rank-based)
    ra, rf, rs = (stats.rankdata(A), stats.rankdata(F), stats.rankdata(S))
    def resid(y, x):
        return y - np.polyval(np.polyfit(x, y, 1), x)
    pr = np.corrcoef(resid(ra, rf), resid(rs, rf))[0, 1]
    print("    partial rho(alpha/lambda, switch | floor) = %+.3f" % pr)

    for rn in ('dorsal', 'ventral'):
        m = np.array([x == rn for x in R])
        print("  %s only (n=%d)" % (rn, m.sum()))
        sp(A[m], S[m], "alpha/lambda -> switch rate")
        sp(F[m], S[m], "floor -> switch rate")

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    cols = {'dorsal': '#2c6fbb', 'ventral': '#d1495b'}
    for rn in ('dorsal', 'ventral'):
        m = np.array([x == rn for x in R])
        ax[0].scatter(A[m], S[m], s=26, alpha=0.7, c=cols[rn], label=rn)
        ax[1].scatter(F[m], S[m], s=26, alpha=0.7, c=cols[rn], label=rn)
        ax[2].scatter(A[m], F[m], s=26, alpha=0.7, c=cols[rn], label=rn)
    ax[0].axvline(0.75, ls='--', c='k', lw=1)
    ax[0].set_xlabel(r'$\alpha/\lambda$')
    ax[0].set_ylabel('max switch rate')
    ax[0].set_title(r'A  Proxy ($\alpha/\lambda$)')
    ax[1].axvline(0.901, ls='--', c='k', lw=1)
    ax[1].set_xlabel('floor occupancy')
    ax[1].set_ylabel('max switch rate')
    ax[1].set_title('B  Mechanism (floor)')
    ax[2].set_xlabel(r'$\alpha/\lambda$')
    ax[2].set_ylabel('floor occupancy')
    ax[2].set_title('C  Proxy vs mechanism')
    for a in ax:
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig('wave3_fig4_mediation.pdf')
    PLOTS.append('wave3_fig4_mediation.pdf')

    OUT['mediation'] = {
        'alpha_over_lambda': A.tolist(), 'floor': F.tolist(),
        'switch_rate': S.tolist(), 'regime': R,
        'partial_rho_controlling_floor': float(pr),
    }


# =============================================================================

def main():
    cfgs = load_configs()
    b2 = rerun_B2(cfgs)
    f2 = rerun_F2(cfgs)
    OUT['B2_levelt_filtered'] = b2
    OUT['F2_sharpness'] = f2

    cmed = load('C_mediator')
    ah4 = load('A_h4')

    analyse_lambda(cmed, f2)
    analyse_prop3(b2)
    analyse_prop24(b2)
    analyse_h4(ah4)
    analyse_mediation(cmed)

    with open('wave3_results.json', 'w', encoding='utf-8') as f:
        json.dump(OUT, f, indent=1, default=float)
    hdr("DONE")
    print("  wave3_results.json")
    for p in PLOTS:
        print("  " + p)


if __name__ == '__main__':
    main()
