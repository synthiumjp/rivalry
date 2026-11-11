"""Pre-specified analysis of the sixth test (analysis_plan_li2017_test6.md, Section 5).
    python analyse_li2017_test6.py [li2017_test6_results.json] [li2017_test6_tests.json]
Decisions use merged scoring (mBm) and a lower Wilson 95% bound above 0.5 unless stated; mean-based hypotheses are also
reported under observer scoring (mOB), unmerged scoring (mB) and the absolute criterion (mBabs)."""
import sys, json
import numpy as np
from scipy.stats import rankdata

SRC = sys.argv[1] if len(sys.argv) > 1 else 'li2017_test6_results.json'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'li2017_test6_tests.json'
D = json.load(open(SRC))
SCORINGS = ('mBm', 'mOB', 'mB', 'mBabs')
N_BOOT, BOOT_SEED, R_HUMAN, MIN_N_C2 = 10000, 20261111, 0.2, 20


def wilson(k, n, z=1.959964):
    if n == 0: return [np.nan, np.nan]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def prop(mask_true, mask_ok):
    k, n = int((mask_true & mask_ok).sum()), int(mask_ok.sum())
    ci = wilson(k, n)
    return dict(k=k, n=n, ci=ci, holds=bool(n > 0 and ci[0] > 0.5))


def med(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def change(R, key, m, sl=slice(None)):
    out = []
    for rb, rc in zip(R['base'], R['cond'][key]):
        b = np.array([x[m] for x in rb[sl]], float); c = np.array([x[m] for x in rc[sl]], float)
        ok = np.isfinite(b) & np.isfinite(c)
        out.append(100 * (c[ok].mean() - b[ok].mean()) / b[ok].mean() if ok.sum() and b[ok].mean() > 0 else np.nan)
    return np.array(out, float)


def dist_change(R, key, stat, qi=None, grp='dists'):
    out = []
    for db, dc in zip(R[grp]['base'], R[grp][key]):
        if db is None or dc is None:
            out.append(np.nan); continue
        a, b = (db['q'][qi], dc['q'][qi]) if stat == 'q' else (db[stat], dc[stat])
        out.append(100 * (b - a) / a if stat != 'frac_short' else 100 * (b - a))
    return np.array(out, float)


def zmean(rs):
    rs = np.asarray(rs, float); rs = rs[np.isfinite(rs)]
    return float(np.tanh(np.mean(np.arctanh(np.clip(rs, -0.999, 0.999))))) if len(rs) else np.nan


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]) if len(a) > 3 else np.nan, int(len(a))


def pooled_within(groups):
    """Correlation of within-model normalised ranks (average ranks / (n + 1)), pooled over models."""
    ra, rb = [], []
    for a, b in groups:
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 3:
            ra.append(rankdata(a[ok]) / (ok.sum() + 1)); rb.append(rankdata(b[ok]) / (ok.sum() + 1))
    ra, rb = np.concatenate(ra), np.concatenate(rb)
    return float(np.corrcoef(ra, rb)[0, 1]), int(len(ra))


def boot_pooled_within(groups):
    """Stratified bootstrap: configurations resampled within each model."""
    rng = np.random.default_rng(BOOT_SEED)
    bs = []
    for _ in range(N_BOOT):
        g2 = []
        for a, b in groups:
            i = rng.integers(0, len(a), len(a)); g2.append((a[i], b[i]))
        bs.append(pooled_within(g2)[0])
    return [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]


QI = {q: i for i, q in enumerate(D['settings']['QS'])}


def distribution_block(R, key):
    T = np.array(R['T'], float)
    out = dict(D1_median=prop(dist_change(R, key, 'q', QI[50]) > 0, np.isfinite(dist_change(R, key, 'q', QI[50]))),
               D2_mean_long=prop(dist_change(R, key, 'mean_long') > 0, np.isfinite(dist_change(R, key, 'mean_long'))))
    out['median_change_by_quantile'] = {str(q): med(dist_change(R, key, 'q', i)) for q, i in QI.items()}
    out['frac_short_change_points'] = med(dist_change(R, key, 'frac_short'))
    out['frac_short_base'] = med([d['frac_short'] if d else np.nan for d in R['dists']['base']])
    out['n_episodes_change_pct'] = med(dist_change(R, key, 'n'))
    out['mean_long_change'] = med(dist_change(R, key, 'mean_long'))
    out['median_change'] = med(dist_change(R, key, 'q', QI[50]))
    out['mean_long05_change'] = med(dist_change(R, key, 'mean_long05'))
    out['D2_at_05T'] = prop(dist_change(R, key, 'mean_long05') > 0, np.isfinite(dist_change(R, key, 'mean_long05')))
    out['rate_long_change_pct'] = med(dist_change(R, key, 'rate_long'))
    if key in R.get('dists_u', {}):
        du = lambda st, qi=None: dist_change(R, key, st, qi, grp='dists_u')
        out['unmerged'] = dict(D1_median=prop(du('q', QI[50]) > 0, np.isfinite(du('q', QI[50]))),
                               D2_mean_long=prop(du('mean_long') > 0, np.isfinite(du('mean_long'))),
                               median_change=med(du('q', QI[50])), mean_long_change=med(du('mean_long')))
    if key in R.get('intr', {}):
        out['brief_intrusions_per_min'] = dict(base=med(R['intr']['base']), cond=med(R['intr'][key]))
    return out


def model_block(name, M):
    R = M['results']; res = dict(n_config=len(M['accepted']), n_drawn=M['n_drawn'], rejected=M['rejected'])
    g = {m: change(R, 'gated', m) for m in SCORINGS}
    u = {m: change(R, 'ungated', m) for m in SCORINGS}
    res['gated_lengthens'] = {m: dict(prop(g[m] > 0, np.isfinite(g[m])), median=med(g[m])) for m in SCORINGS}
    res['continuous_shortens'] = {m: dict(prop(u[m] < 0, np.isfinite(u[m])), median=med(u[m])) for m in SCORINGS}
    res['distribution_gated'] = distribution_block(R, 'gated')
    res['distribution_continuous'] = distribution_block(R, 'ungated')
    rs, ro = change(R, 'replay_self', 'mBm'), change(R, 'replay_other', 'mBm')
    ok = np.isfinite(g['mBm']) & np.isfinite(rs) & np.isfinite(ro)
    du_ = {kk: np.array(v, float) for kk, v in R['duty'].items()}
    res['R1_live_gt_self'] = prop(g['mBm'] > rs, ok)
    res['R1_other_scorings'] = {m: prop(g[m] > change(R, 'replay_self', m), np.isfinite(g[m]) & np.isfinite(change(R, 'replay_self', m))) for m in SCORINGS}
    res['R1_per_unit_duty'] = prop(g['mBm'] / du_['gated'] > rs / du_['replay_self'], ok)
    res['R2_self_gt_other_reported'] = prop(rs > ro, ok)
    dose = lambda key, f: med([np.nanmean([x[f] for x in cfg]) for cfg in R['cond'][key]])
    res['replay_dose'] = {kk: dict(duty=med(R['duty'][kk]), dose_comp=dose(kk, 'dose_comp'), dose_att=dose(kk, 'dose_att'),
                                   agree=med(R['agree'][kk])) for kk in ('gated', 'replay_self', 'replay_other')}
    res['agreement'] = {kk: med(v) for kk, v in R['agree'].items()}
    res['replay_medians'] = dict(live=med(g['mBm'][ok]), self=med(rs[ok]), other=med(ro[ok]),
                                 self_lengthens=prop(rs > 0, np.isfinite(rs)), other_lengthens=prop(ro > 0, np.isfinite(ro)))
    res['replay_other_scorings'] = {m: dict(live=med(g[m]), self=med(change(R, 'replay_self', m)),
                                            other=med(change(R, 'replay_other', m))) for m in SCORINGS}
    res['attended_change'] = {k: med(change(R, k, 'mA')) for k in R['cond']}
    res['duty'] = {k: med(v) for k, v in R['duty'].items()}
    res['T_median_ms'] = med(R['T'])
    r_ab = np.array([zmean(s['r_ab']) for s in R['serial']['base']])
    r_ba = np.array([zmean(s['r_ba']) for s in R['serial']['base']])
    res['serial_base'] = dict(r_ab_median=med(r_ab), r_ab_iqr=[float(np.nanpercentile(r_ab, 25)), float(np.nanpercentile(r_ab, 75))],
                              r_ba_median=med(r_ba), n_below_human=int(np.sum(r_ab < R_HUMAN)))
    rho, n = spearman(r_ab, g['mBm'])
    res['serial_vs_gated_within_model'] = dict(rho=rho, n=n)
    if 'gated' in R['serial']:
        rg_ = np.array([zmean(s['r_ab']) for s in R['serial']['gated']])
        res['serial_gated'] = dict(r_ab_median=med(rg_), change_median=med(rg_ - r_ab))
    if name == 'li':
        nt = {k: change(R, f'gated_nt{k:g}', 'mBm') for k in D['settings']['NT_K']}
        k0, k1 = D['settings']['NT_K'][0], D['settings']['NT_K'][-1]
        okw = np.isfinite(nt[k0]) & np.isfinite(nt[k1])
        d0, d1 = np.array(R['duty'][f'gated_nt{k0:g}'], float), np.array(R['duty'][f'gated_nt{k1:g}'], float)
        res['W1_per_unit_duty'] = prop(nt[k1] / d1 < nt[k0] / d0, okw)
        res['W1_total_reported'] = prop(nt[k1] < nt[k0], okw)
        res['W1_other_scorings'] = {m: prop(change(R, f'gated_nt{k1:g}', m) / d1 < change(R, f'gated_nt{k0:g}', m) / d0,
                                           np.isfinite(change(R, f'gated_nt{k1:g}', m)) & np.isfinite(change(R, f'gated_nt{k0:g}', m)))
                                    for m in SCORINGS}
        res['withdrawal'] = {f'k{k:g}': dict(median=med(v), lead_ms=med(R['lead'][f'gated_nt{k:g}']),
                                              duty=med(R['duty'][f'gated_nt{k:g}'])) for k, v in nt.items()}
        res['withdrawal']['live'] = dict(median=med(g['mBm']), lead_ms=med(R['lead']['gated']), duty=med(R['duty']['gated']))
        res['withdrawal_other_scorings'] = {m: {f'k{k:g}': med(change(R, f'gated_nt{k:g}', m)) for k in D['settings']['NT_K']}
                                            for m in SCORINGS}
        att = {}
        for a in D['settings']['ATT_FRACS']:
            ag = {m: change(R, f'att_gated{a:g}', m) for m in SCORINGS}
            au = {m: change(R, f'att_ungated{a:g}', m) for m in SCORINGS}
            okb = np.isfinite(ag['mBm']) & np.isfinite(au['mBm'])
            att[f'{a:g}'] = dict(
                gated_lengthens={m: dict(prop(ag[m] > 0, np.isfinite(ag[m])), median=med(ag[m])) for m in SCORINGS},
                continuous_shortens={m: dict(prop(au[m] < 0, np.isfinite(au[m])), median=med(au[m])) for m in SCORINGS},
                gated_gt_continuous=prop(ag['mBm'] > au['mBm'], okb),
                gated_gt_continuous_other_scorings={m: prop(ag[m] > au[m], np.isfinite(ag[m]) & np.isfinite(au[m])) for m in SCORINGS},
                attention_response={kk: dict(during_attended=med(R['attn'][kk]['A']), during_competitor=med(R['attn'][kk]['B']))
                                    for kk in ('base', f'att_gated{a:g}', f'att_ungated{a:g}', 'gated', 'ungated')},
                attended_gated=med(change(R, f'att_gated{a:g}', 'mA')),
                attended_continuous=med(change(R, f'att_ungated{a:g}', 'mA')),
                distribution_gated=distribution_block(R, f'att_gated{a:g}'))
        res['attention_layer'] = att
        a_ = f"{D['settings']['ATT_FRACS'][-1]:g}"
        res['AT1_att_gated_lengthens'] = att[a_]['gated_lengthens']['mBm']
        res['AT2_att_continuous_shortens'] = att[a_]['continuous_shortens']['mBm']
        res['AT3_att_gated_gt_continuous'] = att[a_]['gated_gt_continuous']
    # C1 inputs, split-half: r_ab from seeds 0-15, the live gate's change from seeds 16-31
    h = len(R['serial']['base'][0]['r_ab']) // 2
    r_half = np.array([zmean(s['r_ab'][:h]) for s in R['serial']['base']])
    g_half = change(R, 'gated', 'mBm', slice(h, None))
    res['C1_within_model'] = dict(zip(('rho', 'n'), spearman(r_half, g_half)))
    low = r_ab < R_HUMAN
    c2 = prop(g['mBm'] > 0, low & np.isfinite(g['mBm']))
    c2['testable'] = bool(c2['n'] >= MIN_N_C2)
    c2['holds'] = bool(c2['testable'] and c2['holds'])
    c2['median_change'] = med(g['mBm'][low])
    res['C2_gated_lengthens_low_serial'] = c2
    return res, r_half, g_half


res, groups = {}, []
for name, M in D['models'].items():
    res[name], rh, gh = model_block(name, M)
    groups.append((rh, gh))
rho, n = pooled_within(groups)
ci = boot_pooled_within(groups)
res['C1_serial_predicts_gated'] = dict(rho=rho, n=n, ci=ci, holds=bool(ci[0] > 0))
# decisions
dec = {}
for name in D['models']:
    r = res[name]
    dec[name] = dict(G1_gated_lengthens=r['gated_lengthens']['mBm']['holds'],
                     G2_continuous_shortens=r['continuous_shortens']['mBm']['holds'],
                     D1_median=r['distribution_gated']['D1_median']['holds'],
                     D2_mean_long=r['distribution_gated']['D2_mean_long']['holds'])
    c2 = r['C2_gated_lengthens_low_serial']
    dec[name]['C2'] = c2['holds'] if c2['testable'] else 'not testable'
    if name == 'li':
        dec[name].update(R1_live_gt_self=r['R1_live_gt_self']['holds'],
                         W1=r['W1_per_unit_duty']['holds'], AT1=r['AT1_att_gated_lengthens']['holds'],
                         AT2=r['AT2_att_continuous_shortens']['holds'], AT3=r['AT3_att_gated_gt_continuous']['holds'])
dec['C1'] = res['C1_serial_predicts_gated']['holds']
res['decisions'] = dec
json.dump(res, open(OUT, 'w'), indent=1, default=float)
print(json.dumps(dec, indent=1))
