"""Registered analysis for the second test (prereg_li2017_test2.md). Reads li2017_test2_results.json.
    python analyse_li2017_test2.py [results.json] [tests.json]"""
import sys, json, math
import numpy as np
from scipy import stats

src = sys.argv[1] if len(sys.argv) > 1 else 'li2017_test2_results.json'
dst = sys.argv[2] if len(sys.argv) > 2 else 'li2017_test2_tests.json'
D = json.load(open(src))
S_ = D['settings']
DELAYS = [0.0] + list(S_['DELAYS'])
DMAX = DELAYS[-1]
EARLY_W = S_['EARLY_W']
ALL, H0, H1 = list(range(8)), [0, 1, 2, 3], [4, 5, 6, 7]
MIN_SEP = math.log(1.25)
M, NA = D['results']['main'], D['results']['noadapt']


def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'), float('nan'))
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def agg(rows, m, seeds):
    with np.errstate(all='ignore'):
        return np.array([np.nanmean([r[s][m] for s in seeds]) if np.isfinite([r[s][m] for s in seeds]).any() else np.nan
                         for r in rows], float)


def change(S, key, m='mBm', seeds=ALL):
    b, c = agg(S['base'], m, seeds), agg(S['cond'][key], m, seeds)
    with np.errstate(all='ignore'):
        return np.where(b > 0, 100 * (c - b) / b, np.nan)


def in_rivalry(S, key):
    """at least 3 merged competitor episodes per seed on average in the condition"""
    return agg(S['cond'][key], 'nBm', ALL) >= 3


def prop(x, mask=None):
    x = np.asarray(x, float)
    if mask is not None: x = np.where(mask, x, np.nan)
    x = x[np.isfinite(x)]; k, n = int((x > 0).sum()), len(x)
    return dict(k=k, n=n, ci=wilson(k, n), median=float(np.median(x)) if n else float('nan'))


out = {'excluded': {}}
T = np.array(M['T'], float)
dL = np.array(M['dL'], float)
riv_w = in_rivalry(M, 'early_0.2') & in_rivalry(M, 'late_0.2') & np.isfinite(dL)
out['excluded']['windows_out_of_rivalry_or_no_dL'] = int((~riv_w).sum())

# ---- H7 (primary): early minus late window (merged scoring primary)
for m in ('mBm', 'mB', 'mBabs'):
    out[f'H7_{m}'] = prop(change(M, 'early_0.2', m) - change(M, 'late_0.2', m), riv_w)
out['H7'] = dict(holds=bool(out['H7_mBm']['ci'][0] > 0.5))
# ---- H8: late window shortens the competitor
for m in ('mBm', 'mB', 'mBabs'):
    out[f'H8_{m}'] = prop(change(M, 'late_0.2', m), riv_w)
out['H8'] = dict(holds=bool(out['H8_mBm']['median'] < 0 and out['H8_mBm']['ci'][1] < 0.5))
# ---- H9: early window shortens the competitor less than the dose account predicts
s_all, u_all = change(M, 'gated_0.2'), change(M, 'ungated_0.2')
e_all = s_all - u_all
g_b = [d['g'] if d else None for d in M['dists']['base_all']]
q_b = [d['q'] if d else None for d in M['dists']['base_all']]
def g_at(gl, x_T):       # g at x (in units of T), grid 0..3T step 0.01T
    return float(np.interp(x_T, np.linspace(0, 3, 301), gl)) if gl is not None else np.nan
def F_at(ql, x_ms):      # empirical distribution function from 1% quantiles
    if ql is None: return np.nan
    ql = np.asarray(ql); return float(np.interp(x_ms, ql, np.linspace(0, 1, 101), left=0.0, right=1.0))
early = change(M, 'early_0.2')
dose_pred = np.array([-e_all[i] * g_at(g_b[i], EARLY_W) for i in range(len(T))])
ho_early = np.array([-e_all[i] * F_at(q_b[i], EARLY_W * T[i]) for i in range(len(T))])
ho_late = np.array([-e_all[i] * (1 - F_at(q_b[i], dL[i])) for i in range(len(T))])
ok9 = riv_w & np.isfinite(early) & np.isfinite(dose_pred) & (e_all > 0)
out['H9_mBm'] = prop(early - dose_pred, ok9)
out['H9'] = dict(holds=bool(out['H9_mBm']['ci'][0] > 0.5))
out['window_predictions_median'] = dict(dose_both=float(np.nanmedian(dose_pred[ok9])), handover_early=float(np.nanmedian(ho_early[ok9])),
                                        handover_late=float(np.nanmedian(ho_late[ok9])),
                                        observed_early=float(np.nanmedian(early[ok9])),
                                        observed_late=float(np.nanmedian(change(M, 'late_0.2')[ok9])))

# ---- H10: reversal delay of delayed withdrawal (seed split: predictions from seeds 0-3, curve from 4-7)
def pava_decreasing(y):
    """least-squares non-increasing fit (pool adjacent violators)"""
    y = list(map(float, y)); blocks = [[v, 1] for v in y]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:
            v = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [v, blocks[i][1] + blocks[i + 1][1]]; del blocks[i + 1]; i = max(i - 1, 0)
        else:
            i += 1
    out_ = []
    for v, w in blocks: out_ += [v] * w
    return np.array(out_)


def crossing(xs, ys):
    """first zero crossing (linear interpolation); +inf if never <= 0; nan if ys[0] <= 0"""
    if not np.all(np.isfinite(ys)) or ys[0] <= 0: return np.nan
    j = np.flatnonzero(ys <= 0)
    if not len(j): return np.inf
    j = j[0]; x0, x1, y0, y1 = xs[j - 1], xs[j], ys[j - 1], ys[j]
    return x0 + (x1 - x0) * y0 / (y0 - y1) if y0 != y1 else x1


def h10(dist_key_u, dist_key_g, pred_seeds, curve_seeds, isotonic, label):
    s = change(M, 'gated_0.2', 'mBm', pred_seeds); u = change(M, 'ungated_0.2', 'mBm', pred_seeds)
    keys = ['gated_0.2'] + [f'offset_0.2_x{d:g}' for d in DELAYS[1:]]
    riv = np.all([in_rivalry(M, k) for k in keys + ['ungated_0.2']], axis=0)
    curve = np.vstack([change(M, k, 'mBm', curve_seeds) for k in keys]).T
    dp, dz, do = [], [], []
    for i in range(len(T)):
        du, dg = M['dists'][dist_key_u][i], M['dists'][dist_key_g][i]
        if not (riv[i] and np.isfinite(s[i]) and np.isfinite(u[i]) and s[i] > 0 and u[i] < 0 and du and dg):
            dp.append(np.nan); dz.append(np.nan); do.append(np.nan); continue
        p = s[i] / (s[i] - u[i])
        dp.append(float(np.interp(p * 100, np.arange(101), du['q'])) / T[i])
        g = np.asarray(dg['g']); k = np.flatnonzero(g >= p)
        dz.append(float(np.interp(p, g[:k[0] + 1], np.linspace(0, 3, 301)[:k[0] + 1])) if len(k) and k[0] > 0 else np.nan)
        y = pava_decreasing(curve[i]) if isotonic and np.all(np.isfinite(curve[i])) else curve[i]
        do.append(crossing(np.array(DELAYS), np.asarray(y)))
    dp, dz, do = map(np.array, (dp, dz, do))          # all in units of T
    base_ok = np.isfinite(dp) & np.isfinite(dz) & (dp > 0) & (dz > 0) & ~np.isnan(do)
    sep = base_ok & (np.abs(np.log(dp / dz)) > MIN_SEP)
    closer = []
    n_amb = 0
    for i in np.flatnonzero(sep):
        if np.isfinite(do[i]):
            if do[i] <= 0: continue
            closer.append(abs(math.log(do[i] / dp[i])) < abs(math.log(do[i] / dz[i])))
        else:   # censored beyond DMAX
            gm = math.sqrt(dp[i] * dz[i])
            if gm <= DMAX: closer.append(dp[i] > dz[i])
            else: n_amb += 1
    k, n = int(np.sum(closer)), len(closer)
    fin = base_ok & np.isfinite(do) & (do > 0)
    # partial Spearman of observed with handover prediction, controlling for dose prediction
    def pr(a, b, c):
        ra, rb, rc = stats.rankdata(a), stats.rankdata(b), stats.rankdata(c)
        X = np.column_stack([np.ones(len(rc)), rc])
        ea = ra - X @ np.linalg.lstsq(X, ra, rcond=None)[0]; eb = rb - X @ np.linalg.lstsq(X, rb, rcond=None)[0]
        return float(np.corrcoef(ea, eb)[0, 1])
    if fin.sum() >= 10:
        r = pr(do[fin], dp[fin], dz[fin]); prng = np.random.default_rng(20261016 + 1)
        null = np.array([pr(do[fin], prng.permutation(dp[fin]), dz[fin]) for _ in range(5000)])
        pv = float((np.sum(null >= r) + 1) / 5001)
    else:
        r, pv = float('nan'), float('nan')
    return dict(label=label, k=k, n=n, ci=wilson(k, n), holds_b=bool(n > 0 and wilson(k, n)[0] > 0.5),
                partial_rho=r, p_one_sided=pv, n_partial=int(fin.sum()), holds_a=bool(r > 0 and pv < 0.05),
                n_excluded_no_prediction=int((~(np.isfinite(dp) & np.isfinite(dz))).sum()),
                n_excluded_gated_not_positive=int(np.sum(np.isfinite(dp) & np.isnan(do))),
                n_not_separated=int(np.sum(base_ok & ~sep)), n_censored_ambiguous=n_amb,
                n_censored=int(np.sum(base_ok & np.isinf(do))),
                median_obs=float(np.median(do[fin])) if fin.any() else float('nan'),
                median_pred_handover=float(np.nanmedian(dp)), median_pred_dose=float(np.nanmedian(dz)),
                per_config=dict(dpred=dp.tolist(), ddose=dz.tolist(), dobs=[None if not np.isfinite(x) else float(x) for x in do]))

prim = h10('ungated_h0', 'ungated_h0', H0, H1, True, 'primary: ungated distribution, seeds 0-3 / 4-7, isotonic')
out['H10'] = prim
prim['holds'] = prim['holds_b']   # H10 (primary). The partial correlation (holds_a) is descriptive.
out['H10_sensitivity'] = [
    {k: v for k, v in h10('base_h0', 'base_h0', H0, H1, True, 'baseline distribution').items() if k != 'per_config'},
    {k: v for k, v in h10('ungated_all', 'ungated_all', ALL, ALL, True, 'all seeds, no split').items() if k != 'per_config'},
    {k: v for k, v in h10('ungated_h0', 'ungated_h0', H0, H1, False, 'raw crossing (no isotonic fit)').items() if k != 'per_config'}]

# ---- H11: Chong et al.'s manipulation as run lengthens the competitor (out of rivalry = not lengthening)
riv_c = in_rivalry(M, 'chong_1')
out['excluded']['chong_out_of_rivalry'] = int((~riv_c).sum())
for m in ('mBm', 'mB', 'mBabs'):
    x = change(M, 'chong_1', m); x = np.where(riv_c, x, -1.0)           # out of rivalry counts as not lengthening
    out[f'H11_{m}'] = prop(x)
out['H11'] = dict(holds=bool(out['H11_mBm']['ci'][0] > 0.5))
inp = np.array([c['inp'] for c in D['accepted']])
out['H11_sensitivity_input_le_0.5'] = prop(np.where(riv_c, change(M, 'chong_1'), -1.0), inp <= 0.5)

# ---- H12: without adaptation (slow memory), gated delivery leaves the competitor nearly unchanged
g_na = change(NA, 'gated_0.2'); u_na = change(NA, 'ungated_0.2'); a_na = change(NA, 'gated_0.2', 'mA')
g_na = np.where(in_rivalry(NA, 'gated_0.2'), g_na, np.nan)
gm = g_na[np.isfinite(g_na)]; gmain = s_all[np.isfinite(s_all)]
mw = stats.mannwhitneyu(gm, gmain, alternative='less') if len(gm) and len(gmain) else None
H12 = dict(noadapt_gated=prop(g_na), noadapt_ungated=prop(u_na), noadapt_attended=prop(a_na),
           median_main=float(np.median(gmain)) if len(gmain) else float('nan'),
           mw_p=float(mw.pvalue) if mw is not None else float('nan'))
H12['holds'] = bool(H12['noadapt_ungated']['ci'][1] < 0.5 and H12['noadapt_attended']['ci'][0] > 0.5
                    and abs(H12['noadapt_gated']['median']) < 0.25 * abs(H12['median_main'])
                    and H12['mw_p'] < 0.05)
out['H12'] = H12

# ---- descriptive: every condition, three scorings, doses, fragments, rivalry loss
desc = {}
for S, nm in ((M, 'main'), (NA, 'noadapt')):
    for key in S['cond']:
        r = {m: prop(change(S, key, m)) for m in ('mBm', 'mB', 'mBabs')}
        r['attended'] = prop(change(S, key, 'mA'))
        for k in ('dose_comp', 'dose_att'):
            r[k] = float(np.nanmean(agg(S['cond'][key], k, ALL)))
        r['fragments_per_merged'] = float(np.nanmedian(agg(S['cond'][key], 'nB', ALL) / agg(S['cond'][key], 'nBm', ALL)))
        r['out_of_rivalry'] = int((~in_rivalry(S, key)).sum())
        desc[f'{nm}|{key}'] = r
    desc[f'{nm}|base_fragments_per_merged'] = float(np.nanmedian(agg(S['base'], 'nB', ALL) / agg(S['base'], 'nBm', ALL)))
out['descriptive'] = desc
out['sample'] = dict(n_main=len(M['base']), n_drawn=D['n_drawn'], n_na=len(NA['base']), n_drawn_na=D['n_drawn_na'],
                     median_T=float(np.median(T)), median_dL_over_T=float(np.nanmedian(dL / T)), dry_run=D['dry_run'])
json.dump(out, open(dst, 'w'), indent=1, default=float)
for h in ('H7', 'H8', 'H9', 'H10', 'H11', 'H12'):
    print(h, 'holds' if out[h]['holds'] else 'does not hold')
for h in ('H7_mBm', 'H8_mBm', 'H9_mBm', 'H11_mBm'):
    print(' ', h, out[h])
print('  H10', {k: v for k, v in prim.items() if k != 'per_config'})
print('  H12', {k: v for k, v in H12.items()})
print('  window predictions', out['window_predictions_median'])
print('  excluded', out['excluded'])
