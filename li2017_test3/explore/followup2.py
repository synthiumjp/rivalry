"""EXPLORATORY (not registered), second set of follow-up analyses of the third test.
(1) H21 with the first-order curves re-anchored to pass through both the no-hold change and the
    continuous-delivery change: e* = (D0 - u) / (1 - W(d0)), with W the account's weight on [0, d);
    reported with the registered decision rule, the within-25% rates and a constant comparator.
(2) Every third-test hypothesis statistic under the passive observer (300-600 ms), from the
    registered analysis output.
(3) Matched-noise comparison of adaptation: every gate and Chong et al.'s timing, high-noise sample
    with adaptation against the no-adaptation sample, under the three secondary scorings.
    python followup2.py ../li2017_test3_results.json ../li2017_test3_tests.json"""
import sys, io, json, math, contextlib
import numpy as np
src = sys.argv[1] if len(sys.argv) > 1 else '../li2017_test3_results.json'
tst = json.load(open(sys.argv[2] if len(sys.argv) > 2 else '../li2017_test3_tests.json'))
sys.argv = ['analyse', src, '/dev/null']
ns = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(open('../analyse_li2017_test3.py').read(), 'analyse_li2017_test3.py', 'exec'), ns)
g = ns
M, NA, HN, T, HOLDS, D_OFF_MS = g['M'], g['NA'], g['HN'], g['T'], g['HOLDS'], g['D_OFF_MS']
change, in_rivalry, F_at, g_at, inv_grid, pava, crossing, wilson = (g[k] for k in ('change', 'in_rivalry', 'F_at', 'g_at', 'inv_grid', 'pava_decreasing', 'crossing', 'wilson'))
H0, H1, MIN_SEP, GRID_MAX = g['H0'], g['H1'], g['MIN_SEP'], g['GRID_MAX']
out = {}


def h21_reanchored(m='mOB', dist_key='ungated_0.1_h0'):
    keys = ['chong_0.1'] + [f'chong_0.1_x{h:g}' for h in HOLDS[1:]]
    riv = np.all([in_rivalry(M, k) for k in keys + ['gated_0.1', 'ungated_0.1']], axis=0)
    d0 = change(M, 'chong_0.1', m, H0); u = change(M, 'ungated_0.1', m, H0)
    curve = np.vstack([change(M, k, m, H1) for k in keys]).T
    off_ms = D_OFF_MS
    dp, dz, dr, do = [], [], [], []
    for i in range(len(T)):
        dist = M['dists'][dist_key][i]; off = off_ms / T[i]
        nan4 = lambda: (dp.append(np.nan), dz.append(np.nan), dr.append(np.nan), do.append(np.nan))
        if not (riv[i] and dist and np.isfinite(d0[i]) and np.isfinite(u[i]) and d0[i] > 0 and u[i] < 0 and np.all(np.isfinite(curve[i]))):
            nan4(); continue
        Wh0, Wz0, Wr0 = F_at(dist['q'], off_ms), g_at(dist['g'], off), g_at(dist['hz'], off)
        res = []
        for W0, kind in ((Wh0, 'h'), (Wz0, 'z'), (Wr0, 'r')):
            es = (d0[i] - u[i]) / (1 - W0) if W0 < 1 else np.nan
            target = W0 + d0[i] / es if np.isfinite(es) and es > 0 else np.nan
            if not np.isfinite(target) or target >= 1: res.append(np.nan); continue
            if kind == 'h':
                x = float(np.interp(target * 100, np.arange(101), dist['q'])) / T[i]
            elif kind == 'z':
                x = inv_grid(dist['g'], target)
            else:
                x = inv_grid(dist['hz'], target)
            res.append(x if np.isfinite(x) and x <= GRID_MAX else np.nan)
        dp.append(res[0]); dz.append(res[1]); dr.append(res[2])
        c = crossing(np.array(HOLDS), pava(curve[i]))
        do.append(c + off if np.isfinite(c) else c)
    dp, dz, dr, do = map(np.array, (dp, dz, dr, do))
    dmax = HOLDS[-1] + D_OFF_MS / T
    base_ok = np.isfinite(dp) & np.isfinite(dz) & (dp > 0) & (dz > 0) & ~np.isnan(do)
    sep = base_ok & (np.abs(np.log(dp / dz)) > MIN_SEP)
    closer = []
    for i in np.flatnonzero(sep):
        if np.isfinite(do[i]): closer.append(abs(math.log(do[i] / dp[i])) < abs(math.log(do[i] / dz[i])))
        elif math.sqrt(dp[i] * dz[i]) <= dmax[i]: closer.append(dp[i] > dz[i])
    fin = np.isfinite(dp) & np.isfinite(dz) & np.isfinite(dr) & np.isfinite(do) & (do > 0)
    const = np.median(do[fin])
    w = lambda pred: float(np.mean(np.abs(np.log(do[fin] / pred)) < MIN_SEP))
    k, n = int(np.sum(closer)), len(closer)
    return dict(scoring=m, k=k, n=n, ci=wilson(k, n), n_finite=int(fin.sum()),
                median_obs=float(np.median(do[fin])), median_handover=float(np.median(dp[fin])),
                median_hazard=float(np.median(dr[fin])), median_dose=float(np.median(dz[fin])),
                within25=dict(handover=w(dp[fin]), hazard=w(dr[fin]), dose=w(dz[fin]), constant=w(np.full(fin.sum(), const))),
                median_curve_asymptote_check='curves pass through the no-hold change and the continuous change by construction')


out['H21_reanchored'] = [h21_reanchored('mOB'), h21_reanchored('mBm')]
# (2) passive observer
po = {}
for h in ('H13', 'H15', 'H18a', 'H19', 'H20', 'H22', 'H23'):
    v = tst.get(f'{h}_mOB2'); po[h] = dict(k=v['k'], n=v['n'], median=v['median'])
po['H14'] = 'see H14_sensitivity (passive observer): handover closer than hazard in 54 of 160'
po['H16_latency_gate'] = tst['H16']['a_latency_gate']['competitor_mOB2']
po['H16_report_gate'] = tst['H16']['b_report_gate']['competitor_mOB2']
po['H17'] = dict(noadapt=tst['H17']['noadapt_mOB2'], main=tst['H17']['main_mOB2'])
po['H21'] = [s for s in tst['H21_sensitivity'] if 'passive' in s['label']][0]['vs_dose']
po['H24_latency_gate'] = tst['H24']['a_latency_gate']['mOB2']; po['H24_chong'] = tst['H24']['b_chong']['mOB2']
out['passive_observer'] = po
# (3) matched-noise comparison
mn = {}
for key in ('gated_lat_0.2', 'gated_0.2', 'rep_gated_0.2', 'chong_0.1'):
    for S, nm in ((HN, 'highnoise'), (NA, 'noadapt')):
        mn[f'{key}|{nm}'] = {m: dict(median=float(np.nanmedian(change(S, key, m))), k=int(np.nansum(change(S, key, m) > 0)),
                                     n=int(np.isfinite(change(S, key, m)).sum())) for m in ('mBm', 'mB', 'mBabs')}
out['matched_noise'] = mn
json.dump(out, open('followup2_summary.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
