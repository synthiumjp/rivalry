import json, numpy as np
X = json.load(open('explore_results.json')); D = json.load(open('../li2017_test2_results.json'))
def agg(rows, m): return np.array([np.nanmean([s[m] for s in r]) if np.isfinite([s[m] for s in r]).any() else np.nan for r in rows], float)
def wil(k, n, z=1.96):
    p = k/n; c = (p+z*z/(2*n))/(1+z*z/n); h = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n); return c-h, c+h
main = X[:8]; na = X[8:12]; win = X[12]
# reproduction of registered baselines
bm = [r for j in main for r in j['base']]; bn = [r for j in na for r in j['base']]
for nm, B, R in (('main', bm, D['results']['main']['base']), ('noadapt', bn, D['results']['noadapt']['base'])):
    print(nm, 'baseline reproduced:', np.array_equal(agg(B, 'mBm'), agg(R, 'mBm'), equal_nan=True))
out = {}
def report(jobs, key, label):
    B = [r for j in jobs for r in j['base']]; C = [r for j in jobs for r in j[key]]
    riv = agg(C, 'nBm') >= 3
    r = {}
    for m in ('mBm', 'mB', 'mBabs', 'mA'):
        b, c = agg(B, m), agg(C, m); x = np.where(b > 0, 100*(c-b)/b, np.nan)
        xr = np.where(riv, x, np.nan); xf = xr[np.isfinite(xr)]
        r[m] = (float(np.median(xf)) if len(xf) else np.nan, int((xf > 0).sum()), len(xf))
    r['out_of_rivalry'] = int((~riv).sum())
    r['dose_comp'] = float(np.nanmean(agg(C, 'dose_comp'))); r['dose_att'] = float(np.nanmean(agg(C, 'dose_att')))
    out[label] = r; print(label, r)
report(main, 'chong_0.05', 'main chong 5%'); report(main, 'chong_0.1', 'main chong 10%'); report(main, 'chong_0.2', 'main chong 20%')
report(na, 'gated_lat_0.2', 'noadapt gated, 100 ms onset latency'); report(na, 'chong_0.2', 'noadapt chong timing 20%')
# merge-threshold sensitivity for windows (25 configs)
for key in ('early_0.2', 'late_0.2'):
    row = {}
    for t in (20, 50, 100, 200):
        b = agg(win['base_mt'], f'm{t}'); c = agg(win[key + '_mt'], f'm{t}'); x = 100*(c-b)/b
        row[t] = (round(float(np.nanmedian(x)), 1), int(np.sum(x > 0)), int(np.isfinite(x).sum()))
    out[f'merge_threshold_{key}'] = row; print(key, row)
d = {t: None for t in (20, 50, 100, 200)}
for t in (20, 50, 100, 200):
    b = agg(win['base_mt'], f'm{t}'); e = 100*(agg(win['early_0.2_mt'], f'm{t}')-b)/b; l = 100*(agg(win['late_0.2_mt'], f'm{t}')-b)/b
    dd = e - l; d[t] = (int(np.sum(dd > 0)), int(np.isfinite(dd).sum()), round(float(np.nanmedian(dd)), 1))
out['merge_threshold_H7'] = d; print('H7 by merge threshold', d)
json.dump(out, open('explore_summary.json', 'w'), indent=1)
