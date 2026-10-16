import json, numpy as np
from scipy import stats
d = json.load(open('../li2017_test_results.json')); acc, R = d['accepted'], d['results']
osc = np.array(json.load(open('regime.json'))['osc'])
b = {k: np.array([c['base_matched'][k] for c in acc]) for k in ('mA', 'mB', 'mBabs')}
pct = lambda n, o: np.where(o > 0, 100 * (n - o) / o, np.nan)
comp = lambda k, c='mB': pct(np.array([r[c] for r in R[k]], float), b[c])
att = lambda k: pct(np.array([r['mA'] for r in R[k]], float), b['mA'])
T = 0.5 * (b['mA'] + b['mB']); tau = np.array([c['tau_h'] for c in acc])
out = {}
for lab, m in (('oscillating', osc), ('noise-driven', ~osc)):
    row = {'n': int(m.sum())}
    for k in ('gated_0.1', 'gated_0.2', 'ungated_0.1', 'ungated_0.2', 'yoked_0.1', 'offset_0.2_d1', 'onset_0.2_d1.5'):
        for crit in ('mB', 'mBabs'):
            x = comp(k, crit)[m]; x = x[np.isfinite(x)]
            row[f'{k}|{crit}'] = (int((x > 0).sum()), len(x), float(np.median(x)))
    r = comp('gated_0.1')[m] / att('gated_0.1')[m]; pr = np.exp(-T[m] / tau[m]); ok = np.isfinite(r)
    row['H6_rho'] = float(stats.spearmanr(r[ok], pr[ok])[0])
    row['T_median'] = float(np.median(T[m])); row['wa_median'] = float(np.median([c['wa'] for c, mm in zip(acc, m) if mm]))
    row['inp_median'] = float(np.median([c['inp'] for c, mm in zip(acc, m) if mm]))
    row['cv_median'] = float(np.median([c['base']['cv'] for c, mm in zip(acc, m) if mm]))
    out[lab] = row
    print(lab, row)
json.dump(out, open('strat.json', 'w'), indent=1)
