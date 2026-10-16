import json, numpy as np
from scipy import stats
d = json.load(open('../li2017_test_results.json')); acc, R = d['accepted'], d['results']
b = {k: np.array([c['base_matched'][k] for c in acc]) for k in ('mA', 'mB', 'mBabs')}
pct = lambda n, o: np.where(o > 0, 100 * (n - o) / o, np.nan)
comp = lambda k, c='mB': pct(np.array([r[c] for r in R[k]], float), b[c])
att = lambda k: pct(np.array([r['mA'] for r in R[k]], float), b['mA'])
duty = lambda k: np.array([r['duty'] for r in R[k]], float)
rng = np.random.default_rng(1)
def bootmed(x, B=10000):
    x = x[np.isfinite(x)]; m = np.median(rng.choice(x, (B, len(x))), axis=1)
    return np.median(x), np.percentile(m, [2.5, 97.5])
def wil(k, n, z=1.96):
    p = k/n; c = (p+z*z/(2*n))/(1+z*z/n); h = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n); return c-h, c+h
out = {}
print('--- medians with bootstrap CI, and Wilson intervals')
for k in R:
    for crit in ('mB', 'mBabs'):
        x = comp(k, crit); xf = x[np.isfinite(x)]; m, ci = bootmed(x)
        kk = int((xf > 0).sum()); lo, hi = wil(kk, len(xf))
        out[f'{k}|{crit}'] = dict(median=m, ci=list(ci), k=kk, n=len(xf), wilson=[lo, hi])
        print(f'{k:18s} {crit:6s} median {m:+6.1f} [{ci[0]:+6.1f}, {ci[1]:+6.1f}]  {kk}/{len(xf)}  Wilson [{lo:.3f}, {hi:.3f}]')
print('--- H3 paired difference medians with CI')
for f in ('0.1', '0.2'):
    for crit in ('mB', 'mBabs'):
        x = comp(f'gated_{f}', crit) - comp(f'yoked_{f}', crit); m, ci = bootmed(x)
        print(f'H3 {f} {crit}: median {m:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]')
        out[f'H3_{f}|{crit}'] = dict(median=m, ci=list(ci))
print('--- H4/H5 within-configuration trend')
offk = ['gated_0.2'] + [f'offset_0.2_d{x:g}' for x in (0.25, 0.5, 1.0, 1.5)]
onk = ['gated_0.2'] + [f'onset_0.2_d{x:g}' for x in (0.25, 0.5, 1.0, 1.5)]
for lab, keys in (('offset', offk), ('onset', onk)):
    M = np.vstack([comp(k) for k in keys]).T; ok = np.all(np.isfinite(M), 1); M = M[ok]
    # Page test for decreasing trend: reverse order so predicted ranks increase
    pt = stats.page_trend_test(M[:, ::-1], method='asymptotic')
    rho = [stats.spearmanr([0, .25, .5, 1, 1.5], r)[0] for r in M]
    neg = int(np.sum(np.array(rho) < 0))
    print(f'{lab}: n={len(M)}, Page L={pt.statistic:.0f}, p={pt.pvalue:.2e}; per-config Spearman with delay negative in {neg}/{len(M)}, median rho {np.median(rho):+.2f}')
    out[f'trend_{lab}'] = dict(n=int(len(M)), page_L=float(pt.statistic), page_p=float(pt.pvalue), neg=neg, median_rho=float(np.median(rho)))
print('--- onset: change per unit of delivered dose (competitor change / duty)')
for k in onk:
    x = comp(k) / duty(k); m, ci = bootmed(x); print(f'{k}: duty {np.mean(duty(k)):.2f}  change/duty median {m:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]')
    out[f'perdose_{k}'] = dict(duty=float(np.mean(duty(k))), median=m, ci=list(ci))
for k in offk:
    x = comp(k) / duty(k); m, ci = bootmed(x); print(f'{k}: duty {np.mean(duty(k)):.2f}  change/duty median {m:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]')
    out[f'perdose_{k}'] = dict(duty=float(np.mean(duty(k))), median=m, ci=list(ci))
print('--- offset at long delays vs ungated')
for k in ('offset_0.2_d1', 'offset_0.2_d1.5'):
    x = comp(k) - comp('ungated_0.2'); m, ci = bootmed(x); print(f'{k} minus ungated_0.2: median {m:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]')
    out[f'{k}_minus_ungated'] = dict(median=m, ci=list(ci))
print('--- H6 alternatives')
T = 0.5 * (b['mA'] + b['mB']); tau = np.array([c['tau_h'] for c in acc])
a10, c10 = att('gated_0.1'), comp('gated_0.1'); ratio = c10 / a10
print('attended change at gated 10%: min', np.nanmin(a10), 'median', np.nanmedian(a10))
def partial_spearman(x, y, z):
    rx, ry = stats.rankdata(x), stats.rankdata(y); Z = np.column_stack([np.ones(len(x))] + [stats.rankdata(zz) for zz in z])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]; ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return stats.pearsonr(ex, ey)
pr = partial_spearman(ratio, tau, [T]); print('partial Spearman ratio~tau_h | T:', pr)
pr2 = partial_spearman(ratio, -T, [tau]); print('partial Spearman ratio~(-T) | tau_h:', pr2)
# rank regression of competitor change on attended change, T and tau_h
X = np.column_stack([np.ones(len(T)), stats.rankdata(a10), stats.rankdata(T), stats.rankdata(tau)])
yb = stats.rankdata(c10); beta, *_ = np.linalg.lstsq(X, yb, rcond=None)
res = yb - X @ beta; s2 = res @ res / (len(yb) - 4); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
print('rank regression comp ~ att + T + tau: beta', np.round(beta[1:], 3), 't', np.round(beta[1:] / se[1:], 2))
out['H6_alt'] = dict(partial_tau_given_T=[float(pr[0]), float(pr[1])], partial_T_given_tau=[float(pr2[0]), float(pr2[1])],
                     rankreg_beta=beta[1:].tolist(), rankreg_t=(beta[1:] / se[1:]).tolist(), att_min=float(np.nanmin(a10)))
json.dump(out, open('json_analyses.json', 'w'), indent=1, default=float)
