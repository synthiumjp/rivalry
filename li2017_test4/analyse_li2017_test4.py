"""Pre-specified analysis of the fourth test (analysis_plan_li2017_test4.md, Section 6).
    python analyse_li2017_test4.py [li2017_test4_results.json] [li2017_test4_tests.json]
Primary scoring: merged (mBm). Every hypothesis is also reported under observer scoring (mOB), unmerged
scoring (mB) and the absolute criterion (mBabs); decisions use merged scoring only."""
import sys, json
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else 'li2017_test4_results.json'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'li2017_test4_tests.json'
D = json.load(open(SRC))
M = D['results']['main']; S = D['settings']; K = D['kernel']; C = D['accepted']
W = [tuple(w) for w in S['WINDOWS']]
BIN, HA = S['BIN'], S['HA']
SCORINGS = ('mBm', 'mOB', 'mB', 'mBabs')
NBOOT, BOOT_SEED = 2000, 20261023


def wilson(k, n, z=1.959964):
    if n == 0: return [np.nan, np.nan]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def change(key, m):
    out = []
    for rb, rc in zip(M['base'], M['cond'][key]):
        b = np.array([x[m] for x in rb], float); c = np.array([x[m] for x in rc], float)
        ok = np.isfinite(b) & np.isfinite(c)
        out.append(100 * (c[ok].mean() - b[ok].mean()) / b[ok].mean() if ok.sum() and b[ok].mean() > 0 else np.nan)
    return np.array(out, float)


def overlap(b, w, lo, hi):
    edges = np.arange(len(b) + 1) * w
    ov = np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0, None) / w
    return float(np.sum(np.asarray(b) * ov))


# ------------------------------------------------ kernel predictions (percent of the kernel's mean merged duration)
n = len(C)
WK = [f'{a:g}-{b:g}' for a, b in W] + ['late']
pred = {}
for f in S['FRACS']:
    pred[f'gated_{f:g}'] = np.full(n, np.nan); pred[f'ungated_{f:g}'] = np.full(n, np.nan)
    for wk in WK: pred[f'{wk}_{f:g}'] = np.full(n, np.nan)
share_late = np.full(n, np.nan)
for i, (c, k) in enumerate(zip(C, K)):
    if k is None:
        continue
    bA = np.array(k['bA']); nb = len(k['fatt']); bBa, bBo = np.array(k['bB'][:nb]), np.array(k['bB'][nb:])
    fa = np.array(k['fatt']); w = BIN * k['T']; T3, dL = M['T'][i], M['dL'][i]
    for f in S['FRACS']:
        sc = 100 * f * c['inp'] / k['meanD']
        pred[f'gated_{f:g}'][i] = sc * np.sum(bBa * fa)
        pred[f'ungated_{f:g}'][i] = sc * np.sum(bBa * fa + bBo * (1 - fa))
        if np.isfinite(T3):
            for (a, b), wk in zip(W, WK):
                pred[f'{wk}_{f:g}'][i] = sc * overlap(bA, w, a * T3, b * T3)
        if np.isfinite(dL):
            pred[f'late_{f:g}'][i] = sc * overlap(bA, w, dL, HA * k['T'])
    neg = np.minimum(bA, 0.0); tot = np.sum(neg)        # K6 is defined on the negative (shortening) weight
    share_late[i] = overlap(neg, BIN * k['T'], 0.5 * k['T'], 1.5 * k['T']) / tot if tot < 0 else np.nan
q = np.array(M['q'], float)
pred['yoked_0.05'] = q * pred['ungated_0.05']     # the yoked gate is on with probability q, independently of the trial


def obs_key(wk, f):
    return f'late_{f:g}' if wk == 'late' else f'win_{f:g}_{wk}'


def ratio_med(o, p):
    ok = np.isfinite(o) & np.isfinite(p)
    mo, mp = np.median(o[ok]), np.median(p[ok])
    return mo / mp if mp != 0 else np.nan


def abslog(r):
    return abs(np.log(r)) if np.isfinite(r) and r > 0 else np.inf


rng = np.random.default_rng(BOOT_SEED)
BOOT = [rng.integers(0, n, n) for _ in range(NBOOT)]


def tests(m):
    T = {}
    O = {k: change(k, m) for k in M['cond']}
    # K1 continuous delivery at 5% and 10%
    h = {}
    for f in (0.05, 0.10):
        o, p = O[f'ungated_{f:g}'], pred[f'ungated_{f:g}']
        r = ratio_med(o, p); ok = np.isfinite(o) & np.isfinite(p)
        within = np.abs(o[ok] - p[ok]) <= 0.25 * np.abs(p[ok])
        from scipy.stats import spearmanr
        h[f'{f:g}'] = dict(median_obs=float(np.median(o[ok])), median_pred=float(np.median(p[ok])), ratio=float(r),
                          within25=int(within.sum()), n=int(ok.sum()), spearman=float(spearmanr(o[ok], p[ok])[0]))
    h['holds'] = bool(all(0.75 <= h[x]['ratio'] <= 1.25 for x in ('0.05', '0.1')))
    T['K1'] = h
    # K2 small increments follow the kernel: |log R| smaller at 5% than at 20% for the 0.5-0.75 and late windows
    h = {}
    for wk in ('0.5-0.75', 'late'):
        o5, p5 = O[obs_key(wk, 0.05)], pred[f'{wk}_0.05']; o20, p20 = O[obs_key(wk, 0.2)], pred[f'{wk}_0.2']
        r5, r20 = ratio_med(o5, p5), ratio_med(o20, p20)
        diff = abslog(r20) - abslog(r5)
        bd = []
        for ix in BOOT:
            a_, b_ = abslog(ratio_med(o20[ix], p20[ix])), abslog(ratio_med(o5[ix], p5[ix]))
            bd.append(a_ - b_ if (np.isfinite(a_) or np.isfinite(b_)) else -np.inf)   # both infinite: counts against
        bd = np.array(bd, float)
        lo = float(np.quantile(bd, 0.025, method='inverted_cdf'))
        both_inf = np.isinf(abslog(r5)) and np.isinf(abslog(r20))
        h[wk] = dict(R5=float(r5), R20=float(r20), diff=float(diff) if not both_inf else None, boot_lo=lo,
                     holds=bool((not both_inf) and diff > 0 and lo > 0))
    h['holds'] = bool(h['0.5-0.75']['holds'] and h['late']['holds'])
    h['profile_5'] = {wk: dict(obs=float(np.nanmedian(O[obs_key(wk, 0.05)])), pred=float(np.nanmedian(pred[f'{wk}_0.05']))) for wk in WK}
    h['profile_20'] = {wk: dict(obs=float(np.nanmedian(O[obs_key(wk, 0.2)])), pred=float(np.nanmedian(pred[f'{wk}_0.2']))) for wk in WK}
    T['K2'] = h
    # K3 the gated lengthening at 5% is not captured by the kernel
    o, p = O['gated_0.05'], pred['gated_0.05']; ok = np.isfinite(o) & np.isfinite(p)
    k1, k2 = int((o[ok] > 0).sum()), int((p[ok] < 0.5 * o[ok]).sum()); nn = int(ok.sum())   # (b): the kernel accounts for less than half the observed change
    T['K3'] = dict(median_obs=float(np.median(o[ok])), median_pred=float(np.median(p[ok])), lengthen=[k1, nn],
                    ci_lengthen=wilson(k1, nn), exceeds_pred=[k2, nn], ci_exceeds=wilson(k2, nn),
                    holds=bool(wilson(k1, nn)[0] > 0.5 and wilson(k2, nn)[0] > 0.5))
    # K4 yoked replay at 5% is closer to the kernel's yoked prediction than to the live gate; live exceeds yoked
    y, g = O['yoked_0.05'], O['gated_0.05']; p = pred['yoked_0.05']; ok = np.isfinite(y) & np.isfinite(g) & np.isfinite(p)
    k1 = int((np.abs(y[ok] - p[ok]) < np.abs(y[ok] - g[ok])).sum()); k2 = int((g[ok] > y[ok]).sum()); nn = int(ok.sum())
    T['K4'] = dict(median_yoked=float(np.median(y[ok])), median_pred_yoked=float(np.median(p[ok])), median_q=float(np.nanmedian(q)), closer_to_kernel=[k1, nn], ci_closer=wilson(k1, nn),
                    live_exceeds=[k2, nn], ci_live=wilson(k2, nn), holds=bool(wilson(k1, nn)[0] > 0.5 and wilson(k2, nn)[0] > 0.5))
    # K5 the gated effect is first order: per-unit medians at 5% and 20% within [0.67, 1.5]
    ok = np.isfinite(O['gated_0.05']) & np.isfinite(O['gated_0.2'])
    g5, g20 = np.median(O['gated_0.05'][ok]) / 0.05, np.median(O['gated_0.2'][ok]) / 0.2
    r = g5 / g20 if g20 != 0 else np.nan
    T['K5'] = dict(per_unit_5=float(g5), per_unit_10=float(np.nanmedian(O['gated_0.1']) / 0.1), per_unit_20=float(g20),
                    n=int(ok.sum()), ratio=float(r), holds=bool(np.isfinite(r) and 0.67 <= r <= 1.5))
    return T


res = {m: tests(m) for m in SCORINGS}
# K6 (kernel shape; scoring-independent): share of the kernel's negative weight on [0.5T, 1.5T)
res['K6'] = dict(median_share=float(np.nanmedian(share_late)), n=int(np.isfinite(share_late).sum()),
                  holds=bool(np.nanmedian(share_late) > 0.5))
KP = [100 * 0.1 * c['inp'] * np.array(k['bA']) / k['meanD'] for c, k in zip(C, K) if k is not None]
res['kernel_profile'] = dict(bins=[j * BIN for j in range(int(round(HA / BIN)))], n=len(KP),
                             median_pct_per_10pct=[float(x) for x in np.median(np.array(KP), 0)] if KP else None)
res['n_config'] = n
json.dump(res, open(OUT, 'w'), indent=1, default=float)
print(f'n = {n} configurations; decisions on merged scoring')
for h in ('K1', 'K2', 'K3', 'K4', 'K5'):
    print(f'  {h}: ' + ' / '.join(f'{m}: {"holds" if res[m][h]["holds"] else "not"}' for m in SCORINGS))
print(f'  K6: {"holds" if res["K6"]["holds"] else "not"} (median share {res["K6"]["median_share"]:.2f})')
print(json.dumps({h: res['mBm'][h] for h in ('K1', 'K3', 'K4', 'K5')}, indent=1, default=float)[:3000])
