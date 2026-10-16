"""Checks exact reproduction of the registered summaries and produces the numbers in Table S10.
Run from li2017_test/robustness/ after both passes of robustness_li2017.py."""
import json, numpy as np
reg = json.load(open('../li2017_test_results.json')); RR = reg['results']
P = {**json.load(open('robust_results_pass1.json'))}
p2 = json.load(open('robust_results_pass2.json'))
NS = 8; nc = len(P['base']) // NS
agg = lambda rows, k: np.array([np.nanmean([r[k] for r in rows[c*NS:(c+1)*NS]]) for c in range(nc)])
def pooled(rows):
    s = np.array([np.sum([r['sumB'] for r in rows[c*NS:(c+1)*NS]]) for c in range(nc)])
    n = np.array([np.sum([r['nB'] for r in rows[c*NS:(c+1)*NS]]) for c in range(nc)])
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)
with np.errstate(all='ignore'):
    # 1. exact reproduction
    for name, D in (('pass 1', P), ('pass 2', p2)):
        assert np.array_equal(agg(D['base'], 'mB'), np.array([c['base_matched']['mB'] for c in reg['accepted']]), equal_nan=True)
        for k in [k for k in D if k != 'base']:
            for m in ('mA', 'mB', 'mBabs'):
                assert np.array_equal(agg(D[k], m), np.array([x[m] for x in RR[k]], float), equal_nan=True), (k, m)
        print(name, ': registered summaries reproduced exactly')
    # 2. Table S10
    out = {}
    for D in (P, p2):
        for k in [k for k in D if k != 'base']:
            row = {}
            for m in ('mB', 'gapA', 'mB_merge', 'mB_cens', 'pooled'):
                b = pooled(D['base']) if m == 'pooled' else agg(D['base'], m)
                v = pooled(D[k]) if m == 'pooled' else agg(D[k], m)
                x = 100 * (v - b) / b; x = x[np.isfinite(x)]
                row[m] = f'{np.median(x):+.1f}%, {int((x > 0).sum())}/{len(x)}'
            tb, tc, mb = agg(D['base'], 'trans'), agg(D[k], 'trans'), agg(D['base'], 'mB')
            row['transition_ms_base_cond'] = [float(np.nanmedian(tb)), float(np.nanmedian(tc))]
            row['transition_pct_of_comp'] = float(np.nanmedian(100 * tb / mb))
            row['transition_shortening_pct_of_comp'] = float(np.nanmedian(100 * (tb - tc) / mb))
            out[k] = row; print(k, row)
    # 3. merged live minus yoked, and seed-level sign reliability
    for f in ('0.1', '0.2'):
        b = agg(P['base'], 'mB_merge')
        d = 100 * (agg(P[f'gated_{f}'], 'mB_merge') - b) / b - 100 * (agg(P[f'yoked_{f}'], 'mB_merge') - b) / b
        d = d[np.isfinite(d)]; out[f'merged_live_minus_yoked_{f}'] = [int((d > 0).sum()), len(d), float(np.median(d))]
        print('merged live - yoked', f, out[f'merged_live_minus_yoked_{f}'])
    for k in ('gated_0.1', 'ungated_0.1'):
        rel = tot = 0
        for c in range(nc):
            bs = np.array([r['mB'] for r in P['base'][c*NS:(c+1)*NS]]); cs = np.array([r['mB'] for r in P[k][c*NS:(c+1)*NS]])
            ch = 100 * (cs - bs) / bs; ch = ch[np.isfinite(ch)]
            if len(ch) < 3: continue
            tot += 1; rel += abs(ch.mean()) > 2 * ch.std(ddof=1) / np.sqrt(len(ch))
        out[f'sign_reliable_{k}'] = [int(rel), tot]; print('sign reliable', k, rel, tot)
json.dump(out, open('robust_summary.json', 'w'), indent=1)
