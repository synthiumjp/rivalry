import json, numpy as np
A = json.load(open('amp_pilot.json')); P = json.load(open('pilot_kernel.json'))
D = json.load(open('../../li2017_test3/li2017_test3_results.json')); W = D['settings']['WINDOWS']
BIN, HA = P['settings']['BIN'], P['settings']['HA']
def change(key, m):
    out = []
    for rb, rc in zip(A['base'], A['cond'][key]):
        b = np.array([x[m] for x in rb], float); c = np.array([x[m] for x in rc], float)
        ok = np.isfinite(b) & np.isfinite(c)
        out.append(100 * (c[ok].mean() - b[ok].mean()) / b[ok].mean() if ok.any() else np.nan)
    return np.array(out)
def win(b, w, lo, hi):
    edges = np.arange(len(b) + 1) * w
    ov = np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0, None) / w
    return float(np.sum(np.asarray(b) * ov))
names = [f'{a:g}-{b:g}' for a, b in W] + ['late']
for m in ('mBm', 'mOB'):
    print(f'\n=== scoring {m}: median across 10 configs; observed (kernel linear prediction); obs/pred ratio')
    for f in (0.05, 0.1, 0.2):
        obs = [change(f'win_{f:g}_{a:g}-{b:g}', m) for a, b in W] + [change(f'late_{f:g}', m)]
        pr = []
        for r in P['out']:
            k = r['k']; c = P['cfgs'][k]; T3 = A['T'][k]; dL = A['dL'][k]; w = BIN * r['T']
            sc = 100 * f * c['inp'] / r['meanD']; bA = np.array(r['bA'])
            pr.append([sc * win(bA, w, a * T3, b * T3) for a, b in W] + [sc * win(bA, w, dL, HA * r['T'])])
        pr = np.array(pr)
        o = np.array(obs).T
        print(f' {f:4.2f}: ' + '  '.join(f'{n}: {np.nanmedian(o[:,j]):+6.1f} ({np.median(pr[:,j]):+6.1f})' for j, n in enumerate(names)))
        # profile shape: share of total across windows, observed vs kernel
        so = np.nanmedian(o[:, :4] / o[:, :4].sum(1, keepdims=True), 0); sp = np.median(pr[:, :4] / pr[:, :4].sum(1, keepdims=True), 0)
        print('        share of four-window total, obs: ' + ' '.join(f'{x:.2f}' for x in so) + '   kernel: ' + ' '.join(f'{x:.2f}' for x in sp))
    for f in (0.05, 0.1, 0.2):
        g, u = change(f'gated_{f:g}', m), change(f'ungated_{f:g}', m)
        print(f'  gated {f:4.2f}: {np.nanmedian(g):+6.1f} (lengthens {np.sum(g>0)}/10); continuous {np.nanmedian(u):+6.1f}; gated per unit amplitude {np.nanmedian(g)/f:+6.1f}')
