import json, numpy as np
P = json.load(open('pilot_kernel.json')); D = json.load(open('../../li2017_test3/li2017_test3_results.json'))
M = D['results']['main']; S_ = D['settings']
BIN, HA, HB = P['settings']['BIN'], P['settings']['HA'], P['settings']['HB']

def change(key, m='mBm'):
    out = []
    for rb, rc in zip(M['base'], M['cond'][key]):
        b = np.array([x[m] for x in rb], float); c = np.array([x[m] for x in rc], float)
        ok = np.isfinite(b) & np.isfinite(c)
        out.append(100 * (c[ok].mean() - b[ok].mean()) / b[ok].mean() if ok.any() else np.nan)
    return np.array(out)

def win(b, w, lo, hi):
    edges = np.arange(len(b) + 1) * w
    ov = np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0, None) / w
    return float(np.sum(np.asarray(b) * ov))

keys = [f'win_0.2_{a:g}-{b:g}' for a, b in S_['WINDOWS']] + ['late_0.2', 'gated_0.1', 'ungated_0.1', 'gated_0.2', 'ungated_0.2']
obs = {k: change(k) for k in keys}
rows = []
for r in P['out']:
    k = r['k']; c = P['cfgs'][k]; T3 = M['T'][k]; dL = M['dL'][k]; w = BIN * r['T']
    bA = np.array(r['bA']); nB = len(r['fatt']); bBa, bBo = np.array(r['bB'][:nB]), np.array(r['bB'][nB:]); f = np.array(r['fatt'])
    pr = {}
    for frac in (0.1, 0.2):
        inc = frac * c['inp']; sc = 100 * inc / r['meanD']
        pr[f'gated_{frac:g}'] = sc * np.sum(bBa * f)
        pr[f'ungated_{frac:g}'] = sc * np.sum(bBa * f + bBo * (1 - f))
    sc = 100 * 0.2 * c['inp'] / r['meanD']
    for a, b in S_['WINDOWS']:
        pr[f'win_0.2_{a:g}-{b:g}'] = sc * win(bA, w, a * T3, b * T3)
    pr['late_0.2'] = sc * win(bA, w, dL, HA * r['T'])
    rows.append((k, pr))
print('per configuration: predicted / observed (% change, merged scoring)')
print('cfg ' + ' '.join(f'{x[:14]:>16s}' for x in keys))
P_ = {x: [] for x in keys}; O_ = {x: [] for x in keys}
for k, pr in rows:
    print(f'{k:3d} ' + ' '.join(f'{pr[x]:+7.1f}/{obs[x][k]:+6.1f}  ' for x in keys))
    for x in keys: P_[x].append(pr[x]); O_[x].append(obs[x][k])
print('median pred / median obs:')
for x in keys:
    print(f'  {x:18s} {np.median(P_[x]):+7.1f} {np.nanmedian(O_[x]):+7.1f}')
# kernel shape (competitor-start aligned), median across configs, in T units, % per unit-input*0.1inp
print('competitor-aligned kernel (median across configs), effect of 10%-input over one bin, % of D:')
K = np.array([100 * 0.1 * P['cfgs'][r['k']]['inp'] * np.array(r['bA']) / r['meanD'] for r in P['out']])
Z = np.array([np.array(r['bA']) / np.array(r['seA']) for r in P['out']])
for j in range(K.shape[1]):
    print(f'  t={j*BIN:.3f}-{(j+1)*BIN:.3f}T  {np.median(K[:, j]):+6.2f}%   median z {np.median(Z[:, j]):+5.1f}')
