"""Exploratory: classify each accepted configuration by whether its noise-free dynamics oscillate."""
import json, sys, numpy as np
sys.path.insert(0, '..')
import li2017 as L
d = json.load(open('../li2017_test_results.json')); acc = d['accepted']
p = L.default_params(1); p['T'] = 60000.0
N = len(acc); g = lambda k: np.array([c[k] for c in acc])
pv = dict(in1=g('inp'), in2=g('inp'), wo=g('wo'), wa=g('wa'), wh=g('wh'), tau_h=g('tau_h'))
rb, _ = L.simulate(p, N=N, seed=1, noise_sigma=0.0, record_every=2, params_vec=pv)
dt = p['dt'] * 2; b0 = int(30000 / dt)
osc, period = [], []
for i in range(N):
    x = rb[i, :, b0:].astype(float); thr = 0.05 * x.mean()
    dd = x[0] - x[1]; st = np.where(dd > thr, 1, np.where(dd < -thr, -1, 0))
    nz = st[st != 0]; sw = int(np.sum(nz[1:] != nz[:-1]))
    osc.append(sw >= 4); period.append(30000 / max(sw, 1) if sw else np.nan)
osc = np.array(osc)
print('noise-free oscillation in', osc.sum(), 'of', N)
json.dump(dict(osc=osc.tolist(), period=period), open('regime.json', 'w'), default=float)
