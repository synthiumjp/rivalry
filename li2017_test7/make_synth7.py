"""Synthetic check of the analysis: the dry run's configurations replicated three times (12 per model), with imposed
effects on the competitor (merged and other scorings) and the attended grating (mA)."""
import json, copy, numpy as np
d = json.load(open('dry7.json'))
S = copy.deepcopy(d)
rng = np.random.default_rng(7)
F = dict(gated=0.03, ungated=-0.15, replay_self=0.0, replay_masked=0.01)
for m, M in S['models'].items():
    R = M['results']
    for key in ('accepted', 'r_ab_accept', 'cv_accept', 'T_accept'):
        M[key] = [copy.deepcopy(x) for x in M[key] for _ in range(3)]
    for key in ('base', 'T'):
        R[key] = [copy.deepcopy(x) for x in R[key] for _ in range(3)]
    for grp in ('duty', 'dose_comp', 'cover_att', 'dists', 'agree', 'serial'):
        for k in R[grp]:
            R[grp][k] = [copy.deepcopy(x) for x in R[grp][k] for _ in range(3)]
    for k in R['cond']:
        R['cond'][k] = []
        for c in range(len(R['base'])):
            e = F[k] + rng.normal(0, 0.01)
            cc = copy.deepcopy(R['base'][c])
            for x in cc:
                for key in ('mBm', 'mB', 'mBabs', 'mOB'):
                    if x[key] == x[key]:
                        x[key] = x[key] * (1 + e)
                if x['mA'] == x['mA'] and k == 'gated':
                    x['mA'] = x['mA'] * 1.4
            R['cond'][k].append(cc)
json.dump(S, open('synth7.json', 'w'), default=float)
