import sys, json, numpy as np
sys.path.insert(0, '/home/claude/chris/work/t8/li2017_test8')
P = '/home/claude/chris/work/t8/li2017_test8/'
# edge: attended change never reaches 29% (k small) -> zero calibrated; and 12 calibrated of 60
D = json.load(open('synth8_lengthen.json'))
import copy
for model, M in D['models'].items():
    R = M['results']
    for kk in R['cond']:
        for ci, cfg in enumerate(R['cond'][kk]):
            for si, x in enumerate(cfg):
                b = R['base'][ci][si]
                f = 0.5 if model == 'li' else (1.0 if (model == 'sfa' and ci < 12) else 0.1)
                x['mOA'] = b['mOA'] + f * (x['mOA'] - b['mOA'])
json.dump(D, open('edge8.json', 'w'))
sys.argv = ['x', 'edge8.json', 'edge8_tests.json', 'synth8_lengthen_ref7.json']
import analyse_li2017_test8 as A
T = json.load(open('edge8_tests.json'))
for m in ('li', 'sfa', 'dep'):
    print(m, 'n_cal', T[m]['calibrated']['mOB']['n_calibrated'], 'C2', T[m]['C2_not_shortened'])
print('decisions', T['decisions'])
# calibrate unit cases
cases = {'monotone': ([10, 20, 40], [1, 2, 4]), 'nonmono_first': ([30, 10, 50], [3, -5, 5]),
         'dip_then_cross': ([20, 10, 40], [2, -1, 4]), 'never': ([5, 10, 28.9], [1, 1, 1]),
         'nan_first': ([np.nan, 40, 50], [1, 2, 3]), 'comp_nan_at_cross': ([10, 40, 50], [1, np.nan, 3]),
         'exact_at_2.5': ([29, 40, 50], [7, 8, 9]), 'negative_att': ([-10, 5, 40], [1, 2, 3])}
for k, (a, c) in cases.items():
    sz, cc = A.calibrate(np.array([a], float), np.array([c], float))
    print(f'{k:20s} size={sz[0]:.4f} comp={cc[0]:.3f}')
