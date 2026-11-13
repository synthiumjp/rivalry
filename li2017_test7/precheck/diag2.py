"""Pre-check: merged-scoring dose (R3 'dose_comp': on-state outside attended runs >= 50 ms) and brief grating-1
intrusions (<50 ms) per minute, baseline vs conditions; one accepted pilot configuration per model, 4 seeds x 30 s."""
import sys, json
sys.path.insert(0, '/home/claude/chris/work/t7/li2017_test7')
import numpy as np
import run_li2017_test7 as T7, run_li2017_test3 as R3, run_li2017_test6 as T6
src = '/home/claude/chris/work/t7/src6/'
pick = dict(li=('pilot7b_li', 2526), sfa=('pilot7b_sfa', 1911), dep=('pilot7c_dep', 946))
T_MS, NS = 30000.0, 4
for model, (f, Tt) in pick.items():
    d = json.load(open(src + f + '.json'))
    cfg = [x for x in d if x['T'] == x['T'] and round(x['T']) == Tt][0]['cfg']
    seeds = list(range(NS)); tag = 'precheck7_' + model; rg = T7.RANGES[model]
    b0 = int(R3.BURN_MS / R3.DT_REC); mins = NS * (T_MS - R3.BURN_MS) / 60000
    rb0 = T7.run_batch7(model, [cfg], seeds, 'none', 0.0, None, tag, t_ms=T_MS, ranges=rg)[0]
    thr = R3.THETA_FRAC * rb0[:, :, b0:].mean(axis=(1, 2))
    _, on0, rp0, rq0, ex0, Arec = T7.run_batch7(model, [cfg], seeds, 'none_rec', 0.0, thr, tag, t_ms=T_MS, ranges=rg)
    bi0 = sum(T6.brief_intrusions(rb0[i, :, b0:].astype(float), thr[i]) for i in range(NS)) / mins
    print(model, 'baseline brief intrusions/min %.1f' % bi0)
    for name, mode in T7.CONDS:
        rec = Arec if mode.startswith('replay') else None
        rb, on, rp, rq, ex = T7.run_batch7(model, [cfg], seeds, mode, T7.INC, thr, tag, rec=rec, t_ms=T_MS, ranges=rg)
        M = [R3.measures(rb[i, :, b0:].astype(float), on[i, b0:], rp[i, b0:], rq[i, b0:], thr[i], thr[i]) for i in range(NS)]
        bi = sum(T6.brief_intrusions(rb[i, :, b0:].astype(float), thr[i]) for i in range(NS)) / mins
        print('  %-14s merged-scoring dose_comp %.3f  brief intrusions/min %.1f  nBm %s' % (
            name, np.nanmean([m['dose_comp'] for m in M]), bi, [m['nBm'] for m in M]))
