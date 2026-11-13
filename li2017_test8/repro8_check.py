"""Check that the first chunk of the seventh test's configurations of one model reproduces the seventh test's baseline
exactly at the run's settings.  python repro8_check.py sfa"""
import json, sys, numpy as np
sys.path.insert(0, '.')
import run_li2017_test8 as T8, run_li2017_test3 as R3, run_li2017_test7 as T7
D7 = json.load(open('../li2017_test7/li2017_test7_results.json'))
model = sys.argv[1]
acc = D7['models'][model]['accepted'][:10]
seeds = list(range(32)); b = int(R3.BURN_MS / R3.DT_REC)
rb0 = T8.run_batch8(model, acc, seeds, 'none', 0.0, None, f'test7_{model}', 120000.0, T7.RANGES[model])[0]
thr = R3.THETA_FRAC * rb0[:, :, b:].mean(axis=(1, 2))
rb, on, rp, rq, ex = T8.run_batch8(model, acc, seeds, 'none', 0.0, thr, f'test7_{model}', 120000.0, T7.RANGES[model])
if model == 'li':
    thr_abs = np.repeat(np.median(rb0[:, 1, b:], axis=1).reshape(10, 32).mean(1), 32)
else:
    mid = 0.5 * (np.percentile(rb0[:, 1, b:], 10, axis=1) + np.percentile(rb0[:, 1, b:], 90, axis=1)); thr_abs = np.repeat(mid.reshape(10, 32).mean(1), 32)
M0 = [R3.measures(rb[i, :, b:].astype(np.float64), on[i, b:], rp[i, b:], rq[i, b:], thr[i], thr_abs[i]) for i in range(len(thr))]
B7 = D7['models'][model]['results']['base'][:10]
mx = 0
for c in range(10):
    for s in range(32):
        for m in ('mA', 'mBm', 'mOB', 'mOA'):
            a, bb = M0[c*32+s][m], B7[c][s][m]
            if np.isfinite(a) or np.isfinite(bb):
                mx = max(mx, abs(a-bb)/abs(bb) if np.isfinite(a) and np.isfinite(bb) else np.inf)
print(model, 'max rel diff vs test 7 baseline, first chunk:', mx)
