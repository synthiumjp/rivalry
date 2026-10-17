import sys, json, numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-claude/cbc9feeb-684e-5313-8d29-01263ff765f5/scratchpad/run5')
import run_li2017_test3 as R3, run_li2017_test5 as R5
D = json.load(open('/tmp/claude-0/-home-claude/cbc9feeb-684e-5313-8d29-01263ff765f5/scratchpad/run5/codecheck5.json'))
ci = int(sys.argv[1]); c = float(sys.argv[2]); cfg = D['accepted'][ci]
t_ms = 60000.0; seeds = list(range(8)); b = 5000
rb0, *_ = R5.run_batch5([cfg], seeds, 'none', 0.0, None, None, 'chk', t_ms=t_ms)
thr = R3.THETA_FRAC * rb0[:, :, b:].mean(axis=(1, 2)); ta = np.full(8, 1.0)
res = {}
for mode, inc in (('none', 0.0), ('gated', 0.1), ('att_early', 0.1), ('att_late', 0.1)):
    rb, on, rp, rq = R5.run_batch5([cfg], seeds, mode, inc, thr, np.array([1000.0]), 'chk', c=np.array([c]), t_ms=t_ms)
    M = [R3.measures(rb[i, :, b:].astype(float), on[i, b:], rp[i, b:], rq[i, b:], thr[i], ta[i]) for i in range(8)]
    res[mode] = (np.mean([m['mBm'] for m in M]), np.mean([m['mA'] for m in M]), on[:, b:].mean(), np.mean([m['dose_att'] for m in M]))
b0 = res['none']
for k, v in res.items():
    print(f'{k:10s} mBm {v[0]:7.1f} ({100*(v[0]/b0[0]-1):+5.1f}%)  mA {v[1]:7.1f} ({100*(v[1]/b0[1]-1):+5.1f}%)  duty {v[2]:.4f}')
