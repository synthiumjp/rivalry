import sys, json, numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-claude/cbc9feeb-684e-5313-8d29-01263ff765f5/scratchpad/run5')
import li2017 as L, run_li2017_test3 as R3, run_li2017_test4 as R4, run_li2017_test5 as R5
D = json.load(open('/tmp/claude-0/-home-claude/cbc9feeb-684e-5313-8d29-01263ff765f5/scratchpad/run5/codecheck5.json'))
ci = int(sys.argv[1]); inc_frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
cfg = D['accepted'][ci]; t_ms = 60000.0; ns = 8; seeds = list(range(ns)); b = 5000
rb0, *_ = R5.run_batch5([cfg], seeds, 'none', 0.0, None, None, 'chk', t_ms=t_ms)
thr = R3.THETA_FRAC * rb0[:, :, b:].mean(axis=(1, 2))
Aint = []; Aint_first = []; nruns = []
for i in range(ns):
    d = rb0[i, 0, b:].astype(float) - rb0[i, 1, b:]
    st = np.where(d > thr[i], 1, np.where(d < -thr[i], -1, 0)).astype(np.int8)
    eps, att = R4.merged_eps(st); att = np.array(att)
    prev_end = None
    for s_, e in eps:
        j = np.searchsorted(att, s_) - 1
        if j >= 0:
            Aint.append(s_ - att[j])
            # first qualifying attended run after previous episode's end
            lo = prev_end if prev_end is not None else 0
            k = np.searchsorted(att, lo)
            Aint_first.append(s_ - att[k]); nruns.append(j - k + 1)
        prev_end = e
c = R5.split_point(Aint); c_first = R5.split_point(Aint_first)
print('cfg', ci, 'c(last run)', c, 'c(first run)', c_first, 'meanA', np.mean(Aint), np.mean(Aint_first), 'frac periods >1 qualifying run', np.mean(np.array(nruns) > 1))
# online: record early, late, live, in_att resets
p = L.default_params(1); p['T'] = t_ms; N = ns; nt = int(round(t_ms / p['dt'])) + 1
inst = [(tuple(round(cfg[k], 6) for k in sorted(R3.RANGES)), sd) for sd in seeds]
pools = (R3.latency_pools(inst, 'chk', 1, R3.LAT_MS), R3.latency_pools(inst, 'chk', 2, R3.LAT2_MS))
T3 = 1000.0
sch = R5.Schedule5('att_early', np.full(N, c), thr, p['dt'], N, nt, T=np.full(N, T3), pools=pools)
rec = np.zeros((3, N, (nt - 1) // 2 + 1), bool); nstart = np.zeros(N, int)
def S(idx, state):
    before = sch.in_att.copy()
    e = sch(idx, state)
    nstart[:] += (sch.in_att & ~before)
    A = (state['rb'][:, 0] - state['rb'][:, 1]) > thr
    l = A & sch.in_att & (sch.att_since >= c)
    if idx % 2 == 0:
        rec[0, :, idx // 2] = e > 0; rec[1, :, idx // 2] = l; rec[2, :, idx // 2] = A
    return (A if inc_frac else e) .astype(float)  # inc_frac>0: deliver live gate? no -- see below
pv = dict(in1=np.full(N, cfg['inp']), in2=np.full(N, cfg['inp']), wo=np.full(N, cfg['wo']), wa=np.full(N, cfg['wa']), wh=np.full(N, cfg['wh']), tau_h=np.full(N, cfg['tau_h']))
rb, on = L.simulate(p, N=N, seed=R3.seed_of('chk', tuple(seeds), tuple(round(cfg[k], 6) for k in sorted(R3.RANGES))), noise_sigma=1.0, record_every=2, params_vec=pv, inc=np.zeros(N), schedule=S, noise_scale=np.full(N, cfg['noise']))
assert np.array_equal(rb, rb0)
E, Lt, G = rec[:, :, b:rb.shape[2]]
print('duty early %.4f late %.4f live %.4f' % (E.mean(), Lt.mean(), G.mean()))
neps = 0; on_e = on_l = 0; pre_c = 0; nper = 0
for i in range(N):
    d = rb[i, 0, b:].astype(float) - rb[i, 1, b:]
    st = np.where(d > thr[i], 1, np.where(d < -thr[i], -1, 0)).astype(np.int8)
    eps, att = R4.merged_eps(st)
    for s_, e in eps:
        # last grating-1 sample before the episode start
        k = s_ - 1
        while k > 0 and st[k] != 1: k -= 1
        neps += 1; on_e += E[i, k]; on_l += Lt[i, k]
    nper += len(eps)
print('handovers', neps, 'early on at last A sample %.3f' % (on_e / neps), 'late on %.3f' % (on_l / neps), 'P(A<c)=%.3f' % np.mean(np.array(Aint) < c))
print('online attended-period starts', nstart.sum(), 'offline merged episodes', nper)
