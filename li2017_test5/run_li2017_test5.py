"""
Fifth test, with a pre-specified analysis plan, in the attention model of binocular rivalry of Li, Rankin, Rinzel, Carrasco
and Heeger (2017). See analysis_plan_li2017_test5.md. Two questions:
  1. Does the gated lengthening of the competitor come from the increment's presence at the handover? The gate is split
     at a fixed time c after the attended grating becomes dominant, chosen so that the two halves have the same expected
     time on: the early half is off before most handovers, the late half is on at every handover.
  2. Does the kernel's time course hold for brief pulses? Pulses of 0.125 T at six positions in the competitor's episode.

    python run_li2017_test5.py --dry-run --n-config 4 --t-ms 30000 --chunk 2 --seeds 4 --kernel-seeds 16   # code path, increments 0
    python run_li2017_test5.py --workers 6                                                                 # the run
The chunk size (10) is part of the design (it sets the noise streams and the acceptance batches); the number of
workers does not affect the results.

Reuses, unchanged: li2017.py, run_li2017_test3.py (acceptance, dominance, observers, measures, late-window delay) and
run_li2017_test4.py (kernel estimation, merged episodes, the start-up check of sim_noise.py).
"""
import argparse, json, time
import numpy as np
import li2017 as L
import run_li2017_test3 as R3
import run_li2017_test4 as R4

SEED = 20261024            # main sample of this test (fourth test: 20261023)
DRY_SEED = 105
N_TARGET = 60
N_SEEDS = 32
KERNEL_SEEDS = 192
INC = 0.10
PULSE_W = 0.125                                   # pulse width, units of T
PULSES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)        # pulse onsets after the competitor episode begins, units of T

TRANS_K = 3.0                                     # late half without the transition: off once the advantage is below 3 thr

CONDS = ([('gated', INC, None), ('att_early', INC, None), ('att_late', INC, None), ('att_late_nt', INC, None),
          ('ungated', INC, None)] + [('pulse', INC, (p, PULSE_W)) for p in PULSES])
MIN_EP = 50.0


class Schedule5(R3.Schedule3):
    """The third test's schedule, run in mode 'none' so that the observers proceed exactly as in every condition, plus
    two trackers defined as in merged scoring (and as in the kernel's episodes):
      competitor episode: begins when a grating-2 run reaches 50 ms (dated from that run's start) and ends when a
        grating-1 run reaches 50 ms;
      attended period: begins when a grating-1 run reaches 50 ms outside a competitor episode (dated from that run's
        start) and ends when the next competitor episode begins.
    Modes:
      att_early:   grating 1 dominant, in the attended period, less than c into it
      att_late:    grating 1 dominant, in the attended period, at least c into it
      att_late_nt: as att_late, and grating 1's advantage above TRANS_K thresholds (off through the transition)
      pulse:       during [a, a + w) T of each competitor episode, while grating 1 is not dominant"""

    def __init__(self, mode5, c, *args, **kw):
        super().__init__('none', *args, **kw)
        self.mode5, self.c = mode5, c
        N = args[2]
        self.b_run = np.zeros(N); self.in_c5 = np.zeros(N, bool); self.since5 = np.zeros(N)
        self.in_att = np.zeros(N, bool); self.att_since = np.zeros(N)
        if mode5 == 'att_probe':
            nrec = (args[3] - 1) // 2 + 1
            self.probe_att = np.zeros((N, nrec), np.float32); self.probe_on = np.zeros((N, nrec), bool)

    def __call__(self, idx, state):
        super().__call__(idx, state)                      # mode 'none': observers, a_run
        rb = state['rb']; d = rb[:, 0] - rb[:, 1]
        A, B = d > self.thr, d < -self.thr
        dt = self.dt
        self.b_run = np.where(B, self.b_run + dt, 0.0)
        c_start = ~self.in_c5 & (self.b_run >= MIN_EP)
        c_end = self.in_c5 & (self.a_run >= MIN_EP)
        self.in_c5 = (self.in_c5 | c_start) & ~c_end
        self.since5 = np.where(c_start, self.b_run, np.where(self.in_c5, self.since5 + dt, 0.0))
        a_start = ~self.in_att & ~self.in_c5 & (self.a_run >= MIN_EP)
        self.in_att = (self.in_att | a_start) & ~c_start
        self.att_since = np.where(a_start, self.a_run, np.where(self.in_att, self.att_since + dt, 0.0))
        m = self.mode5
        on_att = A & self.in_att
        if m == 'att_probe':                             # zero increment; records time into the attended period
            if idx % 2 == 0 and idx // 2 < self.probe_att.shape[1]:
                self.probe_att[:, idx // 2] = self.att_since; self.probe_on[:, idx // 2] = on_att
            return np.zeros(len(d))
        if m == 'att_early':
            return (on_att & (self.att_since < self.c)).astype(float)
        if m == 'att_late':
            return (on_att & (self.att_since >= self.c)).astype(float)
        if m == 'att_late_nt':
            return (on_att & (self.att_since >= self.c) & (d > TRANS_K * self.thr)).astype(float)
        if m == 'pulse':
            a, w = self.extra
            return (self.in_c5 & ~A & (self.since5 >= a * self.T) & (self.since5 < (a + w) * self.T)).astype(float)
        raise ValueError(m)


def run_batch5(cfgs, seeds, mode, inc_frac, thr, T, tag, c=None, extra=None, t_ms=120000.0, ranges=R3.RANGES):
    """As the third test's run_batch, with Schedule5 for the new modes and Schedule3 otherwise."""
    p = L.default_params(1); p['T'] = t_ms
    N = len(cfgs) * len(seeds)
    nt = int(round(t_ms / p['dt'])) + 1
    rep = lambda k: np.repeat([cc[k] for cc in cfgs], len(seeds))
    pv = dict(in1=rep('inp'), in2=rep('inp'), wo=rep('wo'), wa=rep('wa'), wh=rep('wh'), tau_h=rep('tau_h'))
    key = tuple(round(cc[k], 6) for cc in cfgs for k in sorted(ranges))
    s = R3.seed_of(tag, tuple(seeds), key)
    inst = [(tuple(round(cc[k], 6) for k in sorted(ranges)), sd) for cc in cfgs for sd in seeds]
    pools = (R3.latency_pools(inst, tag, 1, R3.LAT_MS), R3.latency_pools(inst, tag, 2, R3.LAT2_MS))
    thr_ = thr if thr is not None else np.full(N, np.inf)
    Tn = None if T is None else np.repeat(T, len(seeds))
    if mode in ('att_early', 'att_late', 'att_late_nt', 'pulse', 'att_probe'):
        sch = Schedule5(mode, np.repeat(c, len(seeds)), thr_, p['dt'], N, nt, T=Tn, extra=extra, pools=pools)
    else:
        sch = R3.Schedule3(mode, thr_, p['dt'], N, nt, T=Tn, extra=extra, pools=pools)
    rb, on = L.simulate(p, N=N, seed=s, noise_sigma=1.0, record_every=2, params_vec=pv,
                        inc=rep('inp') * inc_frac, schedule=sch, noise_scale=rep('noise'))
    k = rb.shape[2]
    if mode == 'att_probe':
        return sch.probe_att[:, :k], sch.probe_on[:, :k]
    return rb, on, sch.obs.rec[:, :k], sch.obs2.rec[:, :k]


def probe_split(sub, seeds, thr, tag, t_ms, b):
    """Split point c per configuration: the median, over baseline samples at which grating 1 is dominant within an
    attended period, of the time into that period, recorded online by Schedule5 itself in a run with zero increment
    (identical to the baseline), so that the early and late halves are on for equal time at baseline."""
    att, on = run_batch5(sub, seeds, 'att_probe', 0.0, thr, None, tag, c=np.full(len(sub), np.inf), t_ms=t_ms)
    n_seeds = len(seeds); c = []
    for k in range(len(sub)):
        sl = slice(k * n_seeds, (k + 1) * n_seeds)
        v = att[sl, b:][on[sl, b:]]
        c.append(float(np.median(v)) if len(v) >= 1000 else np.nan)
    return np.array(c)


def run_chunk5(sub, scale, t_ms, tag, n_seeds):
    seeds = list(range(n_seeds))
    b = int(R3.BURN_MS / R3.DT_REC)
    rb0, _, _, _ = run_batch5(sub, seeds, 'none', 0.0, None, None, tag, t_ms=t_ms)
    thr = R3.THETA_FRAC * rb0[:, :, b:].mean(axis=(1, 2))
    thr_abs = np.repeat(np.median(rb0[:, 1, b:], axis=1).reshape(len(sub), n_seeds).mean(1), n_seeds)
    rb0b, on0, rp0, rq0 = run_batch5(sub, seeds, 'none', 0.0, thr, None, tag, t_ms=t_ms)
    assert np.array_equal(rb0, rb0b)
    del rb0b
    nc = len(sub)
    cstar = probe_split(sub, seeds, thr, tag, t_ms, b)
    M0 = [R3.measures(rb0[i, :, b:].astype(np.float64), on0[i, b:], rp0[i, b:], rq0[i, b:], thr[i], thr_abs[i]) for i in range(len(thr))]
    del rb0, rp0, rq0
    per = lambda M, cc: [{k: M[i][k] for k in R3.SCAL} for i in range(cc * n_seeds, (cc + 1) * n_seeds)]
    base = [per(M0, cc) for cc in range(nc)]
    T = np.array([0.5 * (np.nanmean([x['mA'] for x in base[cc]]) + np.nanmean([x['mB'] for x in base[cc]])) for cc in range(nc)])
    out, duty = {}, {}
    for mode, frac, extra in CONDS:
        key = R3.cond_key(mode, frac, extra)
        rb, on, rp, rq = run_batch5(sub, seeds, mode, frac * scale, thr, T, tag, c=cstar, extra=extra, t_ms=t_ms)
        M = [R3.measures(rb[i, :, b:].astype(np.float64), on[i, b:], rp[i, b:], rq[i, b:], thr[i], thr_abs[i]) for i in range(len(thr))]
        out[key] = [per(M, cc) for cc in range(nc)]
        duty[key] = on[:, b:].mean(1).reshape(nc, n_seeds).mean(1).tolist()
        del rb, on, rp, rq, M
    return base, out, T.tolist(), cstar.tolist(), duty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--n-config', type=int, default=N_TARGET)
    ap.add_argument('--chunk', type=int, default=10)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--t-ms', type=float, default=120000.0)
    ap.add_argument('--seeds', type=int, default=N_SEEDS)
    ap.add_argument('--kernel-seeds', type=int, default=KERNEL_SEEDS)
    ap.add_argument('--out', default='li2017_test5_results.json')
    a = ap.parse_args()
    R4.check_sim_noise()
    t0 = time.time()
    s_main = DRY_SEED if a.dry_run else SEED
    scale = 0.0 if a.dry_run else 1.0
    acc, nd = R3.accept(a.n_config, s_main, R3.RANGES, a.chunk, a.t_ms, 'test5', True)   # 8 seeds, as in the third test
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        kern = list(ex.map(R4.kernel, acc, [a.kernel_seeds] * len(acc), [a.t_ms] * len(acc), ['test5'] * len(acc)))
    print(f'  kernels done [{time.time() - t0:.0f}s]', flush=True)
    json.dump(dict(accepted=acc, n_drawn=nd, kernel=kern), open(a.out + '.kernels.json', 'w'), default=float)
    chunks = [acc[i:i + a.chunk] for i in range(0, len(acc), a.chunk)]
    Rz = dict(base=[], cond={}, T=[], c=[], duty={})
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_chunk5, ch, scale, a.t_ms, 'test5', a.seeds) for ch in chunks]
        for k, fu in enumerate(futs):
            base, out, T, c, duty = fu.result()
            Rz['base'] += base; Rz['T'] += T; Rz['c'] += c
            for kk, v in out.items(): Rz['cond'].setdefault(kk, []).extend(v)
            for kk, v in duty.items(): Rz['duty'].setdefault(kk, []).extend(v)
            print(f'  chunk {k + 1}/{len(chunks)} [{time.time() - t0:.0f}s]', flush=True)
            json.dump(dict(chunks_done=k + 1, results=Rz), open(a.out + '.partial.json', 'w'), default=float)
    import scipy, platform
    json.dump(dict(dry_run=a.dry_run, n_drawn=nd, accepted=acc, kernel=kern, results=dict(main=Rz),
                   versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                   settings=dict(SEED=SEED, RANGES=R3.RANGES, N_SEEDS=a.seeds, KERNEL_SEEDS=a.kernel_seeds, INC=INC,
                                 PULSE_W=PULSE_W, PULSES=PULSES, TRANS_K=TRANS_K, BIN=R4.BIN, HA=R4.HA, HB=R4.HB, T_MS=a.t_ms,
                                 CHUNK=a.chunk, CONDS=[list(map(str, cc)) for cc in CONDS])),
              open(a.out, 'w'), default=float)
    print(f'wrote {a.out} [{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
