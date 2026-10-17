"""
Fourth test, with a pre-specified analysis plan, in the attention model of binocular rivalry of Li, Rankin, Rinzel, Carrasco
and Heeger (2017): does a linear-response kernel, measured from the intrinsic input noise of baseline
runs with no increment, predict the effects of increments on the competitor? See analysis_plan_li2017_test4.md.

    python run_li2017_test4.py --dry-run --n-config 4 --t-ms 30000 --chunk 2 --seeds 4 --kernel-seeds 16   # code path, increments 0
    python run_li2017_test4.py --workers 2                                                                 # the registered run
The chunk size (10) is part of the registered design: it sets the noise streams and the acceptance batches.
The number of workers does not affect the results.

Uses li2017.py (unchanged from the second and third tests), the third test's code for acceptance,
dominance, the simulated observer, measures and distribution summaries (run_li2017_test3.py, unchanged),
and sim_noise.py, which is li2017.simulate without increment and with the attended-input noise recorded
(identical trajectories; checked at start-up).
"""
import argparse, json, time
import numpy as np
import li2017 as L
import sim_noise as SN
import run_li2017_test3 as R3

SEED = 20261023            # main sample of this test (third test: 20261020, 20261021, 20261022)
DRY_SEED = 104
N_TARGET = 60
N_SEEDS = 32               # seeds per condition
KERNEL_SEEDS = 192         # seeds of the kernel baseline (separate noise)
FRACS_SMALL, FRAC_MID, FRAC_LARGE = 0.05, 0.10, 0.20
BIN, HA, HB = 0.125, 3.0, 4.0   # kernel bins (units of T) and horizons
MIN_EP = 50                 # ms, as in the second and third tests (recorded at 1 ms)
BURN = 5000                 # ms

CONDS = ([('gated', FRACS_SMALL, None), ('yoked', FRACS_SMALL, None), ('ungated', FRACS_SMALL, None)] +
         [('win', FRACS_SMALL, w) for w in R3.WINDOWS] + [('late', FRACS_SMALL, None)] +
         [('gated', FRAC_MID, None), ('ungated', FRAC_MID, None)] +
         [('gated', FRAC_LARGE, None), ('ungated', FRAC_LARGE, None)] +
         [('win', FRAC_LARGE, w) for w in R3.WINDOWS] + [('late', FRAC_LARGE, None)])


# ---------------------------------------------------------------- conditions
def run_batch4(cfgs, seeds, mode, inc_frac, thr, T, tag, dL=None, extra=None, t_ms=120000.0, ranges=R3.RANGES, yoke=None):
    """As R3.run_batch, plus a yoked mode: the increment follows a recorded gate trace (1 ms samples) of another
    seed of the same configuration, so it has the live gate's statistics but no relation to the present trial;
    the third test's schedule runs in mode 'none' alongside, so the observers report as in every condition."""
    p = L.default_params(1); p['T'] = t_ms
    N = len(cfgs) * len(seeds)
    nt = int(round(t_ms / p['dt'])) + 1
    rep = lambda k: np.repeat([c[k] for c in cfgs], len(seeds))
    pv = dict(in1=rep('inp'), in2=rep('inp'), wo=rep('wo'), wa=rep('wa'), wh=rep('wh'), tau_h=rep('tau_h'))
    key = tuple(round(c[k], 6) for c in cfgs for k in sorted(ranges))
    s = R3.seed_of(tag, tuple(seeds), key)
    inst = [(tuple(round(c[k], 6) for k in sorted(ranges)), sd) for c in cfgs for sd in seeds]
    pools = (R3.latency_pools(inst, tag, 1, R3.LAT_MS), R3.latency_pools(inst, tag, 2, R3.LAT2_MS))
    sch = R3.Schedule3('none' if mode == 'yoked' else mode, thr if thr is not None else np.full(N, np.inf), p['dt'], N, nt,
                    T=None if T is None else np.repeat(T, len(seeds)),
                    dL=None if dL is None else np.repeat(dL, len(seeds)), extra=extra, pools=pools)
    if mode == 'yoked':
        def schedule(idx, state):
            sch(idx, state)                      # mode 'none': observers and episode tracking only
            j = min(idx // 2, yoke.shape[1] - 1)
            return yoke[:, j].astype(float)
    else:
        schedule = sch
    rb, on = L.simulate(p, N=N, seed=s, noise_sigma=1.0, record_every=2, params_vec=pv,
                        inc=rep('inp') * inc_frac, schedule=schedule, noise_scale=rep('noise'))
    k = rb.shape[2]
    return rb, on, sch.obs.rec[:, :k], sch.obs2.rec[:, :k]


def run_chunk4(sub, scale, t_ms, tag, n_seeds):
    """As R3.run_chunk, with N_SEEDS seeds, the conditions of this test, and the yoked condition built from
    the realised gate trace of the 5% gated condition (each instance follows the next seed of its own
    configuration)."""
    N_SEEDS = n_seeds
    seeds = list(range(N_SEEDS))
    b = int(R3.BURN_MS / R3.DT_REC)
    rb0, _, _, _ = run_batch4(sub, seeds, 'none', 0.0, None, None, tag, t_ms=t_ms)
    thr = R3.THETA_FRAC * rb0[:, :, b:].mean(axis=(1, 2))
    thr_abs = np.repeat(np.median(rb0[:, 1, b:], axis=1).reshape(len(sub), N_SEEDS).mean(1), N_SEEDS)
    rb0b, on0, rp0, rq0 = run_batch4(sub, seeds, 'none', 0.0, thr, None, tag, t_ms=t_ms)
    assert np.array_equal(rb0, rb0b)
    del rb0b
    M0 = [R3.measures(rb0[i, :, b:].astype(np.float64), on0[i, b:], rp0[i, b:], rq0[i, b:], thr[i], thr_abs[i]) for i in range(len(thr))]
    del rb0, rp0, rq0
    nc = len(sub)
    per = lambda M, c: [{k: M[i][k] for k in R3.SCAL} for i in range(c * N_SEEDS, (c + 1) * N_SEEDS)]
    base = [per(M0, c) for c in range(nc)]
    T = np.array([0.5 * (np.nanmean([x['mA'] for x in base[c]]) + np.nanmean([x['mB'] for x in base[c]])) for c in range(nc)])
    Bm0 = [[l for i in range(c * N_SEEDS, (c + 1) * N_SEEDS) for l in M0[i]['Bm']] for c in range(nc)]
    dL = np.array([R3.match_dL(Bm0[c], T[c]) for c in range(nc)])
    out, yoke, q = {}, None, None
    for mode, frac, extra in CONDS:
        key = R3.cond_key(mode, frac, extra)
        if mode == 'yoked':
            assert yoke is not None
        rb, on, rp, rq = run_batch4(sub, seeds, mode, frac * scale, thr, T, tag, dL=dL, extra=extra, t_ms=t_ms, yoke=yoke)
        if mode == 'gated' and frac == FRACS_SMALL:
            idx = np.roll(np.arange(len(thr)).reshape(nc, N_SEEDS), -1, axis=1).ravel()
            g = (rb[:, 0, :] - rb[:, 1, :]) > thr[:, None]       # realised gate (same rule as 'gated')
            yoke = g[idx]
            q = g[:, b:].mean(1).reshape(nc, N_SEEDS).mean(1)       # fraction of time the 5% gate is on, per configuration
            del g
        M = [R3.measures(rb[i, :, b:].astype(np.float64), on[i, b:], rp[i, b:], rq[i, b:], thr[i], thr_abs[i]) for i in range(len(thr))]
        out[key] = [per(M, c) for c in range(nc)]
        del rb, on, rp, rq, M
    del yoke
    return base, out, T.tolist(), dL.tolist(), q.tolist()


# ---------------------------------------------------------------- kernel
def merged_eps(st):
    """Merged competitor episodes (start, end) in 1 ms samples, as in merged scoring: an episode starts at the
    first competitor run of at least 50 ms after a qualifying attended run (at least 50 ms) and ends at the
    start of the next qualifying attended run. Also returns the starts of qualifying attended runs."""
    ch = np.flatnonzero(np.diff(st)) + 1
    s0 = np.r_[0, ch]; e0 = np.r_[ch, len(st)]; v = st[s0]; Ls = e0 - s0
    eps, att, started, cur0 = [], [], False, None
    for a, l, ss in zip(v, Ls, s0):
        if a == 1 and l >= MIN_EP:
            if started and cur0 is not None:
                eps.append((cur0, ss))
            att.append(ss); started, cur0 = True, None
        elif a == -1 and l >= MIN_EP and started and cur0 is None:
            cur0 = ss
    return eps, att


def ols(X, y):
    Xc = np.c_[np.ones(len(X)), X]
    b, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    res = y - Xc @ b
    cov = np.linalg.pinv(Xc.T @ Xc) * res.var(ddof=Xc.shape[1])
    return b[1:], np.sqrt(np.clip(np.diag(cov), 0, None))[1:]


def kernel(c, seeds_n, t_ms, tag):
    """Baseline runs (no increment) with the attended-input noise recorded; two regressions (Section 5 of the
    registration):
      A: competitor duration on the mean attended-input noise in bins of BIN*T over [0, HA*T) after the
         competitor episode's start;
      B: competitor duration on bin means of noise x 1[attended dominant] and noise x 1[not], over [0, HB*T)
         after the start of the preceding attended episode."""
    p = L.default_params(1); p['T'] = t_ms
    N = seeds_n
    pv = dict(in1=np.full(N, c['inp']), in2=np.full(N, c['inp']), wo=np.full(N, c['wo']), wa=np.full(N, c['wa']),
              wh=np.full(N, c['wh']), tau_h=np.full(N, c['tau_h']))
    s = R3.seed_of(tag, 'kernel', tuple(round(c[k], 6) for k in sorted(R3.RANGES)))
    rb, z = SN.simulate_rec(p, N, s, pv, np.full(N, c['noise']))
    rb, z = rb[:, :, BURN:].astype(np.float64), z[:, BURN:].astype(np.float64)
    thr = R3.THETA_FRAC * rb.mean(axis=(1, 2))
    store, allD, allA = [], [], []
    for i in range(N):
        d = rb[i, 0] - rb[i, 1]
        st = np.where(d > thr[i], 1, np.where(d < -thr[i], -1, 0)).astype(np.int8)
        eps, att = merged_eps(st)
        store.append((st, eps, np.array(att)))
        allD += [e - s_ for s_, e in eps]
        for s_, e in eps:
            j = np.searchsorted(store[-1][2], s_) - 1
            if j >= 0: allA.append(s_ - store[-1][2][j])
    del rb
    if len(allD) < 650 or len(allA) < 650:
        return None
    T = 0.5 * (np.mean(allD) + np.mean(allA))
    w = BIN * T
    nA, nB = int(round(HA / BIN)), int(round(HB / BIN))
    XA, yA, XB, yB, FB = [], [], [], [], []
    for i in range(N):
        st, eps, a_sorted = store[i]; zi = z[i]; n = len(zi)
        cz = np.r_[0.0, np.cumsum(zi)]
        att_m = (st == 1).astype(float)
        cza = np.r_[0.0, np.cumsum(zi * att_m)]; czo = np.r_[0.0, np.cumsum(zi * (1 - att_m))]
        cfa = np.r_[0.0, np.cumsum(att_m)]
        for s_, e in eps:
            edges = (s_ + np.arange(nA + 1) * w).astype(int)
            if edges[-1] < n and s_ >= 1:
                # first column: the noise just before the start (controls for the noise's 100 ms autocorrelation; not used in predictions)
                XA.append(np.r_[zi[s_ - 1], (cz[edges[1:]] - cz[edges[:-1]]) / np.diff(edges)]); yA.append(e - s_)
            j = np.searchsorted(a_sorted, s_) - 1
            if j >= 0:
                a0 = a_sorted[j]
                edges = (a0 + np.arange(nB + 1) * w).astype(int)
                if edges[-1] < n and a0 >= 1:
                    dw = np.diff(edges)
                    XB.append(np.r_[zi[a0 - 1], (cza[edges[1:]] - cza[edges[:-1]]) / dw, (czo[edges[1:]] - czo[edges[:-1]]) / dw])
                    FB.append((cfa[edges[1:]] - cfa[edges[:-1]]) / dw); yB.append(e - s_)
    XA, yA, XB, yB, FB = map(np.asarray, (XA, yA, XB, yB, FB))
    if len(yA) < 650 or len(yB) < 650:
        return None
    bA, seA = ols(XA, yA); bB, seB = ols(XB, yB)
    bA, seA, bB, seB = bA[1:], seA[1:], bB[1:], seB[1:]     # drop the start-noise covariate
    return dict(T=float(T), meanD=float(np.mean(allD)), nA=int(len(yA)), nB=int(len(yB)), bA=bA.tolist(), seA=seA.tolist(),
                bB=bB.tolist(), seB=seB.tolist(), fatt=FB.mean(0).tolist(), sd_noise=float(z.std()))


def check_sim_noise():
    p = L.default_params(1); p['T'] = 2000.0
    N = 3; pv = dict(in1=np.full(N, 0.5), in2=np.full(N, 0.5), wo=np.full(N, 0.6), wa=np.full(N, 0.6),
                     wh=np.full(N, 2.0), tau_h=np.full(N, 2000.0))
    a, _ = L.simulate(p, N=N, seed=5, noise_sigma=1.0, record_every=2, params_vec=pv, noise_scale=np.full(N, 0.05))
    b, _ = SN.simulate_rec(p, N, 5, pv, np.full(N, 0.05))
    assert np.array_equal(a, b), 'sim_noise differs from li2017.simulate'


def main():
    global N_SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--n-config', type=int, default=N_TARGET)
    ap.add_argument('--chunk', type=int, default=10)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--t-ms', type=float, default=120000.0)
    ap.add_argument('--seeds', type=int, default=N_SEEDS)
    ap.add_argument('--kernel-seeds', type=int, default=KERNEL_SEEDS)
    ap.add_argument('--out', default='li2017_test4_results.json')
    a = ap.parse_args()
    check_sim_noise()
    N_SEEDS = a.seeds
    t0 = time.time()
    s_main = DRY_SEED if a.dry_run else SEED
    scale = 0.0 if a.dry_run else 1.0
    acc, nd = R3.accept(a.n_config, s_main, R3.RANGES, a.chunk, a.t_ms, 'test4', True)   # 8 seeds, as in the third test
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        kern = list(ex.map(kernel, acc, [a.kernel_seeds] * len(acc), [a.t_ms] * len(acc), ['test4'] * len(acc)))
    print(f'  kernels done [{time.time() - t0:.0f}s]', flush=True)
    json.dump(dict(accepted=acc, n_drawn=nd, kernel=kern), open(a.out + '.kernels.json', 'w'), default=float)   # checkpoint
    chunks = [acc[i:i + a.chunk] for i in range(0, len(acc), a.chunk)]
    Rz = dict(base=[], cond={}, T=[], dL=[], q=[])
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_chunk4, ch, scale, a.t_ms, 'test4', N_SEEDS) for ch in chunks]
        for k, fu in enumerate(futs):
            base, out, T, dL, q = fu.result()
            Rz['base'] += base; Rz['T'] += T; Rz['dL'] += dL; Rz['q'] += q
            for kk, v in out.items(): Rz['cond'].setdefault(kk, []).extend(v)
            print(f'  chunk {k + 1}/{len(chunks)} [{time.time() - t0:.0f}s]', flush=True)
            json.dump(dict(chunks_done=k + 1, results=Rz), open(a.out + '.partial.json', 'w'), default=float)   # checkpoint
    import scipy, platform
    json.dump(dict(dry_run=a.dry_run, n_drawn=nd, accepted=acc, kernel=kern, results=dict(main=Rz),
                   versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                   settings=dict(SEED=SEED, RANGES=R3.RANGES, N_SEEDS=N_SEEDS, KERNEL_SEEDS=a.kernel_seeds,
                                 FRACS=[FRACS_SMALL, FRAC_MID, FRAC_LARGE], BIN=BIN, HA=HA, HB=HB,
                                 WINDOWS=R3.WINDOWS, T_MS=a.t_ms, CHUNK=a.chunk, CONDS=[list(map(str, c)) for c in CONDS])),
              open(a.out, 'w'), default=float)
    print(f'wrote {a.out} [{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
