"""
Third pre-registered test of the delivery-timing account in the attention model of binocular
rivalry of Li, Rankin, Rinzel, Carrasco and Heeger (2017). See prereg_li2017_test3.md.

    python run_li2017_test3.py --dry-run --n-config 4 --n-config-na 4 --t-ms 30000 --chunk 4   # code path, increments 0
    python run_li2017_test3.py                                                                  # the registered run

Uses the port li2017.py of the second test (unchanged) and the episode helpers of run_li2017_test.py.
New in this test: a simulated observer whose reports are recorded in every condition (observer
scoring, primary), windows tiling the competitor's episode, a ramped early window, a gate with a
100 ms latency, holds at a calibrated increment under Chong et al.'s timing, and the competitor's
response while dominant.
"""
import argparse, json, time, zlib
import numpy as np
import li2017 as L
import run_li2017_test as R   # first test: RANGES, episodes_diff(), episodes_abs()

SEED = 20261020            # main sample (first test 20261007, second test 20261016)
SEED_NA = 20261021         # no-adaptation sample (second test 20261017)
SEED_HN = 20261022         # high-noise sample with adaptation (noise control)
DRY_SEED = 101             # dry run only (101 main, 102 no-adaptation, 103 high-noise)
RANGES = dict(R.RANGES)
RANGES_NA = dict(inp=(0.35, 0.8), wo=(0.5, 0.75), wa=(0.4, 0.9), wh=(0.0, 0.0),
                 tau_h=(2000.0, 2000.0), noise=(0.15, 0.30))
RANGES_HN = dict(R.RANGES, noise=(0.15, 0.30))
N_SEEDS, BURN_MS, MIN_EP_MS, THETA_FRAC = 8, 5000.0, 50.0, 0.05
INC = 0.20                 # mechanism conditions
INC_CAL = 0.10             # calibrated increment (Chong et al.'s timing)
RAMP_MS = 520.0            # Chong et al.'s ramps
LAT_MS = (200.0, 450.0)    # report latency, uniform, drawn independently for each reported switch
LAT2_MS = (300.0, 600.0)   # second, passive observer (sensitivity)
LAT_POOL = 6000            # latencies pre-drawn per instance and observer
GATE_LAT_MS = 100.0        # latency of the latency gate
EARLY_W = 0.25             # early window, units of T
RAMP_EARLY = 0.125         # ramped early window: linear rise over 0.125 T, then on until RAMP_END
RAMP_END = 0.3125          # so that its time on (0.25 T) matches the step window [0, 0.25 T)
WINDOWS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))
HOLDS = (0.25, 0.5, 0.75, 1.0, 1.5)
N_TARGET, N_TARGET_NA, N_TARGET_HN = 200, 100, 50
DT_REC = 1.0               # recorded sample spacing (ms); the integration step is 0.5 ms
GRID_MAX = 10.0            # distribution summaries on 0..GRID_MAX T


def seed_of(*parts):
    return zlib.crc32(repr(parts).encode()) % (2 ** 31 - 1)


class Observer:
    """A change of exclusive dominance is reported after a latency, if the new percept is still the one
    most recently dominant; indeterminate spells do not cancel a pending report, a return to the reported
    percept does. Latencies are pre-drawn per instance (seeded by configuration, seed and observer), so an
    instance's reports do not depend on the other instances in its batch."""

    def __init__(self, N, lat, pools, nrec):
        self.lat, self.pool = lat, pools
        self.k = np.zeros(N, np.int64)
        self.rep = np.zeros(N, np.int8)
        self.pend = np.zeros(N, np.int8)
        self.cd = np.zeros(N)
        self.since_rep = np.full(N, np.inf)
        self.rec = np.zeros((N, nrec), np.int8)

    def step(self, st, dt):
        new = (st != 0) & (st != self.rep)
        start = new & (self.pend != st)
        if start.any():
            ii = np.flatnonzero(start)
            if (self.k[ii] >= self.pool.shape[1]).any():
                raise RuntimeError('latency pool exhausted')
            self.pend[ii] = st[ii]
            self.cd[ii] = self.pool[ii, self.k[ii]]
            self.k[ii] += 1
        cancel = (st != 0) & (st == self.rep)
        self.pend[cancel] = 0
        active = self.pend != 0
        self.cd[active] -= dt
        fire = active & (self.cd <= 0)
        self.rep[fire] = self.pend[fire]
        self.pend[fire] = 0
        self.since_rep = np.where(fire, 0.0, self.since_rep + dt)


def latency_pools(inst_keys, tag, which, lat):
    return np.stack([np.random.default_rng(seed_of(tag, 'observer', which, k)).uniform(lat[0], lat[1], LAT_POOL)
                     for k in inst_keys]).astype(np.float64)


class Schedule3:
    """Increment level per instance, read from the live binocular responses each step.
    Grating 1 (attended) dominant: rb1 - rb2 > thr; grating 2: rb2 - rb1 > thr; else indeterminate.
    Merged competitor episodes are tracked online: one starts at the first grating-2 sample after a
    grating-1 run of at least MIN_EP_MS (or at the start) and ends when a grating-1 run reaches MIN_EP_MS.
    Two simulated observers run in every mode (prereg, Section 4): the primary observer (LAT_MS), whose
    reports drive the increment in the report modes, and a passive observer (LAT2_MS). Reports are recorded
    at every recorded sample."""

    def __init__(self, mode, thr, dt, N, nt, T=None, dL=None, extra=None, pools=None):
        self.mode, self.thr, self.dt = mode, thr, dt
        self.T, self.dL, self.extra = T, dL, extra
        self.a_run = np.zeros(N)
        self.comp = np.zeros(N, bool)
        self.since_comp = np.full(N, np.inf)
        self.level = np.zeros(N)
        nrec = (nt - 1) // 2 + 1
        self.obs = Observer(N, LAT_MS, pools[0], nrec)
        self.obs2 = Observer(N, LAT2_MS, pools[1], nrec)

    def __call__(self, idx, state):
        rb = state['rb']; dt = self.dt
        d = rb[:, 0] - rb[:, 1]
        A, B = d > self.thr, d < -self.thr
        st = np.where(A, 1, np.where(B, 2, 0)).astype(np.int8)
        self.a_run = np.where(A, self.a_run + dt, 0.0)
        end_comp = self.comp & (self.a_run >= MIN_EP_MS)
        self.comp = self.comp & ~end_comp
        start_comp = B & ~self.comp
        self.comp = self.comp | start_comp
        self.since_comp = np.where(start_comp, 0.0, np.where(self.comp, self.since_comp + dt, np.inf))
        self.obs.step(st, dt); self.obs2.step(st, dt)
        if idx % 2 == 0:
            self.obs.rec[:, idx // 2] = self.obs.rep
            self.obs2.rec[:, idx // 2] = self.obs2.rep
        rep, since_rep = self.obs.rep, self.obs.since_rep
        m, n, T = self.mode, len(d), self.T
        if m == 'none':
            return np.zeros(n)
        if m == 'gated':
            return A.astype(float)
        if m == 'gated_lat':
            return (A & (self.a_run >= GATE_LAT_MS)).astype(float)
        if m == 'ungated':
            return np.ones(n)
        if m == 'win':
            a, b = self.extra
            return (self.comp & ~A & (self.since_comp >= a * T) & (self.since_comp < b * T)).astype(float)
        if m == 'late':
            return (self.comp & ~A & (self.since_comp >= self.dL)).astype(float)
        if m == 'rep_gated':
            return (rep == 1).astype(float)
        if m == 'early_ramp':
            on = self.comp & ~A & (self.since_comp < RAMP_END * T)
            return np.where(on, np.minimum(1.0, self.since_comp / (RAMP_EARLY * T)), 0.0)
        if m == 'chong':
            up = rep == 1
            hold = (rep == 2) & (since_rep < (self.extra * T if self.extra else 0.0))
            self.level = np.where(up, np.minimum(1.0, self.level + dt / RAMP_MS),
                                  np.where(hold, self.level, np.maximum(0.0, self.level - dt / RAMP_MS)))
            return self.level.copy()
        if m == 'early_rep':
            return ((rep == 2) & (since_rep < EARLY_W * T)).astype(float)
        if m == 'late_rep':
            return ((rep == 2) & (since_rep >= self.dL)).astype(float)
        raise ValueError(m)


def runs(st):
    ch = np.flatnonzero(np.diff(st)) + 1
    s = np.r_[0, ch]; e = np.r_[ch, len(st)]
    return st[s], s, e


def observer_durations(rep, dt):
    """Durations between reported switches; runs before the first report dropped; first and last
    reported runs dropped."""
    v, s, e = runs(rep)
    keep = v != 0
    v, L_ = v[keep], ((e - s) * dt)[keep]
    v, L_ = v[1:-1], L_[1:-1]
    return L_[v == 1].tolist(), L_[v == 2].tolist()


def measures(x, on, rep, rep2, thr, thr_abs):
    """Registered-style durations (first-test scoring), merged competitor episodes (as in the second
    test), absolute criterion, observer scoring (primary and passive observers), realised dose, the
    competitor's mean response while it is dominant (difference criterion), and the realised
    withdrawal time: the area under the increment level from the start of each merged competitor
    episode until the level first reaches 0 or the episode ends (ms at full level)."""
    dt = DT_REC
    E = R.episodes_diff(x, thr, dt)
    A = [l for s, l in E if s == 1]; B = [l for s, l in E if s == -1]
    Babs = R.episodes_abs(x[1], thr_abs, dt)
    d = x[0] - x[1]
    st = np.where(d > thr, 1, np.where(d < -thr, -1, 0)).astype(np.int8)
    v, s0, e0 = runs(st)
    Ls = (e0 - s0) * dt
    onv = np.asarray(on, float)
    merged, cur, started = [], 0.0, False
    inA = np.zeros(len(st), bool)
    areas, ep0 = [], None
    for a, l, ss, ee in zip(v, Ls, s0, e0):
        if a == 1 and l >= MIN_EP_MS:
            inA[ss:ee] = True
            if started and cur > 0: merged.append(cur)
            if ep0 is not None:
                seg = onv[ep0:ss]; z = np.flatnonzero(seg <= 0)
                areas.append(float(seg[:z[0] if len(z) else len(seg)].sum() * dt))
            cur, started, ep0 = 0.0, True, None
        else:
            if a == -1 and started and ep0 is None: ep0 = ss
            if a == -1 and l >= MIN_EP_MS:
                cur += l
            elif started and cur > 0:
                cur += l
    OA, OB = observer_durations(rep, dt)
    OA2, OB2 = observer_durations(rep2, dt)
    f = lambda z: float(np.mean(z)) if len(z) else np.nan
    domB = st == -1
    return dict(mA=f(A), mB=f(B), mBabs=f(Babs), mBm=f(merged), nBm=len(merged), nB=len(B), nBabs=len(Babs),
                mOA=f(OA), mOB=f(OB), nOB=len(OB), mOA2=f(OA2), mOB2=f(OB2), nOB2=len(OB2),
                rB_dom=float(x[1][domB].mean()) if domB.any() else np.nan, area_comp=f(areas),
                dose_comp=float(onv[~inA].mean()) if (~inA).any() else np.nan,
                dose_att=float(onv[inA].mean()) if inA.any() else np.nan, Bm=merged)


def run_batch(cfgs, seeds, mode, inc_frac, thr, T, tag, dL=None, extra=None, t_ms=120000.0, ranges=RANGES):
    p = L.default_params(1); p['T'] = t_ms
    N = len(cfgs) * len(seeds)
    nt = int(round(t_ms / p['dt'])) + 1
    rep = lambda k: np.repeat([c[k] for c in cfgs], len(seeds))
    pv = dict(in1=rep('inp'), in2=rep('inp'), wo=rep('wo'), wa=rep('wa'), wh=rep('wh'), tau_h=rep('tau_h'))
    key = tuple(round(c[k], 6) for c in cfgs for k in sorted(ranges))
    s = seed_of(tag, tuple(seeds), key)
    inst = [(tuple(round(c[k], 6) for k in sorted(ranges)), sd) for c in cfgs for sd in seeds]
    pools = (latency_pools(inst, tag, 1, LAT_MS), latency_pools(inst, tag, 2, LAT2_MS))
    sch = Schedule3(mode, thr if thr is not None else np.full(N, np.inf), p['dt'], N, nt,
                    T=None if T is None else np.repeat(T, len(seeds)),
                    dL=None if dL is None else np.repeat(dL, len(seeds)), extra=extra, pools=pools)
    rb, on = L.simulate(p, N=N, seed=s, noise_sigma=1.0, record_every=2, params_vec=pv,
                        inc=rep('inp') * inc_frac, schedule=sch, noise_scale=rep('noise'))
    k = rb.shape[2]
    return rb, on, sch.obs.rec[:, :k], sch.obs2.rec[:, :k]


def match_dL(D, T):
    """Late-window delay with the same expected time on per competitor episode as the early window:
    E[(D - dL)+] = E[min(D, EARLY_W*T)], D = pooled baseline merged competitor durations."""
    D = np.asarray(D, float)
    if len(D) < 5:
        return np.nan
    target = np.mean(np.minimum(D, EARLY_W * T))
    lo, hi = 0.0, D.max()
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.mean(np.maximum(D - mid, 0.0)) > target: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


def dist_summary(D, T):
    """Quantiles (1% steps); on d = 0..GRID_MAX*T in steps of 0.01T: g(d) = E[min(D, d)]/E[D] (dose weight)
    and hz(d) = E[m(D); D < d]/E[m(D)], m(t) = E[D - t | D > t] (hazard-multiplier weight)."""
    D = np.sort(np.asarray(D, float))
    if len(D) < 5:
        return None
    n = len(D)
    suf = np.cumsum(D[::-1])[::-1]                     # sum_{j>=i} D_j
    cnt = n - np.arange(n)                             # number of j >= i
    with np.errstate(all='ignore'):
        mres = np.where(cnt > 1, (suf - D * cnt - 0.0) / np.maximum(cnt - 1, 1), 0.0)   # mean of D_j - D_i over j > i
    grid = np.linspace(0, GRID_MAX, int(round(GRID_MAX * 100)) + 1) * T
    cm = np.r_[0.0, np.cumsum(mres)]
    tot = cm[-1]
    hz = [float(cm[np.searchsorted(D, dd, side='left')] / tot) if tot > 0 else float('nan') for dd in grid]
    return dict(q=np.quantile(D, np.linspace(0, 1, 101)).tolist(),
                g=[float(np.mean(np.minimum(D, dd)) / np.mean(D)) for dd in grid],
                hz=hz, n=int(n))


SCAL = ('mA', 'mB', 'mBabs', 'mBm', 'nBm', 'nB', 'nBabs', 'mOA', 'mOB', 'nOB', 'mOA2', 'mOB2', 'nOB2', 'rB_dom',
        'area_comp', 'dose_comp', 'dose_att')


def cond_key(mode, frac, extra):
    k = f'{mode}_{frac:g}'
    if isinstance(extra, tuple): k += f'_{extra[0]:g}-{extra[1]:g}'
    elif extra is not None: k += f'_x{extra:g}'
    return k


def run_chunk(sub, conds, scale, t_ms, tag, ranges):
    seeds = list(range(N_SEEDS))
    b = int(BURN_MS / DT_REC)
    rb0, _, _, _ = run_batch(sub, seeds, 'none', 0.0, None, None, tag, t_ms=t_ms, ranges=ranges)
    thr = THETA_FRAC * rb0[:, :, b:].mean(axis=(1, 2))
    thr_abs = np.repeat(np.median(rb0[:, 1, b:], axis=1).reshape(len(sub), N_SEEDS).mean(1), N_SEEDS)
    # baseline again with the observer reading dominance at thr (no increment, so the trajectory is identical)
    rb0b, on0, rp0, rq0 = run_batch(sub, seeds, 'none', 0.0, thr, None, tag, t_ms=t_ms, ranges=ranges)
    assert np.array_equal(rb0, rb0b)
    del rb0b
    M0 = [measures(rb0[i, :, b:].astype(np.float64), on0[i, b:], rp0[i, b:], rq0[i, b:], thr[i], thr_abs[i]) for i in range(len(thr))]
    del rb0, rp0, rq0
    nc = len(sub)
    per = lambda M, c: [{k: M[i][k] for k in SCAL} for i in range(c * N_SEEDS, (c + 1) * N_SEEDS)]
    base = [per(M0, c) for c in range(nc)]
    T = np.array([0.5 * (np.nanmean([x['mA'] for x in base[c]]) + np.nanmean([x['mB'] for x in base[c]])) for c in range(nc)])
    Bm0 = [[l for i in range(c * N_SEEDS, (c + 1) * N_SEEDS) for l in M0[i]['Bm']] for c in range(nc)]
    dL = np.array([match_dL(Bm0[c], T[c]) for c in range(nc)])
    dists = {'base_all': [dist_summary(Bm0[c], T[c]) for c in range(nc)],
             'base_h0': [dist_summary([l for i in range(c * N_SEEDS, c * N_SEEDS + 4) for l in M0[i]['Bm']], T[c]) for c in range(nc)]}
    out = {}
    for mode, frac, extra in conds:
        key = cond_key(mode, frac, extra)
        rb, on, rp, rq = run_batch(sub, seeds, mode, frac * scale, thr, T, tag, dL=dL, extra=extra, t_ms=t_ms, ranges=ranges)
        M = [measures(rb[i, :, b:].astype(np.float64), on[i, b:], rp[i, b:], rq[i, b:], thr[i], thr_abs[i]) for i in range(len(thr))]
        out[key] = [per(M, c) for c in range(nc)]
        if mode == 'ungated':
            dists[key + '_all'] = [dist_summary([l for i in range(c * N_SEEDS, (c + 1) * N_SEEDS) for l in M[i]['Bm']], T[c]) for c in range(nc)]
            dists[key + '_h0'] = [dist_summary([l for i in range(c * N_SEEDS, c * N_SEEDS + 4) for l in M[i]['Bm']], T[c]) for c in range(nc)]
        del rb, on, rp, rq, M
    return base, out, T.tolist(), dL.tolist(), dists


def accept(n_target, rng_seed, ranges, chunk, t_ms, tag, need_cv):
    rng = np.random.default_rng(rng_seed)
    acc, n_drawn = [], 0
    seeds = list(range(N_SEEDS))
    b = int(BURN_MS / DT_REC)
    while len(acc) < n_target:
        cand = [{k: float(rng.uniform(*v)) for k, v in ranges.items()} for _ in range(chunk)]
        n_drawn += len(cand)
        rb, _, _, _ = run_batch(cand, seeds, 'none', 0.0, None, None, tag + '_accept', t_ms=t_ms, ranges=ranges)
        thr = THETA_FRAC * rb[:, :, b:].mean(axis=(1, 2))
        for c in range(len(cand)):
            n_ep, cvs = [], []
            for i in range(c * N_SEEDS, (c + 1) * N_SEEDS):
                E = R.episodes_diff(rb[i, :, b:].astype(np.float64), thr[i], DT_REC)
                dd = [l for _, l in E]; n_ep.append(len(E))
                cvs.append(np.std(dd) / np.mean(dd) if len(dd) > 2 else np.nan)
            ok = np.mean(n_ep) >= 10 and (not need_cv or 0.3 <= np.nanmedian(cvs) <= 0.7)
            if ok and len(acc) < n_target:
                acc.append(dict(cand[c]))
        del rb
        print(f'  [{tag}] accepted {len(acc)}/{n_target} of {n_drawn} drawn', flush=True)
    return acc, n_drawn


CONDS = [('gated', INC, None), ('gated_lat', INC, None), ('rep_gated', INC, None), ('ungated', INC, None)] + \
        [('win', INC, w) for w in WINDOWS] + [('late', INC, None), ('early_ramp', INC, None)] + \
        [('gated', INC_CAL, None), ('ungated', INC_CAL, None), ('chong', INC_CAL, None)] + \
        [('chong', INC_CAL, h) for h in HOLDS] + [('early_rep', INC_CAL, None), ('late_rep', INC_CAL, None)]
CONDS_NA = [('gated', INC, None), ('gated_lat', INC, None), ('rep_gated', INC, None), ('ungated', INC, None),
            ('win', INC, WINDOWS[0]), ('late', INC, None), ('chong', INC_CAL, None)]
CONDS_HN = CONDS_NA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--n-config', type=int, default=N_TARGET)
    ap.add_argument('--n-config-na', type=int, default=N_TARGET_NA)
    ap.add_argument('--n-config-hn', type=int, default=N_TARGET_HN)
    ap.add_argument('--chunk', type=int, default=25)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--t-ms', type=float, default=120000.0)
    ap.add_argument('--out', default='li2017_test3_results.json')
    a = ap.parse_args()
    t0 = time.time()
    s_main, s_na, s_hn = (DRY_SEED, DRY_SEED + 1, DRY_SEED + 2) if a.dry_run else (SEED, SEED_NA, SEED_HN)
    scale = 0.0 if a.dry_run else 1.0
    acc, nd = accept(a.n_config, s_main, RANGES, a.chunk, a.t_ms, 'main', True)
    acc_na, nd_na = accept(a.n_config_na, s_na, RANGES_NA, a.chunk, a.t_ms, 'noadapt', False)
    acc_hn, nd_hn = accept(a.n_config_hn, s_hn, RANGES_HN, a.chunk, a.t_ms, 'highnoise', False)
    from concurrent.futures import ProcessPoolExecutor
    res = {}
    for name, A_, cd, rg in (('main', acc, CONDS, RANGES), ('noadapt', acc_na, CONDS_NA, RANGES_NA),
                             ('highnoise', acc_hn, CONDS_HN, RANGES_HN)):
        chunks = [A_[i:i + a.chunk] for i in range(0, len(A_), a.chunk)]
        R_ = dict(base=[], cond={}, T=[], dL=[], dists={})
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(run_chunk, ch, cd, scale, a.t_ms, name, rg) for ch in chunks]
            for k, fu in enumerate(futs):
                base, out, T, dL, dists = fu.result()
                R_['base'] += base; R_['T'] += T; R_['dL'] += dL
                for kk, v in out.items(): R_['cond'].setdefault(kk, []).extend(v)
                for kk, v in dists.items(): R_['dists'].setdefault(kk, []).extend(v)
                print(f'  [{name}] chunk {k + 1}/{len(chunks)} [{time.time() - t0:.0f}s]', flush=True)
        res[name] = R_
    import scipy, platform
    json.dump(dict(dry_run=a.dry_run, n_drawn=nd, n_drawn_na=nd_na, n_drawn_hn=nd_hn, accepted=acc, accepted_na=acc_na,
                   accepted_hn=acc_hn, results=res,
                   versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                   settings=dict(SEED=SEED, SEED_NA=SEED_NA, SEED_HN=SEED_HN, RANGES=RANGES, RANGES_NA=RANGES_NA,
                   RANGES_HN=RANGES_HN, INC=INC, INC_CAL=INC_CAL, RAMP_MS=RAMP_MS, LAT_MS=LAT_MS, LAT2_MS=LAT2_MS,
                   GATE_LAT_MS=GATE_LAT_MS, EARLY_W=EARLY_W, RAMP_EARLY=RAMP_EARLY, RAMP_END=RAMP_END,
                   WINDOWS=WINDOWS, HOLDS=HOLDS, GRID_MAX=GRID_MAX, N_SEEDS=N_SEEDS, T_MS=a.t_ms, CHUNK=a.chunk)),
              open(a.out, 'w'), default=float)
    print(f'wrote {a.out} [{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
