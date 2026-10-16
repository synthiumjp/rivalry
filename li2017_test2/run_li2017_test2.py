"""
Second pre-registered test of the delivery-timing account in the attention model of binocular
rivalry of Li, Rankin, Rinzel, Carrasco and Heeger (2017). See prereg_li2017_test2.md.

    python run_li2017_test2.py --dry-run --n-config 4 --n-config-na 4 --t-ms 30000 --chunk 4   # code path, increments 0
    python run_li2017_test2.py                                                                  # the registered run

Uses the port li2017.py (graded increment levels allowed; identical to the first test's port for
on/off gates) and the episode and sampling helpers of run_li2017_test.py.
"""
import argparse, json, time, zlib
import numpy as np
import li2017 as L
import run_li2017_test as R   # first test: RANGES, episodes_diff(), episodes_abs()

SEED = 20261016            # main sample (new configurations; the first test used 20261007)
SEED_NA = 20261017         # no-adaptation sample
DRY_SEED = 99              # dry run only (99 main, 100 no-adaptation): not part of either registered sample
RANGES = dict(R.RANGES)
RANGES_NA = dict(inp=(0.35, 0.8), wo=(0.5, 0.75), wa=(0.4, 0.9), wh=(0.0, 0.0),
                 tau_h=(2000.0, 2000.0), noise=(0.15, 0.30))
N_SEEDS, BURN_MS, MIN_EP_MS, THETA_FRAC = 8, 5000.0, 50.0, 0.05
INC = 0.20                 # increment for the window, delay and no-adaptation conditions
INC_CHONG = 1.00           # Chong et al. doubled contrast
RAMP_MS = 520.0            # ramp up on report of grating 1, ramp down on report of grating 2
LAT_MS = (200.0, 450.0)    # report latency, uniform, drawn independently for each reported switch
EARLY_W = 0.25             # early window length, in units of T
DELAYS = (0.2, 0.35, 0.5, 0.65, 0.8, 1.0, 1.2, 1.6, 2.0)
CHONG_EXTRA = (0.5, 1.0)   # extra hold before the ramp down, in units of T (descriptive)
N_TARGET, N_TARGET_NA = 200, 100
DT_REC = 1.0               # recorded sample spacing (ms)


def seed_of(*parts):
    return zlib.crc32(repr(parts).encode()) % (2 ** 31 - 1)


class Schedule2:
    """Increment level per instance, read from the live binocular responses each step.
    Grating 1 (attended) dominant: rb1 - rb2 > thr; grating 2: rb2 - rb1 > thr; else indeterminate.
    Competitor episodes for the window schedules are tracked online as merged episodes: an episode
    starts at the first grating-2 sample after a grating-1 run of at least MIN_EP_MS (or at the start)
    and ends when a grating-1 run reaches MIN_EP_MS. The increment is never on while grating 1 is
    dominant in the window schedules.
    Reported modes pass the dominant percept through a report latency (see prereg, condition 6)."""

    def __init__(self, mode, thr, dt, N, T=None, dL=None, extra=None, rng=None):
        self.mode, self.thr, self.dt = mode, thr, dt
        self.T, self.dL, self.extra, self.rng = T, dL, extra, rng
        self.prevA = np.zeros(N, bool)
        self.since_Aend = np.full(N, np.inf)
        self.a_run = np.zeros(N)
        self.comp = np.zeros(N, bool)
        self.since_comp = np.full(N, np.inf)
        self.rep = np.zeros(N, np.int8)
        self.pend = np.zeros(N, np.int8)
        self.cd = np.zeros(N)
        self.since_rep = np.full(N, np.inf)
        self.level = np.zeros(N)

    def _report(self, st):
        new = (st != 0) & (st != self.rep)
        start = new & (self.pend != st)
        if start.any():
            self.pend[start] = st[start]
            self.cd[start] = self.rng.uniform(LAT_MS[0], LAT_MS[1], int(start.sum()))
        cancel = (st != 0) & (st == self.rep)
        self.pend[cancel] = 0
        active = self.pend != 0
        self.cd[active] -= self.dt
        fire = active & (self.cd <= 0)
        self.rep[fire] = self.pend[fire]
        self.pend[fire] = 0
        self.since_rep = np.where(fire, 0.0, self.since_rep + self.dt)

    def __call__(self, idx, state):
        rb = state['rb']; dt = self.dt
        d = rb[:, 0] - rb[:, 1]
        A, B = d > self.thr, d < -self.thr
        st = np.where(A, 1, np.where(B, 2, 0)).astype(np.int8)
        self.since_Aend = np.where(~A & self.prevA, 0.0, self.since_Aend + dt)
        self.a_run = np.where(A, self.a_run + dt, 0.0)
        end_comp = self.comp & (self.a_run >= MIN_EP_MS)
        self.comp = self.comp & ~end_comp
        start_comp = B & ~self.comp
        self.comp = self.comp | start_comp
        self.since_comp = np.where(start_comp, 0.0, np.where(self.comp, self.since_comp + dt, np.inf))
        self.prevA = A
        m = self.mode
        n = len(d)
        if m == 'none':
            return np.zeros(n)
        if m == 'gated':
            return A.astype(float)
        if m == 'ungated':
            return np.ones(n)
        if m == 'offset':                       # delayed withdrawal, as in the first test
            return (A | (self.since_Aend < self.extra * self.T)).astype(float)
        if m == 'early':
            return (self.comp & ~A & (self.since_comp < EARLY_W * self.T)).astype(float)
        if m == 'late':
            return (self.comp & ~A & (self.since_comp >= self.dL)).astype(float)
        self._report(st)
        if m == 'chong':
            up = self.rep == 1
            hold = (self.rep == 2) & (self.since_rep < (self.extra * self.T if self.extra else 0.0))
            self.level = np.where(up, np.minimum(1.0, self.level + dt / RAMP_MS),
                                  np.where(hold, self.level, np.maximum(0.0, self.level - dt / RAMP_MS)))
            return self.level.copy()
        if m == 'early_rep':
            return ((self.rep == 2) & (self.since_rep < EARLY_W * self.T)).astype(float)
        if m == 'late_rep':
            return ((self.rep == 2) & (self.since_rep >= self.dL)).astype(float)
        raise ValueError(m)


def runs(st):
    ch = np.flatnonzero(np.diff(st)) + 1
    s = np.r_[0, ch]; e = np.r_[ch, len(st)]
    return st[s], s, e


def measures(x, on, thr, thr_abs):
    """Registered-style durations (first and last episode dropped, <50 ms discarded), merged
    competitor episodes (merged across indeterminate spells and grating-1 excursions < 50 ms; the
    first merged episode starts after the first grating-1 episode, the last unfinished one is dropped),
    and the increment's mean level inside and outside grating-1 episodes."""
    dt = DT_REC
    E = R.episodes_diff(x, thr, dt)
    A = [l for s, l in E if s == 1]; B = [l for s, l in E if s == -1]
    Babs = R.episodes_abs(x[1], thr_abs, dt)
    d = x[0] - x[1]
    st = np.where(d > thr, 1, np.where(d < -thr, -1, 0)).astype(np.int8)
    v, s0, e0 = runs(st)
    Ls = (e0 - s0) * dt
    merged, cur, started = [], 0.0, False
    inA = np.zeros(len(st), bool)
    for a, l, ss, ee in zip(v, Ls, s0, e0):
        if a == 1 and l >= MIN_EP_MS:
            inA[ss:ee] = True
            if started and cur > 0: merged.append(cur)
            cur, started = 0.0, True
        elif a == -1 and l >= MIN_EP_MS:
            cur += l
        elif started and cur > 0:
            cur += l
    f = lambda z: float(np.mean(z)) if len(z) else np.nan
    onv = np.asarray(on, float)
    return dict(mA=f(A), mB=f(B), mBabs=f(Babs), mBm=f(merged), nBm=len(merged), nB=len(B),
                dose_comp=float(onv[~inA].mean()) if (~inA).any() else np.nan,
                dose_att=float(onv[inA].mean()) if inA.any() else np.nan, Bm=merged)


def run_batch(cfgs, seeds, mode, inc_frac, thr, T, tag, dL=None, extra=None, t_ms=120000.0, ranges=RANGES):
    p = L.default_params(1); p['T'] = t_ms
    N = len(cfgs) * len(seeds)
    rep = lambda k: np.repeat([c[k] for c in cfgs], len(seeds))
    pv = dict(in1=rep('inp'), in2=rep('inp'), wo=rep('wo'), wa=rep('wa'), wh=rep('wh'), tau_h=rep('tau_h'))
    key = tuple(round(c[k], 6) for c in cfgs for k in sorted(ranges))
    s = seed_of(tag, tuple(seeds), key)
    srng = np.random.default_rng(seed_of(tag, 'latency', mode, extra, tuple(seeds), key))   # report latencies only
    sch = Schedule2(mode, thr if thr is not None else np.zeros(N), p['dt'], N,
                    T=None if T is None else np.repeat(T, len(seeds)),
                    dL=None if dL is None else np.repeat(dL, len(seeds)), extra=extra, rng=srng)
    return L.simulate(p, N=N, seed=s, noise_sigma=1.0, record_every=2, params_vec=pv,
                      inc=rep('inp') * inc_frac, schedule=sch, noise_scale=rep('noise'))


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
    """Quantiles (1% steps) and g(d) = E[min(D, d)]/E[D] for d = 0..3T in steps of 0.01T."""
    D = np.asarray(D, float)
    if len(D) < 5:
        return None
    return dict(q=np.quantile(D, np.linspace(0, 1, 101)).tolist(),
                g=[float(np.mean(np.minimum(D, dd)) / np.mean(D)) for dd in np.linspace(0, 3, 301) * T],
                n=int(len(D)))


SCAL = ('mA', 'mB', 'mBabs', 'mBm', 'nBm', 'nB', 'dose_comp', 'dose_att')


def run_chunk(sub, conds, scale, t_ms, tag, ranges):
    seeds = list(range(N_SEEDS))
    b = int(BURN_MS / DT_REC)
    rb0, on0 = run_batch(sub, seeds, 'none', 0.0, None, None, tag, t_ms=t_ms, ranges=ranges)
    thr = THETA_FRAC * rb0[:, :, b:].mean(axis=(1, 2))
    thr_abs = np.repeat(np.median(rb0[:, 1, b:], axis=1).reshape(len(sub), N_SEEDS).mean(1), N_SEEDS)
    M0 = [measures(rb0[i, :, b:].astype(np.float64), on0[i, b:], thr[i], thr_abs[i]) for i in range(len(thr))]
    del rb0
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
        key = mode + f'_{frac:g}' + (f'_x{extra:g}' if extra is not None else '')
        rb, on = run_batch(sub, seeds, mode, frac * scale, thr, T, tag, dL=dL, extra=extra, t_ms=t_ms, ranges=ranges)
        M = [measures(rb[i, :, b:].astype(np.float64), on[i, b:], thr[i], thr_abs[i]) for i in range(len(thr))]
        out[key] = [per(M, c) for c in range(nc)]
        if mode == 'ungated':
            dists['ungated_all'] = [dist_summary([l for i in range(c * N_SEEDS, (c + 1) * N_SEEDS) for l in M[i]['Bm']], T[c]) for c in range(nc)]
            dists['ungated_h0'] = [dist_summary([l for i in range(c * N_SEEDS, c * N_SEEDS + 4) for l in M[i]['Bm']], T[c]) for c in range(nc)]
        del rb, on, M
    return base, out, T.tolist(), dL.tolist(), dists


def accept(n_target, rng_seed, ranges, chunk, t_ms, tag, need_cv):
    rng = np.random.default_rng(rng_seed)
    acc, n_drawn = [], 0
    seeds = list(range(N_SEEDS))
    b = int(BURN_MS / DT_REC)
    while len(acc) < n_target:
        cand = [{k: float(rng.uniform(*v)) for k, v in ranges.items()} for _ in range(chunk)]
        n_drawn += len(cand)
        rb, _ = run_batch(cand, seeds, 'none', 0.0, None, None, tag + '_accept', t_ms=t_ms, ranges=ranges)
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
        print(f'  [{tag}] accepted {len(acc)}/{n_target} of {n_drawn} drawn', flush=True)
    return acc, n_drawn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--n-config', type=int, default=N_TARGET)
    ap.add_argument('--n-config-na', type=int, default=N_TARGET_NA)
    ap.add_argument('--chunk', type=int, default=25)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--t-ms', type=float, default=120000.0)
    ap.add_argument('--out', default='li2017_test2_results.json')
    a = ap.parse_args()
    t0 = time.time()
    s_main, s_na = (DRY_SEED, DRY_SEED + 1) if a.dry_run else (SEED, SEED_NA)
    scale = 0.0 if a.dry_run else 1.0
    acc, nd = accept(a.n_config, s_main, RANGES, a.chunk, a.t_ms, 'main', True)
    acc_na, nd_na = accept(a.n_config_na, s_na, RANGES_NA, a.chunk, a.t_ms, 'noadapt', False)
    conds = [('gated', INC, None), ('ungated', INC, None), ('early', INC, None), ('late', INC, None)] + \
            [('offset', INC, d) for d in DELAYS] + \
            [('chong', INC_CHONG, None)] + [('chong', INC_CHONG, x) for x in CHONG_EXTRA] + \
            [('early_rep', INC, None), ('late_rep', INC, None)]
    conds_na = [('gated', INC, None), ('ungated', INC, None)]
    from concurrent.futures import ProcessPoolExecutor
    res = {}
    for name, A_, cd, rg in (('main', acc, conds, RANGES), ('noadapt', acc_na, conds_na, RANGES_NA)):
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
    json.dump(dict(dry_run=a.dry_run, n_drawn=nd, n_drawn_na=nd_na, accepted=acc, accepted_na=acc_na,
                   results=res, settings=dict(SEED=SEED, SEED_NA=SEED_NA, RANGES=RANGES, RANGES_NA=RANGES_NA,
                   INC=INC, INC_CHONG=INC_CHONG, RAMP_MS=RAMP_MS, LAT_MS=LAT_MS, EARLY_W=EARLY_W,
                   DELAYS=DELAYS, CHONG_EXTRA=CHONG_EXTRA, N_SEEDS=N_SEEDS, T_MS=a.t_ms, CHUNK=a.chunk)),
              open(a.out, 'w'), default=float)
    print(f'wrote {a.out} [{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
