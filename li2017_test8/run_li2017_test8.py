"""
Eighth test, with a pre-specified analysis plan (analysis_plan_li2017_test8.md). It runs Chong et al.'s (2005)
report-contingent timing in the seventh test's configurations, whose successive durations are about as weakly
correlated as observers'. The increment ramps up over 520 ms while the simulated observer reports the attended grating
and ramps down over 520 ms otherwise (the third test's 'chong' condition, without a hold), at five sizes, so that it can
be calibrated per configuration to Chong et al.'s attended lengthening of 29%.

The configurations, seeds, chunks and noise streams are those of the seventh test (read from
../li2017_test7/li2017_test7_results.json), so the baseline runs reproduce the seventh test's baseline exactly and every
condition is paired, noise for noise, with the seventh test's conditions.

    python run_li2017_test8.py --dry-run --n-config 4 --t-ms 30000 --chunk 4 --seeds 4    # code path, increments 0
    python run_li2017_test8.py --workers 6                                                # the run
The chunk size (10) must equal the seventh test's (it sets the noise streams); the number of workers does not affect
results.

Reuses, unchanged: li2017.py, li2017_t6.py, rate_models.py, run_li2017_test.py, run_li2017_test3.py,
run_li2017_test6.py and run_li2017_test7.py.
"""
import argparse, json, os, time
import numpy as np
import li2017 as L
import li2017_t6 as L6
import rate_models as RM
import run_li2017_test3 as R3
import run_li2017_test6 as T6
import run_li2017_test7 as T7

SRC7 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'li2017_test7', 'li2017_test7_results.json')
SIZES = (0.025, 0.05, 0.075, 0.10, 0.20)   # increment, fraction of input strength
N_SEEDS = 32                      # as in the seventh test
CHUNK = 10                        # as in the seventh test


class Schedule8(T6.Schedule6):
    """The sixth test's schedule (both simulated observers run in every mode) plus mode 'chong': Chong et al.'s timing,
    as in the third test without a hold. The level rises linearly to 1 over RAMP_MS while the primary observer reports
    grating 1 and falls linearly to 0 over RAMP_MS otherwise; the increment is level x size."""

    def __init__(self, mode6, *args, **kw):
        super().__init__(mode6, None, None, *args, **kw)
        self.lev = np.zeros(args[2])

    def __call__(self, idx, state):
        if self.mode6 != 'chong':
            return super().__call__(idx, state)
        R3.Schedule3.__call__(self, idx, state)          # mode 'none': advances and records both observers
        rb = state['rb']; d = rb[:, 0] - rb[:, 1]
        A, B = d > self.thr, d < -self.thr
        up = self.obs.rep == 1
        self.lev = np.where(up, np.minimum(1.0, self.lev + self.dt / R3.RAMP_MS),
                            np.maximum(0.0, self.lev - self.dt / R3.RAMP_MS))
        on = self.lev.copy()
        self._stats(on, A, B, state)
        return on


def run_batch8(model, cfgs, seeds, mode, size, thr, tag, t_ms, ranges):
    """As the seventh test's run_batch7 (same noise and observer-latency streams for the same tag, seeds and
    configurations), with Schedule8."""
    N = len(cfgs) * len(seeds)
    rep = lambda key: np.repeat([c[key] for c in cfgs], len(seeds))
    keyt = tuple(round(c[kk], 6) for c in cfgs for kk in sorted(ranges))
    s = R3.seed_of(tag, tuple(seeds), keyt)
    inst = [(tuple(round(c[kk], 6) for kk in sorted(ranges)), sd) for c in cfgs for sd in seeds]
    pools = (R3.latency_pools(inst, tag, 1, R3.LAT_MS), R3.latency_pools(inst, tag, 2, R3.LAT2_MS))
    thr_ = thr if thr is not None else np.full(N, np.inf)
    if model == 'li':
        p = L.default_params(1); p['T'] = t_ms
        nt = int(round(t_ms / p['dt'])) + 1
        pv = dict(in1=rep('inp'), in2=rep('inp'), wo=rep('wo'), wa=rep('wa'), wh=rep('wh'), tau_h=rep('tau_h'))
        sch = Schedule8(mode, thr_, p['dt'], N, nt, pools=pools)
        rb, on = L6.simulate(p, N=N, seed=s, noise_sigma=1.0, record_every=2, params_vec=pv, inc=rep('inp') * size,
                             schedule=sch, noise_scale=rep('noise'), target='input')
    else:
        nt = int(round(t_ms / RM.DT)) + 1
        pv = {kk: rep(kk) for kk in ranges if kk != 'noise'}
        sch = Schedule8(mode, thr_, RM.DT, N, nt, pools=pools)
        rb, on = RM.simulate(model, pv, N, s, t_ms, inc=rep('I') * size, schedule=sch, record_every=2,
                             noise_scale=rep('noise'))
    kk = rb.shape[2]
    return rb, on, sch.obs.rec[:, :kk], sch.obs2.rec[:, :kk], sch.extras()


def predominance(rep):
    """Fraction of reported time (primary observer, either grating) during which grating 2 is reported."""
    r = np.asarray(rep)
    n = (r > 0).sum()
    return float((r == 2).sum() / n) if n else np.nan


def run_chunk8(model, sub, scale, t_ms, tag, n_seeds, ranges):
    seeds = list(range(n_seeds))
    b = int(R3.BURN_MS / R3.DT_REC)
    nc = len(sub)
    rb0 = run_batch8(model, sub, seeds, 'none', 0.0, None, tag, t_ms, ranges)[0]
    thr = R3.THETA_FRAC * rb0[:, :, b:].mean(axis=(1, 2))
    if model == 'li':
        thr_abs = np.repeat(np.median(rb0[:, 1, b:], axis=1).reshape(nc, n_seeds).mean(1), n_seeds)
    else:
        mid = 0.5 * (np.percentile(rb0[:, 1, b:], 10, axis=1) + np.percentile(rb0[:, 1, b:], 90, axis=1))
        thr_abs = np.repeat(mid.reshape(nc, n_seeds).mean(1), n_seeds)
    rb0b, on0, rp0, rq0, ex0 = run_batch8(model, sub, seeds, 'none', 0.0, thr, tag, t_ms, ranges)
    assert np.array_equal(rb0, rb0b)
    del rb0b
    M0 = [R3.measures(rb0[i, :, b:].astype(np.float64), on0[i, b:], rp0[i, b:], rq0[i, b:], thr[i], thr_abs[i])
          for i in range(len(thr))]
    per = lambda M, cc: [{kk: M[i][kk] for kk in R3.SCAL} for i in range(cc * n_seeds, (cc + 1) * n_seeds)]
    per_cfg = lambda v: [float(np.nanmean(np.asarray(v, float)[cc * n_seeds:(cc + 1) * n_seeds])) for cc in range(nc)]
    base = [per(M0, cc) for cc in range(nc)]
    T = np.array([0.5 * (np.nanmean([x['mA'] for x in base[cc]]) + np.nanmean([x['mB'] for x in base[cc]]))
                  for cc in range(nc)])
    pred = {'base': per_cfg([predominance(rp0[i, b:]) for i in range(len(thr))])}
    del rb0, rp0, rq0, on0
    out, duty, lev_comp = {}, {}, {}
    for size in SIZES:
        name = f'chong_{size:g}'
        rb, on, rp, rq, ex = run_batch8(model, sub, seeds, 'chong', size * scale, thr, tag, t_ms, ranges)
        M = [R3.measures(rb[i, :, b:].astype(np.float64), on[i, b:], rp[i, b:], rq[i, b:], thr[i], thr_abs[i])
             for i in range(len(thr))]
        out[name] = [per(M, cc) for cc in range(nc)]
        duty[name] = on[:, b:].mean(1).reshape(nc, n_seeds).mean(1).tolist()
        # fraction of the increment's level that falls in the competitor's dominance (difference criterion)
        dc = []
        for i in range(len(thr)):
            x = rb[i, :, b:]; o = on[i, b:]
            Bdom = (x[1] - x[0]) > thr[i]
            dc.append(float((o * Bdom).sum() / o.sum()) if o.sum() > 0 else np.nan)
        lev_comp[name] = per_cfg(dc)
        pred[name] = per_cfg([predominance(rp[i, b:]) for i in range(len(thr))])
        del rb, on, rp, rq, M
    return dict(base=base, out=out, T=T.tolist(), duty=duty, level_in_comp=lev_comp, predominance=pred)


def run_model(model, a, scale, t0, D7):
    from concurrent.futures import ProcessPoolExecutor
    acc = D7['models'][model]['accepted'][:a.n_config]
    ranges = T7.RANGES[model]
    tag = f'test7_{model}'                  # the seventh test's tag: same noise and observer latencies
    chunks = [acc[i:i + a.chunk] for i in range(0, len(acc), a.chunk)]
    Rz = dict(base=[], cond={}, T=[], duty={}, level_in_comp={}, predominance={})
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_chunk8, model, ch, scale, a.t_ms, tag, a.seeds, ranges) for ch in chunks]
        for kk, fu in enumerate(futs):
            r = fu.result()
            if not a.dry_run:                # fail fast: the baseline must reproduce the seventh test's
                b7 = D7['models'][model]['results']['base'][kk * a.chunk:(kk + 1) * a.chunk]
                for c8, c7 in zip(r['base'], b7):
                    for x8, x7 in zip(c8, c7):
                        for m in ('mA', 'mBm', 'mOA', 'mOB'):
                            if not (x8[m] == x7[m] or (np.isnan(x8[m]) and np.isnan(x7[m]))):
                                raise RuntimeError(f'[{model}] chunk {kk + 1}: baseline {m} differs from the seventh test '
                                                   f'({x8[m]} against {x7[m]}); stopping')
            Rz['base'] += r['base']; Rz['T'] += r['T']
            for grp in ('duty', 'level_in_comp', 'predominance'):
                for key, v in r[grp].items():
                    Rz[grp].setdefault(key, []).extend(v)
            for key, v in r['out'].items():
                Rz['cond'].setdefault(key, []).extend(v)
            print(f'  [{model}] chunk {kk + 1}/{len(chunks)} [{time.time() - t0:.0f}s]', flush=True)
    return dict(accepted=acc, results=Rz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--n-config', type=int, default=60)
    ap.add_argument('--chunk', type=int, default=CHUNK)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--t-ms', type=float, default=120000.0)
    ap.add_argument('--seeds', type=int, default=N_SEEDS)
    ap.add_argument('--models', default='li,sfa,dep')
    ap.add_argument('--src7', default=SRC7)
    ap.add_argument('--out', default='li2017_test8_results.json')
    a = ap.parse_args()
    if not a.dry_run:
        assert a.chunk == CHUNK and a.seeds == N_SEEDS and a.t_ms == 120000.0, 'the run must use the seventh test settings'
    D7 = json.load(open(a.src7))
    t0 = time.time()
    scale = 0.0 if a.dry_run else 1.0
    res = {}
    for model in a.models.split(','):
        res[model] = run_model(model, a, scale, t0, D7)
        json.dump(dict(partial=True, models=res), open(a.out + '.partial.json', 'w'), default=float)
    import scipy, platform
    json.dump(dict(dry_run=a.dry_run, models=res,
                   versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                   settings=dict(SRC7=a.src7, SIZES=SIZES, N_SEEDS=a.seeds, T_MS=a.t_ms, CHUNK=a.chunk,
                                 RAMP_MS=R3.RAMP_MS, LAT_MS=R3.LAT_MS, tag='test7_<model>')),
              open(a.out, 'w'), default=float)
    print(f'wrote {a.out} [{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
