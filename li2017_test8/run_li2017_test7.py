"""
Seventh test, with a pre-specified analysis plan (analysis_plan_li2017_test7.md). It asks what the gated lengthening of
the competitor does where observers are: in configurations whose successive durations are as weakly correlated as
observers' (lag-1 correlation below 0.1). In Li et al.'s model and the two rate models of the sixth test, configurations
are drawn from ranges with weaker adaptation and accepted only if their baseline serial correlation, estimated on the
acceptance runs, is below 0.1. Conditions: the live gate, continuous delivery, self-replay of the trial's own baseline
schedule, and a masked self-replay (on only when the baseline schedule says the attended grating was dominant and it is
dominant now). The masked replay is never on during the competitor's dominance and is off once the baseline schedule's
attended period has ended, so, unlike the live gate, it cannot stay on through the delay that the increment causes.

    python run_li2017_test7.py --dry-run --n-config 4 --t-ms 30000 --chunk 4 --seeds 4    # code path, increments 0
    python run_li2017_test7.py --workers 6                                                # the run
The chunk size (10) is part of the design (it sets the noise streams); the number of workers does not affect results.

Reuses, unchanged: li2017.py, li2017_t6.py, rate_models.py, run_li2017_test.py, run_li2017_test3.py and
run_li2017_test6.py (schedules, observers, measures, serial correlation, distributions).
"""
import argparse, json, time
import numpy as np
import li2017 as L
import li2017_t6 as L6
import rate_models as RM
import run_li2017_test as R
import run_li2017_test3 as R3
import run_li2017_test6 as T6

SEED = dict(li=7001001, sfa=7001002, dep=7001003)
DRY_SEED = dict(li=7111, sfa=7112, dep=7113)
N_TARGET = 60
MAX_DRAWS = 3000                 # per model; if reached, the run continues with the configurations accepted so far
N_SEEDS = 32
INC = 0.10
R_MAX = 0.10                     # acceptance: baseline serial correlation (acceptance runs) below this
CV_RANGE = (0.3, 0.8)
MIN_T_RATE = 500.0
RANGES = dict(
    li=dict(inp=(0.35, 0.8), wo=(0.5, 0.75), wa=(0.4, 0.9), wh=(0.3, 1.2), tau_h=(1000.0, 4000.0), noise=(0.04, 0.10)),
    sfa=dict(I=(0.4, 1.0), beta=(0.6, 1.4), g=(0.05, 0.35), tau_a=(1000.0, 6000.0), noise=(0.06, 0.20)),
    dep=dict(I=(0.4, 1.0), beta=(0.9, 1.4), gam=(0.2, 1.0), tau_d=(1000.0, 6000.0), noise=(0.08, 0.18)))
CONDS = [('gated', 'gated'), ('ungated', 'ungated'), ('replay_self', 'replay_self'), ('replay_masked', 'replay_masked')]


class Schedule7(T6.Schedule6):
    """The sixth test's schedule plus mode 'replay_masked': on only while grating 1 was dominant at the same step of
    this instance's own baseline run AND grating 1 is dominant now."""

    def __call__(self, idx, state):
        if self.mode6 != 'replay_masked':
            return super().__call__(idx, state)
        R3.Schedule3.__call__(self, idx, state)
        rb = state['rb']; d = rb[:, 0] - rb[:, 1]
        A, B = d > self.thr, d < -self.thr
        on = (self.rec[:, idx] & A).astype(float)
        self._stats(on, A, B, state)
        return on


def run_batch7(model, cfgs, seeds, mode, size, thr, tag, rec=None, t_ms=120000.0, ranges=None):
    """As run_batch6, input increments only, with Schedule7."""
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
        sch = Schedule7(mode, rec, None, thr_, p['dt'], N, nt, pools=pools)
        rb, on = L6.simulate(p, N=N, seed=s, noise_sigma=1.0, record_every=2, params_vec=pv, inc=rep('inp') * size,
                             schedule=sch, noise_scale=rep('noise'), target='input')
    else:
        nt = int(round(t_ms / RM.DT)) + 1
        pv = {kk: rep(kk) for kk in ranges if kk != 'noise'}
        sch = Schedule7(mode, rec, None, thr_, RM.DT, N, nt, pools=pools)
        rb, on = RM.simulate(model, pv, N, s, t_ms, inc=rep('I') * size, schedule=sch, record_every=2,
                             noise_scale=rep('noise'))
    kk = rb.shape[2]
    out = (rb, on, sch.obs.rec[:, :kk], sch.obs2.rec[:, :kk], sch.extras())
    return out + ((sch.Arec,) if mode == 'none_rec' else ())


def run_chunk7(model, sub, scale, t_ms, tag, n_seeds, ranges):
    seeds = list(range(n_seeds))
    b = int(R3.BURN_MS / R3.DT_REC)
    nc = len(sub)
    minutes = n_seeds * (t_ms - R3.BURN_MS) / 60000.0
    rb0 = run_batch7(model, sub, seeds, 'none', 0.0, None, tag, t_ms=t_ms, ranges=ranges)[0]
    thr = R3.THETA_FRAC * rb0[:, :, b:].mean(axis=(1, 2))
    if model == 'li':
        thr_abs = np.repeat(np.median(rb0[:, 1, b:], axis=1).reshape(nc, n_seeds).mean(1), n_seeds)
    else:
        mid = 0.5 * (np.percentile(rb0[:, 1, b:], 10, axis=1) + np.percentile(rb0[:, 1, b:], 90, axis=1))
        thr_abs = np.repeat(mid.reshape(nc, n_seeds).mean(1), n_seeds)
    rb0b, on0, rp0, rq0, ex0, Arec = run_batch7(model, sub, seeds, 'none_rec', 0.0, thr, tag, t_ms=t_ms, ranges=ranges)
    assert np.array_equal(rb0, rb0b)
    del rb0b
    M0 = [R3.measures(rb0[i, :, b:].astype(np.float64), on0[i, b:], rp0[i, b:], rq0[i, b:], thr[i], thr_abs[i])
          for i in range(len(thr))]
    SR0 = [T6.serial_r(rb0[i, :, b:].astype(np.float64), thr[i]) for i in range(len(thr))]
    del rb0, rp0, rq0, on0
    per = lambda M, cc: [{kk: M[i][kk] for kk in R3.SCAL} for i in range(cc * n_seeds, (cc + 1) * n_seeds)]
    base = [per(M0, cc) for cc in range(nc)]
    T = np.array([0.5 * (np.nanmean([x['mA'] for x in base[cc]]) + np.nanmean([x['mB'] for x in base[cc]])) for cc in range(nc)])
    pooled = lambda M, cc: [l for i in range(cc * n_seeds, (cc + 1) * n_seeds) for l in M[i]['Bm']]
    per_cfg = lambda v: [float(np.nanmean(np.asarray(v, float)[cc * n_seeds:(cc + 1) * n_seeds])) for cc in range(nc)]
    dists = {'base': [T6.dist_stats(pooled(M0, cc), T[cc], minutes) for cc in range(nc)]}
    agree = {'base': per_cfg(ex0['agree'])}
    serial = {'base': [{kk: [SR0[i][kk] for i in range(cc * n_seeds, (cc + 1) * n_seeds)] for kk in ('r_ab', 'r_ba')}
                       for cc in range(nc)]}
    out, duty, dose, cover = {}, {}, {}, {}
    for name, mode in CONDS:
        rec = Arec if mode in ('replay_self', 'replay_masked') else None
        rb, on, rp, rq, ex = run_batch7(model, sub, seeds, mode, INC * scale, thr, tag, rec=rec, t_ms=t_ms, ranges=ranges)
        M = [R3.measures(rb[i, :, b:].astype(np.float64), on[i, b:], rp[i, b:], rq[i, b:], thr[i], thr_abs[i])
             for i in range(len(thr))]
        out[name] = [per(M, cc) for cc in range(nc)]
        duty[name] = on[:, b:].mean(1).reshape(nc, n_seeds).mean(1).tolist()
        # fraction of the increment's time on that falls in the competitor's dominance, and fraction of grating 1's
        # dominance during which the increment is on
        dc, ca = [], []
        for i in range(len(thr)):
            x = rb[i, :, b:]; o = on[i, b:]
            Bdom = (x[1] - x[0]) > thr[i]; Adom = (x[0] - x[1]) > thr[i]
            dc.append(float((o * Bdom).sum() / o.sum()) if o.sum() > 0 else np.nan)
            ca.append(float((o * Adom).sum() / Adom.sum()) if Adom.sum() > 0 else np.nan)
        dose[name] = per_cfg(dc)
        cover[name] = per_cfg(ca)
        dists[name] = [T6.dist_stats(pooled(M, cc), T[cc], minutes) for cc in range(nc)]
        agree[name] = per_cfg(ex['agree'])
        if mode == 'gated':
            SR = [T6.serial_r(rb[i, :, b:].astype(np.float64), thr[i]) for i in range(len(thr))]
            serial[name] = [{kk: [SR[i][kk] for i in range(cc * n_seeds, (cc + 1) * n_seeds)] for kk in ('r_ab', 'r_ba')}
                            for cc in range(nc)]
        del rb, on, rp, rq, M
    del Arec
    return dict(base=base, out=out, T=T.tolist(), duty=duty, dose_comp=dose, cover_att=cover, dists=dists,
                serial=serial, agree=agree)


def accept7(model, n_target, rng_seed, chunk, t_ms, tag):
    """At least 10 episodes per seed on average, median coefficient of variation in CV_RANGE, mean duration of at least
    MIN_T_RATE (rate models), and baseline serial correlation r_ab below R_MAX (8 seeds of t_ms, the first test's
    scoring; mean of the seeds' Fisher-z values). Parameters drawn uniformly within RANGES[model]."""
    ranges = RANGES[model]
    rng = np.random.default_rng(rng_seed)
    acc, r_acc, cv_acc, T_acc, n_drawn = [], [], [], [], 0
    rej = dict(few_episodes=0, cv=0, short=0, serial=0, surplus=0)
    seeds = list(range(R3.N_SEEDS))
    b = int(R3.BURN_MS / R3.DT_REC)
    while len(acc) < n_target and n_drawn < MAX_DRAWS:
        cand = [{kk: float(rng.uniform(*v)) for kk, v in ranges.items()} for _ in range(chunk)]
        n_drawn += len(cand)
        rb = run_batch7(model, cand, seeds, 'none', 0.0, None, tag + '_accept', t_ms=t_ms, ranges=ranges)[0]
        thr = R3.THETA_FRAC * rb[:, :, b:].mean(axis=(1, 2))
        for c in range(len(cand)):
            n_ep, cvs, allD, rs = [], [], [], []
            for i in range(c * len(seeds), (c + 1) * len(seeds)):
                x = rb[i, :, b:].astype(np.float64)
                E = R.episodes_diff(x, thr[i], R3.DT_REC)
                dd = [l for _, l in E]; n_ep.append(len(E)); allD += dd
                cvs.append(np.std(dd) / np.mean(dd) if len(dd) > 2 else np.nan)
                rs.append(T6.serial_r(x, thr[i])['r_ab'])
            if np.mean(n_ep) < 10:
                rej['few_episodes'] += 1; continue
            if not (CV_RANGE[0] <= np.nanmedian(cvs) <= CV_RANGE[1]):
                rej['cv'] += 1; continue
            if model != 'li' and np.mean(allD) < MIN_T_RATE:
                rej['short'] += 1; continue
            rs = np.asarray(rs, float); rs = rs[np.isfinite(rs)]
            r = float(np.tanh(np.mean(np.arctanh(np.clip(rs, -0.999, 0.999))))) if len(rs) else np.nan
            if not (r < R_MAX):
                rej['serial'] += 1; continue
            if len(acc) < n_target:
                acc.append(dict(cand[c])); r_acc.append(r)
                cv_acc.append(float(np.nanmedian(cvs))); T_acc.append(float(np.mean(allD)))
            else:
                rej['surplus'] += 1
        del rb
        print(f'  [{tag}] accepted {len(acc)}/{n_target} of {n_drawn} drawn', flush=True)
    return acc, r_acc, cv_acc, T_acc, n_drawn, rej


def run_model(model, a, scale, t0):
    from concurrent.futures import ProcessPoolExecutor
    seed = DRY_SEED[model] if a.dry_run else SEED[model]
    acc, r_acc, cv_acc, T_acc, nd, rej = accept7(model, a.n_config, seed, a.chunk, a.t_ms, f'test7_{model}')
    chunks = [acc[i:i + a.chunk] for i in range(0, len(acc), a.chunk)]
    Rz = dict(base=[], cond={}, T=[], duty={}, dose_comp={}, cover_att={}, dists={}, serial={}, agree={})
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_chunk7, model, ch, scale, a.t_ms, f'test7_{model}', a.seeds, RANGES[model]) for ch in chunks]
        for kk, fu in enumerate(futs):
            r = fu.result()
            Rz['base'] += r['base']; Rz['T'] += r['T']
            for grp in ('duty', 'dose_comp', 'cover_att', 'dists', 'serial', 'agree'):
                for key, v in r[grp].items():
                    Rz[grp].setdefault(key, []).extend(v)
            for key, v in r['out'].items():
                Rz['cond'].setdefault(key, []).extend(v)
            print(f'  [{model}] chunk {kk + 1}/{len(chunks)} [{time.time() - t0:.0f}s]', flush=True)
    return dict(accepted=acc, r_ab_accept=r_acc, cv_accept=cv_acc, T_accept=T_acc, n_drawn=nd, rejected=rej, results=Rz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--n-config', type=int, default=N_TARGET)
    ap.add_argument('--chunk', type=int, default=10)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--t-ms', type=float, default=120000.0)
    ap.add_argument('--seeds', type=int, default=N_SEEDS)
    ap.add_argument('--models', default='li,sfa,dep')
    ap.add_argument('--out', default='li2017_test7_results.json')
    a = ap.parse_args()
    t0 = time.time()
    scale = 0.0 if a.dry_run else 1.0
    res = {}
    for model in a.models.split(','):
        res[model] = run_model(model, a, scale, t0)
        json.dump(dict(partial=True, models=res), open(a.out + '.partial.json', 'w'), default=float)
    import scipy, platform
    json.dump(dict(dry_run=a.dry_run, models=res,
                   versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                   settings=dict(SEED=SEED, RANGES=RANGES, N_SEEDS=a.seeds, INC=INC, R_MAX=R_MAX, CV_RANGE=CV_RANGE,
                                 MIN_T_RATE=MIN_T_RATE, MAX_DRAWS=MAX_DRAWS, T_MS=a.t_ms, CHUNK=a.chunk)),
              open(a.out, 'w'), default=float)
    print(f'wrote {a.out} [{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
