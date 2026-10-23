"""
Pre-registered test of the delivery-timing account in the attention model of binocular rivalry
of Li, Rankin, Rinzel, Carrasco and Heeger (2017). See prereg_li2017_test.md for hypotheses.

    python run_li2017_test.py --dry-run --n-config 4      # code-path check, every increment set to 0
    python run_li2017_test.py                              # the registered run

The dry run exercises every schedule with a zero increment, so it produces no information about
the hypotheses. The registered run writes li2017_test_results.json and prints the hypothesis tests.
"""
import argparse, json, math, time, zlib
import numpy as np
import li2017 as L

SEED = 20261007
RANGES = dict(inp=(0.35, 0.8), wo=(0.5, 0.75), wa=(0.4, 0.9), wh=(1.5, 2.5),
              tau_h=(1000.0, 4000.0), noise=(0.03, 0.07))
N_SEEDS = 8
T_MS = 120000.0
BURN_MS = 5000.0
MIN_EP_MS = 50.0
THETA_FRAC = 0.05
INC_FRACS = (0.10, 0.20)
DELAYS = (0.25, 0.5, 1.0, 1.5)
DELAY_INC = 0.20
N_TARGET = 200


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    ph = k / n
    c = (ph + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def seed_of(*parts):
    return zlib.crc32(repr(parts).encode()) % (2 ** 31 - 1)


def draw(rng):
    return {k: float(rng.uniform(*v)) for k, v in RANGES.items()}


def episodes_diff(rb, thr, dt_rec):
    """Difference criterion. Returns list of (sign, duration_ms), first and last dropped, short ones removed."""
    d = rb[0] - rb[1]
    st = np.where(d > thr, 1, np.where(d < -thr, -1, 0)).astype(np.int8)
    ch = np.flatnonzero(np.diff(st)) + 1
    starts = np.r_[0, ch]
    ends = np.r_[ch, len(st)]
    vals = st[starts]
    eps = [(int(v), (e - s) * dt_rec) for v, s, e in zip(vals, starts, ends) if v != 0]
    eps = eps[1:-1]
    return [x for x in eps if x[1] >= MIN_EP_MS]


def episodes_abs(x, thr, dt_rec):
    st = (x > thr).astype(np.int8)
    ch = np.flatnonzero(np.diff(st)) + 1
    starts = np.r_[0, ch]
    ends = np.r_[ch, len(st)]
    eps = [(e - s) * dt_rec for v, s, e in zip(st[starts], starts, ends) if v == 1]
    eps = eps[1:-1]
    return [x for x in eps if x >= MIN_EP_MS]


class Schedule:
    """Per-instance gate. mode in {'none','gated','ungated','yoked','offset','onset'}.
    Dominance of grating 1 is read from the live binocular responses with the instance's threshold."""

    def __init__(self, mode, thr, dt, yoke=None, delay_ms=None):
        self.mode, self.thr, self.dt = mode, thr, dt
        self.yoke, self.delay = yoke, delay_ms
        N = len(thr)
        self.t_since_end = np.full(N, np.inf)    # time since grating 1 stopped being dominant
        self.t_since_start = np.zeros(N)         # time since grating 1 became dominant
        self.prev = np.zeros(N, bool)

    def __call__(self, idx, state):
        rb = state['rb']
        domA = (rb[:, 0] - rb[:, 1]) > self.thr
        started = domA & ~self.prev
        ended = ~domA & self.prev
        self.t_since_start = np.where(started, 0.0, self.t_since_start + self.dt)
        self.t_since_end = np.where(ended, 0.0, self.t_since_end + self.dt)
        self.prev = domA
        if self.mode == 'none':
            return np.zeros_like(domA)
        if self.mode == 'ungated':
            return np.ones_like(domA)
        if self.mode == 'gated':
            return domA
        if self.mode == 'yoked':
            j = min(idx // 2, self.yoke.shape[1] - 1)
            return self.yoke[:, j]
        if self.mode == 'offset':
            return domA | (self.t_since_end < self.delay)
        if self.mode == 'onset':
            return domA & (self.t_since_start >= self.delay)
        raise ValueError(self.mode)


def run_batch(cfgs, seeds, mode, inc_frac, thr=None, yoke=None, delay_ms=None, base_p=None):
    """Simulate len(cfgs) * len(seeds) instances. Returns rb (N,2,nrec), on (N,nrec)."""
    p = dict(base_p)
    p['T'] = T_MS
    N = len(cfgs) * len(seeds)
    rep = lambda k: np.repeat([c[k] for c in cfgs], len(seeds))
    pv = dict(in1=rep('inp'), in2=rep('inp'), wo=rep('wo'), wa=rep('wa'), wh=rep('wh'), tau_h=rep('tau_h'))
    noise = rep('noise')
    inc = rep('inp') * inc_frac
    thr_i = np.zeros(N) if thr is None else thr
    dly = None if delay_ms is None else delay_ms
    sch = Schedule(mode, thr_i, p['dt'], yoke=yoke, delay_ms=dly)
    rb, on = simulate_multi(p, pv, noise, inc, sch, seeds, cfgs)
    return rb, on


def simulate_multi(p, pv, noise, inc, sch, seeds, cfgs):
    """Instances share one vectorised integration; noise sd varies per instance, so it is applied
    as a per-instance scale on a unit-sd OU process. The random stream depends only on the batch's
    configurations and seeds, so every condition run on the same batch receives the same noise
    (common random numbers)."""
    N = len(noise)
    s = seed_of('li2017', tuple(int(x) for x in seeds),
                tuple(round(c[k], 6) for c in cfgs for k in sorted(RANGES)))
    rb, on = L.simulate(p, N=N, seed=s, noise_sigma=1.0, record_every=2, params_vec=pv,
                        inc=inc, schedule=sch, noise_scale=noise)
    return rb, on


def summarise(rb, thr_diff, thr_abs, dt_rec, burn):
    """Per instance: mean duration of grating 1 and 2 (difference criterion), of grating 2
    (absolute criterion), episode count and CV of all durations."""
    b = int(burn / dt_rec)
    out = []
    for i in range(rb.shape[0]):
        x = rb[i, :, b:].astype(np.float64)
        E = episodes_diff(x, thr_diff[i], dt_rec)
        A = [l for s, l in E if s == 1]
        B = [l for s, l in E if s == -1]
        Babs = episodes_abs(x[1], thr_abs[i], dt_rec)
        allE = [l for _, l in E]
        out.append(dict(mA=np.mean(A) if A else np.nan, mB=np.mean(B) if B else np.nan,
                        mBabs=np.mean(Babs) if Babs else np.nan, n=len(E),
                        cv=(np.std(allE) / np.mean(allE)) if len(allE) > 2 else np.nan))
    return out


def by_config(rows, ncfg, nseed, key):
    v = np.array([r[key] for r in rows], float).reshape(ncfg, nseed)
    with np.errstate(all='ignore'):
        return np.nanmean(v, axis=1)


def pct(new, old):
    with np.errstate(all='ignore'):
        return np.where(old > 0, 100.0 * (new - old) / old, np.nan)


def run_chunk(sub, conds, scale, t_ms):
    global T_MS
    T_MS = t_ms
    base_p = L.default_params(1)
    dt_rec = base_p['dt'] * 2
    seeds = list(range(N_SEEDS))
    """Baseline and every condition for one batch of configurations, all with the same noise."""
    rb0, _ = run_batch(sub, seeds, 'none', 0.0, base_p=base_p)
    b = int(BURN_MS / dt_rec)
    thr = THETA_FRAC * rb0[:, :, b:].mean(axis=(1, 2))
    thr_abs = np.repeat(np.median(rb0[:, 1, b:], axis=1).reshape(len(sub), N_SEEDS).mean(1), N_SEEDS)
    rows0 = summarise(rb0, thr, thr_abs, dt_rec, BURN_MS)
    domA = (rb0[:, 0, :] - rb0[:, 1, :]) > thr[:, None]
    idx = np.roll(np.arange(len(thr)).reshape(len(sub), N_SEEDS), -1, axis=1).ravel()
    yoke = domA[idx]          # each instance follows the next seed of its own configuration
    base = []
    for c in range(len(sub)):
        r = rows0[c * N_SEEDS:(c + 1) * N_SEEDS]
        base.append(dict(mA=float(np.nanmean([x['mA'] for x in r])),
                         mB=float(np.nanmean([x['mB'] for x in r])),
                         mBabs=float(np.nanmean([x['mBabs'] for x in r]))))
    Tm = np.repeat([0.5 * (bb['mA'] + bb['mB']) for bb in base], N_SEEDS)
    out = {}
    for mode, frac, dmult in conds:
        key = f'{mode}_{frac:g}' + (f'_d{dmult:g}' if dmult is not None else '')
        rb, on = run_batch(sub, seeds, mode, frac * scale, thr=thr,
                           yoke=yoke if mode == 'yoked' else None,
                           delay_ms=None if dmult is None else dmult * Tm, base_p=base_p)
        rows = summarise(rb, thr, thr_abs, dt_rec, BURN_MS)
        out[key] = []
        for c in range(len(sub)):
            r = rows[c * N_SEEDS:(c + 1) * N_SEEDS]
            out[key].append(dict(mA=float(np.nanmean([x['mA'] for x in r])),
                                 mB=float(np.nanmean([x['mB'] for x in r])),
                                 mBabs=float(np.nanmean([x['mBabs'] for x in r])),
                                 duty=float(on[c * N_SEEDS:(c + 1) * N_SEEDS].mean())))
    return base, out



def main():
    global T_MS
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--n-config', type=int, default=N_TARGET)
    ap.add_argument('--chunk', type=int, default=25, help='configurations per vectorised batch')
    ap.add_argument('--out', default='li2017_test_results.json')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--t-ms', type=float, default=120000.0, help='model time per run (registered: 120000)')
    args = ap.parse_args()
    T_MS = args.t_ms
    base_p = L.default_params(1)
    dt_rec = base_p['dt'] * 2
    seeds = list(range(N_SEEDS))
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    # ---- 1. acceptance on baseline only
    accepted, n_drawn = [], 0
    while len(accepted) < args.n_config:
        cand = [draw(rng) for _ in range(args.chunk)]
        n_drawn += len(cand)
        rb, _ = run_batch(cand, seeds, 'none', 0.0, base_p=base_p)
        thr = THETA_FRAC * rb[:, :, int(BURN_MS / dt_rec):].mean(axis=(1, 2))
        thr_abs = np.median(rb[:, 1, int(BURN_MS / dt_rec):], axis=1)
        rows = summarise(rb, thr, thr_abs, dt_rec, BURN_MS)
        for c in range(len(cand)):
            sl = slice(c * N_SEEDS, (c + 1) * N_SEEDS)
            r = rows[sl]
            n_ep = np.mean([x['n'] for x in r])
            durs_cv = np.nanmedian([x['cv'] for x in r])
            if n_ep >= 10 and 0.3 <= durs_cv <= 0.7 and len(accepted) < args.n_config:
                cfg = dict(cand[c])
                cfg['thr'] = thr[sl].tolist()
                cfg['thr_abs'] = float(np.mean(thr_abs[sl]))
                cfg['base'] = dict(mA=float(np.nanmean([x['mA'] for x in r])),
                                   mB=float(np.nanmean([x['mB'] for x in r])),
                                   mBabs=float(np.nanmean([x['mBabs'] for x in r])),
                                   cv=float(durs_cv), n=float(n_ep))
                accepted.append(cfg)
        print(f'  acceptance: {len(accepted)}/{args.n_config} accepted of {n_drawn} drawn '
              f'[{time.time() - t0:.0f}s]', flush=True)
    published = dict(inp=0.5, wo=0.65, wa=0.6, wh=2.0, tau_h=2000.0, noise=0.05)

    # ---- 2. conditions
    conds = [('gated', f, None) for f in INC_FRACS] + [('ungated', f, None) for f in INC_FRACS] + \
            [('yoked', f, None) for f in INC_FRACS] + \
            [('offset', DELAY_INC, d) for d in DELAYS] + [('onset', DELAY_INC, d) for d in DELAYS]
    scale = 0.0 if args.dry_run else 1.0   # dry run: every increment is zero

    results = {}
    bases = []
    chunks = [accepted[c0:c0 + args.chunk] for c0 in range(0, len(accepted), args.chunk)]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_chunk, ch, conds, scale, T_MS) for ch in chunks]
        for k, fu in enumerate(futs):
            bse, out = fu.result()
            bases += bse
            for key, v in out.items():
                results.setdefault(key, []).extend(v)
            print(f'  conditions: chunk {k + 1}/{len(chunks)} done [{time.time() - t0:.0f}s]', flush=True)
    for c, bb in zip(accepted, bases):
        c['base_matched'] = bb

    # ---- 3. published parameter set as a single pre-specified configuration
    pub = dict(published)
    pub_base, pub_out = run_chunk([pub], conds, scale, T_MS)
    pub_res = {k: v[0] for k, v in pub_out.items()}
    pub['base_matched'] = pub_base[0]

    # ---- 4. hypothesis tests
    base = {k: np.array([c['base_matched'][k] for c in accepted]) for k in ('mA', 'mB', 'mBabs')}
    def comp(key, crit='mB'):
        return pct(np.array([r[crit] for r in results[key]]), base[crit])
    def att(key):
        return pct(np.array([r['mA'] for r in results[key]]), base['mA'])
    def prop_pos(x):
        x = x[np.isfinite(x)]
        k, n = int((x > 0).sum()), len(x)
        return dict(k=k, n=n, ci=wilson(k, n), median=float(np.median(x)) if n else float('nan'))
    tests = {}
    for crit, lab in (('mB', 'difference'), ('mBabs', 'absolute')):
        for f in INC_FRACS:
            g, u = prop_pos(comp(f'gated_{f:g}', crit)), prop_pos(comp(f'ungated_{f:g}', crit))
            tests[f'{lab}_{f:g}'] = dict(gated=g, ungated=u,
                                         holds=bool(g['ci'][0] > 0.5 and u['ci'][1] < 0.5))
            diff = comp(f'gated_{f:g}', crit) - comp(f'yoked_{f:g}', crit)
            tests[f'H3_{lab}_{f:g}'] = prop_pos(diff)
            tests[f'H3_{lab}_{f:g}']['holds'] = bool(tests[f'H3_{lab}_{f:g}']['ci'][0] > 0.5)
    off = [float(np.nanmedian(comp(f'gated_{DELAY_INC:g}')))] + \
          [float(np.nanmedian(comp(f'offset_{DELAY_INC:g}_d{d:g}'))) for d in DELAYS]
    ons = [float(np.nanmedian(comp(f'onset_{DELAY_INC:g}_d{d:g}'))) for d in DELAYS]
    tests['H4'] = dict(medians=off, delays=[0.0] + list(DELAYS),
                       holds=bool(off[-1] < 0 and all(b <= a for a, b in zip(off, off[1:]))))
    tests['H5'] = dict(medians=ons, delays=list(DELAYS), holds=bool(all(m > 0 for m in ons)))
    # H6: gated ratio vs exp(-T/tau_h), permutation of tau_h
    f = INC_FRACS[0]
    ratio = comp(f'gated_{f:g}') / att(f'gated_{f:g}')
    Tm = 0.5 * (base['mA'] + base['mB'])
    tau = np.array([c['tau_h'] for c in accepted])
    pred = np.exp(-Tm / tau)
    ok = np.isfinite(ratio) & np.isfinite(pred) & (np.abs(att(f'gated_{f:g}')) > 1e-9)
    rk = lambda v: np.argsort(np.argsort(v))
    sp = lambda a, b: float(np.corrcoef(rk(a), rk(b))[0, 1])
    if ok.sum() < 10:
        tests['H6'] = dict(rho=float('nan'), n=int(ok.sum()), p_one_sided=float('nan'), holds=False)
    obs = sp(ratio[ok], pred[ok]) if ok.sum() >= 10 else float('nan')
    prng = np.random.default_rng(SEED + 1)
    null = np.array([sp(ratio[ok], np.exp(-Tm[ok] / prng.permutation(tau[ok]))) for _ in range(5000)]) \
        if ok.sum() >= 10 else np.array([np.nan])
    if ok.sum() >= 10:
        pval = float((np.sum(null >= obs) + 1) / (len(null) + 1))
        tests['H6'] = dict(rho=obs, n=int(ok.sum()), p_one_sided=pval, holds=bool(obs > 0 and pval < 0.05))
    tests['H1'] = dict(holds=bool(tests[f'difference_{INC_FRACS[0]:g}']['holds'] and
                                  tests[f'difference_{INC_FRACS[1]:g}']['holds']))
    tests['H2'] = dict(holds=bool(tests[f'absolute_{INC_FRACS[0]:g}']['holds'] and
                                  tests[f'absolute_{INC_FRACS[1]:g}']['holds']))
    tests['H3'] = dict(holds=bool(all(tests[f'H3_{lab}_{f:g}']['holds']
                                      for lab in ('difference', 'absolute') for f in INC_FRACS)))

    json.dump(dict(dry_run=args.dry_run, n_drawn=n_drawn, accepted=accepted, results=results,
                   published=dict(config=pub, results=pub_res), tests=tests,
                   settings=dict(SEED=SEED, RANGES=RANGES, N_SEEDS=N_SEEDS, T_MS=T_MS, BURN_MS=BURN_MS,
                                 MIN_EP_MS=MIN_EP_MS, THETA_FRAC=THETA_FRAC, INC_FRACS=INC_FRACS,
                                 DELAYS=DELAYS, DELAY_INC=DELAY_INC)),
              open(args.out, 'w'), indent=1, default=float)
    print('\nHYPOTHESES' + ('  (DRY RUN: increments are zero, results carry no information)' if args.dry_run else ''))
    for h in ('H1', 'H2', 'H3', 'H4', 'H5', 'H6'):
        print(f'  {h}: {"holds" if tests[h]["holds"] else "does not hold"}')
    for k, v in tests.items():
        if k not in ('H1', 'H2', 'H3'):
            print(f'    {k}: {v}')
    print(f'wrote {args.out} [{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
