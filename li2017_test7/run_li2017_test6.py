"""
Sixth test, with a pre-specified analysis plan (analysis_plan_li2017_test6.md). It answers an independent referee's
questions about the gated lengthening of the competitor:
  Part A (Li et al.'s model, 60 new configurations): is the lengthening a shift of the whole duration distribution or the
    loss of short episodes; does it depend on the strength of serial dependence; does a top-down bias in the model's own
    attention layer behave like the input increment; is it reproduced by replaying the trial's own baseline schedule
    (open loop, same noise) or another seed's; and how does it change as the increment is withdrawn earlier before the
    switch.
  Part B (two canonical rate models of the Laing-Chow form, spike-frequency adaptation and synaptic depression, 60
    configurations each): do the sign reversal, the distribution result and the serial-dependence result generalise.

    python run_li2017_test6.py --dry-run --n-config 4 --t-ms 30000 --chunk 4 --seeds 4    # code path, increments 0
    python run_li2017_test6.py --workers 6                                                # the run
The chunk size (10) is part of the design (it sets the noise streams); the number of workers does not affect results.

Reuses, unchanged: li2017.py (through li2017_t6.py, whose simulate() is identical for input increments),
run_li2017_test.py and run_li2017_test3.py (acceptance, dominance, observers, measures).
"""
import argparse, json, time
import numpy as np
import li2017 as L
import li2017_t6 as L6
import rate_models as RM
import run_li2017_test as R
import run_li2017_test3 as R3

SEED_LI, SEED_SFA, SEED_DEP = 20261111, 20261112, 20261113
DRY_SEED = 111
N_TARGET = 60
N_SEEDS = 32
INC = 0.10                                   # increment, fraction of input strength (input target)
ATT_FRACS = (0.25, 0.5)                      # attention-layer bias, fraction of the configuration's median |r1 - r2|
NT_K = (3.0, 10.0, 30.0)                     # gate withdrawn once grating 1's advantage falls below k thresholds
RANGES_RATE = dict(sfa=dict(I=(0.4, 1.0), beta=(0.6, 1.4), g=(0.1, 0.8), tau_a=(1000.0, 6000.0), noise=(0.02, 0.15)),
                   dep=dict(I=(0.4, 1.0), beta=(0.6, 1.4), gam=(0.5, 4.0), tau_d=(1000.0, 6000.0), noise=(0.02, 0.15)))
MIN_EP = 50.0
MIN_T_RATE = 500.0                           # rate models: minimum mean dominance duration (ms) for acceptance
QS = tuple(range(5, 100, 5))                  # quantiles (%) of pooled merged competitor durations

# conditions: (name, schedule mode, target, size); size is the input fraction, or the attention fraction
CONDS_LI = ([('gated', 'gated', 'input', INC), ('ungated', 'ungated', 'input', INC),
             ('replay_self', 'replay_self', 'input', INC), ('replay_other', 'replay_other', 'input', INC)] +
            [(f'gated_nt{k:g}', 'gated_nt', 'input', INC) for k in NT_K] +
            [(f'att_gated{a:g}', 'gated', 'attention', a) for a in ATT_FRACS] +
            [(f'att_ungated{a:g}', 'ungated', 'attention', a) for a in ATT_FRACS])
CONDS_RATE = [('gated', 'gated', 'input', INC), ('ungated', 'ungated', 'input', INC),
              ('replay_self', 'replay_self', 'input', INC), ('replay_other', 'replay_other', 'input', INC)]


class Schedule6(R3.Schedule3):
    """The third test's schedule (observers, dominance at thr), run in mode 'none' for the observers, plus:
      none_rec:     zero increment; records grating 1's dominance (d > thr) at every integration step
      gated/ungated: as in the third test
      replay_self:  on whenever grating 1 was dominant at the same step of this instance's own baseline run
                    (same seed and noise; open loop: the schedule does not depend on the current state)
      replay_other: the same, from the baseline run of another seed of the same configuration (seed s + 1, cyclically)
      gated_nt:     the live gate, withdrawn for the rest of the attended period once grating 1's advantage, having
                    exceeded k thresholds, falls back below k thresholds (hysteretic: re-armed when grating 2 becomes
                    dominant)
    In every mode it accumulates, per instance, the fraction of steps at which the increment's on-state agrees with
    grating 1's current dominance (on > 0 == d > thr), and, when the model exposes it (state['ra'], Li et al.'s
    attention layer), the mean attention response for grating 1 while grating 1 and while grating 2 is dominant."""

    def __init__(self, mode6, rec=None, k=None, *args, **kw):
        super().__init__('none', *args, **kw)
        self.mode6, self.rec, self.k = mode6, rec, k
        N = args[2]
        if mode6 == 'none_rec':
            self.Arec = np.zeros((N, args[3]), bool)
        self.armed = np.zeros(N, bool); self.wd = np.zeros(N, bool)
        self.agree = np.zeros(N); self.nstep = 0
        self.attA = np.zeros(N); self.nA = np.zeros(N); self.attB = np.zeros(N); self.nB = np.zeros(N)

    def _stats(self, on, A, B, state):
        self.agree += ((on > 0) == A); self.nstep += 1
        if 'ra' in state:
            ra = state['ra'][:, 0]
            self.attA += np.where(A, ra, 0.0); self.nA += A
            self.attB += np.where(B, ra, 0.0); self.nB += B

    def extras(self):
        with np.errstate(all='ignore'):
            return dict(agree=self.agree / max(self.nstep, 1), attA=self.attA / self.nA, attB=self.attB / self.nB)

    def __call__(self, idx, state):
        super().__call__(idx, state)
        rb = state['rb']; d = rb[:, 0] - rb[:, 1]
        A, B = d > self.thr, d < -self.thr
        m = self.mode6
        if m in ('none', 'none_rec'):
            if m == 'none_rec':
                self.Arec[:, idx] = A
            on = np.zeros(len(d))
        elif m == 'gated':
            on = A.astype(float)
        elif m == 'ungated':
            on = np.ones(len(d))
        elif m in ('replay_self', 'replay_other'):
            on = self.rec[:, idx].astype(float)
        elif m == 'gated_nt':
            hi = d > self.k * self.thr
            self.wd = (self.wd | (self.armed & A & ~hi)) & ~B
            self.armed = (self.armed | (A & hi)) & ~B
            on = (A & ~self.wd).astype(float)
        else:
            raise ValueError(m)
        self._stats(on, A, B, state)
        return on


def run_batch6(model, cfgs, seeds, mode, size, thr, tag, target='input', rec=None, k=None, bias=None, t_ms=120000.0,
               ranges=None):
    """One batch of instances (configurations x seeds) with common random numbers across conditions."""
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
        sch = Schedule6(mode, rec, k, thr_, p['dt'], N, nt, pools=pools)
        inc = rep('inp') * size if target == 'input' else np.repeat(bias, len(seeds)) * size
        rb, on = L6.simulate(p, N=N, seed=s, noise_sigma=1.0, record_every=2, params_vec=pv, inc=inc, schedule=sch,
                             noise_scale=rep('noise'), target=target)
    else:
        nt = int(round(t_ms / RM.DT)) + 1
        pv = {kk: rep(kk) for kk in ranges if kk != 'noise'}
        sch = Schedule6(mode, rec, k, thr_, RM.DT, N, nt, pools=pools)
        rb, on = RM.simulate(model, pv, N, s, t_ms, inc=rep('I') * size, schedule=sch, record_every=2,
                             noise_scale=rep('noise'))
    kk = rb.shape[2]
    out = (rb, on, sch.obs.rec[:, :kk], sch.obs2.rec[:, :kk], sch.extras())
    return out + ((sch.Arec,) if mode == 'none_rec' else ())


def alt_seq(x, thr):
    """Alternating episode sequence under the first test's scoring: difference criterion, episodes under 50 ms dropped,
    consecutive episodes of the same sign merged. Returns (signs, durations)."""
    E = R.episodes_diff(x, thr, R3.DT_REC)
    sg, du = [], []
    for s_, l in E:
        if sg and sg[-1] == s_:
            du[-1] += l
        else:
            sg.append(s_); du.append(l)
    return np.array(sg), np.array(du, float)


def serial_r(x, thr):
    """Lag-1 Pearson correlations within one run: attended duration with the following competitor duration (r_ab),
    competitor with the following attended (r_ba)."""
    sg, du = alt_seq(x, thr)
    out = {}
    for name, first in (('r_ab', 1), ('r_ba', -1)):
        i = np.flatnonzero(sg[:-1] == first)
        a_, b_ = du[i], du[i + 1]
        out[name] = float(np.corrcoef(a_, b_)[0, 1]) if len(i) >= 5 and a_.std() > 0 and b_.std() > 0 else np.nan
    return out


def lead_times(x, on, thr):
    """For each switch from an attended run (>= 50 ms) to a competitor run (>= 50 ms): the time from the last
    sample at which the increment was on (within that attended run) to the switch. Median over switches (ms)."""
    d = x[0] - x[1]
    st = np.where(d > thr, 1, np.where(d < -thr, -1, 0)).astype(np.int8)
    v, s0, e0 = R3.runs(st)
    nz = np.flatnonzero(v != 0)
    nxt_nz = np.full(len(v), -1)
    nxt_nz[nz[:-1]] = nz[1:]                       # next non-indeterminate run after each non-indeterminate run
    leads = []
    for j in nz:
        q = nxt_nz[j]
        if v[j] != 1 or q < 0 or (e0[j] - s0[j]) * R3.DT_REC < MIN_EP:
            continue
        if v[q] != -1 or (e0[q] - s0[q]) * R3.DT_REC < MIN_EP:
            continue
        seg = on[s0[j]:e0[j]]
        z = np.flatnonzero(seg > 0)
        leads.append(((e0[j] - s0[j]) - (z[-1] + 1)) * R3.DT_REC if len(z) else np.nan)
    leads = np.array(leads, float)
    return float(np.nanmedian(leads)) if np.isfinite(leads).any() else np.nan


def dist_stats(pooled, T, minutes):
    """Pooled competitor durations of one configuration and condition: percentiles, mean, the fraction shorter than
    0.25 T, the mean of those of at least 0.25 T and of at least 0.5 T, and the rate (per minute scored) of episodes of
    at least 0.25 T."""
    D = np.asarray(pooled, float)
    if len(D) < 20:
        return None
    ml = lambda c: float(D[D >= c * T].mean()) if (D >= c * T).any() else np.nan
    return dict(n=int(len(D)), q=np.percentile(D, QS).tolist(), mean=float(D.mean()),
                frac_short=float(np.mean(D < 0.25 * T)), mean_long=ml(0.25), mean_long05=ml(0.5),
                rate_long=float(np.sum(D >= 0.25 * T) / minutes))


def unmerged_B(x, thr):
    """Competitor episodes under the first test's scoring (difference criterion, episodes under 50 ms dropped)."""
    return [l for s_, l in R.episodes_diff(x, thr, R3.DT_REC) if s_ == -1]


def brief_intrusions(x, thr):
    """Number of grating-1 runs shorter than 50 ms (absorbed into competitor episodes by merged scoring)."""
    d = x[0] - x[1]
    st = np.where(d > thr, 1, np.where(d < -thr, -1, 0)).astype(np.int8)
    v, s0, e0 = R3.runs(st)
    return int(np.sum((v == 1) & ((e0 - s0) * R3.DT_REC < MIN_EP)))


def run_chunk6(model, sub, conds, scale, t_ms, tag, n_seeds, ranges):
    seeds = list(range(n_seeds))
    b = int(R3.BURN_MS / R3.DT_REC)
    nc = len(sub)
    minutes = n_seeds * (t_ms - R3.BURN_MS) / 60000.0
    rb0 = run_batch6(model, sub, seeds, 'none', 0.0, None, tag, t_ms=t_ms, ranges=ranges)[0]
    thr = R3.THETA_FRAC * rb0[:, :, b:].mean(axis=(1, 2))
    if model == 'li':
        thr_abs = np.repeat(np.median(rb0[:, 1, b:], axis=1).reshape(nc, n_seeds).mean(1), n_seeds)
    else:   # rate models: u2 is bimodal near 0 and 1, so the midpoint of its 10th and 90th percentiles
        mid = 0.5 * (np.percentile(rb0[:, 1, b:], 10, axis=1) + np.percentile(rb0[:, 1, b:], 90, axis=1))
        thr_abs = np.repeat(mid.reshape(nc, n_seeds).mean(1), n_seeds)
    rb0b, on0, rp0, rq0, ex0, Arec = run_batch6(model, sub, seeds, 'none_rec', 0.0, thr, tag, t_ms=t_ms, ranges=ranges)
    assert np.array_equal(rb0, rb0b)
    del rb0b
    # attention-layer bias scale: median |r1 - r2| over baseline samples outside the indeterminate band
    absd = []
    for c in range(nc):
        x = rb0[c * n_seeds:(c + 1) * n_seeds, :, b:]
        dd = np.abs(x[:, 0] - x[:, 1]); th = thr[c * n_seeds:(c + 1) * n_seeds, None]
        absd.append(float(np.median(dd[dd > th])) if (dd > th).any() else np.nan)
        del x, dd
    absd = np.array(absd)
    M0 = [R3.measures(rb0[i, :, b:].astype(np.float64), on0[i, b:], rp0[i, b:], rq0[i, b:], thr[i], thr_abs[i])
          for i in range(len(thr))]
    SR0 = [serial_r(rb0[i, :, b:].astype(np.float64), thr[i]) for i in range(len(thr))]
    U0 = [unmerged_B(rb0[i, :, b:].astype(np.float64), thr[i]) for i in range(len(thr))]
    BI0 = [brief_intrusions(rb0[i, :, b:].astype(np.float64), thr[i]) for i in range(len(thr))]
    del rb0, rp0, rq0, on0
    per = lambda M, cc: [{kk: M[i][kk] for kk in R3.SCAL} for i in range(cc * n_seeds, (cc + 1) * n_seeds)]
    base = [per(M0, cc) for cc in range(nc)]
    T = np.array([0.5 * (np.nanmean([x['mA'] for x in base[cc]]) + np.nanmean([x['mB'] for x in base[cc]])) for cc in range(nc)])
    pooled = lambda M, cc: [l for i in range(cc * n_seeds, (cc + 1) * n_seeds) for l in M[i]['Bm']]
    pooled_u = lambda U, cc: [l for i in range(cc * n_seeds, (cc + 1) * n_seeds) for l in U[i]]
    per_cfg = lambda v: [float(np.nanmean(np.asarray(v, float)[cc * n_seeds:(cc + 1) * n_seeds])) for cc in range(nc)]
    dists = {'base': [dist_stats(pooled(M0, cc), T[cc], minutes) for cc in range(nc)]}
    dists_u = {'base': [dist_stats(pooled_u(U0, cc), T[cc], minutes) for cc in range(nc)]}
    intr = {'base': [float(np.sum(BI0[cc * n_seeds:(cc + 1) * n_seeds]) / minutes) for cc in range(nc)]}
    agree, attn = {'base': per_cfg(ex0['agree'])}, {'base': dict(A=per_cfg(ex0['attA']), B=per_cfg(ex0['attB']))}
    serial = {'base': [{kk: [SR0[i][kk] for i in range(cc * n_seeds, (cc + 1) * n_seeds)] for kk in ('r_ab', 'r_ba')}
                       for cc in range(nc)]}
    other = np.array([(i // n_seeds) * n_seeds + (i % n_seeds + 1) % n_seeds for i in range(nc * n_seeds)])
    out, duty, lead = {}, {}, {}
    for name, mode, target, size in conds:
        k = None
        if mode == 'gated_nt':
            k = float(name.replace('gated_nt', ''))
        rec = Arec if mode == 'replay_self' else (Arec[other] if mode == 'replay_other' else None)
        rb, on, rp, rq, ex = run_batch6(model, sub, seeds, mode, size * scale, thr, tag, target=target, rec=rec, k=k,
                                        bias=absd, t_ms=t_ms, ranges=ranges)
        M = [R3.measures(rb[i, :, b:].astype(np.float64), on[i, b:], rp[i, b:], rq[i, b:], thr[i], thr_abs[i])
             for i in range(len(thr))]
        out[name] = [per(M, cc) for cc in range(nc)]
        duty[name] = on[:, b:].mean(1).reshape(nc, n_seeds).mean(1).tolist()
        dists[name] = [dist_stats(pooled(M, cc), T[cc], minutes) for cc in range(nc)]
        if name in ('gated', 'ungated') or name.startswith('att_gated'):
            U = [unmerged_B(rb[i, :, b:].astype(np.float64), thr[i]) for i in range(len(thr))]
            dists_u[name] = [dist_stats(pooled_u(U, cc), T[cc], minutes) for cc in range(nc)]
            BI = [brief_intrusions(rb[i, :, b:].astype(np.float64), thr[i]) for i in range(len(thr))]
            intr[name] = [float(np.sum(BI[cc * n_seeds:(cc + 1) * n_seeds]) / minutes) for cc in range(nc)]
        agree[name] = per_cfg(ex['agree'])
        attn[name] = dict(A=per_cfg(ex['attA']), B=per_cfg(ex['attB']))
        if mode == 'gated' and target == 'input':
            SR = [serial_r(rb[i, :, b:].astype(np.float64), thr[i]) for i in range(len(thr))]
            serial[name] = [{kk: [SR[i][kk] for i in range(cc * n_seeds, (cc + 1) * n_seeds)] for kk in ('r_ab', 'r_ba')}
                            for cc in range(nc)]
        if mode in ('gated', 'gated_nt') and target == 'input':
            LT = [lead_times(rb[i, :, b:].astype(np.float64), on[i, b:], thr[i]) for i in range(len(thr))]
            lead[name] = [float(np.nanmedian(LT[cc * n_seeds:(cc + 1) * n_seeds])) for cc in range(nc)]
        del rb, on, rp, rq, M
    del Arec
    return dict(base=base, out=out, T=T.tolist(), duty=duty, dists=dists, serial=serial, lead=lead,
                absd=absd.tolist(), dists_u=dists_u, intr=intr, agree=agree, attn=attn)


def accept_rate(model, n_target, rng_seed, chunk, t_ms, tag):
    """The third test's rule for the rate models: at least 10 episodes per seed on average and a median coefficient of
    variation in [0.3, 0.7] (8 seeds of t_ms, difference criterion), plus a mean dominance duration of at least
    MIN_T_RATE ms (the rate models' time unit is otherwise free, and shorter episodes approach the 50 ms scoring floor);
    parameters drawn uniformly within RANGES_RATE."""
    ranges = RANGES_RATE[model]
    rng = np.random.default_rng(rng_seed)
    acc, n_drawn, rej = [], 0, dict(few_episodes=0, cv=0, short=0)
    seeds = list(range(R3.N_SEEDS))
    b = int(R3.BURN_MS / R3.DT_REC)
    while len(acc) < n_target:
        cand = [{kk: float(rng.uniform(*v)) for kk, v in ranges.items()} for _ in range(chunk)]
        n_drawn += len(cand)
        rb = run_batch6(model, cand, seeds, 'none', 0.0, None, tag + '_accept', t_ms=t_ms, ranges=ranges)[0]
        thr = R3.THETA_FRAC * rb[:, :, b:].mean(axis=(1, 2))
        for c in range(len(cand)):
            n_ep, cvs, allD = [], [], []
            for i in range(c * len(seeds), (c + 1) * len(seeds)):
                E = R.episodes_diff(rb[i, :, b:].astype(np.float64), thr[i], R3.DT_REC)
                dd = [l for _, l in E]; n_ep.append(len(E)); allD += dd
                cvs.append(np.std(dd) / np.mean(dd) if len(dd) > 2 else np.nan)
            if np.mean(n_ep) < 10:
                rej['few_episodes'] += 1; continue
            if not (0.3 <= np.nanmedian(cvs) <= 0.7):
                rej['cv'] += 1; continue
            if np.mean(allD) < MIN_T_RATE:
                rej['short'] += 1; continue
            if len(acc) < n_target:
                acc.append(dict(cand[c]))
        print(f'  [{tag}] accepted {len(acc)}/{n_target} of {n_drawn} drawn', flush=True)
    return acc, n_drawn, rej


def run_model(model, a, scale, t0):
    from concurrent.futures import ProcessPoolExecutor
    if model == 'li':
        seed = DRY_SEED if a.dry_run else SEED_LI
        acc, nd = R3.accept(a.n_config, seed, R3.RANGES, a.chunk, a.t_ms, 'test6_li', True)
        rej, ranges, conds = None, R3.RANGES, CONDS_LI
    else:
        seed = (DRY_SEED + (1 if model == 'sfa' else 2)) if a.dry_run else (SEED_SFA if model == 'sfa' else SEED_DEP)
        acc, nd, rej = accept_rate(model, a.n_config, seed, a.chunk, a.t_ms, f'test6_{model}')
        ranges, conds = RANGES_RATE[model], CONDS_RATE
    chunks = [acc[i:i + a.chunk] for i in range(0, len(acc), a.chunk)]
    Rz = dict(base=[], cond={}, T=[], duty={}, dists={}, serial={}, lead={}, absd=[], dists_u={}, intr={}, agree={},
              attn={})
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_chunk6, model, ch, conds, scale, a.t_ms, f'test6_{model}', a.seeds, ranges) for ch in chunks]
        for kk, fu in enumerate(futs):
            r = fu.result()
            Rz['base'] += r['base']; Rz['T'] += r['T']; Rz['absd'] += r['absd']
            for grp in ('duty', 'dists', 'serial', 'lead', 'dists_u', 'intr', 'agree'):
                for key, v in r[grp].items():
                    Rz[grp].setdefault(key, []).extend(v)
            for key, v in r['attn'].items():
                Rz['attn'].setdefault(key, dict(A=[], B=[]))
                Rz['attn'][key]['A'].extend(v['A']); Rz['attn'][key]['B'].extend(v['B'])
            for key, v in r['out'].items():
                Rz['cond'].setdefault(key, []).extend(v)
            print(f'  [{model}] chunk {kk + 1}/{len(chunks)} [{time.time() - t0:.0f}s]', flush=True)
    return dict(accepted=acc, n_drawn=nd, rejected=rej, results=Rz,
                conds=[list(map(str, c)) for c in conds])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--n-config', type=int, default=N_TARGET)
    ap.add_argument('--chunk', type=int, default=10)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--t-ms', type=float, default=120000.0)
    ap.add_argument('--seeds', type=int, default=N_SEEDS)
    ap.add_argument('--models', default='li,sfa,dep')
    ap.add_argument('--out', default='li2017_test6_results.json')
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
                   settings=dict(SEED_LI=SEED_LI, SEED_SFA=SEED_SFA, SEED_DEP=SEED_DEP, RANGES_LI=R3.RANGES,
                                 RANGES_RATE=RANGES_RATE, N_SEEDS=a.seeds, INC=INC, ATT_FRACS=ATT_FRACS, NT_K=NT_K,
                                 QS=QS, MIN_T_RATE=MIN_T_RATE, T_MS=a.t_ms, CHUNK=a.chunk)),
              open(a.out, 'w'), default=float)
    print(f'wrote {a.out} [{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
