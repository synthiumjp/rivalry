"""
Sampling-sensitivity check of the sign reversal in the attention model of binocular rivalry of Li, Rankin,
Rinzel, Carrasco and Heeger (2017). See analysis_plan_li2017_sens.md.

    python run_li2017_sens.py --dry-run --n-config 4 --t-ms 30000 --chunk 4 --seeds 4   # code path, increments 0
    python run_li2017_sens.py --ensemble L      # log-uniform sampling over widened ranges, CV in [0.2, 0.8]
    python run_li2017_sens.py --ensemble U      # the third test's ranges, uniform, no CV filter

Uses li2017.py and run_li2017_test3.py unchanged (simulation, dominance, observer, measures, run_chunk).
Only the sampler, the acceptance rule, the number of seeds and the conditions are new.
"""
import argparse, json, time
import numpy as np
import run_li2017_test as R
import run_li2017_test3 as R3

N_SEEDS = 16                          # seeds per condition (the fourth and fifth tests used 32)
R3.N_SEEDS = N_SEEDS                  # run_chunk reads this module global
ACC_SEEDS = 8                         # seeds for acceptance, as in the third to fifth tests
INC = 0.10
CONDS = [('gated', INC, None), ('ungated', INC, None)]
SEEDS = dict(L=20261025, U=20261026)
DRY_SEED = 107
N_TARGET = 60
WIDEN = 0.25                          # ensemble L: each range widened by 25% of its log-width on each side


def ranges_L():
    out = {}
    for k, (lo, hi) in R.RANGES.items():
        f = (hi / lo) ** WIDEN
        out[k] = (lo / f, hi * f)
    return out


ENS = dict(L=dict(ranges=ranges_L(), log=True, cv=(0.2, 0.8)),
           U=dict(ranges=dict(R.RANGES), log=False, cv=None))


def draw(rng, ranges, log):
    if log:
        return {k: float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) for k, (lo, hi) in ranges.items()}
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in ranges.items()}


def accept(n_target, rng_seed, ens, chunk, t_ms, tag):
    """As R3.accept: at least 10 episodes per seed on average (difference criterion, 8 seeds of t_ms), and,
    if ens['cv'] is set, a median coefficient of variation within it. Rejections are counted by reason."""
    rng = np.random.default_rng(rng_seed)
    acc, n_drawn, rej = [], 0, dict(few_episodes=0, cv=0)
    seeds = list(range(ACC_SEEDS))
    b = int(R3.BURN_MS / R3.DT_REC)
    while len(acc) < n_target:
        cand = [draw(rng, ens['ranges'], ens['log']) for _ in range(chunk)]
        n_drawn += len(cand)
        rb, _, _, _ = R3.run_batch(cand, seeds, 'none', 0.0, None, None, tag + '_accept', t_ms=t_ms, ranges=ens['ranges'])
        thr = R3.THETA_FRAC * rb[:, :, b:].mean(axis=(1, 2))
        for c in range(len(cand)):
            n_ep, cvs = [], []
            for i in range(c * ACC_SEEDS, (c + 1) * ACC_SEEDS):
                E = R.episodes_diff(rb[i, :, b:].astype(np.float64), thr[i], R3.DT_REC)
                dd = [l for _, l in E]; n_ep.append(len(E))
                cvs.append(np.std(dd) / np.mean(dd) if len(dd) > 2 else np.nan)
            if np.mean(n_ep) < 10:
                rej['few_episodes'] += 1; continue
            if ens['cv'] is not None and not (ens['cv'][0] <= np.nanmedian(cvs) <= ens['cv'][1]):
                rej['cv'] += 1; continue
            if len(acc) < n_target:
                acc.append(dict(cand[c], cv=float(np.nanmedian(cvs))))
        del rb
        print(f'  [{tag}] accepted {len(acc)}/{n_target} of {n_drawn} drawn', flush=True)
    return acc, n_drawn, rej


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ensemble', choices=['L', 'U'], default='L')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--n-config', type=int, default=N_TARGET)
    ap.add_argument('--chunk', type=int, default=10)
    ap.add_argument('--t-ms', type=float, default=120000.0)
    ap.add_argument('--seeds', type=int, default=N_SEEDS)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    R3.N_SEEDS = a.seeds
    ens = ENS[a.ensemble]
    out = a.out or f'li2017_sens_{a.ensemble}{"_dry" if a.dry_run else ""}.json'
    t0 = time.time()
    tag = f'sens{a.ensemble}'
    acc, nd, rej = accept(a.n_config, DRY_SEED if a.dry_run else SEEDS[a.ensemble], ens, a.chunk, a.t_ms, tag)
    scale = 0.0 if a.dry_run else 1.0
    cfgs = [{k: c[k] for k in ens['ranges']} for c in acc]
    Rz = dict(base=[], cond={}, T=[])
    for i in range(0, len(cfgs), a.chunk):
        base, o, T, _, _ = R3.run_chunk(cfgs[i:i + a.chunk], CONDS, scale, a.t_ms, tag, ens['ranges'])
        Rz['base'] += base; Rz['T'] += T
        for k, v in o.items(): Rz['cond'].setdefault(k, []).extend(v)
        print(f'  chunk {i // a.chunk + 1}/{-(-len(cfgs) // a.chunk)} [{time.time() - t0:.0f}s]', flush=True)
        json.dump(dict(chunks_done=i // a.chunk + 1, results=Rz), open(out + '.partial.json', 'w'), default=float)
    import scipy, platform
    json.dump(dict(dry_run=a.dry_run, ensemble=a.ensemble, n_drawn=nd, rejected=rej, accepted=acc, results=Rz,
                   versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                   settings=dict(SEED=SEEDS[a.ensemble], RANGES=ens['ranges'], LOG=ens['log'], CV=ens['cv'],
                                 N_SEEDS=a.seeds, ACC_SEEDS=ACC_SEEDS, INC=INC, T_MS=a.t_ms, CHUNK=a.chunk,
                                 CONDS=[list(map(str, c)) for c in CONDS])),
              open(out, 'w'), default=float)
    print(f'wrote {out} [{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
