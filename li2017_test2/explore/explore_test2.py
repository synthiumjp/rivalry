"""Exploratory follow-up to the second registered test (not registered).
1. Chong et al.'s timing (report latency, 520 ms ramps) with smaller increments (5%, 10%, 20%),
   because the registered doubling locks the attended grating into exclusive dominance.
2. No-adaptation sample: gated delivery whose onset waits 100 ms into the attended episode, and the
   report-contingent gate at 20%, to test whether brief attended excursions captured by a gate without
   latency explain the competitor's shortening there.
3. Early and late windows, competitor durations merged across attended excursions shorter than
   20, 50, 100 and 200 ms (25 configurations).
Same batches and seeds as the registered run, so baselines are identical."""
import sys, json, time
import numpy as np
sys.path.insert(0, '..')
import run_li2017_test2 as R2
import run_li2017_test as R1


class Sched(R2.Schedule2):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.a_since = np.zeros(len(self.thr))
    def __call__(self, idx, state):
        if self.mode != 'gated_lat':
            return super().__call__(idx, state)
        rb = state['rb']; d = rb[:, 0] - rb[:, 1]; A = d > self.thr
        self.a_since = np.where(A, self.a_since + self.dt, 0.0)
        return (A & (self.a_since >= 100.0)).astype(float)
R2.Schedule2 = Sched


def merged_at(x, thr, maxA):
    d = x[0] - x[1]; st = np.where(d > thr, 1, np.where(d < -thr, -1, 0)).astype(np.int8)
    v, s0, e0 = R2.runs(st); L = (e0 - s0) * 1.0
    merged, cur, started = [], 0.0, False
    for a, l in zip(v, L):
        if a == 1 and l >= maxA:
            if started and cur > 0: merged.append(cur)
            cur, started = 0.0, True
        elif a == -1 and l >= 50:
            cur += l
        elif started and cur > 0:
            cur += l
    return float(np.mean(merged)) if merged else np.nan


def run_set(sub, conds, tag, ranges, extra_merge=False):
    seeds = list(range(8)); b = 5000
    rb0, on0 = R2.run_batch(sub, seeds, 'none', 0.0, None, None, tag, ranges=ranges)
    thr = 0.05 * rb0[:, :, b:].mean(axis=(1, 2))
    thr_abs = np.repeat(np.median(rb0[:, 1, b:], axis=1).reshape(len(sub), 8).mean(1), 8)
    M0 = [R2.measures(rb0[i, :, b:].astype(np.float64), on0[i, b:], thr[i], thr_abs[i]) for i in range(len(thr))]
    nc = len(sub)
    T = np.array([0.5 * (np.nanmean([M0[i]['mA'] for i in range(c * 8, c * 8 + 8)]) + np.nanmean([M0[i]['mB'] for i in range(c * 8, c * 8 + 8)])) for c in range(nc)])
    Bm0 = [[l for i in range(c * 8, c * 8 + 8) for l in M0[i]['Bm']] for c in range(nc)]
    dL = np.array([R2.match_dL(Bm0[c], T[c]) for c in range(nc)])
    out = {'base': [[{k: M0[i][k] for k in R2.SCAL} for i in range(c * 8, c * 8 + 8)] for c in range(nc)]}
    if extra_merge:
        out['base_mt'] = [[{f'm{t}': merged_at(rb0[i, :, b:].astype(np.float64), thr[i], t) for t in (20, 50, 100, 200)} for i in range(c * 8, c * 8 + 8)] for c in range(nc)]
    del rb0
    for mode, frac in conds:
        rb, on = R2.run_batch(sub, seeds, mode, frac, thr, T, tag, dL=dL, ranges=ranges)
        M = [R2.measures(rb[i, :, b:].astype(np.float64), on[i, b:], thr[i], thr_abs[i]) for i in range(len(thr))]
        key = f'{mode}_{frac:g}'
        out[key] = [[{k: M[i][k] for k in R2.SCAL} for i in range(c * 8, c * 8 + 8)] for c in range(nc)]
        if extra_merge:
            out[key + '_mt'] = [[{f'm{t}': merged_at(rb[i, :, b:].astype(np.float64), thr[i], t) for t in (20, 50, 100, 200)} for i in range(c * 8, c * 8 + 8)] for c in range(nc)]
        del rb
    return out


def job(args):
    sub, conds, tag, rng_key, em = args
    return run_set(sub, conds, tag, R2.RANGES if tag == 'main' else R2.RANGES_NA, em)


if __name__ == '__main__':
    D = json.load(open('../li2017_test2_results.json'))
    acc, acc_na = D['accepted'], D['accepted_na']
    jobs = [(acc[i:i + 25], [('chong', 0.05), ('chong', 0.10), ('chong', 0.20)], 'main', None, False) for i in range(0, 200, 25)]
    jobs += [(acc_na[i:i + 25], [('gated_lat', 0.20), ('chong', 0.20)], 'noadapt', None, False) for i in range(0, 100, 25)]
    jobs += [(acc[0:25], [('early', 0.20), ('late', 0.20)], 'main', None, True)]
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time(); res = []
    with ProcessPoolExecutor(max_workers=2) as ex:
        for k, r in enumerate(ex.map(job, jobs)):
            res.append(r); print(f'job {k + 1}/{len(jobs)} [{time.time() - t0:.0f}s]', flush=True)
    json.dump(res, open('explore_results.json', 'w'), default=float)
    print('done')
