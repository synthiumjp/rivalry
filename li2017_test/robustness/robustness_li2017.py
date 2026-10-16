"""Exploratory robustness re-analysis of the registered Li et al. (2017) test (not registered).
Re-runs baseline and a subset of registered conditions on the same configuration batches and seeds
as the registered run, so the noise is identical and the registered summaries are reproduced
exactly (checked), then computes additional measures per instance:
  - reg:    registered summaries (mA, mB, mBabs) for the reproduction check
  - gapA:   mean interval from the end of a grating-1 episode to the start of the next one, on the
            gate's own threshold. The gate switches at these crossings, so a step in the readout
            when the gate closes cannot move either end of the interval (artefact-immune measure).
  - trans:  median A->B transition time (d falls below +thr until it falls below -thr, direct
            handovers only)
  - mB_cens: competitor mean including the first and last (censored) episodes
  - mB_merge: competitor mean after merging competitor episodes separated only by indeterminate
            spells or sub-50 ms fragments (no grating-1 episode in between)
  - nB, sumB: competitor episode count and total time per instance (for pooled means)
Usage (from li2017_test/robustness/):
  python robustness_li2017.py ../li2017_test_results.json robust_results_pass1.json
  ROBUST_PASS=2 python robustness_li2017.py ../li2017_test_results.json robust_results_pass2.json
"""
import sys, json, time
import numpy as np
sys.path.insert(0, '..')
import run_li2017_test as R

COND = [('gated', 0.10, None), ('gated', 0.20, None), ('ungated', 0.10, None), ('ungated', 0.20, None),
        ('yoked', 0.10, None), ('yoked', 0.20, None), ('offset', 0.20, 0.25)]
import os
if os.environ.get('ROBUST_PASS') == '2':   # second pass: onset-delayed gates (no zero-latency trigger)
    COND = [('onset', 0.20, 0.25), ('onset', 0.20, 0.5), ('onset', 0.20, 1.0)]


def runs(st):
    ch = np.flatnonzero(np.diff(st)) + 1
    s = np.r_[0, ch]; e = np.r_[ch, len(st)]
    return st[s], s, e


def measures(x, thr, thr_abs, dt):
    d = x[0] - x[1]
    st = np.where(d > thr, 1, np.where(d < -thr, -1, 0)).astype(np.int8)
    v, s, e = runs(st)
    L = (e - s) * dt
    # registered-style lists
    eps = [(int(a), l) for a, l in zip(v, L) if a != 0]
    B_all = [l for a, l in eps if a == -1]
    B_cens = [l for l in B_all if l >= R.MIN_EP_MS]                    # includes first/last
    # artefact-immune: intervals of not-A between two A runs
    isA = (st == 1).astype(np.int8)
    va, sa, ea = runs(isA)
    gaps = [(e2 - s2) * dt for a2, s2, e2 in zip(va, sa, ea) if a2 == 0]
    if va[0] == 0: gaps = gaps[1:]
    if va[-1] == 0: gaps = gaps[:-1]
    gaps = [g for g in gaps if g >= R.MIN_EP_MS]
    # merged competitor episodes: B runs merged across 0-spells and sub-50ms fragments without A
    merged, cur, started = [], 0.0, False
    for a, l in zip(v, L):
        if a == 1 and l >= R.MIN_EP_MS:
            if started and cur > 0: merged.append(cur)
            cur, started = 0.0, True
        elif a == -1 and l >= R.MIN_EP_MS:
            cur += l
        elif started and cur > 0:
            cur += l
    # merged[0] starts at a grating-2 run that follows a complete grating-1 run, so it is not truncated
    # A->B transition times
    tr = []
    for k in range(1, len(v) - 1):
        if v[k] == 0 and v[k - 1] == 1 and v[k + 1] == -1:
            tr.append(L[k])
    reg = R.episodes_diff(x, thr, dt)
    A = [l for a, l in reg if a == 1]; B = [l for a, l in reg if a == -1]
    Babs = R.episodes_abs(x[1], thr_abs, dt)
    f = lambda z: float(np.mean(z)) if len(z) else np.nan
    return dict(mA=f(A), mB=f(B), mBabs=f(Babs), gapA=f(gaps), mB_cens=f(B_cens), mB_merge=f(merged),
                trans=float(np.median(tr)) if tr else np.nan, nB=len(B), sumB=float(np.sum(B)))


def chunk(sub):
    base_p = R.L.default_params(1)
    dt = base_p['dt'] * 2
    seeds = list(range(R.N_SEEDS))
    rb0, _ = R.run_batch(sub, seeds, 'none', 0.0, base_p=base_p)
    b = int(R.BURN_MS / dt)
    thr = R.THETA_FRAC * rb0[:, :, b:].mean(axis=(1, 2))
    thr_abs = np.repeat(np.median(rb0[:, 1, b:], axis=1).reshape(len(sub), R.N_SEEDS).mean(1), R.N_SEEDS)
    domA = (rb0[:, 0, :] - rb0[:, 1, :]) > thr[:, None]
    idx = np.roll(np.arange(len(thr)).reshape(len(sub), R.N_SEEDS), -1, axis=1).ravel()
    yoke = domA[idx]
    out = {'base': [measures(rb0[i, :, b:].astype(np.float64), thr[i], thr_abs[i], dt) for i in range(len(thr))]}
    T = []
    for c in range(len(sub)):
        r = out['base'][c * R.N_SEEDS:(c + 1) * R.N_SEEDS]
        T.append(0.5 * (np.nanmean([x['mA'] for x in r]) + np.nanmean([x['mB'] for x in r])))
    Tm = np.repeat(T, R.N_SEEDS)
    del rb0
    for mode, frac, dm in COND:
        key = f'{mode}_{frac:g}' + (f'_d{dm:g}' if dm is not None else '')
        rb, on = R.run_batch(sub, seeds, mode, frac, thr=thr, yoke=yoke if mode == 'yoked' else None,
                             delay_ms=None if dm is None else dm * Tm, base_p=base_p)
        out[key] = [measures(rb[i, :, b:].astype(np.float64), thr[i], thr_abs[i], dt) for i in range(len(thr))]
        del rb
    return out


if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    acc = json.load(open(src))['accepted']
    chunks = [acc[i:i + 25] for i in range(0, len(acc), 25)]
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time(); res = {}
    with ProcessPoolExecutor(max_workers=2) as ex:
        for k, o in enumerate(ex.map(chunk, chunks)):
            for key, v in o.items(): res.setdefault(key, []).extend(v)
            print(f'chunk {k + 1}/{len(chunks)} [{time.time() - t0:.0f}s]', flush=True)
            json.dump(res, open(dst, 'w'), default=float)
    print('done', flush=True)
