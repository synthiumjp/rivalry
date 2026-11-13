"""Pilot for the seventh test (run from this folder: python pilot7.py <model> <n> <seed>): baseline only (zero increment). For draws from widened ranges, record acceptance under the
third test's rule and the baseline lag-1 serial correlation r_ab, to find ranges that yield observer-like serial
correlation (r_ab < 0.1). No condition with an increment is run."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the test's folder
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import run_li2017_test6 as T6, run_li2017_test3 as R3, run_li2017_test as R

RANGES_WIDE = dict(
    li=dict(inp=(0.35, 0.8), wo=(0.5, 0.75), wa=(0.4, 0.9), wh=(0.3, 2.5), tau_h=(1000.0, 4000.0), noise=(0.03, 0.25)),
    sfa=dict(I=(0.4, 1.0), beta=(0.6, 1.4), g=(0.05, 0.8), tau_a=(1000.0, 6000.0), noise=(0.02, 0.40)),
    dep=dict(I=(0.4, 1.0), beta=(0.6, 1.4), gam=(0.2, 4.0), tau_d=(1000.0, 6000.0), noise=(0.02, 0.40)))


RANGES = dict(
    li=dict(inp=(0.35, 0.8), wo=(0.5, 0.75), wa=(0.4, 0.9), wh=(0.3, 1.2), tau_h=(1000.0, 4000.0), noise=(0.04, 0.10)),
    sfa=dict(I=(0.4, 1.0), beta=(0.6, 1.4), g=(0.05, 0.35), tau_a=(1000.0, 6000.0), noise=(0.06, 0.20)),
    dep=dict(I=(0.4, 1.0), beta=(0.9, 1.4), gam=(0.2, 1.0), tau_d=(1000.0, 6000.0), noise=(0.08, 0.18)))


def chunk(model, cfgs, tag, t_ms=120000.0, n_seeds=8):
    seeds = list(range(n_seeds))
    b = int(R3.BURN_MS / R3.DT_REC)
    rb = T6.run_batch6(model, cfgs, seeds, 'none', 0.0, None, tag, t_ms=t_ms, ranges=RANGES[model])[0]
    thr = R3.THETA_FRAC * rb[:, :, b:].mean(axis=(1, 2))
    out = []
    for c in range(len(cfgs)):
        n_ep, cvs, allD, rs = [], [], [], []
        for i in range(c * n_seeds, (c + 1) * n_seeds):
            x = rb[i, :, b:].astype(np.float64)
            E = R.episodes_diff(x, thr[i], R3.DT_REC)
            dd = [l for _, l in E]; n_ep.append(len(E)); allD += dd
            cvs.append(np.std(dd) / np.mean(dd) if len(dd) > 2 else np.nan)
            rs.append(T6.serial_r(x, thr[i])['r_ab'])
        rs = np.array(rs, float); rs = rs[np.isfinite(rs)]
        r = float(np.tanh(np.mean(np.arctanh(np.clip(rs, -.999, .999))))) if len(rs) else np.nan
        out.append(dict(cfg=cfgs[c], n_ep=float(np.mean(n_ep)), cv=float(np.nanmedian(cvs)) if np.isfinite(cvs).any() else np.nan,
                        T=float(np.mean(allD)) if allD else np.nan, r_ab=r))
    return out


if __name__ == '__main__':
    model, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rng = np.random.default_rng(seed)
    cfgs = [{k: float(rng.uniform(*v)) for k, v in RANGES[model].items()} for _ in range(n)]
    ch = [cfgs[i:i + 5] for i in range(0, n, 5)]
    t0 = time.time(); res = []
    with ProcessPoolExecutor(max_workers=2) as ex:
        for r in ex.map(chunk, [model] * len(ch), ch, [f'pilot7_{model}_{i}' for i in range(len(ch))]):
            res += r; print(model, len(res), f'{time.time() - t0:.0f}s', flush=True)
    json.dump(res, open(f'pilot7c_{model}.json', 'w'))
