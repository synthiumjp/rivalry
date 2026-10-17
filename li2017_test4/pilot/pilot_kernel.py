"""Pilot (exploratory, not registered): does a linear-response kernel, estimated from the intrinsic
input noise of baseline runs (no increment), predict the effects of increments in Li et al.'s model?
For each of the first N_CFG configurations of the third test's main sample:
  - simulate SEEDS x T_MS of baseline with the attended-input noise recorded (the noise enters exactly
    where the increment does: eye L, orientation 1);
  - Regression A (aligned to merged competitor-episode start, causal): competitor duration on the mean
    attended-input noise in bins of BIN*T over [0, HA*T)  -> window and late-window predictions;
  - Regression B (aligned to the start of the preceding attended episode, causal): the following
    competitor duration on bin means of noise x 1[attended dominant] and noise x 1[not], over [0, HB*T)
    -> gated and continuous predictions.
Predictions (percent of the baseline mean merged competitor duration) are written per configuration."""
import sys, json, time, numpy as np
sys.path.insert(0, '..')
import li2017 as L, sim_noise as S
from concurrent.futures import ProcessPoolExecutor

RES = '../../li2017_test3/li2017_test3_results.json'
N_CFG, SEEDS, T_MS, BURN = int(sys.argv[1]) if len(sys.argv) > 1 else 10, 192, 120000.0, 5000
BIN, HA, HB = 0.125, 3.0, 4.0
MIN_EP = 50


def merged_eps(st):
    """Merged competitor episodes (start, end) in samples, as in the second and third tests' merged scoring:
    start at the first competitor run >= 50 ms after a qualifying attended run (>= 50 ms); end at the start
    of the next qualifying attended run. Also returns the starts of qualifying attended runs."""
    ch = np.flatnonzero(np.diff(st)) + 1
    s0 = np.r_[0, ch]; e0 = np.r_[ch, len(st)]; v = st[s0]; Ls = e0 - s0
    eps, att, started, cur0 = [], [], False, None
    for a, l, ss in zip(v, Ls, s0):
        if a == 1 and l >= MIN_EP:
            if started and cur0 is not None:
                eps.append((cur0, ss))
            att.append(ss); started, cur0 = True, None
        elif a == -1 and l >= MIN_EP and started and cur0 is None:
            cur0 = ss
    return eps, att


def one(c, k):
    p = L.default_params(1); p['T'] = T_MS
    N = SEEDS
    pv = dict(in1=np.full(N, c['inp']), in2=np.full(N, c['inp']), wo=np.full(N, c['wo']), wa=np.full(N, c['wa']),
              wh=np.full(N, c['wh']), tau_h=np.full(N, c['tau_h']))
    rb, z = S.simulate_rec(p, N, 900 + k, pv, np.full(N, c['noise']))
    rb, z = rb[:, :, BURN:].astype(np.float64), z[:, BURN:].astype(np.float64)
    thr = 0.05 * rb.mean(axis=(1, 2))
    XA, yA, XB, yB, FB, Ds = [], [], [], [], [], []
    Tm = None
    # first pass: mean duration -> T (mean of attended and competitor merged-ish durations) per config
    allD, allA = [], []
    store = []
    for i in range(N):
        d = rb[i, 0] - rb[i, 1]
        st = np.where(d > thr[i], 1, np.where(d < -thr[i], -1, 0)).astype(np.int8)
        eps, att = merged_eps(st)
        store.append((st, eps, att))
        allD += [e - s for s, e in eps]
        a_sorted = np.array(att)
        for s, e in eps:
            j = np.searchsorted(a_sorted, s) - 1
            if j >= 0: allA.append(s - a_sorted[j])
    T = 0.5 * (np.mean(allD) + np.mean(allA))
    w = BIN * T
    nA, nB = int(round(HA / BIN)), int(round(HB / BIN))
    for i in range(N):
        st, eps, att = store[i]; zi = z[i]; n = len(zi)
        cz = np.r_[0.0, np.cumsum(zi)]
        att_m = (st == 1).astype(float)
        cza = np.r_[0.0, np.cumsum(zi * att_m)]; czo = np.r_[0.0, np.cumsum(zi * (1 - att_m))]
        cfa = np.r_[0.0, np.cumsum(att_m)]
        a_sorted = np.array(att)
        for s, e in eps:
            edges = (s + np.arange(nA + 1) * w).astype(int)
            if edges[-1] < n:
                XA.append((cz[edges[1:]] - cz[edges[:-1]]) / np.diff(edges)); yA.append(e - s)
            j = np.searchsorted(a_sorted, s) - 1
            if j >= 0:
                a0 = a_sorted[j]
                edges = (a0 + np.arange(nB + 1) * w).astype(int)
                if edges[-1] < n:
                    dw = np.diff(edges)
                    XB.append(np.r_[(cza[edges[1:]] - cza[edges[:-1]]) / dw, (czo[edges[1:]] - czo[edges[:-1]]) / dw])
                    FB.append((cfa[edges[1:]] - cfa[edges[:-1]]) / dw); yB.append(e - s)
    XA, yA, XB, yB, FB = map(np.asarray, (XA, yA, XB, yB, FB))
    def ols(X, y):
        Xc = np.c_[np.ones(len(X)), X]
        b, *_ = np.linalg.lstsq(Xc, y, rcond=None)
        res = y - Xc @ b
        cov = np.linalg.inv(Xc.T @ Xc) * res.var(ddof=Xc.shape[1])
        return b[1:], np.sqrt(np.diag(cov))[1:]
    bA, seA = ols(XA, yA); bB, seB = ols(XB, yB)
    return dict(k=k, T=T, meanD=float(np.mean(allD)), nA=len(yA), nB=len(yB), bA=bA.tolist(), seA=seA.tolist(),
                bB=bB.tolist(), seB=seB.tolist(), fatt=FB.mean(0).tolist(), sd_noise=float(z.std()))


if __name__ == '__main__':
    D = json.load(open(RES))
    cfgs = D['accepted'][:N_CFG]
    t0 = time.time(); out = []
    with ProcessPoolExecutor(max_workers=2) as ex:
        for r in ex.map(one, cfgs, range(N_CFG)):
            out.append(r); print(f"cfg {r['k']}: T={r['T']:.0f} ms, nA={r['nA']}, nB={r['nB']} [{time.time()-t0:.0f}s]", flush=True)
            json.dump(dict(settings=dict(N_CFG=N_CFG, SEEDS=SEEDS, T_MS=T_MS, BIN=BIN, HA=HA, HB=HB), cfgs=cfgs, out=out),
                      open('pilot_kernel.json', 'w'))
    print('done', time.time() - t0)
