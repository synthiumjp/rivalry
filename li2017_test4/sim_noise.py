"""li2017.simulate with the attended-input noise (eye L, orientation 1, where the increment enters) recorded.
Dynamics identical to li2017.simulate with schedule=None, inc=0."""
import numpy as np
import li2017 as L


def simulate_rec(p, N, seed, params_vec, noise_scale, record_every=2):
    pv = {k: L.as_col(params_vec[k], N) for k in params_vec}
    g = lambda k: pv[k] if k in pv else p[k]
    dt, T = p['dt'], p['T']
    nt = int(round(T / dt)) + 1
    in1, in2 = L.as_col(pv['in1'], N), L.as_col(pv['in2'], N)
    wh, wo, wa = L.as_col(g('wh'), N), L.as_col(g('wo'), N), L.as_col(g('wa'), N)
    tau_h = L.as_col(g('tau_h'), N)
    sigma, m, n, n_m = p['sigma'], p['m'], p['n'], p['n_m']
    tau_s, tau_o, tau_a, sigma_a = p['tau_s'], p['tau_o'], p['tau_a'], p['sigma_a']
    mod = L.stim_modulator(p, nt)
    OL, OR = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    rng = np.random.default_rng(seed)
    r = [np.zeros((N, 2)) for _ in range(6)]
    hh = [np.zeros((N, 2)) for _ in range(6)]
    o = [np.zeros((N, 2)), np.zeros((N, 2))]
    r[0][:] = (rng.random(N) * 0.2)[:, None]
    r[3][:] = (rng.random(N) * 0.2)[:, None]
    nz = np.zeros((N, 2, 2))
    tau_n, noise_sigma = 100.0, 1.0
    a_ou = np.sqrt(2 * tau_n)
    nrec = (nt - 1) // record_every + 1
    rb = np.zeros((N, 2, nrec), np.float32)
    zr = np.zeros((N, nrec), np.float32)
    ki = 1
    ns = np.asarray(noise_scale, float)[:, None, None]
    for idx in range(1, nt):
        base = mod[idx]
        iL = np.outer(in1, OL) * base
        iR = np.outer(in2, OR) * base
        xi = rng.standard_normal((N, 2, 2)) * ns
        nz += (dt / tau_n) * (-nz) + noise_sigma * a_ou * np.sqrt(dt) / tau_n * xi
        iL = iL + nz[:, 0, :]
        iR = iR + nz[:, 1, :]
        gain = L.h_smooth(1.0 + r[5] * wa[:, None])
        dL = L.h_smooth(iL ** n_m - o[0] * wo[:, None]) * gain
        dR = L.h_smooth(iR ** n_m - o[1] * wo[:, None]) * gain
        S = (dL.sum(1) + dR.sum(1))[:, None]
        newr = []
        for lay, d in ((0, dL), (1, dR)):
            f = m * d / (S + sigma ** n_m + hh[lay] ** n_m)
            rn = r[lay] + (dt / tau_s) * (-r[lay] + f)
            hh[lay] = hh[lay] + (dt / tau_h[:, None]) * (-hh[lay] + r[lay] * wh[:, None])
            newr.append(rn)
        r_prev0, r_prev1 = r[0], r[1]
        r[0], r[1] = newr
        d3 = (r_prev0 + r_prev1) ** n
        d4 = L.h_smooth(r_prev0 - r_prev1) ** n
        d5 = L.h_smooth(r_prev1 - r_prev0) ** n
        f3 = d3 / (d3 + sigma ** n + hh[2] ** n)
        r[2] = r[2] + (dt / tau_s) * (-r[2] + f3)
        hh[2] = hh[2] + (dt / tau_h[:, None]) * (-hh[2] + r[2] * wh[:, None])
        f4 = d4 / (d4.sum(1, keepdims=True) + sigma ** n)
        r[3] = r[3] + (dt / tau_o) * (-r[3] + f4)
        o[1] = np.repeat(r[3].sum(1, keepdims=True), 2, axis=1)
        f5 = d5 / (d5.sum(1, keepdims=True) + sigma ** n)
        r[4] = r[4] + (dt / tau_o) * (-r[4] + f5)
        o[0] = np.repeat(r[4].sum(1, keepdims=True), 2, axis=1)
        inp = r[2]
        ak = np.stack([inp[:, 0] - inp[:, 1], inp[:, 1] - inp[:, 0]], 1)
        aDrive, aSign = np.abs(ak), np.sign(ak)
        d6 = aSign * aDrive ** n
        s6 = (aDrive ** n).sum(1, keepdims=True) + sigma_a ** n
        r[5] = r[5] + (dt / tau_a) * (-r[5] + d6 / s6)
        if idx % record_every == 0:
            rb[:, :, ki] = r[2]
            zr[:, ki] = nz[:, 0, 0]
            ki += 1
    return rb[:, :, :ki], zr[:, :ki]
