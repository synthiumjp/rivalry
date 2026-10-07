"""
Python port of the attention model of binocular rivalry of
Li H-H, Rankin J, Rinzel J, Carrasco M, Heeger DJ (2017) Attention Model of Binocular Rivalry. PNAS.
Original MATLAB code: archive.nyu.edu/handle/2451/38721, CC BY-SA 3.0. This port is distributed
under the same licence and must be accompanied by the original README.ME.

The port follows n_model.m line by line (forward Euler, same update order, smoothed rectifier),
vectorised over N independent instances (rows). Conditions 1-4 only (static stimuli).
Additions, all off by default: an Ornstein-Uhlenbeck noise term on the input drive (as described
in the paper's Methods, tau_n = 100 ms) and an additive increment on one grating's input, gated by
a caller-supplied schedule.
"""
import numpy as np


def default_params(cond=1):
    p = dict(input=(0.5, 0.5), dt=0.5, T=15000.0, n_m=1, n=2, sigma=0.5, m=2.0, wh=2.0,
             wo=0.65, wa=0.6, tau_s=5.0, tau_o=20.0, tau_a=150.0, tau_h=2000.0,
             sigma_a=0.2, alpha_t=3.0, alphaAmp=0.5, cond=cond)
    if cond in (2, 4):
        p['wa'] = 0.0
    return p


def h_smooth(x):
    # halfExp_smooth: y = x / (1 + exp(-slope*(x - thresh))) for x > 0, else 0
    y = np.zeros_like(x)
    pos = x > 0
    y[pos] = x[pos] / (1.0 + np.exp(-30.0 * (x[pos] - 0.05)))
    return y


def makealpha(dt, T, tau, bound=10e-4):
    t = np.arange(0, T + dt / 2, dt)
    a = t / tau * np.exp(1 - t / tau)
    keep = ~((t > tau) & (a < bound))
    return a[keep]


def stim_modulator(p, nt):
    mod = np.ones(nt)
    onset = makealpha(p['dt'], p['alpha_t'] * 100, p['alpha_t'])
    k = min(len(onset), nt)
    mod[:k] += onset[:k] * p['alphaAmp']
    return mod


def as_col(v, N):
    v = np.asarray(v, float)
    return np.full(N, float(v)) if v.ndim == 0 else v.astype(float)


def simulate(p, N=1, seed=0, noise_sigma=0.0, tau_n=100.0, inc=None, schedule=None,
             record_every=1, rng_init=None, params_vec=None, noise_scale=None):
    """
    p           : parameter dict (scalars); entries in params_vec (dict of length-N arrays) override per instance
    N           : number of instances
    noise_sigma : OU noise sd added to each monocular input (per eye, per orientation)
    inc         : length-N array, additive increment on grating 1 (left eye, orientation 1)
    schedule    : callable(step, state) -> boolean array (N,) saying whether the increment is on
    Returns binocular-summation responses rb (N, 2, nrec) and attention responses ra.
    """
    pv = {k: as_col(params_vec[k], N) for k in params_vec} if params_vec else {}
    g = lambda k: pv[k] if k in pv else p[k]
    dt, T = p['dt'], p['T']
    nt = int(round(T / dt)) + 1
    in1 = as_col(g('in1') if 'in1' in pv else p['input'][0], N)
    in2 = as_col(g('in2') if 'in2' in pv else p['input'][1], N)
    wh, wo, wa = as_col(g('wh'), N), as_col(g('wo'), N), as_col(g('wa'), N)
    tau_h = as_col(g('tau_h'), N)
    sigma, m, n, n_m = p['sigma'], p['m'], p['n'], p['n_m']
    tau_s, tau_o, tau_a, sigma_a = p['tau_s'], p['tau_o'], p['tau_a'], p['sigma_a']
    mod = stim_modulator(p, nt)
    cond = p['cond']
    if cond in (1, 2):
        OL, OR = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    else:
        OL, OR = np.array([1.0, 1.0]), np.array([0.0, 0.0])

    rng = np.random.default_rng(seed)
    r = [np.zeros((N, 2)) for _ in range(6)]
    hh = [np.zeros((N, 2)) for _ in range(6)]
    o = [np.zeros((N, 2)), np.zeros((N, 2))]
    r[0][:] = (rng.random(N) * 0.2)[:, None]
    r[3][:] = (rng.random(N) * 0.2)[:, None]
    nz = np.zeros((N, 2, 2))
    a_ou = np.sqrt(2 * tau_n) if noise_sigma > 0 else 0.0
    inc = np.zeros(N) if inc is None else as_col(inc, N)
    nrec = (nt - 1) // record_every + 1
    rb = np.zeros((N, 2, nrec), np.float32)
    rb[:, :, 0] = r[2]
    on_rec = np.zeros((N, nrec), bool)
    ki = 1
    state = dict(rb=r[2])
    for idx in range(1, nt):
        base = mod[idx]
        iL = np.outer(in1, OL) * base
        iR = np.outer(in2, OR) * base
        on = schedule(idx, state) if schedule is not None else np.zeros(N, bool)
        iL[:, 0] += np.where(on, inc, 0.0)
        if noise_sigma > 0:
            xi = rng.standard_normal((N, 2, 2))
            if noise_scale is not None:
                xi = xi * np.asarray(noise_scale, float)[:, None, None]
            nz += (dt / tau_n) * (-nz) + noise_sigma * a_ou * np.sqrt(dt) / tau_n * xi
            iL = iL + nz[:, 0, :]
            iR = iR + nz[:, 1, :]
        # monocular drives (both computed from previous-step o and attention)
        gain = h_smooth(1.0 + r[5] * wa[:, None])
        dL = h_smooth(iL ** n_m - o[0] * wo[:, None]) * gain
        dR = h_smooth(iR ** n_m - o[1] * wo[:, None]) * gain
        S = (dL.sum(1) + dR.sum(1))[:, None]
        newr = []
        for lay, d in ((0, dL), (1, dR)):
            f = m * d / (S + sigma ** n_m + hh[lay] ** n_m)
            rn = r[lay] + (dt / tau_s) * (-r[lay] + f)
            hh[lay] = hh[lay] + (dt / tau_h[:, None]) * (-hh[lay] + r[lay] * wh[:, None])
            newr.append(rn)
        r_prev0, r_prev1 = r[0], r[1]
        r[0], r[1] = newr
        # binocular summation and opponency use previous-step monocular responses
        d3 = (r_prev0 + r_prev1) ** n
        d4 = h_smooth(r_prev0 - r_prev1) ** n
        d5 = h_smooth(r_prev1 - r_prev0) ** n
        f3 = d3 / (d3 + sigma ** n + hh[2] ** n)
        r[2] = r[2] + (dt / tau_s) * (-r[2] + f3)
        hh[2] = hh[2] + (dt / tau_h[:, None]) * (-hh[2] + r[2] * wh[:, None])
        f4 = d4 / (d4.sum(1, keepdims=True) + sigma ** n)
        r[3] = r[3] + (dt / tau_o) * (-r[3] + f4)
        o[1] = np.repeat(r[3].sum(1, keepdims=True), 2, axis=1)
        f5 = d5 / (d5.sum(1, keepdims=True) + sigma ** n)
        r[4] = r[4] + (dt / tau_o) * (-r[4] + f5)
        o[0] = np.repeat(r[4].sum(1, keepdims=True), 2, axis=1)
        # attention
        inp = r[2]
        ak = np.stack([inp[:, 0] - inp[:, 1], inp[:, 1] - inp[:, 0]], 1)
        aDrive, aSign = np.abs(ak), np.sign(ak)
        d6 = aSign * aDrive ** n
        s6 = (aDrive ** n).sum(1, keepdims=True) + sigma_a ** n
        r[5] = r[5] + (dt / tau_a) * (-r[5] + d6 / s6)
        state['rb'] = r[2]
        if idx % record_every == 0:
            rb[:, :, ki] = r[2]
            on_rec[:, ki] = on
            ki += 1
    return rb[:, :, :ki], on_rec[:, :ki]
