"""
Two canonical continuous-time firing-rate models of rivalry, of the Laing and Chow (2002) form analysed by Shpiro,
Curtu, Rinzel and Rubin (2007) and Shpiro, Moreno-Bote, Rubin and Rinzel (2009), with the same interface as
li2017.simulate so that the third test's schedules, observers and measures can be reused unchanged.

  'sfa'  (spike-frequency adaptation):
      tau   du_i/dt = -u_i + f(I_i + s_i - beta u_j - g a_i + n_i)
      tau_a da_i/dt = -a_i + u_i
  'dep'  (synaptic depression of the cross-inhibition):
      tau   du_i/dt = -u_i + f(I_i + s_i - beta d_j u_j + n_i)
      tau_d dd_i/dt = 1 - d_i - gam d_i u_i

f(x) = 1 / (1 + exp(-(x - theta) / k)), theta = 0.2, k = 0.1, tau = 10 ms. n_i: Ornstein-Uhlenbeck noise on each
channel's input (tau_n = 100 ms, sd sig), as in the port of Li et al.'s model. s_i: the gated increment, on channel 1
only. Forward Euler, dt = 0.5 ms. Parameters are per instance (params_vec): I, beta, g, tau_a (sfa) or gam, tau_d (dep),
and the noise sd via noise_scale.
"""
import numpy as np

THETA, K, TAU, TAU_N, DT = 0.2, 0.1, 10.0, 100.0, 0.5


def f(x):
    return 1.0 / (1.0 + np.exp(-(x - THETA) / K))


def simulate(model, params_vec, N, seed, t_ms, inc=None, schedule=None, record_every=2, noise_scale=None):
    nt = int(round(t_ms / DT)) + 1
    I = np.asarray(params_vec['I'], float)
    beta = np.asarray(params_vec['beta'], float)
    rng = np.random.default_rng(seed)
    u = np.zeros((N, 2)); u[:, 0] = rng.random(N) * 0.2; u[:, 1] = rng.random(N) * 0.2
    if model == 'sfa':
        g, tau_a = np.asarray(params_vec['g'], float), np.asarray(params_vec['tau_a'], float)
        a = np.zeros((N, 2))
    elif model == 'dep':
        gam, tau_d = np.asarray(params_vec['gam'], float), np.asarray(params_vec['tau_d'], float)
        dd = np.ones((N, 2))
    else:
        raise ValueError(model)
    sig = np.ones(N) if noise_scale is None else np.asarray(noise_scale, float)
    nz = np.zeros((N, 2))
    a_ou = np.sqrt(2 * TAU_N)
    inc = np.zeros(N) if inc is None else np.asarray(inc, float)
    nrec = (nt - 1) // record_every + 1
    rb = np.zeros((N, 2, nrec), np.float32); rb[:, :, 0] = u
    on_rec = np.zeros((N, nrec), np.float32)
    state = dict(rb=u)
    ki = 1
    for idx in range(1, nt):
        on = schedule(idx, state) if schedule is not None else np.zeros(N)
        xi = rng.standard_normal((N, 2)) * sig[:, None]
        nz += (DT / TAU_N) * (-nz) + a_ou * np.sqrt(DT) / TAU_N * xi
        drive = np.repeat(I[:, None], 2, axis=1) + nz
        drive[:, 0] += inc * on
        if model == 'sfa':
            x = drive - beta[:, None] * u[:, ::-1] - g[:, None] * a
            a = a + (DT / tau_a[:, None]) * (-a + u)
        else:
            x = drive - beta[:, None] * dd[:, ::-1] * u[:, ::-1]
            dd = dd + (DT / tau_d[:, None]) * (1.0 - dd - gam[:, None] * dd * u)
        u = u + (DT / TAU) * (-u + f(x))
        state['rb'] = u
        if idx % record_every == 0:
            rb[:, :, ki] = u
            on_rec[:, ki] = on
            ki += 1
    return rb[:, :, :ki], on_rec[:, :ki]
