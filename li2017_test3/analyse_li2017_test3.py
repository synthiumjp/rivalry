"""Registered analysis for the third test (prereg_li2017_test3.md). Reads li2017_test3_results.json.
    python analyse_li2017_test3.py [results.json] [tests.json]"""
import sys, json, math
import numpy as np
from scipy import stats

src = sys.argv[1] if len(sys.argv) > 1 else 'li2017_test3_results.json'
dst = sys.argv[2] if len(sys.argv) > 2 else 'li2017_test3_tests.json'
D = json.load(open(src))
S_ = D['settings']
EARLY_W = S_['EARLY_W']
WINDOWS = [tuple(w) for w in S_['WINDOWS']]
HOLDS = [0.0] + list(S_['HOLDS'])
D_OFF_MS = 0.5 * (S_['LAT_MS'][0] + S_['LAT_MS'][1]) + 0.5 * S_['RAMP_MS']   # 585 ms: mean latency + half ramp
ALL, H0, H1 = list(range(8)), [0, 1, 2, 3], [4, 5, 6, 7]
MIN_SEP = math.log(1.25)
MIN_COS_SEP = 0.98
PRIMARY = 'mOB'
SCORINGS = ('mOB', 'mOB2', 'mBm', 'mB', 'mBabs')
M, NA, HN = D['results']['main'], D['results']['noadapt'], D['results']['highnoise']
GRID_MAX = S_['GRID_MAX']
GRID = np.linspace(0, GRID_MAX, int(round(GRID_MAX * 100)) + 1)
W1 = 'win_0.2_0-0.25'
WKEYS = [f'win_0.2_{a:g}-{b:g}' for a, b in WINDOWS] + ['late_0.2']


def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'), float('nan'))
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def agg(rows, m, seeds):
    with np.errstate(all='ignore'):
        return np.array([np.nanmean([r[s][m] for s in seeds]) if np.isfinite([r[s][m] for s in seeds]).any() else np.nan
                         for r in rows], float)


def change(S, key, m=PRIMARY, seeds=ALL):
    """per configuration: mean over seeds where both baseline and condition are finite (paired)"""
    out_ = []
    for rb, rc in zip(S['base'], S['cond'][key]):
        b = np.array([rb[s][m] for s in seeds], float); c = np.array([rc[s][m] for s in seeds], float)
        ok = np.isfinite(b) & np.isfinite(c)
        if not ok.any(): out_.append(np.nan); continue
        bm, cm = b[ok].mean(), c[ok].mean()
        out_.append(100 * (cm - bm) / bm if bm > 0 else np.nan)
    T_ = np.array(S['T'], float)
    return np.where(np.isfinite(T_), np.array(out_, float), np.nan)


def in_rivalry(S, key):
    """at least 3 merged and 3 observer-reported competitor episodes per seed on average"""
    return (agg(S['cond'][key], 'nBm', ALL) >= 3) & (agg(S['cond'][key], 'nOB', ALL) >= 3)


def prop(x, mask=None):
    x = np.asarray(x, float)
    if mask is not None: x = np.where(mask, x, np.nan)
    x = x[np.isfinite(x)]; k, n = int((x > 0).sum()), len(x)
    return dict(k=k, n=n, ci=wilson(k, n), median=float(np.median(x)) if n else float('nan'))


def g_at(gl, x_T):
    return float(np.interp(x_T, GRID, gl)) if gl is not None else np.nan


def F_at(ql, x_ms):
    if ql is None: return np.nan
    return float(np.interp(x_ms, np.asarray(ql), np.linspace(0, 1, 101), left=0.0, right=1.0))


def inv_grid(curve, p):
    """smallest x (units of T) on the grid with curve(x) = p, linear interpolation; nan if never reached"""
    c = np.asarray(curve, float); k = np.flatnonzero(c >= p)
    if not len(k): return np.nan
    k = k[0]
    if k == 0: return 0.0
    return float(np.interp(p, c[k - 1:k + 1], GRID[k - 1:k + 1]))


def pava_decreasing(y):
    y = list(map(float, y)); blocks = [[v, 1] for v in y]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:
            v = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [v, blocks[i][1] + blocks[i + 1][1]]; del blocks[i + 1]; i = max(i - 1, 0)
        else:
            i += 1
    out_ = []
    for v, w in blocks: out_ += [v] * w
    return np.array(out_)


def crossing(xs, ys):
    if not np.all(np.isfinite(ys)) or ys[0] <= 0: return np.nan
    j = np.flatnonzero(ys <= 0)
    if not len(j): return np.inf
    j = j[0]; x0, x1, y0, y1 = xs[j - 1], xs[j], ys[j - 1], ys[j]
    return x0 + (x1 - x0) * y0 / (y0 - y1) if y0 != y1 else x1


out = {'excluded': {}, 'settings': dict(primary_scoring=PRIMARY, D_OFF_MS=D_OFF_MS)}
T = np.array(M['T'], float); dL = np.array(M['dL'], float)
T_na = np.array(NA['T'], float); dL_na = np.array(NA['dL'], float)

# ---- H13: early minus late window (observer scoring primary)
riv_w = in_rivalry(M, W1) & in_rivalry(M, 'late_0.2') & np.isfinite(dL)
out['excluded']['H13_out_of_rivalry_or_no_dL'] = int((~riv_w).sum())
d_main = change(M, W1) - change(M, 'late_0.2')
for m in SCORINGS:
    out[f'H13_{m}'] = prop(change(M, W1, m) - change(M, 'late_0.2', m), riv_w)
out['H13'] = dict(holds=bool(out[f'H13_{PRIMARY}']['ci'][0] > 0.5))


# ---- H14 (primary): profile of the five windows, handover vs hazard (and vs dose)
def window_weights(dist, Ti, dLi):
    q, gl, hz = dist['q'], dist['g'], dist['hz']
    wh = [F_at(q, b * Ti) - F_at(q, a * Ti) for a, b in WINDOWS] + [1 - F_at(q, dLi)]
    wz = [g_at(gl, b) - g_at(gl, a) for a, b in WINDOWS] + [1 - g_at(gl, dLi / Ti)]
    wr = [g_at(hz, b) - g_at(hz, a) for a, b in WINDOWS] + [1 - g_at(hz, dLi / Ti)]
    return np.array(wh), np.array(wz), np.array(wr)


def h14(m, pred_seeds, obs_seeds, dist_key, scaled, label, idx=(0, 1, 2, 3, 4)):
    """idx: which of the five windows (four fixed, then late) enter the comparison"""
    idx = list(idx)
    e = change(M, 'gated_0.2', m, pred_seeds) - change(M, 'ungated_0.2', m, pred_seeds)
    O = np.vstack([change(M, k, m, obs_seeds) for k in WKEYS]).T
    riv = np.all([in_rivalry(M, k) for k in WKEYS + ['gated_0.2', 'ungated_0.2']], axis=0) & np.isfinite(dL)
    res = {'hazard': [], 'dose': []}; n_ns = {'hazard': 0, 'dose': 0}; n_excl = 0
    beta_h = []
    for i in range(len(T)):
        dist = M['dists'][dist_key][i]
        if not (riv[i] and dist and np.isfinite(e[i]) and e[i] > 0 and np.all(np.isfinite(O[i]))):
            n_excl += 1; continue
        wh, wz, wr = window_weights(dist, T[i], dL[i])
        if not (np.all(np.isfinite(wh)) and np.all(np.isfinite(wz)) and np.all(np.isfinite(wr))):
            n_excl += 1; continue
        wh, wz, wr, o = wh[idx], wz[idx], wr[idx], O[i][idx]

        def ssr(w):
            p = -e[i] * w
            bb = max(0.0, float(o @ p) / float(p @ p)) if scaled and p @ p > 0 else 1.0
            return float(np.sum((o - bb * p) ** 2)), bb
        sh, bh = ssr(wh); beta_h.append(bh)
        for name, w in (('hazard', wr), ('dose', wz)):
            nw, nh = np.linalg.norm(w), np.linalg.norm(wh)
            cos = float(wh @ w / (nh * nw)) if nw > 0 and nh > 0 else 1.0
            if cos >= MIN_COS_SEP:
                n_ns[name] += 1; continue
            res[name].append(sh < ssr(w)[0])        # ties count against the handover account
    r = {'label': label, 'windows': idx, 'n_excluded': n_excl,
         'median_beta_handover': float(np.median(beta_h)) if beta_h else float('nan')}
    for name in ('hazard', 'dose'):
        k, n = int(np.sum(res[name])), len(res[name])
        r[f'vs_{name}'] = dict(k=k, n=n, ci=wilson(k, n), n_not_separated=n_ns[name], holds=bool(n > 0 and wilson(k, n)[0] > 0.5))
    return r


LATE3 = (2, 3, 4)
prim14 = h14(PRIMARY, ALL, ALL, 'base_all', True, 'primary (i): five windows, observer scoring, all seeds, scaled fit')
prim14b = h14(PRIMARY, ALL, ALL, 'base_all', True, 'primary (ii): three late windows only, scaled fit', LATE3)
# H14 holds if the five-window comparison favours the handover account (lower Wilson bound above 0.5) and
# the three late windows, which second-order lengthening early in the episode cannot affect, do not
# contradict it (upper Wilson bound above 0.5).
out['H14'] = dict(five_windows=prim14, late_windows=prim14b,
                  holds=bool(prim14['vs_hazard']['holds'] and prim14b['vs_hazard']['n'] > 0 and prim14b['vs_hazard']['ci'][1] > 0.5),
                  holds_vs_dose=bool(prim14['vs_dose']['holds'] and prim14b['vs_dose']['n'] > 0 and prim14b['vs_dose']['ci'][1] > 0.5))
out['H14_sensitivity'] = [h14(PRIMARY, ALL, ALL, 'base_all', False, 'five windows, unscaled (beta = 1)'),
                          h14(PRIMARY, ALL, ALL, 'base_all', False, 'three late windows, unscaled (beta = 1)', LATE3),
                          h14('mBm', ALL, ALL, 'base_all', True, 'five windows, merged scoring'),
                          h14('mOB2', ALL, ALL, 'base_all', True, 'five windows, passive observer 300-600 ms'),
                          h14(PRIMARY, H0, H1, 'base_h0', True, 'five windows, seed split: weights and e from seeds 0-3, windows from 4-7')]
# window profile, descriptive: observed medians and each account's median prediction
e20 = change(M, 'gated_0.2') - change(M, 'ungated_0.2')
prof = {k: {} for k in WKEYS}
P = {'handover': [], 'dose': [], 'hazard': []}
for i in range(len(T)):
    dist = M['dists']['base_all'][i]
    if dist and np.isfinite(dL[i]) and np.isfinite(e20[i]):
        wh, wz, wr = window_weights(dist, T[i], dL[i])
        P['handover'].append(-e20[i] * wh); P['dose'].append(-e20[i] * wz); P['hazard'].append(-e20[i] * wr)
for j, k in enumerate(WKEYS):
    for m in SCORINGS:
        prof[k][m] = prop(change(M, k, m))
    for a in P:
        prof[k][f'pred_{a}_median'] = float(np.nanmedian([p[j] for p in P[a]])) if P[a] else float('nan')
    prof[k]['dose_comp'] = float(np.nanmean(agg(M['cond'][k], 'dose_comp', ALL)))
out['window_profile'] = prof

# ---- H15: a 100 ms latency gate lengthens the competitor more than the zero-latency gate
riv15 = in_rivalry(M, 'gated_0.2') & in_rivalry(M, 'gated_lat_0.2')
out['excluded']['H15_out_of_rivalry'] = int((~riv15).sum())
for m in SCORINGS:
    out[f'H15_{m}'] = prop(change(M, 'gated_lat_0.2', m) - change(M, 'gated_0.2', m), riv15)
out['H15'] = dict(holds=bool(out[f'H15_{PRIMARY}']['ci'][0] > 0.5))


# ---- H16: without adaptation, the latency gate (a) and the report gate (b) leave the competitor nearly unchanged
def nearly_unchanged(S, key, ref_S, ref_key):
    riv = in_rivalry(S, key)
    g = np.where(riv, change(S, key), np.nan)
    u = np.where(in_rivalry(S, 'ungated_0.2'), change(S, 'ungated_0.2'), np.nan)
    att = np.where(riv, change(S, key, 'mOA'), np.nan)
    ref = np.where(in_rivalry(ref_S, ref_key), change(ref_S, ref_key), np.nan)
    gm_, rm_ = g[np.isfinite(g)], ref[np.isfinite(ref)]
    mw_ = stats.mannwhitneyu(gm_, rm_, alternative='less') if len(gm_) and len(rm_) else None
    r = dict(competitor=prop(g), ungated=prop(u), attended=prop(att), n_out_of_rivalry=int((~riv).sum()),
             median_main=float(np.median(rm_)) if len(rm_) else float('nan'),
             mw_p=float(mw_.pvalue) if mw_ is not None else float('nan'))
    r['holds'] = bool(r['ungated']['ci'][1] < 0.5 and r['attended']['ci'][0] > 0.5 and np.isfinite(r['median_main'])
                      and abs(r['competitor']['median']) < 0.25 * abs(r['median_main']) and r['mw_p'] < 0.05)
    for m in SCORINGS:
        r[f'competitor_{m}'] = prop(np.where(riv, change(S, key, m), np.nan))
    return r


out['H16'] = dict(a_latency_gate=nearly_unchanged(NA, 'gated_lat_0.2', M, 'gated_lat_0.2'),
                  b_report_gate=nearly_unchanged(NA, 'rep_gated_0.2', M, 'rep_gated_0.2'),
                  zero_latency_gate={m: prop(np.where(in_rivalry(NA, 'gated_0.2'), change(NA, 'gated_0.2', m), np.nan)) for m in SCORINGS})
out['H16']['holds'] = out['H16']['a_latency_gate']['holds']
out['H16']['b_holds'] = out['H16']['b_report_gate']['holds']


# ---- H17: without adaptation, early and late windows differ less, relative to e, than with adaptation
def norm_diff(S, m=PRIMARY):
    dL_ = np.array(S['dL'], float)
    riv = in_rivalry(S, W1) & in_rivalry(S, 'late_0.2') & np.isfinite(dL_)
    e = change(S, 'gated_0.2', m) - change(S, 'ungated_0.2', m)
    x = (change(S, W1, m) - change(S, 'late_0.2', m)) / e
    return np.where(riv & np.isfinite(e) & (e > 0), x, np.nan)


def predicted_norm_diff(S):
    T_, dL_ = np.array(S['T'], float), np.array(S['dL'], float)
    P_ = {'handover': [], 'dose': [], 'hazard': []}
    for i in range(len(T_)):
        dist = S['dists']['base_all'][i]
        if not (dist and np.isfinite(dL_[i]) and np.isfinite(T_[i])):
            continue
        q, gl, hz = dist['q'], dist['g'], dist['hz']
        we = (F_at(q, EARLY_W * T_[i]), g_at(gl, EARLY_W), g_at(hz, EARLY_W))
        wl = (1 - F_at(q, dL_[i]), 1 - g_at(gl, dL_[i] / T_[i]), 1 - g_at(hz, dL_[i] / T_[i]))
        for j, a_ in enumerate(('handover', 'dose', 'hazard')):
            P_[a_].append(-(we[j] - wl[j]))
    return {a_: float(np.nanmedian(v)) if v else float('nan') for a_, v in P_.items()}


x_na, x_main = norm_diff(NA), norm_diff(M)
a_, b_ = x_na[np.isfinite(x_na)], x_main[np.isfinite(x_main)]
mw17 = stats.mannwhitneyu(a_, b_, alternative='less') if len(a_) and len(b_) else None
H17 = dict(noadapt=prop(x_na), main=prop(x_main), mw_p=float(mw17.pvalue) if mw17 is not None else float('nan'))
H17['testable'] = bool(H17['main']['median'] > 0)
H17['holds'] = bool(H17['testable'] and abs(H17['noadapt']['median']) < 0.5 * H17['main']['median'] and H17['mw_p'] < 0.05)
for m in SCORINGS:
    H17[f'noadapt_{m}'] = prop(norm_diff(NA, m)); H17[f'main_{m}'] = prop(norm_diff(M, m))
H17['predicted_normalised_difference_median'] = dict(main=predicted_norm_diff(M), noadapt=predicted_norm_diff(NA),
                                                     highnoise=predicted_norm_diff(HN))
H17['highnoise'] = prop(norm_diff(HN))
out['H17'] = H17

# ---- H18: ramped early window
riv18 = in_rivalry(M, 'early_ramp_0.2') & in_rivalry(M, 'late_0.2') & np.isfinite(dL)
for m in SCORINGS:
    out[f'H18a_{m}'] = prop(change(M, 'early_ramp_0.2', m) - change(M, 'late_0.2', m), riv18)
abs_r = prop(change(M, 'early_ramp_0.2', 'mBabs'), riv18); abs_s = prop(change(M, W1, 'mBabs'), riv_w)
rB = prop(change(M, W1, 'rB_dom'), riv_w)
out['H18'] = dict(a_holds=bool(out[f'H18a_{PRIMARY}']['ci'][0] > 0.5),
                  b_abs_ramped=abs_r, b_abs_step=abs_s,
                  b_holds=bool(abs_s['median'] < 0 and abs_r['median'] <= 0.5 * abs_s['median']),
                  c_rB_dom_change_step=rB, c_holds=bool(rB['ci'][1] < 0.5),
                  c_rB_dom_change_ramped=prop(change(M, 'early_ramp_0.2', 'rB_dom'), riv18),
                  c_rB_dom_change_late=prop(change(M, 'late_0.2', 'rB_dom'), riv_w),
                  fragments_abs_per_episode={k: float(np.nanmedian(agg(M['cond'][k], 'nBabs', ALL) / agg(M['base'], 'nBabs', ALL)))
                                             for k in (W1, 'early_ramp_0.2', 'late_0.2')})
out['H18']['holds'] = bool(out['H18']['a_holds'] and out['H18']['b_holds'] and out['H18']['c_holds'])

# ---- H19 (primary): Chong et al.'s timing at the calibrated increment lengthens the competitor
riv_c = in_rivalry(M, 'chong_0.1')
out['excluded']['H19_out_of_rivalry'] = int((~riv_c).sum())
for m in SCORINGS:
    out[f'H19_{m}'] = prop(np.where(riv_c, change(M, 'chong_0.1', m), -1.0))
out['H19_attended_observer'] = prop(change(M, 'chong_0.1', 'mOA'), riv_c)
out['H19_attended_first_test'] = prop(change(M, 'chong_0.1', 'mA'), riv_c)
out['H19'] = dict(holds=bool(out[f'H19_{PRIMARY}']['ci'][0] > 0.5),
                  positive_control_attended=bool(out['H19_attended_observer']['ci'][0] > 0.5))

# ---- H20: holding the increment 1.5 T after the report of the competitor shortens the competitor
riv20 = in_rivalry(M, 'chong_0.1_x1.5')
for m in SCORINGS:
    out[f'H20_{m}'] = prop(np.where(riv20, change(M, 'chong_0.1_x1.5', m), np.nan))
out['H20'] = dict(holds=bool(out[f'H20_{PRIMARY}']['median'] < 0 and out[f'H20_{PRIMARY}']['ci'][1] < 0.5))
out['hold_curve_median'] = {f'{h:g}': {m: prop(change(M, 'chong_0.1' if h == 0 else f'chong_0.1_x{h:g}', m))['median'] for m in SCORINGS}
                            for h in HOLDS}


# ---- H21 (primary): where the hold reverses the effect, handover vs dose (and vs hazard)
def h21(dist_key, pred_seeds, curve_seeds, isotonic, label, m=PRIMARY, empirical_d0=False):
    keys = ['chong_0.1'] + [f'chong_0.1_x{h:g}' for h in HOLDS[1:]]
    riv = np.all([in_rivalry(M, k) for k in keys + ['gated_0.1', 'ungated_0.1']], axis=0)
    d0 = change(M, 'chong_0.1', m, pred_seeds)
    e = change(M, 'gated_0.1', m, pred_seeds) - change(M, 'ungated_0.1', m, pred_seeds)
    curve = np.vstack([change(M, k, m, curve_seeds) for k in keys]).T
    areas = np.vstack([agg(M['cond'][k], 'area_comp', curve_seeds) for k in keys]).T     # realised withdrawal (ms)
    a0 = agg(M['cond']['chong_0.1'], 'area_comp', pred_seeds)
    dp, dz, dr, do = [], [], [], []
    n_nan_curve = 0
    for i in range(len(T)):
        dist = M['dists'][dist_key][i]
        off_ms = a0[i] if empirical_d0 else D_OFF_MS
        off = off_ms / T[i]
        if not (riv[i] and dist and np.isfinite(d0[i]) and np.isfinite(e[i]) and d0[i] > 0 and e[i] > 0):
            dp.append(np.nan); dz.append(np.nan); dr.append(np.nan); do.append(np.nan); continue
        ph = F_at(dist['q'], off_ms) + d0[i] / e[i]
        x_ = float(np.interp(ph * 100, np.arange(101), dist['q'])) / T[i] if ph < 1 else np.nan
        dp.append(x_ if np.isfinite(x_) and x_ <= GRID_MAX else np.nan)
        dz.append(inv_grid(dist['g'], g_at(dist['g'], off) + d0[i] / e[i]))
        dr.append(inv_grid(dist['hz'], g_at(dist['hz'], off) + d0[i] / e[i]))
        if not np.all(np.isfinite(curve[i])):
            n_nan_curve += 1; dp[-1] = dz[-1] = dr[-1] = np.nan; do.append(np.nan); continue
        y = pava_decreasing(curve[i]) if isotonic else curve[i]
        if empirical_d0:
            xs = areas[i] / T[i]
            if not (np.all(np.isfinite(xs)) and np.all(np.diff(xs) > 0)):
                n_nan_curve += 1; dp[-1] = dz[-1] = dr[-1] = np.nan; do.append(np.nan); continue
            c = crossing(xs, np.asarray(y)); do.append(c)
        else:
            c = crossing(np.array(HOLDS), np.asarray(y)); do.append(c + off if np.isfinite(c) else c)
    dp, dz, dr, do = map(np.array, (dp, dz, dr, do))       # effective delays, units of T
    dmax = (areas[:, -1] / T) if empirical_d0 else (HOLDS[-1] + D_OFF_MS / T)
    res = {'label': label}
    for name, alt in (('dose', dz), ('hazard', dr)):
        base_ok = np.isfinite(dp) & np.isfinite(alt) & (dp > 0) & (alt > 0) & ~np.isnan(do)
        sep = base_ok & (np.abs(np.log(dp / alt)) > MIN_SEP)
        closer, n_amb = [], 0
        for i in np.flatnonzero(sep):
            if np.isfinite(do[i]):
                closer.append(abs(math.log(do[i] / dp[i])) < abs(math.log(do[i] / alt[i])))
            else:
                gm_ = math.sqrt(dp[i] * alt[i])
                if gm_ <= dmax[i]: closer.append(dp[i] > alt[i])
                else: n_amb += 1
        k, n = int(np.sum(closer)), len(closer)
        res[f'vs_{name}'] = dict(k=k, n=n, ci=wilson(k, n), holds=bool(n > 0 and wilson(k, n)[0] > 0.5),
                                 n_not_separated=int(np.sum(base_ok & ~sep)), n_censored_ambiguous=n_amb)
    fin = np.isfinite(dp) & np.isfinite(dz) & np.isfinite(dr) & np.isfinite(do) & (do > 0)
    res.update(n_curve_not_finite=n_nan_curve, n_no_prediction=int(np.sum(~(np.isfinite(dp) & np.isfinite(dz) & np.isfinite(dr)))),
               n_excluded_hold0_not_positive=int(np.sum(np.isfinite(dp) & np.isnan(do))),
               n_censored=int(np.sum(np.isfinite(dp) & np.isinf(do))),
               median_obs=float(np.median(do[fin])) if fin.any() else float('nan'),
               median_pred_handover=float(np.nanmedian(dp)), median_pred_dose=float(np.nanmedian(dz)),
               median_pred_hazard=float(np.nanmedian(dr)), n_finite=int(fin.sum()))
    if fin.sum() >= 10:
        res['spearman'] = {a: float(stats.spearmanr(do[fin], v[fin])[0]) for a, v in (('handover', dp), ('dose', dz), ('hazard', dr))}
        res['within_25pct'] = {a: float(np.mean(np.abs(np.log(do[fin] / v[fin])) < MIN_SEP)) for a, v in (('handover', dp), ('dose', dz), ('hazard', dr))}
    res['per_config'] = dict(dpred=dp.tolist(), ddose=dz.tolist(), dhaz=dr.tolist(),
                             dobs=[None if not np.isfinite(x) else float(x) for x in do])
    return res


prim21 = h21('ungated_0.1_h0', H0, H1, True, 'primary: continuous-delivery distribution, seeds 0-3 / 4-7, isotonic')
out['H21'] = prim21
out['H21']['holds'] = prim21['vs_dose']['holds']             # primary comparison: handover vs dose
out['H21_sensitivity'] = [{k: v for k, v in h21('base_h0', H0, H1, True, 'baseline distribution').items() if k != 'per_config'},
                          {k: v for k, v in h21('ungated_0.1_all', ALL, ALL, True, 'all seeds, no split').items() if k != 'per_config'},
                          {k: v for k, v in h21('ungated_0.1_h0', H0, H1, False, 'raw crossing').items() if k != 'per_config'},
                          {k: v for k, v in h21('ungated_0.1_h0', H0, H1, True, 'merged scoring', m='mBm').items() if k != 'per_config'},
                          {k: v for k, v in h21('ungated_0.1_h0', H0, H1, True, 'passive observer 300-600 ms', m='mOB2').items() if k != 'per_config'},
                          {k: v for k, v in h21('ungated_0.1_h0', H0, H1, True, 'empirical withdrawal time', empirical_d0=True).items() if k != 'per_config'}]

# ---- H22: report-contingent windows at the calibrated increment
riv22 = in_rivalry(M, 'early_rep_0.1') & in_rivalry(M, 'late_rep_0.1') & np.isfinite(dL)
for m in SCORINGS:
    out[f'H22_{m}'] = prop(change(M, 'early_rep_0.1', m) - change(M, 'late_rep_0.1', m), riv22)
out['H22'] = dict(holds=bool(out[f'H22_{PRIMARY}']['ci'][0] > 0.5))

# ---- H23: without adaptation, Chong et al.'s timing does not lengthen the competitor
riv23 = in_rivalry(NA, 'chong_0.1')
out['excluded']['H23_out_of_rivalry'] = int((~riv23).sum())
for m in SCORINGS:
    out[f'H23_{m}'] = prop(np.where(riv23, change(NA, 'chong_0.1', m), np.nan))
out['H23_attended_observer'] = prop(change(NA, 'chong_0.1', 'mOA'), riv23)
out['H23'] = dict(holds=bool(out[f'H23_{PRIMARY}']['ci'][1] < 0.5))

# ---- H24 (noise control): with adaptation and the no-adaptation sample's noise, the latency gate (a)
#      and Chong et al.'s timing (b) lengthen the competitor (out of rivalry counted as not lengthening)
H24 = {}
for nm_, key in (('a_latency_gate', 'gated_lat_0.2'), ('b_chong', 'chong_0.1')):
    riv = in_rivalry(HN, key)
    H24[nm_] = {m: prop(np.where(riv, change(HN, key, m), -1.0)) for m in SCORINGS}
    H24[nm_]['n_out_of_rivalry'] = int((~riv).sum())
    H24[nm_]['holds'] = bool(H24[nm_][PRIMARY]['ci'][0] > 0.5)
H24['holds'] = bool(H24['a_latency_gate']['holds'] and H24['b_chong']['holds'])
out['H24'] = H24

# ---- descriptive: every condition, four scorings, attended, doses, fragments, rivalry loss
desc = {}
for S, nm in ((M, 'main'), (NA, 'noadapt'), (HN, 'highnoise')):
    for key in S['cond']:
        r = {m: prop(change(S, key, m)) for m in SCORINGS}
        r['attended_observer'] = prop(change(S, key, 'mOA')); r['attended_first_test'] = prop(change(S, key, 'mA'))
        r['rB_dom'] = prop(change(S, key, 'rB_dom'))
        for k in ('dose_comp', 'dose_att'):
            r[k] = float(np.nanmean(agg(S['cond'][key], k, ALL)))
        r['fragments_per_merged'] = float(np.nanmedian(agg(S['cond'][key], 'nB', ALL) / agg(S['cond'][key], 'nBm', ALL)))
        r['out_of_rivalry'] = int((~in_rivalry(S, key)).sum())
        r['realised_withdrawal_ms'] = float(np.nanmedian(agg(S['cond'][key], 'area_comp', ALL)))
        desc[f'{nm}|{key}'] = r
    desc[f'{nm}|base_fragments_per_merged'] = float(np.nanmedian(agg(S['base'], 'nB', ALL) / agg(S['base'], 'nBm', ALL)))
    desc[f'{nm}|base_ratio_merged_over_T'] = float(np.nanmedian(agg(S['base'], 'mBm', ALL) / np.array(S['T'])))
    desc[f'{nm}|base_ratio_observer_over_T'] = float(np.nanmedian(agg(S['base'], 'mOB', ALL) / np.array(S['T'])))
out['descriptive'] = desc
out['sample'] = dict(n_main=len(M['base']), n_drawn=D['n_drawn'], n_na=len(NA['base']), n_drawn_na=D['n_drawn_na'],
                     median_T=float(np.median(T)), median_dL_over_T=float(np.nanmedian(dL / T)),
                     median_T_na=float(np.median(T_na)), n_hn=len(HN['base']), n_drawn_hn=D['n_drawn_hn'],
                     median_T_hn=float(np.median(HN['T'])), dry_run=D['dry_run'], versions=D.get('versions'))
json.dump(out, open(dst, 'w'), indent=1, default=float)
for h in ('H13', 'H14', 'H15', 'H16', 'H17', 'H18', 'H19', 'H20', 'H21', 'H22', 'H23', 'H24'):
    print(h, 'holds' if out[h]['holds'] else 'does not hold')
print('  H14', prim14, '\n  H14 late', prim14b)
print('  H21', {k: v for k, v in prim21.items() if k != 'per_config'})
print('  excluded', out['excluded'])
