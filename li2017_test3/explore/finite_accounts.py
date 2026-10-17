"""EXPLORATORY (not registered). Finite-effect versions of three accounts, each calibrated on the
continuous-delivery condition of the same configuration and compared with the window profile of the
third test (H14 inclusion set), with the second test's windows and delayed-withdrawal curve, and with
the third test's hold curve (out of sample: none of these was used to define or calibrate them).

  finite shift (threshold account): an increment present during [a, b) ends an episode of natural
      duration D at max(a, D - delta) if D >= a and D - delta < b; delta = E[D] - E[D_u].
  finite hazard multiplier: while the increment is on, the switching rate is multiplied by (1 + k);
      k solves E[D] under continuous presence = E[D_u].
  dose (calibrated): change = (E[D_u]/E[D] - 1) x E[time on in [a, b)]/E[D].
D, D_u: pooled merged competitor durations at baseline and under continuous delivery (4,000 values
interpolated from the stored 1% quantiles). Predictions are percentage changes of the mean.
    python finite_accounts.py ../li2017_test3_results.json ../../li2017_test2/li2017_test2_results.json [e|u]
Calibration 'e' (primary): presence throughout the competitor's episode produces -e, e = s - u (merged
scoring), the effect of presence at the end of the episode, as in the registered predictions; the gated
change s is then added for schedules that are also present at its start (delayed withdrawal). Calibration
'u': presence throughout produces the continuous-delivery change u. The first-order handover and
hazard-multiplier accounts (Section 3.3) are included with the same target; for windows they use the
baseline distribution (as registered for H14), and for delayed withdrawal and holds the
continuous-delivery distribution (as registered for H10 and H21).
"""
import sys, json
import numpy as np

R3 = json.load(open(sys.argv[1] if len(sys.argv) > 1 else '../li2017_test3_results.json'))
R2 = json.load(open(sys.argv[2] if len(sys.argv) > 2 else '../../li2017_test2/li2017_test2_results.json'))
CAL = sys.argv[3] if len(sys.argv) > 3 else 'e'   # 'e' (primary) or 'u'
NQ = 4000


def sample(dist):
    return np.sort(np.interp(np.linspace(0, 100, NQ), np.arange(101), dist['q']))


class Accounts:
    def __init__(self, Db, Du, target_pct=None):
        """target_pct: change produced by presence throughout the competitor's episode. None: the
        continuous-delivery change E[D_u]/E[D] - 1 (calibration 'u'); otherwise -e, e = s - u
        (calibration 'e', the presence-at-the-end effect of Section 3.3)."""
        self.Db, self.Du = Db, Du
        self.mb = Db.mean()
        self.mu = Du.mean() if target_pct is None else self.mb * (1 + target_pct / 100)
        self.t_pct = 100 * (self.mu / self.mb - 1)
        lo_, hi_ = 0.0, float(Db.max())                       # delta such that presence throughout gives mu
        for _ in range(60):
            mid = 0.5 * (lo_ + hi_)
            if np.maximum(Db - mid, 0).mean() > self.mu: lo_ = mid
            else: hi_ = mid
        self.delta = 0.5 * (lo_ + hi_)
        self.t = np.linspace(0, Db.max() * 1.0001, 3000)
        self.dt = self.t[1] - self.t[0]
        self.S = 1.0 - np.searchsorted(Db, self.t, side='right') / len(Db)
        lo, hi = 0.0, 200.0
        f = lambda k: np.sum(self.S ** (1 + k)) * self.dt
        if f(hi) > self.mu:
            self.k = np.nan
        else:
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if f(mid) > self.mu: lo = mid
                else: hi = mid
            self.k = 0.5 * (lo + hi)

    def shift(self, a, b):
        D = self.Db; new = D.copy(); hit = (D >= a) & (D - self.delta < b)
        new[hit] = np.maximum(a, D[hit] - self.delta)
        return 100 * (new.mean() - self.mb) / self.mb

    def hazard(self, a, b):
        if not np.isfinite(self.k): return np.nan
        S, t, k = self.S, self.t, self.k
        Sa = np.interp(a, t, S); Sb = np.interp(min(b, t[-1]), t, S)
        with np.errstate(divide='ignore', invalid='ignore'):
            r_in = np.where(Sa > 0, np.clip(S / Sa, 0, 1), 0.0) ** k
            r_out = (Sb / Sa) ** k if Sa > 0 else 1.0
        Sp = np.where(t < a, S, np.where(t < b, S * r_in, S * r_out))
        return 100 * (np.sum(Sp) * self.dt - self.mb) / self.mb

    def first_handover(self, a, b, D=None):
        D = self.Db if D is None else D
        return self.t_pct * (np.mean(D < b) - np.mean(D < a))

    def first_hazard(self, a, b, D=None):
        D = np.sort(self.Db if D is None else D); n = len(D)
        suf = np.cumsum(D[::-1])[::-1]; cnt = n - np.arange(n)
        mres = np.where(cnt > 1, (suf - D * cnt) / np.maximum(cnt - 1, 1), 0.0)
        tot = mres.sum()
        H = lambda x: mres[:np.searchsorted(D, x, side='left')].sum() / tot
        return self.t_pct * (H(b) - H(a))

    def dose(self, a, b):
        ton = np.mean(np.clip(np.minimum(self.Db, b) - a, 0, None))
        return 100 * (self.mu / self.mb - 1) * ton / self.mb


def chg(S, key, m, seeds=range(8)):
    out = []
    for rb, rc in zip(S['base'], S['cond'][key]):
        b = np.array([rb[s][m] for s in seeds], float); c = np.array([rc[s][m] for s in seeds], float)
        ok = np.isfinite(b) & np.isfinite(c)
        out.append(100 * (c[ok].mean() - b[ok].mean()) / b[ok].mean() if ok.any() and b[ok].mean() > 0 else np.nan)
    return np.array(out, float)


def compare(obs, preds, label):
    """obs: (n, k); preds: dict name -> (n, k). Count configurations best described by each account
    (sum of squared errors, beta = 1) and medians."""
    ok = np.all(np.isfinite(obs), 1)
    for p in preds.values(): ok &= np.all(np.isfinite(p), 1)
    sse = {n: ((obs[ok] - p[ok]) ** 2).sum(1) for n, p in preds.items()}
    names = list(preds); best = np.argmin(np.vstack([sse[n] for n in names]), 0)
    return dict(label=label, n=int(ok.sum()), best={n: int(np.sum(best == j)) for j, n in enumerate(names)},
                median_sse={n: float(np.median(v)) for n, v in sse.items()},
                observed_median=np.median(obs[ok], 0).round(2).tolist(),
                predicted_median={n: np.median(p[ok], 0).round(2).tolist() for n, p in preds.items()})


out = {}
# ---------- third test: window profile
M = R3['results']['main']; S3 = R3['settings']
T, dL = np.array(M['T']), np.array(M['dL'])
WIN = [tuple(w) for w in S3['WINDOWS']]
WKEYS = [f'win_0.2_{a:g}-{b:g}' for a, b in WIN] + ['late_0.2']
e3 = chg(M, 'gated_0.2', 'mBm') - chg(M, 'ungated_0.2', 'mBm')
A3 = []
for i in range(len(T)):
    db, du = M['dists']['base_all'][i], M['dists']['ungated_0.2_all'][i]
    ok_ = db and du and (CAL == 'u' or (np.isfinite(e3[i]) and e3[i] > 0))
    A3.append(Accounts(sample(db), sample(du), None if CAL == 'u' else -e3[i]) if ok_ else None)
for m in ('mOB', 'mBm', 'mB', 'mBabs'):
    e = chg(M, 'gated_0.2', m) - chg(M, 'ungated_0.2', m)
    riv = np.ones(len(T), bool)
    for k in WKEYS + ['gated_0.2', 'ungated_0.2']:
        riv &= np.array([np.nanmean([s['nBm'] for s in r]) >= 3 and np.nanmean([s['nOB'] for s in r]) >= 3 for r in M['cond'][k]])
    inc = riv & np.isfinite(dL) & (e > 0) & np.array([a is not None for a in A3])
    obs = np.vstack([chg(M, k, m) for k in WKEYS]).T[inc]
    P = {n: [] for n in ('finite shift', 'finite hazard', 'dose', 'first-order handover', 'first-order hazard')}
    for i in np.flatnonzero(inc):
        A = A3[i]; lims = [(a * T[i], b * T[i]) for a, b in WIN] + [(dL[i], 1e12)]
        P['finite shift'].append([A.shift(a, b) for a, b in lims]); P['finite hazard'].append([A.hazard(a, b) for a, b in lims])
        P['dose'].append([A.dose(a, b) for a, b in lims])
        P['first-order handover'].append([A.first_handover(a, b) for a, b in lims]); P['first-order hazard'].append([A.first_hazard(a, b) for a, b in lims])
    out[f'test3_windows_{m}'] = compare(obs, {n: np.array(v) for n, v in P.items()}, f'third test, five windows, {m}, H14 inclusion set (in rivalry, e > 0)')
# per unit realised dose
out['test3_change_per_unit_dose_mOB'] = {k: float(np.nanmedian(chg(M, k, 'mOB')) / np.nanmean([s['dose_comp'] for r in M['cond'][k] for s in r])) for k in WKEYS}
# ---------- third test: hold curve (observer scoring), change(h) = D0 + W(0, d0 + hT) - W(0, d0)
HOLDS = [0.0] + list(S3['HOLDS']); d0 = 0.5 * sum(S3['LAT_MS']) + 0.5 * S3['RAMP_MS']
keys = ['chong_0.1'] + [f'chong_0.1_x{h:g}' for h in HOLDS[1:]]
e31 = chg(M, 'gated_0.1', 'mBm') - chg(M, 'ungated_0.1', 'mBm')
for m in ('mOB', 'mBm'):
    obs = np.vstack([chg(M, k, m) for k in keys]).T
    D0 = obs[:, 0]
    P = {n: [] for n in ('finite shift', 'finite hazard', 'dose', 'first-order handover', 'first-order hazard')}; keep = []
    for i in range(len(T)):
        db, du = M['dists']['base_all'][i], M['dists']['ungated_0.1_all'][i]
        if not (db and du and np.isfinite(D0[i]) and (CAL == 'u' or (np.isfinite(e31[i]) and e31[i] > 0))): continue
        A = Accounts(sample(db), sample(du), None if CAL == 'u' else -e31[i]); keep.append(i)
        Du_ = sample(du)
        for n, f in (('finite shift', A.shift), ('finite hazard', A.hazard), ('dose', A.dose),
                     ('first-order handover', lambda a_, b_: A.first_handover(a_, b_, Du_)),
                     ('first-order hazard', lambda a_, b_: A.first_hazard(a_, b_, Du_))):
            P[n].append([D0[i] + f(0, d0 + h * T[i]) - f(0, d0) for h in HOLDS])
    out[f'test3_holds_{m}'] = compare(obs[keep][:, 1:], {n: np.array(v)[:, 1:] for n, v in P.items()},
                                     f'third test, hold curve (holds 0.25 to 1.5 T; no-hold value used as anchor), {m}')
# ---------- second test: windows and delayed withdrawal (merged scoring, its primary)
M2 = R2['results']['main']; T2, dL2 = np.array(M2['T']), np.array(M2['dL'])
DEL = list(R2['settings']['DELAYS'])
s2 = chg(M2, 'gated_0.2', 'mBm'); u2 = chg(M2, 'ungated_0.2', 'mBm')
obs_w = np.vstack([chg(M2, k, 'mBm') for k in ('early_0.2', 'late_0.2')]).T
obs_d = np.vstack([chg(M2, f'offset_0.2_x{d:g}', 'mBm') for d in DEL]).T
Pw = {n: [] for n in ('finite shift', 'finite hazard', 'dose', 'first-order handover', 'first-order hazard')}
Pd = {n: [] for n in ('finite shift', 'finite hazard', 'dose', 'first-order handover', 'first-order hazard')}; keep = []
for i in range(len(T2)):
    db, du = M2['dists']['base_all'][i], M2['dists']['ungated_all'][i]
    if not (db and du and np.isfinite(dL2[i]) and np.isfinite(s2[i]) and np.isfinite(u2[i]) and s2[i] - u2[i] > 0): continue
    A = Accounts(sample(db), sample(du), None if CAL == 'u' else -(s2[i] - u2[i])); keep.append(i)
    Du_ = sample(du)
    for n, f in (('finite shift', A.shift), ('finite hazard', A.hazard), ('dose', A.dose),
                 ('first-order handover', A.first_handover), ('first-order hazard', A.first_hazard)):
        Pw[n].append([f(0, 0.25 * T2[i]), f(dL2[i], 1e12)])
    for n, f in (('finite shift', A.shift), ('finite hazard', A.hazard), ('dose', A.dose),
                 ('first-order handover', lambda a_, b_: A.first_handover(a_, b_, Du_)),
                 ('first-order hazard', lambda a_, b_: A.first_hazard(a_, b_, Du_))):
        Pd[n].append([s2[i] + f(0, d * T2[i]) for d in DEL])
out['test2_windows_mBm'] = compare(obs_w[keep], {n: np.array(v) for n, v in Pw.items()}, 'second test, early and late windows, merged scoring')
out['test2_delayed_withdrawal_mBm'] = compare(obs_d[keep], {n: np.array(v) for n, v in Pd.items()}, 'second test, delayed withdrawal (9 delays), merged scoring; gated change s added to each account; first-order accounts use the continuous-delivery distribution, as registered for H10 (under calibration e the first-order handover row is the H10 formula)')
out['calibration'] = dict(test3_median_delta_over_T=float(np.nanmedian([A3[i].delta / T[i] for i in range(len(T)) if A3[i]])),
                          test3_median_k=float(np.nanmedian([A3[i].k for i in range(len(T)) if A3[i]])))
json.dump(out, open(f'finite_accounts_summary_{CAL}.json', 'w'), indent=1)
for k, v in out.items():
    print(k, json.dumps(v)[:900])
