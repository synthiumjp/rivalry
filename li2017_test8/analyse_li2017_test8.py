"""Pre-specified analysis of the eighth test (analysis_plan_li2017_test8.md, Section 5).
    python analyse_li2017_test8.py [li2017_test8_results.json] [li2017_test8_tests.json] [../li2017_test7/li2017_test7_results.json]
Primary scoring is observer scoring (mOA for the attended grating, mOB for the competitor), as in Chong et al.'s
experiment and the third test's calibrated condition, in observer-valid configurations (baseline ratio of reported to
merged competitor episodes at least 0.8). Changes are per configuration: 100 x (mean over seeds with the
increment - mean over seeds without) / mean over seeds without, over seeds with a finite value in both runs."""
import sys, json
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else 'li2017_test8_results.json'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'li2017_test8_tests.json'
REF = sys.argv[3] if len(sys.argv) > 3 else '../li2017_test7/li2017_test7_results.json'
D = json.load(open(SRC))
D7 = json.load(open(REF))
SIZES = (0.025, 0.05, 0.075, 0.10, 0.20)
KEYS = [f'chong_{s:g}' for s in SIZES]
TARGET = 29.0                    # Chong et al.'s (2005) attended lengthening (%), observer reports
MARGIN = -2.0                    # C2: the smallest effect of Table 6
MIN_N = 10
SCORINGS = ('mOB', 'mBm', 'mB', 'mBabs', 'mOB2')   # mOB2: the passive observer (latency 300-600 ms)
VALID = 0.8                      # observer-valid: baseline reported / merged competitor episodes at least this
N_BOOT, BOOT_SEED = 10000, 8001001


def wilson(k, n, z=1.959964):
    if n == 0: return [np.nan, np.nan]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def prop(mask_true, mask_ok):
    k, n = int((mask_true & mask_ok).sum()), int(mask_ok.sum())
    ci = wilson(k, n)
    return dict(k=k, n=n, ci=ci, holds=bool(n > 0 and ci[0] > 0.5))


def med(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def boot_median_ci(x, seed=BOOT_SEED):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    bs = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(N_BOOT)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def change(base, cond, m):
    out = []
    for rb, rc in zip(base, cond):
        b = np.array([x[m] for x in rb], float); c = np.array([x[m] for x in rc], float)
        ok = np.isfinite(b) & np.isfinite(c)
        out.append(100 * (c[ok].mean() - b[ok].mean()) / b[ok].mean() if ok.sum() and b[ok].mean() > 0 else np.nan)
    return np.array(out, float)


def calibrate(att):
    """att: array (n_config, len(SIZES)) of attended changes (observer scoring). Adds the point (0, 0). Per configuration,
    the bracket j and weight w at which the attended change first reaches TARGET, by linear interpolation along the
    sizes: the calibrated size is s[j-1] + w (s[j] - s[j-1]). Reason codes: 'ok', 'never' (does not reach TARGET by the
    largest size), 'nonfinite' (a value needed is not finite)."""
    s = np.r_[0.0, SIZES]
    J, W, why = np.zeros(len(att), int), np.full(len(att), np.nan), np.array(['never'] * len(att), dtype=object)
    for i in range(len(att)):
        a = np.r_[0.0, att[i]]
        for j in range(1, len(s)):
            if not np.isfinite(a[j]):
                why[i] = 'nonfinite'; break
            if a[j] >= TARGET:
                J[i], W[i], why[i] = j, (TARGET - a[j - 1]) / (a[j] - a[j - 1]), 'ok'
                break
    return J, W, why


def at_cal(J, W, why, x):
    """x: array (n_config, len(SIZES)) of a change; its value at each configuration's calibrated size (NaN if not
    calibrated or if a value needed is not finite)."""
    s = np.r_[0.0, SIZES]
    out = np.full(len(x), np.nan)
    sz = np.full(len(x), np.nan)
    for i in range(len(x)):
        if why[i] != 'ok':
            continue
        c = np.r_[0.0, x[i]]; j, w = J[i], W[i]
        sz[i] = s[j - 1] + w * (s[j] - s[j - 1])
        if np.isfinite(c[j]) and np.isfinite(c[j - 1]):
            out[i] = c[j - 1] + w * (c[j] - c[j - 1])
    return out, sz


def model_block(name, M):
    R = M['results']
    n = len(M['accepted'])
    res = dict(n_config=n)
    # reproduction of the seventh test's baseline (same configurations, seeds and noise)
    R7 = D7['models'][name]['results']
    diffs = []
    for b8, b7 in zip(R['base'], R7['base']):
        for x8, x7 in zip(b8, b7):
            for m in ('mA', 'mBm', 'mOB', 'mOA'):
                if np.isfinite(x8[m]) or np.isfinite(x7[m]):
                    diffs.append(abs(x8[m] - x7[m]) / max(abs(x7[m]), 1e-12) if np.isfinite(x8[m]) and np.isfinite(x7[m]) else np.inf)
    res['baseline_reproduces_test7'] = dict(max_rel_diff=float(max(diffs)) if diffs else np.nan,
                                            identical=bool(diffs and max(diffs) < 1e-9))
    att = np.stack([change(R['base'], R['cond'][k], 'mOA') for k in KEYS], 1)
    res['attended_change_obs'] = {k: dict(median=med(att[:, j]), **prop(att[:, j] > 0, np.isfinite(att[:, j])))
                                  for j, k in enumerate(KEYS)}
    if n < MIN_N:
        res['testable'] = False
        return res
    res['testable'] = True
    # observer-valid stratum (pre-specified from the baseline alone): reported / merged competitor episodes >= VALID
    ratio = np.array([np.nanmean([x['nOB'] for x in c]) / np.nanmean([x['nBm'] for x in c]) for c in R['base']])
    valid = ratio >= VALID
    res['stratum'] = dict(n_valid=int(valid.sum()), n_all=n, ratio_median=med(ratio), T_median_valid_ms=med(np.array(R['T'])[valid]),
                          T_median_invalid_ms=med(np.array(R['T'])[~valid]))
    # C1 (manipulation check): the 10% increment lengthens the attended grating (observer scoring)
    res['C1_attended_lengthens'] = res['attended_change_obs']['chong_0.1']
    comp = {m: np.stack([change(R['base'], R['cond'][k], m) for k in KEYS], 1) for m in SCORINGS + ('mA', 'mOA2')}
    J, W, why = calibrate(att)
    groups = dict(valid=valid, all=np.ones(n, bool))
    cal = {}
    for gname, g in groups.items():
        cov = dict(n=int(g.sum()), calibrated=int(((why == 'ok') & g).sum()), never=int(((why == 'never') & g).sum()),
                   nonfinite=int(((why == 'nonfinite') & g).sum()))
        cov['reach_29_by_largest'] = prop(np.isfinite(att[:, -1]) & (att[:, -1] >= TARGET), g & np.isfinite(att[:, -1]))
        blk = dict(coverage=cov)
        for m in SCORINGS + ('mA', 'mOA2'):
            v, sz = at_cal(J, W, why, comp[m])
            v = np.where(g, v, np.nan); ok = np.isfinite(v)
            blk[m] = dict(n=int(ok.sum()), median=med(v), median_ci=boot_median_ci(v),
                          lengthens=prop(v > 0, ok), shortens=prop(v < 0, ok))
        blk['size_median'] = med(np.where(g, at_cal(J, W, why, comp['mOB'])[1], np.nan))
        # uncalibrated configurations: the competitor change (observer scoring) at the largest size
        unc = g & (why != 'ok')
        blk['uncalibrated_comp_obs_at_largest'] = dict(n=int(unc.sum()), median=med(comp['mOB'][unc, -1]))
        cal[gname] = blk
    res['calibrated'] = cal
    p = cal['valid']['mOB']
    testable = p['n'] >= MIN_N
    lo, hi = p['median_ci']
    # C2 (primary: observer-valid configurations): the competitor (observer scoring) is not shortened by more than 2%
    res['C2_not_shortened'] = dict(n=p['n'], median=p['median'], ci=p['median_ci'], holds=bool(testable and lo > MARGIN),
                                   testable=testable)
    # C3: the competitor (observer scoring) is lengthened
    res['C3_lengthened'] = dict(n=p['n'], median=p['median'], ci=p['median_ci'], holds=bool(testable and lo > 0),
                                testable=testable)
    # C4 (reported): the competitor is shortened (upper bound below 0), and by more than 2% (upper bound below -2%)
    res['C4_shortened_reported'] = dict(below_0=bool(testable and hi < 0), below_margin=bool(testable and hi < MARGIN),
                                        testable=testable)
    # reported: each size and scoring, uncalibrated (in Li et al.'s model the larger sizes approach lock-in)
    res['by_size'] = {k: {m: dict(median=med(comp[m][:, j]), median_ci=boot_median_ci(comp[m][:, j]),
                                  **prop(comp[m][:, j] > 0, np.isfinite(comp[m][:, j]))) for m in SCORINGS}
                      for j, k in enumerate(KEYS)}
    res['attended_by_size'] = {k: dict(obs=med(att[:, j]), model_unmerged=med(comp['mA'][:, j])) for j, k in enumerate(KEYS)}
    # reported: the seventh test's live gate at 10%, same configurations and noise
    g7 = {m: change(R7['base'], R7['cond']['gated'], m) for m in SCORINGS}
    j10 = KEYS.index('chong_0.1')
    res['live_gate_test7_10pct'] = {m: med(g7[m]) for m in SCORINGS}
    res['chong_10_minus_live_gate'] = {m: dict(median=med(comp[m][:, j10] - g7[m]),
                                              chong_gt_live=prop(comp[m][:, j10] > g7[m],
                                                                 np.isfinite(comp[m][:, j10]) & np.isfinite(g7[m])))
                                       for m in SCORINGS}
    res['live_gate_attended_obs_test7_10pct'] = med(change(R7['base'], R7['cond']['gated'], 'mOA'))
    # reported: the competitor's predominance (reported time) and number of reported competitor episodes
    pb = np.array(R['predominance']['base'], float)
    res['predominance'] = {k: dict(base=med(pb), cond=med(R['predominance'][k]),
                                   change_points=med(100 * (np.array(R['predominance'][k], float) - pb)))
                           for k in KEYS}
    res['n_reported_competitor_episodes_change'] = {k: med(change(R['base'], R['cond'][k], 'nOB')) for k in KEYS}
    res['duty'] = {k: med(v) for k, v in R['duty'].items()}
    res['level_in_comp'] = {k: med(v) for k, v in R['level_in_comp'].items()}
    return res
    res['testable'] = True
    # C1: the 10% increment under Chong et al.'s timing lengthens the attended grating (observer scoring)
    res['C1_attended_lengthens'] = res['attended_change_obs']['chong_0.1']
    comp = {m: np.stack([change(R['base'], R['cond'][k], m) for k in KEYS], 1) for m in SCORINGS}
    cal = {}
    for m in SCORINGS:
        sz, cc = calibrate(att, comp[m])
        ok = np.isfinite(cc)
        cal[m] = dict(n_calibrated=int(ok.sum()), size_median=med(sz), median=med(cc), median_ci=boot_median_ci(cc),
                      lengthens=prop(cc > 0, ok), shortens=prop(cc < 0, ok))
    res['calibrated'] = cal
    p = cal['mOB']
    testable = p['n_calibrated'] >= MIN_N
    # C2: at the calibrated size, the competitor (observer scoring) is not shortened by more than 2%
    res['C2_not_shortened'] = dict(n=p['n_calibrated'], median=p['median'], ci=p['median_ci'],
                                   holds=bool(testable and p['median_ci'][0] > MARGIN), testable=testable)
    # C3: at the calibrated size, the competitor (observer scoring) is lengthened
    res['C3_lengthened'] = dict(n=p['n_calibrated'], median=p['median'], ci=p['median_ci'],
                                holds=bool(testable and p['median_ci'][0] > 0), testable=testable)
    # reported: each size and scoring, uncalibrated
    res['by_size'] = {k: {m: dict(median=med(comp[m][:, j]), median_ci=boot_median_ci(comp[m][:, j]),
                                  **prop(comp[m][:, j] > 0, np.isfinite(comp[m][:, j]))) for m in SCORINGS}
                      for j, k in enumerate(KEYS)}
    # reported: the seventh test's live gate at 10%, same configurations and noise
    g7 = {m: change(R7['base'], R7['cond']['gated'], m) for m in SCORINGS}
    res['live_gate_test7_10pct'] = {m: med(g7[m]) for m in SCORINGS}
    res['chong_10_minus_live_gate'] = {m: dict(median=med(comp[m][:, 2] - g7[m]),
                                              chong_gt_live=prop(comp[m][:, 2] > g7[m],
                                                                 np.isfinite(comp[m][:, 2]) & np.isfinite(g7[m])))
                                       for m in SCORINGS}
    res['attended_change_unmerged_10'] = med(change(R['base'], R['cond']['chong_0.1'], 'mA'))
    res['live_gate_attended_obs_test7_10pct'] = med(change(R7['base'], R7['cond']['gated'], 'mOA'))
    # reported: the competitor's predominance (reported time) and number of reported competitor episodes
    pb = np.array(R['predominance']['base'], float)
    res['predominance'] = {k: dict(base=med(pb), cond=med(R['predominance'][k]),
                                   change_points=med(100 * (np.array(R['predominance'][k], float) - pb)))
                           for k in KEYS}
    res['n_reported_competitor_episodes_change'] = {k: med(change(R['base'], R['cond'][k], 'nOB')) for k in KEYS}
    res['duty'] = {k: med(v) for k, v in R['duty'].items()}
    res['dose_comp'] = {k: med(v) for k, v in R['dose_comp'].items()}
    return res


res = {name: model_block(name, M) for name, M in D['models'].items()}
dec = {}
for name, r in res.items():
    if not r.get('testable'):
        dec[name] = 'not testable (fewer than %d configurations)' % MIN_N
        continue
    nt = 'not tested (%d calibrated observer-valid configurations, fewer than %d)' % (r['C2_not_shortened']['n'], MIN_N)
    dec[name] = dict(C1=r['C1_attended_lengthens']['holds'],
                     C2=r['C2_not_shortened']['holds'] if r['C2_not_shortened']['testable'] else nt,
                     C3=r['C3_lengthened']['holds'] if r['C3_lengthened']['testable'] else nt,
                     baseline_identical=r['baseline_reproduces_test7']['identical'])
res['decisions'] = dec
json.dump(res, open(OUT, 'w'), indent=1, default=float)
print(json.dumps(dec, indent=1))
