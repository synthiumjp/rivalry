"""EXPLORATORY follow-up analyses of the third test (not registered), using the registered analysis's
own functions. (1) H14 and H21 under the two scorings the analysis script did not compute; (2) H21 as
a function of the assumed effective withdrawal offset d0, and against a constant comparator; (3) the
fraction of merged competitor episodes the observer reports, per sample and condition; (4) H19
decomposed into the change in the number of reported competitor episodes and in reported time.
    python followup.py ../li2017_test3_results.json"""
import sys, io, json, math, contextlib
import numpy as np
src = sys.argv[1] if len(sys.argv) > 1 else '../li2017_test3_results.json'
sys.argv = ['analyse', src, '/dev/null']
code = open('../analyse_li2017_test3.py').read()
ns = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(code, 'analyse_li2017_test3.py', 'exec'), ns)
h14, h21, M, NA, HN, agg, ALL, H0, H1 = (ns[k] for k in ('h14', 'h21', 'M', 'NA', 'HN', 'agg', 'ALL', 'H0', 'H1'))
out = {}
# (1) remaining scorings
for m in ('mB', 'mBabs'):
    r = h14(m, ALL, ALL, 'base_all', True, f'five windows, {m}')
    out[f'H14_{m}'] = {k: r[k] for k in ('vs_hazard', 'vs_dose', 'n_excluded')}
    r = h21('ungated_0.1_h0', H0, H1, True, f'H21, {m}', m=m)
    out[f'H21_{m}'] = {k: r[k] for k in ('vs_dose', 'vs_hazard', 'n_no_prediction', 'n_excluded_hold0_not_positive', 'median_obs', 'median_pred_handover', 'median_pred_dose')}
# (2) d0 sweep and constant comparator
sweep = {}
for d0 in (300, 400, 500, 585, 700, 800, 900):
    ns['D_OFF_MS'] = float(d0)
    r = ns['h21']('ungated_0.1_h0', H0, H1, True, f'd0 {d0}')
    pc = r['per_config']; do = np.array([np.nan if v is None else v for v in pc['dobs']], float)
    dp, dz = np.array(pc['dpred'], float), np.array(pc['ddose'], float)
    fin = np.isfinite(do) & np.isfinite(dp) & np.isfinite(dz) & (do > 0)
    const = np.median(do[fin])
    sweep[d0] = dict(k=r['vs_dose']['k'], n=r['vs_dose']['n'], ci=r['vs_dose']['ci'],
                     within25_handover=float(np.mean(np.abs(np.log(do[fin] / dp[fin])) < math.log(1.25))),
                     within25_dose=float(np.mean(np.abs(np.log(do[fin] / dz[fin])) < math.log(1.25))),
                     within25_constant=float(np.mean(np.abs(np.log(do[fin] / const)) < math.log(1.25))))
ns['D_OFF_MS'] = 585.0
out['H21_d0_sweep'] = sweep
# (3) reported fraction nOB / nBm
def frac(S, key=None):
    rows = S['base'] if key is None else S['cond'][key]
    return float(np.nanmedian(agg(rows, 'nOB', ALL) / agg(rows, 'nBm', ALL)))
rf = {}
for S, nm in ((M, 'main'), (NA, 'noadapt'), (HN, 'highnoise')):
    rf[nm] = {'baseline': frac(S)}
    for k in S['cond']:
        rf[nm][k] = frac(S, k)
out['reported_fraction'] = rf
# (4) H19 decomposition: % change in reported competitor episodes and in total reported competitor time
def pct(a, b): return 100 * (a - b) / b
for key in ('chong_0.1', 'gated_0.1', 'rep_gated_0.2', 'gated_0.2'):
    nb, nc = agg(M['base'], 'nOB', ALL), agg(M['cond'][key], 'nOB', ALL)
    tb = agg(M['base'], 'mOB', ALL) * nb; tc = agg(M['cond'][key], 'mOB', ALL) * nc
    nmb, nmc = agg(M['base'], 'nBm', ALL), agg(M['cond'][key], 'nBm', ALL)
    out[f'decomposition_{key}'] = dict(reported_episodes=float(np.nanmedian(pct(nc, nb))), reported_time=float(np.nanmedian(pct(tc, tb))),
                                       merged_episodes=float(np.nanmedian(pct(nmc, nmb))),
                                       observer_duration=float(np.nanmedian(pct(agg(M['cond'][key], 'mOB', ALL), agg(M['base'], 'mOB', ALL)))))
json.dump(out, open('followup_summary.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)[:6000])
