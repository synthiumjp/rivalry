"""Pre-run behaviour check (precheck8). Usage: python3 behav8.py <model>
Run from the li2017_test8 folder with sys.path pointing there; writes only into precheck/."""
import sys, json, time
sys.path.insert(0, '/home/claude/chris/work/t8/li2017_test8')
import numpy as np
import run_li2017_test8 as T8, run_li2017_test3 as R3, run_li2017_test7 as T7
import li2017 as L, li2017_t6 as L6, rate_models as RM
OUTD = '/home/claude/chris/work/t8/li2017_test8/precheck/'
model = sys.argv[1]
NS, TMS = 4, float(sys.argv[2]) if len(sys.argv) > 2 else 30000.0
D7 = json.load(open('/home/claude/chris/work/t8/li2017_test7/li2017_test7_results.json'))
cfg = D7['models'][model]['accepted'][:1]
ranges = T7.RANGES[model]
tag = f'precheck8_{model}'
t0 = time.time()
r = T8.run_chunk8(model, cfg, 1.0, TMS, tag, NS, ranges)
t_chunk = time.time() - t0
def chg(base, cond, m):
    b = np.array([x[m] for x in base], float); c = np.array([x[m] for x in cond], float)
    ok = np.isfinite(b) & np.isfinite(c)
    return float(100 * (c[ok].mean() - b[ok].mean()) / b[ok].mean()) if ok.sum() else np.nan
summ = dict(model=model, cfg=cfg[0], secs_chunk=t_chunk, T=r['T'][0])
summ['base'] = {m: float(np.nanmean([x[m] for x in r['base'][0]])) for m in ('mOA', 'mOB', 'mA', 'mB', 'mBm', 'nOB')}
for k, v in r['out'].items():
    summ[k] = dict({m: chg(r['base'][0], v[0], m) for m in ('mOA', 'mOB', 'mA', 'mBm', 'mB', 'mBabs', 'nOB')},
                   duty=r['duty'][k][0], dose_comp=r['dose_comp'][k][0], pred=r['predominance'][k][0],
                   pred_base=r['predominance']['base'][0])
# per-seed values at 10% to see noise in the attended change
summ['per_seed_mOA_base'] = [x['mOA'] for x in r['base'][0]]
summ['per_seed_mOA_10'] = [x['mOA'] for x in r['out']['chong_0.1'][0]]
summ['per_seed_mOB_base'] = [x['mOB'] for x in r['base'][0]]
summ['per_seed_mOB_10'] = [x['mOB'] for x in r['out']['chong_0.1'][0]]

# --- equivalence with the third test's Schedule3 'chong' (no hold), same seed, same simulator, 10% ---
seeds = list(range(NS)); b = int(R3.BURN_MS / R3.DT_REC)
rb0 = T8.run_batch8(model, cfg, seeds, 'none', 0.0, None, tag, TMS, ranges)[0]
thr = R3.THETA_FRAC * rb0[:, :, b:].mean(axis=(1, 2))
rb8, on8, rp8, rq8, ex8 = T8.run_batch8(model, cfg, seeds, 'chong', 0.10, thr, tag, TMS, ranges)
N = len(seeds)
rep = lambda key: np.repeat([c[key] for c in cfg], N)
keyt = tuple(round(c[kk], 6) for c in cfg for kk in sorted(ranges))
s = R3.seed_of(tag, tuple(seeds), keyt)
inst = [(tuple(round(c[kk], 6) for kk in sorted(ranges)), sd) for c in cfg for sd in seeds]
pools = (R3.latency_pools(inst, tag, 1, R3.LAT_MS), R3.latency_pools(inst, tag, 2, R3.LAT2_MS))
if model == 'li':
    p = L.default_params(1); p['T'] = TMS; nt = int(round(TMS / p['dt'])) + 1
    pv = dict(in1=rep('inp'), in2=rep('inp'), wo=rep('wo'), wa=rep('wa'), wh=rep('wh'), tau_h=rep('tau_h'))
    sch3 = R3.Schedule3('chong', thr, p['dt'], N, nt, T=None, dL=None, extra=None, pools=pools)
    rb3, on3 = L.simulate(p, N=N, seed=s, noise_sigma=1.0, record_every=2, params_vec=pv, inc=rep('inp') * 0.10,
                          schedule=sch3, noise_scale=rep('noise'))
else:
    nt = int(round(TMS / RM.DT)) + 1
    pv = {kk: rep(kk) for kk in ranges if kk != 'noise'}
    sch3 = R3.Schedule3('chong', thr, RM.DT, N, nt, T=None, dL=None, extra=None, pools=pools)
    rb3, on3 = RM.simulate(model, pv, N, s, TMS, inc=rep('I') * 0.10, schedule=sch3, record_every=2,
                           noise_scale=rep('noise'))
kk = rb3.shape[2]
summ['equiv_schedule3_chong'] = dict(rb=bool(np.array_equal(rb3, rb8)), on=bool(np.array_equal(on3, on8)),
                                     rep1=bool(np.array_equal(sch3.obs.rec[:, :kk], rp8)),
                                     rep2=bool(np.array_equal(sch3.obs2.rec[:, :kk], rq8)))
# --- level trace against reports, instance 0 ---
i = 0
lev, rp = on8[i].astype(float), rp8[i]
dl = np.diff(lev)
summ['level'] = dict(min=float(lev.min()), max=float(lev.max()),
                     max_abs_step=float(np.abs(dl).max()), expected_step=float(R3.DT_REC * 1.0 / R3.RAMP_MS * 1.0),
                     frac_up_while_rep1=float(np.mean(dl[rp[1:] == 1] >= -1e-7)),
                     frac_down_while_not_rep1=float(np.mean(dl[rp[1:] != 1] <= 1e-7)),
                     mean_lev_rep1=float(lev[rp == 1].mean()), mean_lev_rep2=float(lev[rp == 2].mean()),
                     frac_at_1_rep1=float(np.mean(lev[rp == 1] >= 0.999)))
# report-onset aligned: level 0..1000 ms after each report of grating 1 and grating 2
ch = np.flatnonzero(np.diff(rp.astype(int)) != 0) + 1
on1 = [c for c in ch if rp[c] == 1 and c + 1100 < len(rp)]
on2 = [c for c in ch if rp[c] == 2 and c + 1100 < len(rp)]
summ['level_after_report1_ms'] = {t: float(np.mean([lev[c + t] for c in on1])) for t in (0, 100, 260, 520, 800)}
summ['level_after_report2_ms'] = {t: float(np.mean([lev[c + t] for c in on2])) for t in (0, 100, 260, 520, 800)}
# level at perceptual (model) switch to competitor, and how long the level stays > 0 after the competitor becomes dominant
x = rb8[i].astype(float); d = x[0] - x[1]
st = np.where(d > thr[i], 1, np.where(d < -thr[i], 2, 0))
v, s0, e0 = R3.runs(st)
idxB = [ss for a, ss, ee in zip(v, s0, e0) if a == 2 and (ee - ss) >= 50]
summ['level_at_competitor_onset_mean'] = float(np.mean([lev[ss] for ss in idxB])) if idxB else np.nan
# reported vs model durations (instance 0)
OA, OB = R3.observer_durations(rp, 1.0)
summ['inst0'] = dict(n_rep_A=len(OA), n_rep_B=len(OB), mean_rep_A=float(np.mean(OA)), mean_rep_B=float(np.mean(OB)))
np.savez_compressed(OUTD + f'trace_{model}_{int(TMS//1000)}s.npz', lev=lev, rp=rp, rb=rb8[i], thr=thr[i])
summ['secs_total'] = time.time() - t0
json.dump(summ, open(OUTD + f'behav8_{model}_{int(TMS//1000)}s.json', 'w'), indent=1, default=float)
print(json.dumps(summ, indent=1, default=float))
