"""Pre-check diagnostic: one accepted pilot configuration per model, nonzero increment (scale 1), 4 seeds x 30 s.
(a) run_chunk7 as written; (b) direct run_batch7 calls to measure, per condition: duty, duty within current grating-1
dominance, fraction of on-time in competitor dominance, and synchrony of current dominance with the baseline record."""
import sys, json, time
sys.path.insert(0, '/home/claude/chris/work/t7/li2017_test7')
import numpy as np
import run_li2017_test7 as T7, run_li2017_test3 as R3, run_li2017_test as R
src = '/home/claude/chris/work/t7/src6/'
pick = dict(li=('pilot7b_li', 2526), sfa=('pilot7b_sfa', 1911), dep=('pilot7c_dep', 946))
T_MS, NS = 30000.0, 4
out = {}
for model, (f, Tt) in pick.items():
    d = json.load(open(src + f + '.json'))
    cfg = [x for x in d if x['T'] == x['T'] and round(x['T']) == Tt][0]['cfg']
    t0 = time.time()
    rc = T7.run_chunk7(model, [cfg], 1.0, T_MS, 'precheck7_' + model, NS, T7.RANGES[model])
    ch = {}
    for k in rc['out']:
        for m in ('mBm', 'mA', 'mOB'):
            b = np.array([x[m] for x in rc['base'][0]], float); c = np.array([x[m] for x in rc['out'][k][0]], float)
            ok = np.isfinite(b) & np.isfinite(c)
            ch[(k, m)] = 100 * (c[ok].mean() - b[ok].mean()) / b[ok].mean() if ok.any() else np.nan
    # direct diagnostics
    seeds = list(range(NS)); tag = 'precheck7_' + model; rg = T7.RANGES[model]
    b0 = int(R3.BURN_MS / R3.DT_REC)
    rb0 = T7.run_batch7(model, [cfg], seeds, 'none', 0.0, None, tag, t_ms=T_MS, ranges=rg)[0]
    thr = R3.THETA_FRAC * rb0[:, :, b0:].mean(axis=(1, 2))
    rb0b, on0, rp0, rq0, ex0, Arec = T7.run_batch7(model, [cfg], seeds, 'none_rec', 0.0, thr, tag, t_ms=T_MS, ranges=rg)
    Ads = Arec[:, ::2][:, :rb0.shape[2]]
    A0 = (rb0[:, 0] - rb0[:, 1]) > thr[:, None]
    # Arec is read at the start of step idx from the state after step idx-1; rb recorded after step idx.
    lagchk = [np.mean(Ads[i, 1:] == ((rb0[i, 0, :-1] - rb0[i, 1, :-1]) > thr[i])) for i in range(NS)]
    print(model, 'Arec alignment check (should be ~1): %.5f' % np.mean(lagchk), 'n episodes base mBm', [x['nBm'] for x in rc['base'][0]])
    row = {}
    for name, mode in T7.CONDS:
        rec = Arec if mode.startswith('replay') else None
        rb, on, rp, rq, ex = T7.run_batch7(model, [cfg], seeds, mode, T7.INC, thr, tag, rec=rec, t_ms=T_MS, ranges=rg)
        x = rb[:, :, b0:]; o = on[:, b0:] > 0
        A = (x[:, 0] - x[:, 1]) > thr[:, None]; B = (x[:, 1] - x[:, 0]) > thr[:, None]
        Ab = Ads[:, b0:]
        # first divergence from baseline (ms after burn-in) and later synchrony
        div = [int(np.argmax(np.abs(rb[i] - rb0[i]).max(0) > 1e-6)) for i in range(NS)]
        row[name] = dict(duty=float(o.mean()), duty_in_A=float(o[A].mean()), dose_B=float((o & B).sum() / max(o.sum(), 1)),
                         sync_with_base=float((A == Ab).mean()),
                         on_after_base_end=float((o & ~Ab).sum() / max(o.sum(), 1)),
                         first_div_s=float(np.median(div)) / 1000.0, agree=float(np.mean(ex['agree'])))
    out[model] = dict(cfg=cfg, changes={f'{k}|{m}': float(v) for (k, m), v in ch.items()}, diag=row,
                      dose_comp=rc['dose_comp'], duty=rc['duty'], secs=time.time() - t0)
    print(model, json.dumps(out[model], indent=1), flush=True)
json.dump(out, open('diag_masked.json', 'w'), indent=1)
