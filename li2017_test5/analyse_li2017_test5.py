"""Pre-specified analysis of the fifth test (analysis_plan_li2017_test5.md, Section 5).
    python analyse_li2017_test5.py [li2017_test5_results.json] [li2017_test5_tests.json]
Decisions use merged scoring (mBm); every hypothesis is also reported under observer scoring (mOB), unmerged scoring
(mB) and the absolute criterion (mBabs)."""
import sys, json
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else 'li2017_test5_results.json'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'li2017_test5_tests.json'
D = json.load(open(SRC))
M = D['results']['main']; S = D['settings']; K = D['kernel']; C = D['accepted']
BIN, HA, HB, INC, PW = S['BIN'], S['HA'], S['HB'], S['INC'], S['PULSE_W']
PULSES = S['PULSES']
SCORINGS = ('mBm', 'mOB', 'mB', 'mBabs')
n = len(C)


def wilson(k, nn, z=1.959964):
    if nn == 0: return [np.nan, np.nan]
    p = k / nn; d = 1 + z * z / nn
    c = (p + z * z / (2 * nn)) / d; h = z * np.sqrt(p * (1 - p) / nn + z * z / (4 * nn * nn)) / d
    return [float(c - h), float(c + h)]


def change(key, m):
    out = []
    for rb, rc in zip(M['base'], M['cond'][key]):
        b = np.array([x[m] for x in rb], float); c = np.array([x[m] for x in rc], float)
        ok = np.isfinite(b) & np.isfinite(c)
        out.append(100 * (c[ok].mean() - b[ok].mean()) / b[ok].mean() if ok.sum() and b[ok].mean() > 0 else np.nan)
    return np.array(out, float)


def overlap(b, w, lo, hi):
    edges = np.arange(len(b) + 1) * w
    ov = np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0, None) / w
    return float(np.sum(np.asarray(b) * ov))


key = lambda mode, extra=None: f'{mode}_{INC:g}' + (f'_{extra[0]:g}-{extra[1]:g}' if extra else '')
cvec = np.array(M['c'], float)
okc = np.isfinite(cvec)                                   # configurations with a split point

# kernel predictions (percent of the kernel's mean merged competitor duration)
pred = {k_: np.full(n, np.nan) for k_ in ['gated', 'ungated'] + [f'pulse{p:g}' for p in PULSES]}
for i, (c, k) in enumerate(zip(C, K)):
    if k is None:
        continue
    bA = np.array(k['bA']); nb = len(k['fatt']); bBa, bBo = np.array(k['bB'][:nb]), np.array(k['bB'][nb:])
    fa = np.array(k['fatt']); w = BIN * k['T']; sc = 100 * INC * c['inp'] / k['meanD']
    pred['gated'][i] = sc * np.sum(bBa * fa)
    pred['ungated'][i] = sc * np.sum(bBa * fa + bBo * (1 - fa))
    T3 = M['T'][i]
    if np.isfinite(T3):
        for p in PULSES:
            pred[f'pulse{p:g}'][i] = sc * overlap(bA, w, p * T3, (p + PW) * T3)


def med(x):
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def tests(m):
    T = {}
    live, early, late = change(key('gated'), m), change(key('att_early'), m), change(key('att_late'), m)
    late_nt, cont = change(key('att_late_nt'), m), change(key('ungated'), m)
    du = {k_: np.array(v, float) for k_, v in M['duty'].items()}
    ok = okc & np.isfinite(early) & np.isfinite(late)
    nn = int(ok.sum())
    k1 = int((late[ok] > early[ok]).sum())
    T['L1'] = dict(late_gt_early=[k1, nn], ci=wilson(k1, nn), median_early=med(early[ok]), median_late=med(late[ok]),
                   holds=bool(wilson(k1, nn)[0] > 0.5))
    k2 = int((late[ok] > 0).sum())
    T['L2'] = dict(late_lengthens=[k2, nn], ci=wilson(k2, nn), holds=bool(wilson(k2, nn)[0] > 0.5))
    ok3 = ok & np.isfinite(late_nt)
    k3, n3 = int((late[ok3] > late_nt[ok3]).sum()), int(ok3.sum())
    T['L3'] = dict(late_gt_late_without_transition=[k3, n3], ci=wilson(k3, n3), median_late=med(late[ok3]),
                   median_late_nt=med(late_nt[ok3]), duty_late=med(du[key('att_late')][ok3]),
                   duty_late_nt=med(du[key('att_late_nt')][ok3]), holds=bool(wilson(k3, n3)[0] > 0.5))
    # L4: kernel time course for pulses (configurations with a kernel and T)
    okp = np.all([np.isfinite(pred[f'pulse{p:g}']) for p in PULSES], axis=0)
    om = np.array([med(change(key('pulse', (p, PW)), m)[okp]) for p in PULSES])
    pm = np.array([med(pred[f'pulse{p:g}'][okp]) for p in PULSES])
    ratio = om.sum() / pm.sum() if pm.sum() != 0 else np.nan
    r = float(np.corrcoef(om, pm)[0, 1]) if np.all(np.isfinite(om)) and np.std(om) > 0 and np.std(pm) > 0 else np.nan
    T['L4'] = dict(positions=list(PULSES), observed=om.tolist(), kernel=pm.tolist(), sum_ratio=float(ratio),
                   profile_r=r, n=int(okp.sum()),
                   holds=bool(np.isfinite(ratio) and 0.67 <= ratio <= 1.5 and np.isfinite(r) and r >= 0.7))
    # reported without decisions
    per = lambda x, k_: x / du[k_]
    ke, kl = key('att_early'), key('att_late')
    okd = ok & (du[ke] > 0) & (du[kl] > 0)
    kd = int((per(late, kl)[okd] > per(early, ke)[okd]).sum())
    att = {k_: change(k_, 'mA') for k_ in (ke, kl, key('att_late_nt'), key('gated'))}
    okk = np.isfinite(pred['ungated']) & np.isfinite(cont)
    T['reported'] = dict(
        per_unit_duty=dict(late_gt_early=[kd, int(okd.sum())], ci=wilson(kd, int(okd.sum())),
                           median_early=med(per(early, ke)[okd]), median_late=med(per(late, kl)[okd])),
        attended_change={k_: med(v) for k_, v in att.items()},
        live=med(live), live_lengthen=[int((live > 0).sum()), int(np.isfinite(live).sum())],
        ratio_late_live=med(late[ok]) / med(live[ok]) if med(live[ok]) else np.nan,
        ratio_early_live=med(early[ok]) / med(live[ok]) if med(live[ok]) else np.nan,
        continuous=med(cont[okk]), continuous_pred=med(pred['ungated'][okk]), gated_pred=med(pred['gated']),
        duty={k_: med(v) for k_, v in du.items()}, n_split=int(okc.sum()), n_kernel=int(sum(k is not None for k in K)))
    return T


res = {m: tests(m) for m in SCORINGS}
res['n_config'] = n
json.dump(res, open(OUT, 'w'), indent=1, default=float)
print(f'n = {n} configurations; decisions on merged scoring')
for h in ('L1', 'L2', 'L3', 'L4'):
    print(f'  {h}: ' + ' / '.join(f'{m}: {"holds" if res[m][h]["holds"] else "not"}' for m in SCORINGS))
print(json.dumps(res['mBm'], indent=1, default=float)[:3500])
