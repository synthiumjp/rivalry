"""Pre-specified analysis of the seventh test (analysis_plan_li2017_test7.md, Section 5).
    python analyse_li2017_test7.py [li2017_test7_results.json] [li2017_test7_tests.json]
Decisions use merged scoring (mBm) and a lower Wilson 95% bound above 0.5 unless stated; mean-based hypotheses are also
reported under observer scoring (mOB), unmerged scoring (mB) and the absolute criterion (mBabs)."""
import sys, json
import numpy as np
from scipy.stats import rankdata

SRC = sys.argv[1] if len(sys.argv) > 1 else 'li2017_test7_results.json'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'li2017_test7_tests.json'
REF = sys.argv[3] if len(sys.argv) > 3 else '../li2017_test6/li2017_test6_results.json'   # the sixth test, for P1
D = json.load(open(SRC))
D6 = json.load(open(REF))
MIN_N = 10                           # a hypothesis is tested in a model only with at least this many configurations
MIN_ATT = 5.0                        # P1: configurations whose attended change is at least 5%
SCORINGS = ('mBm', 'mOB', 'mB', 'mBabs')
N_BOOT, BOOT_SEED = 10000, 7001001
ATT_OBS = 29.0                       # Chong et al.'s (2005) attended lengthening (%)


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


def change(R, key, m, sl=slice(None)):
    out = []
    for rb, rc in zip(R['base'], R['cond'][key]):
        b = np.array([x[m] for x in rb[sl]], float); c = np.array([x[m] for x in rc[sl]], float)
        ok = np.isfinite(b) & np.isfinite(c)
        out.append(100 * (c[ok].mean() - b[ok].mean()) / b[ok].mean() if ok.sum() and b[ok].mean() > 0 else np.nan)
    return np.array(out, float)


def zmean(rs):
    rs = np.asarray(rs, float); rs = rs[np.isfinite(rs)]
    return float(np.tanh(np.mean(np.arctanh(np.clip(rs, -0.999, 0.999))))) if len(rs) else np.nan


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]) if len(a) > 3 else np.nan, int(len(a))


def boot_median_ci(x, seed=BOOT_SEED):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    bs = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(N_BOOT)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def ratio_comp_att(R):
    """Per configuration: the live gate's competitor change (merged) divided by its attended change (unmerged), for
    configurations whose attended change is at least MIN_ATT."""
    comp, att = change(R, 'gated', 'mBm'), change(R, 'gated', 'mA')
    ok = np.isfinite(comp) & np.isfinite(att) & (att >= MIN_ATT)
    return comp[ok] / att[ok]


def model_block(M):
    R = M['results']
    if len(M['accepted']) < MIN_N:
        return dict(n_config=len(M['accepted']), n_drawn=M['n_drawn'], rejected=M['rejected'], testable=False)
    res = dict(n_config=len(M['accepted']), n_drawn=M['n_drawn'], rejected=M['rejected'],
               r_ab_accept_median=med(M['r_ab_accept']), T_median_ms=med(R['T']))
    ch = {k: {m: change(R, k, m) for m in SCORINGS} for k in R['cond']}
    g, u, s, mk = ch['gated'], ch['ungated'], ch['replay_self'], ch['replay_masked']
    res['testable'] = True
    res['G1_gated_lengthens'] = {m: dict(prop(g[m] > 0, np.isfinite(g[m])), median=med(g[m]), median_ci=boot_median_ci(g[m]))
                                 for m in SCORINGS}
    res['G2_continuous_shortens'] = {m: dict(prop(u[m] < 0, np.isfinite(u[m])), median=med(u[m])) for m in SCORINGS}
    res['M1_live_gt_masked'] = {m: prop(g[m] > mk[m], np.isfinite(g[m]) & np.isfinite(mk[m])) for m in SCORINGS}
    res['R1_live_gt_self_reported'] = {m: prop(g[m] > s[m], np.isfinite(g[m]) & np.isfinite(s[m])) for m in SCORINGS}
    res['masked_lengthens_reported'] = {m: dict(prop(mk[m] > 0, np.isfinite(mk[m])), median=med(mk[m])) for m in SCORINGS}
    res['self_lengthens_reported'] = {m: dict(prop(s[m] > 0, np.isfinite(s[m])), median=med(s[m])) for m in SCORINGS}
    res['medians'] = {m: dict(live=med(g[m]), continuous=med(u[m]), self=med(s[m]), masked=med(mk[m])) for m in SCORINGS}
    res['attended_change'] = {k: med(change(R, k, 'mA')) for k in R['cond']}
    with np.errstate(divide='ignore', invalid='ignore'):
        res['ratio_comp_to_att'] = {k: med(change(R, k, 'mBm') / change(R, k, 'mA')) for k in R['cond']}
    res['duty'] = {k: med(v) for k, v in R['duty'].items()}
    res['dose_comp'] = {k: med(v) for k, v in R['dose_comp'].items()}
    res['cover_att'] = {k: med(v) for k, v in R['cover_att'].items()}
    res['agreement'] = {k: med(v) for k, v in R['agree'].items()}
    # serial correlation in the main run (all seeds), and the gate's lengthening where it is below 0.1
    r_ab = np.array([zmean(x['r_ab']) for x in R['serial']['base']])
    res['serial_main'] = dict(median=med(r_ab), iqr=[float(np.nanpercentile(r_ab, 25)), float(np.nanpercentile(r_ab, 75))],
                              n_below_0_1=int(np.sum(r_ab < 0.1)), n_below_0_2=int(np.sum(r_ab < 0.2)))
    h = len(R['serial']['base'][0]['r_ab']) // 2
    r_h = np.array([zmean(x['r_ab'][:h]) for x in R['serial']['base']])
    comp_h = change(R, 'gated', 'mBm', slice(h, None))
    low = r_h < 0.1
    res['G1_main_r_below_0_1_reported'] = dict(prop(comp_h > 0, low & np.isfinite(comp_h)), median=med(comp_h[low]),
                                               note='r_ab from seeds 0-15, change from seeds 16-31')
    # G1 by mean duration (above / below the model's median T) and among configurations with acceptance CV <= 0.7
    T = np.array(R['T'], float); hiT = T >= np.nanmedian(T)
    cv = np.array(M['cv_accept'], float)
    res['G1_by_duration_reported'] = dict(long=dict(prop(g['mBm'] > 0, hiT & np.isfinite(g['mBm'])), median=med(g['mBm'][hiT])),
                                          short=dict(prop(g['mBm'] > 0, ~hiT & np.isfinite(g['mBm'])), median=med(g['mBm'][~hiT])),
                                          T_median_ms=med(T))
    res['G1_cv_le_0_7_reported'] = dict(prop(g['mBm'] > 0, (cv <= 0.7) & np.isfinite(g['mBm'])), median=med(g['mBm'][cv <= 0.7]))
    # distribution (median of pooled merged competitor durations)
    QI = 9   # QS = 5..95 in steps of 5: index 9 is the 50th percentile
    dm = np.array([100 * (c['q'][QI] - b['q'][QI]) / b['q'][QI] if b and c else np.nan
                   for b, c in zip(R['dists']['base'], R['dists']['gated'])])
    res['median_duration_change_reported'] = dict(prop(dm > 0, np.isfinite(dm)), median=med(dm))
    # reported: the product r_ab x attended change (seeds 0-15) against the competitor change (seeds 16-31), and each
    # factor alone
    att_h = change(R, 'gated', 'mA', slice(None, h))
    pred = r_h * att_h
    res['product_predictor_reported'] = dict(pred=spearman(pred, comp_h)[0], r_only=spearman(r_h, comp_h)[0],
                                             att_only=spearman(att_h, comp_h)[0], n=spearman(pred, comp_h)[1])
    rt = ratio_comp_att(R)
    res['ratio_comp_to_att_gated'] = dict(median=med(rt), median_ci=boot_median_ci(rt), n=int(len(rt)),
                                          competitor_at_attended_29=float(med(rt) * ATT_OBS))
    return res


from scipy.stats import mannwhitneyu
res = {}
for name, M in D['models'].items():
    res[name] = model_block(M)
    if not res[name].get('testable'):
        continue
    # P1: the competitor-to-attended ratio is lower than in the sixth test's configurations of the same model
    r7 = ratio_comp_att(M['results'])
    r6 = ratio_comp_att(D6['models'][name]['results'])
    p = float(mannwhitneyu(r7, r6, alternative='less').pvalue) if len(r7) >= MIN_N and len(r6) >= MIN_N else np.nan
    res[name]['P1_ratio_lower_than_test6'] = dict(median_test7=med(r7), median_test6=med(r6), n7=int(len(r7)),
                                                  n6=int(len(r6)), p_one_sided=p, holds=bool(np.isfinite(p) and p < 0.05))
dec = {}
for name in D['models']:
    r = res[name]
    if not r.get('testable'):
        dec[name] = 'not testable (fewer than %d configurations)' % MIN_N
        continue
    dec[name] = dict(G1=r['G1_gated_lengthens']['mBm']['holds'], G2=r['G2_continuous_shortens']['mBm']['holds'],
                     M1=r['M1_live_gt_masked']['mBm']['holds'], P1=r['P1_ratio_lower_than_test6']['holds'])
res['decisions'] = dec
json.dump(res, open(OUT, 'w'), indent=1, default=float)
print(json.dumps(dec, indent=1))
