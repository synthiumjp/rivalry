"""Pre-specified analysis of the sampling-sensitivity check (analysis_plan_li2017_sens.md, Section 4).
    python analyse_li2017_sens.py li2017_sens_L.json li2017_sens_U.json [li2017_sens_tests.json]"""
import sys, json
import numpy as np

SCORINGS = ('mBm', 'mOB', 'mB', 'mBabs')
BOUND = 0.75


def wilson(k, n, z=1.959964):
    if n == 0: return [np.nan, np.nan]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def change(M, key, m):
    out = []
    for rb, rc in zip(M['base'], M['cond'][key]):
        b = np.array([x[m] for x in rb], float); c = np.array([x[m] for x in rc], float)
        ok = np.isfinite(b) & np.isfinite(c)
        out.append(100 * (c[ok].mean() - b[ok].mean()) / b[ok].mean() if ok.sum() and b[ok].mean() > 0 else np.nan)
    return np.array(out, float)


def analyse(D):
    M = D['results']; res = dict(n_config=len(D['accepted']), n_drawn=D['n_drawn'], rejected=D['rejected'])
    for m in SCORINGS:
        g, u = change(M, 'gated_0.1', m), change(M, 'ungated_0.1', m)
        okg, oku, okb = np.isfinite(g), np.isfinite(u), np.isfinite(g) & np.isfinite(u)
        k1, n1 = int((g[okg] > 0).sum()), int(okg.sum())
        k2, n2 = int((u[oku] < 0).sum()), int(oku.sum())
        k3, n3 = int((g[okb] > u[okb]).sum()), int(okb.sum())
        res[m] = dict(
            S1_gated_lengthens=dict(k=k1, n=n1, ci=wilson(k1, n1), median=float(np.median(g[okg])) if n1 else np.nan,
                                    holds=bool(wilson(k1, n1)[0] > BOUND)),
            S2_continuous_shortens=dict(k=k2, n=n2, ci=wilson(k2, n2), median=float(np.median(u[oku])) if n2 else np.nan,
                                        holds=bool(wilson(k2, n2)[0] > BOUND)),
            S3_gated_exceeds_continuous=dict(k=k3, n=n3, ci=wilson(k3, n3), holds=bool(wilson(k3, n3)[0] > BOUND)),
            not_measurable=dict(gated=int((~okg).sum()), continuous=int((~oku).sum())))
    return res


if __name__ == '__main__':
    srcs = [a for a in sys.argv[1:] if a.endswith('.json') and 'tests' not in a]
    dst = next((a for a in sys.argv[1:] if 'tests' in a), 'li2017_sens_tests.json')
    R = {json.load(open(s))['ensemble']: analyse(json.load(open(s))) for s in srcs}
    json.dump(R, open(dst, 'w'), indent=1, default=float)
    for e, r in R.items():
        print(f'ensemble {e}: {r["n_config"]} accepted of {r["n_drawn"]} drawn; rejected {r["rejected"]}')
        for m in SCORINGS:
            x = r[m]
            print(f'  {m}: gated lengthens {x["S1_gated_lengthens"]["k"]}/{x["S1_gated_lengthens"]["n"]} '
                  f'(median {x["S1_gated_lengthens"]["median"]:+.1f}%), continuous shortens '
                  f'{x["S2_continuous_shortens"]["k"]}/{x["S2_continuous_shortens"]["n"]} '
                  f'(median {x["S2_continuous_shortens"]["median"]:+.1f}%), gated > continuous '
                  f'{x["S3_gated_exceeds_continuous"]["k"]}/{x["S3_gated_exceeds_continuous"]["n"]}; '
                  f'S1 {x["S1_gated_lengthens"]["holds"]}, S2 {x["S2_continuous_shortens"]["holds"]}, '
                  f'S3 {x["S3_gated_exceeds_continuous"]["holds"]}')
