"""Decomposition of the continuous-to-gated contrast, as reported in Section 4.3.3 and Table S2.
Estimator: difference of medians on complete cases (configurations with all four conditions),
95% percentile bootstrap over configurations, 10,000 resamples, seed 0. Reproduces the published
values exactly from the earlier campaign's output."""
import json, sys, numpy as np
d = json.load(open(sys.argv[1])); R = d.get('rows', d)
u, y, s, l = (np.array(R[k], float) for k in ('ungated', 'yoked', 'shuffled', 'live'))
ok = np.isfinite(u) & np.isfinite(y) & np.isfinite(s) & np.isfinite(l)
u, y, s, l = u[ok], y[ok], s[ok], l[ok]
f = lambda u, y, s, l: np.array([np.median(y)-np.median(u), np.median(s)-np.median(y),
                                 np.median(l)-np.median(s), np.median(l)-np.median(u)])
est = f(u, y, s, l)
rng = np.random.default_rng(0); n = len(u)
B = np.array([f(u[i], y[i], s[i], l[i]) for i in rng.integers(0, n, (10000, n))])
lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
print(f"complete cases: {int(ok.sum())} of {len(ok)}")
for k, name in enumerate(('schedule statistics (yoked - continuous)', 'ordering (shuffled - yoked)',
                          'contingency (live - shuffled)', 'full span (live - continuous)')):
    print(f"  {name:<42} {est[k]:+6.1f}  [{lo[k]:+.1f}, {hi[k]:+.1f}]")
