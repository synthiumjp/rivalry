# Check that the modified port (graded increment levels) reproduces the first test exactly.
import json, numpy as np, run_li2017_test as R
d = json.load(open('../li2017_test/li2017_test_results.json')); acc = d['accepted'][:25]
base, out = R.run_chunk(acc, [('gated', 0.10, None), ('offset', 0.20, 1.0)], 1.0, 120000.0)
ok = all(np.array_equal(np.array([b['mB'] for b in base]), np.array([c['base_matched']['mB'] for c in acc])) for _ in [0])
for k, v in out.items():
    for m in ('mA', 'mB', 'mBabs', 'duty'):
        a = np.array([x[m] for x in v]); r = np.array([x[m] for x in d['results'][k][:25]])
        ok &= np.array_equal(a, r, equal_nan=True) if m != 'duty' else np.allclose(a, r)
print('modified port reproduces first test exactly:', ok)
