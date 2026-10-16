"""Synthetic check of the third test's analysis with realistic distributions and noise.

Structure: the 200 main and 100 no-adaptation configurations of the SECOND registered test (their
baseline per-seed values, T, d_L and competitor-duration quantiles; li2017_test2_results.json), so
that the distributions have the shapes this model produces. Window profiles and hold curves are
injected as each account predicts, scaled by 0.6 (the second test's late window was about half its
handover prediction), with noise: a per-configuration baseline error of 2.4% shared by all
conditions, and a per-seed error of 10% (about 3.5% per configuration, as in the second test's seed
halves). Scenarios: handover, hazard-multiplier, dose, and handover and hazard with +4% added to the
two earliest windows (second-order lengthening).

    python synth_test3.py ../li2017_test2/li2017_test2_results.json
"""
import json, copy, subprocess, sys, zlib
import numpy as np
import run_li2017_test3 as R3

src = sys.argv[1] if len(sys.argv) > 1 else '../li2017_test2/li2017_test2_results.json'
D2 = json.load(open(src))
S_ = dict(R3.__dict__)
WIN = list(R3.WINDOWS); HOLDS = [0.0] + list(R3.HOLDS)
D_OFF = 0.5 * sum(R3.LAT_MS) + 0.5 * R3.RAMP_MS
GRID = np.linspace(0, R3.GRID_MAX, int(round(R3.GRID_MAX * 100)) + 1)
F = lambda q, x: float(np.interp(x, q, np.linspace(0, 1, 101), left=0, right=1))
G = lambda gl, xT: float(np.interp(xT, GRID, gl))
MS = ('mOB', 'mOB2', 'mBm', 'mB', 'mBabs')
KEYS = [R3.cond_key(*c) for c in R3.CONDS]
KEYS_NA = [R3.cond_key(*c) for c in R3.CONDS_NA]


def pseudo(dist, T):
    """a pseudo-sample of 2000 durations from the 1% quantiles, summarised as in the registered run"""
    if dist is None: return None
    q = np.asarray(dist['q'])
    return R3.dist_summary(np.interp(np.linspace(0, 100, 2000), np.arange(101), q), T)


def build(S2):
    S = dict(T=list(S2['T']), dL=list(S2['dL']), base=[], cond={}, dists={})
    for r in S2['base']:
        rows = []
        for x in r:
            y = dict(x)
            y.update(mOB=x['mBm'] * 1.08 if x['mBm'] == x['mBm'] else np.nan, mOA=x['mA'] * 1.1, nOB=x['nBm'],
                     mOB2=x['mBm'] * 1.12 if x['mBm'] == x['mBm'] else np.nan, mOA2=x['mA'] * 1.15, nOB2=x['nBm'],
                     nBabs=x['nB'], rB_dom=0.4, area_comp=D_OFF)
            rows.append(y)
        S['base'].append(rows)
    for k in ('base_all', 'base_h0'):
        S['dists'][k] = [pseudo(d, t) for d, t in zip(S2['dists'][k], S2['T'])]
    for k in ('ungated_0.2', 'ungated_0.1'):
        for h in ('all', 'h0'):
            S['dists'][f'{k}_{h}'] = [pseudo(d, t) for d, t in zip(S2['dists'][f'ungated_{h}'], S2['T'])]
    return S


base = dict(dry_run=True, n_drawn=275, n_drawn_na=100, n_drawn_hn=50,
            settings=dict(EARLY_W=R3.EARLY_W, WINDOWS=WIN, HOLDS=list(R3.HOLDS), LAT_MS=R3.LAT_MS,
                          RAMP_MS=R3.RAMP_MS, GRID_MAX=R3.GRID_MAX),
            results=dict(main=build(D2['results']['main']), noadapt=build(D2['results']['noadapt'])))
base['results']['highnoise'] = copy.deepcopy(base['results']['noadapt'])


def weight(acc, dist, Ti, a_ms, b_ms):
    if acc == 'handover': return F(dist['q'], b_ms) - F(dist['q'], a_ms)
    key = 'g' if acc == 'dose' else 'hz'
    return G(dist[key], b_ms / Ti) - G(dist[key], a_ms / Ti)


def scenario(acc, offset, seed):
    rng = np.random.default_rng(seed)
    dd = copy.deepcopy(base)
    for nm, keys in (('main', KEYS), ('noadapt', KEYS_NA), ('highnoise', KEYS_NA)):
        S = dd['results'][nm]
        for k in keys:
            S['cond'][k] = copy.deepcopy(S['base'])
        for i in range(len(S['T'])):
            common = rng.normal(0, 2.4)

            def setc(key, pct, att=None):
                for s in range(8):
                    eps = rng.normal(0, 10)
                    for m in MS:
                        S['cond'][key][i][s][m] = S['base'][i][s][m] * (1 + (pct + common + eps) / 100)
                    if att is not None:
                        for m in ('mOA', 'mA', 'mOA2'):
                            S['cond'][key][i][s][m] = S['base'][i][s][m] * (1 + att / 100)
            Ti, dLi = S['T'][i], S['dL'][i]; dist = S['dists']['base_all'][i]
            if nm != 'main':
                setc('gated_0.2', -6, att=40); setc('gated_lat_0.2', 0.5 if nm == 'noadapt' else 8, att=40)
                setc('rep_gated_0.2', 0.0 if nm == 'noadapt' else 6, att=40); setc('ungated_0.2', -20)
                setc('win_0.2_0-0.25', -4); setc('late_0.2', -4.5); setc('chong_0.1', -8 if nm == 'noadapt' else 5, att=25)
                continue
            if dist is None or not np.isfinite(dLi):
                continue
            s, u = 16 + rng.normal(0, 2), -32 + rng.normal(0, 2); e = s - u
            setc('gated_0.2', s); setc('ungated_0.2', u); setc('gated_lat_0.2', s + 5); setc('rep_gated_0.2', s + 6)
            for j, (a, b) in enumerate(WIN):
                setc(f'win_0.2_{a:g}-{b:g}', -0.6 * e * weight(acc, dist, Ti, a * Ti, b * Ti) + (offset if j < 2 else 0))
            setc('late_0.2', -0.6 * e * weight(acc, dist, Ti, dLi, 1e12))
            setc('early_ramp_0.2', -0.6 * e * weight(acc, dist, Ti, 0, 0.25 * Ti) + offset)
            s1, u1 = 8 + rng.normal(0, 1), -15 + rng.normal(0, 1); e1 = s1 - u1
            setc('gated_0.1', s1); setc('ungated_0.1', u1)
            du = S['dists']['ungated_0.1_h0'][i]
            if du is None: continue
            sc = 5 + e1 * weight(acc, du, Ti, 0, D_OFF)
            for h in HOLDS:
                setc('chong_0.1' if h == 0 else f'chong_0.1_x{h:g}', sc - e1 * weight(acc, du, Ti, 0, D_OFF + h * Ti), att=25)
            setc('early_rep_0.1', -2); setc('late_rep_0.1', -6)
    return dd


summary = {}
for acc, off in (('handover', 0), ('hazard', 0), ('dose', 0), ('handover', 4), ('hazard', 4)):
    name = f'{acc}{"_offset" if off else ""}'
    dd = scenario(acc, off, seed=zlib.crc32(name.encode()))
    json.dump(dd, open(f'synth3_{name}.json', 'w'), default=float)
    subprocess.run([sys.executable, 'analyse_li2017_test3.py', f'synth3_{name}.json', f'synth3_{name}_tests.json'],
                   capture_output=True)
    t = json.load(open(f'synth3_{name}_tests.json'))
    h14 = t['H14']
    summary[name] = dict(
        H14_five_vs_hazard=(h14['five_windows']['vs_hazard']['k'], h14['five_windows']['vs_hazard']['n']),
        H14_late_vs_hazard=(h14['late_windows']['vs_hazard']['k'], h14['late_windows']['vs_hazard']['n']),
        H14_five_vs_dose=(h14['five_windows']['vs_dose']['k'], h14['five_windows']['vs_dose']['n']),
        H14_holds=h14['holds'], H14_holds_vs_dose=h14['holds_vs_dose'], H13=t['H13']['holds'], H17=t['H17']['holds'],
        H21_vs_dose=(t['H21']['vs_dose']['k'], t['H21']['vs_dose']['n']), H21_holds=t['H21']['holds'],
        H21_vs_hazard=(t['H21']['vs_hazard']['k'], t['H21']['vs_hazard']['n'], t['H21']['vs_hazard']['n_not_separated']))
    print(name, summary[name], flush=True)
json.dump(summary, open('synth3_summary.json', 'w'), indent=1)
