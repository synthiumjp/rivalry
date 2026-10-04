"""
analysis_einhauser.py - the registered analysis for osf.io/3rthv.

Tests two predictions of the dominance-gating account of attention in binocular rivalry on the open
dataset of Einhauser, Sandrock and Schutz (2021), Zenodo record 4575552.

  H1  The competitor's duration change, relative to the pre-task blocks, is more positive in the task
      phase (gated) than in the post-task phase (ungated): Delta_O,task - Delta_O,post > 0.
  H2  The task-phase coupling ratio Delta_O,task / Delta_T,task is negatively rank-correlated with the
      observer's mean pre-task dominance duration.

Implements Sections 4 and 5 of the registration and the clarifications in the design note of
4 October 2026 (einhauser_design_note.md). Registered results are printed first; everything after
them is labelled exploratory.

USAGE
    python analysis_einhauser.py --data allData.mat
    python analysis_einhauser.py --selftest        # synthetic data only, for checking the code

The registration requires this file to be committed before it is run on the real data.
"""

import argparse
import json
import sys

import numpy as np
from scipy import stats
from scipy.io import loadmat

PHASES = {'pre': (range(0, 4), 8), 'task': (range(4, 12), 16), 'post': (range(12, 16), 8)}
REG = dict(thr=0.3, min_ep=300, max_gap=300, max_nan=0.5, min_n=10, floor=5.0)
B_BOOT, SEED = 10000, 0


# ----------------------------------------------------------------------------- dominance extraction
def sample_states(gain, dir_blue, thr):
    """+1 blue dominant, -1 red dominant, 0 indeterminate, 9 missing (NaN)."""
    g = np.asarray(gain, float).ravel()
    s = np.zeros(g.size, np.int8)
    with np.errstate(invalid='ignore'):
        v = g * dir_blue
        s[v >= thr] = 1
        s[v <= -thr] = -1
    s[~np.isfinite(g)] = 9
    return s


def fill_missing(s, max_gap):
    """NaN gaps shorter than max_gap take the preceding state; longer ones become indeterminate."""
    out = s.copy()
    n, i = len(s), 0
    while i < n:
        if out[i] == 9:
            j = i
            while j < n and s[j] == 9:
                j += 1
            out[i:j] = out[i - 1] if (i > 0 and (j - i) < max_gap) else 0
            i = j
        else:
            i += 1
    return out


def episodes(s, max_gap):
    """Maximal runs of one state, allowing indeterminate gaps shorter than max_gap.
    Returns (state, start, end) in samples (ms)."""
    runs, n, i = [], len(s), 0
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        runs.append([int(s[i]), i, j])
        i = j
    merged = []
    for r in runs:
        if r[0] == 0:
            merged.append(r)
            continue
        if (len(merged) >= 2 and merged[-1][0] == 0 and merged[-1][2] - merged[-1][1] < max_gap
                and merged[-2][0] == r[0]):
            merged.pop()
            merged[-1][2] = r[2]
        elif merged and merged[-1][0] == r[0]:
            merged[-1][2] = r[2]
        else:
            merged.append(r)
    return [tuple(m) for m in merged if m[0] != 0]


def trial_durations(gain, dir_blue, p):
    """Valid dominance durations in one trial as (colour, ms), or None if the trial is excluded."""
    g = np.asarray(gain, float).ravel()
    if g.size == 0 or np.mean(~np.isfinite(g)) > p['max_nan']:
        return None
    eps = episodes(fill_missing(sample_states(g, dir_blue, p['thr']), p['max_gap']), p['max_gap'])
    eps = eps[1:-1]                                   # first and last are truncated by the trial
    return [('blue' if st == 1 else 'red', e - b) for st, b, e in eps if e - b >= p['min_ep']]


# ----------------------------------------------------------------------------- per observer
def observer_table(D, s, p, pre_response=False):
    """Per phase: mean duration and episode count for T and O on exactly-one-hard trials, the mean
    on symmetric trials, and the mean of all episodes on all trials."""
    okn, db, dr, eb = D['OKNrawGain'], D['diffBlue'], D['diffRed'], D['eyeBlue']
    out = {}
    for ph, (blocks, n_tr) in PHASES.items():
        role = {'T': [], 'O': []}
        sym, every = [], []
        for b in blocks:
            for t in range(n_tr):
                gain = np.asarray(okn[s, b, t], float).ravel()
                if pre_response and ph == 'task':
                    ib = D['idxButton'][s, b, t]
                    if not np.isfinite(ib):
                        continue
                    gain = gain[:int(ib)]
                dir_blue = 1.0 if eb[s, b, t] == 1 else -1.0
                durs = trial_durations(gain, dir_blue, p)
                if durs is None:
                    continue
                every += [d for _, d in durs]
                hb, hr = db[s, b, t] == 2, dr[s, b, t] == 2
                if hb != hr:
                    hard = 'blue' if hb else 'red'
                    for c, d in durs:
                        role['T' if c == hard else 'O'].append(d)
                else:
                    sym += [d for _, d in durs]
        out[ph] = dict(T=float(np.mean(role['T'])) if role['T'] else np.nan,
                       O=float(np.mean(role['O'])) if role['O'] else np.nan,
                       nT=len(role['T']), nO=len(role['O']),
                       sym=float(np.mean(sym)) if sym else np.nan,
                       all=float(np.mean(every)) if every else np.nan)
    return out


def deltas(tab, drift=False):
    def norm(ph, r):
        v = tab[ph][r]
        return v / tab[ph]['sym'] if drift else v
    d = {}
    for ph in ('task', 'post'):
        for r in ('T', 'O'):
            base = norm('pre', r)
            d[f'{r}_{ph}'] = 100.0 * (norm(ph, r) - base) / base if np.isfinite(base) and base > 0 else np.nan
    return d


# ----------------------------------------------------------------------------- statistics
def boot_ci(x, fn, rng):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (B_BOOT, len(x)))
    return np.percentile([fn(x[i]) for i in idx], [2.5, 97.5])


def boot_ci_pair(x, y, fn, rng):
    x, y = np.asarray(x, float), np.asarray(y, float)
    idx = rng.integers(0, len(x), (B_BOOT, len(x)))
    return np.percentile([fn(x[i], y[i]) for i in idx], [2.5, 97.5])


def run(D, p, label, drift=False, pre_response=False, floor=True, verbose=True):
    rng = np.random.default_rng(SEED)
    rows = []
    for s in range(D['OKNrawGain'].shape[0]):
        tab = observer_table(D, s, p, pre_response=pre_response)
        rows.append(dict(obs=s + 1, tab=tab, d=deltas(tab, drift=drift)))

    def enough(tab, phases):
        return all(tab[ph]['nT'] >= p['min_n'] and tab[ph]['nO'] >= p['min_n'] for ph in phases)

    res = dict(label=label, params=dict(p), drift=drift, pre_response=pre_response, floor=floor)

    # ---- H1
    h1 = [r for r in rows if enough(r['tab'], ('pre', 'task', 'post'))
          and all(np.isfinite(r['d'][k]) for k in ('T_task', 'T_post', 'O_task', 'O_post'))]
    res['H1_n'] = len(h1)
    if len(h1) >= 5:
        Tt = np.array([r['d']['T_task'] for r in h1]); Tp = np.array([r['d']['T_post'] for r in h1])
        Ot = np.array([r['d']['O_task'] for r in h1]); Op = np.array([r['d']['O_post'] for r in h1])
        diff = Ot - Op
        res['precondition'] = dict(med_T_task=float(np.median(Tt)), med_T_post=float(np.median(Tp)),
                                   met=bool(np.median(Tt) > 0 and np.median(Tp) > 0))
        res['H1'] = dict(median_diff=float(np.median(diff)),
                         ci=boot_ci(diff, np.median, rng).tolist(),
                         p=float(stats.wilcoxon(diff, alternative='greater').pvalue))
        res['H1a'] = dict(median_O_task=float(np.median(Ot)), ci=boot_ci(Ot, np.median, rng).tolist())
        res['H1b'] = dict(median_O_post=float(np.median(Op)), ci=boot_ci(Op, np.median, rng).tolist())
        res['T_changes'] = dict(T_task_ci=boot_ci(Tt, np.median, rng).tolist(),
                                T_post_ci=boot_ci(Tp, np.median, rng).tolist())

    # ---- H2
    h2 = [r for r in rows if enough(r['tab'], ('pre', 'task'))
          and np.isfinite(r['d']['T_task']) and np.isfinite(r['d']['O_task'])
          and np.isfinite(r['tab']['pre']['all'])
          and (r['d']['T_task'] > p['floor'] if floor else r['d']['T_task'] != 0)]
    res['H2_n'] = len(h2)
    if len(h2) >= 5:
        ratio = np.array([r['d']['O_task'] / r['d']['T_task'] for r in h2])
        dpre = np.array([r['tab']['pre']['all'] for r in h2])
        sp = stats.spearmanr(ratio, dpre, alternative='less')
        rho_fn = lambda a, b: stats.spearmanr(a, b).correlation
        res['H2'] = dict(rho=float(sp.correlation), p=float(sp.pvalue),
                         ci=boot_ci_pair(ratio, dpre, rho_fn, rng).tolist(),
                         median_ratio=float(np.median(ratio)))

    # ---- Holm across the two primary tests
    ps = [(k, res[k]['p']) for k in ('H1', 'H2') if k in res]
    if len(ps) == 2:
        (k1, p1), (k2, p2) = sorted(ps, key=lambda kv: kv[1])
        rej1 = p1 <= 0.025
        res['holm'] = {k1: bool(rej1), k2: bool(rej1 and p2 <= 0.05)}
    res['per_observer'] = [dict(obs=r['obs'], **{k: (None if not np.isfinite(v) else round(v, 3))
                                                 for k, v in r['d'].items()},
                                pre_all_ms=r['tab']['pre']['all'],
                                n=[r['tab'][ph][x] for ph in PHASES for x in ('nT', 'nO')])
                           for r in rows]
    if verbose:
        report(res)
    return res


def report(r):
    print(f"\n{'=' * 78}\n{r['label']}\n{'=' * 78}")
    if 'precondition' in r:
        pc = r['precondition']
        print(f"  precondition: median Delta_T,task {pc['med_T_task']:+.1f}%, Delta_T,post "
              f"{pc['med_T_post']:+.1f}%  -> {'MET' if pc['met'] else 'NOT MET: H1 is uninformative'}")
    if 'H1' in r:
        h = r['H1']
        print(f"  H1  n = {r['H1_n']}: median Delta_O,task - Delta_O,post = {h['median_diff']:+.1f} pp "
              f"[{h['ci'][0]:+.1f}, {h['ci'][1]:+.1f}], Wilcoxon one-sided p = {h['p']:.4f}")
        print(f"      H1a median Delta_O,task {r['H1a']['median_O_task']:+.1f}% "
              f"[{r['H1a']['ci'][0]:+.1f}, {r['H1a']['ci'][1]:+.1f}]")
        print(f"      H1b median Delta_O,post {r['H1b']['median_O_post']:+.1f}% "
              f"[{r['H1b']['ci'][0]:+.1f}, {r['H1b']['ci'][1]:+.1f}]")
    else:
        print(f"  H1  too few observers ({r['H1_n']})")
    if 'H2' in r:
        h = r['H2']
        print(f"  H2  n = {r['H2_n']}: Spearman rho = {h['rho']:+.3f} [{h['ci'][0]:+.3f}, "
              f"{h['ci'][1]:+.3f}], one-sided p = {h['p']:.4f}; median ratio {h['median_ratio']:+.3f}")
    else:
        print(f"  H2  too few observers ({r['H2_n']})")
    if 'holm' in r:
        print(f"  Holm, family-wise .05: " + ", ".join(f"{k} {'supported' if v else 'not supported'}"
                                                    for k, v in r['holm'].items()))


# ----------------------------------------------------------------------------- data
def load(path):
    D = loadmat(path, variable_names=['OKNrawGain', 'diffBlue', 'diffRed', 'eyeBlue', 'eyeRed', 'idxButton'])
    return {k: D[k] for k in ('OKNrawGain', 'diffBlue', 'diffRed', 'eyeBlue', 'eyeRed', 'idxButton')}


def synthetic(seed=1, n_obs=24, trial_ms=20000, effects=None):
    """Same structure as allData.mat, invented values. For testing the code only.
    effects maps (phase, role) to a duration multiplier on exactly-one-hard trials, where role is
    'T' (the hard stimulus) or 'O' (the easy one). Each observer keeps one baseline throughout."""
    effects = effects or {('task', 'T'): 1.4}
    rng = np.random.default_rng(seed)
    shape = (n_obs, 16, 16)
    okn = np.empty(shape, dtype=object)
    db = np.ones(shape); dr = np.ones(shape); eb = np.ones(shape); idx = np.full(shape, np.nan)
    for s in range(n_obs):
        base = rng.uniform(1500, 4000)
        for b in range(16):
            n_tr = 16 if 4 <= b < 12 else 8
            combos = np.repeat(np.arange(4), n_tr // 4); rng.shuffle(combos)
            for t in range(16):
                if t >= n_tr:
                    okn[s, b, t] = np.zeros((0, 0))
                    continue
                db[s, b, t] = 1 + combos[t] % 2; dr[s, b, t] = 1 + combos[t] // 2
                eb[s, b, t] = 1 + rng.integers(0, 2)
                dirb = 1.0 if eb[s, b, t] == 1 else -1.0
                g = np.empty(trial_ms); k, col = 0, rng.integers(0, 2)
                ph = 'pre' if b < 4 else ('task' if b < 12 else 'post')
                hb, hr = db[s, b, t] == 2, dr[s, b, t] == 2
                while k < trial_ms:
                    mine, theirs = (hb, hr) if col == 0 else (hr, hb)
                    boost = 1.0
                    if mine != theirs:
                        boost = effects.get((ph, 'T' if mine else 'O'), 1.0)
                    d = int(rng.gamma(4, base * boost / 4))
                    sign = dirb if col == 0 else -dirb
                    g[k:k + d] = sign * 0.7 + rng.normal(0, 0.15, size=len(g[k:k + d]))
                    k += d; col = 1 - col
                for _ in range(trial_ms // 400):
                    a = rng.integers(0, trial_ms - 60); g[a:a + 50] = np.nan
                okn[s, b, t] = g[None, :]
                if 4 <= b < 12:
                    idx[s, b, t] = rng.integers(trial_ms // 3, trial_ms)
    eyeRed = 3 - eb
    return dict(OKNrawGain=okn, diffBlue=db, diffRed=dr, eyeBlue=eb, eyeRed=eyeRed, idxButton=idx)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='allData.mat')
    ap.add_argument('--out', default='einhauser_results.json')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        print("SELF-TEST ON SYNTHETIC DATA. These numbers mean nothing about the real dataset.")
        D = synthetic()
    else:
        D = load(a.data)
    results = {'registered': run(D, REG, 'REGISTERED ANALYSIS (osf.io/3rthv)')}
    print("\n\nEverything below is exploratory, as registered in Section 5 and the design note.")
    variants = [('gain threshold 0.2', dict(REG, thr=0.2), {}),
                ('gain threshold 0.5', dict(REG, thr=0.5), {}),
                ('minimum episode 200 ms', dict(REG, min_ep=200), {}),
                ('minimum episode 500 ms', dict(REG, min_ep=500), {}),
                ('H2 without the 5% floor', REG, dict(floor=False)),
                ('drift-corrected by symmetric trials (design note)', REG, dict(drift=True)),
                ('task phase before the button press (design note)', REG, dict(pre_response=True))]
    for lab, p, kw in variants:
        results[lab] = run(D, p, 'EXPLORATORY: ' + lab, **kw)
    if not a.selftest:
        json.dump(results, open(a.out, 'w'), indent=1, default=float)
        print(f"\nwrote {a.out}")


if __name__ == '__main__':
    main()
