"""
wave22_cv_audit.py — is the frozen phase1 CV correct?

No simulation. Reads phase1_grid_results.json and phase2_dissociation_results.json
and asks whether the stored coefficient of variation behaves the way it physically
must.

WHY THIS EXISTS
---------------
Block V of wave22_adaptation.py found that two of the first eight Phase II
configurations carry phase1_cv = 0.500 and 0.501 while an independent kernel gives
0.088 and 0.090 for the same parameters. Those two are the only ones in the sample
with gamma = 0.050 (highest grid value) and sigma = 0.040-0.060 (lowest two). A
sweep of the dominance threshold from 0 to 0.10 moves their CV by less than 0.06,
so the disagreement is not an episode-extraction convention.

Low noise plus fast adaptation decay gives a near-deterministic relaxation
oscillator. Its dominance durations are regular by construction, so CV near 0.09
is what the dynamics permit and CV near 0.50 is not reachable by any extraction
rule. That makes the stored value the thing under suspicion.

The test below does not depend on any reimplementation. Duration variability in
this model comes from noise. CV must therefore increase with sigma. If the stored
CV is flat in sigma, or if low-sigma configurations cluster at CV ~ 0.5, the grid's
CV computation is wrong -- and since the 30 Phase II configurations were SELECTED
on proximity of that value to 0.5, the selection is then contaminated.

WHAT DEPENDS ON THE ANSWER
--------------------------
The 30 selected configurations carry manuscript 4.2 (Props I and III: 29/30,
25/30, 21/30), 4.3, 4.4 (volitional control, 5/30 strong support) and
pre-registered prediction 2. The 762-config eligible pool is filtered on the same
statistic.

USAGE
    python wave22_cv_audit.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

try:
    from wave22_adaptation import as_list, params_from, ALIASES
except ImportError:
    sys.exit("run from C:\\crewther with wave22_adaptation.py present")


def fnum(v, default=float('nan')):
    """Coerce to float, tolerating None, '', and non-numeric.

    phase1 stores levelt_rho as null for configurations that did not produce
    rivalry, and any of these fields may be absent or null in a given record.
    """
    if v is None or isinstance(v, (dict, list)):
        return default
    if isinstance(v, bool):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def find_cv(rec):
    """Locate the stored CV, wherever the grid put it."""
    for path in (('rivalry_metrics', 'cv'), ('rivalry_metrics', 'duration_cv'),
                 ('metrics', 'cv'), ('cv',), ('duration_cv',)):
        node = rec
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        v = fnum(node)
        if np.isfinite(v):
            return v
    return float('nan')


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return float('nan')
    rx = np.argsort(np.argsort(x[ok]))
    ry = np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def band(v, lo, hi):
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    return float(np.mean((v[ok] >= lo) & (v[ok] <= hi))) if ok.any() else float('nan')


def main():
    # ---------------- phase1 grid ----------------
    with open('phase1_grid_results.json') as f:
        grid_raw = json.load(f)
    rows = as_list(grid_raw, 'results')
    print(f"phase1_grid_results.json: {len(rows)} records\n")

    recs = []
    for i, r in enumerate(rows):
        try:
            p = params_from(r, i)
        except KeyError:
            continue
        p['cv'] = find_cv(r)
        p['rivalry'] = bool(r.get('rivalry_producing') or False) \
            if 'rivalry_producing' in r else True
        p['rho'] = fnum(r.get('levelt_rho'))
        recs.append(p)

    riv = [r for r in recs if r['rivalry']]
    cv = np.array([r['cv'] for r in riv], float)
    if not np.isfinite(cv).any():
        sys.exit("no CV field found in phase1 records. Print one record's keys:\n"
                 f"  {sorted(rows[0])}")

    n_cv = int(np.isfinite(cv).sum())
    print(f"  CV present for {n_cv}/{len(riv)} rivalry-producing records"
          + ("" if n_cv == len(riv) else "  <-- gaps, interpret with care"))
    n_rho = int(np.isfinite([r['rho'] for r in recs]).sum())
    print(f"  levelt_rho present for {n_rho}/{len(recs)} records "
          f"(null for non-rivalry cells is expected)\n")

    print("=== 1. distribution against the manuscript ===")
    print(f"  rivalry-producing        {len(riv)}       (4.1 reports 6,814)")
    print(f"  median CV                {np.nanmedian(cv):.3f}   (4.3 reports 0.563)")
    print(f"  fraction in [0.35,0.65]  {100*band(cv,0.35,0.65):.1f}%   "
          f"(4.3 reports 13.7%)")
    print(f"  fraction in [0.25,0.75]  {100*band(cv,0.25,0.75):.1f}%   "
          f"(4.3 reports 25.2%)")

    print("\n=== 2. THE TEST: does CV rise with noise? ===")
    print("  Duration variability in this model is noise-driven. CV must increase")
    print("  with sigma. If it does not, the stored statistic is not CV.\n")
    by_sig = defaultdict(list)
    for r in riv:
        by_sig[round(r['sigma'], 4)].append(r['cv'])
    print(f"  {'sigma':>7} {'n':>6} {'medCV':>7} {'IQR':>17} {'frac in [.45,.55]':>18}")
    for s in sorted(by_sig):
        v = np.array(by_sig[s], float)
        q1, q3 = np.nanpercentile(v, [25, 75])
        print(f"  {s:>7.3f} {len(v):>6} {np.nanmedian(v):>7.3f} "
              f"  [{q1:>6.3f}, {q3:>6.3f}] {100*band(v,0.45,0.55):>16.1f}%")
    rho_sig = spearman([r['sigma'] for r in riv], [r['cv'] for r in riv])
    print(f"\n  Spearman(sigma, CV) = {rho_sig:+.3f}")
    if not np.isfinite(rho_sig):
        print("  UNINTERPRETABLE")
    elif rho_sig > 0.3:
        print("  Positive as required. The stored CV behaves like a CV, so the")
        print("  disagreement on configs 0 and 4 is specific to those cells rather")
        print("  than a systematic defect. Go to section 4.")
    else:
        print("  *** NOT POSITIVE. Duration variability cannot be independent of")
        print("  *** the noise amplitude that generates it. The stored phase1 CV is")
        print("  *** not measuring what the manuscript says it measures, and the")
        print("  *** 30-config selection and the 762-config eligible pool are both")
        print("  *** filtered on it. This supersedes the wave22 question.")

    print("\n=== 3. CV against adaptation decay ===")
    by_gam = defaultdict(list)
    for r in riv:
        by_gam[round(r['gam'], 4)].append(r['cv'])
    print(f"  {'gamma':>7} {'n':>6} {'medCV':>7}")
    for g in sorted(by_gam):
        print(f"  {g:>7.3f} {len(by_gam[g]):>6} {np.nanmedian(by_gam[g]):>7.3f}")

    print("\n=== 4. the specific suspects: low sigma, high gamma ===")
    sus = [r for r in riv if r['sigma'] <= 0.06 and r['gam'] >= 0.05]
    oth = [r for r in riv if not (r['sigma'] <= 0.06 and r['gam'] >= 0.05)]
    if sus:
        sv = np.array([r['cv'] for r in sus], float)
        ov = np.array([r['cv'] for r in oth], float)
        print(f"  sigma<=0.06 and gamma>=0.05:  n={len(sus):>5}  "
              f"medCV {np.nanmedian(sv):.3f}  in [.45,.55] {100*band(sv,.45,.55):.1f}%")
        print(f"  all others:                   n={len(oth):>5}  "
              f"medCV {np.nanmedian(ov):.3f}  in [.45,.55] {100*band(ov,.45,.55):.1f}%")
        print("\n  An independent kernel puts this cell near CV 0.09 at any threshold.")
        print("  If the grid puts it near 0.5, that is the defect, localised.")

    # ---------------- phase2 cross-check ----------------
    print("\n=== 5. phase2 records against the grid they came from ===")
    try:
        with open('phase2_dissociation_results.json') as f:
            ph2 = as_list(json.load(f), 'config_results')
    except FileNotFoundError:
        print("  phase2_dissociation_results.json not found, skipped")
        ph2 = []

    if ph2:
        key = lambda p: tuple(round(p[q], 6) for q in
                              ('lam', 'beta', 'alpha', 'sigma', 'gam', 'kap'))
        index = {}
        for r in recs:
            index.setdefault(key(r), r)
        print(f"  {'cfg':>4} {'sig':>6} {'gam':>6} {'ph2 cv':>8} {'grid cv':>8} "
              f"{'delta':>8}  match")
        n_bad = 0
        for i, c in enumerate(ph2):
            try:
                p = params_from(c, i)
            except KeyError:
                continue
            g = index.get(key(p))
            stored = fnum(c.get('phase1_cv'))
            gcv = g['cv'] if g else float('nan')
            d = stored - gcv
            if g is None:
                flag = 'NO GRID MATCH'
            elif not np.isfinite(stored):
                flag = 'ph2 cv null'
            elif not np.isfinite(gcv):
                flag = 'grid cv null'
            elif abs(d) < 0.01:
                flag = 'OK'
            else:
                flag = 'MISMATCH'
                n_bad += 1
            print(f"  {int(c.get('config_idx', i)):>4} {p['sigma']:>6.3f} "
                  f"{p['gam']:>6.3f} {stored:>8.3f} {gcv:>8.3f} {d:>+8.3f}  {flag}")
        print(f"\n  {n_bad}/{len(ph2)} genuine mismatches between the phase2 record\n"
              f"  and the grid measurement for the same parameters.")
        if n_bad:
            print("  A mismatch means phase1_cv in the phase2 file is not the grid's")
            print("  own measurement -- it may be a stored selection target rather")
            print("  than a measured value, in which case block V was comparing")
            print("  against a constant.")

        sel = [params_from(c, i) for i, c in enumerate(ph2)]
        s_low = sum(1 for p in sel if p['sigma'] <= 0.06)
        print(f"\n  {s_low}/{len(sel)} selected configurations have sigma <= 0.06.")
        if s_low:
            print("  Those are the cells where an independent kernel cannot reach")
            print("  CV 0.5. If their stored CV is wrong, they entered the Phase II")
            print("  set on a spurious value, and 4.2, 4.3 and 4.4 rest on that set.")

    out = dict(n_rivalry=len(riv), median_cv=float(np.nanmedian(cv)),
               frac_035_065=band(cv, .35, .65), frac_025_075=band(cv, .25, .75),
               spearman_sigma_cv=rho_sig,
               by_sigma={str(s): float(np.nanmedian(by_sig[s])) for s in sorted(by_sig)},
               by_gamma={str(g): float(np.nanmedian(by_gam[g])) for g in sorted(by_gam)})
    with open('wave22_cv_audit.json', 'w') as f:
        json.dump(out, f, indent=1, default=float)
    print("\nwrote wave22_cv_audit.json")


if __name__ == '__main__':
    main()
