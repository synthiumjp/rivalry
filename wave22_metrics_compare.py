"""
wave22_metrics_compare.py — dynamics disagreement, or statistic disagreement?

The CV audit established three things:

  1. phase1_grid_results.json is the file manuscript 4.3 was computed from
     (6,814 rivalry-producing, median CV 0.563, 13.7% in [0.35,0.65], 25.2% in
     [0.25,0.75] -- all exact).
  2. phase1_cv in the phase2 records is the grid's own measurement, matching to
     +/-0.000 across all 30. It is not a stored selection target.
  3. Median CV rises monotonically with sigma (0.183, 0.337, 0.506, 0.639,
     0.760), so the stored statistic does respond to noise as it must.

What remains is a single localised disagreement. The grid's median CV at
gamma = 0.050 is 0.890, against 0.427 and 0.485 at gamma = 0.020 and 0.030. An
independent kernel gives CV 0.088 and 0.090 for the two gamma = 0.050
configurations in the Phase II set, unchanged across dominance thresholds from 0
to 0.10. Their durations, 35.1 and 34.8 timesteps at ~545 switches per run, are
those of a regular relaxation oscillator, which is what low noise plus fast
adaptation decay should produce.

THE QUESTION THIS SCRIPT ANSWERS
--------------------------------
Two possibilities, with different consequences:

  (A) DYNAMICS DIFFER. The grid's mean durations and switch counts for these
      configurations disagree with the independent kernel too. Then the two
      state updates are not the same model, and wave22 cannot proceed until the
      difference is found. Nothing about the frozen results is impugned.

  (B) ONLY THE STATISTIC DIFFERS. The grid's mean durations and switch counts
      agree, but its CV does not. Then both kernels are simulating the same
      dynamics and the disagreement is in how variability is computed from the
      episode list. Since the 30-config Phase II set was selected on that
      statistic, and manuscript 4.2 (Props I and III), 4.3, 4.4 and
      pre-registered prediction 2 all rest on that set, this is a finding about
      the frozen results rather than an integration detail.

The discriminator is mean duration, not CV. Mean duration is insensitive to how
the variance is computed and sensitive to the dynamics.

USAGE
    python wave22_metrics_compare.py
    python wave22_metrics_compare.py --seeds 30 --steps 20000   # match phase1
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

try:
    from wave22_adaptation import (as_list, params_from, run_cell, sm, G_NONE,
                                   BLK_V, THETA, O_DURA, O_DURB, O_CVA, O_CVB,
                                   O_SW, O_SWF, O_NA, O_NB, O_ACTA, O_FLOOR,
                                   O_PREDA)
    from wave22_cv_audit import fnum
except ImportError as e:
    sys.exit(f"run from C:\\crewther with wave22_adaptation.py and "
             f"wave22_cv_audit.py present ({e})")


def flatten(obj, prefix=''):
    """All scalar leaves of a nested dict, as prefix.key -> float."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}{k}."))
    else:
        v = fnum(obj)
        if np.isfinite(v):
            out[prefix.rstrip('.')] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=30)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--n', type=int, default=30)
    args = ap.parse_args()

    with open('phase1_grid_results.json') as f:
        rows = as_list(json.load(f), 'results')
    with open('phase2_dissociation_results.json') as f:
        ph2 = as_list(json.load(f), 'config_results')

    key = lambda p: tuple(round(p[q], 6) for q in
                          ('lam', 'beta', 'alpha', 'sigma', 'gam', 'kap'))
    index = {}
    for i, r in enumerate(rows):
        try:
            index.setdefault(key(params_from(r, i)), r)
        except KeyError:
            continue

    # what does the grid actually store per configuration?
    sample = index.get(key(params_from(ph2[0], 0)))
    if sample is None:
        sys.exit("first phase2 configuration not found in the grid by parameters")
    fields = flatten(sample)
    print("=== stored numeric fields per grid record ===")
    for k, v in sorted(fields.items()):
        print(f"  {k:<40} {v:>12.4f}")

    print(f"\n=== 30 selected configurations, {args.seeds} seeds x "
          f"{args.steps} steps, theta = {THETA} ===")
    print("  mean duration is the discriminator. CV is the disputed quantity.\n")
    print(f"  {'cfg':>4} {'sig':>5} {'gam':>5} | {'gridCV':>7} {'mineCV':>7} "
          f"{'ratio':>6} | {'gridDur':>8} {'mineDur':>8} {'ratio':>6} | "
          f"{'gridSw':>7} {'mineSw':>7} {'ratio':>6}")
    print("  " + "-" * 104)

    out = []
    dur_ratio, cv_ratio, sw_ratio = [], [], []
    for i, c in enumerate(ph2[:args.n]):
        p = params_from(c, i)
        g = index.get(key(p))
        if g is None:
            print(f"  {i:>4}  no grid match")
            continue
        gf = flatten(g)

        def pick(*names):
            for n in names:
                for k in gf:
                    if k.lower().endswith(n):
                        return gf[k]
            return float('nan')

        gcv = pick('cv', 'duration_cv')
        gdur = pick('mean_duration', 'duration_mean', 'mean_dur', 'dur_mean',
                    'mean_dominance_duration')
        gsw = pick('n_switches', 'switches', 'switch_count', 'mean_switches',
                   'switches_per_seed')

        m = run_cell(p, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0,
                     args.steps, args.seeds, BLK_V, 0)
        mcv = 0.5 * (sm(m, O_CVA) + sm(m, O_CVB))
        mdur = 0.5 * (sm(m, O_DURA) + sm(m, O_DURB))
        msw = sm(m, O_SW)

        rc = mcv / gcv if (np.isfinite(gcv) and gcv > 0) else float('nan')
        rd = mdur / gdur if (np.isfinite(gdur) and gdur > 0) else float('nan')
        rs = msw / gsw if (np.isfinite(gsw) and gsw > 0) else float('nan')
        for lst, val in ((cv_ratio, rc), (dur_ratio, rd), (sw_ratio, rs)):
            if np.isfinite(val):
                lst.append(val)

        print(f"  {int(c.get('config_idx', i)):>4} {p['sigma']:>5.3f} "
              f"{p['gam']:>5.3f} | {gcv:>7.3f} {mcv:>7.3f} {rc:>6.2f} | "
              f"{gdur:>8.1f} {mdur:>8.1f} {rd:>6.2f} | "
              f"{gsw:>7.1f} {msw:>7.1f} {rs:>6.2f}")
        out.append(dict(config=int(c.get('config_idx', i)), params=p,
                        grid={'cv': gcv, 'dur': gdur, 'sw': gsw},
                        mine={'cv': mcv, 'dur': mdur, 'sw': msw},
                        grid_all=gf))

    def summarise(name, vals, tol=0.10, diagnostic=True):
        if not vals:
            print(f"  {name}: not stored in the grid, cannot compare")
            return None
        v = np.array(vals)
        within = int(np.sum(np.abs(v - 1.0) <= tol))
        note = '' if diagnostic else '   (not diagnostic, see note)'
        print(f"  {name}: median ratio {np.median(v):.3f}, "
              f"within {int(100*tol)}% in {within}/{len(v)}{note}")
        return float(np.median(v))

    print("\n=== mine / grid ===")
    md = summarise("mean duration", dur_ratio)
    ms = summarise("switches    ", sw_ratio, diagnostic=False)
    mc = summarise("CV          ", cv_ratio)
    print(f"\n  Note: raw switch counts scale with run length. This run used "
          f"{args.steps} steps;\n  the grid's stored count is for whatever length "
          f"phase1 used. Only the duration\n  and CV ratios are comparable. Pass "
          f"--steps to match phase1 if you want the\n  switch column to mean "
          f"anything.")

    print("\n=== VERDICT ===")
    if md is None:
        print("""  Mean duration is not stored in the grid, so this script cannot
  discriminate. Print the keys listed at the top of this output and tell me which
  of them is a duration; if none is, the grid discarded everything except CV and
  levelt_rho, and the comparison has to be made against wave2_A_h4.json or
  wave6_G_dissociation.json instead.""")
    elif abs(md - 1.0) <= 0.10:
        print(f"""  DURATIONS AGREE (median ratio {md:.3f}) WHILE CV DOES NOT
  (median ratio {mc:.3f}).

  Case (B). The two kernels are simulating the same dynamics, so the
  disagreement is in how variability is computed from the episode list, not in
  the model. Since the Phase II set was selected on that statistic, this needs
  resolving on its own terms and ahead of wave22.

  The one thing that would explain it: CV computed by pooling all episodes from
  all seeds without removing between-seed differences in mean. For a regular
  oscillator each seed settles onto the same period, so pooling adds little --
  unless the seeds are not independent. Note that wave22_A_h4.json and the
  original phase2 run both carry documented seed-collision defects (handover
  section 3, bugs 1 and 2). If phase1 shares that seeding scheme, pooled CV
  would be inflated exactly where the dynamics are most regular, which is the
  gamma = 0.050 cell.

  Check gc_lca_phase1_grid.py for two things: how the per-seed trial seed is
  formed, and whether CV is computed per seed then averaged, or on pooled
  episodes.""")
    else:
        print(f"""  DURATIONS DISAGREE (median ratio {md:.3f}).

  Case (A). The two state updates are not the same model, so nothing is impugned
  about the frozen results and wave22 cannot proceed until the difference is
  found. The disagreement is in the update, not the bookkeeping.

  Paste the update loop from gc_lca_phase1_grid.py, or wave2_campaign's
  run_trace and _rectify. Roughly 20 lines settles it.""")

    with open('wave22_metrics_compare.json', 'w') as f:
        json.dump(dict(seeds=args.seeds, steps=args.steps, theta=THETA,
                       median_ratio=dict(duration=md, switches=ms, cv=mc),
                       rows=out), f, indent=1, default=float)
    print("\nwrote wave22_metrics_compare.json")


if __name__ == '__main__':
    main()
