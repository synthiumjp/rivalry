"""
wave22_verify_claims.py — check three claims the manuscript now makes.

Two of these are load-bearing on text written after block D and were asserted
rather than tested. The third fills a [TO SUPPLY] marker from data already on disk.

CLAIM 1 (Section 4.7.7). The jointly-satisfying configurations are few but real.
    The conditional-rate table rests on denominators of 6 and 5 at low exponent, so
    four jointly-satisfying configurations at p = 1 could in principle be one region
    of parameter space sampled repeatedly. Check that they are distinct in
    parameters, and report how far apart they sit.

CLAIM 2 (Section 4.7.7). "We have not established whether the same configurations
    satisfy both at more than one exponent." True when written, and cheap to settle:
    the per-configuration rows carry config identifiers at every exponent.

CLAIM 3 (Section 4.7.7). "...or whether they retain the increasing-duration
    behaviour." This one needs simulation, but only on the handful of jointly
    satisfying configurations. If they DO retain it, the claim in the manuscript is
    unnecessarily defensive and should be strengthened. If they do not, the
    caveat stands and becomes concrete: a region satisfying two constraints while
    violating a third is not a solution.

MARKER (Section 4.7.4). The manuscript asserts that gated and ungated
    competitor-to-attended ratios are non-overlapping across the parameter space,
    but reports interquartile ranges, which do not establish that. The per-
    configuration rows in block C carry what is needed: sign counts and full ranges.

USAGE
    python wave22_verify_claims.py                    # inspection only
    python wave22_verify_claims.py --id-check         # adds the ID simulation
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations

import numpy as np

try:
    from wave22_adaptation import (load_pool, run_cell, sm, G_NONE, SPLIT_INDET,
                                   BLK_D_SWEEP, O_SW, O_SWF, O_ACTA, O_ACTB,
                                   O_DURA, O_CVA)
except ImportError as e:
    sys.exit(f"run from C:\\crewther ({e})")

PARAMS = ('lam', 'beta', 'alpha', 'sigma', 'gam', 'kap')
GRID_STEPS = dict(lam=0.04, beta=0.05, alpha=0.02, sigma=0.02, gam=0.01,
                  kap=0.02)   # approximate spacing, for "adjacent" judgements


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--joint', default='w22_wave22_D_joint.json')
    ap.add_argument('--gates', default='w22_wave22_C_gates.json')
    ap.add_argument('--tol', type=float, default=0.10)
    ap.add_argument('--id-check', action='store_true', dest='id_check')
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--steps', type=int, default=20000)
    args = ap.parse_args()

    D = load(args.joint)
    C = load(args.gates)
    CHONG = 0.310

    # ------------------------------------------------------------------
    # CLAIMS 1 and 2
    # ------------------------------------------------------------------
    if D is None:
        print(f"{args.joint} not found, skipping claims 1-3")
    else:
        print("=" * 76)
        print("CLAIMS 1 AND 2: are the jointly-satisfying configurations distinct,")
        print("and do any satisfy both constraints at more than one exponent?")
        print("=" * 76)

        both_by_p = {}
        for pk in sorted(D, key=float):
            rows = D[pk].get('rows', [])
            both = [r for r in rows
                    if r.get('levelt')
                    and np.isfinite(r.get('ratio_1x', float('nan')))
                    and abs(r['ratio_1x'] - CHONG) <= args.tol]
            both_by_p[float(pk)] = both

        print(f"\n  tolerance ±{args.tol} on the ratio at 1×\n")
        for p, both in sorted(both_by_p.items()):
            print(f"  p = {p:<5}  n_both = {len(both)}")
            if not both:
                continue
            print(f"    {'cfg':>6} {'lam':>6} {'beta':>6} {'alpha':>6} {'sigma':>6} "
                  f"{'gam':>6} {'kap':>7} {'ratio1x':>8} {'ratio2x':>8}")
            for r in sorted(both, key=lambda x: x['config']):
                print(f"    {r['config']:>6} {r['lam']:>6.3f} {r['beta']:>6.3f} "
                      f"{r['alpha']:>6.3f} {r['sigma']:>6.3f} {r['gam']:>6.3f} "
                      f"{r['kap']:>7.4f} {r['ratio_1x']:>8.3f} "
                      f"{r.get('ratio_2x', float('nan')):>8.3f}")
            # distinctness: how many grid steps apart is the closest pair?
            if len(both) > 1:
                dists = []
                for a, b in combinations(both, 2):
                    steps = sum(abs(a[k] - b[k]) / GRID_STEPS[k] for k in PARAMS)
                    dists.append((steps, a['config'], b['config']))
                dists.sort()
                print(f"    closest pair: configs {dists[0][1]} and {dists[0][2]}, "
                      f"~{dists[0][0]:.1f} grid steps apart "
                      f"(summed over six parameters)")
                if dists[0][0] < 2.0:
                    print("    -> ADJACENT. These are near-duplicates in parameter "
                          "space, so the\n       count overstates how many "
                          "independent regions satisfy both.")
                else:
                    print("    -> DISTINCT. Not a single region sampled repeatedly.")

        # cross-exponent persistence
        ids = {p: {r['config'] for r in b} for p, b in both_by_p.items()}
        allp = sorted(ids)
        persistent = set.intersection(*[ids[p] for p in allp]) if allp else set()
        print("\n  cross-exponent overlap of the jointly-satisfying sets:")
        for p in allp:
            others = [q for q in allp if q != p]
            shared = {c: [q for q in others if c in ids[q]] for c in ids[p]}
            multi = {c: v for c, v in shared.items() if v}
            print(f"    p = {p:<5}  {len(ids[p])} configs, "
                  f"{len(multi)} of which also satisfy both at another exponent"
                  + (f" ({', '.join(str(c) for c in sorted(multi))})" if multi else ""))
        print(f"\n    satisfying both at EVERY exponent: "
              f"{sorted(persistent) if persistent else 'none'}")
        if persistent:
            print("""    -> At least one configuration satisfies both constraints
       regardless of the exponent. That is a stronger statement than the
       manuscript makes and Section 4.7.7 should say it: the trade-off is a
       property of the population, not of every configuration in it.""")
        else:
            print("""    -> No configuration satisfies both at every exponent. The
       jointly-satisfying set is reconstituted from different configurations at
       each exponent, which supports reading the intersection as incidental
       rather than as a feasible region. Section 4.7.7 can say this directly
       instead of leaving it open.""")

    # ------------------------------------------------------------------
    # CLAIM 3
    # ------------------------------------------------------------------
    if D is not None and args.id_check:
        print("\n" + "=" * 76)
        print("CLAIM 3: do the jointly-satisfying configurations retain the")
        print("increasing-duration behaviour that Section 4.2 reports?")
        print("=" * 76)
        pool = {c['idx']: c for c in load_pool(False)}
        levels = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
        print(f"\n  equal-strength sweep S_A = S_B over {levels}")
        print(f"  Section 4.2: 97/100 configurations violate Modified Proposition IV")
        print(f"  (alternation rate FALLS with drive). Violation is the norm here,")
        print(f"  so retaining it is expected; NOT violating it would be the finding.\n")
        print(f"  {'p':>5} {'cfg':>6} {'rho(S, altrate)':>16} {'direction':>12}")
        out = []
        for p, both in sorted(both_by_p.items()):
            for r in both:
                c = pool.get(r['config'])
                if c is None:
                    continue
                bl = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps,
                              max(4, args.seeds // 2), BLK_D_SWEEP, 900,
                              split_indet=SPLIT_INDET)
                xbar = 0.5 * (sm(bl, O_ACTA) + sm(bl, O_ACTB))
                keff = c['kap'] / (xbar ** (p - 1.0)) if p != 1.0 else c['kap']
                cp = dict(c, kap=keff)
                alt = []
                for lv, s in enumerate(levels):
                    rr = run_cell(cp, p, s, s, 0.0, G_NONE, 0.0, args.steps,
                                  args.seeds, BLK_D_SWEEP, 910 + lv,
                                  split_indet=SPLIT_INDET)
                    alt.append(sm(rr, O_SWF))
                a = np.asarray(alt, float)
                if not np.isfinite(a).all():
                    continue
                rho = float(np.corrcoef(np.argsort(np.argsort(levels)),
                                        np.argsort(np.argsort(a)))[0, 1])
                dirn = 'ID (violates)' if rho < 0 else 'DD (satisfies)'
                print(f"  {p:>5.2f} {r['config']:>6} {rho:>16.3f} {dirn:>14}")
                out.append(dict(p=p, config=r['config'], rho=rho))
        if out:
            n_id = sum(1 for o in out if o['rho'] < 0)
            print(f"\n  {n_id}/{len(out)} of the jointly-satisfying configurations "
                  f"remain in the increasing-duration regime.")
            if n_id == len(out):
                print("""  -> All of them. So the jointly-satisfying set satisfies two
     constraints while violating a third, exactly as Section 4.7.7 cautions.
     The caveat is now concrete rather than hypothetical and should be stated
     as a result: no configuration in this study satisfies Levelt's second
     proposition, the attentional coupling ratio, and Modified Proposition IV
     together.""")
            elif n_id == 0:
                print("""  -> None of them. The jointly-satisfying configurations have
     ALSO left the increasing-duration regime, which no other configuration in
     this study does. That is a substantially stronger result than the
     manuscript claims and needs its own treatment: a small region satisfies
     three constraints that the rest of the parameter space cannot.""")
            else:
                print("""  -> Mixed. Report the split and treat the subset that
     satisfies all three as the interesting one, with the caveat that it is
     smaller still.""")

    # ------------------------------------------------------------------
    # MARKER for Section 4.7.4
    # ------------------------------------------------------------------
    if C is None:
        print(f"\n{args.gates} not found, skipping the ratio marker")
        return
    print("\n" + "=" * 76)
    print("MARKER (4.7.4): gated vs ungated ratio, sign counts and FULL ranges")
    print("=" * 76)
    print("""
  The manuscript states the two distributions are non-overlapping but reports
  interquartile ranges, which cannot establish that. Full ranges can.
""")
    for pk in sorted(C, key=float):
        per = C[pk].get('per_config', {})
        print(f"  p = {pk}")
        for cond in ('gated_1x', 'gated_2x', 'ungated_1x', 'ungated_2x'):
            rows = per.get(cond, [])
            rat = []
            for r in rows:
                a, b = r.get('pA'), r.get('pB')
                if (a is not None and b is not None and np.isfinite(a)
                        and np.isfinite(b) and abs(a) > 1e-9):
                    rat.append(b / a)
            if not rat:
                continue
            v = np.asarray(rat, float)
            pos = int(np.sum(v > 0))
            q1, q3 = np.percentile(v, [25, 75])
            print(f"    {cond:<12} n={len(v):>4}  positive {pos:>4}/{len(v):<4}  "
                  f"range [{v.min():+.3f}, {v.max():+.3f}]  "
                  f"IQR [{q1:+.3f}, {q3:+.3f}]  median {np.median(v):+.3f}")
        g = [b / a for r in per.get('gated_1x', [])
             if (a := r.get('pA')) and (b := r.get('pB')) and abs(a) > 1e-9]
        u = [b / a for r in per.get('ungated_1x', [])
             if (a := r.get('pA')) and (b := r.get('pB')) and abs(a) > 1e-9]
        if g and u:
            gv, uv = np.asarray(g), np.asarray(u)
            overlap = (min(gv.max(), uv.max()) >= max(gv.min(), uv.min()))
            print(f"    -> gated vs ungated at 1×: "
                  f"{'RANGES OVERLAP' if overlap else 'ranges do NOT overlap'}"
                  f"  (gated min {gv.min():+.3f}, ungated max {uv.max():+.3f})")
            if overlap:
                print("       The claim of non-overlapping distributions is too "
                      "strong. Report\n       sign counts and the overlap extent "
                      "instead.")
    print("""
  Use whichever of these the ranges support. Sign counts are the safe claim; the
  stronger 'non-overlapping distributions' claim in 4.7.4 and the Abstract needs
  the full ranges to be disjoint, and the check above says whether they are.
""")


if __name__ == '__main__':
    main()
