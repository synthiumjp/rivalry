"""
wave22_reconcile.py — why has the gated 2x competitor response been reported
three times with three values?

  +23.6%   an earlier manuscript draft, believed to be wave17
  +16.2%   the current draft, designated canonical (wave18)
  +17.1%   wave22 block C at p = 1

The last two agree to within Monte Carlo. The first does not, and 7 percentage
points is well outside the ~1 pp campaign-to-campaign variation that Section 3.7
of the manuscript documents. One of these campaigns differs in more than its
seeds.

There are only three candidate explanations and they are distinguishable:

  (A) SAMPLE. The campaigns drew different configurations, or different numbers
      of them, and the reported value is a median across a heterogeneous
      quantity. Diagnostic: the medians agree when computed on the
      configurations the campaigns share.

  (B) CONDITION. The gate was implemented differently -- a lag constant applied
      in one and not the other, a different amplitude convention, a different
      threshold. Diagnostic: the medians disagree on shared configurations, and
      the delivered dose differs.

  (C) OUTCOME. The competitor's duration change was measured against a
      different baseline, or under the difference criterion in one campaign and
      the absolute criterion in the other. Diagnostic: as (B), but dose agrees
      while the outcome does not.

This script does no simulation. It reads the stored campaign outputs, reports
what each one recorded for the gated conditions, and where per-configuration
values are available it recomputes the medians on the intersection.

USAGE
    python wave22_reconcile.py
    python wave22_reconcile.py --files wave17_results.json wave18_yoked.json w22_wave22_C_gates.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT = ['wave17_results.json', 'wave17_A_gates.json', 'wave18_yoked.json',
           'wave18_results.json', 'w22_wave22_C_gates.json',
           'wave22_C_gates.json', 'wave11_gated.json', 'wave11_results.json']

GATE_WORDS = ('gate', 'gated', 'antigate', 'anti_gate', 'anti-gated', 'yoked',
              'ungated')
COMP_WORDS = ('competitor', 'rival', 'pb', 'p_b', 'b_pct', 'rival_pct',
              'competitor_pct', 'delta_b')
DOSE_WORDS = ('dose', 'delivered')
CONF_WORDS = ('config', 'config_idx', 'cfg', 'idx')


def walk(obj, path=''):
    """Yield (path, value) for every leaf, and (path, list) for lists of dicts."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj):
            yield path, obj
        else:
            yield path, obj
    else:
        yield path, obj


def outline(obj, prefix='', depth=0, maxdepth=3, out=None):
    if out is None:
        out = []
    if depth > maxdepth:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.append(f"{'  '*depth}{p}/  ({len(v)} keys: "
                           f"{', '.join(list(v)[:6])})")
                outline(v, p, depth + 1, maxdepth, out)
            elif isinstance(v, list):
                kind = (f"list[{len(v)}] of dict, keys: "
                        f"{', '.join(list(v[0])[:8])}"
                        if v and isinstance(v[0], dict) else f"list[{len(v)}]")
                out.append(f"{'  '*depth}{p}  {kind}")
            else:
                out.append(f"{'  '*depth}{p} = {repr(v)[:70]}")
    return out


def find_gate_records(obj):
    """Return {label: list-of-per-config-dicts} for anything gate-shaped."""
    found = {}
    for path, val in walk(obj):
        if not (isinstance(val, list) and val and isinstance(val[0], dict)):
            continue
        keys = ' '.join(k.lower() for k in val[0])
        if any(w in keys for w in COMP_WORDS) or any(
                w in path.lower() for w in GATE_WORDS):
            found[path] = val
    return found


def pick(d, words):
    for k in d:
        kl = k.lower()
        for w in words:
            if kl == w or kl.endswith(w) or w in kl:
                v = d[k]
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    return k, float(v)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='*', default=None)
    ap.add_argument('--outline', action='store_true',
                    help='print the full structure of each file and stop')
    args = ap.parse_args()

    files = args.files or [f for f in DEFAULT if Path(f).exists()]
    if not files:
        sys.exit("no campaign JSONs found. Pass them with --files. Looked for:\n  "
                 + "\n  ".join(DEFAULT))

    loaded = {}
    for f in files:
        try:
            with open(f) as fh:
                loaded[f] = json.load(fh)
        except Exception as e:
            print(f"  could not read {f}: {e}")
    print(f"loaded {len(loaded)} campaign files\n")

    # ---------- structure ----------
    print("=" * 78)
    print("STRUCTURE")
    print("=" * 78)
    for f, obj in loaded.items():
        print(f"\n--- {f} ---")
        for line in outline(obj)[:40]:
            print("  " + line)
    if args.outline:
        return

    # ---------- gate-shaped records ----------
    print("\n" + "=" * 78)
    print("GATE-SHAPED RECORD SETS")
    print("=" * 78)
    per_campaign = {}
    for f, obj in loaded.items():
        recs = find_gate_records(obj)
        if not recs:
            print(f"\n--- {f}: none found ---")
            continue
        print(f"\n--- {f} ---")
        for path, rows in recs.items():
            ck, _ = pick(rows[0], COMP_WORDS)
            idk, _ = pick(rows[0], CONF_WORDS)
            dk, _ = pick(rows[0], DOSE_WORDS)
            print(f"  {path}: n={len(rows)}  competitor field={ck!r}  "
                  f"config field={idk!r}  dose field={dk!r}")
            if ck:
                vals, ids, doses = [], [], []
                for r in rows:
                    _, cv = pick(r, COMP_WORDS)
                    _, ci = pick(r, CONF_WORDS)
                    _, dv = pick(r, DOSE_WORDS)
                    if cv is not None:
                        vals.append(cv)
                        ids.append(int(ci) if ci is not None else None)
                        doses.append(dv)
                v = np.array(vals, float)
                print(f"      median {np.nanmedian(v):+.2f}  "
                      f"mean {np.nanmean(v):+.2f}  "
                      f"n_finite {int(np.isfinite(v).sum())}"
                      + (f"  median dose {np.nanmedian([d for d in doses if d is not None]):.3f}"
                         if any(d is not None for d in doses) else ""))
                if any(i is not None for i in ids):
                    per_campaign[f"{f}::{path}"] = {
                        i: val for i, val in zip(ids, vals) if i is not None}

    # ---------- the decisive test ----------
    print("\n" + "=" * 78)
    print("PAIRED COMPARISON ON SHARED CONFIGURATIONS")
    print("=" * 78)
    keys = list(per_campaign)
    if len(keys) < 2:
        print("""
  Fewer than two record sets carry per-configuration values with an identifier,
  so the paired test cannot be run from these files. That is itself informative:
  a campaign that stored only aggregate medians cannot be reconciled against
  another except by rerunning it.

  If the raw per-configuration values exist in a file not listed above, pass it
  with --files. Otherwise the practical resolution is to treat the canonical
  campaign as authoritative, delete the superseded value from the manuscript
  rather than trying to explain it, and record in Section 3.7 that the earlier
  figure came from a campaign whose per-configuration output was not retained.""")
        return

    def cond(k):
        return k.rsplit('.', 1)[-1].lower()

    pairs = [(i, j) for i in range(len(keys)) for j in range(i + 1, len(keys))
             if cond(keys[i]) == cond(keys[j])]
    if not pairs:
        print("\n  No two record sets name the same condition, so there is nothing"
              "\n  to pair. Comparing different conditions is meaningless -- of course"
              "\n  gated differs from ungated. Check the condition labels above.")
    for i, j in pairs:
            a, b = per_campaign[keys[i]], per_campaign[keys[j]]
            shared = sorted(set(a) & set(b))
            print(f"\n  {keys[i]}\n  vs {keys[j]}")
            print(f"    configurations: {len(a)} and {len(b)}, "
                  f"shared {len(shared)}")
            if len(shared) < 5:
                print("    too few shared to compare -> different samples, "
                      "which is explanation (A) on its own")
                continue
            va = np.array([a[k] for k in shared], float)
            vb = np.array([b[k] for k in shared], float)
            print(f"    median on ALL:    {np.nanmedian(list(a.values())):+.2f} "
                  f"vs {np.nanmedian(list(b.values())):+.2f}")
            print(f"    median on SHARED: {np.nanmedian(va):+.2f} vs "
                  f"{np.nanmedian(vb):+.2f}")
            d = vb - va
            print(f"    paired difference: median {np.nanmedian(d):+.2f}, "
                  f"mean {np.nanmean(d):+.2f}, "
                  f"same sign in {int(np.nansum(np.sign(va) == np.sign(vb)))}"
                  f"/{len(shared)}")
            if abs(np.nanmedian(d)) < 2.0:
                print("""    -> AGREE on shared configurations. The discrepancy is
       explanation (A): the campaigns sampled different configurations. Quote
       the canonical campaign, state its n, and note that the earlier figure
       came from a different sample of the same pool.""")
            else:
                print("""    -> DISAGREE on shared configurations. The condition or the
       outcome measure differs between campaigns, explanation (B) or (C).
       Compare the median dose printed above: if dose differs, the gate
       implementation changed; if dose agrees, the outcome measure did.""")

    print("""
=== WHAT TO DO WITH THE ANSWER ===

Either way, one value goes in the manuscript and the others are deleted rather
than reconciled in prose. Section 3.7 designates a canonical campaign; the fix is
to recompute every gate-timing number from it and quote nothing else, which is
the convention the manuscript already states and has not fully applied. If the
paired test shows the campaigns disagree on shared configurations, the superseded
campaign should also be named in the deviations list, because a changed condition
definition is a deviation and not a Monte Carlo difference.
""")


if __name__ == '__main__':
    main()
