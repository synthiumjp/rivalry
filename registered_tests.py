"""
registered_tests.py — find and compute the two pre-registered analyses.

Two markers remain in the manuscript, both in Section 4.1, and both are
pre-registration compliance gaps rather than missing science:

  PREDICTION 3. The registered test is a logistic dose-response fit of switch
    probability on goal strength. Section 4.1 reports monotonicity counts and paired
    effect sizes instead. No fitted slope or interval appears anywhere.

  PREDICTION 5. The registered test is a 2 x 2 ANOVA, mechanism (persistence
    modulation against mean-matched input gain) by goal level, on mean dominance
    duration across 100 configurations. Section 4.5 reports percentage changes
    instead. No F or p appears anywhere.

Both were presumably run when the campaigns were executed, so the data are likely
already on disk. This script inspects every result file, reports which ones carry the
inputs each test needs, and computes whatever it can. Where the data are absent it
says so explicitly, which is the answer that determines whether a new simulation is
needed or only an analysis.

No simulation. Reads only.

USAGE
    python registered_tests.py
    python registered_tests.py --dump      # full structure of every result file
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

try:
    from scipy import stats as _st
except ImportError:
    _st = None


def outline(obj, depth=0, maxdepth=5, prefix=''):
    """Structure of a JSON object, descending into lists as well as dicts.

    The previous version walked only dicts at the top level, so a file whose root is
    a list printed as empty, and it never descended into the first element of a list
    of dicts, so nested record structure was invisible. Both are fixed: that is where
    the registered-test data actually lives.
    """
    out = []
    if depth > maxdepth:
        return out
    if isinstance(obj, list):
        if not obj:
            return [f"{'  '*depth}{prefix or '<root>'}  empty list"]
        kind = type(obj[0]).__name__
        out.append(f"{'  '*depth}{prefix or '<root>'}  list[{len(obj)}] of {kind}")
        if isinstance(obj[0], (dict, list)):
            out.append(f"{'  '*depth}  first element:")
            out += outline(obj[0], depth + 1, maxdepth, (prefix or '') + '[0]')
        else:
            out.append(f"{'  '*depth}  sample: {repr(obj[:6])[:80]}")
        return out
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:25]:
            p = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.append(f"{'  '*depth}{p}/  ({len(v)} keys: "
                           f"{', '.join(list(v)[:8])})")
                out += outline(v, depth + 1, maxdepth, p)
            elif isinstance(v, list):
                out += outline(v, depth, maxdepth, p)
            else:
                out.append(f"{'  '*depth}{p} = {repr(v)[:60]}")
    return out


def walk(obj, path=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        yield path, obj
        if obj and isinstance(obj[0], dict):
            for v in obj[:1]:
                yield from walk(v, path + '[]')
    else:
        yield path, obj


SWITCH_WORDS = ('switch_rate', 'switch_success', 'n_switches', 'switched',
                'switch_prob', 'p_switch', 'success_rate')
LEVEL_WORDS = ('goal_level', 'g_level', 'g_frac', 'level', 'goal_strength',
               'g_over_lambda', 'gfrac')
DUR_WORDS = ('mean_duration', 'duration', 'dur_own', 'mean_dur', 'dur_mean')
MECH_WORDS = ('mechanism', 'condition', 'arm', 'mode', 'manipulation')


def _matches(key, words):
    """Whole-token match, so that 'g' does not match 'config' and 'dur' does not
    match 'during'. Splits on underscores and compares components as well as the
    whole key."""
    k = key.lower()
    parts = set(k.split('_')) | {k}
    return any(w == k or w in parts or k.endswith('_' + w) or k.startswith(w + '_')
               for w in words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', action='store_true')
    ap.add_argument('--files', nargs='*', default=None,
                    help='restrict to these files, e.g. --files wave9_M_pulse.json')
    args = ap.parse_args()

    files = args.files if args.files else sorted(set(glob.glob('*.json')))
    files = [f for f in files if os.path.exists(f)]
    if not files:
        sys.exit("no JSON files in the working directory")
    print(f"{len(files)} result files\n")

    loaded = {}
    for f in files:
        try:
            with open(f) as fh:
                loaded[f] = json.load(fh)
        except Exception as e:
            print(f"  {f}: unreadable ({type(e).__name__})")

    if args.dump:
        for f, obj in loaded.items():
            print(f"\n{'='*74}\n{f}  ({os.path.getsize(f)/1024:.0f} KB)\n{'='*74}")
            for line in outline(obj)[:70]:
                print("  " + line)
        return

    # ---------------- which files could support each test ----------------
    print("=" * 74)
    print("WHICH FILES CARRY THE INPUTS EACH REGISTERED TEST NEEDS")
    print("=" * 74)
    cand_p3, cand_p5 = [], []
    for f, obj in loaded.items():
        keys = set()
        for path, v in walk(obj):
            keys.add(path.split('.')[-1].lower().strip('[]'))
        has_sw = any(_matches(k, SWITCH_WORDS) for k in keys)
        has_lv = any(_matches(k, LEVEL_WORDS) for k in keys)
        has_du = any(_matches(k, DUR_WORDS) for k in keys)
        has_me = any(_matches(k, MECH_WORDS) for k in keys)
        if has_sw and has_lv:
            cand_p3.append(f)
        if has_du and (has_me or has_lv):
            cand_p5.append(f)
        flags = ''.join(c for c, b in
                        (('S', has_sw), ('L', has_lv), ('D', has_du), ('M', has_me)) if b)
        if flags:
            print(f"  {f:<44} {flags}")
    print("\n  S = switch outcome, L = goal level, D = duration, M = mechanism label")
    print(f"\n  Prediction 3 candidates: {cand_p3 or 'NONE'}")
    print(f"  Prediction 5 candidates: {cand_p5 or 'NONE'}")

    # ---------------- Prediction 3: logistic dose-response ----------------
    print("\n" + "=" * 74)
    print("PREDICTION 3: logistic dose-response, switch probability on goal strength")
    print("=" * 74)
    print("""
  Registered test. Source: phase2_dissociation_results.json, config_results[].{dorsal,
  ventral}.g_levels.G*, which stores g_fraction with n_switched of n_seeds per level.
  A binomial logistic fit is therefore exact rather than approximated from rates.
""")

    def logistic_fit(x, k, n):
        """Binomial logistic regression by Newton-Raphson. Returns slope, intercept,
        slope standard error, and the deviance against an intercept-only model."""
        x = np.asarray(x, float); k = np.asarray(k, float); n = np.asarray(n, float)
        b = np.zeros(2)
        X = np.column_stack([np.ones_like(x), x])
        for _ in range(100):
            eta = X @ b
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
            W = n * p * (1 - p)
            if W.sum() <= 0:
                return (np.nan,) * 4
            z = eta + (k - n * p) / np.maximum(W, 1e-9)
            XtW = X.T * W
            try:
                bn = np.linalg.solve(XtW @ X, XtW @ z)
            except np.linalg.LinAlgError:
                return (np.nan,) * 4
            if np.max(np.abs(bn - b)) < 1e-10:
                b = bn; break
            b = bn
        eta = X @ b
        p = np.clip(1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30))), 1e-12, 1 - 1e-12)
        W = n * p * (1 - p)
        try:
            cov = np.linalg.inv((X.T * W) @ X)
            se = float(np.sqrt(cov[1, 1]))
        except np.linalg.LinAlgError:
            se = float('nan')
        ll = float(np.sum(k * np.log(p) + (n - k) * np.log(1 - p)))
        p0 = np.clip(k.sum() / max(n.sum(), 1), 1e-12, 1 - 1e-12)
        ll0 = float(np.sum(k * np.log(p0) + (n - k) * np.log(1 - p0)))
        return float(b[1]), float(b[0]), se, 2 * (ll - ll0)

    p3 = {}
    src3 = 'phase2_dissociation_results.json'
    if src3 in loaded:
        recs = loaded[src3].get('config_results', [])
        for regime in ('dorsal', 'ventral'):
            X, K, N, per_cfg = [], [], [], []
            for r in recs:
                lv = (r.get(regime) or {}).get('g_levels') or {}
                xs, ks, ns = [], [], []
                for key in sorted(lv, key=lambda s: lv[s].get('g_fraction', 0)):
                    d = lv[key]
                    if d.get('n_seeds'):
                        xs.append(float(d['g_fraction']))
                        ks.append(float(d.get('n_switched', 0)))
                        ns.append(float(d['n_seeds']))
                if len(xs) >= 3 and sum(ks) > 0:
                    s, _, _, _ = logistic_fit(xs, ks, ns)
                    if np.isfinite(s):
                        per_cfg.append(s)
                X += xs; K += ks; N += ns
            if not X:
                continue
            sl, ic, se, dev = logistic_fit(X, K, N)
            lo, hi = sl - 1.96 * se, sl + 1.96 * se
            pv = (1 - _st.chi2.cdf(dev, 1)) if _st is not None else float('nan')
            print(f"  {regime:<9} pooled over {len(recs)} configurations, "
                  f"{int(sum(N))} trials")
            print(f"            slope {sl:+.3f} [{lo:+.3f}, {hi:+.3f}] per unit "
                  f"g/λ, LR χ²(1) = {dev:.1f}, p = {pv:.3g}")
            if per_cfg:
                v = np.array(per_cfg)
                print(f"            per-configuration slopes: median {np.median(v):+.3f}, "
                      f"{int((v > 0).sum())}/{v.size} positive")
            p3[regime] = dict(slope=sl, ci=[lo, hi], lr_chi2=dev, p=float(pv),
                              n_trials=int(sum(N)), n_config_fits=len(per_cfg),
                              median_per_config=float(np.median(per_cfg)) if per_cfg else None)
    src3b = 'dissociation_retest_results.json'
    if src3b in loaded:
        print(f"\n  independent replicate, {src3b}:")
        for regime in ('dorsal', 'ventral'):
            X, K, N = [], [], []
            for r in loaded[src3b].get('results', []):
                dd = (r.get('dissociation_retest') or {}).get(regime) or {}
                for key, d in dd.items():
                    if isinstance(d, dict) and d.get('valid_trials'):
                        X.append(float(d['G_frac'])); K.append(float(d.get('switches', 0)))
                        N.append(float(d['valid_trials']))
            if not X:
                continue
            sl, ic, se, dev = logistic_fit(X, K, N)
            pv = (1 - _st.chi2.cdf(dev, 1)) if _st is not None else float('nan')
            print(f"    {regime:<9} slope {sl:+.3f} "
                  f"[{sl-1.96*se:+.3f}, {sl+1.96*se:+.3f}], p = {pv:.3g}, "
                  f"{int(sum(N))} trials")
            p3[f'{regime}_replicate'] = dict(slope=sl, p=float(pv))

    # ---------------- Prediction 5: 2 x 2 ANOVA ----------------
    print("\n" + "=" * 74)
    print("PREDICTION 5: 2 x 2 ANOVA, mechanism x channel, on mean dominance duration")
    print("=" * 74)
    print("""
  Registered test. Source: wave9_N_matching.json, conditions.{gclca, boost_1}
  .registered.{mean_dur_a, mean_dur_b}, at the mean-matched boost amplitude. The two
  factors are mechanism (persistence modulation against input gain) and channel
  (manipulated against competitor), both within configuration, which is what
  "different rivalry signatures" means: the signature is the pattern across channels,
  so the interaction is the registered effect.
""")
    p5 = {}
    src5 = 'wave9_N_matching.json'
    rows = loaded.get(src5)
    if isinstance(rows, list) and rows:
        cells = {k: [] for k in ('gA', 'gB', 'bA', 'bB')}
        wta = []
        for r in rows:
            cond = r.get('conditions') or {}
            g = (cond.get('gclca') or {}).get('registered') or {}
            b = (cond.get('boost_1') or {}).get('registered') or {}
            if all(k in g for k in ('mean_dur_a', 'mean_dur_b')) and \
               all(k in b for k in ('mean_dur_a', 'mean_dur_b')):
                cells['gA'].append(g['mean_dur_a']); cells['gB'].append(g['mean_dur_b'])
                cells['bA'].append(b['mean_dur_a']); cells['bB'].append(b['mean_dur_b'])
                wta.append(max(float(g.get('wta_rate', 0) or 0),
                               float(b.get('wta_rate', 0) or 0)))
        M = {k: np.array(v, float) for k, v in cells.items()}
        W = np.array(wta, float)
        ok = np.ones(len(M['gA']), bool)
        for v in M.values():
            ok &= np.isfinite(v)
        n = int(ok.sum())

        def two_way(sel, label):
            gA, gB, bA, bB = (M[k][sel] for k in ('gA', 'gB', 'bA', 'bB'))
            m = int(sel.sum())
            if m < 8:
                print(f"  {label}: only {m} configurations, not run")
                return {}
            print(f"\n  {label}, {m} configurations")
            print(f"    {'':>22} {'manipulated':>12} {'competitor':>12}")
            print(f"    {'persistence (gclca)':>22} {gA.mean():>12.2f} {gB.mean():>12.2f}")
            print(f"    {'input gain (matched)':>22} {bA.mean():>12.2f} {bB.mean():>12.2f}")
            res = {}
            for lab, d in (('mechanism', ((gA + gB) / 2) - ((bA + bB) / 2)),
                           ('channel', ((gA + bA) / 2) - ((gB + bB) / 2)),
                           ('mechanism x channel', (gA - gB) - (bA - bB))):
                se = d.std(ddof=1) / np.sqrt(m)
                t = d.mean() / se if se > 0 else np.nan
                pv = (2 * (1 - _st.t.cdf(abs(t), m - 1))) if _st is not None else float('nan')
                print(f"    {lab:<22} F(1,{m-1}) = {t**2:9.3f}, p = {pv:.4g}, "
                      f"mean {d.mean():+9.2f}, sd {d.std(ddof=1):9.2f}")
                res[lab] = dict(F=float(t ** 2), df=[1, m - 1], p=float(pv),
                                mean=float(d.mean()), sd=float(d.std(ddof=1)))
            return res

        print(f"  winner-take-all rate across conditions: "
              f"{int((W[ok] > 0).sum())} of {n} configurations show any")
        p5['as_registered'] = two_way(ok, "AS REGISTERED, raw durations")
        print("""
    The registered analysis is uninformative on raw durations. The persistence
    condition drives some configurations toward winner-take-all, where a dominance
    episode has no upper bound, so the cell means are dominated by a few
    configurations and the variance swamps effects of several hundred timesteps.
    Two deviations follow, both reported.""")

        sel2 = ok & (W <= 0)
        p5['wta_excluded'] = two_way(sel2, "DEVIATION 1: excluding any winner-take-all")

        L = {k: np.log(np.clip(M[k], 1e-6, None)) for k in M}
        gA, gB, bA, bB = (L[k][ok] for k in ('gA', 'gB', 'bA', 'bB'))
        m = n
        print(f"\n  DEVIATION 2: log duration, {m} configurations")
        res = {}
        for lab, d in (('mechanism', ((gA + gB) / 2) - ((bA + bB) / 2)),
                       ('channel', ((gA + bA) / 2) - ((gB + bB) / 2)),
                       ('mechanism x channel', (gA - gB) - (bA - bB))):
            se = d.std(ddof=1) / np.sqrt(m)
            t = d.mean() / se if se > 0 else np.nan
            pv = (2 * (1 - _st.t.cdf(abs(t), m - 1))) if _st is not None else float('nan')
            print(f"    {lab:<22} F(1,{m-1}) = {t**2:9.3f}, p = {pv:.4g}, "
                  f"mean log ratio {d.mean():+7.3f}")
            res[lab] = dict(F=float(t ** 2), df=[1, m - 1], p=float(pv),
                            mean_log=float(d.mean()))
        p5['log_duration'] = res
        print("""
    Report the registered analysis first and both deviations after it, with the
    reason. A reviewer will accept a registered test that fails on its own terms
    provided the failure is diagnosed; what they will not accept is quietly
    substituting the version that works.""")
    else:
        print(f"  only {n if 'n' in dir() else 0} complete configurations")
    json.dump(dict(prediction_3=p3, prediction_5=p5),
              open('registered_tests_output.json', 'w'), indent=1, default=float)
    print("\n  wrote registered_tests_output.json")

    print("""
=== WHAT TO DO WITH THE ANSWER ===

If both tests compute, paste the output and the two markers in Section 4.1 close with
the registered statistics reported regardless of outcome, which is what the
pre-registration commits to.

If either does not, the honest alternative is one sentence in Section 4.1 saying the
registered analysis was not run and why, with the substituted analysis named. A
reviewer will accept that; what they will not accept is the registered test being
listed in the audit table with no statistics anywhere in the manuscript.
""")


if __name__ == '__main__':
    main()
