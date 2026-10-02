"""
check_geometry.py - two checks on block R's output, in the paper's own architecture.

(1) DOES THE GEOMETRY EXPLAIN THE ATTRITION?
    A gated increment raises the attended channel's handover threshold from c to about
    c + delta, where delta = increment / alpha. If that reaches the asymptote L, the
    attended channel never hands over and the competitor has no episodes to measure.
    So the configurations excluded from the gated counts should be those where delta
    is large relative to the margin L - c. The prediction is a threshold at
    delta / (L - c) = 1 in the deterministic limit, softened by noise; the rank test
    (AUC) does not depend on where exactly it falls. The larger increment should also
    exclude more configurations than the smaller one.

(2) IS exp(-gamma T) DOING WORK THAT ITS PARTS DO NOT?
    exp(-gamma T) predicts the gated ratio at Spearman +0.85. That could be carried by
    T alone or gamma alone. Compare each, and permute gamma across configurations to
    ask whether pairing each configuration's own gamma with its own T matters.

Reads w22_wave22_R_geometry.json; recomputes each baseline with block R's own seeds
(20,000 steps, 12 seeds) to get the activation that set the increment.

USAGE
    python check_geometry.py                    # reads w22_wave22_R_geometry.json
    python check_geometry.py --prefix final_    # reads final_wave22_R_geometry.json
"""
import json
import numpy as np
import wave22_adaptation as w

STEPS, SEEDS, AMPS = 20000, 12, (0.25, 0.50)


def rank(v):
    return np.argsort(np.argsort(np.asarray(v, float)))


def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(rank(a[ok]), rank(b[ok]))[0, 1]) if ok.sum() > 4 else float('nan')


def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if pos.size == 0 or neg.size == 0:
        return float('nan')
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (pos.size * neg.size))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='w22_', help='prefix block R wrote under')
    args = ap.parse_args()
    R = json.load(open(args.prefix + 'wave22_R_geometry.json'))
    rows = R['rows']
    pool = {c['idx']: c for c in w.load_pool(False)}
    print(f"{len(rows)} configurations from block R\n")

    # ---------------- (1) attrition ----------------
    print("=" * 74 + "\n(1) DOES THE GEOMETRY EXPLAIN WHICH CONFIGURATIONS DROP OUT?\n" + "=" * 74)
    recs = []
    for r in rows:
        c = pool[r['config']]
        b = w.run_cell(c, 1.0, 0.5, 0.5, 0.0, w.G_NONE, 0.0, STEPS, SEEDS,
                       w.BLK_R_BASE, 0, split_indet=w.SPLIT_INDET)
        actA = w.sm(b, w.O_ACTA)
        margin = r['L'] - r['c']
        for a in AMPS:
            pB = r.get(f'gated_{a}_pB')
            excluded = pB is None or not np.isfinite(pB)
            delta = a * c['lam'] * actA / c['alpha']
            recs.append(dict(a=a, x=delta / margin if margin > 0 else np.inf,
                             rho=r['rho'], excl=excluded))
    for a in AMPS:
        rr = [q for q in recs if q['a'] == a]
        x = np.array([q['x'] for q in rr]); ex = np.array([q['excl'] for q in rr])
        rho = np.array([q['rho'] for q in rr])
        print(f"\n  increment {a} x lambda x baseline activation: "
              f"{int(ex.sum())} of {ex.size} excluded")
        if ex.sum():
            print(f"    delta / margin: excluded median {np.median(x[ex]):.2f}, "
                  f"retained median {np.median(x[~ex]):.2f}")
            print(f"    AUC of delta / margin for exclusion: {auc(x, ex):.3f}")
            print(f"    rho: excluded median {np.median(rho[ex]):.3f}, "
                  f"retained median {np.median(rho[~ex]):.3f}")
            for thr in (0.5, 1.0):
                hi = x >= thr
                print(f"    delta / margin >= {thr}: {int((ex & hi).sum())}/{int(hi.sum())} "
                      f"excluded;  below: {int((ex & ~hi).sum())}/{int((~hi).sum())}")
    xa = np.array([q['x'] for q in recs]); ea = np.array([q['excl'] for q in recs])
    print(f"\n  pooled over both increments: AUC {auc(xa, ea):.3f} on "
          f"{int(ea.sum())} exclusions of {ea.size}")
    print("""
  HOW TO READ THIS. An AUC well above 0.5 means the configurations that drop out are
  the ones whose handover threshold the increment pushes closest to the asymptote,
  which is what the geometry predicts. The count should also rise with the increment.
  If both hold, the exclusion stated in Sections 4.3.3 and 4.3.5 is a derived
  consequence of the mechanism rather than a caveat. If the AUC is near 0.5, it is
  not, and the exclusion stays a caveat.""")

    # ---------------- (2) parts versus combination ----------------
    print("=" * 74 + "\n(2) IS exp(-gamma T) DOING WORK ITS PARTS DO NOT?\n" + "=" * 74)
    g = np.array([r.get('gated_0.25', np.nan) for r in rows], float)
    T = np.array([r['T_obs'] for r in rows], float)
    gam = np.array([pool[r['config']]['gam'] for r in rows], float)
    rho = np.array([r['rho'] for r in rows], float)
    ok = np.isfinite(g)
    pred = np.exp(-gam * T)
    obs = sp(g[ok], pred[ok])
    rng = np.random.default_rng(0)
    null = np.array([sp(g[ok], np.exp(-rng.permutation(gam[ok]) * T[ok])) for _ in range(5000)])
    print(f"\n  gated ratio at 0.25, n = {int(ok.sum())}")
    print(f"    Spearman with exp(-gamma T)     {obs:+.3f}")
    print(f"    Spearman with -T alone          {sp(g[ok], -T[ok]):+.3f}")
    print(f"    Spearman with -gamma alone      {sp(g[ok], -gam[ok]):+.3f}")
    print(f"    Spearman with rho               {sp(g[ok], rho[ok]):+.3f}")
    print(f"    gamma permuted across configurations: null median {np.median(null):+.3f}, "
          f"95th pct {np.percentile(null, 95):+.3f}, p = {(np.sum(null >= obs) + 1) / (null.size + 1):.4f}")
    print(f"    grid values of gamma: {sorted(set(np.round(gam, 4)))}")
    print("""
  HOW TO READ THIS. If T alone correlates nearly as strongly as exp(-gamma T), most of
  the prediction is carried by duration, which would be a weaker claim: long episodes
  go with small ratios. The permutation asks whether matching each configuration's
  own gamma to its own T matters. With only a few grid values of gamma the permutation
  is coarse, so read it alongside the parts.""")
    json.dump(dict(attrition=recs, parts=dict(sp_pred=obs, sp_T=sp(g[ok], -T[ok]),
                                              sp_gam=sp(g[ok], -gam[ok]), sp_rho=sp(g[ok], rho[ok]),
                                              perm_p=float((np.sum(null >= obs) + 1) / (null.size + 1)))),
              open(args.prefix + 'check_geometry_output.json', 'w'), indent=1, default=float)
    print("\n  wrote check_geometry_output.json")


if __name__ == '__main__':
    main()
