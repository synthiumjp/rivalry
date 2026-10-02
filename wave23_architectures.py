"""
wave23_architectures.py — does the gate-timing result belong to rivalry or to one
accumulator?

THE HOLE THIS FILLS
-------------------
Every result in the companion paper is obtained in one architecture: subtractive
mutual inhibition with adaptation proportional to a channel's own output. The paper
concedes the limitation three times and never tests it. Section 1.2 explicitly
exempts normalisation models of attention from its input-gain argument. Section 1.1
builds the introduction around Li, Rankin, Rinzel, Carrasco and Heeger (2017), whose
model uses divisive rather than subtractive suppression. Section 6 states that
running the gate-timing series in a normalisation architecture "is the obvious next
test and has not been done", and notes that response gain is a poor stand-in because
the architecture has no divisive term to modulate.

So the central claim currently reads "in the GC-LCA". If the sign reversal is a
property of competitive networks with adaptation, the paper is about rivalry. If it
is a property of this accumulator, it is about this accumulator.

WHAT IS TESTED
--------------
Four architectures, each a generic representative of a family rather than a
reimplementation of a specific published model. That distinction is deliberate and
must be reported: the point is whether the result survives changes of architectural
kind, not whether it reproduces any particular author's parameter set.

    A  LCA-SUB     subtractive inhibition, adaptation subtracted from the
                   accumulator and driven by the channel's own output. This is the
                   companion paper's architecture and serves as the control.

    B  DIVISIVE    the competitor enters the denominator of a normalised drive
                   rather than being subtracted. The family of Wilson (2003) and
                   Li et al. (2017), and the case Section 1.2 exempts.

    C  SIGMOID-FR  firing-rate competition with a sigmoid transfer function on the
                   net input. The family of Laing and Chow (2002) and Shpiro,
                   Curtu, Rinzel and Rubin (2007), whose analyses of the
                   increasing-duration regime the paper relies on.

    D  INPUT-ADAPT subtractive inhibition, but adaptation multiplies the input
                   drive rather than subtracting from the accumulator -- synaptic
                   depression rather than spike-frequency adaptation. Changes the
                   LOCUS of adaptation while keeping its dynamics.

For each, the gate-timing series: an identical increment on channel A delivered only
while A is dominant, at every timestep, or only while A is suppressed. The
prediction, if the result is architectural rather than incidental, is that gated
delivery lengthens the competitor and ungated delivery shortens it, in every family.

GUARDING AGAINST STRAWMEN
-------------------------
An architecture can be made to confirm anything if its parameters are chosen after
seeing the answer. Each family is therefore given an independent random parameter
search, and a configuration enters the test only if it produces rivalry on its own
terms: at least 10 switches per seed, a coefficient of variation of dominance
durations in [0.30, 0.70], and Levelt's first proposition satisfied, meaning
predominance increases with that channel's input strength. Configurations are
accepted or rejected before any gated condition is run. The search yield is reported
per family, since a family that only rivals in a narrow corner of its parameter space
is a weaker test than one that rivals broadly.

USAGE
    python wave23_architectures.py                 # ~10 min
    python wave23_architectures.py --n-config 60 --seeds 12 --steps 20000
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*a, **k):
        return (lambda f: f) if not a else a[0]
    print("WARNING: numba unavailable, this will be slow. Reduce --n-config.")

THETA = 0.05
BURN = 500
MIN_DUR = 5
A_LCA, A_DIV, A_SIG, A_INP = 0, 1, 2, 3
ARCH_NAME = {A_LCA: 'A LCA-SUB', A_DIV: 'B DIVISIVE',
             A_SIG: 'C SIGMOID-FR', A_INP: 'D INPUT-ADAPT'}
G_NONE, G_UNGATED, G_ON, G_OFF = 0, 1, 2, 3


@njit(cache=True)
def _trace(arch, lam, beta, alpha, sigma, gam, kap, sig_n, w, thr, slope,
           S_A, S_B, inc, gate, n_steps, seed, tau, delay):
    """One trace. Returns [durA, durB, cvA, nEpisodes, switchesFiltered,
    predomA, actA, actB]."""
    np.random.seed(seed)
    xA = 0.1
    xB = 0.1
    aA = 0.0
    aB = 0.0
    gsig = 0.0

    cur = 0
    cl = 0
    pend = 0
    pl = 0
    seen = 0
    s1 = 0.0
    q1 = 0.0
    c1 = 0
    s2 = 0.0
    c2 = 0
    nswf = 0
    prev = 0
    domA = 0
    domB = 0
    actA = 0.0
    actB = 0.0
    npost = 0
    L = delay + 1
    hist = np.zeros(L, dtype=np.int64)     # ring buffer of gate states, for a pure delay

    for t in range(n_steps):
        d = xA - xB
        if gate == G_NONE:
            on = 0
        elif gate == G_UNGATED:
            on = 1
        elif gate == G_ON:
            on = 1 if d > THETA else 0
        else:
            on = 1 if d < -THETA else 0
        if delay > 0:
            hist[t % L] = on
            on = hist[(t - delay) % L] if t >= delay else 0
        target = inc if on == 1 else 0.0
        if tau <= 1.0:
            gsig = target
        else:
            gsig = gsig + (target - gsig) / tau
        add = gsig

        nA = np.random.normal(0.0, sigma)
        nB = np.random.normal(0.0, sigma)
        dA = S_A + add
        dB = S_B

        if arch == A_LCA:
            newA = (1.0 - lam) * xA + dA - beta * xB - alpha * aA + nA
            newB = (1.0 - lam) * xB + dB - beta * xA - alpha * aB + nB
        elif arch == A_DIV:
            # competitor pools into the denominator of a normalised drive
            rA = dA / (sig_n + dA + w * xB)
            rB = dB / (sig_n + dB + w * xA)
            newA = (1.0 - lam) * xA + rA - alpha * aA + nA
            newB = (1.0 - lam) * xB + rB - alpha * aB + nB
        elif arch == A_SIG:
            uA = dA - beta * xB - alpha * aA
            uB = dB - beta * xA - alpha * aB
            fA = 1.0 / (1.0 + math.exp(-(uA - thr) / slope))
            fB = 1.0 / (1.0 + math.exp(-(uB - thr) / slope))
            newA = (1.0 - lam) * xA + lam * fA + nA
            newB = (1.0 - lam) * xB + lam * fB + nB
        else:
            # adaptation gates the input multiplicatively (synaptic depression)
            gA = 1.0 - alpha * aA
            gB = 1.0 - alpha * aB
            if gA < 0.0:
                gA = 0.0
            if gB < 0.0:
                gB = 0.0
            newA = (1.0 - lam) * xA + dA * gA - beta * xB + nA
            newB = (1.0 - lam) * xB + dB * gB - beta * xA + nB

        if newA < 0.0:
            newA = 0.0
        elif newA > 5.0:
            newA = 5.0
        if newB < 0.0:
            newB = 0.0
        elif newB > 5.0:
            newB = 5.0

        xA = newA
        xB = newB
        aA = (1.0 - gam) * aA + kap * xA
        aB = (1.0 - gam) * aB + kap * xB

        if t < BURN:
            continue
        npost += 1
        dd = xA - xB
        if dd > THETA:
            new = 1
            domA += 1
        elif dd < -THETA:
            new = -1
            domB += 1
        else:
            new = 0

        if new != cur:
            if cur != 0:
                if seen == 0:
                    seen = 1
                else:
                    if pend != 0:
                        fl = float(pl)
                        if pend == 1:
                            s1 += fl
                            q1 += fl * fl
                            c1 += 1
                        else:
                            s2 += fl
                            c2 += 1
                        if pl >= MIN_DUR:
                            if prev != 0 and pend != prev:
                                nswf += 1
                            prev = pend
                    pend = cur
                    pl = cl
            cl = 1
        else:
            cl += 1
        cur = new
        actA += xA
        actB += xB

    out = np.zeros(8)
    if c1 > 0:
        m1 = s1 / c1
        v1 = q1 / c1 - m1 * m1
        out[0] = m1
        out[2] = (math.sqrt(v1) / m1) if (v1 > 0.0 and m1 > 0.0) else 0.0
    else:
        out[0] = np.nan
        out[2] = np.nan
    out[1] = (s2 / c2) if c2 > 0 else np.nan
    out[3] = c1 + c2
    out[4] = nswf
    out[5] = (domA / (domA + domB)) if (domA + domB) > 0 else np.nan
    out[6] = actA / npost if npost > 0 else np.nan
    out[7] = actB / npost if npost > 0 else np.nan
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    ph = k / n
    c = (ph + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def _seed(*parts):
    """Deterministic across processes. The previous version used hash() on a tuple
    containing a string, which Python salts per process unless PYTHONHASHSEED is set,
    so runs were statistically stable but not bit-reproducible."""
    import zlib
    return zlib.crc32(repr(parts).encode()) % (2 ** 31 - 1)


def cell(arch, p, S_A, S_B, inc, gate, n_seeds, n_steps, tag, tau=0.0, delay=0):
    r = np.empty((n_seeds, 8))
    for s in range(n_seeds):
        seed = _seed(arch, tag, round(p['lam'], 5), round(p['beta'], 5),
                     round(p['sigma'], 5), s)
        r[s] = _trace(arch, p['lam'], p['beta'], p['alpha'], p['sigma'], p['gam'],
                      p['kap'], p['sig_n'], p['w'], p['thr'], p['slope'],
                      S_A, S_B, inc, gate, n_steps, seed, tau, int(delay))
    return np.nanmean(r, axis=0)


def draw(arch, rng):
    """Wide, family-appropriate parameter draws. Deliberately not tuned."""
    lam = 10 ** rng.uniform(-1.4, -0.6)
    p = dict(lam=lam,
             beta=10 ** rng.uniform(-1.3, -0.3),
             alpha=10 ** rng.uniform(-1.8, -0.7),
             sigma=10 ** rng.uniform(-1.7, -0.8),
             gam=10 ** rng.uniform(-2.0, -1.1),
             kap=10 ** rng.uniform(-1.8, -0.7),
             sig_n=10 ** rng.uniform(-1.3, 0.3),
             w=10 ** rng.uniform(-0.5, 1.0),
             thr=rng.uniform(0.05, 0.6),
             slope=10 ** rng.uniform(-1.5, -0.5))
    if arch == A_SIG:
        p['beta'] = 10 ** rng.uniform(-0.7, 0.4)
    if arch == A_INP:
        p['alpha'] = 10 ** rng.uniform(-0.8, 0.3)
    return p


def eligible(arch, p, args):
    """Rivalry on its own terms, decided before any gated condition is run."""
    b = cell(arch, p, 0.5, 0.5, 0.0, G_NONE, max(4, args.seeds // 2),
             args.steps, 'elig')
    dA, cv, nep = b[0], b[2], b[3]
    if not (np.isfinite(dA) and np.isfinite(cv)):
        return None
    sw_per_seed = nep
    if sw_per_seed < 10 or not (0.30 <= cv <= 0.70):
        return None
    # Levelt I: predominance must rise with that channel's input strength
    pred = []
    for lv, sa in enumerate((0.40, 0.50, 0.60)):
        r = cell(arch, p, sa, 0.5, 0.0, G_NONE, max(3, args.seeds // 3),
                 args.steps, f'lev{lv}')
        pred.append(r[5])
    if not all(np.isfinite(v) for v in pred):
        return None
    if not (pred[0] < pred[1] < pred[2]):
        return None
    return dict(cv=float(cv), durA=float(dA), n_ep=float(nep),
                actA=float(b[6]), actB=float(b[7]))



def _rank(v):
    return np.argsort(np.argsort(np.asarray(v, float)))


def _sp(a, b, minn=5):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < minn:
        return float('nan')
    return float(np.corrcoef(_rank(a[ok]), _rank(b[ok]))[0, 1])


def geometry_check(rows, tag, seed=0):
    """Does exp(-gamma T) predict the gated coupling ratio in this family?

    The threshold-geometry reduction of the main paper gives the gated ratio as
    rho = (L - c)/(L + c) and episode duration as T = ln(1/rho)/gamma, hence
    rho = exp(-gamma T). In the paper's own architecture exp(-gamma T) predicts the
    gated ratio at Spearman +0.85. This asks whether it does in each family, whether
    the combination beats either part alone, and whether pairing each configuration's
    gamma with its own T matters, by permuting gamma within the family."""
    rr = [x for x in rows.get(tag, []) if np.isfinite(x['pA']) and np.isfinite(x['pB'])
          and abs(x['pA']) > 1e-9 and np.isfinite(x.get('T0', np.nan))]
    if len(rr) < 8:
        return None
    ratio = np.array([x['pB'] / x['pA'] for x in rr])
    gam = np.array([x['gam'] for x in rr]); T = np.array([x['T0'] for x in rr])
    pred = np.exp(-gam * T)
    obs_sp = _sp(ratio, pred)
    rng = np.random.default_rng(seed)
    null = np.array([_sp(ratio, np.exp(-rng.permutation(gam) * T)) for _ in range(2000)])
    return dict(n=len(rr), sp_pred=obs_sp, sp_T=_sp(ratio, -T), sp_gam=_sp(ratio, -gam),
                perm_p=float((np.sum(null >= obs_sp) + 1) / (null.size + 1)),
                perm_95=float(np.percentile(null, 95)),
                calib=float(np.median(ratio / pred)),
                med_ratio=float(np.median(ratio)), med_pred=float(np.median(pred)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-config', type=int, default=40, dest='n_config')
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--steps', type=int, default=15000)
    ap.add_argument('--max-draws', type=int, default=4000, dest='max_draws')
    ap.add_argument('--out', default='wave23_architectures.json')
    ap.add_argument('--delay-mode', action='store_true', dest='delay_mode',
                    help='with --crossover: sweep a pure delay on the gate instead of '
                         'a first-order lag, to place a report-contingent gate')
    ap.add_argument('--crossover', action='store_true',
                    help='also sweep the ramp time constant and locate '
                         'the tau/D crossover in each architecture')
    args = ap.parse_args()

    results = {}
    for arch in (A_LCA, A_DIV, A_SIG, A_INP):
        name = ARCH_NAME[arch]
        rng = np.random.default_rng(2300 + arch)
        print(f"\n{'=' * 74}\n{name}\n{'=' * 74}")
        found, tried = [], 0
        while len(found) < args.n_config and tried < args.max_draws:
            tried += 1
            p = draw(arch, rng)
            e = eligible(arch, p, args)
            if e is not None:
                found.append((p, e))
        yld = len(found) / max(tried, 1)
        print(f"  search: {len(found)} configurations from {tried} draws "
              f"(yield {100 * yld:.1f}%)")
        if not found:
            print("  no rivalry-producing configurations found; family untestable "
                  "at these ranges")
            results[name] = dict(n=0, yield_frac=0.0)
            continue
        cvs = [e['cv'] for _, e in found]
        print(f"  median CV {np.median(cvs):.3f}, "
              f"median episodes/seed {np.median([e['n_ep'] for _, e in found]):.0f}")

        rows = {k: [] for k in ('gated_1x', 'gated_2x', 'ungated_1x',
                                'antigated_2x')}
        for p, e in found:
            ref = 0.5 * p['lam'] * e['actA']
            base = cell(arch, p, 0.5, 0.5, 0.0, G_NONE, args.seeds, args.steps, 'b')
            dA0, dB0 = base[0], base[1]
            if not (np.isfinite(dA0) and np.isfinite(dB0)):
                continue
            for tag, (gate, amp) in dict(gated_1x=(G_ON, 1.0),
                                         gated_2x=(G_ON, 2.0),
                                         ungated_1x=(G_UNGATED, 1.0),
                                         antigated_2x=(G_OFF, 2.0)).items():
                r = cell(arch, p, 0.5, 0.5, amp * ref, gate, args.seeds,
                         args.steps, tag)
                pA = 100 * (r[0] - dA0) / dA0 if dA0 > 0 else np.nan
                pB = 100 * (r[1] - dB0) / dB0 if dB0 > 0 else np.nan
                rows[tag].append(dict(pA=float(pA), pB=float(pB),
                                      cv=float(r[2]), gam=float(p['gam']),
                                      T0=float(0.5 * (dA0 + dB0))))

        print(f"\n  {'condition':>14} {'n':>4} {'attended':>9} {'competitor':>11} "
              f"{'pB>0':>10} {'ratio':>8} {'CV':>7}")
        summ = {}
        for tag in ('gated_1x', 'gated_2x', 'ungated_1x', 'antigated_2x'):
            rr = [x for x in rows[tag] if np.isfinite(x['pA']) and np.isfinite(x['pB'])]
            if not rr:
                continue
            pa = np.array([x['pA'] for x in rr])
            pb = np.array([x['pB'] for x in rr])
            rat = pb[np.abs(pa) > 1e-9] / pa[np.abs(pa) > 1e-9]
            summ[tag] = dict(n=len(rr), med_pA=float(np.median(pa)),
                             med_pB=float(np.median(pb)),
                             n_pos=int(np.sum(pb > 0)),
                             med_ratio=float(np.median(rat)) if rat.size else None,
                             med_cv=float(np.median([x['cv'] for x in rr])))
            s = summ[tag]
            print(f"  {tag:>14} {s['n']:>4} {s['med_pA']:>+9.1f} {s['med_pB']:>+11.1f} "
                  f"{s['n_pos']:>4}/{s['n']:<4} "
                  f"{(s['med_ratio'] if s['med_ratio'] is not None else float('nan')):>+8.3f} "
                  f"{s['med_cv']:>7.3f}")

        g = summ.get('gated_1x')
        u = summ.get('ungated_1x')
        if g and u:
            # sample-size aware: the two proportions must lie on opposite sides of
            # chance with 95% intervals excluding it. A fixed threshold cannot be
            # met at small n and would report a false negative.
            gl, gh = wilson(g['n_pos'], g['n'])
            ul, uh = wilson(u['n_pos'], u['n'])
            holds = bool(gl > 0.5 and uh < 0.5)
            direction = bool(g['med_pB'] > 0 > u['med_pB'])
            print(f"\n  gated positive in {g['n_pos']}/{g['n']} "
                  f"[{100*gl:.0f}%, {100*gh:.0f}%];  "
                  f"ungated positive in {u['n_pos']}/{u['n']} "
                  f"[{100*ul:.0f}%, {100*uh:.0f}%]")
            print(f"  medians opposite in sign: {'yes' if direction else 'NO'}")
            print(f"  SIGN REVERSAL: {'PRESENT' if holds else 'not established'}"
                  + ("" if holds or g['n'] >= 30 else
                     f"  (n = {g['n']} is too small to establish it either way)"))
            summ['sign_reversal'] = holds
            summ['medians_opposite'] = direction
            summ['wilson_gated'] = [gl, gh]
            summ['wilson_ungated'] = [ul, uh]
        geo = {t: geometry_check(rows, t, seed=arch) for t in ('gated_1x', 'gated_2x')}
        g1 = geo.get('gated_1x')
        if g1:
            print(f"\n  GEOMETRY CHECK, gated 1x, n = {g1['n']}")
            print(f"    Spearman(ratio, exp(-gamma T))  {g1['sp_pred']:+.3f}   "
                  f"permuting gamma: 95th pct {g1['perm_95']:+.3f}, p = {g1['perm_p']:.4f}")
            print(f"    Spearman(ratio, -T alone)       {g1['sp_T']:+.3f}")
            print(f"    Spearman(ratio, -gamma alone)   {g1['sp_gam']:+.3f}")
            print(f"    median ratio / prediction       {g1['calib']:.2f}   "
                  f"(median ratio {g1['med_ratio']:+.3f}, median prediction {g1['med_pred']:.3f})")
        summ['geometry'] = geo
        results[name] = dict(n=len(found), yield_frac=float(yld),
                             median_cv=float(np.median(cvs)), summary=summ,
                             rows=rows, params=found)

    print(f"\n{'=' * 74}\nSUMMARY\n{'=' * 74}")
    print(f"  {'architecture':>16} {'n':>4} {'yield':>7} {'medCV':>7} "
          f"{'gated pB':>10} {'ungated pB':>11} {'reversal':>10}")
    for name, r in results.items():
        if not r.get('n'):
            print(f"  {name:>16} {0:>4} {'--':>7} {'--':>7} {'--':>10} "
                  f"{'--':>11} {'untestable':>10}")
            continue
        s = r['summary']
        g, u = s.get('gated_1x', {}), s.get('ungated_1x', {})
        print(f"  {name:>16} {r['n']:>4} {100*r['yield_frac']:>6.1f}% "
              f"{r['median_cv']:>7.3f} {g.get('med_pB', float('nan')):>+10.1f} "
              f"{u.get('med_pB', float('nan')):>+11.1f} "
              f"{('yes' if s.get('sign_reversal') else ('signs only' if s.get('medians_opposite') else 'NO')):>10}")

    print(f"\n{'=' * 74}\nGEOMETRY ACROSS FAMILIES: does exp(-gamma T) explain the spread?\n{'=' * 74}")
    print(f"  {'architecture':>16} {'n':>4} {'Spearman':>9} {'perm p':>8} {'vs T':>7} "
          f"{'vs gam':>7} {'obs/pred':>9} {'med obs':>8} {'med pred':>9}")
    fam_obs, fam_pred, pooled_r, pooled_p = [], [], [], []
    for name, r in results.items():
        g1 = (r.get('summary') or {}).get('geometry', {}).get('gated_1x')
        if not g1:
            continue
        print(f"  {name:>16} {g1['n']:>4} {g1['sp_pred']:>+9.3f} {g1['perm_p']:>8.4f} "
              f"{g1['sp_T']:>+7.3f} {g1['sp_gam']:>+7.3f} {g1['calib']:>9.2f} "
              f"{g1['med_ratio']:>+8.3f} {g1['med_pred']:>9.3f}")
        fam_obs.append(g1['med_ratio']); fam_pred.append(g1['med_pred'])
        for x in r['rows'].get('gated_1x', []):
            if (np.isfinite(x['pA']) and np.isfinite(x['pB']) and abs(x['pA']) > 1e-9
                    and np.isfinite(x.get('T0', np.nan))):
                pooled_r.append(x['pB'] / x['pA'])
                pooled_p.append(np.exp(-x['gam'] * x['T0']))
    if len(fam_obs) >= 3:
        print(f"\n  pooled across families: Spearman {_sp(pooled_r, pooled_p):+.3f} "
              f"on {len(pooled_r)} configurations")
        print(f"  family medians, observed vs predicted: Spearman "
              f"{_sp(fam_obs, fam_pred, minn=3):+.3f} on {len(fam_obs)} families "
              f"(four points: read the ordering, not the coefficient)")
        print("""
  HOW TO READ THIS. Within a family, a Spearman well above the permutation 95th
  percentile means pairing each configuration's own gamma and T matters, so the
  combination is doing work that neither part does. If that holds in every family the
  prediction is architecture-general. If the family medians also line up, the
  thirtyfold spread in coupling ratio across architectures is explained rather than
  merely reported. Where a family fails, the claim in Section 4.4 must be scoped to
  the families where it holds, and the failure reported.""")

    n_arch = sum(1 for r in results.values() if r.get('n'))
    n_rev = sum(1 for r in results.values()
                if r.get('summary', {}).get('sign_reversal'))
    print(f"""
  {n_rev} of {n_arch} testable architectures show the sign reversal.
""")
    if n_arch and n_rev == n_arch:
        print("""  THE RESULT IS ARCHITECTURAL. Gated delivery lengthens the competitor and
  ungated delivery shortens it in every family tested, including the divisive
  normalisation case that Section 1.2 exempts from the paper's input-gain argument
  and the sigmoid firing-rate case whose analyses the paper's Proposition IV
  discussion relies on. The claim can be stated for competitive networks with
  adaptation rather than for one accumulator, and Section 6's admission that this
  had not been tested can be replaced with the test. State clearly that these are
  generic family representatives and not reimplementations of specific published
  models -- the claim is robustness to architectural kind, not reproduction of
  anyone's parameter set.""")
    elif n_rev == 0:
        print("""  THE RESULT IS SPECIFIC TO THE COMPANION PAPER'S ARCHITECTURE. That is a
  serious limitation and must be stated as one: the central claim then concerns
  subtractive inhibition with output adaptation, not rivalry. Check the search
  yields before concluding it -- a family that barely rivals at these parameter
  ranges has not been given a fair test.""")
    else:
        print("""  MIXED. The families where it holds and where it fails should be compared for
  what distinguishes them: whether suppression is subtractive or divisive, and
  whether adaptation acts on the output or on the input. That comparison is more
  informative than either a uniform positive or a uniform negative, because it
  identifies which architectural feature the result depends on.""")

    # ------------------------------------------------------------------
    # crossover phase: is tau/D ~ 0.59 architecture-invariant?
    # ------------------------------------------------------------------
    if args.crossover:
        print(f"\n{'=' * 74}")
        print("CROSSOVER IN tau/D: does the predicted 0.59 generalise?")
        print("=" * 74)
        print("""
  Section 4.7.8 establishes that the SIGN reversal is architectural while the
  coupling magnitudes are not, spanning fifteen-fold. The crossover in tau/D is a
  magnitude. If it varies as widely as the ratio does, the experimental prediction
  is directional only -- lengthen the ramp enough and the sign inverts -- and the
  ramp durations for a test cannot be chosen from 0.59. If it is invariant, the
  prediction carries a number.

  tau is set per configuration as a fraction of that configuration's own mean
  dominance duration, so the sweep samples tau/D directly rather than tau.
""")
        FR = (0.02, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.50)
        cross = {}
        for arch in (A_LCA, A_DIV, A_SIG, A_INP):
            name = ARCH_NAME[arch]
            cfgs = results.get(name, {}).get('params', [])
            if not cfgs:
                continue
            per_cfg, curves = [], {f: [] for f in FR}
            for p, e in cfgs:
                ref = 0.5 * p['lam'] * e['actA']
                base = cell(arch, p, 0.5, 0.5, 0.0, G_NONE, args.seeds,
                            args.steps, 'xb')
                dA0, dB0, D = base[0], base[1], base[0]
                if not (np.isfinite(dA0) and np.isfinite(dB0) and D > 1):
                    continue
                vals = []
                for k, f in enumerate(FR):
                    r = cell(arch, p, 0.5, 0.5, ref, G_ON, args.seeds,
                             args.steps, f'x{k}',
                             tau=(1.0 if args.delay_mode else max(1.0, f * D)),
                             delay=(int(round(f * D)) if args.delay_mode else 0))
                    v = 100 * (r[1] - dB0) / dB0 if dB0 > 0 else np.nan
                    vals.append(v)
                    if np.isfinite(v):
                        curves[f].append(v)
                # first downward zero crossing, linearly interpolated
                xc = None
                for k in range(len(FR) - 1):
                    a, b = vals[k], vals[k + 1]
                    if np.isfinite(a) and np.isfinite(b) and a > 0 >= b:
                        xc = FR[k] + (FR[k + 1] - FR[k]) * a / (a - b)
                        break
                if xc is not None:
                    per_cfg.append(xc)
            med = {f: float(np.nanmedian(curves[f])) if curves[f] else float('nan')
                   for f in FR}
            print(f"  {name}   n with a crossing: {len(per_cfg)}/{len(cfgs)}")
            print("    " + "  ".join(f"{f:>6.2f}" for f in FR))
            print("    " + "  ".join(f"{med[f]:>+6.1f}" for f in FR)
                  + "   <- median competitor % change")
            if per_cfg:
                v = np.asarray(per_cfg)
                q1, q3 = np.percentile(v, [25, 75])
                print(f"    crossing at tau/D = {np.median(v):.2f} "
                      f"IQR [{q1:.2f}, {q3:.2f}]\n")
                cross[name] = dict(n=len(per_cfg), median=float(np.median(v)),
                                   iqr=[float(q1), float(q3)],
                                   curve=med, per_config=list(map(float, v)))
            else:
                print("    no configuration crossed zero in the swept range\n")
                cross[name] = dict(n=0, curve=med)
        results['crossover'] = cross
        meds = [c['median'] for c in cross.values() if c.get('n')]
        if len(meds) >= 2:
            lo, hi = min(meds), max(meds)
            print(f"  crossings across architectures: "
                  + ", ".join(f"{n.split()[0]} {c['median']:.2f}"
                              for n, c in cross.items() if c.get('n')))
            print(f"  spread {lo:.2f} to {hi:.2f}, a factor of {hi/max(lo,1e-9):.1f}")
            if hi / max(lo, 1e-9) <= 2.0:
                print("""
  INVARIANT to within a factor of two. The crossover is a property of the
  competition rather than of the architecture, so the experimental prediction
  carries a number and ramp durations can be chosen from it. State the range
  across architectures rather than the single value from one.""")
            else:
                print("""
  NOT INVARIANT. The crossover varies as the coupling magnitudes do, so it is a
  magnitude and not a structural feature. The experimental prediction is
  directional -- lengthening the ramp far enough inverts the sign -- and the design
  must sweep tau/D broadly rather than target 0.59. Section 4.7.4 and the design
  note both need amending, and the paper should report the across-architecture
  spread instead of the single figure.""")

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=1, default=float)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
