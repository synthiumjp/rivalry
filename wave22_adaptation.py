"""
wave22_adaptation.py — superlinear adaptation law and Levelt's second proposition.

Tests the diagnosis in manuscript 5.4: that the model cannot reproduce Levelt II
because activation-proportional adaptation scales with the increment-induced gain
instead of cancelling it, and that cancelling it requires adaptation superlinear
in activation.

    Eq 2 becomes   a_i(t+1) = (1-gam) a_i(t) + kap' * x_i(t)**p
    with           kap' = kap / xbar**(p-1)

The renormalisation matches baseline adaptation drive at the operating point, so p
varies the CURVATURE of the law and not its level. An unnormalised arm is also run.

USAGE (Windows, from C:\\crewther with venv active)

    venv\\Scripts\\activate
    python wave22_adaptation.py --verify              # do this first, see below
    python wave22_adaptation.py --quick               # ~2 min smoke test
    python wave22_adaptation.py --block A             # eligibility per p
    python wave22_adaptation.py --block B             # Levelt II search
    python wave22_adaptation.py --block C             # gate timing per p
    python wave22_adaptation.py --analyse             # reanalysis only, no simulation

READ BEFORE RUNNING
-------------------
1. --verify is not optional. It cross-checks this file's kernel against
   wave2_campaign.run_trace at p=1 on shared configurations and seeds. If they
   disagree beyond Monte Carlo, every number below is uninterpretable and the
   script aborts. An exploratory reimplementation of this model produced 7/200
   Levelt II hits at p=1 where the real pipeline produces 0/100, so kernel
   agreement is the load-bearing control here, not a formality.

2. Block A re-derives the eligible pool under each p. Do NOT inherit the 762.
   The exponent shifts CV downward (median CV fell to ~0.35 at p>=1.75 in the
   exploratory run), so the eligibility window selects a different region.

3. The reference increment is 0.5 * lambda * mean ACTIVATION, measured in a
   dedicated baseline run. It is NOT summarise()['own'], which is duration
   (~50-150) and would make every increment ~100x too large. The script asserts
   the magnitude and aborts if it is outside [0.005, 0.5].

4. All ratio statistics are reported under both aggregation conventions.
   Alternation rate and transition counts are reported under the 5-timestep
   minimum as well as raw.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# pipeline imports. wave6_robustness is the canonical source of make_seed.
# --------------------------------------------------------------------------
try:
    import wave6_robustness as w6
except ImportError:
    sys.exit("wave6_robustness.py not importable. Run from C:\\crewther.")
try:
    import wave2_campaign as w2
except ImportError:
    sys.exit("wave2_campaign.py not importable. Run from C:\\crewther.")

try:
    from numba import njit
except ImportError:                                        # pragma: no cover
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not a else a[0]
    print("WARNING: numba unavailable, falling back to pure Python. Use --quick.")

THETA = 0.05
XMAX = 5.0
BURN = 500
MIN_DUR = 5                     # transitional/sustained boundary, manuscript 4.3
SPLIT_INDET = 1                 # set from block V's verdict before running A-C
PROGRESS_EVERY = 100            # block A progress line every N configurations
P_VALUES = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
MULTS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
GATE_P = (1.0, 2.0, 3.0)
LEVELT_A = 10.0                 # |% change attended| <=
LEVELT_B = (-45.0, -15.0)       # % change competitor within
ABS_THRESH_FRAC = 0.5           # absolute criterion: x_B > frac * baseline mean x_B

# gate codes
G_NONE, G_UNGATED, G_ON, G_OFF, G_REPLAY = 0, 1, 2, 3, 4
# adaptation modulation: `inc` is reinterpreted as a fractional reduction of the
# attended channel's adaptation gain, in (0, 1). No input increment is delivered.
G_ADAPT_ON, G_ADAPT_ALL = 5, 6

# integer block IDs for w6.make_seed, which does int(block).
# 2200-block reserved for wave22 to avoid collision with waves 2-21.
BLK_V = 2200
BLK_A_ACT, BLK_A_BASE, BLK_A_SWEEP = 2210, 2211, 2212
BLK_B_ACT, BLK_B_BASE, BLK_B_SWEEP = 2220, 2221, 2222
BLK_C_ACT, BLK_C_BASE = 2230, 2231
BLK_C_REC_ON, BLK_C_REC_OFF, BLK_C_COND = 2232, 2233, 2234
BLK_D_ACT, BLK_D_BASE, BLK_D_SWEEP, BLK_D_GATE = 2240, 2241, 2242, 2243
BLK_E_ACT, BLK_E_SWEEP = 2250, 2251
BLK_F_ACT, BLK_F_BASE, BLK_F_P4, BLK_F_L2, BLK_F_GATE = 2260, 2261, 2262, 2263, 2264
BLK_G_ACT, BLK_G_BASE, BLK_G_COND = 2270, 2271, 2272
BLK_H_ACT, BLK_H_BASE, BLK_H_P4 = 2280, 2281, 2282
BLK_H_L2, BLK_H_GATE, BLK_H_UNG = 2283, 2284, 2285
BLK_I_BASE, BLK_I_GOAL, BLK_I_WIN, BLK_I_COND = 2290, 2291, 2292, 2293
BLK_J_BASE = 2300
BLK_K_ACT, BLK_K_BASE, BLK_K_SWEEP, BLK_K_P4 = 2310, 2311, 2312, 2313
BLK_L_SCREEN, BLK_L_BASE, BLK_L_COND = 2320, 2321, 2322
BLK_M_BASE, BLK_M_REC, BLK_M_COND = 2330, 2331, 2332
BLK_N_BASE, BLK_N_REC = 2340, 2341
BLK_O_BASE, BLK_O_REC = 2350, 2351
BLK_Q_BASE, BLK_Q_BURST, BLK_Q_REC, BLK_Q_UNF = 2360, 2361, 2362, 2363
BLK_R_GEO, BLK_R_BASE, BLK_R_COND = 2370, 2371, 2372

# kernel output slots
(O_DURA, O_DURB, O_CVA, O_CVB, O_NA, O_NB, O_SW, O_SWF,
 O_ACTA, O_ACTB, O_FLOOR, O_DOSE, O_PREDA, O_DURAF, O_DURBF, O_ABSOCC,
 O_SPA, O_SPA10, O_SPA25, O_SPA50, O_DURPOOL, O_CEIL) = range(22)
N_OUT = 22
# O_ABSOCC: occupancy fraction with x_B above a fixed threshold. This is NOT the
# absolute dominance criterion of Section 3.5, which needs episode extraction under
# that rule and yields a duration. An earlier version wrote pooled mean duration into
# this slot and two blocks read it as though it were the absolute criterion; both are
# corrected and neither figure reached the manuscript.
# O_DURPOOL: pooled mean episode duration across both channels, difference criterion.
# O_CEIL: ceiling occupancy, for the figure Section 2.1 quotes.
# O_SPA*: mean activation of channel A during episodes in which B is dominant --
# the 'suppressed-phase activation' of Section 4.7.1. O_SPA averages over the whole
# episode, which makes the averaging window an outcome; O_SPA10/25/50 average over
# a fixed number of timesteps from episode onset, which does not.


# ==========================================================================
# kernel
# ==========================================================================
# adaptation update order. The pipeline (gc_lca_phase1_grid.simulate_gclca and
# wave2_campaign.run_trace) both compute a(t+1) from the POST-update x(t+1):
#     x <- rectify(new);  a <- (1-gam)*a + kap*x
# Manuscript Eq 2 states a(t+1) = (1-gam)a(t) + kap*x(t), which is the lagged
# form. The two differ by O(alpha*kappa) in the adaptation feedback loop:
# negligible for weak adaptation, and enough to change the dynamical regime for
# strong adaptation. On config 0 the pipeline order gives CV 0.5002 and the
# lagged form 0.1082. ADAPT_PIPELINE = 1 reproduces the frozen results.
ADAPT_PIPELINE = 1
X_INIT = 0.1                    # pipeline initialises both channels at 0.1


@njit(cache=True)
def _trace(lam, beta, alpha, sigma, gam, kap, p,
           S_A, S_B, inc, gate, abs_thresh,
           n_steps, seed, schedule, record, split_indet, theta,
           adapt_pipeline, delta, mu, g_a):
    """One (config, seed) trace. Returns a length-16 summary vector.

    Mirrors the pipeline's loop structure exactly so that p=1 with
    adapt_pipeline=1 reproduces phase1_grid_results.json to the stored digits:
      * state is recorded at the TOP of the iteration, before the update
      * both channels initialise at X_INIT
      * every episode is committed including the final in-progress one, then
        the first and last are dropped (equivalent to durations[1:-1])

    gate: 0 none, 1 ungated, 2 gated on A dominant, 3 gated on A suppressed,
          4 replay `schedule` (yoked control).
    record: if 1, write the realised gate mask into `schedule` for later replay.
    theta: dominance margin.
    split_indet: if 1, |x_A - x_B| <= theta ends the current episode.
    adapt_pipeline: 1 -> a(t+1) from x(t+1) (pipeline); 0 -> from x(t) (Eq 2).
    g_a: goal signal on channel A, per Equation 1. Enters A's recurrent coefficient as
          (1 - lam + g_a) unconditionally, unlike `delta`, which is a symmetric
          dominance-phase gain applied to whichever channel is currently dominant.
          Blocks measuring the goal signal must use this and not `delta`.
    mu: weight on inhibition-driven adaptation. Adaptation becomes
          a_i <- (1-gam) a_i + kap*(x_i**p + mu*beta*x_j), so a channel adapts to
          the inhibition it RECEIVES as well as to its own activation. mu = 0
          recovers the model used everywhere else. A suppressed channel receives
          strong inhibition, so it stays adapted rather than de-adapting, which is
          the mechanism by which the competitor currently lengthens under a
          dominance-gated increment.
    delta: dominance-phase self-gain, added to the recurrent coefficient of
          whichever channel is currently dominant. An architectural feature,
          symmetric across channels, distinct from the goal signal G which is
          applied to one channel as a manipulation. delta = 0 recovers the model
          used everywhere else in this file. Must satisfy delta < lam for the
          effective leak of the dominant channel to stay positive.
    """
    np.random.seed(seed)
    out = np.zeros(N_OUT, dtype=np.float64)

    xA = X_INIT
    xB = X_INIT
    aA = 0.0
    aB = 0.0
    rec = 1.0 - lam

    cur = 0
    cur_len = 0
    pend = 0
    pend_len = 0
    seen_first = 0
    last_nz = 0
    n_sw = 0
    n_swf = 0
    prev_state = 0

    sA = 0.0
    qA = 0.0
    cA = 0
    sB = 0.0
    qB = 0.0
    cB = 0
    sAf = 0.0
    cAf = 0
    sBf = 0.0
    cBf = 0
    actA = 0.0
    actB = 0.0
    floor_n = 0
    dose = 0.0
    domA = 0
    domB = 0
    absB = 0
    ceil_n = 0
    n_post = 0
    spa = 0.0
    spa_n = 0
    spa10 = 0.0
    spa10_n = 0
    spa25 = 0.0
    spa25_n = 0
    spa50 = 0.0
    spa50_n = 0

    for t in range(n_steps):
        # ---- bookkeeping on the state as recorded at the top of the loop ----
        if t >= BURN:
            n_post += 1
            dd = xA - xB
            if dd > theta:
                new = 1
            elif dd < -theta:
                new = -1
            else:
                new = 0 if split_indet == 1 else cur

            if new != cur:
                if cur != 0:
                    if seen_first == 0:
                        seen_first = 1
                    else:
                        if pend != 0:
                            dl = float(pend_len)
                            if pend == 1:
                                sA += dl
                                qA += dl * dl
                                cA += 1
                                if pend_len >= MIN_DUR:
                                    sAf += dl
                                    cAf += 1
                            else:
                                sB += dl
                                qB += dl * dl
                                cB += 1
                                if pend_len >= MIN_DUR:
                                    sBf += dl
                                    cBf += 1
                            if pend_len >= MIN_DUR:
                                if prev_state != 0 and pend != prev_state:
                                    n_swf += 1
                                prev_state = pend
                        pend = cur
                        pend_len = cur_len
                cur_len = 1
            else:
                cur_len += 1
            cur = new

            if new != 0:
                if last_nz != 0 and new != last_nz:
                    n_sw += 1
                last_nz = new
            if dd > theta:
                domA += 1
            elif dd < -theta:
                domB += 1
            actA += xA
            actB += xB
            if (xA if xA < xB else xB) <= 0.0:
                floor_n += 1
            if xB > abs_thresh:
                absB += 1
            if (xA if xA > xB else xB) >= XMAX:
                ceil_n += 1
            if cur == -1:                      # competitor B dominant
                spa += xA
                spa_n += 1
                if cur_len <= 10:
                    spa10 += xA
                    spa10_n += 1
                if cur_len <= 25:
                    spa25 += xA
                    spa25_n += 1
                if cur_len <= 50:
                    spa50 += xA
                    spa50_n += 1

        # ---- gate ----
        d = xA - xB
        if gate == G_NONE:
            on = 0
        elif gate == G_UNGATED:
            on = 1
        elif gate == G_ON:
            on = 1 if d > theta else 0
        elif gate == G_OFF:
            on = 1 if d < -theta else 0
        elif gate == G_ADAPT_ON:
            on = 1 if d > theta else 0
        elif gate == G_ADAPT_ALL:
            on = 1
        else:
            on = 1 if schedule[t] == 1 else 0
        if record == 1:
            schedule[t] = np.uint8(on)
        # adaptation gates modulate kappa rather than the input
        if gate == G_ADAPT_ON or gate == G_ADAPT_ALL:
            add = 0.0
            kapA = kap * (1.0 - inc) if (gate == G_ADAPT_ALL or on == 1) else kap
        else:
            add = inc if on == 1 else 0.0
            kapA = kap
        if t >= BURN:
            dose += add

        # ---- update ----
        nA = np.random.normal(0.0, sigma)
        nB = np.random.normal(0.0, sigma)
        recA = rec + g_a
        recB = rec
        if delta != 0.0:
            if d > theta:
                recA = recA + delta
            elif d < -theta:
                recB = recB + delta
        newA = recA * xA + S_A + add - beta * xB - alpha * aA + nA
        newB = recB * xB + S_B - beta * xA - alpha * aB + nB
        if newA < 0.0:
            newA = 0.0
        elif newA > XMAX:
            newA = XMAX
        if newB < 0.0:
            newB = 0.0
        elif newB > XMAX:
            newB = XMAX

        if adapt_pipeline == 1:
            xA = newA
            xB = newB
            dA = xA if p == 1.0 else xA ** p
            dB = xB if p == 1.0 else xB ** p
            if mu != 0.0:
                dA = dA + mu * beta * xB
                dB = dB + mu * beta * xA
            aA = (1.0 - gam) * aA + kapA * dA
            aB = (1.0 - gam) * aB + kap * dB
        else:
            dA = xA if p == 1.0 else xA ** p
            dB = xB if p == 1.0 else xB ** p
            if mu != 0.0:
                dA = dA + mu * beta * xB
                dB = dB + mu * beta * xA
            aA = (1.0 - gam) * aA + kapA * dA
            aB = (1.0 - gam) * aB + kap * dB
            xA = newA
            xB = newB

    # final in-progress episode enters the buffer, so it is the one dropped
    if cur != 0 and cur_len > 0 and seen_first == 1:
        if pend != 0:
            dl = float(pend_len)
            if pend == 1:
                sA += dl
                qA += dl * dl
                cA += 1
                if pend_len >= MIN_DUR:
                    sAf += dl
                    cAf += 1
            else:
                sB += dl
                qB += dl * dl
                cB += 1
                if pend_len >= MIN_DUR:
                    sBf += dl
                    cBf += 1

    inv = 1.0 / n_post if n_post > 0 else 0.0
    nAB = cA + cB
    if nAB > 0:
        mAll = (sA + sB) / nAB
        vAll = (qA + qB) / nAB - mAll * mAll
    else:
        mAll = np.nan
        vAll = np.nan

    if cA > 0:
        mA = sA / cA
        vA = qA / cA - mA * mA
        out[O_DURA] = mA
        out[O_CVA] = (math.sqrt(vA) / mA) if (vA > 0.0 and mA > 0.0) else 0.0
    else:
        out[O_DURA] = np.nan
        out[O_CVA] = np.nan
    if cB > 0:
        mB = sB / cB
        vB = qB / cB - mB * mB
        out[O_DURB] = mB
        out[O_CVB] = (math.sqrt(vB) / mB) if (vB > 0.0 and mB > 0.0) else 0.0
    else:
        out[O_DURB] = np.nan
        out[O_CVB] = np.nan

    out[O_NA] = cA
    out[O_NB] = cB
    # episodes minus one, matching gc_lca_phase1_grid's
    # n_switches = max(0, len(channels) - 1). This counts an
    # A -> indeterminate -> A excursion as a switch, as the pipeline does.
    out[O_SW] = nAB - 1 if nAB > 0 else 0
    out[O_SWF] = n_swf
    out[O_ACTA] = actA * inv
    out[O_ACTB] = actB * inv
    out[O_FLOOR] = floor_n * inv
    out[O_DOSE] = dose
    out[O_PREDA] = (domA / (domA + domB)) if (domA + domB) > 0 else np.nan
    out[O_DURAF] = (sAf / cAf) if cAf > 0 else np.nan
    out[O_DURBF] = (sBf / cBf) if cBf > 0 else np.nan
    out[O_ABSOCC] = absB * inv
    out[O_DURPOOL] = mAll if not np.isnan(mAll) else np.nan
    out[O_CEIL] = ceil_n * inv
    out[O_SPA] = (spa / spa_n) if spa_n > 0 else np.nan
    out[O_SPA10] = (spa10 / spa10_n) if spa10_n > 0 else np.nan
    out[O_SPA25] = (spa25 / spa25_n) if spa25_n > 0 else np.nan
    out[O_SPA50] = (spa50 / spa50_n) if spa50_n > 0 else np.nan
    return out


def LEVEL(*parts):
    """Injective integer index over a tuple of condition parameters.

    Replaces arithmetic blends such as int(p*100 + dfrac*10), where p = 1.0 with
    dfrac = 0.0 and dfrac = 0.05 both round to 100 and therefore share a seed. That is
    the third instance in this project of the defect Section A.3 documents twice.
    """
    key = tuple(round(float(x), 6) if isinstance(x, (int, float, np.floating, np.integer))
                else str(x) for x in parts)
    if key not in _LEVEL_CACHE:
        _LEVEL_CACHE[key] = len(_LEVEL_CACHE) + 1
    return _LEVEL_CACHE[key]


_LEVEL_CACHE = {}


def canonical_seed(block, config, regime, level, seed):
    """w6.make_seed with defensive coercion.

    w6.make_seed does int() on every argument, so all five must be integers.
    Its return type is not documented here, so accept an int, an array, or a
    SeedSequence and reduce to a single value in numpy's legacy seed range.
    """
    v = w6.make_seed(int(block), int(config), int(regime), int(level), int(seed))
    if isinstance(v, np.random.SeedSequence):
        v = v.generate_state(1, dtype=np.uint32)[0]
    v = np.asarray(v).reshape(-1)[0]
    # % 2**32 is injective on the uint32 SeedSequence output; the previous
    # % (2**31 - 1) folded roughly half the space onto the other half.
    return int(v) % (2 ** 32)



@njit(cache=True)
def _trace_full(lam, beta, alpha, sigma, gam, kap, S_A, S_B, n_steps, seed,
                adapt_pipeline, ta, tb):
    """Mirror of _trace's update with no manipulation, writing the traces out.

    Exists only for block P. Any divergence between this and _trace is a bug in one
    of them; any divergence between this and wave2_campaign.run_trace at the same
    seed is a difference in the model.
    """
    np.random.seed(seed)
    xA = X_INIT
    xB = X_INIT
    aA = 0.0
    aB = 0.0
    rec = 1.0 - lam
    for t in range(n_steps):
        ta[t] = xA
        tb[t] = xB
        nA = np.random.normal(0.0, sigma)
        nB = np.random.normal(0.0, sigma)
        newA = rec * xA + S_A - beta * xB - alpha * aA + nA
        newB = rec * xB + S_B - beta * xA - alpha * aB + nB
        if newA < 0.0:
            newA = 0.0
        elif newA > XMAX:
            newA = XMAX
        if newB < 0.0:
            newB = 0.0
        elif newB > XMAX:
            newB = XMAX
        if adapt_pipeline == 1:
            xA = newA
            xB = newB
            aA = (1.0 - gam) * aA + kap * xA
            aB = (1.0 - gam) * aB + kap * xB
        else:
            aA = (1.0 - gam) * aA + kap * xA
            aB = (1.0 - gam) * aB + kap * xB
            xA = newA
            xB = newB


def run_cell(prm, p, S_A, S_B, inc, gate, abs_thresh, n_steps, n_seeds,
             block, level, regime=0, schedule=None, record=False,
             split_indet=1, theta=None, adapt=None, delta=0.0, mu=0.0,
             g_a=0.0):
    """Run n_seeds traces for one configuration. Seeds from w6.make_seed."""
    res = np.empty((n_seeds, N_OUT))
    sched = (np.zeros(n_steps, dtype=np.uint8) if schedule is None
             else schedule)
    for s in range(n_seeds):
        seed = canonical_seed(block, prm['idx'], regime, level, s)
        res[s] = _trace(prm['lam'], prm['beta'], prm['alpha'], prm['sigma'],
                        prm['gam'], prm['kap'], p, S_A, S_B, inc, gate,
                        abs_thresh, n_steps, seed, sched,
                        1 if (record and s == 0) else 0, int(split_indet),
                        float(THETA if theta is None else theta),
                        int(ADAPT_PIPELINE if adapt is None else adapt),
                        float(delta), float(mu), float(g_a))
    return res


# ==========================================================================
# helpers
# ==========================================================================
def sm(res, slot):
    """Seed-mean of one output slot, nan-safe."""
    v = res[:, slot]
    return float(np.nanmean(v)) if np.isfinite(v).any() else float('nan')


def pct(new, old):
    return 100.0 * (new - old) / old if (old and old > 0) else float('nan')


def both_conventions(num, den):
    """Return (median of per-config ratios, ratio of medians). Trap 4.1."""
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    ok = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-9)
    per = float(np.median(num[ok] / den[ok])) if ok.any() else float('nan')
    pooled = (float(np.median(num[ok]) / np.median(den[ok]))
              if ok.any() and abs(np.median(den[ok])) > 1e-9 else float('nan'))
    return per, pooled


def wilson(k, n):
    try:
        return w6.wilson(k, n)
    except Exception:
        if n == 0:
            return (float('nan'), float('nan'))
        z, ph = 1.96, k / n
        c = (ph + z * z / (2 * n)) / (1 + z * z / n)
        h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return (max(0.0, c - h), min(1.0, c + h))


def fnum(v, default=float('nan')):
    """Coerce to float, tolerating None, '', and non-numeric.

    phase1 stores levelt_rho and rivalry_metrics as null for configurations that
    did not produce rivalry, so every read of a record field must tolerate it.
    """
    if v is None or isinstance(v, (dict, list)):
        return default
    if isinstance(v, bool):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


ALIASES = dict(lam=('lambda', 'lam', 'leak'),
               beta=('beta', 'b'),
               alpha=('alpha', 'a'),
               sigma=('sigma', 'sig', 'noise'),
               gam=('gamma', 'gam', 'g'),
               kap=('kappa', 'kap', 'k'))


def as_list(obj, *prefer):
    """Normalise a pipeline return value to a list of records.

    Handles: a list; a dict with a known list-valued key; a dict of id -> record.
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in prefer + ('config_results', 'configs', 'results',
                             'selected', 'per_config', 'rows'):
            v = obj.get(key)
            if isinstance(v, list):
                return v
        vals = list(obj.values())
        if vals and all(isinstance(v, dict) for v in vals):
            try:                                  # preserve numeric key order
                order = sorted(obj.keys(), key=lambda k: int(k))
            except (TypeError, ValueError):
                order = list(obj.keys())
            return [obj[k] for k in order]
    raise TypeError(f"cannot normalise {type(obj).__name__} to a config list; "
                    f"keys={list(obj)[:12] if isinstance(obj, dict) else 'n/a'}")


def params_from(rec, idx=0):
    """Pull the six parameters out of a record, searching nested blocks."""
    cands = [rec]
    for key in ('base_params', 'params', 'config', 'parameters'):
        v = rec.get(key) if isinstance(rec, dict) else None
        if isinstance(v, dict):
            cands.append(v)
    out = dict(idx=idx)
    for canon, names in ALIASES.items():
        for src in cands:
            hit = next((n for n in names if n in src), None)
            if hit is not None:
                out[canon] = float(src[hit])
                break
        else:
            raise KeyError(f"no parameter matching {names} in record; "
                           f"available keys: {sorted(set().union(*(set(c) for c in cands)))}")
    return out


def load_pool(quick):
    """Configurations from phase1 grid; parameters preserved verbatim."""
    with open('phase1_grid_results.json') as f:
        grid = json.load(f)
    rows = as_list(grid, 'results')
    out = []
    for i, r in enumerate(rows):
        c = params_from(r, i)
        c['rivalry'] = (bool(r.get('rivalry_producing') or False)
                        if 'rivalry_producing' in r else True)
        c['levelt_rho'] = fnum(r.get('levelt_rho'))
        out.append(c)
    if quick:
        out = [c for c in out if c['rivalry']][:400]
    return out


def _provenance(args=None):
    """Every constant that changes the numbers, recorded with every output file.

    Section 5.6 draws a lesson about a discretisation choice that was invisible because
    it lived in code and not in the record. These constants are the same hazard one
    level up: THETA, SPLIT_INDET and ADAPT_PIPELINE each change every figure and none
    of them was written to any result file.
    """
    import subprocess
    try:
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        commit = 'unavailable'
    try:
        import numba
        nbv = numba.__version__
    except ImportError:
        nbv = 'ABSENT: pure-Python fallback, RNG stream differs from the JIT path'
    return dict(THETA=THETA, SPLIT_INDET=SPLIT_INDET, ADAPT_PIPELINE=ADAPT_PIPELINE,
                MIN_DUR=MIN_DUR, X_INIT=X_INIT, BURN=BURN, XMAX=XMAX,
                numba=nbv, git_commit=commit, argv=' '.join(sys.argv),
                timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'))


def save(path, obj, force):
    if isinstance(obj, dict):
        obj = dict(obj, _provenance=_provenance())
    p = Path(path)
    if p.exists() and not force:
        sys.exit(f"{p} exists. Use --force to overwrite, or pass --out to redirect.")
    with open(p, 'w') as f:
        json.dump(obj, f, indent=1, default=float)
    print(f"  wrote {p}")



def _load_elig(args):
    """The eligibility pool, keyed by exponent. Every saved output carries a
    _provenance entry; it is metadata, not an exponent, so it is dropped here.
    Blocks run on their own load the pool from disk and would otherwise try to
    read '_provenance' as a number."""
    raw = json.load(open((args.elig or args.out) + 'wave22_A_eligibility.json'))
    return {k: v for k, v in raw.items() if not str(k).startswith('_')}


# ==========================================================================
# BLOCK V — kernel agreement against wave2_campaign at p = 1
# ==========================================================================
def block_verify(args):
    print("\n=== BLOCK V: kernel agreement with wave2_campaign at p = 1 ===")

    # ---- pipeline introspection, so signature mismatches surface here -----
    print("\n  pipeline surface:")
    surface = {}
    for mod, name in ((w2, 'wave2_campaign'), (w6, 'wave6_robustness')):
        fns = [f for f in dir(mod) if not f.startswith('__')
               and callable(getattr(mod, f, None))]
        surface[name] = {}
        for f in sorted(fns):
            try:
                import inspect
                sig = str(inspect.signature(getattr(mod, f)))
            except (TypeError, ValueError):
                sig = '(?)'
            surface[name][f] = sig
        print(f"    {name}: {', '.join(sorted(fns))}")

    lc = getattr(w2, 'load_configs', None)
    if lc is not None:
        try:
            raw = lc()
            kind = type(raw).__name__
            keys = (list(raw)[:12] if isinstance(raw, dict) else f"len={len(raw)}")
            print(f"    load_configs() -> {kind}, {keys}")
            surface['load_configs_shape'] = dict(kind=kind, keys=str(keys))
        except Exception as e:
            print(f"    load_configs() raised {type(e).__name__}: {e}")

    # ---- configurations: read phase2 directly, structure is documented ----
    with open('phase2_dissociation_results.json') as f:
        ph2 = json.load(f)
    recs = as_list(ph2, 'config_results')
    take = recs[:args.verify_n]
    print(f"\n  {len(recs)} configurations in phase2_dissociation_results.json, "
          f"using first {len(take)}")
    print(f"  record keys: {sorted(take[0])[:14]}")

    THETA_SWEEP = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10)

    rows = []
    for k, c in enumerate(take):
        prm = params_from(c, k)
        ref_cv = fnum(c.get('phase1_cv'))
        r = dict(config=int(c.get('config_idx') or k),
                 params={q: prm[q] for q in
                         ('lam', 'beta', 'alpha', 'sigma', 'gam', 'kap')},
                 phase1_cv=ref_cv,
                 phase1_levelt_rho=fnum(c.get('phase1_levelt_rho')),
                 theta={}, split={})
        for th in THETA_SWEEP:
            m_ = run_cell(prm, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0,
                          args.steps, args.seeds, BLK_V, 0,
                          split_indet=1, theta=th)
            sw, swf = sm(m_, O_SW), sm(m_, O_SWF)
            r['theta'][str(th)] = dict(
                cvA=sm(m_, O_CVA), durA=sm(m_, O_DURA), sw=sw, swF=swf,
                trans_frac=(1.0 - swf / sw) if sw > 0 else float('nan'),
                floor=sm(m_, O_FLOOR), actA=sm(m_, O_ACTA),
                cv_delta=sm(m_, O_CVA) - ref_cv)
        for tag, sp in (('split', 1), ('nosplit', 0)):
            m_ = run_cell(prm, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0,
                          args.steps, args.seeds, BLK_V, 0,
                          split_indet=sp, theta=THETA)
            r['split'][tag] = dict(cvA=sm(m_, O_CVA),
                                   cv_delta=sm(m_, O_CVA) - ref_cv)
        rows.append(r)

    print(f"\n  {args.seeds} seeds, {args.steps} steps, S_A = S_B = 0.5, G = 0")
    print("  These 30 configurations were selected for CV proximity to 0.5, so "
          "phase1_cv is\n  the authoritative per-config comparison. Sweeping theta "
          "because the CV ~ 0.5\n  target is carried by the transitional episode "
          "population (manuscript 4.3),\n  and theta controls how many margin "
          "crossings become episodes.\n")

    hdr = f"  {'cfg':>4} {'sig':>5} {'gam':>5} | {'ph1CV':>6} |"
    for th in THETA_SWEEP:
        hdr += f" {('th=' + str(th)):>13}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        q = r['params']
        line = (f"  {r['config']:>4} {q['sigma']:>5.3f} {q['gam']:>5.3f} | "
                f"{r['phase1_cv']:>6.3f} |")
        for th in THETA_SWEEP:
            t = r['theta'][str(th)]
            line += f" {t['cvA']:>6.3f}/{100*t['trans_frac']:>5.1f}%"
        print(line)
    print("\n  cells are CV / transitional%.  manuscript 4.3 reports 15% "
          "transitional.\n")

    print(f"  {'theta':>7} {'mean|CVerr|':>12} {'medCVerr':>10} "
          f"{'within .05':>11} {'med trans%':>11} {'med durA':>9}")
    scores = {}
    for th in THETA_SWEEP:
        d = np.array([r['theta'][str(th)]['cv_delta'] for r in rows], float)
        tr = np.array([r['theta'][str(th)]['trans_frac'] for r in rows], float)
        du = np.array([r['theta'][str(th)]['durA'] for r in rows], float)
        scores[th] = float(np.nanmean(np.abs(d)))
        print(f"  {th:>7.3f} {np.nanmean(np.abs(d)):>12.3f} "
              f"{np.nanmedian(d):>+10.3f} "
              f"{int(np.nansum(np.abs(d) <= 0.05)):>7}/{len(rows):<3} "
              f"{100*np.nanmedian(tr):>10.1f}% {np.nanmedian(du):>9.1f}")

    best_th = min(scores, key=scores.get)
    err = scores[best_th]
    sp = {t: float(np.nanmean(np.abs([r['split'][t]['cv_delta'] for r in rows])))
          for t in ('split', 'nosplit')}
    best_sp = min(sp, key=sp.get)
    print(f"\n  split on indeterminate at theta=0.05: mean|CVerr| {sp['split']:.3f}; "
          f"no-split: {sp['nosplit']:.3f}")

    print(f"""
  VERDICT: closest is theta = {best_th}, splitting = {best_sp}, mean |CV error|
  {err:.3f} against phase1_cv.""")
    if err <= 0.05:
        print(f"""
  That is agreement. Set THETA = {best_th} and SPLIT_INDET =
  {1 if best_sp == 'split' else 0} at the top of this file, rerun --verify once to
  confirm, then proceed to blocks A-C.

  If the recovered theta is NOT 0.05, that is a finding and not just an
  integration detail: manuscript 2.2 states theta = 0.05, and the value that
  reproduces the frozen CV is what the frozen results actually used. Check
  wave2_campaign.extract_durations for the literal, and note the discrepancy in
  2.2 and in the deviations list.""")
    else:
        print("""
  NOT agreement at any theta, so the difference is upstream of episode
  extraction, in the state update itself. Two candidates remain and both are
  settled by three functions:

    (a) _rectify -- if it is anything other than clip to [0, x_max], or is
        applied outside rather than inside the noise term, the dynamics differ.
    (b) noise -- this file draws one N(0, sigma) per channel per step, added
        inside the rectifier per Eq 1. A shared draw, a different generator, or
        noise on the difference rather than per channel all change the
        transitional population, which is what carries CV.

  Paste wave2_campaign._rectify, run_trace and extract_durations. Those three
  determine every number in the table above; nothing else in the pipeline does.""")

    save(args.out + 'wave22_V_verify.json',
         dict(n=args.verify_n, seeds=args.seeds, steps=args.steps, rows=rows),
         True)


# ==========================================================================
# BLOCK A — eligibility re-derived under each exponent
# ==========================================================================
def block_A(args):
    print("\n=== BLOCK A: eligibility per adaptation exponent ===")
    allc = load_pool(args.quick)
    pool = [c for c in allc if c['rivalry']]
    n_rho = sum(1 for c in pool if np.isfinite(c['levelt_rho']))
    print(f"  {len(allc)} records, {len(pool)} rivalry-producing, "
          f"{n_rho} with a stored levelt_rho")
    print(f"  (4.1 reports 6,814 rivalry-producing of 9,000)")
    if args.screen_n and args.screen_n < len(pool):
        _rng = np.random.default_rng(2201)
        pool = [pool[i] for i in
                sorted(_rng.choice(len(pool), args.screen_n, replace=False))]
        print(f"  --screen-n: screening a random subsample of {len(pool)}")
    if args.no_rho:
        print("  --no-rho: eligibility is rivalry + CV only. The Levelt-rho > 0.7\n"
              "  criterion is NOT applied, so this pool is not the registered pool\n"
              "  and the p=1 row will exceed 762. Report as a deviation.")

    sweep = (0.30, 0.40, 0.50, 0.60, 0.70)

    # baseline mean activation under the linear law does not depend on p, so it is
    # computed once per configuration and reused across exponents
    print("  measuring baseline activation (once per configuration)...", flush=True)
    t0 = time.time()
    xbars = {}
    for j, c in enumerate(pool):
        base = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0,
                        args.steps, max(4, args.seeds // 2), BLK_A_ACT, 0,
                        split_indet=SPLIT_INDET)
        xb = 0.5 * (sm(base, O_ACTA) + sm(base, O_ACTB))
        if np.isfinite(xb) and xb > 0:
            xbars[c['idx']] = xb
        if (j + 1) % PROGRESS_EVERY == 0:
            el = time.time() - t0
            print(f"    {j+1}/{len(pool)}  [{el:.0f}s elapsed, "
                  f"~{el/(j+1)*(len(pool)-j-1):.0f}s left]", flush=True)
    print(f"  activation measured for {len(xbars)}/{len(pool)} "
          f"[{time.time()-t0:.0f}s]\n", flush=True)

    p_list = (1.0, 2.0) if args.quick else P_VALUES
    res = {}
    for p in p_list:
        t0 = time.time()
        keep, cvs, rhos = [], [], []
        for j, c in enumerate(pool):
            if (j + 1) % PROGRESS_EVERY == 0:
                el = time.time() - t0
                print(f"    p={p}  {j+1}/{len(pool)}  eligible so far {len(keep)}"
                      f"  [{el:.0f}s, ~{el/(j+1)*(len(pool)-j-1):.0f}s left]",
                      flush=True)
            xbar = xbars.get(c['idx'])
            if xbar is None:
                continue
            kap_eff = c['kap'] / (xbar ** (p - 1.0)) if p != 1.0 else c['kap']
            cp = dict(c, kap=kap_eff)

            b = run_cell(cp, p, 0.5, 0.5, 0.0, G_NONE, 0.0,
                         args.steps, args.seeds, BLK_A_BASE, int(p * 100), split_indet=SPLIT_INDET)
            sw, cv = sm(b, O_SW), sm(b, O_CVA)
            if not (np.isfinite(sw) and np.isfinite(cv)) or sw < 10:
                continue

            if args.no_rho:
                rho = float('nan')
                rho_ok = True
            else:
                pred = []
                for lv, sa in enumerate(sweep):
                    r = run_cell(cp, p, sa, 0.5, 0.0, G_NONE, 0.0,
                                 args.steps, max(3, args.seeds // 3), BLK_A_SWEEP,
                                 int(p * 100) + lv, split_indet=SPLIT_INDET)
                    pred.append(sm(r, O_PREDA))
                pred = np.asarray(pred, float)
                if not np.isfinite(pred).all():
                    continue
                rho = float(np.corrcoef(np.argsort(np.argsort(sweep)),
                                        np.argsort(np.argsort(pred)))[0, 1])
                rho_ok = rho > 0.7
            # eligibility: switches >= 10, rho > 0.7 (strict: 762 not 764), CV window
            if rho_ok and 0.35 <= cv <= 0.65:
                keep.append(c['idx'])
            cvs.append(cv)
            rhos.append(rho)

        res[str(p)] = dict(n_screened=len(pool), n_eligible=len(keep),
                           median_cv=float(np.median(cvs)) if cvs else None,
                           median_rho=float(np.median(rhos)) if rhos else None,
                           eligible_idx=keep)
        print(f"  p={p:<5} eligible {len(keep):>5}/{len(pool)}  "
              f"medCV {np.median(cvs) if cvs else float('nan'):.3f}  "
              f"medRho {np.median(rhos) if rhos else float('nan'):.3f}  "
              f"[{time.time()-t0:.0f}s]")

    print("""
  NOTE on the p=1 row. It does NOT reproduce the registered 762, and is not
  expected to. gc_lca_phase1_grid.py computes Levelt rho over 11 signal levels
  (0.25 to 0.75 in steps of 0.05) at 8 seeds x 12,000 steps via scipy's
  spearmanr; this block uses 5 levels at 3 seeds, which is too coarse to
  discriminate and passes almost everything (median rho 1.000). The resulting
  pool is therefore larger than the registered one -- roughly 946 against 762 at
  p=1 -- and more permissive in the same direction at every exponent.

  This is acceptable for the question at hand and must be reported as a
  deviation. The Levelt-rho criterion is a compliance filter, not part of the
  adaptation-exponent question; block B's Levelt II test is explicit and
  independent of rho; and the identical criterion is applied at every exponent,
  so the across-exponent comparison is internally consistent. Matching the
  registered sweep would cost 88 runs per configuration per exponent instead of
  15, for no gain in what the comparison can show.

  What IS diagnostic here: eligibility should rise with p while median CV falls
  toward the middle of the window, because adaptation superlinear in activation
  cancels the drive gain more strongly and makes durations more regular. If
  median CV rises with p instead, the renormalisation of kappa has the wrong
  sign somewhere.""")
    save(args.out + 'wave22_A_eligibility.json', res, args.force)
    return res


# ==========================================================================
# BLOCK B — Levelt II reachability
# ==========================================================================
def block_B(args, elig=None):
    print("\n=== BLOCK B: Levelt II reachability vs exponent ===")
    if elig is None:
        elig = _load_elig(args)
    pool = {c['idx']: c for c in load_pool(args.quick)}

    rng = np.random.default_rng(22)
    res = {}
    for p_key, rec in elig.items():
        p = float(p_key)
        idx = rec['eligible_idx']
        if not idx:
            print(f"  p={p}: no eligible configurations, skipped")
            continue
        take = rng.choice(idx, min(args.n_config, len(idx)), replace=False)
        t0 = time.time()

        hits = 0
        alive = 0
        per_cfg = []
        for ci in take:
            c = pool[int(ci)]
            base_lin = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0,
                                args.steps, args.seeds, BLK_B_ACT, 0, split_indet=SPLIT_INDET)
            xbar = 0.5 * (sm(base_lin, O_ACTA) + sm(base_lin, O_ACTB))
            actA = sm(base_lin, O_ACTA)
            ref = 0.5 * c['lam'] * actA
            if not (0.005 <= ref <= 0.5):
                sys.exit(f"reference increment {ref:.4g} outside [0.005, 0.5] for "
                         f"config {ci}. This is the units trap (4.2): ref must be "
                         f"0.5*lambda*mean ACTIVATION, not duration.")

            kap_eff = c['kap'] / (xbar ** (p - 1.0)) if p != 1.0 else c['kap']
            cp = dict(c, kap=kap_eff)
            b = run_cell(cp, p, 0.5, 0.5, 0.0, G_NONE, 0.0,
                         args.steps, args.seeds, BLK_B_BASE, int(p * 100), split_indet=SPLIT_INDET)
            dA0, dB0, sw0 = sm(b, O_DURA), sm(b, O_DURB), sm(b, O_SW)
            dA0f, dB0f = sm(b, O_DURAF), sm(b, O_DURBF)
            if not (np.isfinite(dA0) and np.isfinite(dB0) and sw0 >= 10):
                per_cfg.append(dict(config=int(ci), alive=False))
                continue
            alive += 1

            row = dict(config=int(ci), alive=True, ref=ref, p=p, sweep=[])
            hit_any = False
            for lv, m in enumerate(MULTS):
                r = run_cell(cp, p, 0.5 + m * ref, 0.5, 0.0, G_NONE, 0.0,
                             args.steps, args.seeds, BLK_B_SWEEP,
                             int(p * 100) + lv + 1, split_indet=SPLIT_INDET)
                pA, pB = pct(sm(r, O_DURA), dA0), pct(sm(r, O_DURB), dB0)
                pAf, pBf = pct(sm(r, O_DURAF), dA0f), pct(sm(r, O_DURBF), dB0f)
                hit = (abs(pA) <= LEVELT_A and LEVELT_B[0] <= pB <= LEVELT_B[1])
                hit_f = (abs(pAf) <= LEVELT_A and LEVELT_B[0] <= pBf <= LEVELT_B[1])
                hit_any |= hit
                row['sweep'].append(dict(mult=m, pA=pA, pB=pB, pA_filt=pAf,
                                         pB_filt=pBf, hit=hit, hit_filt=hit_f))
            row['hit_any'] = hit_any
            hits += int(hit_any)
            per_cfg.append(row)

        lo, hi = wilson(hits, alive)
        res[p_key] = dict(p=p, n_config=len(take), n_alive=alive,
                          n_hit=hits, wilson=[lo, hi], per_config=per_cfg)
        print(f"  p={p:<5} alive {alive:>4}/{len(take)}  hits {hits:>4}  "
              f"[{100*lo:.1f}%, {100*hi:.1f}%]  [{time.time()-t0:.0f}s]")

    if '1.0' in res and res['1.0']['n_hit'] > 0:
        print(f"\n  *** WARNING: p=1 gives {res['1.0']['n_hit']} hits, not 0. ***\n"
              "  wave15's Levelt-valid search found 0/100. Reconcile before\n"
              "  interpreting any p>1 row: check the sweep grid (MULTS here spans\n"
              "  32x), the eligibility criteria, and the boundary-exclusion rule.\n")
    save(args.out + 'wave22_B_levelt.json', res, args.force)
    return res


# ==========================================================================
# BLOCK C — does the gate-timing result survive the fix
# ==========================================================================
def block_C(args, elig=None):
    print("\n=== BLOCK C: gate-timing sign contrast vs exponent ===")
    if elig is None:
        elig = _load_elig(args)
    pool = {c['idx']: c for c in load_pool(args.quick)}
    rng = np.random.default_rng(23)

    conditions = [('gated_1x', G_ON, 1.0), ('gated_2x', G_ON, 2.0),
                  ('ungated_1x', G_UNGATED, 1.0), ('ungated_2x', G_UNGATED, 2.0),
                  ('antigated_2x', G_OFF, 2.0),
                  ('yoked_gated_2x', G_REPLAY, 2.0),
                  ('yoked_antigated_2x', G_REPLAY, 2.0)]

    res = {}
    for p in (GATE_P if not args.quick else (1.0, 2.0)):
        key = f"{p}"
        if key not in elig or not elig[key]['eligible_idx']:
            print(f"  p={p}: no eligible pool from block A, skipped")
            continue
        idx = elig[key]['eligible_idx']
        take = rng.choice(idx, min(args.n_config, len(idx)), replace=False)
        t0 = time.time()
        rows = {n: [] for n, _, _ in conditions}
        diag = []

        for ci in take:
            c = pool[int(ci)]
            bl = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0,
                          args.steps, args.seeds, BLK_C_ACT, 0, split_indet=SPLIT_INDET)
            xbar = 0.5 * (sm(bl, O_ACTA) + sm(bl, O_ACTB))
            ref = 0.5 * c['lam'] * sm(bl, O_ACTA)
            kap_eff = c['kap'] / (xbar ** (p - 1.0)) if p != 1.0 else c['kap']
            cp = dict(c, kap=kap_eff)

            b = run_cell(cp, p, 0.5, 0.5, 0.0, G_NONE, 0.0,
                         args.steps, args.seeds, BLK_C_BASE, int(p * 100), split_indet=SPLIT_INDET)
            dA0, dB0 = sm(b, O_DURA), sm(b, O_DURB)
            dA0f, dB0f = sm(b, O_DURAF), sm(b, O_DURBF)
            sw0, cv0 = sm(b, O_SW), sm(b, O_CVA)
            absB0 = sm(b, O_ACTB)
            if not (np.isfinite(dA0) and np.isfinite(dB0) and sw0 >= 10):
                continue
            athr = ABS_THRESH_FRAC * absB0

            # record schedules for the yoked arms from live runs
            sched_on = np.zeros(args.steps, dtype=np.uint8)
            sched_off = np.zeros(args.steps, dtype=np.uint8)
            run_cell(cp, p, 0.5, 0.5, 2.0 * ref, G_ON, athr, args.steps, 1,
                     BLK_C_REC_ON, int(p * 100), schedule=sched_on, record=True, split_indet=SPLIT_INDET)
            run_cell(cp, p, 0.5, 0.5, 2.0 * ref, G_OFF, athr, args.steps, 1,
                     BLK_C_REC_OFF, int(p * 100), schedule=sched_off, record=True, split_indet=SPLIT_INDET)

            for lv, (name, gate, amp) in enumerate(conditions):
                sch = (sched_on if name == 'yoked_gated_2x'
                       else sched_off if name == 'yoked_antigated_2x' else None)
                r = run_cell(cp, p, 0.5, 0.5, amp * ref, gate, athr,
                             args.steps, args.seeds, BLK_C_COND,
                             int(p * 100) + lv + 1, schedule=sch, split_indet=SPLIT_INDET)
                rows[name].append(dict(
                    config=int(ci),
                    pA=pct(sm(r, O_DURA), dA0), pB=pct(sm(r, O_DURB), dB0),
                    pA_filt=pct(sm(r, O_DURAF), dA0f),
                    pB_filt=pct(sm(r, O_DURBF), dB0f),
                    absB=sm(r, O_DURPOOL), absB0=sm(b, O_DURPOOL),
                    dose=sm(r, O_DOSE), sw=sm(r, O_SW), swf=sm(r, O_SWF),
                    cv=sm(r, O_CVA), floor=sm(r, O_FLOOR)))
            diag.append(dict(config=int(ci), cv_base=cv0, sw_base=sw0,
                             swf_base=sm(b, O_SWF), ref=ref))

        summary = {}
        for name, _, _ in conditions:
            rr = rows[name]
            pA = np.array([x['pA'] for x in rr], float)
            pB = np.array([x['pB'] for x in rr], float)
            pBf = np.array([x['pB_filt'] for x in rr], float)
            n = int(np.isfinite(pB).sum())
            k = int(np.nansum(pB > 0))
            lo, hi = wilson(k, n)
            per, pooled = both_conventions(pB, pA)
            summary[name] = dict(
                n=n, med_pA=float(np.nanmedian(pA)), med_pB=float(np.nanmedian(pB)),
                med_pB_filt=float(np.nanmedian(pBf)),
                n_pB_positive=k, wilson=[lo, hi],
                ratio_median_of_ratios=per, ratio_of_medians=pooled,
                med_dose=float(np.nanmedian([x['dose'] for x in rr])),
                med_absB_change=float(np.nanmedian(
                    [100 * (x['absB'] - x['absB0']) / x['absB0']
                     for x in rr if x['absB0'] > 0])),
                med_sw=float(np.nanmedian([x['sw'] for x in rr])),
                med_swf=float(np.nanmedian([x['swf'] for x in rr])),
                med_cv=float(np.nanmedian([x['cv'] for x in rr])))

        res[key] = dict(p=p, summary=summary, per_config=rows, diagnostics=diag)
        print(f"\n  p = {p}   [{time.time()-t0:.0f}s]")
        print(f"    {'condition':<20} {'attended':>9} {'competitor':>11} "
              f"{'filt':>8} {'pB>0':>10} {'ratio(per/pooled)':>20} {'swF':>7} {'CV':>6}")
        for name, _, _ in conditions:
            s = summary[name]
            print(f"    {name:<20} {s['med_pA']:>+9.1f} {s['med_pB']:>+11.1f} "
                  f"{s['med_pB_filt']:>+8.1f} {s['n_pB_positive']:>5}/{s['n']:<4} "
                  f"{s['ratio_median_of_ratios']:>+9.3f}/{s['ratio_of_medians']:>+9.3f} "
                  f"{s['med_swf']:>7.1f} {s['med_cv']:>6.3f}")

    print("""
  READ THE swF AND CV COLUMNS. The anti-gated condition shortens both channels
  together, which may mean it has left the rivalry regime rather than shifted
  coupling within it. If filtered switch count rises sharply or CV collapses
  relative to baseline for that condition, the anti-gate number describes regime
  destruction and must be relabelled in 4.7.2, 1.5 and the Abstract.
""")
    save(args.out + 'wave22_C_gates.json', res, args.force)
    return res



# ==========================================================================
# BLOCK D - is there a parameter region satisfying BOTH published datasets?
# ==========================================================================
def block_D(args, elig=None):
    """Joint feasibility of Levelt II and the Chong gated ratio.

    Section 4.7.5 establishes that the gated competitor-to-attended ratio is ~90%
    predictable from the model's six parameters. That means the ratio does not
    identify the mechanism -- but it does identify the parameters. Levelt II
    reachability constrains the adaptation exponent. Those are constraints on
    different objects, so the tension between them is not necessarily a
    contradiction: there may be configurations that satisfy both.

    This block asks whether the intersection is empty. Both answers matter:

      NON-EMPTY -> the parameter region consistent with both the attentional and
        the stimulus-strength literature, which is a positive result and what a
        modeller would want from this study.

      EMPTY -> two independent published datasets jointly exclude this model
        class, which is a sharper structural finding than locating the defect.

    Chong et al. (2005) Experiment 3 give 9/29 = 0.310. Tolerances of +/-0.05,
    0.10 and 0.15 are reported, since with n = 4 observers the interval on that
    ratio is wide and no single tolerance is defensible.
    """
    print("\n=== BLOCK D: joint feasibility, Levelt II and the Chong ratio ===")
    if elig is None:
        elig = _load_elig(args)
    pool = {c['idx']: c for c in load_pool(args.quick)}
    rng = np.random.default_rng(24)
    CHONG = 0.310
    TOLS = (0.05, 0.10, 0.15)

    res = {}
    for p_key, rec in sorted(elig.items(), key=lambda kv: float(kv[0])):
        p = float(p_key)
        idx = rec['eligible_idx']
        if not idx:
            continue
        take = rng.choice(idx, min(args.n_config, len(idx)), replace=False)
        t0 = time.time()
        rows = []
        for j, ci in enumerate(take):
            if (j + 1) % PROGRESS_EVERY == 0:
                el = time.time() - t0
                print(f"    p={p}  {j+1}/{len(take)}  [{el:.0f}s, "
                      f"~{el/(j+1)*(len(take)-j-1):.0f}s left]", flush=True)
            c = pool[int(ci)]
            bl = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps,
                          args.seeds, BLK_D_ACT, 0, split_indet=SPLIT_INDET)
            xbar = 0.5 * (sm(bl, O_ACTA) + sm(bl, O_ACTB))
            ref = 0.5 * c['lam'] * sm(bl, O_ACTA)
            if not (0.005 <= ref <= 0.5) or not np.isfinite(xbar) or xbar <= 0:
                continue
            keff = c['kap'] / (xbar ** (p - 1.0)) if p != 1.0 else c['kap']
            cp = dict(c, kap=keff)

            b = run_cell(cp, p, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps,
                         args.seeds, BLK_D_BASE, int(p * 100),
                         split_indet=SPLIT_INDET)
            dA0, dB0 = sm(b, O_DURA), sm(b, O_DURB)
            if not (np.isfinite(dA0) and np.isfinite(dB0) and sm(b, O_SW) >= 10):
                continue

            # (i) Levelt II reachable at any swept magnitude?
            hit = False
            for lv, m in enumerate(MULTS):
                r = run_cell(cp, p, 0.5 + m * ref, 0.5, 0.0, G_NONE, 0.0,
                             args.steps, args.seeds, BLK_D_SWEEP,
                             int(p * 100) + lv, split_indet=SPLIT_INDET)
                pA, pB = pct(sm(r, O_DURA), dA0), pct(sm(r, O_DURB), dB0)
                if abs(pA) <= LEVELT_A and LEVELT_B[0] <= pB <= LEVELT_B[1]:
                    hit = True
                    break

            # (ii) gated ratio, at both amplitudes
            ratios = {}
            for lv, amp in ((0, 1.0), (1, 2.0)):
                g = run_cell(cp, p, 0.5, 0.5, amp * ref, G_ON, 0.0, args.steps,
                             args.seeds, BLK_D_GATE, int(p * 100) + lv,
                             split_indet=SPLIT_INDET)
                a_, b_ = pct(sm(g, O_DURA), dA0), pct(sm(g, O_DURB), dB0)
                ratios[amp] = (b_ / a_) if (np.isfinite(a_) and abs(a_) > 1e-9) \
                    else float('nan')

            rows.append(dict(config=int(ci), p=p, levelt=bool(hit),
                             ratio_1x=ratios[1.0], ratio_2x=ratios[2.0],
                             lam=c['lam'], beta=c['beta'], alpha=c['alpha'],
                             sigma=c['sigma'], gam=c['gam'], kap=c['kap']))

        n = len(rows)
        lev = sum(1 for r in rows if r['levelt'])
        summ = dict(p=p, n=n, n_levelt=lev, tolerances={})
        for tol in TOLS:
            near = [r for r in rows
                    if np.isfinite(r['ratio_1x'])
                    and abs(r['ratio_1x'] - CHONG) <= tol]
            both = [r for r in near if r['levelt']]
            lo, hi = wilson(len(both), n) if n else (float('nan'),) * 2
            summ['tolerances'][str(tol)] = dict(
                n_near_chong=len(near), n_both=len(both), wilson=[lo, hi],
                params_both={k: float(np.median([r[k] for r in both]))
                             for k in ('lam', 'beta', 'alpha', 'sigma', 'gam', 'kap')}
                if both else None)
        summ['rows'] = rows
        res[p_key] = summ
        print(f"  p={p:<5} n={n:>4}  Levelt II {lev:>4}  "
              + "  ".join(f"both(+/-{t}) {summ['tolerances'][str(t)]['n_both']:>3}"
                          for t in TOLS)
              + f"  [{time.time()-t0:.0f}s]")

    print(f"\n  {'p':>5} {'n':>5} {'LeveltII':>9} {'nearChong':>10} "
          f"{'BOTH':>6} {'Wilson95':>16}")
    any_both = 0
    for k in sorted(res, key=float):
        s = res[k]
        t = s['tolerances']['0.1']
        any_both += t['n_both']
        print(f"  {s['p']:>5.2f} {s['n']:>5} {s['n_levelt']:>9} "
              f"{t['n_near_chong']:>10} {t['n_both']:>6} "
              f"  [{100*t['wilson'][0]:.1f}%, {100*t['wilson'][1]:.1f}%]")

    if any_both == 0:
        print("""
  THE INTERSECTION IS EMPTY at every exponent tested.

  No configuration reproduces Levelt's second proposition while also producing
  Chong et al.'s gated competitor-to-attended ratio. Two independent published
  datasets therefore jointly exclude this architecture, which is a stronger and
  more useful claim than locating the defect in the adaptation law: it says the
  repair of Section 4.7.7 cannot be completed within this model class, and it
  bounds what any successor must do. Report the exclusion, its tolerance
  sensitivity, and which of the two constraints each region of parameter space
  can satisfy.""")
    else:
        print("""
  THE INTERSECTION IS NON-EMPTY.

  Configurations exist that satisfy both published constraints simultaneously.
  The median parameters of that set are printed per exponent in the JSON. This
  supersedes the framing in Sections 4.7.7 and 5.4, which report the two
  constraints as pulling against one another: they pull against one another
  along the exponent, but not jointly across the parameter space. Report the
  feasible region, check whether it is contiguous, and state whether it also
  satisfies the eligibility criteria and the increasing-duration behaviour --
  a region that satisfies two constraints while violating a third is not a
  solution.""")

    save(args.out + 'wave22_D_joint.json', res, args.force)
    return res


# ==========================================================================
# BLOCK E - does superlinear adaptation repair Modified Proposition IV
#           in general, or only in the jointly-satisfying subset?
# ==========================================================================
def block_E(args, elig=None):
    """Modified Proposition IV under a superlinear adaptation law.

    Section 4.2 reports that the model occupies only the increasing-duration
    regime: alternation rate FALLS with equal bilateral drive in 97 of 100
    configurations, at every strength tested, and neither a saturating input
    transfer nor anything else in the literature's list of remedies fixes it here.
    Signal-dependent noise does, but it acts on the noise process rather than the
    deterministic dynamics.

    A spot check on the eighteen configurations that satisfy both Levelt II and the
    Chong gated ratio found that all twelve with an exponent of 1.5 or above are in
    the DECREASING-duration regime, with rank correlations reaching +1.000, and all
    six below it are not. If that generalises, then one change to the adaptation law
    repairs both of the failures this paper reports, and it does so through the
    deterministic dynamics.

    Those eighteen were selected on two other criteria, so the spot check cannot
    establish it. This block runs the registered equal-strength sweep on an
    unselected sample from each exponent's eligible pool.

    Reported both raw and under the registered 5-timestep minimum, because Section
    4.2's transitional-episode population inflates raw transition counts and the
    remedy that section does demonstrate weakens considerably under filtering.
    """
    print("\n=== BLOCK E: Modified Proposition IV vs adaptation exponent ===")
    if elig is None:
        elig = _load_elig(args)
    pool = {c['idx']: c for c in load_pool(args.quick)}
    rng = np.random.default_rng(25)
    LEVELS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
    ranks = np.argsort(np.argsort(LEVELS))

    res = {}
    for p_key, rec in sorted(elig.items(), key=lambda kv: float(kv[0])):
        p = float(p_key)
        idx = rec['eligible_idx']
        if not idx:
            continue
        take = rng.choice(idx, min(args.n_config, len(idx)), replace=False)
        t0 = time.time()
        rows = []
        for j, ci in enumerate(take):
            if (j + 1) % PROGRESS_EVERY == 0:
                el = time.time() - t0
                print(f"    p={p}  {j+1}/{len(take)}  [{el:.0f}s, "
                      f"~{el/(j+1)*(len(take)-j-1):.0f}s left]", flush=True)
            c = pool[int(ci)]
            bl = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps,
                          max(4, args.seeds // 2), BLK_E_ACT, 0,
                          split_indet=SPLIT_INDET)
            xbar = 0.5 * (sm(bl, O_ACTA) + sm(bl, O_ACTB))
            if not np.isfinite(xbar) or xbar <= 0:
                continue
            keff = c['kap'] / (xbar ** (p - 1.0)) if p != 1.0 else c['kap']
            cp = dict(c, kap=keff)

            raw, filt, wta = [], [], []
            for lv, s in enumerate(LEVELS):
                r = run_cell(cp, p, s, s, 0.0, G_NONE, 0.0, args.steps,
                             args.seeds, BLK_E_SWEEP, int(p * 100) + lv,
                             split_indet=SPLIT_INDET)
                raw.append(sm(r, O_SW))
                filt.append(sm(r, O_SWF))
                wta.append(1.0 if sm(r, O_SW) < 2 else 0.0)
            a, b = np.asarray(raw, float), np.asarray(filt, float)
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                continue
            rho_raw = float(np.corrcoef(ranks, np.argsort(np.argsort(a)))[0, 1])
            rho_f = float(np.corrcoef(ranks, np.argsort(np.argsort(b)))[0, 1])
            rows.append(dict(config=int(ci), rho_raw=rho_raw, rho_filt=rho_f,
                             wta_frac=float(np.mean(wta)),
                             lam=c['lam'], beta=c['beta'], alpha=c['alpha'],
                             sigma=c['sigma'], gam=c['gam'], kap=c['kap']))

        n = len(rows)
        def tally(key):
            v = np.array([r[key] for r in rows], float)
            dd = int(np.sum(v > 0))
            lo, hi = wilson(dd, n) if n else (float('nan'),) * 2
            return dict(n=n, n_dd=dd, frac_dd=(dd / n if n else float('nan')),
                        median_rho=float(np.median(v)), wilson=[lo, hi])
        res[p_key] = dict(p=p, raw=tally('rho_raw'), filt=tally('rho_filt'),
                          rows=rows)
        rr, ff = res[p_key]['raw'], res[p_key]['filt']
        print(f"  p={p:<5} n={n:>4}  raw: DD {rr['n_dd']:>4}/{n} "
              f"(med rho {rr['median_rho']:+.3f})   "
              f"filtered: DD {ff['n_dd']:>4}/{n} "
              f"(med rho {ff['median_rho']:+.3f})   [{time.time()-t0:.0f}s]")

    print(f"\n  {'p':>5} {'n':>5} | {'DD raw':>8} {'medrho':>8} {'Wilson95':>16} "
          f"| {'DD filt':>8} {'medrho':>8} {'Wilson95':>16}")
    for k in sorted(res, key=float):
        s = res[k]
        r_, f_ = s['raw'], s['filt']
        print(f"  {s['p']:>5.2f} {r_['n']:>5} | {r_['n_dd']:>8} "
              f"{r_['median_rho']:>+8.3f} "
              f"[{100*r_['wilson'][0]:>5.1f}%, {100*r_['wilson'][1]:>5.1f}%] "
              f"| {f_['n_dd']:>8} {f_['median_rho']:>+8.3f} "
              f"[{100*f_['wilson'][0]:>5.1f}%, {100*f_['wilson'][1]:>5.1f}%]")

    print("""
  HOW TO READ THIS

  Section 4.2 reports 97 of 100 configurations VIOLATING the proposition at p = 1,
  i.e. DD in roughly 3%. That is the baseline the p = 1 row here should reproduce;
  if it does not, this block's sweep differs from the registered one and the
  comparison across exponents is still valid but the absolute rates are not.

  The filtered column is the one that matters. Section 4.2's demonstrated remedy,
  signal-dependent noise, recovers the proposition in 28 of 30 configurations
  unfiltered but falls to a median rho of +0.299 under the 5-timestep minimum,
  because scaled noise inflates the transitional population. A remedy that holds
  up under filtering is doing something the noise remedy is not.

  If the DD fraction rises with the exponent in BOTH columns, then one change to
  the adaptation law repairs both failures this paper reports -- the unreachability
  of Levelt's second proposition and the confinement to the increasing-duration
  regime -- and it does so through the deterministic dynamics rather than the noise
  process. That is a contribution to the literature Section 4.2 reviews, not just a
  repair of this model, and Sections 4.2, 4.7.7, 5.4 and the Abstract all change.

  If it rises raw but not filtered, the repair is the same kind as signal-dependent
  noise and should be reported alongside it with the same caveat.

  If it does not rise, the spot check on the eighteen jointly-satisfying
  configurations was an artefact of their selection, and Section 4.7.7's caveat
  stands as written.
""")
    save(args.out + 'wave22_E_prop4.json', res, args.force)
    return res


# ==========================================================================
# BLOCK F - can a dominance-gated adaptation modulation recover the Chong
#           ratio while the superlinear law holds both structural results?
# ==========================================================================
def block_G(args, elig=None):
    """The untested cell in the design.

    Section 4.7.1 compared five formulations of attentional modulation, all applied
    continuously. Reducing the attended channel's adaptation gain was the one
    state-dependent formulation that produced the STIMULUS-STRENGTH signature rather
    than the attentional one: the competitor shortened, 0 of 44 configurations
    same-signed. Section 4.7.2 then showed that delivery timing, not
    state-dependence, sets the coupling sign. Nobody has combined the two: an
    adaptation modulation GATED ON DOMINANCE.

    That combination is what Section 5.4's successor argument points at. Under the
    superlinear law of Section 4.7.7 the attended channel's own response shrinks as
    the exponent rises, because adaptation increasingly cancels the drive increase,
    and that is what pushes the competitor-to-attended ratio above Chong et al.'s
    observed 0.310 - from 0.32 at p = 1 to 1.50 at p = 3. A modulation that lengthens
    the attended channel's dominance WITHOUT paying the adaptation penalty should
    restore its own response while leaving the competitor's mechanism intact, and so
    lower the ratio.

    PREDICTIONS, stated before running:
      1. Gated adaptation modulation gives a POSITIVE competitor response, unlike the
         same modulation applied continuously (0/44 in Section 4.7.1). If it does
         not, delivery timing does not govern this formulation and Section 4.7.2's
         account is narrower than claimed.
      2. Its ratio is LOWER than the gated input increment's at the same exponent,
         because the attended response is larger for the same competitor response.
      3. At p = 2 the ratio approaches 0.310, which the gated increment overshoots.

    Levelt II and Modified Proposition IV are measured at zero modulation, so an
    attentional mechanism cannot disturb them. They are preserved by construction
    rather than by test, and this block does not re-measure them.
    """
    print("\n=== BLOCK G: dominance-gated adaptation modulation ===")
    if elig is None:
        elig = _load_elig(args)
    pool = {c['idx']: c for c in load_pool(args.quick)}
    rng = np.random.default_rng(26)
    CHONG = 0.310
    # A smoke test at p = 2 gives ratio +0.874 at zero reduction and -0.628 at 0.4,
    # so the crossing through Chong's +0.310 lies below 0.2 and the original range
    # started past it. Extended downward.
    MS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.40)  # fractional reduction of kappa_A
    P_LIST = (1.0, 2.0) if args.quick else (1.0, 1.75, 2.0, 3.0)

    res = {}
    for p in P_LIST:
        key = f"{p}"
        if key not in elig or not elig[key]['eligible_idx']:
            print(f"  p={p}: no eligible pool, skipped")
            continue
        take = rng.choice(elig[key]['eligible_idx'],
                          min(args.n_config, len(elig[key]['eligible_idx'])),
                          replace=False)
        t0 = time.time()
        conds = ([('adapt_gated_%.2f' % m, G_ADAPT_ON, m) for m in MS]
                 + [('adapt_always_%.2f' % m, G_ADAPT_ALL, m) for m in MS]
                 + [('incr_gated_1x', G_ON, 1.0), ('incr_gated_2x', G_ON, 2.0)])
        acc = {n: [] for n, _, _ in conds}

        for j, ci in enumerate(take):
            if (j + 1) % PROGRESS_EVERY == 0:
                el = time.time() - t0
                print(f"    p={p}  {j+1}/{len(take)}  [{el:.0f}s, "
                      f"~{el/(j+1)*(len(take)-j-1):.0f}s left]", flush=True)
            c = pool[int(ci)]
            bl = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps,
                          max(4, args.seeds // 2), BLK_G_ACT, 0,
                          split_indet=SPLIT_INDET)
            xbar = 0.5 * (sm(bl, O_ACTA) + sm(bl, O_ACTB))
            ref = 0.5 * c['lam'] * sm(bl, O_ACTA)
            if not (0.005 <= ref <= 0.5) or not np.isfinite(xbar) or xbar <= 0:
                continue
            keff = c['kap'] / (xbar ** (p - 1.0)) if p != 1.0 else c['kap']
            cp = dict(c, kap=keff)
            b = run_cell(cp, p, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps,
                         args.seeds, BLK_G_BASE, int(p * 100),
                         split_indet=SPLIT_INDET)
            dA0, dB0 = sm(b, O_DURA), sm(b, O_DURB)
            dA0f, dB0f = sm(b, O_DURAF), sm(b, O_DURBF)
            if not (np.isfinite(dA0) and np.isfinite(dB0) and sm(b, O_SW) >= 10):
                continue

            for lv, (name, gate, amt) in enumerate(conds):
                amp = amt if gate in (G_ADAPT_ON, G_ADAPT_ALL) else amt * ref
                r = run_cell(cp, p, 0.5, 0.5, amp, gate, 0.0, args.steps,
                             args.seeds, BLK_G_COND, int(p * 100) + lv,
                             split_indet=SPLIT_INDET)
                pA, pB = pct(sm(r, O_DURA), dA0), pct(sm(r, O_DURB), dB0)
                acc[name].append(dict(
                    config=int(ci), pA=pA, pB=pB,
                    pA_filt=pct(sm(r, O_DURAF), dA0f),
                    pB_filt=pct(sm(r, O_DURBF), dB0f),
                    ratio=(pB / pA) if (np.isfinite(pA) and abs(pA) > 1e-9)
                    else float('nan'),
                    cv=sm(r, O_CVA), swf=sm(r, O_SWF)))

        summ = {}
        for name, _, _ in conds:
            rr = acc[name]
            if not rr:
                continue
            pA = np.array([x['pA'] for x in rr], float)
            pB = np.array([x['pB'] for x in rr], float)
            rt = np.array([x['ratio'] for x in rr], float)
            n = int(np.isfinite(pB).sum())
            k = int(np.nansum(pB > 0))
            lo, hi = wilson(k, n)
            per, pooled = both_conventions(pB, pA)
            near = int(np.nansum(np.abs(rt - CHONG) <= 0.10))
            summ[name] = dict(
                n=n, med_pA=float(np.nanmedian(pA)), med_pB=float(np.nanmedian(pB)),
                n_pB_positive=k, wilson=[lo, hi],
                ratio_per_config=per, ratio_of_medians=pooled,
                n_near_chong=near,
                med_cv=float(np.nanmedian([x['cv'] for x in rr])),
                med_swf=float(np.nanmedian([x['swf'] for x in rr])))
        res[key] = dict(p=p, summary=summ, per_config=acc)

        print(f"\n  p = {p}   [{time.time()-t0:.0f}s]")
        print(f"    {'condition':<20} {'attended':>9} {'competitor':>11} "
              f"{'pB>0':>10} {'ratio(per/pooled)':>20} {'|r-0.31|<=.1':>13} {'CV':>6}")
        for name, _, _ in conds:
            if name not in summ:
                continue
            s = summ[name]
            print(f"    {name:<20} {s['med_pA']:>+9.1f} {s['med_pB']:>+11.1f} "
                  f"{s['n_pB_positive']:>5}/{s['n']:<4} "
                  f"{s['ratio_per_config']:>+9.3f}/{s['ratio_of_medians']:>+9.3f} "
                  f"{s['n_near_chong']:>9}/{s['n']:<4} {s['med_cv']:>6.3f}")

    print(f"""
  HOW TO READ THIS.  Chong et al. (2005) observed a ratio of 0.310.

  Prediction 1: the GATED adaptation rows should show a positive competitor
  response, where the ALWAYS-ON rows should not. Section 4.7.1 found 0 of 44
  same-signed for the always-on version, so if the gated rows are also negative,
  delivery timing does not govern this formulation and Section 4.7.2's account is
  narrower than the paper claims. That would be a real limitation and should be
  reported as one.

  Prediction 2: at a given exponent the gated adaptation ratio should sit BELOW the
  gated increment ratio, because it buys the same competitor response with a larger
  attended response.

  Prediction 3: at p = 2 the gated adaptation ratio should approach 0.310, where the
  gated increment gives roughly 0.63 to 0.95.

  If all three hold, this is no longer a diagnosis. The combination -- superlinear
  adaptation for the two structural results, dominance-gated adaptation modulation
  for the attentional one -- satisfies every constraint the paper tests, and
  Sections 4.7.7, 5.4, the Abstract and the Conclusion all need rewriting around a
  corrected model rather than a located defect. Check the CV column before
  concluding that: a condition outside [0.35, 0.65] has left the regime the rest of
  the study is conducted in, as the anti-gate does.

  If prediction 1 holds but 3 does not, the mechanism is right in kind and wrong in
  magnitude, and the successor problem is narrower again rather than solved.
""")
    save(args.out + 'wave22_F_adaptgate.json', res, args.force)
    return res


# ==========================================================================
# BLOCK F - can a dominance-phase self-gain recover the attentional coupling
#           ratio while holding both structural results?
# ==========================================================================
def block_F(args, elig=None):
    """The successor mechanism named in Section 5.4, tested.

    Section 4.7.7 leaves the architecture in this position: a superlinear
    adaptation law at p ~ 1.75-2 recovers Levelt's second proposition and Modified
    Proposition IV, and displaces the one published measurement of the attentional
    coupling ratio, which rises from 0.318 at p = 1 to 0.746 at p = 2 against Chong
    et al.'s 0.310. Section 5.4 argues that recovering it needs a mechanism acting
    on the attended channel's dominance-phase gain independently of its adaptation,
    and notes that the gate-timing result of Section 4.7.2 is itself a
    dominance-phase manipulation.

    This block tests the obvious candidate. Equation 1's recurrent coefficient
    becomes (1 - lam + delta) for whichever channel is currently dominant, delta
    applied symmetrically to both channels as an architectural feature rather than
    as a manipulation of one. It raises the dominant channel's gain without touching
    the input drive, so it should not reintroduce the input-response defect that
    superlinear adaptation repairs.

    WHY THIS IS NOT JUST ADDING A KNOB. delta changes the baseline dynamics, so it
    can break what has already been won. A smoke test at p = 1 on one configuration
    found the coefficient of variation falling from 0.498 to 0.057 at delta =
    0.25*lam: the gain is positive feedback during dominance and regularises
    alternation sharply. Eligibility requires CV in [0.35, 0.65], so the mechanism
    has to thread between recovering the ratio and destroying the duration
    variability that every configuration in this study was admitted on. Four things
    are therefore checked at once, and a value of delta only counts if it holds all
    four:

        (1) eligibility   CV in [0.35, 0.65] and at least 10 switches per seed
        (2) Proposition IV  alternation rate rises with equal bilateral drive
        (3) Levelt II       reachable at some increment magnitude
        (4) the ratio       gated competitor-to-attended near 0.310

    A single delta holding all four across configurations would make this a
    corrected model rather than a diagnosis. Per-configuration tuning would not:
    report the fraction of configurations satisfying all four at each delta, and
    whether the same delta works for more than a handful.
    """
    print("\n=== BLOCK F: dominance-phase self-gain ===")
    if elig is None:
        elig = _load_elig(args)
    pool = {c['idx']: c for c in load_pool(args.quick)}
    rng = np.random.default_rng(26)
    CHONG, RTOL = 0.310, 0.10
    DELTAS = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50)   # as fractions of lambda
    LEVELS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
    ranks = np.argsort(np.argsort(LEVELS))
    p_list = (1.0, 2.0) if args.quick else (1.0, 1.75, 2.0)

    res = {}
    for p in p_list:
        key = f"{p}"
        pk = key if key in elig else next((k for k in elig
                                           if abs(float(k) - p) < 1e-9), None)
        if pk is None or not elig[pk]['eligible_idx']:
            print(f"  p={p}: no eligible pool, skipped")
            continue
        take = rng.choice(elig[pk]['eligible_idx'],
                          min(args.n_config, len(elig[pk]['eligible_idx'])),
                          replace=False)
        res[key] = {}
        for dfrac in DELTAS:
            t0 = time.time()
            rows = []
            for j, ci in enumerate(take):
                c = pool[int(ci)]
                delta = dfrac * c['lam']
                bl = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps,
                              max(4, args.seeds // 2), BLK_F_ACT, 0,
                              split_indet=SPLIT_INDET)
                xbar = 0.5 * (sm(bl, O_ACTA) + sm(bl, O_ACTB))
                actA = sm(bl, O_ACTA)
                if not np.isfinite(xbar) or xbar <= 0:
                    continue
                ref = 0.5 * c['lam'] * actA
                keff = c['kap'] / (xbar ** (p - 1.0)) if p != 1.0 else c['kap']
                cp = dict(c, kap=keff)

                b = run_cell(cp, p, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps,
                             args.seeds, BLK_F_BASE, LEVEL(p, dfrac),
                             split_indet=SPLIT_INDET, delta=delta)
                dA0, dB0 = sm(b, O_DURA), sm(b, O_DURB)
                cv0, sw0 = sm(b, O_CVA), sm(b, O_SW)
                if not (np.isfinite(dA0) and np.isfinite(dB0)):
                    continue
                ok_elig = bool(sw0 >= 10 and 0.35 <= cv0 <= 0.65)

                # (2) Proposition IV, filtered
                alt = []
                for lv, s in enumerate(LEVELS):
                    r = run_cell(cp, p, s, s, 0.0, G_NONE, 0.0, args.steps,
                                 args.seeds, BLK_F_P4,
                                 LEVEL(p, dfrac) + lv,
                                 split_indet=SPLIT_INDET, delta=delta)
                    alt.append(sm(r, O_SWF))
                a = np.asarray(alt, float)
                rho4 = (float(np.corrcoef(ranks, np.argsort(np.argsort(a)))[0, 1])
                        if np.isfinite(a).all() else float('nan'))
                ok_p4 = bool(np.isfinite(rho4) and rho4 > 0)

                # (3) Levelt II
                ok_l2 = False
                for lv, mlt in enumerate(MULTS):
                    r = run_cell(cp, p, 0.5 + mlt * ref, 0.5, 0.0, G_NONE, 0.0,
                                 args.steps, args.seeds, BLK_F_L2,
                                 LEVEL(p, dfrac) + lv,
                                 split_indet=SPLIT_INDET, delta=delta)
                    pA, pB = pct(sm(r, O_DURA), dA0), pct(sm(r, O_DURB), dB0)
                    if abs(pA) <= LEVELT_A and LEVELT_B[0] <= pB <= LEVELT_B[1]:
                        ok_l2 = True
                        break

                # (4) gated ratio
                g = run_cell(cp, p, 0.5, 0.5, ref, G_ON, 0.0, args.steps,
                             args.seeds, BLK_F_GATE, LEVEL(p, dfrac),
                             split_indet=SPLIT_INDET, delta=delta)
                gA, gB = pct(sm(g, O_DURA), dA0), pct(sm(g, O_DURB), dB0)
                ratio = (gB / gA) if (np.isfinite(gA) and abs(gA) > 1e-9) \
                    else float('nan')
                ok_r = bool(np.isfinite(ratio) and abs(ratio - CHONG) <= RTOL)

                rows.append(dict(config=int(ci), delta_frac=dfrac, cv=cv0,
                                 sw=sw0, rho_prop4=rho4, ratio=ratio,
                                 elig=ok_elig, prop4=ok_p4, levelt2=ok_l2,
                                 ratio_ok=ok_r,
                                 all_four=bool(ok_elig and ok_p4 and ok_l2 and ok_r)))
            n = len(rows)
            cnt = {k: sum(1 for r in rows if r[k])
                   for k in ('elig', 'prop4', 'levelt2', 'ratio_ok', 'all_four')}
            lo, hi = wilson(cnt['all_four'], n) if n else (float('nan'),) * 2
            res[key][str(dfrac)] = dict(
                p=p, delta_frac=dfrac, n=n, counts=cnt, wilson=[lo, hi],
                median_cv=float(np.nanmedian([r['cv'] for r in rows])) if n else None,
                median_ratio=float(np.nanmedian([r['ratio'] for r in rows])) if n else None,
                rows=rows)
            print(f"  p={p:<5} delta={dfrac:<5.2f}*lam  n={n:>4}  "
                  f"medCV {res[key][str(dfrac)]['median_cv']:.3f}  "
                  f"medRatio {res[key][str(dfrac)]['median_ratio']:+.3f}  |  "
                  f"elig {cnt['elig']:>4} propIV {cnt['prop4']:>4} "
                  f"LeveltII {cnt['levelt2']:>4} ratio {cnt['ratio_ok']:>4}  "
                  f"ALL FOUR {cnt['all_four']:>4}  [{time.time()-t0:.0f}s]")

    print(f"\n  {'p':>5} {'delta/lam':>10} {'medCV':>7} {'medRatio':>9} "
          f"{'elig':>6} {'propIV':>7} {'LeveltII':>9} {'ratio':>6} {'ALL 4':>7} "
          f"{'Wilson95':>16}")
    best = (None, -1)
    for k in sorted(res, key=float):
        for dk in sorted(res[k], key=float):
            s = res[k][dk]
            c = s['counts']
            if c['all_four'] > best[1]:
                best = ((s['p'], s['delta_frac']), c['all_four'])
            print(f"  {s['p']:>5.2f} {s['delta_frac']:>10.2f} "
                  f"{s['median_cv']:>7.3f} {s['median_ratio']:>+9.3f} "
                  f"{c['elig']:>6} {c['prop4']:>7} {c['levelt2']:>9} "
                  f"{c['ratio_ok']:>6} {c['all_four']:>7} "
                  f"  [{100*s['wilson'][0]:.1f}%, {100*s['wilson'][1]:.1f}%]")

    (bp, bd), bn = best
    # a mechanism only counts if it beats delta = 0 at the same exponent
    null_at = {}
    for k in res:
        z = res[k].get('0.0')
        if z:
            null_at[float(k)] = z['counts']['all_four']
    beats_null = any(
        s['counts']['all_four'] > null_at.get(s['p'], 0)
        for k in res for s in res[k].values() if s['delta_frac'] > 0)
    print(f"""
  Best cell: p = {bp}, delta = {bd}*lambda, satisfying all four in {bn} configurations.
  Best with delta = 0 (no mechanism): {null_at}
  Any delta > 0 beating its own delta = 0 baseline: {beats_null}
""")
    if bn > 0 and not beats_null:
        print("""  THE MECHANISM CONTRIBUTES NOTHING. Cells satisfying all four exist, but no
  positive delta beats delta = 0 at the same exponent, so the successes belong to
  the baseline model and not to the dominance-phase gain. Read the per-criterion
  columns for the shape of the failure: if the ratio moves the right way at low
  exponent and the wrong way at high exponent, the gain is being cancelled by the
  same superlinear adaptation that repairs the structural results, because both act
  on the dominant channel's activation. That is a stronger negative than a flat
  null, and it tells a successor where NOT to intervene.""")
    elif bn == 0:
        print("""  NO CELL SATISFIES ALL FOUR. The mechanism proposed in Section 5.4 does not
  work: a dominance-phase self-gain cannot recover the attentional coupling ratio
  while holding eligibility, Modified Proposition IV and Levelt's second
  proposition together. Read the per-criterion columns to see which constraint it
  breaks first -- if eligibility falls away as delta rises, the gain is
  regularising the durations, which the smoke test predicted.

  That is a clean negative and worth reporting. It closes the obvious door,
  narrows what a successor must do, and strengthens rather than weakens the
  paper's claim that the displacement of the ratio is structural: the first
  mechanism anyone would reach for does not recover it.""")
    else:
        print("""  AT LEAST ONE CELL SATISFIES ALL FOUR. Before treating this as a corrected
  model, check three things in the JSON. Does a SINGLE delta work across many
  configurations, or does each need its own? Are the satisfying configurations
  distinct in parameter space or adjacent? And does the gate-timing sign contrast
  of Section 4.7.2 survive at that delta -- the block does not test it, and a
  mechanism that recovers the ratio by abolishing the paper's central result is
  not a solution.""")

    save(args.out + 'wave22_F_dominance_gain.json', res, args.force)
    return res


# ==========================================================================
# BLOCK H - inhibition-driven adaptation: does blocking the competitor's
#           de-adaptation recover the coupling ratio, and is adaptation
#           recovery even the mechanism?
# ==========================================================================
def block_H(args, elig=None):
    """Two questions, one manipulation.

    WHERE THIS COMES FROM. Block F ruled out a dominance-phase self-gain: it is
    cancelled by the same superlinear adaptation that repairs the two structural
    failures, because both act on the dominant channel's activation. Section 5.4
    therefore relocates the target to the other side of the network. The gated
    coupling ratio is too high because the COMPETITOR lengthens too much, and Section
    4.7.2 attributes that to the competitor de-adapting while suppressed. Blocking
    that de-adaptation should lower the ratio.

    Adaptation becomes a_i <- (1-gam) a_i + kap*(x_i**p + mu*beta*x_j), so a channel
    adapts to the inhibition it receives as well as to its own activation. A
    suppressed channel receives strong inhibition and therefore stays adapted. kappa
    is renormalised so the baseline adaptation drive is unchanged, which makes mu vary
    the COMPOSITION of adaptation rather than its level.

    Unlike the self-gain of block F this is architectural and always on, so it can
    break eligibility, Modified Proposition IV and Levelt's second proposition. All
    five criteria are therefore checked, including the one block F omitted:

        (1) eligibility        CV in [0.35, 0.65], at least 10 switches per seed
        (2) Proposition IV     alternation rate rises with equal bilateral drive
        (3) Levelt II          reachable at some increment magnitude
        (4) ratio              gated competitor-to-attended near 0.310
        (5) sign contrast      gated competitor > 0 AND ungated competitor < 0

    Criterion 5 matters because a mechanism that recovers the ratio by abolishing the
    paper's central result is not a solution.

    THE SECOND QUESTION, WHICH MAY MATTER MORE. A smoke test on one configuration at
    p = 2 found the ratio falling only from +0.909 to +0.794 as mu went from 0 to 4,
    and saturating. The saturation has a reason: during the attended channel's
    dominance the competitor's adaptation settles to kap*mu*beta*xbar/gam, a plateau
    rather than a decay, so raising mu raises the plateau but stops changing the phase
    structure once the renormalisation compensates the mean.

    If that holds across configurations, it is a problem for the paper rather than for
    the mechanism. Section 4.7.2 states that the gated condition lengthens the
    competitor because a lengthened attended episode gives it more time to de-adapt.
    If de-adaptation can be largely blocked and the coupling barely moves, that
    explanation is incomplete and the coupling has another source -- most likely the
    direct inhibition term, which falls whenever the attended channel is not being
    driven. The block reports the ratio's asymptote in mu explicitly so this can be
    read off.
    """
    print("\n=== BLOCK H: inhibition-driven adaptation ===")
    if elig is None:
        elig = _load_elig(args)
    pool = {c['idx']: c for c in load_pool(args.quick)}
    rng = np.random.default_rng(27)
    CHONG, RTOL = 0.310, 0.10
    MUS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)
    LEVELS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
    ranks = np.argsort(np.argsort(LEVELS))
    p_list = (2.0,) if args.quick else (1.75, 2.0)

    res = {}
    for p in p_list:
        key = f"{p}"
        pk = key if key in elig else next((k for k in elig
                                           if abs(float(k) - p) < 1e-9), None)
        if pk is None or not elig[pk]['eligible_idx']:
            print(f"  p={p}: no eligible pool, skipped")
            continue
        take = rng.choice(elig[pk]['eligible_idx'],
                          min(args.n_config, len(elig[pk]['eligible_idx'])),
                          replace=False)
        res[key] = {}
        for mu in MUS:
            t0 = time.time()
            rows = []
            for ci in take:
                c = pool[int(ci)]
                bl = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps,
                              max(4, args.seeds // 2), BLK_H_ACT, 0,
                              split_indet=SPLIT_INDET)
                xbar = 0.5 * (sm(bl, O_ACTA) + sm(bl, O_ACTB))
                actA = sm(bl, O_ACTA)
                if not np.isfinite(xbar) or xbar <= 0:
                    continue
                ref = 0.5 * c['lam'] * actA
                if not (0.005 <= ref <= 0.5):
                    continue
                # hold baseline adaptation drive fixed: kap*(xbar**p + mu*beta*xbar)
                denom = xbar ** p + mu * c['beta'] * xbar
                keff = c['kap'] * xbar / denom if denom > 0 else c['kap']
                cp = dict(c, kap=keff)

                b = run_cell(cp, p, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps,
                             args.seeds, BLK_H_BASE, LEVEL(p, mu),
                             split_indet=SPLIT_INDET, mu=mu)
                dA0, dB0 = sm(b, O_DURA), sm(b, O_DURB)
                cv0, sw0 = sm(b, O_CVA), sm(b, O_SW)
                if not (np.isfinite(dA0) and np.isfinite(dB0)):
                    continue
                ok_e = bool(sw0 >= 10 and 0.35 <= cv0 <= 0.65)

                alt = []
                for lv, s in enumerate(LEVELS):
                    r = run_cell(cp, p, s, s, 0.0, G_NONE, 0.0, args.steps,
                                 args.seeds, BLK_H_P4, LEVEL(p, mu) + lv,
                                 split_indet=SPLIT_INDET, mu=mu)
                    alt.append(sm(r, O_SWF))
                a = np.asarray(alt, float)
                rho4 = (float(np.corrcoef(ranks, np.argsort(np.argsort(a)))[0, 1])
                        if np.isfinite(a).all() else float('nan'))
                ok_4 = bool(np.isfinite(rho4) and rho4 > 0)

                ok_l2 = False
                for lv, mlt in enumerate(MULTS):
                    r = run_cell(cp, p, 0.5 + mlt * ref, 0.5, 0.0, G_NONE, 0.0,
                                 args.steps, args.seeds, BLK_H_L2,
                                 LEVEL(p, mu) + lv, split_indet=SPLIT_INDET,
                                 mu=mu)
                    pA_, pB_ = pct(sm(r, O_DURA), dA0), pct(sm(r, O_DURB), dB0)
                    if abs(pA_) <= LEVELT_A and LEVELT_B[0] <= pB_ <= LEVELT_B[1]:
                        ok_l2 = True
                        break

                g = run_cell(cp, p, 0.5, 0.5, ref, G_ON, 0.0, args.steps,
                             args.seeds, BLK_H_GATE, LEVEL(p, mu),
                             split_indet=SPLIT_INDET, mu=mu)
                u = run_cell(cp, p, 0.5, 0.5, ref, G_UNGATED, 0.0, args.steps,
                             args.seeds, BLK_H_UNG, LEVEL(p, mu),
                             split_indet=SPLIT_INDET, mu=mu)
                gA, gB = pct(sm(g, O_DURA), dA0), pct(sm(g, O_DURB), dB0)
                uB = pct(sm(u, O_DURB), dB0)
                ratio = (gB / gA) if (np.isfinite(gA) and abs(gA) > 1e-9) \
                    else float('nan')
                ok_r = bool(np.isfinite(ratio) and abs(ratio - CHONG) <= RTOL)
                ok_s = bool(np.isfinite(gB) and np.isfinite(uB) and gB > 0 > uB)

                rows.append(dict(config=int(ci), mu=mu, cv=cv0, rho_prop4=rho4,
                                 ratio=ratio, gB=gB, uB=uB,
                                 elig=ok_e, prop4=ok_4, levelt2=ok_l2,
                                 ratio_ok=ok_r, sign=ok_s,
                                 all_five=bool(ok_e and ok_4 and ok_l2 and ok_r and ok_s)))
            n = len(rows)
            cnt = {k: sum(1 for r in rows if r[k]) for k in
                   ('elig', 'prop4', 'levelt2', 'ratio_ok', 'sign', 'all_five')}
            lo, hi = wilson(cnt['all_five'], n) if n else (float('nan'),) * 2
            med_r = float(np.nanmedian([r['ratio'] for r in rows])) if n else None
            res[key][str(mu)] = dict(p=p, mu=mu, n=n, counts=cnt, wilson=[lo, hi],
                                     median_cv=float(np.nanmedian([r['cv'] for r in rows])) if n else None,
                                     median_ratio=med_r, rows=rows)
            print(f"  p={p:<5} mu={mu:<5.1f}  n={n:>4}  medCV "
                  f"{res[key][str(mu)]['median_cv']:.3f}  medRatio {med_r:+.3f}  |  "
                  f"elig {cnt['elig']:>4} propIV {cnt['prop4']:>4} "
                  f"L2 {cnt['levelt2']:>4} ratio {cnt['ratio_ok']:>4} "
                  f"sign {cnt['sign']:>4}  ALL FIVE {cnt['all_five']:>4}  "
                  f"[{time.time()-t0:.0f}s]")

    print(f"\n  {'p':>5} {'mu':>6} {'medCV':>7} {'medRatio':>9} {'elig':>6} "
          f"{'propIV':>7} {'L2':>5} {'ratio':>6} {'sign':>5} {'ALL 5':>7}")
    for k in sorted(res, key=float):
        for mk in sorted(res[k], key=float):
            s = res[k][mk]
            c = s['counts']
            print(f"  {s['p']:>5.2f} {s['mu']:>6.1f} {s['median_cv']:>7.3f} "
                  f"{s['median_ratio']:>+9.3f} {c['elig']:>6} {c['prop4']:>7} "
                  f"{c['levelt2']:>5} {c['ratio_ok']:>6} {c['sign']:>5} "
                  f"{c['all_five']:>7}")

    # asymptote in mu, per exponent
    print("\n  RATIO ASYMPTOTE IN MU (bears on the mechanism claim in 4.7.2)")
    for k in sorted(res, key=float):
        mus = sorted(res[k], key=float)
        r0 = res[k][mus[0]]['median_ratio']
        rN = res[k][mus[-1]]['median_ratio']
        rP = res[k][mus[-2]]['median_ratio'] if len(mus) > 1 else rN
        print(f"    p = {k}: ratio {r0:+.3f} at mu = {mus[0]} -> {rN:+.3f} at "
              f"mu = {mus[-1]}   (previous step {rP:+.3f}, "
              f"change {abs(rN - rP):.3f})")
        if abs(rN - rP) < 0.03 and abs(rN - CHONG) > 2 * RTOL:
            print(f"""      -> SATURATED well above {CHONG}. Blocking the competitor's
         de-adaptation does not recover the ratio, so adaptation recovery is not
         the main source of the gated coupling. Section 4.7.2 attributes it to
         adaptation recovery and that attribution needs weakening: the residual
         is most plausibly the direct inhibition term, which falls whenever the
         attended channel is not being driven. This is a correction to the
         paper's own mechanism, and it does not touch the gate-timing result
         itself, which is established by manipulation rather than by mechanism.""")
        elif abs(rN - CHONG) <= RTOL:
            print("      -> reaches the observed value. Check the ALL FIVE column: "
                  "the ratio\n         alone is not the claim.")

    best = max((s for k in res for s in res[k].values()),
               key=lambda s: s['counts']['all_five'], default=None)
    if best is not None:
        null = res[f"{best['p']}"].get('0.0', {}).get('counts', {}).get('all_five', 0)
        print(f"\n  Best cell: p = {best['p']}, mu = {best['mu']}, all five in "
              f"{best['counts']['all_five']} configurations (mu = 0 baseline: {null})")
        if best['counts']['all_five'] <= null:
            print("""  THE MECHANISM CONTRIBUTES NOTHING over its own mu = 0 baseline. Read the
  per-criterion columns for which constraint it fails, and the asymptote above for
  whether it moved the ratio at all.""")

    save(args.out + 'wave22_H_inhib_adapt.json', res, args.force)
    return res


def _spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return float('nan')
    rx = np.argsort(np.argsort(x[ok]))
    ry = np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def _partial(x, y, z):
    """Spearman partial correlation of x with y controlling for z."""
    rxy, rxz, ryz = _spear(x, y), _spear(x, z), _spear(y, z)
    d = math.sqrt(max(1e-12, (1 - rxz ** 2) * (1 - ryz ** 2)))
    return (rxy - rxz * ryz) / d


# ==========================================================================
# BLOCK I - closes the four data-backed markers left in the manuscript
# ==========================================================================
def block_I(args, elig=None):
    """Four quantities the manuscript flags as needing computation.

    (A) Section 4.7.6 reports the architectural bound lam/(lam+beta+alpha*kappa/gam)
        against the maximum achievable attentional effect at rho = +0.774, with
        +0.722 in a partly overlapping recomputation. Section 3.7 requires four
        independent draws with the range reported, and the bound has lambda in its
        numerator while lambda takes only four grid values, so rho for lambda alone
        and the partial correlation controlling for it are both needed before the
        composite can be said to add anything.

    (B) Section 4.7.1 reports the coupling relationship with the predictor averaged
        over the competitor's whole dominance episode, which makes the averaging
        window an outcome. The fixed-window control needs stating on one sample and
        across window lengths.

    (C) Section 4.7.2 reports the anti-gate comparison on 91 of 100 configurations
        without saying why nine dropped, and its criterion table lacks the
        anti-gated 1x row.

    (D) Section 4.7.4's ratio sign counts come from block C, which ran on the
        wave22 pool. They should be quoted from settings matching the canonical
        campaign.
    """
    print("\n=== BLOCK I: closing the outstanding manuscript figures ===")
    if elig is None:
        elig = _load_elig(args)
    pk = next((k for k in elig if abs(float(k) - 1.0) < 1e-9), None) or \
        sorted(elig, key=float)[0]
    pool_idx = elig[pk]['eligible_idx']
    pool = {c['idx']: c for c in load_pool(args.quick)}
    out = {}

    # ---------------- (A) four draws on the bound ----------------
    print("\n  (A) architectural bound, four independent draws")
    GS = (0.25, 0.50, 0.75, 0.95)
    rows_by_draw = []
    for d in range(4):
        rng = np.random.default_rng(2400 + d)
        take = rng.choice(pool_idx, min(args.n_config, len(pool_idx)), replace=False)
        rec = []
        for ci in take:
            c = pool[int(ci)]
            b = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps,
                         args.seeds, BLK_I_BASE, d, split_indet=SPLIT_INDET)
            dA0 = sm(b, O_DURA)
            if not np.isfinite(dA0) or sm(b, O_SW) < 10:
                continue
            best = -1e9
            for lv, gf in enumerate(GS):
                r = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps,
                             args.seeds, BLK_I_GOAL, d * 10 + lv,
                             split_indet=SPLIT_INDET, g_a=gf * c['lam'])
                v = pct(sm(r, O_DURA), dA0)
                if np.isfinite(v):
                    best = max(best, v)
            if best <= -1e8:
                continue
            bound = c['lam'] / (c['lam'] + c['beta']
                                + c['alpha'] * c['kap'] / c['gam'])
            rec.append(dict(config=int(ci), max_gain=best, bound=bound,
                            lam=c['lam']))
        if not rec:
            continue
        mg = [r['max_gain'] for r in rec]
        bd = [r['bound'] for r in rec]
        lm = [r['lam'] for r in rec]
        rows_by_draw.append(dict(
            draw=d, n=len(rec),
            rho_bound=_spear(bd, mg), rho_lam=_spear(lm, mg),
            partial_bound_given_lam=_partial(bd, mg, lm), rows=rec))
        s = rows_by_draw[-1]
        print(f"    draw {d}: n={s['n']:>4}  rho(bound) {s['rho_bound']:+.3f}  "
              f"rho(lambda alone) {s['rho_lam']:+.3f}  "
              f"partial(bound | lambda) {s['partial_bound_given_lam']:+.3f}")
    if rows_by_draw:
        rb = [s['rho_bound'] for s in rows_by_draw]
        rl = [s['rho_lam'] for s in rows_by_draw]
        rp = [s['partial_bound_given_lam'] for s in rows_by_draw]
        print(f"\n    ACROSS DRAWS  rho(bound) median {np.median(rb):+.3f} "
              f"range [{min(rb):+.3f}, {max(rb):+.3f}]")
        print(f"                  rho(lambda) median {np.median(rl):+.3f} "
              f"range [{min(rl):+.3f}, {max(rl):+.3f}]")
        print(f"                  partial      median {np.median(rp):+.3f} "
              f"range [{min(rp):+.3f}, {max(rp):+.3f}]")
        if abs(np.median(rp)) < 0.2:
            print("""    -> The composite adds little once lambda is controlled. Section 4.7.6
       should say the bound is largely a statement about leak rate, which is
       still a result but a narrower one than the expression suggests.""")
        else:
            print("""    -> The composite survives controlling for lambda, so it is not merely
       restating that high leak permits large modulation.""")
    out['A_bound'] = rows_by_draw

    # ---------------- (B) fixed-window control ----------------
    print("\n  (B) coupling relationship by averaging window")
    rng = np.random.default_rng(2410)
    take = rng.choice(pool_idx, min(args.n_config, len(pool_idx)), replace=False)
    WIN = (('whole episode', O_SPA), ('first 10', O_SPA10),
           ('first 25', O_SPA25), ('first 50', O_SPA50))
    per_cfg = {lab: [] for lab, _ in WIN}
    for ci in take:
        c = pool[int(ci)]
        b = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps, args.seeds,
                     BLK_I_BASE, 99, split_indet=SPLIT_INDET)
        dB0 = sm(b, O_DURB)
        if not np.isfinite(dB0) or sm(b, O_SW) < 10:
            continue
        spa = {lab: [] for lab, _ in WIN}
        comp = []
        for lv, gf in enumerate((0.25, 0.50, 0.75, 0.90)):
            r = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps,
                         args.seeds, BLK_I_WIN, lv, split_indet=SPLIT_INDET,
                         g_a=gf * c['lam'])
            comp.append(pct(sm(r, O_DURB), dB0))
            for lab, slot in WIN:
                spa[lab].append(sm(r, slot))
        for lab, _ in WIN:
            rho = _spear(spa[lab], comp)
            if np.isfinite(rho):
                per_cfg[lab].append(rho)
    print(f"    {'window':>16} {'n':>5} {'median rho':>11} {'IQR':>20}")
    for lab, _ in WIN:
        v = np.asarray(per_cfg[lab], float)
        if v.size == 0:
            continue
        q1, q3 = np.percentile(v, [25, 75])
        print(f"    {lab:>16} {v.size:>5} {np.median(v):>+11.3f} "
              f"  [{q1:+.3f}, {q3:+.3f}]")
    out['B_windows'] = {lab: list(map(float, per_cfg[lab])) for lab, _ in WIN}
    print("""    -> If the fixed-window correlations are close to the whole-episode one,
       the relationship does not depend on the averaging window being an outcome.
       If they fall away as the window shortens, it partly does, and Section
       4.7.1 must say so.""")

    # ---------------- (C) anti-gate: 1x row and the exclusion rule ----------
    print("\n  (C) anti-gated 1x, and why nine configurations dropped")
    rng = np.random.default_rng(2420)
    take = rng.choice(pool_idx, min(args.n_config, len(pool_idx)), replace=False)
    conds = [('gated_1x', G_ON, 1.0), ('gated_2x', G_ON, 2.0),
             ('ungated_1x', G_UNGATED, 1.0), ('ungated_2x', G_UNGATED, 2.0),
             ('antigated_1x', G_OFF, 1.0), ('antigated_2x', G_OFF, 2.0)]
    acc = {n: [] for n, _, _ in conds}
    drop = dict(no_baseline=0, few_switches=0, nan_duration=0, kept=0)
    for ci in take:
        c = pool[int(ci)]
        bl = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps,
                      max(4, args.seeds // 2), BLK_I_BASE, 7,
                      split_indet=SPLIT_INDET)
        ref = 0.5 * c['lam'] * sm(bl, O_ACTA)
        if not (0.005 <= ref <= 0.5):
            drop['no_baseline'] += 1
            continue
        b = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps, args.seeds,
                     BLK_I_BASE, 8, split_indet=SPLIT_INDET)
        dA0, dB0 = sm(b, O_DURA), sm(b, O_DURB)
        absthr = 0.5 * sm(b, O_ACTB)
        if sm(b, O_SW) < 10:
            drop['few_switches'] += 1
            continue
        if not (np.isfinite(dA0) and np.isfinite(dB0)):
            drop['nan_duration'] += 1
            continue
        drop['kept'] += 1
        for lv, (nm, gate, amp) in enumerate(conds):
            r = run_cell(c, 1.0, 0.5, 0.5, amp * ref, gate, absthr, args.steps,
                         args.seeds, BLK_I_COND, lv, split_indet=SPLIT_INDET)
            acc[nm].append(dict(
                config=int(ci), pA=pct(sm(r, O_DURA), dA0),
                pB=pct(sm(r, O_DURB), dB0), cv=sm(r, O_CVA), swf=sm(r, O_SWF),
                absB=sm(r, O_DURPOOL)))
    print(f"    exclusions: {drop}")
    print(f"    {'condition':>14} {'n':>5} {'attended':>9} {'competitor':>11} "
          f"{'CV':>7} {'swF':>7}")
    summ = {}
    for nm, _, _ in conds:
        rr = acc[nm]
        if not rr:
            continue
        pa = np.array([x['pA'] for x in rr], float)
        pb = np.array([x['pB'] for x in rr], float)
        summ[nm] = dict(n=len(rr), med_pA=float(np.nanmedian(pa)),
                        med_pB=float(np.nanmedian(pb)),
                        n_pB_pos=int(np.nansum(pb > 0)),
                        med_cv=float(np.nanmedian([x['cv'] for x in rr])),
                        med_swf=float(np.nanmedian([x['swf'] for x in rr])))
        s = summ[nm]
        print(f"    {nm:>14} {s['n']:>5} {s['med_pA']:>+9.1f} {s['med_pB']:>+11.1f} "
              f"{s['med_cv']:>7.3f} {s['med_swf']:>7.1f}")
    out['C_conditions'] = dict(exclusions=drop, summary=summ)

    # ---------------- (D) ratio sign counts ----------------
    print("\n  (D) ratio sign counts and ranges, canonical settings")
    for nm in ('gated_1x', 'gated_2x', 'ungated_1x', 'ungated_2x'):
        rr = acc.get(nm, [])
        rat = [x['pB'] / x['pA'] for x in rr
               if np.isfinite(x['pA']) and abs(x['pA']) > 1e-9
               and np.isfinite(x['pB'])]
        if not rat:
            continue
        v = np.asarray(rat, float)
        q1, q3 = np.percentile(v, [25, 75])
        print(f"    {nm:>12} n={v.size:>4}  positive {int(np.sum(v > 0)):>4}/{v.size:<4}"
              f"  median {np.median(v):+.3f}  IQR [{q1:+.3f}, {q3:+.3f}]"
              f"  range [{v.min():+.3f}, {v.max():+.3f}]")
    out['D_ratios'] = {nm: [float(x['pB'] / x['pA']) for x in acc.get(nm, [])
                            if np.isfinite(x['pA']) and abs(x['pA']) > 1e-9]
                       for nm in ('gated_1x', 'gated_2x', 'ungated_1x', 'ungated_2x')}
    print("""    -> Quote the sign counts. The ranges are contaminated by configurations
       whose attended-channel change passes near zero, which is the ratio
       instability Section 5.5 warns about; that is a reason not to quote ranges
       rather than a finding about gating.""")

    save(args.out + 'wave22_I_closeout.json', out, args.force)
    return out


# ==========================================================================
# BLOCK J - the Eq 2 discrepancy at registered resolution
# ==========================================================================
def block_J(args, elig=None):
    """Grid-wide scope of the adaptation-update-order discrepancy.

    Section 5.6 reports that the manuscript's Equation 2 as originally written
    specified adaptation driven by x(t) while the pipeline drives it from x(t+1),
    and quantifies the scope: rank correlation +0.985 between the two coefficients
    of variation, 14.3% against 13.6% of configurations inside the registered CV
    window, and an eligible-pool overlap of 69%. Those figures were computed on a
    reduced run of 6 seeds x 6,000 timesteps and carry a marker in the manuscript
    saying so. This block recomputes them at registered resolution.

    Both arms are run over every rivalry-producing configuration in the grid, once
    with adapt=1 (the pipeline, a(t+1) from x(t+1)) and once with adapt=0 (Equation 2
    as published, a(t+1) from x(t)). Nothing else differs, including the seeds, which
    come from the same make_seed calls in both arms so the comparison is paired.

    The reported quantities are those the manuscript cites: median CV per arm, the
    fraction inside [0.35, 0.65], the Spearman correlation between the two arms'
    CVs, and the overlap between the eligible pools each arm defines. The pool
    comparison uses rivalry plus the CV window only, not the Levelt-rho criterion,
    which is what Section 5.6's 69% figure was based on.
    """
    print("\n=== BLOCK J: Eq 2 update order, grid-wide, registered resolution ===")
    pool = [c for c in load_pool(args.quick) if c['rivalry']]
    print(f"  {len(pool)} rivalry-producing configurations")
    print(f"  {args.seeds} seeds x {args.steps} timesteps, both arms, paired seeds")
    est = len(pool) * args.seeds * args.steps * 2 / 2.0e7
    print(f"  rough estimate {est/60:.0f} min\n", flush=True)

    rows = []
    t0 = time.time()
    for j, c in enumerate(pool):
        if (j + 1) % PROGRESS_EVERY == 0:
            el = time.time() - t0
            print(f"    {j+1}/{len(pool)}  [{el:.0f}s, "
                  f"~{el/(j+1)*(len(pool)-j-1):.0f}s left]", flush=True)
        rec = dict(config=c['idx'], lam=c['lam'], beta=c['beta'], alpha=c['alpha'],
                   sigma=c['sigma'], gam=c['gam'], kap=c['kap'])
        for tag, ad in (('pipeline', 1), ('eq2', 0)):
            r = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps,
                         args.seeds, BLK_J_BASE, 0, split_indet=SPLIT_INDET,
                         adapt=ad)
            rec[f'cv_{tag}'] = sm(r, O_CVA)
            rec[f'sw_{tag}'] = sm(r, O_SW)
            rec[f'dur_{tag}'] = sm(r, O_DURA)
        rows.append(rec)

    def arr(k):
        return np.array([r[k] for r in rows], float)
    cvP, cvE = arr('cv_pipeline'), arr('cv_eq2')
    swP, swE = arr('sw_pipeline'), arr('sw_eq2')
    ok = np.isfinite(cvP) & np.isfinite(cvE)
    inP = ok & (swP >= 10) & (cvP >= 0.35) & (cvP <= 0.65)
    inE = ok & (swE >= 10) & (cvE >= 0.35) & (cvE <= 0.65)

    rho = _spear(cvP[ok], cvE[ok])
    both = int(np.sum(inP & inE))
    overlap = both / max(int(np.sum(inP)), 1)

    print(f"\n  {'arm':>10} {'median CV':>10} {'in [.35,.65]':>13} {'eligible':>9} "
          f"{'median dur':>11}")
    for tag, cv, sw in (('pipeline', cvP, swP), ('Eq 2', cvE, swE)):
        m = ok & (sw >= 10)
        frac = np.mean((cv[m] >= 0.35) & (cv[m] <= 0.65)) if m.any() else np.nan
        print(f"  {tag:>10} {np.nanmedian(cv[ok]):>10.3f} {100*frac:>12.1f}% "
              f"{int(np.sum(m & (cv >= 0.35) & (cv <= 0.65))):>9} "
              f"{np.nanmedian(arr('dur_pipeline' if tag=='pipeline' else 'dur_eq2')[ok]):>11.1f}")
    print(f"\n  Spearman(CV_pipeline, CV_Eq2) = {rho:+.3f}  on {int(ok.sum())} configurations")
    print(f"  eligible under pipeline only: {int(np.sum(inP & ~inE))}")
    print(f"  eligible under Eq 2 only:     {int(np.sum(inE & ~inP))}")
    print(f"  eligible under both:          {both}")
    print(f"  overlap = {100*overlap:.1f}% of the pipeline's pool")

    # divergence by adaptation coupling strength, which is where the effect lives
    ak = arr('alpha') * arr('kap')
    print(f"\n  {'alpha*kappa':>15} {'n':>6} {'medCV pipe':>11} {'medCV Eq2':>10} {'ratio':>7}")
    for lo, hi in ((0, .002), (.002, .005), (.005, .01), (.01, .02)):
        s = ok & (ak >= lo) & (ak < hi)
        if s.sum() < 20:
            continue
        a_, b_ = np.nanmedian(cvP[s]), np.nanmedian(cvE[s])
        print(f"  [{lo:.3f},{hi:.3f}) {int(s.sum()):>6} {a_:>11.3f} {b_:>10.3f} "
              f"{a_/b_ if b_ else float('nan'):>7.2f}")

    print(f"""
  These four numbers replace the marked ones in Section 5.6: the Spearman
  correlation, the two window fractions, and the pool overlap. Remove the
  [TO SUPPLY] marker and state the resolution used here.

  The comparison is paired, since both arms draw the same seeds from the same
  make_seed calls, so the correlation is between two measurements of the same
  configurations and not between two samples.
""")
    save(args.out + 'wave22_J_updateorder.json',
         dict(n=len(rows), seeds=args.seeds, steps=args.steps,
              spearman=rho, median_cv_pipeline=float(np.nanmedian(cvP[ok])),
              median_cv_eq2=float(np.nanmedian(cvE[ok])),
              n_eligible_pipeline=int(inP.sum()), n_eligible_eq2=int(inE.sum()),
              n_eligible_both=both, overlap=overlap, rows=rows), args.force)
    return rows


# ==========================================================================
# BLOCK K - the exponent scan on a FIXED configuration set
# ==========================================================================
def block_K(args, elig=None):
    """Does superlinear adaptation change behaviour, or change which configurations
    are tested?

    Section 4.8 re-derives eligibility at each exponent, and the eligible count rises
    from 946 at p = 1 to 1,655 at p = 3. The defence offered there is that the same
    criterion is applied throughout, which is not sufficient: the same criterion
    applied to different dynamics selects different regions of parameter space. So
    monotone Levelt II reachability in p could be a change in WHICH configurations are
    tested rather than a change in behaviour at fixed configuration, and until that is
    ruled out the table establishes only that the population satisfying the filter
    shifts with p.

    This block runs the same sweep on a fixed set, two ways:

      FIXED-INTERSECTION  configurations eligible at EVERY exponent. The cleanest
        comparison, but the set is selected on being robust to the exponent, which is
        its own filter.
      UNFILTERED          a random sample of rivalry-producing configurations with no
        CV and no Levelt criterion. Not selected on anything the exponent affects, at
        the cost of including configurations the rest of the paper excludes.

    Both are reported. If reachability rises with p on either fixed set, the effect is
    behavioural. If it rises only when eligibility is re-derived, Section 4.8 must be
    rewritten as a statement about pool composition.
    """
    print("\n=== BLOCK K: exponent scan on a fixed configuration set ===")
    if elig is None:
        elig = _load_elig(args)
    pool = {c['idx']: c for c in load_pool(args.quick)}
    rng = np.random.default_rng(28)

    # (a) intersection of the per-exponent eligible sets
    sets = [set(v['eligible_idx']) for v in elig.values() if v['eligible_idx']]
    inter = sorted(set.intersection(*sets)) if sets else []
    print(f"  intersection of all {len(sets)} per-exponent pools: {len(inter)} configurations")

    # (b) unfiltered rivalry-producing sample
    riv = [c['idx'] for c in load_pool(args.quick) if c['rivalry']]
    unfil = sorted(rng.choice(riv, min(args.n_config, len(riv)), replace=False))
    print(f"  unfiltered rivalry-producing sample: {len(unfil)} configurations")

    LEVELS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
    ranks = np.argsort(np.argsort(LEVELS))
    res = {}
    for label, idx in (('fixed_intersection', inter[:args.n_config]),
                       ('unfiltered', unfil)):
        if not idx:
            print(f"  {label}: empty, skipped")
            continue
        res[label] = {}
        print(f"\n  --- {label}, n = {len(idx)} ---")
        for p in (P_VALUES if not args.quick else (1.0, 2.0)):
            t0 = time.time()
            hit = alive = dd = 0
            rows = []
            for j, ci in enumerate(idx):
                c = pool[int(ci)]
                bl = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps,
                              max(4, args.seeds // 2), BLK_K_ACT, 0,
                              split_indet=SPLIT_INDET)
                xbar = 0.5 * (sm(bl, O_ACTA) + sm(bl, O_ACTB))
                ref = 0.5 * c['lam'] * sm(bl, O_ACTA)
                if not (np.isfinite(xbar) and xbar > 0 and 0.005 <= ref <= 0.5):
                    continue
                keff = c['kap'] / (xbar ** (p - 1.0)) if p != 1.0 else c['kap']
                cp = dict(c, kap=keff)
                b = run_cell(cp, p, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps,
                             args.seeds, BLK_K_BASE, int(p * 100),
                             split_indet=SPLIT_INDET)
                dA0, dB0 = sm(b, O_DURA), sm(b, O_DURB)
                if not (np.isfinite(dA0) and np.isfinite(dB0) and sm(b, O_SW) >= 10):
                    continue
                alive += 1
                h = False
                for lv, m in enumerate(MULTS):
                    r = run_cell(cp, p, 0.5 + m * ref, 0.5, 0.0, G_NONE, 0.0,
                                 args.steps, args.seeds, BLK_K_SWEEP,
                                 int(p * 100) + lv, split_indet=SPLIT_INDET)
                    pA, pB = pct(sm(r, O_DURA), dA0), pct(sm(r, O_DURB), dB0)
                    if abs(pA) <= LEVELT_A and LEVELT_B[0] <= pB <= LEVELT_B[1]:
                        h = True
                        break
                hit += int(h)
                alt = []
                for lv, s in enumerate(LEVELS):
                    r = run_cell(cp, p, s, s, 0.0, G_NONE, 0.0, args.steps,
                                 args.seeds, BLK_K_P4, int(p * 100) + lv,
                                 split_indet=SPLIT_INDET)
                    alt.append(sm(r, O_SWF))
                a = np.asarray(alt, float)
                if np.isfinite(a).all():
                    rho = float(np.corrcoef(ranks, np.argsort(np.argsort(a)))[0, 1])
                    dd += int(rho > 0)
                    rows.append(dict(config=int(ci), levelt=h, rho4=rho))
            lo, hi = wilson(hit, alive) if alive else (float('nan'),) * 2
            res[label][str(p)] = dict(p=p, n=len(idx), alive=alive, levelt=hit,
                                      wilson=[lo, hi], prop4_dd=dd, rows=rows)
            print(f"    p={p:<5} alive {alive:>4}  LeveltII {hit:>4} "
                  f"[{100*lo:.1f}%, {100*hi:.1f}%]  PropIV DD {dd:>4}  "
                  f"[{time.time()-t0:.0f}s]")

    print("""
  HOW TO READ THIS. Compare each row against Section 4.8's re-derived-eligibility
  figures (3, 6, 9, 20, 44, 78, 91 of 200). If reachability still rises with p on a
  set that does not change with p, the effect is behavioural and Section 4.8 stands
  with a note that the fixed-set version reproduces it. If it is flat, Section 4.8 is
  a statement about pool composition and must be rewritten as one.
""")
    save(args.out + 'wave22_K_fixedset.json', res, args.force)
    return res


# ==========================================================================
# BLOCK L - the gate-timing series on the pool defined by published Eq. 2
# ==========================================================================
def block_L(args, elig=None):
    """Is the central result an artefact of pool composition?

    Section 5.6 reports that the eligible pool under the published form of Equation 2
    overlaps the pipeline's pool by only 71.3%, and Section 4.5.1 reports that the
    coupling sign contrast weakens substantially outside the eligibility filter (61 of
    100 versus 100 of 100). Pool composition is therefore doing real work, and the
    most reputationally awkward disclosure in the paper is that the published equation
    is not the executed one.

    This block runs the gate-timing series on 100 configurations drawn from the pool
    defined under the PUBLISHED form, and simulates them under that form. If the sign
    contrast is unchanged, the disclosure converts into a robustness result: the
    central claim does not depend on which of the two update orders is used, either
    for selecting configurations or for running them.
    """
    print("\n=== BLOCK L: gate-timing series under the published Eq. 2 form ===")
    riv = [c for c in load_pool(args.quick) if c['rivalry']]
    print(f"  screening {len(riv)} rivalry-producing configurations under adapt=0")
    rng = np.random.default_rng(29)
    keep = []
    t0 = time.time()
    for j, c in enumerate(riv):
        if (j + 1) % PROGRESS_EVERY == 0:
            el = time.time() - t0
            print(f"    {j+1}/{len(riv)}  eligible {len(keep)}  [{el:.0f}s, "
                  f"~{el/(j+1)*(len(riv)-j-1):.0f}s left]", flush=True)
        b = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps,
                     max(4, args.seeds // 2), BLK_L_SCREEN, 0,
                     split_indet=SPLIT_INDET, adapt=0)
        cv, sw = sm(b, O_CVA), sm(b, O_SW)
        if np.isfinite(cv) and sw >= 10 and 0.35 <= cv <= 0.65:
            keep.append(c)
    print(f"  eligible under the published form: {len(keep)}")
    if not keep:
        print("  none eligible, cannot proceed")
        return {}
    take = [keep[i] for i in
            sorted(rng.choice(len(keep), min(args.n_config, len(keep)), replace=False))]

    conds = [('gated_1x', G_ON, 1.0), ('gated_2x', G_ON, 2.0),
             ('ungated_1x', G_UNGATED, 1.0), ('ungated_2x', G_UNGATED, 2.0)]
    acc = {n: [] for n, _, _ in conds}
    for c in take:
        bl = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps,
                      max(4, args.seeds // 2), BLK_L_BASE, 0,
                      split_indet=SPLIT_INDET, adapt=0)
        ref = 0.5 * c['lam'] * sm(bl, O_ACTA)
        if not (0.005 <= ref <= 0.5):
            continue
        b = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps, args.seeds,
                     BLK_L_BASE, 1, split_indet=SPLIT_INDET, adapt=0)
        dA0, dB0 = sm(b, O_DURA), sm(b, O_DURB)
        athr = 0.5 * sm(b, O_ACTB)
        if not (np.isfinite(dA0) and np.isfinite(dB0)):
            continue
        for lv, (nm, gate, amp) in enumerate(conds):
            r = run_cell(c, 1.0, 0.5, 0.5, amp * ref, gate, athr, args.steps,
                         args.seeds, BLK_L_COND, lv, split_indet=SPLIT_INDET,
                         adapt=0)
            acc[nm].append(dict(pA=pct(sm(r, O_DURA), dA0),
                                pB=pct(sm(r, O_DURB), dB0),
                                pB_abs=pct(sm(r, O_DURPOOL), sm(b, O_DURPOOL))))

    print(f"\n  {'condition':>12} {'n':>5} {'attended':>9} {'competitor':>11} "
          f"{'pB>0':>10} {'Wilson95':>16}")
    out = {}
    for nm, _, _ in conds:
        rr = [x for x in acc[nm] if np.isfinite(x['pA']) and np.isfinite(x['pB'])]
        if not rr:
            continue
        pb = np.array([x['pB'] for x in rr])
        k = int(np.sum(pb > 0))
        lo, hi = wilson(k, len(rr))
        out[nm] = dict(n=len(rr), med_pA=float(np.median([x['pA'] for x in rr])),
                       med_pB=float(np.median(pb)), n_pos=k, wilson=[lo, hi])
        s = out[nm]
        print(f"  {nm:>12} {s['n']:>5} {s['med_pA']:>+9.1f} {s['med_pB']:>+11.1f} "
              f"{k:>4}/{s['n']:<4} [{100*lo:>5.1f}%, {100*hi:>5.1f}%]")

    g, u = out.get('gated_1x'), out.get('ungated_1x')
    if g and u:
        holds = g['wilson'][0] > 0.5 and u['wilson'][1] < 0.5
        print(f"""
  SIGN CONTRAST UNDER THE PUBLISHED FORM: {'PRESENT' if holds else 'NOT ESTABLISHED'}

  Compare with the pipeline form in Section 4.5.4: gated positive in 180 of 188 and
  ungated positive in 9 of 189. If this reproduces, Section 5.6 gains a sentence
  saying the central result is invariant to the update order in both selection and
  simulation, which converts the discrepancy from a liability into a robustness
  result. If it does not, the result depends on the discretisation and that must be
  stated in Section 4.5.2 and the Abstract.""")
    save(args.out + 'wave22_L_eq2pool.json', dict(n_eligible=len(keep), summary=out,
                                                  rows=acc), args.force)
    return out


# ==========================================================================
# BLOCK M - phase-shifted self-gating: a tighter control than yoked replay
# ==========================================================================
def block_M(args, elig=None):
    """Decompose the gated effect into intermittency, contingency and phase.

    Yoked replay destroys contingency, but a schedule taken from a different trial is
    state-independent and therefore delivers roughly in proportion to each channel's
    predominance. It is approximately a reduced-amplitude, noisy UNGATED increment. So
    live-minus-yoked conflates two things: that the increment is contingent on the
    channel's state at all, and that it is contingent with the correct PHASE.

    The tighter control gates on the same trial's own dominance time-course, delayed by
    a fraction of a mean episode. That preserves duty cycle, episode-length
    distribution and autocorrelation with the trial's own dynamics, and varies phase
    alone.

    NOISE IS HELD FIXED ACROSS THE COMPARISON. For each configuration and each seed the
    live gate is recorded and then replayed at the SAME seed, so shift 0 reproduces the
    live condition exactly and any deviation at shift 0 is a bug rather than a result.
    That validity check is printed. An earlier version of this block replayed on fresh
    noise, which made the shift-0 condition another yoked condition and the
    decomposition meaningless.

        live gated  minus  yoked          = contingency + phase
        live gated  minus  phase-shifted  = phase alone
        yoked       minus  ungated        = intermittency

    Section 4.5.2 says contingency accounts for the majority of the effect. That is
    under-identified until phase is separated from it.
    """
    print("\n=== BLOCK M: phase-shifted self-gating ===")
    if elig is None:
        elig = _load_elig(args)
    pk = next((k for k in elig if abs(float(k) - 1.0) < 1e-9), sorted(elig, key=float)[0])
    pool = {c['idx']: c for c in load_pool(args.quick)}
    rng = np.random.default_rng(30)
    take = rng.choice(elig[pk]['eligible_idx'],
                      min(args.n_config, len(elig[pk]['eligible_idx'])), replace=False)
    FRACS = (0.0, 0.25, 0.50, 0.75, 1.00)
    KEYS = [f"shift_{f:.2f}" for f in FRACS] + ['live', 'yoked', 'ungated',
                                                'ungated_2x']
    rows = {k: [] for k in KEYS}

    t0 = time.time()
    for j2, ci in enumerate(take):
        if (j2 + 1) % PROGRESS_EVERY == 0:
            el = time.time() - t0
            print(f"    {j2+1}/{len(take)}  [{el:.0f}s, "
                  f"~{el/(j2+1)*(len(take)-j2-1):.0f}s left]", flush=True)
        c = pool[int(ci)]
        bl = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps,
                      max(4, args.seeds // 2), BLK_M_BASE, 0, split_indet=SPLIT_INDET)
        ref = 0.5 * c['lam'] * sm(bl, O_ACTA)
        b = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps, args.seeds,
                     BLK_M_BASE, 1, split_indet=SPLIT_INDET)
        dB0, D = sm(b, O_DURB), sm(b, O_DURA)
        if not (np.isfinite(dB0) and D > 2 and 0.005 <= ref <= 0.5):
            continue

        # per-seed: record the live gate, then replay it shifted at the SAME seed
        acc = {k: [] for k in KEYS}
        for s in range(args.seeds):
            lvl = 100 + s                       # same level -> same seed for both calls
            sch = np.zeros(args.steps, dtype=np.uint8)
            live = run_cell(c, 1.0, 0.5, 0.5, 2.0 * ref, G_ON, 0.0, args.steps, 1,
                            BLK_M_REC, lvl, schedule=sch, record=True,
                            split_indet=SPLIT_INDET)
            acc['live'].append(sm(live, O_DURB))
            for f in FRACS:
                shift = int(round(f * D)) % args.steps
                sh = np.roll(sch, shift) if shift else sch.copy()
                r = run_cell(c, 1.0, 0.5, 0.5, 2.0 * ref, G_REPLAY, 0.0, args.steps,
                             1, BLK_M_REC, lvl, schedule=sh, split_indet=SPLIT_INDET)
                acc[f"shift_{f:.2f}"].append(sm(r, O_DURB))
            # yoked: schedule from a DIFFERENT seed, dynamics at this one
            yk = np.zeros(args.steps, dtype=np.uint8)
            run_cell(c, 1.0, 0.5, 0.5, 2.0 * ref, G_ON, 0.0, args.steps, 1,
                     BLK_M_REC, 500 + s, schedule=yk, record=True,
                     split_indet=SPLIT_INDET)
            r = run_cell(c, 1.0, 0.5, 0.5, 2.0 * ref, G_REPLAY, 0.0, args.steps, 1,
                         BLK_M_REC, lvl, schedule=yk, split_indet=SPLIT_INDET)
            acc['yoked'].append(sm(r, O_DURB))
            # dose-matched reference: the gated conditions run at 2x amplitude on a
            # duty cycle near 0.5, so the comparable continuous condition is 1x.
            # A 2x continuous condition delivers roughly twice the dose and its
            # difference from yoked is confounded with dose, not intermittency.
            r = run_cell(c, 1.0, 0.5, 0.5, 1.0 * ref, G_UNGATED, 0.0, args.steps, 1,
                         BLK_M_REC, lvl, split_indet=SPLIT_INDET)
            acc['ungated'].append(sm(r, O_DURB))
            r = run_cell(c, 1.0, 0.5, 0.5, 2.0 * ref, G_UNGATED, 0.0, args.steps, 1,
                         BLK_M_REC, lvl, split_indet=SPLIT_INDET)
            acc['ungated_2x'].append(sm(r, O_DURB))
        for k in KEYS:
            v = np.nanmean(acc[k]) if acc[k] else np.nan
            rows[k].append(pct(v, dB0))

    def med(k):
        v = np.array([x for x in rows[k] if np.isfinite(x)], float)
        return (float(np.median(v)), v.size) if v.size else (float('nan'), 0)

    print(f"\n  {'condition':>24} {'n':>5} {'competitor':>11}")
    for f in FRACS:
        m_, n_ = med(f"shift_{f:.2f}")
        print(f"  {('replay, shift ' + format(f, '.2f') + ' D'):>24} {n_:>5} {m_:>+11.1f}")
    for k, lab in (('live', 'live gate (contingent)'), ('yoked', 'yoked (other seed)'),
                   ('ungated', 'ungated 1x (dose-matched)'),
                   ('ungated_2x', 'ungated 2x (double dose)')):
        m_, n_ = med(k)
        print(f"  {lab:>24} {n_:>5} {m_:>+11.1f}")

    live, _ = med('live')
    z, _ = med('shift_0.00')
    half, _ = med('shift_0.50')
    yok, _ = med('yoked')
    ung, _ = med('ungated')
    print(f"\n  VALIDITY CHECK: live {live:+.2f} against replay at shift 0 {z:+.2f}, "
          f"difference {abs(live - z):.2f} pp")
    if abs(live - z) > 1.0:
        print("""  ** These must agree. Replay at shift 0 delivers the identical schedule on
  ** identical noise, so any difference is an implementation fault and the
  ** decomposition below cannot be read.""")
    else:
        print("  agree, so the replay path is faithful and the decomposition is readable.\n")
        u2, _ = med('ungated_2x')
        print(f"  DECOMPOSITION (competitor response, percentage points)")
        print(f"    intermittency   yoked minus ungated 1x         {yok - ung:+.1f}"
              f"   (dose-matched)")
        print(f"    phase           live minus shift 0.5 D         {live - half:+.1f}"
              f"   (identical schedule and dose)")
        print(f"    contingency     (live minus yoked) less phase  "
              f"{(live - yok) - (live - half):+.1f}   (dose-matched)")
        print(f"\n    for reference, ungated at 2x (double dose) gives {u2:+.1f}; the "
              f"difference\n    from yoked, {yok - u2:+.1f}, is confounded with dose and is "
              f"not the intermittency term.")
        tot = live - ung
        if abs(tot) > 1e-6:
            print(f"\n    shares of the total live-minus-ungated effect of {tot:+.1f} pp: "
                  f"intermittency {100*(yok-ung)/tot:.0f}%, "
                  f"phase {100*(live-half)/tot:.0f}%, "
                  f"contingency {100*((live-yok)-(live-half))/tot:.0f}%")
        print("""
  Section 4.5.2 attributes the majority of the effect to contingency. If the phase
  term is comparable to or larger than the residual contingency term, that
  attribution is wrong and the section should report this three-way split instead. A
  phase-shifted schedule that reproduces the live effect would be the strongest
  negative available: it would mean duty cycle and autocorrelation carry the effect,
  not alignment with the channel's own state.
""")
    save(args.out + 'wave22_M_phase.json',
         dict(fracs=[float(f) for f in FRACS],
              medians={k: med(k)[0] for k in KEYS},
              rows={k: [float(x) for x in v] for k, v in rows.items()}), args.force)
    return rows


# ==========================================================================
# BLOCK N - what is the intermittency term actually measuring?
# ==========================================================================
def block_N(args, elig=None):
    """Three controls that decide between two materially different papers.

    Block M found that a yoked schedule, gate-shaped in its temporal statistics but
    uncorrelated with the present trial's percept, already flips the competitor's
    response from -15.3% to +6.5%, and that the live gate adds only a further +9.9.
    Intermittency carries 69% of the effect. Three objections follow, and each has a
    control.

    (1) FAST CHOPPING: is it intermittency as such, or EPISODE-SCALE intermittency?
        The yoked schedule has bursts whose durations match dominance episodes. If the
        same amplitude, duty cycle and dose delivered as fast on/off chopping, at a
        timescale far below an episode, reproduces +6.5%, then the effect has nothing
        to do with the rivalry timescale and is a property of intermittent delivery
        alone. If fast chopping instead returns to the ungated value near -15%, the
        yoked schedule's contribution is episode-scale structure, which IS a timing
        property, and the paper's framing survives.

    (2) EPISODE-BOUNDARY DELAY: is the delay series a phase axis or a decorrelation
        axis? Block M delayed by fixed multiples of the BASELINE mean duration, but
        the live gate lengthens the attended channel's episodes by 61%, so a
        one-episode offset drifts and the series is monotone rather than periodic. The
        one-episode point even falls below the uncorrelated yoked condition, which a
        phase axis cannot do. Shifting instead by exactly k of the trial's OWN gate
        episodes restores phase while changing which episode is tracked. If k = 1
        recovers something near live, phase is real and separable from contingency. If
        it lands near yoked, the two are the same quantity and the decomposition
        should be reported as a two-way split.

    (3) FLOOR OCCUPANCY: is the rectifier account right? Section 4.5.2 explains the
        intermittency term by partial self-gating, a burst arriving while the attended
        channel sits near the floor being absorbed. That requires the channel to BE
        near the floor, and Section 4.1 reports median floor occupancy of 0.045 among
        switching configurations. If the intermittency term does not scale with floor
        occupancy across configurations, the mechanism is elsewhere and the
        explanation must be withdrawn.
    """
    print("\n=== BLOCK N: what the intermittency term measures ===")
    if elig is None:
        elig = _load_elig(args)
    pk = next((k for k in elig if abs(float(k) - 1.0) < 1e-9), sorted(elig, key=float)[0])
    pool = {c['idx']: c for c in load_pool(args.quick)}
    rng = np.random.default_rng(31)
    take = rng.choice(elig[pk]['eligible_idx'],
                      min(args.n_config, len(elig[pk]['eligible_idx'])), replace=False)
    # burst length in timesteps. The short end is far below the adaptation time
    # constant 1/gamma (20 to 50 on this grid); the long end approaches and exceeds
    # both it and a typical dominance episode, so the series brackets the threshold
    # the fast-chopping result implies rather than stopping short of it.
    CHOP = (2, 5, 10, 25, 50, 75, 100, 150, 200)
    KVALS = (1, 2, 3)              # delay in whole gate episodes
    rows = {k: [] for k in ('ungated', 'yoked', 'live', 'floor', 'dur', 'tau_a',
                            'shuffled', 'A_ungated', 'A_yoked', 'A_live', 'A_shuffled',
                            *[f'chop_{c}' for c in CHOP],
                            *[f'ep_{k}' for k in KVALS])}

    t0 = time.time()
    for j, ci in enumerate(take):
        if (j + 1) % PROGRESS_EVERY == 0:
            el = time.time() - t0
            print(f"    {j+1}/{len(take)}  [{el:.0f}s, "
                  f"~{el/(j+1)*(len(take)-j-1):.0f}s left]", flush=True)
        c = pool[int(ci)]
        bl = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps,
                      max(4, args.seeds // 2), BLK_N_BASE, 0, split_indet=SPLIT_INDET)
        ref = 0.5 * c['lam'] * sm(bl, O_ACTA)
        b = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps, args.seeds,
                     BLK_N_BASE, 1, split_indet=SPLIT_INDET)
        dB0, dA0, floor = sm(b, O_DURB), sm(b, O_DURA), sm(b, O_FLOOR)
        D = dA0                      # attended baseline, also the episode scale
        if not (np.isfinite(dB0) and np.isfinite(dA0) and D > 4
                and 0.005 <= ref <= 0.5):
            continue
        acc = {k: [] for k in rows}
        for s in range(args.seeds):
            lvl = 200 + s
            sch = np.zeros(args.steps, dtype=np.uint8)
            live = run_cell(c, 1.0, 0.5, 0.5, 2.0 * ref, G_ON, 0.0, args.steps, 1,
                            BLK_N_REC, lvl, schedule=sch, record=True,
                            split_indet=SPLIT_INDET)
            acc['live'].append(sm(live, O_DURB))
            acc['A_live'].append(sm(live, O_DURA))
            duty = float(sch.mean())

            # (1) fast chopping at the SAME duty cycle and amplitude
            for cl in CHOP:
                period = max(2, int(round(cl / max(duty, 1e-6))))
                idx = np.arange(args.steps)
                ch = ((idx % period) < cl).astype(np.uint8)
                r = run_cell(c, 1.0, 0.5, 0.5, 2.0 * ref, G_REPLAY, 0.0, args.steps,
                             1, BLK_N_REC, lvl, schedule=ch, split_indet=SPLIT_INDET)
                acc[f'chop_{cl}'].append(sm(r, O_DURB))

            # (2) delay by whole gate episodes of the schedule itself
            on = np.flatnonzero(np.diff(sch.astype(np.int8)) == 1) + 1
            for k in KVALS:
                if len(on) > k:
                    shift = int(on[k] - on[0])
                    sh = np.roll(sch, shift)
                    r = run_cell(c, 1.0, 0.5, 0.5, 2.0 * ref, G_REPLAY, 0.0,
                                 args.steps, 1, BLK_N_REC, lvl, schedule=sh,
                                 split_indet=SPLIT_INDET)
                    acc[f'ep_{k}'].append(sm(r, O_DURB))

            yk = np.zeros(args.steps, dtype=np.uint8)
            run_cell(c, 1.0, 0.5, 0.5, 2.0 * ref, G_ON, 0.0, args.steps, 1,
                     BLK_N_REC, 700 + s, schedule=yk, record=True,
                     split_indet=SPLIT_INDET)

            # (1b) duration-shuffled: the same burst and gap LENGTHS as the live
            # gate, reordered at random. Preserves the duration distribution,
            # including its variability, and destroys the ordering. A fixed-period
            # square wave has the right timescale but no variability; this has both
            # the timescale and the variability but no sequence.
            d0 = np.diff(np.concatenate(([0], sch.astype(np.int8), [0])))
            starts, ends = np.flatnonzero(d0 == 1), np.flatnonzero(d0 == -1)
            if len(starts) > 2 and len(ends) > 2:
                bl_ = (ends - starts)[:min(len(starts), len(ends))]
                gp_ = (starts[1:] - ends[:len(starts) - 1])
                gp_ = gp_[gp_ > 0]
                if len(bl_) > 1 and len(gp_) > 1:
                    rr = np.random.default_rng(1000 + s + int(ci))
                    bl_s = rr.permutation(bl_)
                    gp_s = rr.permutation(gp_)
                    sh2 = np.zeros(args.steps, dtype=np.uint8)
                    t_, i_ = 0, 0
                    while t_ < args.steps and i_ < len(bl_s):
                        b_ = int(bl_s[i_ % len(bl_s)])
                        sh2[t_:min(t_ + b_, args.steps)] = 1
                        t_ += b_ + int(gp_s[i_ % len(gp_s)])
                        i_ += 1
                    r = run_cell(c, 1.0, 0.5, 0.5, 2.0 * ref, G_REPLAY, 0.0,
                                 args.steps, 1, BLK_N_REC, lvl, schedule=sh2,
                                 split_indet=SPLIT_INDET)
                    acc['shuffled'].append(sm(r, O_DURB))
                    acc['A_shuffled'].append(sm(r, O_DURA))
            r = run_cell(c, 1.0, 0.5, 0.5, 2.0 * ref, G_REPLAY, 0.0, args.steps, 1,
                         BLK_N_REC, lvl, schedule=yk, split_indet=SPLIT_INDET)
            acc['yoked'].append(sm(r, O_DURB))
            acc['A_yoked'].append(sm(r, O_DURA))
            r = run_cell(c, 1.0, 0.5, 0.5, 1.0 * ref, G_UNGATED, 0.0, args.steps, 1,
                         BLK_N_REC, lvl, split_indet=SPLIT_INDET)
            acc['ungated'].append(sm(r, O_DURB))
            acc['A_ungated'].append(sm(r, O_DURA))
        for k in rows:
            if k in ('floor', 'dur', 'tau_a'):
                continue
            v = [x for x in acc[k] if np.isfinite(x)]
            base = dA0 if k.startswith('A_') else dB0
            rows[k].append(pct(np.mean(v), base) if v else np.nan)
        rows['floor'].append(floor)
        rows['dur'].append(D)
        rows['tau_a'].append(1.0 / c['gam'])

    def med(k):
        v = np.array([x for x in rows[k] if np.isfinite(x)], float)
        return (float(np.median(v)), v.size) if v.size else (float('nan'), 0)

    ung, _ = med('ungated'); yok, _ = med('yoked'); live, _ = med('live')
    print(f"\n  {'condition':>28} {'n':>5} {'competitor':>11}")
    for k, lab in (('ungated', 'ungated 1x (dose-matched)'), ('live', 'live gate'),
                   ('yoked', 'yoked (other seed)')):
        m_, n_ = med(k); print(f"  {lab:>28} {n_:>5} {m_:>+11.1f}")
    tau_a = float(np.nanmedian(rows['tau_a'])) if rows['tau_a'] else float('nan')
    dur_med = float(np.nanmedian(rows['dur'])) if rows['dur'] else float('nan')
    print(f"\n  (1) BURST DURATION SERIES, same duty cycle, amplitude and dose")
    print(f"      median adaptation constant 1/gamma = {tau_a:.0f} timesteps, "
          f"median episode = {dur_med:.0f}")
    print(f"  {'burst length':>28} {'n':>5} {'competitor':>11} {'vs 1/gamma':>11}")
    prev_v, prev_c, cross = None, None, None
    for cl in CHOP:
        m_, n_ = med(f'chop_{cl}')
        rel = cl / tau_a if np.isfinite(tau_a) and tau_a > 0 else float('nan')
        print(f"  {(str(cl) + ' steps'):>28} {n_:>5} {m_:>+11.1f} {rel:>11.2f}")
        if prev_v is not None and np.isfinite(m_) and np.isfinite(prev_v):
            if prev_v < 0 <= m_:
                cross = prev_c + (cl - prev_c) * (-prev_v) / (m_ - prev_v)
        prev_v, prev_c = m_, cl
    if cross is not None:
        print(f"\n      sign crossing at a burst length of about {cross:.0f} timesteps, "
              f"{cross/tau_a:.2f} x 1/gamma\n      and {cross/dur_med:.2f} x a mean episode")
    else:
        print("\n      no sign crossing within the swept range")
    sh_, shn = med('shuffled')
    print(f"\n  ATTENDED AND COMPETITOR TOGETHER, the comparison 4.5.2 needs")
    print(f"  {'condition':>28} {'attended':>10} {'competitor':>12}")
    for k, lab in (('ungated', 'continuous, dose-matched'), ('yoked', 'yoked'),
                   ('shuffled', 'duration-shuffled'), ('live', 'live gate')):
        a_, _ = med('A_' + k)
        c_, _ = med(k)
        print(f"  {lab:>28} {a_:>+10.1f} {c_:>+12.1f}")
    ay, _ = med('A_yoked'); al, _ = med('A_live')
    cy, _ = med('yoked'); cl, _ = med('live')
    print(f"""
      If yoked lengthens the attended channel about as much as the live gate does while
      producing a much smaller competitor response, attended lengthening is not
      sufficient and Section 4.5.2's adaptation-recovery mechanism is not the operative
      variable. If yoked lengthens it far less, the mechanism survives and the schedule
      term acts through attended duration as the account says.
      yoked attended {ay:+.1f} against live {al:+.1f}; competitor {cy:+.1f} against {cl:+.1f}.""")

    print(f"\n  (1b) DURATION-SHUFFLED, same burst and gap lengths, random order")
    print(f"  {'shuffled':>28} {shn:>5} {sh_:>+11.1f}")
    print(f"\n  (2) DELAY BY WHOLE GATE EPISODES")
    for k in KVALS:
        m_, n_ = med(f'ep_{k}')
        print(f"  {('shifted ' + str(k) + ' episode(s)'):>28} {n_:>5} {m_:>+11.1f}")

    chop_long, _ = med(f'chop_{CHOP[-1]}')
    chop_fast, _ = med(f'chop_{CHOP[0]}')
    ep1, _ = med('ep_1')
    print(f"""
  VERDICT ON (1). Chopping at {CHOP[0]} timesteps gives {chop_fast:+.1f} and at
  {CHOP[-1]} timesteps {chop_long:+.1f}, against continuous {ung:+.1f} and yoked
  {yok:+.1f}. The short end reproducing the continuous condition shows that
  fragmentation alone does nothing. Where the series crosses zero locates the
  timescale the modulation must exceed, and whether that crossing sits nearer
  1/gamma or nearer a mean episode says whether the constraint is set by adaptation
  or by the alternation itself. If it tracks 1/gamma the mechanism is the adaptation
  integration window, which is the reading Section 4.5.2 currently gives; if it tracks
  the episode, the constraint is the competition and the adaptation reading should be
  withdrawn in turn.

  VERDICT ON (2). Shifting by one whole gate episode gives {ep1:+.1f} against live
  {live:+.1f} and yoked {yok:+.1f}. Near live means phase is real and separable from
  contingency. Near yoked means the block M delay series was a decorrelation axis and
  the phase and contingency terms are one quantity, so the decomposition should be
  reported two ways: intermittency against everything contingent on the present trial.
""")

    # (3) does the intermittency term scale with floor occupancy?
    fo = np.array(rows['floor'], float)
    inter = np.array(rows['yoked'], float) - np.array(rows['ungated'], float)
    ok = np.isfinite(fo) & np.isfinite(inter)
    r_fo = _spear(fo[ok], inter[ok])
    print(f"  (3) RECTIFIER ACCOUNT: Spearman(floor occupancy, intermittency term) "
          f"= {r_fo:+.3f} on {int(ok.sum())} configurations")
    print(f"      median floor occupancy in this sample: {np.nanmedian(fo):.3f}")
    # the rectifier account predicts a POSITIVE correlation: more floor occupancy
    # means more of a suppressed-phase burst is absorbed. Testing |r| would treat a
    # strong negative correlation as confirmation, which is the wrong test.
    if r_fo < 0.2:
        print(f"""      The correlation is {r_fo:+.3f}. The rectifier account requires it to be
      positive and substantial. It is not, so partial
      self-gating by the rectifier cannot be carrying it. Section 4.5.2's mechanism
      must be withdrawn and replaced. The likely alternative is curvature of the
      effective transfer at the suppressed operating point, which is a Jensen
      argument comparing E[f(x + delta.1_gate)] with f(x + mean delta), and which is
      also the derivation Section 4.10 should point at.""")
    else:
        print("""      The term scales with floor occupancy, which supports the rectifier
      account. Report the correlation and stratify the decomposition by floor
      occupancy to show the mechanism operating.""")

    save(args.out + 'wave22_N_intermittency.json',
         dict(medians={k: med(k)[0] for k in rows if k not in ('floor', 'dur')},
              spearman_floor_intermittency=r_fo,
              median_floor=float(np.nanmedian(fo)),
              rows={k: [float(x) for x in v] for k, v in rows.items()}), args.force)
    return rows


# ==========================================================================
# BLOCK O - the convexity account of the schedule-statistics term
# ==========================================================================
def block_O(args, elig=None):
    """Does a Jensen argument explain the schedule-statistics term?

    THE ACCOUNT. Section 4.10 identifies the operative regime as noise-induced escape
    between basins rather than adaptation-driven oscillation. If dwell time is
    exponential in barrier height, as escape-rate arguments give, and the barrier moves
    with the delivered perturbation, then the attended channel's dwell is CONVEX in
    burst duration. By Jensen's inequality a schedule whose burst durations are drawn
    from a distribution produces a longer mean attended dwell than a fixed schedule at
    the same mean, since E[exp(cT)] > exp(c E[T]). A longer attended dwell gives the
    competitor more time to de-adapt, so it returns stronger and its episodes lengthen.
    Positive coupling, from variance in burst duration alone, at matched mean and dose.

    That single premise predicts every result in Section 4.5.2's table: a fixed period
    fails at every period because it has no variance; shuffled equals yoked because
    Jensen depends on the distribution and not the sequence; the effect scales with
    variance rather than mean; and the term shrinks with floor occupancy because a
    floored channel is not in the escape regime and has no barrier to modulate.

    THREE TESTS.

    (A) VARIANCE SWEEP AT FIXED MEAN. Burst durations drawn from gamma distributions
        with shape swept from very large, which approximates a fixed period, through
        intermediate values to below 1, all at the same mean and duty cycle. Jensen
        predicts the competitor response rises monotonically with variance and
        interpolates between the fixed-period and yoked rows already reported.

    (B) THE SINGLE-BURST RESPONSE FUNCTION. One isolated burst of duration T delivered
        into an otherwise unperturbed trial, phase-locked to the attended channel's
        dominance onset, measuring the change in that episode and in the competitor's
        next. Sweeping T gives f(T) empirically. Integrating f(T) over each schedule's
        own duration distribution then PREDICTS the yoked, shuffled and fixed-period
        medians without further fitting. If the predictions land, the mechanism is
        established and the decomposition collapses to one figure. If they do not, the
        effect involves interaction between successive bursts, which the
        shuffled-equals-yoked result sharply constrains.

    (C) FALSIFICATION. Convexity applies to the attended channel's own dwell in the
        first instance, so the attended gain should show the same variance sensitivity
        and more steeply than the competitor's. If it does not, the account is wrong.
    """
    print("\n=== BLOCK O: convexity account of the schedule-statistics term ===")
    if elig is None:
        elig = _load_elig(args)
    pk = next((k for k in elig if abs(float(k) - 1.0) < 1e-9), sorted(elig, key=float)[0])
    pool = {c['idx']: c for c in load_pool(args.quick)}
    rng = np.random.default_rng(32)
    take = rng.choice(elig[pk]['eligible_idx'],
                      min(args.n_config, len(elig[pk]['eligible_idx'])), replace=False)
    SHAPES = (50.0, 10.0, 3.8, 2.0, 1.0, 0.5)     # gamma shape: high = near-fixed
    TS = (5, 10, 20, 40, 60, 90, 130, 180)         # single-burst durations
    keys = ([f'k_{s}' for s in SHAPES] + [f'T_{t}' for t in TS]
            + [f'kA_{s}' for s in SHAPES] + ['ungated', 'yoked', 'live', 'dur', 'floor'])
    rows = {k: [] for k in keys}

    t0 = time.time()
    for j, ci in enumerate(take):
        if (j + 1) % PROGRESS_EVERY == 0:
            el = time.time() - t0
            print(f"    {j+1}/{len(take)}  [{el:.0f}s, "
                  f"~{el/(j+1)*(len(take)-j-1):.0f}s left]", flush=True)
        c = pool[int(ci)]
        bl = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps,
                      max(4, args.seeds // 2), BLK_O_BASE, 0, split_indet=SPLIT_INDET)
        ref = 0.5 * c['lam'] * sm(bl, O_ACTA)
        b = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps, args.seeds,
                     BLK_O_BASE, 1, split_indet=SPLIT_INDET)
        dA0, dB0, D, fo = sm(b, O_DURA), sm(b, O_DURB), sm(b, O_DURA), sm(b, O_FLOOR)
        if not (np.isfinite(dA0) and np.isfinite(dB0) and D > 4
                and 0.005 <= ref <= 0.5):
            continue
        acc = {k: [] for k in keys}
        for s in range(args.seeds):
            lvl = 300 + s
            sch = np.zeros(args.steps, dtype=np.uint8)
            live = run_cell(c, 1.0, 0.5, 0.5, 2.0 * ref, G_ON, 0.0, args.steps, 1,
                            BLK_O_REC, lvl, schedule=sch, record=True,
                            split_indet=SPLIT_INDET)
            acc['live'].append(sm(live, O_DURB))
            duty = max(float(sch.mean()), 1e-6)
            mean_burst = D                      # match the live gate's mean burst

            # (A) gamma-distributed burst durations at fixed mean and duty cycle
            rr = np.random.default_rng(5000 + s + int(ci))
            for shp in SHAPES:
                sc = mean_burst / shp
                g = np.zeros(args.steps, dtype=np.uint8)
                t_ = 0
                while t_ < args.steps:
                    bdur = max(1, int(round(rr.gamma(shp, sc))))
                    gap = max(1, int(round(bdur * (1.0 - duty) / duty)))
                    g[t_:min(t_ + bdur, args.steps)] = 1
                    t_ += bdur + gap
                r = run_cell(c, 1.0, 0.5, 0.5, 2.0 * ref, G_REPLAY, 0.0, args.steps,
                             1, BLK_O_REC, lvl, schedule=g, split_indet=SPLIT_INDET)
                acc[f'k_{shp}'].append(sm(r, O_DURB))
                acc[f'kA_{shp}'].append(sm(r, O_DURA))

            # (B) single isolated burst of duration T, one per run
            for T in TS:
                g = np.zeros(args.steps, dtype=np.uint8)
                g[BURN + 200: BURN + 200 + T] = 1
                r = run_cell(c, 1.0, 0.5, 0.5, 2.0 * ref, G_REPLAY, 0.0, args.steps,
                             1, BLK_O_REC, lvl, schedule=g, split_indet=SPLIT_INDET)
                acc[f'T_{T}'].append(sm(r, O_DURB))

            yk = np.zeros(args.steps, dtype=np.uint8)
            run_cell(c, 1.0, 0.5, 0.5, 2.0 * ref, G_ON, 0.0, args.steps, 1,
                     BLK_O_REC, 800 + s, schedule=yk, record=True,
                     split_indet=SPLIT_INDET)
            r = run_cell(c, 1.0, 0.5, 0.5, 2.0 * ref, G_REPLAY, 0.0, args.steps, 1,
                         BLK_O_REC, lvl, schedule=yk, split_indet=SPLIT_INDET)
            acc['yoked'].append(sm(r, O_DURB))
            r = run_cell(c, 1.0, 0.5, 0.5, 1.0 * ref, G_UNGATED, 0.0, args.steps, 1,
                         BLK_O_REC, lvl, split_indet=SPLIT_INDET)
            acc['ungated'].append(sm(r, O_DURB))
        for k in keys:
            if k in ('dur', 'floor'):
                continue
            v = [x for x in acc[k] if np.isfinite(x)]
            base = dA0 if k.startswith('kA_') else dB0
            rows[k].append(pct(np.mean(v), base) if v else np.nan)
        rows['dur'].append(D)
        rows['floor'].append(fo)

    def med(k):
        v = np.array([x for x in rows[k] if np.isfinite(x)], float)
        return (float(np.median(v)), v.size) if v.size else (float('nan'), 0)

    ung, _ = med('ungated'); yok, _ = med('yoked'); live, _ = med('live')
    Dm = float(np.nanmedian(rows['dur']))
    print(f"\n  reference conditions: continuous {ung:+.1f}, yoked {yok:+.1f}, "
          f"live {live:+.1f}; mean episode {Dm:.0f}")

    print(f"\n  (A) VARIANCE SWEEP at fixed mean burst {Dm:.0f} and matched duty cycle")
    print(f"  {'gamma shape':>13} {'CV of burst':>12} {'n':>5} {'competitor':>11} {'attended':>10}")
    cvs, comps = [], []
    for shp in SHAPES:
        m_, n_ = med(f'k_{shp}'); a_, _ = med(f'kA_{shp}')
        cv = 1.0 / math.sqrt(shp)
        cvs.append(cv); comps.append(m_)
        print(f"  {shp:>13.1f} {cv:>12.2f} {n_:>5} {m_:>+11.1f} {a_:>+10.1f}")
    rho_var = _spear(cvs, comps)
    print(f"\n      Spearman(CV of burst duration, competitor response) = {rho_var:+.3f}")
    if rho_var > 0.7:
        print("""      Monotone in variance at fixed mean, which is the central prediction of the
      convexity account and is not predicted by any timescale or intermittency
      story.""")
    else:
        print("""      NOT monotone in variance. The convexity account predicts it should be, so
      this is the result that refutes it.""")

    print(f"\n  (B) SINGLE-BURST RESPONSE f(T), one isolated burst per run")
    print(f"  {'T':>6} {'T/episode':>11} {'n':>5} {'competitor':>11}")
    fT = {}
    for T in TS:
        m_, n_ = med(f'T_{T}'); fT[T] = m_
        print(f"  {T:>6} {T/Dm:>11.2f} {n_:>5} {m_:>+11.1f}")
    xs = np.array(TS, float); ys = np.array([fT[T] for T in TS], float)
    good = np.isfinite(ys)
    if good.sum() >= 4:
        # curvature: quadratic fit, positive second-order term means convex
        cfit = np.polyfit(xs[good], ys[good], 2)
        print(f"\n      quadratic fit: curvature coefficient {cfit[0]:+.5f}")
        print(f"      {'CONVEX' if cfit[0] > 0 else 'CONCAVE'} in burst duration"
              f" -- Jensen requires convex")
        # predict each schedule by integrating f(T) over its duration distribution
        def predict(shape):
            rr2 = np.random.default_rng(7)
            draws = rr2.gamma(shape, Dm / shape, 20000)
            return float(np.mean(np.polyval(cfit, np.clip(draws, 1, xs.max()))))
        print(f"\n      predicted from f(T) alone, no further fitting:")
        print(f"  {'gamma shape':>13} {'predicted':>11} {'observed':>10}")
        for shp in SHAPES:
            print(f"  {shp:>13.1f} {predict(shp):>+11.1f} {med(f'k_{shp}')[0]:>+10.1f}")
        print("""
      If predicted tracks observed, the schedule-statistics term is fully accounted
      for by the single-burst response and its curvature, and Section 4.5.2 can
      replace three withdrawn mechanisms with one derivation. Systematic
      underprediction means successive bursts interact, which the
      shuffled-equals-yoked result constrains to be order-independent.""")

    print(f"\n  (C) FALSIFICATION: attended-channel variance sensitivity")
    aA = [med(f'kA_{s}')[0] for s in SHAPES]
    rho_a = _spear(cvs, aA)
    print(f"      Spearman(CV, attended gain) = {rho_a:+.3f} against "
          f"{rho_var:+.3f} for the competitor")
    print("""      Convexity acts on the attended channel's dwell first, so its variance
      sensitivity should be at least as strong. A weaker or absent attended effect
      with a strong competitor effect would falsify the account.""")

    save(args.out + 'wave22_O_convexity.json',
         dict(shapes=list(SHAPES), Ts=list(TS), mean_episode=Dm,
              medians={k: med(k)[0] for k in keys if k not in ('dur', 'floor')},
              spearman_cv_competitor=rho_var, spearman_cv_attended=rho_a,
              rows={k: [float(x) for x in v] for k, v in rows.items()}), args.force)
    return rows


# ==========================================================================
# BLOCK P - the paired kernel comparison --verify has always promised
# ==========================================================================
def block_P(args, elig=None):
    """Call wave2_campaign.run_trace and this file's _trace on identical parameters
    and identical seeds, and compare the traces themselves.

    WHY THIS EXISTS. Block V compares this kernel's coefficient of variation against
    phase1_cv read from JSON. That is a comparison against a stored summary statistic,
    which cannot separate a kernel difference from a seed difference from a threshold
    difference, which is why block V ends up sweeping theta to find the value that
    minimises disagreement. This project's most expensive discovery was a one-line
    update-order divergence between the published Equation 2 and the executed code,
    found only because an independent reimplementation disagreed on a second moment
    while agreeing on a first. A paired same-seed trace comparison would have caught it
    in one run instead of six rounds.

    The obstacle is that run_trace's signature is not documented here. This block tries
    a series of plausible calling conventions, reports which one worked, and prints the
    shape and type of what came back so the comparison can be completed by hand if none
    of them fits. It never guesses silently.
    """
    print("\n=== BLOCK P: paired kernel comparison against wave2_campaign ===")
    import inspect
    rt = getattr(w2, 'run_trace', None)
    if rt is None:
        print("  wave2_campaign has no run_trace; nothing to compare against")
        return {}
    try:
        sig = inspect.signature(rt)
        print(f"  run_trace{sig}")
        params = list(sig.parameters)
    except (TypeError, ValueError):
        print("  run_trace signature unavailable (compiled?)")
        params = []

    with open('phase2_dissociation_results.json') as f:
        recs = as_list(json.load(f), 'config_results')
    c = params_from(recs[0], 0)
    n_steps, seed = args.steps, 12345
    print(f"\n  configuration: " + ", ".join(f"{k}={c[k]:.4g}" for k in
          ('lam', 'beta', 'alpha', 'sigma', 'gam', 'kap')))
    print(f"  {n_steps} steps, seed {seed}\n")

    # candidate calling conventions, in decreasing order of likelihood
    kw = dict(lam=c['lam'], beta=c['beta'], alpha=c['alpha'], sigma=c['sigma'],
              gam=c['gam'], kap=c['kap'])
    alt = dict(lambda_=c['lam'], gamma=c['gam'], kappa=c['kap'], gamma_adapt=c['gam'],
               lam_leak=c['lam'])
    attempts = [
        # recovered from the signature: 18 positional arguments, with separate
        # per-channel goal signals g_a and g_b and separate boosts, k_sharp <= 0
        # selecting the hard rectifier
        ('full 18-argument positional',
         lambda: rt(c['lam'], c['beta'], c['alpha'], c['sigma'], c['gam'], c['kap'],
                    0.5, 0.5, 0.0, 0.0, 0.0, 0.0, n_steps, seed, XMAX, 0.0, 0.0, 0.0)),
        ('positional 6 + steps + seed',
         lambda: rt(c['lam'], c['beta'], c['alpha'], c['sigma'], c['gam'], c['kap'],
                    0.5, 0.5, n_steps, seed)),
        ('positional 6 + inputs + steps',
         lambda: rt(c['lam'], c['beta'], c['alpha'], c['sigma'], c['gam'], c['kap'],
                    0.5, 0.5, n_steps)),
        ('dict of params',
         lambda: rt(dict(kw, n_steps=n_steps, seed=seed))),
        ('kwargs',
         lambda: rt(**dict(kw, S_A=0.5, S_B=0.5, n_steps=n_steps, seed=seed))),
        ('kwargs, alternative names',
         lambda: rt(**dict(alt, beta=c['beta'], alpha=c['alpha'], sigma=c['sigma'],
                           n_steps=n_steps, seed=seed))),
    ]
    theirs, how = None, None
    for label, fn in attempts:
        try:
            theirs = fn()
            how = label
            break
        except Exception as e:
            print(f"    {label:<32} {type(e).__name__}: {str(e)[:60]}")
    if theirs is None:
        print(f"""
  None of the calling conventions fits. Its parameters are {params}.
  Add the correct call to `attempts` above; the comparison below then runs
  unchanged. Do not proceed to interpret block V's theta sweep as a kernel check
  until this passes, because that sweep cannot distinguish a kernel difference from
  a threshold difference.""")
        return {}

    print(f"\n  called successfully as: {how}")
    arrs = theirs if isinstance(theirs, (tuple, list)) else (theirs,)
    arrs = [np.asarray(a, dtype=np.float64) for a in arrs if hasattr(a, '__len__')]
    print(f"  returned {len(arrs)} array(s), shapes {[a.shape for a in arrs]}")
    if len(arrs) < 2 or arrs[0].size != n_steps:
        print("  expected two traces of length n_steps; cannot compare element-wise")
        return {}

    ta = np.zeros(n_steps)
    tb = np.zeros(n_steps)
    _trace_full(c['lam'], c['beta'], c['alpha'], c['sigma'], c['gam'], c['kap'],
                0.5, 0.5, n_steps, seed, ADAPT_PIPELINE, ta, tb)

    out = dict(called_as=how, n_steps=n_steps, seed=seed)
    print(f"\n  ELEMENT-WISE, same seed, no manipulation")
    print(f"  {'':>10} {'theirs':>26} {'ours':>26}")
    for lab, th, ou in (('channel A', arrs[0], ta), ('channel B', arrs[1], tb)):
        print(f"  {lab:>10} mean {th[BURN:].mean():>8.5f} sd {th[BURN:].std():>7.5f}"
              f"   mean {ou[BURN:].mean():>8.5f} sd {ou[BURN:].std():>7.5f}")
    for lab, th, ou in (('A', arrs[0], ta), ('B', arrs[1], tb)):
        d = np.abs(th - ou)
        first = int(np.argmax(d > 1e-9)) if (d > 1e-9).any() else -1
        out[f'max_abs_diff_{lab}'] = float(d.max())
        out[f'first_divergence_{lab}'] = first
        print(f"\n  channel {lab}: max |difference| {d.max():.3e}, "
              f"first index above 1e-9: {first if first >= 0 else 'none'}")
        if 0 <= first < 12:
            print(f"    first 12 timesteps")
            print(f"    {'t':>3} {'theirs':>12} {'ours':>12} {'diff':>12}")
            for t in range(12):
                print(f"    {t:>3} {th[t]:>12.6f} {ou[t]:>12.6f} {th[t]-ou[t]:>12.3e}")

    agree = max(out['max_abs_diff_A'], out['max_abs_diff_B']) < 1e-6
    out['agree'] = bool(agree)
    if agree:
        print("""
  THE KERNELS AGREE to within 1e-6 at every timestep on a shared seed. The noise
  streams match, so this is a genuine paired comparison and not a distributional
  one, and it establishes that the update rules are identical. This is the check
  block V was described as performing and never did.""")
    else:
        print("""
  THE KERNELS DIVERGE. Read the first-divergence index above. Divergence at t = 1
  means the update itself differs; divergence later with agreement early means the
  noise streams differ, which points at the seeding rather than the model; and a
  difference that grows smoothly from a small value points at the adaptation order,
  which is the defect this project spent six rounds finding.

  Try ADAPT_PIPELINE = 0 and rerun. If that agrees, the constant is set wrong for
  this comparison.""")
    save(args.out + 'wave22_P_paired.json', out, True)
    return out


# ==========================================================================
# BLOCK Q - the three runs the CM review asks for that fit this kernel
# ==========================================================================
@njit(cache=True)
def _trace_burst(lam, beta, alpha, sigma, gam, kap, S_A, S_B, inc, T, every,
                 n_steps, seed, theta, ta, tb, flag):
    """Unmanipulated dynamics except for a burst of length T on channel A, triggered
    at the onset of every `every`-th A-dominance episode. Writes both traces and a
    per-timestep flag marking when a burst is being delivered. Update order matches
    the pipeline: activation first, adaptation from the new activation."""
    np.random.seed(seed)
    xA = X_INIT
    xB = X_INIT
    aA = 0.0
    aB = 0.0
    rec = 1.0 - lam
    prevA = 0
    nA_on = 0
    left = 0
    for t in range(n_steps):
        d = xA - xB
        isA = 1 if d > theta else 0
        if isA == 1 and prevA == 0:
            nA_on += 1
            if nA_on % every == 0:
                left = T
        prevA = isA
        add = 0.0
        if left > 0:
            add = inc
            flag[t] = 1
            left -= 1
        ta[t] = xA
        tb[t] = xB
        eA = np.random.normal(0.0, sigma)
        eB = np.random.normal(0.0, sigma)
        newA = rec * xA + S_A + add - beta * xB - alpha * aA + eA
        newB = rec * xB + S_B - beta * xA - alpha * aB + eB
        if newA < 0.0:
            newA = 0.0
        elif newA > XMAX:
            newA = XMAX
        if newB < 0.0:
            newB = 0.0
        elif newB > XMAX:
            newB = XMAX
        xA = newA
        xB = newB
        aA = (1.0 - gam) * aA + kap * xA
        aB = (1.0 - gam) * aB + kap * xB


def _episodes(ta, tb, flag, theta):
    """Episodes after burn-in as (channel, start, length, burst_inside)."""
    d = ta[BURN:] - tb[BURN:]
    st = np.where(d > theta, 1, np.where(d < -theta, -1, 0))
    fl = flag[BURN:]
    eps, cur, start = [], 0, 0
    for t, s in enumerate(st):
        if s != 0 and s != cur:
            if cur != 0:
                eps.append((cur, start, t - start, int(fl[start:t].any())))
            cur, start = s, t
    return eps[1:]                           # drop the first, as the pipeline does


def block_Q(args, elig=None):
    """(M1a) single-burst response phase-locked to dominance onset;
    (M1b) gamma schedules with gaps drawn independently of bursts;
    (M4) the gated-versus-ungated contrast on an unfiltered sample."""
    print("\n=== BLOCK Q: CM review, M1a, M1b and M4 ===")
    if elig is None:
        elig = _load_elig(args)
    pk = next((k for k in elig if abs(float(k) - 1.0) < 1e-9), sorted(elig, key=float)[0])
    pool = {c['idx']: c for c in load_pool(args.quick)}
    rng = np.random.default_rng(33)
    take = rng.choice(elig[pk]['eligible_idx'],
                      min(args.n_config, len(elig[pk]['eligible_idx'])), replace=False)
    out = {}

    # ---------------- M1a ----------------
    print("\n  (M1a) single burst at A-dominance onset, every 4th episode, paired "
          "within run")
    TS = (5, 10, 20, 40, 80)
    resA = {T: {'own': [], 'next': []} for T in TS}
    for ci in take:
        c = pool[int(ci)]
        bl = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps,
                      max(4, args.seeds // 2), BLK_Q_BASE, 0, split_indet=SPLIT_INDET)
        ref = 0.5 * c['lam'] * sm(bl, O_ACTA)
        if not (0.005 <= ref <= 0.5):
            continue
        for T in TS:
            own_d, next_d = [], []
            for s in range(args.seeds):
                ta = np.zeros(args.steps); tb = np.zeros(args.steps)
                fl = np.zeros(args.steps, dtype=np.uint8)
                seed = canonical_seed(BLK_Q_BURST, int(ci), 0, LEVEL(T), s)
                _trace_burst(c['lam'], c['beta'], c['alpha'], c['sigma'], c['gam'],
                             c['kap'], 0.5, 0.5, 2.0 * ref, T, 4, args.steps, seed,
                             THETA, ta, tb, fl)
                eps = _episodes(ta, tb, fl, THETA)
                a_b = [e[2] for e in eps if e[0] == 1 and e[3]]
                a_n = [e[2] for e in eps if e[0] == 1 and not e[3]]
                nb_b, nb_n = [], []
                for k in range(len(eps) - 1):
                    if eps[k][0] == 1 and eps[k + 1][0] == -1:
                        (nb_b if eps[k][3] else nb_n).append(eps[k + 1][2])
                if a_b and a_n:
                    own_d.append(100 * (np.mean(a_b) - np.mean(a_n)) / np.mean(a_n))
                if nb_b and nb_n:
                    next_d.append(100 * (np.mean(nb_b) - np.mean(nb_n)) / np.mean(nb_n))
            if own_d:
                resA[T]['own'].append(float(np.mean(own_d)))
            if next_d:
                resA[T]['next'].append(float(np.mean(next_d)))
    print(f"  {'burst T':>8} {'n':>5} {'containing A episode':>21} {'following B episode':>21}")
    for T in TS:
        o, nx = np.array(resA[T]['own']), np.array(resA[T]['next'])
        print(f"  {T:>8} {o.size:>5} {np.median(o) if o.size else np.nan:>+20.1f}% "
              f"{np.median(nx) if nx.size else np.nan:>+20.1f}%")
    out['M1a'] = {str(T): {k: list(v) for k, v in resA[T].items()} for T in TS}
    print("""
      Read the following-B column against T. If it rises with T and is convex, the
      single-burst response has the curvature the withdrawn convexity account needed
      and the earlier flat result was the dilution artefact Section 4.5.3 names. If it
      is flat here too, the schedule-statistics term needs interaction between bursts.""")

    # ---------------- M1b ----------------
    print("\n  (M1b) gamma bursts with independently drawn gaps, against yoked")
    SH = (50.0, 3.8, 1.0)
    resB = {'yoked': [], 'ungated': [], **{f'k{s}': [] for s in SH}}
    for ci in take:
        c = pool[int(ci)]
        bl = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps,
                      max(4, args.seeds // 2), BLK_Q_BASE, 1, split_indet=SPLIT_INDET)
        ref = 0.5 * c['lam'] * sm(bl, O_ACTA)
        b = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps, args.seeds,
                     BLK_Q_BASE, 2, split_indet=SPLIT_INDET)
        dB0, D = sm(b, O_DURB), sm(b, O_DURA)
        if not (np.isfinite(dB0) and D > 4 and 0.005 <= ref <= 0.5):
            continue
        acc = {k: [] for k in resB}
        for s in range(args.seeds):
            lvl = LEVEL('Q', s)
            sch = np.zeros(args.steps, dtype=np.uint8)
            run_cell(c, 1.0, 0.5, 0.5, 2.0 * ref, G_ON, 0.0, args.steps, 1,
                     BLK_Q_REC, lvl, schedule=sch, record=True,
                     split_indet=SPLIT_INDET)
            duty = max(float(sch.mean()), 1e-3)
            gap_mean = D * (1.0 - duty) / duty
            yk = np.zeros(args.steps, dtype=np.uint8)
            run_cell(c, 1.0, 0.5, 0.5, 2.0 * ref, G_ON, 0.0, args.steps, 1,
                     BLK_Q_REC, LEVEL('Qy', s), schedule=yk, record=True,
                     split_indet=SPLIT_INDET)
            r = run_cell(c, 1.0, 0.5, 0.5, 2.0 * ref, G_REPLAY, 0.0, args.steps, 1,
                         BLK_Q_REC, lvl, schedule=yk, split_indet=SPLIT_INDET)
            acc['yoked'].append(sm(r, O_DURB))
            r = run_cell(c, 1.0, 0.5, 0.5, 1.0 * ref, G_UNGATED, 0.0, args.steps, 1,
                         BLK_Q_REC, lvl, split_indet=SPLIT_INDET)
            acc['ungated'].append(sm(r, O_DURB))
            rr = np.random.default_rng(9000 + s + int(ci))
            for shp in SH:
                g = np.zeros(args.steps, dtype=np.uint8)
                t_ = 0
                while t_ < args.steps:
                    bd = max(1, int(round(rr.gamma(shp, D / shp))))
                    gp = max(1, int(round(rr.gamma(shp, gap_mean / shp))))
                    g[t_:min(t_ + bd, args.steps)] = 1
                    t_ += bd + gp
                r = run_cell(c, 1.0, 0.5, 0.5, 2.0 * ref, G_REPLAY, 0.0, args.steps,
                             1, BLK_Q_REC, lvl, schedule=g, split_indet=SPLIT_INDET)
                acc[f'k{shp}'].append(sm(r, O_DURB))
        for k in resB:
            v = [x for x in acc[k] if np.isfinite(x)]
            resB[k].append(pct(np.mean(v), dB0) if v else np.nan)
    print(f"  {'schedule':>28} {'n':>5} {'competitor':>11}")
    for k, lab in (('ungated', 'continuous, dose-matched'), ('yoked', 'yoked'),
                   *[(f'k{s}', f'gamma shape {s}, indep. gaps') for s in SH]):
        v = np.array([x for x in resB[k] if np.isfinite(x)])
        print(f"  {lab:>28} {v.size:>5} {np.median(v) if v.size else np.nan:>+11.1f}")
    out['M1b'] = {k: [float(x) for x in v] for k, v in resB.items()}
    print("""
      If independent gaps bring the gamma schedules up to the yoked value, the burst-gap
      coupling in the earlier synthetic schedules was the missing ingredient and the
      schedule-statistics term is a property of the joint distribution. If they stay
      near the tied-gap values, it is not, and the term remains unexplained.""")

    # ---------------- M4 ----------------
    print("\n  (M4) gated vs ungated on an UNFILTERED rivalry-producing sample")
    riv = [c['idx'] for c in load_pool(args.quick) if c['rivalry']]
    unf = rng.choice(riv, min(args.n_config, len(riv)), replace=False)
    g_pos = g_n = u_pos = u_n = 0
    gv, uv = [], []
    for ci in unf:
        c = pool[int(ci)]
        bl = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps,
                      max(4, args.seeds // 2), BLK_Q_BASE, 3, split_indet=SPLIT_INDET)
        ref = 0.5 * c['lam'] * sm(bl, O_ACTA)
        b = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps, args.seeds,
                     BLK_Q_BASE, 4, split_indet=SPLIT_INDET)
        dB0 = sm(b, O_DURB)
        if not (np.isfinite(dB0) and sm(b, O_SW) >= 10 and 0.005 <= ref <= 0.5):
            continue
        g = run_cell(c, 1.0, 0.5, 0.5, ref, G_ON, 0.0, args.steps, args.seeds,
                     BLK_Q_UNF, 1, split_indet=SPLIT_INDET)
        u = run_cell(c, 1.0, 0.5, 0.5, ref, G_UNGATED, 0.0, args.steps, args.seeds,
                     BLK_Q_UNF, 2, split_indet=SPLIT_INDET)
        pg, pu = pct(sm(g, O_DURB), dB0), pct(sm(u, O_DURB), dB0)
        if np.isfinite(pg):
            g_n += 1; g_pos += int(pg > 0); gv.append(pg)
        if np.isfinite(pu):
            u_n += 1; u_pos += int(pu > 0); uv.append(pu)
    gl, gh = wilson(g_pos, g_n) if g_n else (np.nan, np.nan)
    ul, uh = wilson(u_pos, u_n) if u_n else (np.nan, np.nan)
    print(f"    gated 1x   competitor > 0 in {g_pos}/{g_n} [{100*gl:.0f}%, {100*gh:.0f}%], "
          f"median {np.median(gv) if gv else np.nan:+.1f}%")
    print(f"    ungated 1x competitor > 0 in {u_pos}/{u_n} [{100*ul:.0f}%, {100*uh:.0f}%], "
          f"median {np.median(uv) if uv else np.nan:+.1f}%")
    print("""
      This is the number Section 6 needs. Intervals on opposite sides of chance mean the
      central result holds without the eligibility filter; a gated proportion near 61%
      means it behaves like the goal signal and weakens outside it.""")
    out['M4'] = dict(gated=[g_pos, g_n], ungated=[u_pos, u_n], gated_vals=gv, ungated_vals=uv)
    save(args.out + 'wave22_Q_cmreview.json', out, args.force)
    return out


# ==========================================================================
# BLOCK R - the threshold-geometry account of the coupling sign
# ==========================================================================
def _geometry(c, n_steps, seed):
    """Switching geometry of one unmanipulated trace.

    Reduces the dynamics to the adaptation imbalance u = a_A - a_B. During B's
    dominance u relaxes towards -L, where L = kappa (x_B - x_A) / gamma with the
    activations averaged over B-dominant timesteps. Switches happen at |u| = c,
    measured as the median imbalance at each handover. A B-episode then lasts
    T = ln((L + c) / (L - c)) / gamma, and rho = (L - c) / (L + c).
    Adaptation is reconstructed from the activation trace in the pipeline's
    update order, a(t+1) = (1 - gamma) a(t) + kappa x(t+1)."""
    ta = np.zeros(n_steps); tb = np.zeros(n_steps)
    _trace_full(c['lam'], c['beta'], c['alpha'], c['sigma'], c['gam'], c['kap'],
                0.5, 0.5, n_steps, seed, ADAPT_PIPELINE, ta, tb)
    g, k = c['gam'], c['kap']
    aA = np.zeros(n_steps); aB = np.zeros(n_steps)
    for t in range(n_steps - 1):
        aA[t + 1] = (1 - g) * aA[t] + k * ta[t + 1]
        aB[t + 1] = (1 - g) * aB[t] + k * tb[t + 1]
    s0 = BURN
    d = ta[s0:] - tb[s0:]
    u = (aA - aB)[s0:]
    st = np.where(d > THETA, 1, np.where(d < -THETA, -1, 0))
    last, c_ab, c_ba = 0, [], []
    for t, s in enumerate(st):
        if s == 0:
            continue
        if last == 1 and s == -1:
            c_ab.append(u[t])
        if last == -1 and s == 1:
            c_ba.append(-u[t])
        last = s
    if len(c_ab) < 5 or len(c_ba) < 5:
        return None
    cc = 0.5 * (np.median(c_ab) + np.median(c_ba))
    Bdom = st == -1
    if Bdom.sum() < 50:
        return None
    L = k * (np.mean(tb[s0:][Bdom]) - np.mean(ta[s0:][Bdom])) / g
    if not (L > cc > 0):
        return None
    return dict(c=float(cc), L=float(L), rho=float((L - cc) / (L + cc)),
                T_pred=float(np.log((L + cc) / (L - cc)) / g))


def block_R(args, elig=None):
    """Does switching-threshold geometry explain the coupling sign?

    THE ACCOUNT. Treat the adaptation imbalance u as the one slow variable. An
    episode is the time for u to travel between two switching thresholds while
    relaxing towards an asymptote. An increment on channel A displaces thresholds:

      gated     displaces only the A-to-B handover threshold, so B's episode starts
                further from its own end point and lengthens;
      ungated   displaces both thresholds equally, but B's episode ends near its
                asymptote where the logarithm is steep, so the end shift dominates
                and B shortens; to first order the two channels trade off
                symmetrically, which is why Levelt II is unreachable;
      anti      displaces only the threshold that ends B's episode, so B shortens
                by more than A does.

    To first order the gated competitor-to-attended ratio is rho = (L-c)/(L+c), the
    ungated ratio is -1, and the anti-gated ratio is 1/rho.

    WHAT THIS BLOCK TESTS, in order of how much the paper will lean on it:
      (1) whether (c, L) measured from an unmanipulated trace predicts episode
          duration, i.e. whether the reduction describes the dynamics at all;
      (2) the three signs;
      (3) the first-order magnitude predictions, which a sandbox check on random
          configurations found weak for the gated ratio and absent for the
          anti-gated ratio. They are reported whatever they show.
    """
    print("\n=== BLOCK R: threshold geometry of the coupling sign ===")
    if elig is None:
        elig = _load_elig(args)
    pk = next((k for k in elig if abs(float(k) - 1.0) < 1e-9), sorted(elig, key=float)[0])
    pool = {c['idx']: c for c in load_pool(args.quick)}
    rng = np.random.default_rng(34)
    take = rng.choice(elig[pk]['eligible_idx'],
                      min(args.n_config, len(elig[pk]['eligible_idx'])), replace=False)
    AMPS = (0.25, 0.50)                       # fraction of lambda * baseline activation
    rows = []
    t0 = time.time()
    for j, ci in enumerate(take):
        if (j + 1) % PROGRESS_EVERY == 0:
            el = time.time() - t0
            print(f"    {j+1}/{len(take)}  [{el:.0f}s, ~{el/(j+1)*(len(take)-j-1):.0f}s left]",
                  flush=True)
        c = pool[int(ci)]
        geo = _geometry(c, 2 * args.steps, canonical_seed(BLK_R_GEO, int(ci), 0, 0, 0))
        if geo is None:
            continue
        b = run_cell(c, 1.0, 0.5, 0.5, 0.0, G_NONE, 0.0, args.steps, args.seeds,
                     BLK_R_BASE, 0, split_indet=SPLIT_INDET)
        dA0, dB0 = sm(b, O_DURA), sm(b, O_DURB)
        if not (np.isfinite(dA0) and np.isfinite(dB0) and sm(b, O_SW) >= 20):
            continue
        rec = dict(config=int(ci), **geo, T_obs=float(0.5 * (dA0 + dB0)))
        for a in AMPS:
            inc = a * c['lam'] * sm(b, O_ACTA)
            for lab, gate in (('gated', G_ON), ('ungated', G_UNGATED), ('anti', G_OFF)):
                o = run_cell(c, 1.0, 0.5, 0.5, inc, gate, 0.0, args.steps, args.seeds,
                             BLK_R_COND, LEVEL(lab, a), split_indet=SPLIT_INDET)
                pA, pB = pct(sm(o, O_DURA), dA0), pct(sm(o, O_DURB), dB0)
                rec[f'{lab}_{a}'] = (pB / pA) if (np.isfinite(pA) and np.isfinite(pB)
                                                 and abs(pA) > 1e-6) else np.nan
                rec[f'{lab}_{a}_pB'] = pB
        rows.append(rec)

    R = {k: np.array([r[k] for r in rows], float) for k in rows[0]} if rows else {}
    n = len(rows)
    print(f"\n  {n} configurations with a measurable switching geometry")
    if n < 10:
        print("  too few to test")
        return rows
    rs = _spear(R['T_pred'], R['T_obs'])
    print(f"\n  (1) DOES THE REDUCTION DESCRIBE THE DYNAMICS")
    print(f"      Spearman(predicted, observed episode duration) = {rs:+.3f}")
    print(f"      median predicted / observed = {np.nanmedian(R['T_pred'] / R['T_obs']):.2f}")
    print(f"      median rho = {np.median(R['rho']):.3f}, IQR "
          f"[{np.percentile(R['rho'], 25):.3f}, {np.percentile(R['rho'], 75):.3f}]")
    out = dict(n=n, spearman_duration=rs)
    for a in AMPS:
        print(f"\n  (2) SIGNS, increment {a} x lambda x baseline activation")
        for lab, want in (('gated', '+'), ('ungated', '-'), ('anti', '-')):
            v = R[f'{lab}_{a}_pB']; v = v[np.isfinite(v)]
            pos = int((v > 0).sum())
            lo, hi = wilson(pos, v.size)
            print(f"      {lab:<8} competitor lengthens in {pos:>4}/{v.size:<4} "
                  f"[{100*lo:.0f}%, {100*hi:.0f}%]   theory: {want}")
            out[f'sign_{lab}_{a}'] = [pos, int(v.size)]
        g, u, an = R[f'gated_{a}'], R[f'ungated_{a}'], R[f'anti_{a}']
        print(f"\n  (3) FIRST-ORDER MAGNITUDES, increment {a}")
        print(f"      gated ratio vs rho      Spearman {_spear(R['rho'], g):+.3f}, "
              f"median gated/rho {np.nanmedian(g / R['rho']):.2f}")
        print(f"      ungated ratio vs -1     median {np.nanmedian(u):+.3f}")
        print(f"      anti ratio vs 1/rho     Spearman {_spear(1 / R['rho'], an):+.3f}, "
              f"median anti x rho {np.nanmedian(an * R['rho']):.2f}")
        out[f'mag_{a}'] = dict(sp_gated_rho=_spear(R['rho'], g),
                               gated_over_rho=float(np.nanmedian(g / R['rho'])),
                               ungated=float(np.nanmedian(u)),
                               sp_anti_invrho=_spear(1 / R['rho'], an),
                               anti_times_rho=float(np.nanmedian(an * R['rho'])))
    print("""
  HOW TO READ THIS. (1) is the licence for everything else: a strong duration
  correlation means the one-variable reduction describes these dynamics. (2) is what
  the paper will claim as derived. (3) says whether the reduction also predicts
  magnitudes; the sandbox check says mostly not, and the paper should report
  whichever way it falls rather than lean on it.""")
    save(args.out + 'wave22_R_geometry.json',
         dict(summary=out, rows=[{k: (float(v) if isinstance(v, (float, np.floating))
                                      else v) for k, v in r.items()} for r in rows]),
         args.force)
    return rows

# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--block', choices=['V', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'all'], default='all')
    ap.add_argument('--verify', action='store_true', help='run block V only')
    ap.add_argument('--analyse', action='store_true',
                    help='reanalyse existing JSON, no simulation')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--out', default='', help='output filename prefix')
    ap.add_argument('--elig', default=None,
                    help='prefix of an EXISTING wave22_A_eligibility.json to read. '
                         'Defaults to --out. Use this when writing to a new prefix but\n                         reusing a pool already built, e.g. --elig w22_ --out w22_full_')
    ap.add_argument('--n-config', type=int, default=200, dest='n_config')
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--verify-n', type=int, default=8, dest='verify_n')
    ap.add_argument('--screen-n', type=int, default=0, dest='screen_n',
                    help='block A: screen a random subsample of this many '
                         'rivalry-producing configurations (0 = all)')
    ap.add_argument('--no-rho', action='store_true', dest='no_rho',
                    help='block A: skip the Levelt-rho eligibility criterion. '
                         'Halves runtime. Eligibility becomes rivalry + CV only, '
                         'which is a deviation and must be reported as one.')
    args = ap.parse_args()

    if args.quick:
        args.n_config = min(args.n_config, 30)
        args.seeds = min(args.seeds, 4)
        args.steps = min(args.steps, 6000)
        print(f"QUICK: {args.n_config} configs, {args.seeds} seeds, {args.steps} steps")

    if args.analyse:
        for tag in ('A_eligibility', 'B_levelt', 'C_gates'):
            path = Path(args.out + f'wave22_{tag}.json')
            print(f"{path}: {'present' if path.exists() else 'MISSING'}")
        return

    if args.verify or args.block == 'V':
        block_verify(args)
        return

    elig = None
    if args.block in ('A', 'all'):
        elig = block_A(args)
    if args.block in ('B', 'all'):
        block_B(args, elig)
    if args.block in ('C', 'all'):
        block_C(args, elig)
    if args.block in ('D', 'all'):
        block_D(args, elig)
    if args.block in ('E', 'all'):
        block_E(args, elig)
    if args.block in ('F', 'all'):
        block_F(args, elig)
    if args.block in ('G', 'all'):
        block_G(args, elig)
    if args.block in ('H', 'all'):
        block_H(args, elig)
    if args.block in ('I', 'all'):
        block_I(args, elig)
    if args.block in ('J', 'all'):
        block_J(args, elig)
    if args.block in ('K', 'all'):
        block_K(args, elig)
    if args.block in ('L', 'all'):
        block_L(args, elig)
    if args.block in ('M', 'all'):
        block_M(args, elig)
    if args.block in ('N', 'all'):
        block_N(args, elig)
    if args.block in ('O', 'all'):
        block_O(args, elig)
    if args.block in ('P', 'all'):
        block_P(args, elig)
    if args.block in ('Q', 'all'):
        block_Q(args, elig)
    if args.block in ('R', 'all'):
        block_R(args, elig)


if __name__ == '__main__':
    main()
