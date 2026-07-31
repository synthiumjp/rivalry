"""
wave5_prop4_diagnostic.py

Modified Prop IV came out reversed in 30/30 configurations (rho = -1.000 in
most). Two questions before that is written up as a falsification:

  1. FLICKER. wave4 counted every dominance episode, including 1-2 step
     crossings of the +/-0.05 margin. Those are densest at low signal, which is
     exactly where the correlation is anchored. Re-test with a minimum-duration
     filter at the transitional/sustained boundary (5 steps, from the b13
     mixture analysis) and at 10 steps.

  2. MECHANISM. At symmetric steady state x = S / (lambda + beta + alpha*kappa/
     gamma), so activation scales linearly with S while sigma is fixed. Raising
     S therefore raises SNR and stabilises dominance. If that is the cause,
     scaling sigma with S should restore the proposition. Tested at
     sigma = sigma_0 * (S / 0.5), i.e. constant coefficient of variation of the
     input drive.

Also reports mean dominance duration and predominance as consistency checks:
predominance must stay at 0.5 under equal signals regardless of level.

Run from C:\\crewther. Writes wave5_prop4.json, wave5_fig_prop4.pdf
"""

import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wave2_campaign import (
    run_trace, extract_durations, make_seed,
    BURN_IN, MARGIN, X_MAX, load_configs,
)

N_SEEDS = 8
N_STEPS = 12000
SWEEP = [round(0.20 + 0.05 * i, 2) for i in range(11)]   # 0.20 .. 0.70
MIN_DURS = [1, 5, 10]


def measure(p, s, sigma, block, ci, li):
    """Alternation rate at several min-duration thresholds, plus durations."""
    acc = {md: [] for md in MIN_DURS}
    durs = {md: [] for md in MIN_DURS}
    preds = []
    for seed in range(N_SEEDS):
        ta, tb = run_trace(p['lambda'], p['beta'], p['alpha'], sigma,
                           p['gamma'], p['kappa'], s, s,
                           0.0, 0.0, 0.0, 0.0,
                           N_STEPS, make_seed(block, ci, 0, li, seed),
                           X_MAX, -1.0, 0.0, 0.0)
        ch, dur = extract_durations(ta, tb, BURN_IN, MARGIN)
        if len(ch) < 3:
            continue
        ch, dur = ch[1:-1], dur[1:-1].astype(float)
        for md in MIN_DURS:
            k = dur >= md
            c, d = ch[k], dur[k]
            if len(c) < 2:
                continue
            acc[md].append(int(np.sum(c[1:] != c[:-1])) / (N_STEPS - BURN_IN))
            durs[md].append(float(d.mean()))
        if len(ch):
            ta_ = dur[ch == 0].sum()
            tb_ = dur[ch == 1].sum()
            if ta_ + tb_ > 0:
                preds.append(ta_ / (ta_ + tb_))
    out = {}
    for md in MIN_DURS:
        out['alt_%d' % md] = float(np.mean(acc[md])) if acc[md] else np.nan
        out['dur_%d' % md] = float(np.mean(durs[md])) if durs[md] else np.nan
    out['predominance'] = float(np.mean(preds)) if preds else np.nan
    return out


def run(cfgs, scaled_noise, block, label):
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    print("  cfg |  rho(min=1)  rho(min=5)  rho(min=10) | rho(duration,min=5)")
    out = []
    for ci in sorted(cfgs.keys()):
        p = cfgs[ci]
        rows = []
        for li, s in enumerate(SWEEP):
            sig = p['sigma'] * (s / 0.5) if scaled_noise else p['sigma']
            r = measure(p, s, sig, block, ci, li)
            r['signal'] = s
            r['sigma'] = sig
            rows.append(r)
        S = np.array([r['signal'] for r in rows])
        rec = {'config_idx': ci, 'params': p,
               'scaled_noise': scaled_noise, 'levels': rows}
        line = "  %3d |" % ci
        for md in MIN_DURS:
            A = np.array([r['alt_%d' % md] for r in rows], dtype=float)
            m = np.isfinite(A)
            if m.sum() < 5:
                rec['rho_alt_%d' % md] = None
                line += "      n/a  "
                continue
            rho, pv = stats.spearmanr(S[m], A[m])
            rec['rho_alt_%d' % md] = float(rho)
            rec['p_alt_%d' % md] = float(pv)
            line += "  %+7.3f  " % rho
        D = np.array([r['dur_5'] for r in rows], dtype=float)
        m = np.isfinite(D)
        if m.sum() >= 5:
            rho_d, _ = stats.spearmanr(S[m], D[m])
            rec['rho_dur_5'] = float(rho_d)
            line += "|  %+7.3f" % rho_d
        P = np.array([r['predominance'] for r in rows], dtype=float)
        rec['predominance_mean'] = float(np.nanmean(P))
        out.append(rec)
        print(line)

    for md in MIN_DURS:
        v = [r['rho_alt_%d' % md] for r in out
             if r.get('rho_alt_%d' % md) is not None]
        if not v:
            continue
        print("\n  min_dur=%d : %d/%d increasing (rho>0.7), "
              "%d/%d decreasing (rho<-0.7), median rho %+.3f"
              % (md, sum(1 for x in v if x > 0.7), len(v),
                 sum(1 for x in v if x < -0.7), len(v), np.median(v)))
    pm = np.nanmean([r['predominance_mean'] for r in out])
    print("  predominance under equal signals: %.4f (must be ~0.500)" % pm)
    return out


def figure(fixed, scaled):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for rec in fixed:
        S = [r['signal'] for r in rec['levels']]
        ax[0].plot(S, [r['alt_1'] for r in rec['levels']], '-',
                   lw=0.7, alpha=0.4, color='#999999')
        ax[0].plot(S, [r['alt_5'] for r in rec['levels']], '-',
                   lw=0.8, alpha=0.5, color='#2c6fbb')
    ax[0].set_xlabel(r'$S_A = S_B$')
    ax[0].set_ylabel('alternation rate')
    ax[0].set_title('A  Fixed noise\ngrey: unfiltered, blue: min 5 steps')

    for rec in scaled:
        S = [r['signal'] for r in rec['levels']]
        ax[1].plot(S, [r['alt_5'] for r in rec['levels']], '-',
                   lw=0.8, alpha=0.5, color='#d1495b')
    ax[1].set_xlabel(r'$S_A = S_B$')
    ax[1].set_ylabel('alternation rate')
    ax[1].set_title(r'B  Noise scaled with signal')

    rf = [r['rho_alt_5'] for r in fixed if r.get('rho_alt_5') is not None]
    rs = [r['rho_alt_5'] for r in scaled if r.get('rho_alt_5') is not None]
    ax[2].hist([rf, rs], bins=12, label=['fixed', 'scaled'],
               color=['#2c6fbb', '#d1495b'])
    ax[2].axvline(0, ls='--', c='k', lw=1)
    ax[2].set_xlabel(r'Spearman $\rho$ (signal vs alternation rate)')
    ax[2].set_ylabel('configs')
    ax[2].set_title('C  Modified Prop IV')
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig('wave5_fig_prop4.pdf')
    print("\n  wave5_fig_prop4.pdf")


def main():
    cfgs = load_configs()
    print("Loaded %d configs" % len(cfgs))
    fixed = run(cfgs, False, 14, "MODIFIED PROP IV -- fixed noise (as published)")
    scaled = run(cfgs, True, 15,
                 "MODIFIED PROP IV -- noise scaled with signal (mechanism test)")
    figure(fixed, scaled)
    with open('wave5_prop4.json', 'w', encoding='utf-8') as f:
        json.dump({'fixed_noise': fixed, 'scaled_noise': scaled},
                  f, indent=1, default=float)
    print("  wave5_prop4.json")


if __name__ == '__main__':
    main()
