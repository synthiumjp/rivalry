"""
make_figures.py — draw the eight figures specified in the manuscript.

Reads the stored campaign JSONs, draws what the data support, and for any panel whose
inputs are absent prints exactly which file and field it needs rather than inventing
something. Nothing here re-runs a simulation.

Run from C:\\crewther. Writes PNG and PDF at 300 dpi into ./figures/.

USAGE
    python make_figures.py                 # all available panels
    python make_figures.py --figs 3 6 7    # a subset
    python make_figures.py --audit         # report inputs only, draw nothing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib required: pip install matplotlib")

OUT = Path('figures')
MISSING = []

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelsize': 8, 'axes.titlesize': 9, 'legend.fontsize': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'figure.constrained_layout.use': True,
    'figure.constrained_layout.w_pad': 0.09,
    'figure.constrained_layout.h_pad': 0.06,
})
INK, ACC, WARN, MUT = '#1a1a1a', '#2b6cb0', '#c05621', '#9aa5b1'


def load(name, quiet=False):
    for cand in (name, f'w22_{name}', f'w22_fix_{name}'):
        if os.path.exists(cand):
            with open(cand) as f:
                return json.load(f)
    if not quiet:
        MISSING.append(name)
    return None


def need(fig, panel, what):
    MISSING.append(f"Figure {fig}{panel}: {what}")


def finish(fig, num, title):
    OUT.mkdir(exist_ok=True)
    fig.suptitle(title, fontsize=9.5, ha='left', x=0.01)
    fig.savefig(OUT / f'figure{num}.png', bbox_inches='tight')
    fig.savefig(OUT / f'figure{num}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote figures/figure{num}.png and .pdf")


# ---------------------------------------------------------------- Figure 1
def figure1():
    grid = load('phase1_grid_results.json')
    if grid is None:
        return
    recs = grid.get('results', grid if isinstance(grid, list) else [])
    cv, riv = [], 0
    for r in recs:
        m = r.get('rivalry_metrics') or {}
        if r.get('rivalry_producing'):
            riv += 1
            if isinstance(m, dict) and np.isfinite(m.get('cv', np.nan) or np.nan):
                cv.append(float(m['cv']))
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 2.6))
    ax[0].axis('off')
    ax[0].text(0.0, 0.95, 'Architecture and example traces', va='top', fontsize=8.5)
    ax[0].text(0.0, 0.78, 'Panel A is a schematic and a pair of example traces.\\n'
               'Draw by hand or from a saved trace; this script does not\\n'
               'simulate. See block P in wave22_adaptation.py for a\\n'
               'trace-returning kernel if you want to generate one.',
               va='top', fontsize=7, color=MUT)
    need(1, 'A', 'architecture schematic and example traces, drawn by hand')
    if cv:
        ax[1].hist(cv, bins=60, color=ACC, alpha=.85, edgecolor='none')
        ax[1].axvspan(0.35, 0.65, color=WARN, alpha=.13, lw=0)
        ax[1].axvline(float(np.median(cv)), color=INK, lw=1, ls='--')
        ax[1].set_xlabel('coefficient of variation of dominance durations')
        ax[1].set_ylabel('configurations')
        ax[1].set_title(f'B  {riv:,} rivalry-producing of {len(recs):,}\\n'
                        f'shaded band is the registered eligibility window',
                        loc='left')
        ax[1].set_xlim(0, 1.6)
    finish(fig, 1, 'Figure 1. Parameter space and duration variability')


# ---------------------------------------------------------------- Figure 3
def figure3():
    d = load('wave22_M_phase.json')
    n = load('wave22_N_intermittency.json')
    fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.6))

    # A: the causal series, hard-coded from the canonical campaign table in 4.5.2
    conds = ['gated 2x', 'ungated 1x', 'anti-gated 2x']
    diff = [16.2, -10.4, -76.1]
    absol = [3.8, -6.5, -54.1]
    x = np.arange(3)
    ax[0].bar(x - .19, diff, .36, label='difference criterion', color=MUT)
    ax[0].bar(x + .19, absol, .36, label='absolute criterion', color=ACC)
    ax[0].axhline(0, color=INK, lw=.7)
    ax[0].set_xticks(x); ax[0].set_xticklabels(conds, rotation=18, ha='right')
    ax[0].set_ylabel('competitor duration change (%)')
    ax[0].set_title('A  one increment, three schedules', loc='left')
    ax[0].legend(frameon=False, loc='lower left')

    # B: the decomposition, both channels
    if n:
        m = n.get('medians', {})
        keys = [('ungated', 'continuous'), ('yoked', 'yoked'),
                ('shuffled', 'shuffled'), ('live', 'live gate')]
        att = [m.get('A_' + k) for k, _ in keys]
        comp = [m.get(k) for k, _ in keys]
        if all(v is not None for v in att):
            xs = np.arange(len(keys))
            ax[1].plot(xs, att, 'o-', color=WARN, label='attended', ms=4)
            ax[1].plot(xs, comp, 's-', color=ACC, label='competitor', ms=4)
            ax[1].axhline(0, color=INK, lw=.7)
            ax[1].set_xticks(xs)
            ax[1].set_xticklabels([l for _, l in keys], rotation=18, ha='right')
            ax[1].set_ylabel('duration change (%)')
            ax[1].set_title('B  both channels move together', loc='left')
            ax[1].legend(frameon=False)
        else:
            need(3, 'B', 'A_ungated/A_yoked/A_shuffled/A_live in wave22_N_intermittency.json')

    # C: burst-duration series
    if n:
        m = n.get('medians', {})
        Ts = [2, 5, 10, 25, 50, 75, 100, 150, 200]
        ys = [m.get(f'chop_{t}') for t in Ts]
        if all(y is not None for y in ys):
            ax[2].semilogx(Ts, ys, 'o-', color=INK, ms=4)
            for lab, v, c in (('yoked', m.get('yoked'), ACC),
                              ('continuous', m.get('ungated'), MUT)):
                if v is not None:
                    ax[2].axhline(v, color=c, ls='--', lw=1)
                    ax[2].text(Ts[-1], v, f' {lab}', va='center', fontsize=6.5, color=c)
            ax[2].axhline(0, color=INK, lw=.7)
            ax[2].set_xlabel('burst length (timesteps)')
            ax[2].set_ylabel('competitor change (%)')
            ax[2].set_title('C  no fixed period reaches yoked', loc='left')
        else:
            need(3, 'C', 'chop_* in wave22_N_intermittency.json')
    finish(fig, 3, 'Figure 3. Gate timing as an independent variable')


# ---------------------------------------------------------------- Figure 6
def figure6():
    b = load('wave22_B_levelt.json')
    e = load('wave22_E_prop4.json')
    k = load('wave22_K_fixedset.json')
    fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.6))
    P = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

    def series(obj, get):
        out = []
        for p in P:
            rec = (obj or {}).get(str(p)) or (obj or {}).get(f'{p:.1f}')
            out.append(get(rec) if rec else None)
        return out

    hits = series(b, lambda r: r.get('n_hit', r.get('levelt')))
    if any(h is not None for h in hits):
        ax[0].plot(P, [h if h is not None else np.nan for h in hits], 'o-',
                   color=ACC, ms=4)
        ax[0].set_xlabel('adaptation exponent p')
        ax[0].set_ylabel('Levelt II reachable (of 200)')
        ax[0].set_title('A  reachability rises with p', loc='left')
    else:
        need(6, 'A', 'n_hit per exponent in wave22_B_levelt.json')

    dd = series(e, lambda r: (r.get('filt') or {}).get('n_dd'))
    raw = series(e, lambda r: (r.get('raw') or {}).get('n_dd'))
    if any(d is not None for d in dd):
        ax[1].plot(P, [d if d is not None else np.nan for d in dd], 'o-',
                   color=ACC, ms=4, label='5-timestep filter')
        ax[1].plot(P, [d if d is not None else np.nan for d in raw], 's--',
                   color=MUT, ms=4, label='raw transitions')
        ax[1].set_xlabel('adaptation exponent p')
        ax[1].set_ylabel('Proposition IV recovered (of 200)')
        ax[1].set_title('B  and so does Proposition IV', loc='left')
        ax[1].legend(frameon=False, loc='upper left')
    else:
        need(6, 'B', 'raw/filt n_dd per exponent in wave22_E_prop4.json')

    if k:
        for name, style, col in (('fixed_intersection', 'o-', ACC),
                                 ('unfiltered', 's--', WARN)):
            sub = k.get(name) or {}
            ys = [(sub.get(str(p)) or {}).get('levelt') for p in P]
            if any(y is not None for y in ys):
                ax[2].plot(P, [y if y is not None else np.nan for y in ys], style,
                           color=col, ms=4, label=name.replace('_', ' '))
        ax[2].set_xlabel('adaptation exponent p')
        ax[2].set_ylabel('Levelt II reachable (of 200)')
        ax[2].set_title('C  on sets fixed across p', loc='left')
        ax[2].legend(frameon=False, loc='upper left')
    else:
        need(6, 'C', 'wave22_K_fixedset.json')
    finish(fig, 6, 'Figure 6. One modification repairs both structural failures')


# ---------------------------------------------------------------- Figure 7
def figure7():
    a = load('wave23_architectures.json')
    if a is None:
        return
    fams = [k for k in a if k != 'crossover' and isinstance(a[k], dict) and a[k].get('n')]
    if not fams:
        need(7, '', 'per-family summaries in wave23_architectures.json')
        return
    fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.6))
    labs = [f.split(None, 1)[-1] for f in fams]
    xs = np.arange(len(fams))

    g = [a[f]['summary'].get('gated_1x', {}) for f in fams]
    u = [a[f]['summary'].get('ungated_1x', {}) for f in fams]
    for i, (gg, uu) in enumerate(zip(g, u)):
        for d, col, off in ((gg, ACC, -.16), (uu, WARN, .16)):
            if d.get('n'):
                p = d['n_pos'] / d['n']
                lo, hi = d.get('wilson', [np.nan, np.nan])
                ax[0].errorbar(i + off, p, yerr=[[p - lo], [hi - p]], fmt='o',
                               color=col, ms=4, capsize=2)
    ax[0].axhline(.5, color=INK, lw=.7, ls=':')
    ax[0].set_xticks(xs); ax[0].set_xticklabels(labs, rotation=18, ha='right')
    ax[0].set_ylabel('proportion with positive competitor')
    ax[0].set_title('A  gated (blue) vs ungated (orange)', loc='left')
    ax[0].set_ylim(0, 1)

    r = [a[f]['summary'].get('gated_1x', {}).get('med_ratio') for f in fams]
    ax[1].bar(xs, [x if x is not None else np.nan for x in r], .55, color=MUT)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(labs, rotation=18, ha='right')
    ax[1].set_ylabel('gated coupling ratio')
    ax[1].set_title('B  magnitudes span fifteenfold', loc='left')

    c = a.get('crossover') or {}
    got = [(f, c[f]) for f in fams if f in c and c[f].get('n')]
    if got:
        for i, (f, d) in enumerate(got):
            q1, q3 = d.get('iqr', [np.nan, np.nan])
            ax[2].errorbar(i, d['median'], yerr=[[d['median'] - q1], [q3 - d['median']]],
                           fmt='o', color=ACC, ms=4, capsize=2)
        ax[2].set_xticks(range(len(got)))
        ax[2].set_xticklabels([f.split(None, 1)[-1] for f, _ in got],
                              rotation=18, ha='right')
        ax[2].set_ylabel('crossover in tau / D')
        ax[2].set_title('C  and the crossover does not', loc='left')
    else:
        need(7, 'C', "run wave23_architectures.py with --crossover")
    finish(fig, 7, 'Figure 7. The result is architectural')


# ---------------------------------------------------------------- Figure 8
def figure8():
    w = load('wave9_results.json')
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
    cl = (w or {}).get('classifiers') or {}
    rows = []
    for key, d in cl.items():
        if not isinstance(d, dict):
            continue
        for pred, v in d.items():
            if isinstance(v, dict) and 'auc' in v:
                rows.append((key, pred, float(v['auc'])))
    if rows:
        preds = sorted({p for _, p, _ in rows})
        regs = sorted({k for k, _, _ in rows})
        wdt = .8 / max(len(regs), 1)
        for i, reg in enumerate(regs):
            ys = [next((a for k, p, a in rows if k == reg and p == pr), np.nan)
                  for pr in preds]
            ax[0].bar(np.arange(len(preds)) + i * wdt, ys, wdt,
                      label=reg[:22], alpha=.9)
        ax[0].axhline(.5, color=INK, lw=.7, ls=':')
        ax[0].set_xticks(np.arange(len(preds)) + .4 - wdt / 2)
        ax[0].set_xticklabels(preds, rotation=25, ha='right')
        ax[0].set_ylabel('AUC'); ax[0].set_ylim(0, 1)
        ax[0].set_title('A  what classifies controllability', loc='left')
        ax[0].legend(frameon=False, fontsize=5.5)
    else:
        need(8, 'A', 'classifiers.*.auc in wave9_results.json')
    ax[1].axis('off')
    ax[1].text(0.0, .95, 'B  rectifier sharpness', va='top', fontsize=8.5)
    ax[1].text(0.0, .80, 'Switch rate against $k$ at each goal level, with\\n'
               'f(0) annotated. The table is in Appendix A.1; the\\n'
               'underlying sweep is not in any stored JSON, so this\\n'
               'panel needs a rerun of the softplus sweep or can be\\n'
               'drawn from the six numbers in that table.',
               va='top', fontsize=7, color=MUT)
    need(8, 'B', 'softplus sweep, or transcribe Appendix A.1 table')
    finish(fig, 8, 'Figure 8. Controllability and the graded rectifier')




# ---------------------------------------------------------------- panel audit
PANELS = {
 '2A': ('dominance duration against S_A, both channels',
        ('signal', 's_a', 'signal_a', 'sa'), ('mean_dur_a', 'dur_a', 'mean_duration_a')),
 '2B': ('slope ratios either side of equidominance',
        ('slope', 'ratio'), ('levelt', 'prop2', 'proposition')),
 '2C': ('alternation rate against predominance',
        ('predominance',), ('alternation_rate', 'alt_rate')),
 '2D': ('equal-strength sweep, fixed and signal-scaled noise',
        ('signal', 's_equal', 'scaled'), ('alternation_rate', 'n_switches')),
 '4A': ('competitor change against suppressed-phase activation, five formulations',
        ('formulation', 'mechanism', 'mode'), ('resid', 'supp', 'suppressed')),
 '4B': ('the same under four averaging windows',
        ('window', 'win'), ('resid', 'supp')),
 '4C': ('formulations ordered by suppressed-phase activation',
        ('formulation', 'mechanism'), ('resid', 'supp')),
 '5A': ('gated and ungated ratio distributions',
        ('ratio',), ('gated', 'gate')),
 '5B': ('ratio against adaptation gain, with the leave-one-out fit',
        ('ratio',), ('kappa', 'kap')),
 '5C': ('competitor response against tau / duration',
        ('tau', 'ramp', 'lag'), ('ratio', 'pb', 'competitor')),
}


def audit_panels():
    import glob
    files = sorted(glob.glob('*.json'))
    print(f"scanning {len(files)} result files for the panels with no generator\n")

    def keys_of(obj, acc, depth=0):
        if depth > 6:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                acc.add(str(k).lower())
                keys_of(v, acc, depth + 1)
        elif isinstance(obj, list) and obj:
            keys_of(obj[0], acc, depth + 1)

    index = {}
    for f in files:
        try:
            with open(f) as fh:
                acc = set()
                keys_of(json.load(fh), acc)
                index[f] = acc
        except Exception:
            pass

    for pid, (what, needA, needB) in PANELS.items():
        hits = []
        for f, ks in index.items():
            a = any(any(w in k for k in ks) for w in needA)
            b = any(any(w in k for k in ks) for w in needB)
            if a and b:
                hits.append(f)
        print(f"  {pid}  {what}")
        if hits:
            for h in hits[:4]:
                print(f"        candidate: {h}")
        else:
            print(f"        NO CANDIDATE. Needs a field matching {needA} and one "
                  f"matching {needB}.")
        print()
    print("""  Run  python registered_tests.py --dump --files <candidate>  on the
  candidates above and paste the structure; I will write the extraction. Where
  there is no candidate the campaign has to be re-run, and the block that does it
  is named in the manuscript section the panel illustrates.""")


# ---------------------------------------------------------------- Figure 2
def figure2():
    """From wave2_B_levelt.json: 30 configurations x 11 signal levels, each with
    mean_dur_a, mean_dur_b, predominance_mean, alternation_rate_mean and signal_a."""
    d = load('wave2_B_levelt.json')
    if d is None:
        return
    recs = d if isinstance(d, list) else d.get('results', [])
    fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.6))

    # A: duration against S_A, both channels, normalised to equidominance
    curves_a, curves_b, sig = [], [], None
    for r in recs:
        lv = r.get('levels') or []
        s = [l.get('signal_a') for l in lv]
        a = [l.get('mean_dur_a') for l in lv]
        b = [l.get('mean_dur_b') for l in lv]
        if not all(x is not None for x in s):
            continue
        mid = min(range(len(s)), key=lambda i: abs(s[i] - 0.5))
        if a[mid] and b[mid]:
            curves_a.append([(x / a[mid] if x else np.nan) for x in a])
            curves_b.append([(x / b[mid] if x else np.nan) for x in b])
            sig = s
    if curves_a:
        A = np.array(curves_a, float); B = np.array(curves_b, float)
        for M, col, lab in ((A, WARN, 'varied channel'), (B, ACC, 'fixed channel')):
            med = np.nanmedian(M, axis=0)
            q1, q3 = np.nanpercentile(M, [25, 75], axis=0)
            ax[0].fill_between(sig, q1, q3, color=col, alpha=.18, lw=0)
            ax[0].plot(sig, med, 'o-', color=col, ms=3, label=lab)
        ax[0].set_yscale('log'); ax[0].axvline(.5, color=INK, lw=.6, ls=':')
        ax[0].axhline(1, color=INK, lw=.6, ls=':')
        ax[0].set_xlabel('signal to the varied channel')
        ax[0].set_ylabel('duration, relative to equidominance')
        ax[0].set_title(f'A  Levelt I and II, n = {len(curves_a)}', loc='left')
        ax[0].legend(frameon=False)

    # B: slope ratio either side of equidominance
    ratios = []
    for r in recs:
        lv = r.get('levels') or []
        s = [l.get('signal_a') for l in lv]
        a = [l.get('mean_dur_a') for l in lv]
        b = [l.get('mean_dur_b') for l in lv]
        ok = [i for i in range(len(s)) if s[i] is not None and a[i] and b[i]]
        hi = [i for i in ok if s[i] > 0.52]; lo = [i for i in ok if s[i] < 0.48]
        if len(hi) >= 2 and len(lo) >= 2:
            sa = np.polyfit([s[i] for i in hi], [a[i] for i in hi], 1)[0]
            sb = np.polyfit([s[i] for i in hi], [b[i] for i in hi], 1)[0]
            if abs(sb) > 1e-9:
                ratios.append(abs(sa / sb))
    if ratios:
        v = np.array(ratios)
        pos = v[v > 0]
        ax[1].hist(np.log10(pos), bins=24, color=ACC, alpha=.85, edgecolor='none')
        ax[1].axvline(0, color=INK, lw=.8, ls='--')
        ax[1].set_xlabel('log10 |stronger slope| / |weaker slope|')
        ax[1].set_ylabel('configurations')
        ax[1].set_title(f'B  Proposition II\n{int((v>1).sum())} of {v.size} above 1',
                        loc='left')

    # C: alternation rate against predominance
    P, R = [], []
    for r in recs:
        for l in (r.get('levels') or []):
            p, a = l.get('predominance_mean'), l.get('alternation_rate_mean')
            if p is not None and a is not None and a > 0:
                P.append(p); R.append(a)
    if P:
        ax[2].plot(P, R, '.', color=MUT, ms=2, alpha=.5)
        bins = np.linspace(0, 1, 21)
        idx = np.digitize(P, bins)
        bm = [np.median([R[i] for i in range(len(P)) if idx[i] == k]) if
              any(idx[i] == k for i in range(len(P))) else np.nan
              for k in range(1, len(bins))]
        ax[2].plot(bins[:-1] + .025, bm, 'o-', color=ACC, ms=3)
        ax[2].axvline(.5, color=INK, lw=.6, ls=':')
        ax[2].set_xlabel('predominance of the varied channel')
        ax[2].set_ylabel('alternation rate')
        ax[2].set_title('C  Proposition III', loc='left')
    need(2, 'D', 'equal-strength sweep with signal-scaled noise; not in '
                 'wave2_B_levelt.json, which varies one channel only')
    finish(fig, 2, 'Figure 2. Modified Levelt propositions')


# ---------------------------------------------------------------- Figure 4
def figure4():
    """From wave15_A_paired.json: 100 configurations, modes 1-5 each swept over
    magnitude m, with supp_act and rival recorded, against a common baseline."""
    d = load('wave15_A_paired.json')
    if d is None:
        return
    recs = d if isinstance(d, list) else d.get('results', [])
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
    pts = {k: [] for k in '12345'}
    for r in recs:
        base = r.get('baseline') or {}
        b_riv, b_sup = base.get('rival'), base.get('supp_act')
        if not b_riv or b_sup is None:
            continue
        for k, lst in (r.get('modes') or {}).items():
            for e in lst:
                if e.get('rival') and e.get('supp_act') is not None:
                    pts[str(k)].append((100 * (e['supp_act'] - b_sup) / b_sup,
                                        100 * (e['rival'] - b_riv) / b_riv))
    cols = [ACC, '#38a169', WARN, '#805ad5', INK]
    for i, k in enumerate('12345'):
        if pts[k]:
            xy = np.array(pts[k], float)
            g = np.isfinite(xy).all(axis=1)
            ax[0].plot(xy[g, 0], xy[g, 1], '.', color=cols[i], ms=2.5, alpha=.45,
                       label=f'mode {k}')
    ax[0].axhline(0, color=INK, lw=.6); ax[0].axvline(0, color=INK, lw=.6)
    ax[0].set_xlabel('suppressed-phase activation change (%)')
    ax[0].set_ylabel('competitor duration change (%)')
    ax[0].set_title('A  the coupling variable', loc='left')
    ax[0].legend(frameon=False, ncol=2, fontsize=6)

    meds = []
    for k in '12345':
        if not pts[k]:
            print(f"    mode {k}: no points")
            continue
        xs = np.array([p[0] for p in pts[k]], float)
        ys = np.array([p[1] for p in pts[k]], float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        if ok.sum() == 0:
            print(f"    mode {k}: {len(xs)} points, all non-finite")
            continue
        s, c = float(np.median(xs[ok])), float(np.median(ys[ok]))
        meds.append((k, s, c))
        if ok.sum() < len(xs):
            print(f"    mode {k}: {int(ok.sum())} of {len(xs)} points finite")
    meds.sort(key=lambda t: t[1])
    if meds:
        ypos = np.arange(len(meds))
        ax[1].barh(ypos, [c for _, _, c in meds], height=.62,
                   color=[cols[int(k) - 1] for k, _, _ in meds])
        ax[1].set_yticks(ypos)
        ax[1].set_yticklabels([f'mode {k}\n({s:+.0f}% supp)' for k, s, _ in meds])
        ax[1].axvline(0, color=INK, lw=.7)
        ax[1].set_xlabel('competitor duration change (%)')
        ax[1].set_title('B  ordered by suppressed-phase activation', loc='left')
        print('    mode order by suppressed-phase activation, lowest first:')
        for k, s, c in meds:
            print(f'      mode {k}: suppressed-phase {s:+7.1f}%, '
                  f'competitor {c:+7.1f}%')
    else:
        need(4, 'B', 'finite medians per mode in wave15_A_paired.json')
    finish(fig, 4, 'Figure 4. What the coupling sign tracks')
    print("""    NOTE: modes are numbered 1 to 5 in the stored file with no names. Map them
    to the five formulations of Section 4.5.1 before use; the ordering by
    suppressed-phase activation in panel B should match that section's table.""")


# ---------------------------------------------------------------- Figure 5
def figure5():
    """From w22_wave22_C_gates.json, keyed by exponent, summary and per_config."""
    d = load('wave22_C_gates.json')
    if d is None:
        return
    key = '1.0' if '1.0' in d else sorted(d, key=float)[0]
    blk = d[key]
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
    per = blk.get('per_config') or {}
    got = False
    for cond, col, lab in (('gated_1x', ACC, 'gated'), ('ungated_1x', WARN, 'ungated')):
        rows = per.get(cond) or []
        rat = [r['pB'] / r['pA'] for r in rows
               if r.get('pA') and abs(r['pA']) > 1e-9 and r.get('pB') is not None]
        if rat:
            got = True
            ax[0].hist(np.clip(rat, -2, 2), bins=40, color=col, alpha=.6,
                       label=lab, edgecolor='none')
    if got:
        ax[0].axvline(0.310, color=INK, lw=1, ls='--')
        ax[0].text(0.315, ax[0].get_ylim()[1] * .92, ' Chong 0.310',
                   fontsize=6.5, color=INK)
        ax[0].set_xlabel('competitor-to-attended ratio')
        ax[0].set_ylabel('configurations')
        ax[0].set_title('A  gated and ungated distributions', loc='left')
        ax[0].legend(frameon=False)
    else:
        need(5, 'A', 'per_config pA/pB in wave22_C_gates.json')

    rows = per.get('gated_1x') or []

    def find_kappa(r):
        for key in ('kap', 'kappa'):
            if r.get(key) is not None:
                return float(r[key])
        p = r.get('params') or {}
        for key in ('kappa', 'kap'):
            if p.get(key) is not None:
                return float(p[key])
        return None

    kap = [find_kappa(r) for r in rows]
    if rows and not any(k is not None for k in kap):
        print(f"    panel B: no kappa in per_config rows. Keys present: "
              f"{sorted(rows[0])[:12]}")
    rat = [r['pB'] / r['pA'] if r.get('pA') and abs(r['pA']) > 1e-9 else None
           for r in rows]
    pairs = [(k, v) for k, v in zip(kap, rat) if k is not None and v is not None]
    if pairs:
        k, v = zip(*pairs)
        ax[1].plot(k, np.clip(v, -1, 2), '.', color=MUT, ms=3)
        ax[1].set_xlabel('adaptation gain kappa')
        ax[1].set_ylabel('gated coupling ratio')
        ax[1].set_title('B  ratio against adaptation gain', loc='left')
    else:
        grid = load('phase1_grid_results.json', quiet=True)
        recs = (grid or {}).get('results', grid if isinstance(grid, list) else [])
        by_idx = {i: (r.get('params') or {}).get('kappa') for i, r in enumerate(recs)}
        def cfg_id(r):
            for key in ('config', 'config_idx', 'grid_index', 'idx'):
                if r.get(key) is not None:
                    return int(r[key])
            return None
        pairs = [(by_idx.get(cfg_id(r)), rr) for r, rr in zip(rows, rat)
                 if rr is not None and by_idx.get(cfg_id(r)) is not None]
        if pairs:
            k, v = zip(*pairs)
            ax[1].plot(k, np.clip(v, -1, 2), '.', color=MUT, ms=3)
            ax[1].set_xlabel('adaptation gain kappa')
            ax[1].set_ylabel('gated coupling ratio')
            ax[1].set_title('B  ratio against adaptation gain', loc='left')
        else:
            need(5, 'B', "kappa per configuration; not in wave22_C_gates.json "
                         "per_config nor recoverable from phase1 by config index")
    need(5, 'C', 'ramp sweep, tau against competitor response; run block N or the '
                 'wave11 ramp campaign')
    finish(fig, 5, 'Figure 5. Reproducing a published gated manipulation')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--figs', nargs='*', type=int, default=None)
    ap.add_argument('--audit', action='store_true')
    ap.add_argument('--audit-panels', action='store_true', dest='ap')
    args = ap.parse_args()
    todo = {1: figure1, 2: figure2, 3: figure3, 4: figure4,
            5: figure5, 6: figure6, 7: figure7, 8: figure8}
    if args.ap:
        audit_panels()
        return
    if args.audit:
        print("panels this script can draw from stored JSONs: 1B, 3A-C, 6A-C, 7A-C, 8A")
        print("panels needing new work:              1A, 8B, and Figures 2, 4, 5")
        return
    want = args.figs or sorted(todo)
    for k in want:
        if k in todo:
            print(f"\nFigure {k}")
            todo[k]()
        else:
            print(f"\nFigure {k}: no generator.")
    if MISSING:
        print("\n" + "=" * 70)
        print("INPUTS NOT FOUND, nothing invented for these:")
        for m in dict.fromkeys(MISSING):
            print("  -", m)
        print("""
  Three panels have no stored source and need a run rather than an extraction:

    1A  architecture schematic and two example traces. Draw by hand, or use the
        trace-returning kernel in block P of wave22_adaptation.py.
    2D  equal-strength sweep under fixed and signal-scaled noise. Section 4.3's
        remedy. wave2_B_levelt.json varies one channel only, so this needs the
        bilateral sweep re-run.
    5C  ramp speed against competitor response. Block N of wave22_adaptation.py
        has the machinery; the crossover values are also in wave23 --crossover.
    8B  softplus rectifier sweep. Six numbers, transcribable from Appendix A.1.

  Everything else on this list is an extraction whose source was found.""")


if __name__ == '__main__':
    main()
