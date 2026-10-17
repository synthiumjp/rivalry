"""Figure for the fifth test (exploratory presentation of the pre-specified analysis).
    python figure_li2017_test5.py [results.json] [outdir]
(A) competitor change under the early half, late half, late half without the transition, and the live gate;
(B) late half against late half without the transition, per configuration;
(C) brief pulses within the competitor's episode, observed against the kernel's prediction."""
import sys, os, runpy, contextlib, io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
src = sys.argv[1] if len(sys.argv) > 1 else 'li2017_test5_results.json'
outdir = sys.argv[2] if len(sys.argv) > 2 else '.'
argv0 = sys.argv; sys.argv = ['analyse_li2017_test5.py', src, os.devnull]
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analyse_li2017_test5.py'))
sys.argv = argv0
change, pred, key, PULSES, PW = G['change'], G['pred'], G['key'], G['PULSES'], G['PW']
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.labelsize': 8, 'axes.titlesize': 9, 'legend.fontsize': 6.5,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'figure.constrained_layout.use': True})
INK, ACC, WARN, MUT = '#1a1a1a', '#2b6cb0', '#c05621', '#9aa5b1'
m = 'mBm'
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.55), gridspec_kw={'width_ratios': [1.25, 0.9, 1.1]})
e, l, nt, g = (change(key(k), m) for k in ('att_early', 'att_late', 'att_late_nt', 'gated'))
rng = np.random.default_rng(1)
cols = [(0, e, 'early\nhalf'), (1, l, 'late\nhalf'), (2, nt, 'late half,\nno transition'), (3, g, 'live\ngate')]
for x0, v, _ in cols:
    v = v[np.isfinite(v)]
    ax[0].plot(x0 + rng.uniform(-0.15, 0.15, len(v)), v, '.', color=INK, ms=3, alpha=0.45)
    ax[0].hlines(np.median(v), x0 - 0.3, x0 + 0.3, color=WARN, lw=2)
ax[0].axhline(0, color=INK, lw=0.6)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels([c[2] for c in cols])
ax[0].set_ylabel('competitor change (%)')
ax[0].set_title('A  splitting the gate', loc='left')
ok = np.isfinite(l) & np.isfinite(nt)
ax[1].plot(nt[ok], l[ok], 'o', color=ACC, ms=3, alpha=0.7)
lo, hi = min(nt[ok].min(), l[ok].min()) - 1, max(nt[ok].max(), l[ok].max()) + 1
ax[1].plot([lo, hi], [lo, hi], '--', color=INK, lw=0.7)
ax[1].set_xlim(lo, hi); ax[1].set_ylim(lo, hi)
ax[1].set_xlabel('late half, no transition (%)'); ax[1].set_ylabel('late half (%)')
ax[1].set_title('B  the switch itself', loc='left')
okp = np.all([np.isfinite(pred[f'pulse{p:g}']) for p in PULSES], axis=0)
o = np.array([change(key('pulse', (p, PW)), m)[okp] for p in PULSES]); pk = np.array([pred[f'pulse{p:g}'][okp] for p in PULSES])
x = np.array(PULSES) + PW / 2
om = np.median(o, 1); oq = np.percentile(o, [25, 75], 1)
ax[2].errorbar(x, om, yerr=[om - oq[0], oq[1] - om], fmt='-o', color=ACC, ms=3.5, lw=1, label='observed')
pm = np.median(pk, 1); pq = np.percentile(pk, [25, 75], 1)
ax[2].fill_between(x, pq[0], pq[1], color=MUT, alpha=0.35, lw=0)
ax[2].plot(x, pm, 's--', color=INK, ms=3, lw=0.8, mfc='white', label='kernel')
ax[2].axhline(0, color=INK, lw=0.6)
ax[2].set_xlabel('pulse time since the competitor\'s\nepisode began (T)'); ax[2].set_ylabel('competitor change (%)')
ax[2].set_title('C  brief pulses', loc='left')
ax[2].legend(frameon=False, loc='lower left')
fig.savefig(os.path.join(outdir, 'figure_halves.pdf')); fig.savefig(os.path.join(outdir, 'figure_halves.png'), dpi=130)
print('wrote figure_halves')
