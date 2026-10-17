"""Figures for the fourth test (exploratory presentation of the pre-specified analysis).
    python figure_li2017_test4.py [results.json] [outdir]
Figure A (kernel): (A) the kernel; (B) window profile at 5% and 20% per unit increment, observed vs kernel;
  (C) continuous delivery, observed vs kernel prediction per configuration.
Figure B (closed loop): (A) live gate, yoked replay and kernel predictions at 5%; (B) gated and continuous
  changes against increment size, observed vs kernel."""
import sys, os, runpy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
src = sys.argv[1] if len(sys.argv) > 1 else 'li2017_test4_results.json'
outdir = sys.argv[2] if len(sys.argv) > 2 else '.'
argv0 = sys.argv; sys.argv = ['analyse_li2017_test4.py', src, os.devnull]
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analyse_li2017_test4.py'))
sys.argv = argv0
change, pred, C, K, WK, obs_key, BIN = G['change'], G['pred'], G['C'], G['K'], G['WK'], G['obs_key'], G['BIN']
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.labelsize': 8, 'axes.titlesize': 9, 'legend.fontsize': 6.5,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'figure.constrained_layout.use': True})
INK, ACC, WARN, HAZ, MUT = '#1a1a1a', '#2b6cb0', '#c05621', '#2f855a', '#9aa5b1'
m = 'mBm'

# ================= kernel figure
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5), gridspec_kw={'width_ratios': [1.2, 1.2, 1]})
KP = np.array([100 * 0.1 * c['inp'] * np.array(k['bA']) / k['meanD'] for c, k in zip(C, K) if k is not None])
t = (np.arange(KP.shape[1]) + 0.5) * BIN
q1, md, q3 = np.percentile(KP, [25, 50, 75], 0)
ax[0].fill_between(t, q1, q3, color=MUT, alpha=0.4, lw=0)
ax[0].plot(t, md, '-o', color=INK, ms=2.5, lw=1)
ax[0].axhline(0, color=INK, lw=0.6)
ax[0].set_xlabel('time since the competitor\'s episode began (T)')
ax[0].set_ylabel('competitor change (%)\nper 10% increment for 0.125 T')
ax[0].set_title('A  kernel from baseline noise', loc='left')
xs = np.arange(len(WK))
for f, off, col, lab in ((0.05, -0.12, ACC, '5%'), (0.2, 0.12, WARN, '20%')):
    o = np.array([change(obs_key(wk, f), m) for wk in WK]) * (0.1 / f)
    p = np.array([pred[f'{wk}_{f:g}'] for wk in WK]) * (0.1 / f)
    om = np.nanmedian(o, 1); oq = np.nanpercentile(o, [25, 75], 1)
    ax[1].errorbar(xs + off, om, yerr=[om - oq[0], oq[1] - om], fmt='o', color=col, ms=3.5, lw=0.8, label=f'observed, {lab}')
pm = np.array([np.nanmedian(pred[f'{wk}_0.05']) * 2 for wk in WK])
ax[1].plot(xs, pm, 's--', color=INK, ms=3, lw=0.8, mfc='white', label='kernel (either size)')
ax[1].axhline(0, color=INK, lw=0.6)
ax[1].set_xticks(xs); ax[1].set_xticklabels(['0 to\n0.25', '0.25 to\n0.5', '0.5 to\n0.75', '0.75 to\n1', 'late'])
ax[1].set_ylabel('competitor change (%)\nper 10% increment')
ax[1].set_xlabel('window (T)')
ax[1].set_title('B  windows: small vs large', loc='left')
ax[1].legend(frameon=False, loc='lower left')
for f, col, mk in ((0.05, ACC, 'o'), (0.1, WARN, '^')):
    o, p = change(f'ungated_{f:g}', m), pred[f'ungated_{f:g}']
    ax[2].plot(p, o, mk, color=col, ms=3, alpha=0.7, label=f'{int(f*100)}%')
lim = [-45, 0]
ax[2].plot(lim, lim, '--', color=INK, lw=0.7)
ax[2].set_xlim(lim); ax[2].set_ylim(lim)
ax[2].set_xlabel('kernel prediction (%)'); ax[2].set_ylabel('observed (%)')
ax[2].set_title('C  continuous delivery', loc='left')
ax[2].legend(frameon=False, loc='upper left')
fig.savefig(os.path.join(outdir, 'figure_kernel.pdf')); fig.savefig(os.path.join(outdir, 'figure_kernel.png'), dpi=130)

# ================= closed-loop figure
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw={'width_ratios': [1.1, 1]})
g, y = change('gated_0.05', m), change('yoked_0.05', m)
pg, py = pred['gated_0.05'], pred['yoked_0.05']
rng = np.random.default_rng(1)
cols = [(0, g, INK, 'live gate,\nobserved'), (1, pg, ACC, 'live gate,\nkernel'), (2, y, INK, 'yoked,\nobserved'), (3, py, ACC, 'yoked,\nkernel')]
for x0, v, col, _ in cols:
    v = v[np.isfinite(v)]
    ax[0].plot(x0 + rng.uniform(-0.15, 0.15, len(v)), v, '.', color=col, ms=3, alpha=0.5)
    ax[0].hlines(np.median(v), x0 - 0.3, x0 + 0.3, color=WARN, lw=2)
ok = np.isfinite(g) & np.isfinite(y)
for a_, b_ in zip(g[ok], y[ok]):
    pass
ax[0].axhline(0, color=INK, lw=0.6)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels([c[3] for c in cols])
ax[0].set_ylabel('competitor change (%)')
ax[0].set_title('A  5% increment', loc='left')
F = [0.05, 0.1, 0.2]
for key, col, lab in (('gated', ACC, 'gated'), ('ungated', WARN, 'continuous')):
    o = np.array([change(f'{key}_{f:g}', m) for f in F]); om = np.nanmedian(o, 1); oq = np.nanpercentile(o, [25, 75], 1)
    ax[1].errorbar(np.array(F) * 100, om, yerr=[om - oq[0], oq[1] - om], fmt='-o', color=col, ms=3.5, lw=1, label=f'{lab}, observed')
    p = np.array([np.nanmedian(pred[f'{key}_{f:g}']) for f in F])
    ax[1].plot(np.array(F) * 100, p, '--', color=col, lw=1, label=f'{lab}, kernel')
ax[1].axhline(0, color=INK, lw=0.6)
ax[1].set_xlabel('increment (% of input strength)'); ax[1].set_ylabel('competitor change (%)')
ax[1].set_xticks([5, 10, 20])
ax[1].set_title('B  increment size', loc='left')
ax[1].legend(frameon=False, loc='lower left')
fig.savefig(os.path.join(outdir, 'figure_closedloop.pdf')); fig.savefig(os.path.join(outdir, 'figure_closedloop.png'), dpi=130)
print('wrote figures')
