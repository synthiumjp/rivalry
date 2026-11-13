"""Figure for the sixth test (presentation of the pre-specified analysis; main text Figure 6).
    python figure_li2017_test6.py [li2017_test6_results.json] [outdir]
(A) change in each percentile of the merged competitor duration distribution, live gate and continuous delivery;
(B) live gate, self-replay and other-seed replay; (C) baseline serial dependence against the live gate's change;
(D) Li et al.'s model: input increment against attention-layer bias, gated and continuous."""
import sys, os, runpy, contextlib, io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
src = sys.argv[1] if len(sys.argv) > 1 else 'li2017_test6_results.json'
outdir = sys.argv[2] if len(sys.argv) > 2 else '.'
argv0 = sys.argv; sys.argv = ['analyse_li2017_test6.py', src, os.devnull]
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analyse_li2017_test6.py'))
sys.argv = argv0
D, change, dist_change, zmean, QI = G['D'], G['change'], G['dist_change'], G['zmean'], G['QI']
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.labelsize': 8, 'axes.titlesize': 9, 'legend.fontsize': 6.5,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'figure.constrained_layout.use': True})
INK = '#1a1a1a'
COL = {'li': '#2b6cb0', 'sfa': '#c05621', 'dep': '#2f855a'}
NAME = {'li': 'Li et al.', 'sfa': 'rate, adaptation', 'dep': 'rate, depression'}
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4))
a = ax[0, 0]
qs = sorted(QI)
for m, M in D['models'].items():
    R = M['results']
    for key, ls in (('gated', '-'), ('ungated', '--')):
        y = [np.nanmedian(dist_change(R, key, 'q', QI[q])) for q in qs]
        a.plot(qs, y, ls, color=COL[m], lw=1.3, label=f'{NAME[m]}, {"gated" if key == "gated" else "continuous"}')
a.axhline(0, color=INK, lw=0.6); a.set_ylim(-35, 62)
a.set_xlabel('percentile of competitor durations'); a.set_ylabel('change (%)')
a.set_title('A  the percentiles shift with timing', loc='left'); a.legend(frameon=False, ncol=2, loc='upper right', fontsize=5.5)
a = ax[0, 1]
rng = np.random.default_rng(1)
for j, (m, M) in enumerate(D['models'].items()):
    R = M['results']
    for i, (key, lab) in enumerate((('gated', 'live'), ('replay_self', 'self'), ('replay_other', 'other'))):
        v = change(R, key, 'mBm'); v = v[np.isfinite(v)]
        x0 = j * 3.6 + i
        a.plot(x0 + rng.uniform(-0.18, 0.18, len(v)), v, '.', color=COL[m], ms=2.2, alpha=0.45)
        a.hlines(np.median(v), x0 - 0.32, x0 + 0.32, color=INK, lw=1.8)
a.set_xticks([j * 3.6 + i for j in range(3) for i in range(3)])
a.set_xticklabels(['live', 'self', 'other'] * 3, rotation=0)
for j, m in enumerate(D['models']):
    a.text(j * 3.6 + 1, 1.02, NAME[m], transform=a.get_xaxis_transform(), ha='center', fontsize=6.5, color=COL[m])
a.axhline(0, color=INK, lw=0.6); a.set_ylabel('competitor change (%)')
a.set_title('B  open-loop replays lose most of the effect', loc='left', pad=12)
a = ax[1, 0]
for m, M in D['models'].items():
    R = M['results']
    r = np.array([zmean(s['r_ab']) for s in R['serial']['base']]); g = change(R, 'gated', 'mBm')
    a.plot(r, g, 'o', color=COL[m], ms=3, alpha=0.7, label=NAME[m])
a.axvline(0.2, color=INK, lw=0.6, ls=':'); a.axhline(0, color=INK, lw=0.6)
a.set_xlabel('baseline lag-1 correlation, attended to next competitor'); a.set_ylabel('gated change (%)')
a.set_title('C  stronger serial correlation, larger effect', loc='left'); a.legend(frameon=False, loc='upper left')
a = ax[1, 1]
R = D['models']['li']['results']
conds = [('gated', 'input\ngated'), ('ungated', 'input\ncont.'), ('att_gated0.25', 'attention\ngated 0.25'),
         ('att_ungated0.25', 'attention\ncont. 0.25'), ('att_gated0.5', 'attention\ngated 0.5'),
         ('att_ungated0.5', 'attention\ncont. 0.5')]
for i, (key, lab) in enumerate(conds):
    v = change(R, key, 'mBm'); v = v[np.isfinite(v)]
    a.plot(i + rng.uniform(-0.18, 0.18, len(v)), v, '.', color=COL['li'], ms=2.2, alpha=0.45)
    a.hlines(np.median(v), i - 0.32, i + 0.32, color=INK, lw=1.8)
a.set_xticks(range(len(conds))); a.set_xticklabels([c[1] for c in conds], fontsize=6)
a.axhline(0, color=INK, lw=0.6); a.set_ylabel('competitor change (%)')
a.set_title('D  the attention layer behaves the same way', loc='left')
fig.savefig(os.path.join(outdir, 'figure_test6.pdf')); fig.savefig(os.path.join(outdir, 'figure_test6.png'), dpi=130)
print('wrote figure_test6')
