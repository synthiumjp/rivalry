"""Figure for the seventh test (presentation of the pre-specified analysis; main text Figure 7).
    python figure_li2017_test7.py [li2017_test7_results.json] [../li2017_test6/li2017_test6_results.json] [outdir]
(A) the live gate's change in the competitor under each scoring, sixth test (stronger serial correlation) against
seventh test (observer-like serial correlation); (B) the ratio of the competitor's change to the attended grating's,
sixth against seventh test; (C) seventh test: live gate, masked self-replay and self-replay, merged scoring."""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
src = sys.argv[1] if len(sys.argv) > 1 else 'li2017_test7_results.json'
ref = sys.argv[2] if len(sys.argv) > 2 else '../li2017_test6/li2017_test6_results.json'
outdir = sys.argv[3] if len(sys.argv) > 3 else '.'
D7, D6 = json.load(open(src)), json.load(open(ref))


def change(R, key, m):
    out = []
    for rb, rc in zip(R['base'], R['cond'][key]):
        b = np.array([x[m] for x in rb], float); c = np.array([x[m] for x in rc], float)
        ok = np.isfinite(b) & np.isfinite(c)
        out.append(100 * (c[ok].mean() - b[ok].mean()) / b[ok].mean() if ok.sum() and b[ok].mean() > 0 else np.nan)
    return np.array(out, float)


def ratio(R):
    c, a = change(R, 'gated', 'mBm'), change(R, 'gated', 'mA')
    ok = np.isfinite(c) & np.isfinite(a) & (a >= 5.0)
    return c[ok] / a[ok]


plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.labelsize': 8, 'axes.titlesize': 9, 'legend.fontsize': 6.5,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'figure.constrained_layout.use': True})
INK, GREY = '#1a1a1a', '#9a9a9a'
COL = {'li': '#2b6cb0', 'sfa': '#c05621', 'dep': '#2f855a'}
NAME = {'li': 'Li et al.', 'sfa': 'rate, adaptation', 'dep': 'rate, depression'}
MODELS = ['li', 'sfa', 'dep']
SC = [('mBm', 'merged'), ('mB', 'unmerged'), ('mBabs', 'absolute'), ('mOB', 'observer')]
fig = plt.figure(figsize=(7.2, 4.9))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
rng = np.random.default_rng(1)

a = fig.add_subplot(gs[0, :])
for j, m in enumerate(MODELS):
    for i, (s, lab) in enumerate(SC):
        x0 = j * 5 + i
        for dx, D, fill in ((-0.18, D6, 'none'), (0.18, D7, COL[m])):
            v = change(D['models'][m]['results'], 'gated', s)
            q = np.nanpercentile(v, [25, 50, 75])
            a.vlines(x0 + dx, q[0], q[2], color=COL[m], lw=1.1)
            a.plot(x0 + dx, q[1], 'o', ms=4.2, mfc=fill, mec=COL[m], mew=1.1)
    a.text(j * 5 + 1.5, -0.3, NAME[m], transform=a.get_xaxis_transform(), ha='center', fontsize=7.5, color=COL[m])
a.set_xticks([j * 5 + i for j in range(3) for i in range(4)])
a.set_xticklabels([lab for _ in range(3) for _, lab in SC], rotation=30, ha='right')
a.axhline(0, color=INK, lw=0.6)
a.set_ylabel('competitor change, live gate (%)')
a.plot([], [], 'o', mfc='none', mec=GREY, label='sixth test (serial correlation 0.27 to 0.31)')
a.plot([], [], 'o', mfc=GREY, mec=GREY, label='seventh test (0.09 to 0.13)')
a.legend(frameon=False, loc='upper right', ncol=2)
a.set_title('A  at observer-like serial correlation the lengthening shrinks, and vanishes on report-like scoring',
            loc='left', pad=12)

a = fig.add_subplot(gs[1, 0])
for j, m in enumerate(MODELS):
    for i, (D, fill) in enumerate(((D6, 'none'), (D7, COL[m]))):
        v = ratio(D['models'][m]['results'])
        x0 = j * 2.6 + i
        a.plot(x0 + rng.uniform(-0.18, 0.18, len(v)), v, '.', color=COL[m], ms=2.2, alpha=0.4)
        a.hlines(np.median(v), x0 - 0.3, x0 + 0.3, color=INK, lw=1.8)
a.set_xticks([j * 2.6 + i for j in range(3) for i in range(2)])
a.set_xticklabels(['6th', '7th'] * 3)
for j, m in enumerate(MODELS):
    a.text(j * 2.6 + 0.5, 1.0, NAME[m], transform=a.get_xaxis_transform(), ha='center', fontsize=6.5, color=COL[m])
a.axhline(0, color=INK, lw=0.6); a.set_ylim(-0.3, 1.0)
a.set_ylabel('competitor change / attended change')
a.set_title('B  the competitor gains less per unit\n    of attended lengthening', loc='left', pad=10)

a = fig.add_subplot(gs[1, 1])
for j, m in enumerate(MODELS):
    R = D7['models'][m]['results']
    for i, (key, lab) in enumerate((('gated', 'live'), ('replay_masked', 'masked'), ('replay_self', 'self'))):
        v = change(R, key, 'mBm'); v = v[np.isfinite(v)]
        x0 = j * 3.6 + i
        a.plot(x0 + rng.uniform(-0.18, 0.18, len(v)), v, '.', color=COL[m], ms=2.2, alpha=0.45)
        a.hlines(np.median(v), x0 - 0.32, x0 + 0.32, color=INK, lw=1.8)
a.set_xticks([j * 3.6 + i for j in range(3) for i in range(3)])
a.set_xticklabels(['live', 'masked', 'self'] * 3, rotation=45)
for j, m in enumerate(MODELS):
    a.text(j * 3.6 + 1, 1.0, NAME[m], transform=a.get_xaxis_transform(), ha='center', fontsize=6.5, color=COL[m])
a.axhline(0, color=INK, lw=0.6); a.set_ylim(-20, 25)
a.set_ylabel('competitor change (%)')
a.set_title('C  a replay never on during the competitor\n    keeps about half of the effect', loc='left', pad=10)
fig.savefig(os.path.join(outdir, 'figure_test7.pdf'))
fig.savefig(os.path.join(outdir, 'figure_test7.png'), dpi=150)
print('wrote figure_test7.pdf')
