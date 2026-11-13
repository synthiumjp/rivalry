"""Figure for the eighth test (presentation of the pre-specified analysis; main text Figure 8).
    python figure_li2017_test8.py [li2017_test8_results.json] [outdir]
(A) Chong et al.'s report-contingent timing at five sizes: median competitor change against median attended change,
both under observer scoring, observer-valid configurations; (B) the competitor's change at the size calibrated to a 29%
attended lengthening, per observer-valid configuration, under observer and merged scoring."""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
src = sys.argv[1] if len(sys.argv) > 1 else 'li2017_test8_results.json'
outdir = sys.argv[2] if len(sys.argv) > 2 else '.'
D = json.load(open(src))
SIZES = (0.025, 0.05, 0.075, 0.10, 0.20)
KEYS = [f'chong_{s:g}' for s in SIZES]
TARGET = 29.0


def change(R, key, m):
    out = []
    for rb, rc in zip(R['base'], R['cond'][key]):
        b = np.array([x[m] for x in rb], float); c = np.array([x[m] for x in rc], float)
        ok = np.isfinite(b) & np.isfinite(c)
        out.append(100 * (c[ok].mean() - b[ok].mean()) / b[ok].mean() if ok.sum() and b[ok].mean() > 0 else np.nan)
    return np.array(out, float)


def calibrated(att, x):
    """As analyse_li2017_test8.py: the value of x at the size where att first reaches TARGET."""
    out = np.full(len(att), np.nan)
    for i in range(len(att)):
        a = np.r_[0.0, att[i]]; c = np.r_[0.0, x[i]]
        for j in range(1, len(a)):
            if not np.isfinite(a[j]):
                break
            if a[j] >= TARGET:
                w = (TARGET - a[j - 1]) / (a[j] - a[j - 1])
                if np.isfinite(c[j]) and np.isfinite(c[j - 1]):
                    out[i] = c[j - 1] + w * (c[j] - c[j - 1])
                break
    return out


plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.labelsize': 8, 'axes.titlesize': 9, 'legend.fontsize': 6.5,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'figure.constrained_layout.use': True})
INK, GREY = '#1a1a1a', '#9a9a9a'
COL = {'li': '#2b6cb0', 'sfa': '#c05621', 'dep': '#2f855a'}
NAME = {'li': 'Li et al.', 'sfa': 'rate, adaptation', 'dep': 'rate, depression'}
MODELS = ['li', 'sfa', 'dep']
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1, 1.15]))

a = ax[0]
for m in MODELS:
    R = D['models'][m]['results']
    valid = np.array([np.nanmean([x['nOB'] for x in c]) / np.nanmean([x['nBm'] for x in c]) for c in R['base']]) >= 0.8
    att = np.stack([change(R, k, 'mOA') for k in KEYS], 1)[valid]
    comp = np.stack([change(R, k, 'mOB') for k in KEYS], 1)[valid]
    xa = np.r_[0.0, np.nanmedian(att, 0)]; yc = np.r_[0.0, np.nanmedian(comp, 0)]
    keep = xa <= 160
    a.plot(xa[keep], yc[keep], '-o', color=COL[m], ms=3, lw=1.2, label=f'{NAME[m]} ({valid.sum()})')
a.axvline(TARGET, color=GREY, ls=':', lw=0.9)
a.axhline(0, color=INK, lw=0.6)
a.set_xlabel('attended grating, change in reported duration (%)')
a.set_ylabel('competitor, change in reported duration (%)')
a.set_title('A  Chong et al.\'s timing, sizes 2.5% to 20%', loc='left')
a.legend(frameon=False, loc='lower right')
a.text(TARGET + 2, a.get_ylim()[0] + 0.3, '29%', color=GREY, fontsize=6.5)

a = ax[1]
pos = 0
ticks, labels = [], []
for m in MODELS:
    R = D['models'][m]['results']
    valid = np.array([np.nanmean([x['nOB'] for x in c]) / np.nanmean([x['nBm'] for x in c]) for c in R['base']]) >= 0.8
    att = np.stack([change(R, k, 'mOA') for k in KEYS], 1)
    for sc, lab, mk in (('mOB', 'observer', 'o'), ('mBm', 'merged', 's')):
        x = np.stack([change(R, k, sc) for k in KEYS], 1)
        v = calibrated(att, x)[valid]; v = v[np.isfinite(v)]
        jit = (np.random.default_rng(8).random(len(v)) - 0.5) * 0.35
        a.scatter(pos + jit, v, s=7, color=COL[m], alpha=0.55 if sc == 'mOB' else 0.3, marker=mk, lw=0)
        a.plot([pos - 0.28, pos + 0.28], [np.median(v)] * 2, color=INK, lw=1.4)
        ticks.append(pos); labels.append(lab)
        pos += 1
    pos += 0.6
a.axhline(0, color=INK, lw=0.6)
a.axhline(-2, color=GREY, lw=0.8, ls='--')
a.set_xticks(ticks); a.set_xticklabels(labels, rotation=0)
for k, m in enumerate(MODELS):
    a.text(ticks[2 * k] + 0.5, 1.02, NAME[m], transform=a.get_xaxis_transform(), ha='center', fontsize=6.5, color=COL[m])
a.set_ylabel('competitor change at the calibrated size (%)')
a.set_title('B  calibrated to a 29% attended lengthening', loc='left', pad=12)
ylo, yhi = np.percentile(np.concatenate([c.get_offsets()[:, 1] for c in a.collections]), [1, 99])
a.set_ylim(min(-8, ylo - 1), max(12, yhi + 1))
fig.savefig(os.path.join(outdir, 'figure_test8.pdf')); fig.savefig(os.path.join(outdir, 'figure_test8.png'), dpi=130)
print('wrote figure_test8')
