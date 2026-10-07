"""Figure for the pre-registered test in the Li et al. (2017) model. Reads li2017_test_results.json."""
import json, sys, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelsize': 8, 'axes.titlesize': 9, 'legend.fontsize': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'figure.constrained_layout.use': True,
})
INK, ACC, WARN, MUT = '#1a1a1a', '#2b6cb0', '#c05621', '#9aa5b1'

src = sys.argv[1] if len(sys.argv) > 1 else 'li2017_test_results.json'
out = sys.argv[2] if len(sys.argv) > 2 else 'Fig8.pdf'
d = json.load(open(src))
acc, R = d['accepted'], d['results']
base = {k: np.array([c['base_matched'][k] for c in acc]) for k in ('mA', 'mB', 'mBabs')}
pct = lambda n, o: np.where(o > 0, 100 * (n - o) / o, np.nan)
comp = lambda key, crit='mB': pct(np.array([r[crit] for r in R[key]], float), base[crit])
att = lambda key: pct(np.array([r['mA'] for r in R[key]], float), base['mA'])

def wilson(k, n, z=1.96):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h

fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7))

# A: proportion lengthening, gated / ungated / yoked, both criteria, 10% increment
labels = ['gated', 'ungated', 'yoked']
for j, (crit, col, lab) in enumerate((('mB', ACC, 'difference criterion'), ('mBabs', WARN, 'absolute criterion'))):
    ps, lo, hi = [], [], []
    for m in labels:
        x = comp(f'{m}_0.1', crit); x = x[np.isfinite(x)]
        k, n = int((x > 0).sum()), len(x)
        a, b = wilson(k, n); ps.append(k / n); lo.append(k / n - a); hi.append(b - k / n)
    xs = np.arange(3) + (j - 0.5) * 0.36
    ax[0].bar(xs, ps, 0.34, color=col, label=lab)
    ax[0].errorbar(xs, ps, yerr=[lo, hi], fmt='none', ecolor=INK, lw=0.7, capsize=2)
ax[0].axhline(0.5, color=INK, lw=0.6, ls=':')
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(labels)
ax[0].set_ylim(0, 1.05); ax[0].set_ylabel('proportion: competitor lengthens')
ax[0].set_title('A  sign by schedule', loc='left')
ax[0].legend(frameon=False, loc='upper right')

# B: offset vs onset delay, 20% increment, median competitor change with IQR
dl = [0.0, 0.25, 0.5, 1.0, 1.5]
for mode, col, lab in (('offset', ACC, 'withdrawal delayed'), ('onset', WARN, 'onset delayed')):
    meds, q1, q3 = [], [], []
    for dv in dl:
        key = 'gated_0.2' if dv == 0 else f'{mode}_0.2_d{dv:g}'
        x = comp(key); x = x[np.isfinite(x)]
        meds.append(np.median(x)); q1.append(np.percentile(x, 25)); q3.append(np.percentile(x, 75))
    ax[1].plot(dl, meds, 'o-', color=col, ms=3.5, label=lab)
    ax[1].fill_between(dl, q1, q3, color=col, alpha=0.15, lw=0)
ax[1].axhline(0, color=INK, lw=0.6)
ax[1].set_xlabel('delay (mean dominance durations)')
ax[1].set_ylabel('competitor duration change (%)')
ax[1].set_title('B  delayed withdrawal vs onset', loc='left')
ax[1].legend(frameon=False, loc='lower left')

# C: gated ratio vs exp(-T/tau_h)
T = 0.5 * (base['mA'] + base['mB']); tau = np.array([c['tau_h'] for c in acc])
ratio = comp('gated_0.1') / att('gated_0.1'); pred = np.exp(-T / tau)
ok = np.isfinite(ratio) & np.isfinite(pred)
rk = lambda v: np.argsort(np.argsort(v))
rho = np.corrcoef(rk(ratio[ok]), rk(pred[ok]))[0, 1]
ax[2].plot(pred[ok], ratio[ok], '.', color=ACC, ms=3, alpha=0.6)
lim = [0, max(1.0, float(np.nanmax(pred[ok])) * 1.05)]
ax[2].plot(lim, lim, '--', color=INK, lw=0.7)
ax[2].set_xlim(lim); ax[2].set_ylim(-0.1, max(1.0, float(np.nanpercentile(ratio[ok], 99)) * 1.1))
ax[2].set_xlabel('exp(−T/τ$_h$)')
ax[2].set_ylabel('gated coupling ratio')
ax[2].set_title('C  gated ratio', loc='left')
ax[2].text(0.04, 0.95, f'Spearman {rho:+.2f}, n = {ok.sum()}', transform=ax[2].transAxes, va='top')
fig.savefig(out)
print('wrote', out, 'rho', round(rho, 3))
