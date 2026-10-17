"""Figure for the second registered test in Li et al.'s (2017) model (revised layout).
    python figure_li2017_test2_v2.py [results.json] [tests.json] [out.pdf]
(A) early and late windows under the three registered scorings, with each account's predicted median;
(B) delayed withdrawal: observed median curve and the first-order curves of the two accounts;
(C) observed reversal delay against each account's prediction, per configuration (H10)."""
import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.labelsize': 8, 'axes.titlesize': 9, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'figure.constrained_layout.use': True})
INK, ACC, WARN, MUT = '#1a1a1a', '#2b6cb0', '#c05621', '#9aa5b1'
src = sys.argv[1] if len(sys.argv) > 1 else 'li2017_test2_results.json'
tst = sys.argv[2] if len(sys.argv) > 2 else 'li2017_test2_tests.json'
out = sys.argv[3] if len(sys.argv) > 3 else 'figure_li2017_test2_v2.pdf'
D, Tt = json.load(open(src)), json.load(open(tst))
M = D['results']['main']
T = np.array(M['T'], float)
DEL = [0.0] + list(D['settings']['DELAYS'])


def agg(rows, m):
    return np.array([np.nanmean([s[m] for s in r]) for r in rows], float)


def change(key, m='mBm'):
    b, c = agg(M['base'], m), agg(M['cond'][key], m)
    with np.errstate(all='ignore'):
        return np.where(b > 0, 100 * (c - b) / b, np.nan)


fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
# A
pos = {'mBm': -0.22, 'mB': 0.0, 'mBabs': 0.22}
lab = {'mBm': 'merged (primary)', 'mB': 'first-test scoring', 'mBabs': 'absolute criterion'}
col = {'mBm': MUT, 'mB': '#cbd2d9', 'mBabs': '#e4e7eb'}
for j, key in enumerate(('early_0.2', 'late_0.2')):
    for m in ('mBm', 'mB', 'mBabs'):
        x = change(key, m); x = x[np.isfinite(x)]
        bp = ax[0].boxplot([x], positions=[j + pos[m]], widths=0.18, showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor=col[m], color=INK, lw=0.6), medianprops=dict(color=INK),
                           whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
wp = Tt['window_predictions_median']
for j, (h_, d_) in enumerate(((wp['handover_early'], wp['dose_both']), (wp['handover_late'], wp['dose_both']))):
    ax[0].hlines(h_, j - 0.36, j + 0.36, color=ACC, lw=1.6, label='handover prediction' if j == 0 else None)
    ax[0].hlines(d_, j - 0.36, j + 0.36, color=WARN, lw=1.6, linestyles='dashed', label='dose prediction' if j == 0 else None)
for m in ('mBm', 'mB', 'mBabs'):
    ax[0].bar([0], [0], color=col[m], edgecolor=INK, lw=0.6, label=lab[m])
ax[0].axhline(0, color=INK, lw=0.6)
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(['early\nwindow', 'late\nwindow'])
ax[0].set_xlim(-0.5, 1.5)
ax[0].set_ylabel('competitor duration change (%)')
ax[0].set_title('A  windows', loc='left')
ax[0].legend(frameon=False, loc='lower right', fontsize=5.5, handlelength=1.2)
# B
keys = ['gated_0.2'] + [f'offset_0.2_x{d:g}' for d in DEL[1:]]
C = np.vstack([change(k) for k in keys]).T
s, u = change('gated_0.2'), change('ungated_0.2'); e = s - u
ph, pz = [], []
for i in range(len(T)):
    du = M['dists']['ungated_all'][i]
    if du is None or not np.isfinite(e[i]):
        ph.append([np.nan] * len(DEL)); pz.append([np.nan] * len(DEL)); continue
    Fq = lambda x: float(np.interp(x, np.asarray(du['q']), np.linspace(0, 1, 101), left=0, right=1))
    gq = lambda xT: float(np.interp(xT, np.linspace(0, 3, 301), du['g']))
    ph.append([s[i] - e[i] * Fq(d * T[i]) for d in DEL]); pz.append([s[i] - e[i] * gq(d) for d in DEL])
ph, pz = np.array(ph), np.array(pz)
med = np.nanmedian(C, 0); q1, q3 = np.nanpercentile(C, [25, 75], 0)
ax[1].fill_between(DEL, q1, q3, color=MUT, alpha=0.35, lw=0)
ax[1].plot(DEL, med, '-o', color=INK, ms=3, lw=1, label='observed (median, IQR)')
ax[1].plot(DEL, np.nanmedian(ph, 0), '--', color=ACC, lw=1.2, label='handover, first order')
ax[1].plot(DEL, np.nanmedian(pz, 0), ':', color=WARN, lw=1.4, label='dose, first order')
ax[1].axhline(0, color=INK, lw=0.6)
ax[1].set_xlabel('delay of withdrawal (T)'); ax[1].set_ylabel('competitor duration change (%)')
ax[1].set_title('B  delayed withdrawal', loc='left')
ax[1].legend(frameon=False, loc='upper right', fontsize=6)
# C
pc = Tt['H10']['per_config']
dp, dz = np.array(pc['dpred'], float), np.array(pc['ddose'], float)
do = np.array([np.nan if v is None else v for v in pc['dobs']], float)
ok = np.isfinite(dp) & np.isfinite(dz) & np.isfinite(do) & (do > 0)
ax[2].plot(dp[ok], do[ok], '.', color=ACC, ms=3, alpha=0.6, label='handover')
ax[2].plot(dz[ok], do[ok], '.', color=WARN, ms=3, alpha=0.6, label='dose')
lim = [0.05, 2.2]
ax[2].plot(lim, lim, '--', color=INK, lw=0.7)
ax[2].set_xscale('log'); ax[2].set_yscale('log'); ax[2].set_xlim(lim); ax[2].set_ylim(lim)
ax[2].set_xlabel('predicted reversal delay (T)'); ax[2].set_ylabel('observed reversal delay (T)')
ax[2].set_title('C  reversal delay (H10)', loc='left')
ax[2].legend(frameon=False, loc='upper left', markerscale=2)
fig.savefig(out)
print('wrote', out)
