"""Figure for the second registered test in Li et al.'s (2017) model.
    python figure_li2017_test2.py [results.json] [tests.json] [out.pdf]"""
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
out = sys.argv[3] if len(sys.argv) > 3 else 'Fig_test2.pdf'
D, Tt = json.load(open(src)), json.load(open(tst))
M, NA = D['results']['main'], D['results']['noadapt']


def agg(rows, m):
    return np.array([np.nanmean([s[m] for s in r]) for r in rows], float)


def change(S, key, m='mBm'):
    b, c = agg(S['base'], m), agg(S['cond'][key], m)
    with np.errstate(all='ignore'):
        return np.where(b > 0, 100 * (c - b) / b, np.nan)


fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.6))
# A: early vs late windows, with each account's predicted medians
e, l = change(M, 'early_0.2'), change(M, 'late_0.2')
for j, (x, lab) in enumerate(((e, 'early'), (l, 'late'))):
    x = x[np.isfinite(x)]
    ax[0].boxplot([x], positions=[j], widths=0.5, showfliers=False, patch_artist=True,
                  boxprops=dict(facecolor=MUT, alpha=0.5, color=INK), medianprops=dict(color=INK))
wp = Tt['window_predictions_median']
ax[0].plot([0, 1], [wp['handover_early'], wp['handover_late']], 'o', color=ACC, ms=5, label='handover prediction')
ax[0].plot([0, 1], [wp['dose_both'], wp['dose_both']], 's', color=WARN, ms=5, label='dose prediction')
ax[0].axhline(0, color=INK, lw=0.6)
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(['early\nwindow', 'late\nwindow'])
ax[0].set_ylabel('competitor duration change (%)')
ax[0].set_title('A  windows', loc='left')
ax[0].legend(frameon=False, loc='lower left')
# B: reversal delay, observed against each prediction
pc = Tt['H10']['per_config']
dp, dz = np.array(pc['dpred'], float), np.array(pc['ddose'], float)
do = np.array([np.nan if v is None else v for v in pc['dobs']], float)
ok = np.isfinite(dp) & np.isfinite(dz) & np.isfinite(do) & (do > 0)
ax[1].plot(dp[ok], do[ok], '.', color=ACC, ms=3, alpha=0.6, label='handover')
ax[1].plot(dz[ok], do[ok], '.', color=WARN, ms=3, alpha=0.6, label='dose')
lim = [0.05, 2.2]
ax[1].plot(lim, lim, '--', color=INK, lw=0.7)
ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlim(lim); ax[1].set_ylim(lim)
ax[1].set_xlabel('predicted delay (T)'); ax[1].set_ylabel('observed reversal delay (T)')
ax[1].set_title('B  reversal delay', loc='left')
ax[1].legend(frameon=False, loc='upper left', markerscale=2)
# C: Chong et al.'s timing (report latency, 520 ms ramps) at several increments.
#    5-20%: exploratory follow-up (explore/explore_results.json); 100%: registered (H11).
X = json.load(open(sys.argv[4] if len(sys.argv) > 4 else 'explore/explore_results.json'))
mainX, naX = X[:8], X[8:12]
def chg(jobs, key, m='mBm'):
    B = [r for j in jobs for r in j['base']]; C = [r for j in jobs for r in j[key]]
    b, c = agg(B, m), agg(C, m)
    riv = agg(C, 'nBm') >= 3
    with np.errstate(all='ignore'):
        x = np.where(b > 0, 100 * (c - b) / b, np.nan)
    return np.where(riv, x, np.nan)
for j, (k, lab) in enumerate((('chong_0.05', '5%'), ('chong_0.1', '10%'), ('chong_0.2', '20%'))):
    x = chg(mainX, k); x = x[np.isfinite(x)]
    ax[2].boxplot([x], positions=[j], widths=0.5, showfliers=False, patch_artist=True,
                  boxprops=dict(facecolor=MUT, alpha=0.5, color=INK), medianprops=dict(color=INK))
nout = int(np.sum(agg(M['cond']['chong_1'], 'nBm') < 3))
ax[2].text(3, 2, f'out of\nrivalry\n{nout}/200', ha='center', va='bottom', fontsize=6.5, color=WARN)
ax[2].axhline(0, color=INK, lw=0.6)
ax[2].set_xticks(range(4)); ax[2].set_xticklabels(['5%', '10%', '20%', '100%'])
ax[2].set_xlim(-0.6, 3.6)
ax[2].set_xlabel('increment')
ax[2].set_ylabel('competitor duration change (%)')
ax[2].set_title('C  Chong paradigm', loc='left')
# D: gated delivery with and without adaptation
gm, gn = change(M, 'gated_0.2'), change(NA, 'gated_0.2')
gl = chg(naX, 'gated_lat_0.2')
for j, x in enumerate((gm, gn, gl)):
    x = x[np.isfinite(x)]
    ax[3].boxplot([x], positions=[j], widths=0.5, showfliers=False, patch_artist=True,
                  boxprops=dict(facecolor=MUT, alpha=0.5, color=INK), medianprops=dict(color=INK))
ax[3].axhline(0, color=INK, lw=0.6)
ax[3].set_xticks([0, 1, 2]); ax[3].set_xticklabels(['adapt.', 'none', 'none,\n100 ms'])
ax[3].set_ylabel('competitor change, gated 20% (%)')
ax[3].set_title('D  no adaptation', loc='left')
fig.savefig(out)
print('wrote', out)
