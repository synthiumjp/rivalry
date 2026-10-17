# Supplementary Figures S10 and S11 (tests 2 and 3). Usage: python figure_supp_S10_S11.py ../li2017_test2/li2017_test2_results.json ../li2017_test2/li2017_test2_tests.json li2017_test3_results.json li2017_test3_tests.json .
"""Main-text Figures 3 and 4 of the short version, composed from the second and third registered tests.
    python figure_v19.py test2_results.json test2_tests.json test3_results.json test3_tests.json outdir
Figure 3 (where in the episode): (A) third test, five windows, observer scoring, with first-order predictions;
  (B) second test, early and late windows under three scorings, with handover and dose predictions.
Figure 4 (reversal points): (A) second test, delayed withdrawal; (B) second test, reversal delay per configuration (H10);
  (C) third test, hold after report under Chong et al.'s timing."""
import sys, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.labelsize': 8, 'axes.titlesize': 9, 'legend.fontsize': 6.5,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'figure.constrained_layout.use': True})
INK, ACC, WARN, HAZ, MUT = '#1a1a1a', '#2b6cb0', '#c05621', '#2f855a', '#9aa5b1'
r2, t2, r3, t3, outdir = sys.argv[1:6]
D2, T2 = json.load(open(r2)), json.load(open(t2))
D3, T3 = json.load(open(r3)), json.load(open(t3))


def change(S, key, m):
    out_ = []
    for rb, rc in zip(S['base'], S['cond'][key]):
        b = np.array([x[m] for x in rb], float); c = np.array([x[m] for x in rc], float)
        ok = np.isfinite(b) & np.isfinite(c)
        out_.append(100 * (c[ok].mean() - b[ok].mean()) / b[ok].mean() if ok.any() and b[ok].mean() > 0 else np.nan)
    return np.array(out_, float)


def box(ax, data, pos, color, w=0.5):
    data = [d[np.isfinite(d)] for d in data]
    ax.boxplot(data, positions=pos, widths=w, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor=color, color=INK, lw=0.6), medianprops=dict(color=INK),
               whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))


# ---------- third-test helpers
S3 = D3['settings']; M3 = D3['results']['main']
GRID = np.linspace(0, S3['GRID_MAX'], int(round(S3['GRID_MAX'] * 100)) + 1)
HOLDS = [0.0] + list(S3['HOLDS'])
D_OFF = 0.5 * sum(S3['LAT_MS']) + 0.5 * S3['RAMP_MS']
WIN = [tuple(w) for w in S3['WINDOWS']]
WKEYS = [f'win_0.2_{a:g}-{b:g}' for a, b in WIN] + ['late_0.2']
T3v = np.array(M3['T'], float); dL3 = np.array(M3['dL'], float)


def W(acc, dist, Ti, a_ms, b_ms):
    if dist is None: return np.nan
    if acc == 'handover':
        F = lambda x: float(np.interp(x, np.asarray(dist['q']), np.linspace(0, 1, 101), left=0, right=1))
        return F(b_ms) - F(a_ms)
    g = dist['g'] if acc == 'dose' else dist['hz']
    G = lambda x: float(np.interp(x / Ti, GRID, g))
    return G(b_ms) - G(a_ms)


ACCS = (('handover', ACC, 'o'), ('hazard multiplier', HAZ, '^'), ('dose', WARN, 's'))
AKEY = {'handover': 'handover', 'hazard multiplier': 'hazard', 'dose': 'dose'}

# ================= Figure 3
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={'width_ratios': [1.35, 1]})
e3 = change(M3, 'gated_0.2', 'mOB') - change(M3, 'ungated_0.2', 'mOB')
box(ax[0], [change(M3, k, 'mOB') for k in WKEYS], list(range(5)), MUT)
for j, (acc, colr, mk) in enumerate(ACCS):
    pr = []
    for i in range(len(T3v)):
        dist = M3['dists']['base_all'][i]
        if dist is None or not np.isfinite(dL3[i]) or not np.isfinite(e3[i]):
            continue
        pr.append([-e3[i] * W(AKEY[acc], dist, T3v[i], a * T3v[i], b * T3v[i]) for a, b in WIN]
                  + [-e3[i] * W(AKEY[acc], dist, T3v[i], dL3[i], 1e12)])
    ax[0].plot(np.arange(5) + (j - 1) * 0.12, np.nanmedian(np.array(pr), 0), mk, color=colr, ms=4,
               label=acc + ', first order')
ax[0].axhline(0, color=INK, lw=0.6)
ax[0].set_xticks(range(5)); ax[0].set_xticklabels(['0 to\n0.25', '0.25 to\n0.5', '0.5 to\n0.75', '0.75 to\n1', 'late'])
ax[0].set_xlabel('window within the competitor\'s episode (T)')
ax[0].set_ylabel('competitor duration change (%)')
ax[0].set_title('A  third test: four windows and the late window', loc='left')
ax[0].legend(frameon=False, loc='lower left')

M2 = D2['results']['main']
pos = {'mBm': -0.22, 'mB': 0.0, 'mBabs': 0.22}
lab = {'mBm': 'merged', 'mB': 'unmerged', 'mBabs': 'absolute-criterion'}
col = {'mBm': MUT, 'mB': '#cbd2d9', 'mBabs': '#eef0f2'}
for j, key in enumerate(('early_0.2', 'late_0.2')):
    for m in ('mBm', 'mB', 'mBabs'):
        box(ax[1], [change(M2, key, m)], [j + pos[m]], col[m], w=0.18)
wp = T2['window_predictions_median']
for j, h_ in enumerate((wp['handover_early'], wp['handover_late'])):
    ax[1].hlines(h_, j - 0.36, j + 0.36, color=ACC, lw=1.6, label='handover, first order' if j == 0 else None)
    ax[1].hlines(wp['dose_both'], j - 0.36, j + 0.36, color=WARN, lw=1.6, linestyles='dashed',
                 label='dose, first order' if j == 0 else None)
for m in ('mBm', 'mB', 'mBabs'):
    ax[1].bar([0], [0], color=col[m], edgecolor=INK, lw=0.6, label=lab[m] + (' scoring' if m != 'mBabs' else ''))
ax[1].axhline(0, color=INK, lw=0.6)
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(['early\nwindow', 'late\nwindow']); ax[1].set_xlim(-0.5, 1.5)
ax[1].set_ylabel('competitor duration change (%)')
ax[1].set_title('B  second test: scorings', loc='left')
ax[1].legend(frameon=False, loc='lower right', fontsize=5.5, handlelength=1.2)
fig.savefig(os.path.join(outdir, 'figure_where.pdf')); fig.savefig(os.path.join(outdir, 'figure_where.png'), dpi=130)

# ================= Figure 4
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6))
T2v = np.array(M2['T'], float)
DEL = [0.0] + list(D2['settings']['DELAYS'])
keys = ['gated_0.2'] + [f'offset_0.2_x{d:g}' for d in DEL[1:]]
C = np.vstack([change(M2, k, 'mBm') for k in keys]).T
s, u = change(M2, 'gated_0.2', 'mBm'), change(M2, 'ungated_0.2', 'mBm'); e = s - u
ph, pz = [], []
for i in range(len(T2v)):
    du = M2['dists']['ungated_all'][i]
    if du is None or not np.isfinite(e[i]):
        continue
    Fq = lambda x: float(np.interp(x, np.asarray(du['q']), np.linspace(0, 1, 101), left=0, right=1))
    gq = lambda xT: float(np.interp(xT, np.linspace(0, 3, 301), du['g']))
    ph.append([s[i] - e[i] * Fq(d * T2v[i]) for d in DEL]); pz.append([s[i] - e[i] * gq(d) for d in DEL])
med = np.nanmedian(C, 0); q1, q3 = np.nanpercentile(C, [25, 75], 0)
ax[0].fill_between(DEL, q1, q3, color=MUT, alpha=0.35, lw=0)
ax[0].plot(DEL, med, '-o', color=INK, ms=3, lw=1, label='observed')
ax[0].plot(DEL, np.nanmedian(np.array(ph), 0), '--', color=ACC, lw=1.2, label='handover')
ax[0].plot(DEL, np.nanmedian(np.array(pz), 0), ':', color=WARN, lw=1.4, label='dose')
ax[0].axhline(0, color=INK, lw=0.6)
ax[0].set_xlabel('delay before withdrawal (T)'); ax[0].set_ylabel('competitor duration change (%)')
ax[0].set_title('A  delayed withdrawal', loc='left')
ax[0].legend(frameon=False, loc='upper right', fontsize=6)

pc = T2['H10']['per_config']
dp, dz = np.array(pc['dpred'], float), np.array(pc['ddose'], float)
do = np.array([np.nan if v is None else v for v in pc['dobs']], float)
ok = np.isfinite(dp) & np.isfinite(dz) & np.isfinite(do) & (do > 0)
ax[1].plot(dp[ok], do[ok], '.', color=ACC, ms=3, alpha=0.6, label='handover')
ax[1].plot(dz[ok], do[ok], '.', color=WARN, ms=3, alpha=0.6, label='dose')
lim = [0.05, 2.2]
ax[1].plot(lim, lim, '--', color=INK, lw=0.7)
ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlim(lim); ax[1].set_ylim(lim)
ax[1].set_xlabel('predicted reversal delay (T)'); ax[1].set_ylabel('observed reversal delay (T)')
ax[1].set_title('B  per configuration', loc='left')
ax[1].legend(frameon=False, loc='upper left', markerscale=2)

keys = ['chong_0.1'] + [f'chong_0.1_x{h:g}' for h in HOLDS[1:]]
C = np.vstack([change(M3, k, 'mOB') for k in keys]).T
med = np.nanmedian(C, 0); q1, q3 = np.nanpercentile(C, [25, 75], 0)
ax[2].fill_between(HOLDS, q1, q3, color=MUT, alpha=0.35, lw=0)
ax[2].plot(HOLDS, med, '-o', color=INK, ms=3, lw=1, label='observed')
d0 = change(M3, 'chong_0.1', 'mOB'); e1 = change(M3, 'gated_0.1', 'mOB') - change(M3, 'ungated_0.1', 'mOB')
for acc, colr, ls in (('handover', ACC, '--'), ('hazard multiplier', HAZ, '-.'), ('dose', WARN, ':')):
    pr = []
    for i in range(len(T3v)):
        du = M3['dists']['ungated_0.1_all'][i]
        if du is None or not (np.isfinite(d0[i]) and np.isfinite(e1[i])):
            continue
        sc = d0[i] + e1[i] * W(AKEY[acc], du, T3v[i], 0, D_OFF)
        pr.append([sc - e1[i] * W(AKEY[acc], du, T3v[i], 0, D_OFF + h * T3v[i]) for h in HOLDS])
    ax[2].plot(HOLDS, np.nanmedian(np.array(pr), 0), ls, color=colr, lw=1.1, label=acc)
ax[2].axhline(0, color=INK, lw=0.6)
ax[2].set_xlabel('hold after the report (T)'); ax[2].set_ylabel('competitor duration change (%)')
ax[2].set_title('C  Chong et al.\'s timing, held', loc='left')
ax[2].legend(frameon=False, loc='lower left', fontsize=6)
fig.savefig(os.path.join(outdir, 'figure_reversal.pdf')); fig.savefig(os.path.join(outdir, 'figure_reversal.png'), dpi=130)
print('wrote figures')
