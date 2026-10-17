"""Figure for the third registered test (revised layout: A windows, B holds, C-E gates x scorings) in Li et al.'s (2017) model.
    python figure_li2017_test3.py [results.json] [tests.json] [out.pdf]
(A) window profile (observer scoring) with each account's first-order prediction (medians);
(B) Chong et al.'s timing at 10%: competitor change against hold, with each account's first-order curve;
(C) gates: zero-latency, 100 ms latency and report gate in the three samples;
(D) early minus late window, relative to e, in the three samples (H13, H17)."""
import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.labelsize': 8, 'axes.titlesize': 9, 'legend.fontsize': 6.5,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'figure.constrained_layout.use': True})
INK, ACC, WARN, HAZ, MUT = '#1a1a1a', '#2b6cb0', '#c05621', '#2f855a', '#9aa5b1'
src = sys.argv[1] if len(sys.argv) > 1 else 'li2017_test3_results.json'
tst = sys.argv[2] if len(sys.argv) > 2 else 'li2017_test3_tests.json'
out = sys.argv[3] if len(sys.argv) > 3 else 'figure_li2017_test3.pdf'
D, Tt = json.load(open(src)), json.load(open(tst))
S_ = D['settings']
M, NA, HN = D['results']['main'], D['results']['noadapt'], D['results']['highnoise']
GRID = np.linspace(0, S_['GRID_MAX'], int(round(S_['GRID_MAX'] * 100)) + 1)
HOLDS = [0.0] + list(S_['HOLDS'])
D_OFF = 0.5 * sum(S_['LAT_MS']) + 0.5 * S_['RAMP_MS']
WIN = [tuple(w) for w in S_['WINDOWS']]
WKEYS = [f'win_0.2_{a:g}-{b:g}' for a, b in WIN] + ['late_0.2']
P = 'mOB'


def change(S, key, m=P):
    out_ = []
    for rb, rc in zip(S['base'], S['cond'][key]):
        b = np.array([x[m] for x in rb], float); c = np.array([x[m] for x in rc], float)
        ok = np.isfinite(b) & np.isfinite(c)
        out_.append(100 * (c[ok].mean() - b[ok].mean()) / b[ok].mean() if ok.any() and b[ok].mean() > 0 else np.nan)
    return np.array(out_, float)


def W(acc, dist, Ti, a_ms, b_ms):
    if dist is None: return np.nan
    if acc == 'handover':
        F = lambda x: float(np.interp(x, np.asarray(dist['q']), np.linspace(0, 1, 101), left=0, right=1))
        return F(b_ms) - F(a_ms)
    g = dist['g'] if acc == 'dose' else dist['hz']
    G = lambda x: float(np.interp(x / Ti, GRID, g))
    return G(b_ms) - G(a_ms)


def box(ax, data, pos, color, w=0.5):
    data = [d[np.isfinite(d)] for d in data]
    ax.boxplot(data, positions=pos, widths=w, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor=color, color=INK, lw=0.6), medianprops=dict(color=INK),
               whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))


fig = plt.figure(figsize=(7.4, 5.0))
gs = fig.add_gridspec(2, 6)
ax = [fig.add_subplot(gs[0, 0:3]), fig.add_subplot(gs[0, 3:6])]
hx = [fig.add_subplot(gs[1, 2 * j:2 * j + 2]) for j in range(3)]
T = np.array(M['T'], float); dL = np.array(M['dL'], float)
ACCS = (('handover', ACC, 'o'), ('hazard', HAZ, '^'), ('dose', WARN, 's'))
# A: window profile
e = change(M, 'gated_0.2') - change(M, 'ungated_0.2')
box(ax[0], [change(M, k) for k in WKEYS], list(range(5)), MUT)
for acc, colr, mk in ACCS:
    pr = []
    for i in range(len(T)):
        dist = M['dists']['base_all'][i]
        if dist is None or not np.isfinite(dL[i]) or not np.isfinite(e[i]):
            continue
        pr.append([-e[i] * W(acc, dist, T[i], a * T[i], b * T[i]) for a, b in WIN] + [-e[i] * W(acc, dist, T[i], dL[i], 1e12)])
    if pr:
        ax[0].plot(np.arange(5) + {'handover': -0.12, 'hazard': 0.0, 'dose': 0.12}[acc], np.nanmedian(np.array(pr), 0), mk, color=colr, ms=4, label=acc + ' (first order)')
ax[0].axhline(0, color=INK, lw=0.6)
ax[0].set_xticks(range(5)); ax[0].set_xticklabels(['0-\n.25', '.25-\n.5', '.5-\n.75', '.75-\n1', 'late'])
ax[0].set_xlabel('window (T from episode start)')
ax[0].set_ylabel('competitor change (%)')
ax[0].set_title('A  windows, 20%', loc='left')
ax[0].legend(frameon=False, loc='lower left', fontsize=6)
# B: holds under Chong et al.'s timing, 10%
keys = ['chong_0.1'] + [f'chong_0.1_x{h:g}' for h in HOLDS[1:]]
C = np.vstack([change(M, k) for k in keys]).T
med = np.nanmedian(C, 0); q1, q3 = np.nanpercentile(C, [25, 75], 0)
ax[1].fill_between(HOLDS, q1, q3, color=MUT, alpha=0.35, lw=0)
ax[1].plot(HOLDS, med, '-o', color=INK, ms=3, lw=1, label='observed')
d0 = change(M, 'chong_0.1'); e1 = change(M, 'gated_0.1') - change(M, 'ungated_0.1')
for acc, colr, mk in ACCS:
    pr = []
    for i in range(len(T)):
        du = M['dists']['ungated_0.1_all'][i]
        if du is None or not (np.isfinite(d0[i]) and np.isfinite(e1[i])):
            continue
        sc = d0[i] + e1[i] * W(acc, du, T[i], 0, D_OFF)
        pr.append([sc - e1[i] * W(acc, du, T[i], 0, D_OFF + h * T[i]) for h in HOLDS])
    if pr:
        ax[1].plot(HOLDS, np.nanmedian(np.array(pr), 0), '--', color=colr, lw=1.1, label=acc)
ax[1].axhline(0, color=INK, lw=0.6)
ax[1].set_xlabel('hold after report (T)'); ax[1].set_ylabel('competitor change (%)')
ax[1].set_title('B  Chong timing, 10%', loc='left')
ax[1].legend(frameon=False, loc='lower left')
# D-F: gate x scoring, median competitor change at 20%
GATES = (('gated_0.2', 'no latency'), ('gated_lat_0.2', '100 ms'), ('rep_gated_0.2', 'report'), ('chong_0.1', 'Chong, 10%'))
SCS = (('mOB', 'observer'), ('mBm', 'merged'), ('mB', 'first test'), ('mBabs', 'absolute'))
from matplotlib.colors import TwoSlopeNorm
norm = TwoSlopeNorm(vmin=-25, vcenter=0, vmax=25)
for j, (S, nm, lt) in enumerate(((M, 'adaptation', 'C'), (NA, 'no adaptation', 'D'), (HN, 'adapt., high noise', 'E'))):
    Z = np.array([[np.nanmedian(change(S, g, m)) for m, _ in SCS] for g, _ in GATES])
    hx[j].imshow(Z, cmap='RdBu', norm=norm, aspect='auto')
    for r in range(Z.shape[0]):
        for c in range(Z.shape[1]):
            hx[j].text(c, r, f'{Z[r, c]:+.1f}', ha='center', va='center', fontsize=6.5,
                       color='white' if abs(Z[r, c]) > 15 else INK)
    hx[j].set_xticks(range(4)); hx[j].set_xticklabels([l for _, l in SCS], rotation=30, ha='right')
    hx[j].set_yticks(range(len(GATES))); hx[j].set_yticklabels([l for _, l in GATES] if j == 0 else [])
    if j > 0:   # the observer reports fewer than half of the episodes in the control samples
        from matplotlib.patches import Rectangle
        hx[j].add_patch(Rectangle((-0.5, -0.5), 1, len(GATES), fill=False, hatch='///', edgecolor='#52606d', lw=0))
    hx[j].set_title(f'{lt}  {nm}', loc='left', fontsize=8)
    for sp in hx[j].spines.values(): sp.set_visible(False)
fig.savefig(out)
print('wrote', out)
