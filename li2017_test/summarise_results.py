import json, numpy as np, math, sys
d=json.load(open(sys.argv[1] if len(sys.argv)>1 else 'li2017_test_results.json'))
acc=d['accepted']; R=d['results']
base={k:np.array([c['base_matched'][k] for c in acc]) for k in ('mA','mB','mBabs')}
def pct(n,o):
    with np.errstate(all='ignore'): return np.where(o>0,100*(n-o)/o,np.nan)
def wil(k,n,z=1.96):
    if n==0: return (np.nan,np.nan)
    p=k/n; c=(p+z*z/(2*n))/(1+z*z/n); h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n); return (c-h,c+h)
print('accepted',len(acc),'of',d['n_drawn'],'drawn; median baseline dur',np.median(0.5*(base['mA']+base['mB'])).round(0),'ms')
print(f"{'condition':22s} {'att':>7s} {'comp(diff)':>10s} {'len k/n':>9s} {'CI':>14s} {'comp(abs)':>10s} {'len k/n':>9s} {'duty':>5s}")
for k,v in R.items():
    a=pct(np.array([r['mA'] for r in v]),base['mA']); b=pct(np.array([r['mB'] for r in v]),base['mB']); ba=pct(np.array([r['mBabs'] for r in v]),base['mBabs'])
    fb=b[np.isfinite(b)]; fa=ba[np.isfinite(ba)]
    kb,nb=int((fb>0).sum()),len(fb); ka,na=int((fa>0).sum()),len(fa)
    lo,hi=wil(kb,nb)
    print(f"{k:22s} {np.nanmedian(a):+7.1f} {np.nanmedian(b):+10.1f} {kb:4d}/{nb:<4d} [{100*lo:4.0f}%,{100*hi:4.0f}%] {np.nanmedian(ba):+10.1f} {ka:4d}/{na:<4d} {np.mean([r['duty'] for r in v]):5.2f}")
print('\nTESTS'); 
for k,v in d['tests'].items(): print(' ',k,v)
print('\nPUBLISHED CONFIG')
pb=d['published']['config']['base_matched']
for k,v in d['published']['results'].items():
    print(f"  {k:22s} att {100*(v['mA']-pb['mA'])/pb['mA']:+6.1f}  comp {100*(v['mB']-pb['mB'])/pb['mB']:+6.1f}  comp(abs) {100*(v['mBabs']-pb['mBabs'])/pb['mBabs']:+6.1f}")
