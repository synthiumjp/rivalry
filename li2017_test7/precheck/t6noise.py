import json, numpy as np
from scipy.stats import spearmanr
D=json.load(open('/home/claude/chris/work/res6/li2017_test6_results.json'))
def change(R,key,m,sl=slice(None)):
    out=[]
    for rb,rc in zip(R['base'],R['cond'][key]):
        b=np.array([x[m] for x in rb[sl]],float); c=np.array([x[m] for x in rc[sl]],float); ok=np.isfinite(b)&np.isfinite(c)
        out.append(100*(c[ok].mean()-b[ok].mean())/b[ok].mean() if ok.sum() and b[ok].mean()>0 else np.nan)
    return np.array(out)
def zmean(rs):
    rs=np.asarray(rs,float); rs=rs[np.isfinite(rs)]; return float(np.tanh(np.mean(np.arctanh(np.clip(rs,-.999,.999))))) if len(rs) else np.nan
for name,M in D['models'].items():
    R=M['results']; T=np.array(R['T'])
    for m in ('mBm','mA'):
        h1=change(R,'gated',m,slice(0,16)); h2=change(R,'gated',m,slice(16,32))
        se=np.abs(h1-h2)/2  # approx SE of full estimate per config
        lo=T>np.median(T)
        print(name,m,'SE(full) median all %.1f%%, T<med %.1f%%, T>med %.1f%%'%(np.median(se),np.median(se[~lo]),np.median(se[lo])), ' corr(SE,T)=%.2f'%spearmanr(se,T).correlation)
    # r_ab SE: half-difference
    r1=np.array([zmean(x['r_ab'][:16]) for x in R['serial']['base']]); r2=np.array([zmean(x['r_ab'][16:]) for x in R['serial']['base']])
    print(name,'r_ab half SE (one half) median %.3f; T>med %.3f'%(np.median(np.abs(r1-r2)/np.sqrt(2)), np.median(np.abs(r1-r2)[T>np.median(T)]/np.sqrt(2))))
    # number of finite per-seed r_ab
    nf=np.array([np.isfinite(x['r_ab']).sum() for x in R['serial']['base']]); print('   finite seeds min', nf.min())
