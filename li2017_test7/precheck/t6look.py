import json, numpy as np, sys
sys.path.insert(0,'/home/claude/chris/work/t7/li2017_test7')
D=json.load(open('/home/claude/chris/work/res6/li2017_test6_results.json'))
from scipy.stats import spearmanr
def change(R,key,m,sl=slice(None)):
    out=[]
    for rb,rc in zip(R['base'],R['cond'][key]):
        b=np.array([x[m] for x in rb[sl]],float); c=np.array([x[m] for x in rc[sl]],float); ok=np.isfinite(b)&np.isfinite(c)
        out.append(100*(c[ok].mean()-b[ok].mean())/b[ok].mean() if ok.sum() and b[ok].mean()>0 else np.nan)
    return np.array(out)
def zmean(rs):
    rs=np.asarray(rs,float); rs=rs[np.isfinite(rs)]; return float(np.tanh(np.mean(np.arctanh(np.clip(rs,-.999,.999))))) if len(rs) else np.nan
for name,M in D['models'].items():
    R=M['results']
    g=change(R,'gated','mBm'); a=change(R,'gated','mA'); u=change(R,'ungated','mBm'); s=change(R,'replay_self','mBm')
    r=np.array([zmean(x['r_ab']) for x in R['serial']['base']])
    T=np.array(R['T'])
    h=16
    rh=np.array([zmean(x['r_ab'][:h]) for x in R['serial']['base']]); rh2=np.array([zmean(x['r_ab'][h:]) for x in R['serial']['base']])
    print(name, 'n',len(g),'r med %.3f'%np.median(r), 'gated comp med %.1f'%np.nanmedian(g),'att med %.1f'%np.nanmedian(a),'ungated %.1f'%np.nanmedian(u),'self %.1f'%np.nanmedian(s))
    print('   split-half r reliability (rho between halves): %.2f'%spearmanr(rh,rh2).correlation, ' SD r across cfgs %.3f'%np.std(r))
    lo=r<0.2
    print('   r<0.2: n',lo.sum(),'gated comp',np.round(g[lo],1))
    print('   spearman comp~r %.2f comp~att %.2f comp~r*att %.2f'%(spearmanr(g,r).correlation,spearmanr(g,a).correlation,spearmanr(g,r*a).correlation))
    print('   T range', np.round(np.percentile(T,[0,50,100])), 'duty', {k:round(float(np.median(v)),3) for k,v in R['duty'].items() if k in('gated','replay_self','ungated')})
    nseeds=[len(x['r_ab']) for x in R['serial']['base']][:1]; nfin=np.mean([np.isfinite(x['r_ab']).sum() for x in R['serial']['base']])
    print('   seeds',nseeds,'finite r per cfg',nfin)
