"""Rough power of P1 under assumed noise (illustrative; numbers from pilot/test-6 summaries)."""
import numpy as np
from scipy.stats import rankdata
rng=np.random.default_rng(1)
def pw(groups):
    ra,rb=[],[]
    for a,b in groups:
        ra.append(rankdata(a)/(len(a)+1)); rb.append(rankdata(b)/(len(b)+1))
    return np.corrcoef(np.concatenate(ra),np.concatenate(rb))[0,1]
def boot(groups,nb=500):
    bs=[]
    for _ in range(nb):
        g2=[]
        for a,b in groups:
            i=rng.integers(0,len(a),len(a)); g2.append((a[i],b[i]))
        bs.append(pw(g2))
    return np.percentile(bs,2.5)
def sim(scen, r_sd_true=0.04, r_se=(0.09,0.05,0.05), comp_se=3.0, reps=200):
    hits=dict(pred=0,att=0,r=0); rhos=dict(pred=[],att=[],r=[])
    for _ in range(reps):
        G={k:[] for k in hits}
        for m in range(3):
            n=60
            r=rng.normal(0.06,r_sd_true,n); att=np.clip(rng.normal(40,15,n),2,None)
            if scen=='product': comp=0.6*r*att
            elif scen=='att':   comp=0.08*att
            else:               comp=np.zeros(n)+2
            rh=r+rng.normal(0,r_se[m],n); ah=att+rng.normal(0,3,n); ch=comp+rng.normal(0,comp_se,n)
            G['pred'].append((rh*ah,ch)); G['att'].append((ah,ch)); G['r'].append((rh,ch))
        for k in hits:
            rhos[k].append(pw(G[k])); hits[k]+= boot(G[k])>0
    return {k:(round(hits[k]/reps,2), round(float(np.median(rhos[k])),3)) for k in hits}
for scen in ('product','att','null'):
    print(scen, 'P(lower CI>0), median rho for predictors [product, att alone, r alone]:', sim(scen))
