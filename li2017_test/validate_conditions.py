import numpy as np, time, sys
sys.path.insert(0,'.')
import li2017 as L
def episodes(d, thr, dt):
    st=np.where(d>thr,1,np.where(d<-thr,-1,0))
    out={1:[],-1:[]}; cur=0; ln=0
    for v in st:
        if v!=cur:
            if cur!=0: out[cur].append(ln*dt)
            cur=v; ln=1
        else: ln+=1
    for k in out: out[k]=out[k][1:]  # drop first
    return out
for cond in (1,2,3,4):
    p=L.default_params(cond); p['T']=30000
    t=time.time(); rb,_=L.simulate(p,N=1,seed=1,record_every=2)
    dt=p['dt']*2
    d=rb[0,0,:]-rb[0,1,:]
    ep=episodes(d[int(2000/dt):],0.05*rb[0].mean(),dt)
    print(f"cond {cond}: rb mean {rb[0].mean():.3f}  max|d| {np.abs(d[int(2000/dt):]).max():.3f}  episodes A {len(ep[1])} B {len(ep[-1])}  meanA {np.mean(ep[1]) if ep[1] else float('nan'):.0f} ms  meanB {np.mean(ep[-1]) if ep[-1] else float('nan'):.0f} ms  ({time.time()-t:.1f}s)")
