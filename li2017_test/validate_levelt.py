import numpy as np, time, json
import li2017 as L
def episodes(d, thr, dt, minlen=50):
    st=np.where(d>thr,1,np.where(d<-thr,-1,0))
    ch=np.flatnonzero(np.diff(st))+1; seg=np.split(st,ch)
    lens=[(int(s[0]),len(s)*dt) for s in seg if s[0]!=0]
    lens=lens[1:-1]; return [l for l in lens if l[1]>=minlen]
def run(in1, in2, seeds=6, T=120000, sig=0.05):
    p=L.default_params(1); p['T']=T
    in1=np.repeat(in1,seeds); in2=np.repeat(in2,seeds)
    N=len(in1)
    rb,_=L.simulate(p,N=N,seed=21,noise_sigma=sig,record_every=2,params_vec={'in1':in1,'in2':in2})
    dt=1.0; b=5000
    out=[]
    for i in range(N):
        x=rb[i,:,b:]; d=x[0]-x[1]; thr=0.05*x.mean()
        E=episodes(d,thr,dt)
        A=[l for s,l in E if s==1]; B=[l for s,l in E if s==-1]
        dom=(d>thr).sum(); domB=(d<-thr).sum()
        out.append(dict(in1=float(in1[i]),in2=float(in2[i]),mA=np.mean(A) if A else np.nan,mB=np.mean(B) if B else np.nan,
                        pred=dom/max(dom+domB,1), rate=len(E)/((x.shape[1])*dt/1000)))
    return out
t=time.time()
lv=np.array([0.3,0.4,0.45,0.5,0.55,0.6,0.7,0.8])
asym=run(lv, np.full(len(lv),0.5))
eq=run(np.array([0.35,0.45,0.5,0.6,0.7,0.8]),np.array([0.35,0.45,0.5,0.6,0.7,0.8]))
def agg(rows,key):
    from collections import defaultdict
    g=defaultdict(list)
    for r in rows: g[r[key]].append(r)
    return {k:{m:float(np.nanmean([r[m] for r in v])) for m in ('mA','mB','pred','rate')} for k,v in sorted(g.items())}
A=agg(asym,'in1'); E=agg(eq,'in1')
print('ASYMMETRIC (B fixed 0.5): in1, predominance A, mean dur A, mean dur B, alternations/s')
for k,v in A.items(): print(f"  {k:.2f}  {v['pred']:.2f}  {v['mA']:6.0f}  {v['mB']:6.0f}  {v['rate']:.2f}")
print('EQUAL STRENGTH: input, mean dur, alternations/s')
for k,v in E.items(): print(f"  {k:.2f}  {(v['mA']+v['mB'])/2:6.0f}  {v['rate']:.2f}")
json.dump(dict(asym=A,eq=E),open('levelt_validation.json','w'),indent=1)
print(f'{time.time()-t:.0f}s')
