import numpy as np, time, li2017 as L
def eps_all(d, thr, dt, minlen=50):
    N=d.shape[0]; res=[]
    for i in range(N):
        st=np.where(d[i]>thr[i],1,np.where(d[i]<-thr[i],-1,0))
        ch=np.flatnonzero(np.diff(st))+1; seg=np.split(st,ch)
        lens=[(s[0],len(s)*dt) for s in seg if s[0]!=0]
        lens=lens[1:-1]; lens=[l for l in lens if l[1]>=minlen]
        res.append(lens)
    return res
p=L.default_params(1); p['T']=120000
for sig in (0.0,0.02,0.05,0.1):
    t=time.time()
    rb,_=L.simulate(p,N=8,seed=11,noise_sigma=sig,record_every=2)
    dt=1.0; b=int(5000/dt)
    d=(rb[:,0,b:]-rb[:,1,b:]); thr=0.05*rb[:,:,b:].mean(axis=(1,2))
    E=eps_all(d,thr,dt)
    durs=np.concatenate([[l for _,l in e] for e in E]) if E else []
    cvs=[np.std([l for _,l in e])/np.mean([l for _,l in e]) for e in E if len(e)>3]
    print(f"sigma {sig}: episodes/run {np.mean([len(e) for e in E]):.1f}  mean dur {np.mean(durs):.0f} ms  median CV {np.median(cvs):.2f}  ({time.time()-t:.0f}s)")
