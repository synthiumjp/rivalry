# Feasibility only (baseline, no increment): does Li et al.'s model alternate without adaptation (w_h = 0)?
import numpy as np, run_li2017_test as R
bp = R.L.default_params(1); bp['T'] = 60000.0; dt = bp['dt']*2
rng = np.random.default_rng(777)
cfgs=[]
for noise in (0.05, 0.1, 0.15, 0.2, 0.3):
    for wa in (0.6, 0.9):
        for inp in (0.5, 0.8):
            cfgs.append(dict(inp=inp, wo=0.65, wa=wa, wh=0.0, tau_h=2000.0, noise=noise))
N=len(cfgs); pv=dict(in1=np.array([c['inp'] for c in cfgs]), in2=np.array([c['inp'] for c in cfgs]), wo=np.full(N,0.65), wa=np.array([c['wa'] for c in cfgs]), wh=np.zeros(N), tau_h=np.full(N,2000.))
rb,_=R.L.simulate(bp, N=N, seed=5, noise_sigma=1.0, record_every=2, params_vec=pv, noise_scale=np.array([c['noise'] for c in cfgs]))
b=int(5000/dt)
for i,c in enumerate(cfgs):
    x=rb[i,:,b:].astype(float); thr=0.05*x.mean()
    E=R.episodes_diff(x,thr,dt); d=[l for _,l in E]
    print(c['noise'],c['wa'],c['inp'],'episodes',len(d),'mean %.0f'%(np.mean(d) if d else 0),'cv %.2f'%(np.std(d)/np.mean(d) if len(d)>2 else np.nan))
