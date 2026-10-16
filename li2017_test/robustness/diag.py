import json, sys, numpy as np
sys.path.insert(0, '..')
import run_li2017_test as R
acc = json.load(open('../li2017_test_results.json'))['accepted'][:6]
bp = R.L.default_params(1); dt = bp['dt']*2; seeds = list(range(8)); b = int(R.BURN_MS/dt)
rb0,_ = R.run_batch(acc, seeds, 'none', 0.0, base_p=bp)
thr = R.THETA_FRAC*rb0[:,:,b:].mean(axis=(1,2))
rb1,on1 = R.run_batch(acc, seeds, 'gated', 0.1, thr=thr, base_p=bp)
def frag(x, t):
    d = x[0]-x[1]; st = np.where(d>t,1,np.where(d<-t,-1,0))
    ch = np.flatnonzero(np.diff(st))+1; s=np.r_[0,ch]; e=np.r_[ch,len(st)]; v=st[s]; L=(e-s)*dt
    # separators inside B: sequences between two B runs (>=50ms) that contain no A run >=50ms
    sepA=sep0=0; Ablip_lens=[]
    lastB=None
    for i,(vv,ll) in enumerate(zip(v,L)):
        if vv==-1 and ll>=50:
            if lastB is not None:
                seg = list(zip(v[lastB+1:i], L[lastB+1:i]))
                if not any(a==1 and l>=50 for a,l in seg):
                    if any(a==1 for a,l in seg): sepA+=1; Ablip_lens += [l for a,l in seg if a==1]
                    else: sep0+=1
            lastB=i
        elif vv==1 and ll>=50: lastB=None
    nA_short = sum(1 for vv,ll in zip(v,L) if vv==1 and ll<50); nA = sum(1 for vv,ll in zip(v,L) if vv==1 and ll>=50)
    nAshortgt = sum(1 for vv,ll in zip(v,L) if vv==1 and 50<=ll<300)
    return sepA, sep0, nA_short, nA, nAshortgt
for name, rb in (('base', rb0), ('gated', rb1)):
    tot = np.array([frag(rb[i,:,b:].astype(float), thr[i]) for i in range(rb.shape[0])]).sum(0)
    print(name, 'B split by short A blips', tot[0], '| split by indeterminate only', tot[1], '| A runs <50ms', tot[2], '| A runs >=50ms', tot[3], '| A runs 50-300ms', tot[4])
