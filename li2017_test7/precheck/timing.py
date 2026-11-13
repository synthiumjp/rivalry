import sys, time, json
sys.path.insert(0,'/home/claude/chris/work/t7/li2017_test7')
import numpy as np, run_li2017_test7 as T7
src='/home/claude/chris/work/t7/src6/'
cfg=dict(li=json.load(open(src+'pilot7b_li.json'))[0]['cfg'], sfa=json.load(open(src+'pilot7b_sfa.json'))[0]['cfg'], dep=json.load(open(src+'pilot7c_dep.json'))[0]['cfg'])
for model in ('li','sfa','dep'):
    for nc,ns in ((1,4),(10,8),(10,32)):
        t0=time.time()
        T7.run_batch7(model,[cfg[model]]*nc,list(range(ns)),'none',0.0,None,'timing',t_ms=2000.0,ranges=T7.RANGES[model])
        dt=time.time()-t0
        print(model,'N=%d'%(nc*ns),'%.1f us/step'%(dt/4000*1e6),' -> 120 s run: %.0f s'%(dt/4000*240000),flush=True)
