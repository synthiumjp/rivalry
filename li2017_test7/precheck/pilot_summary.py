import json, numpy as np, sys
src='/home/claude/chris/work/t7/src6/'
for f in ['pilot7_li','pilot7b_li','pilot7_sfa','pilot7b_sfa','pilot7_dep','pilot7b_dep','pilot7c_dep']:
    d=json.load(open(src+f+'.json'))
    model=f.split('_')[1]
    n=len(d)
    ok=[x for x in d if x['n_ep']>=10 and x['cv']==x['cv'] and 0.3<=x['cv']<=0.8 and (model=='li' or x['T']>=500)]
    acc=[x for x in ok if x['r_ab']<0.1]
    acc7=[x for x in acc if x['cv']<=0.7]
    print(f, 'n',n,'pass_basic',len(ok),'acc',len(acc),'acc_cv<=.7',len(acc7))
    if acc:
        r=np.array([x['r_ab'] for x in acc]); T=np.array([x['T'] for x in acc]); cv=np.array([x['cv'] for x in acc]); ne=np.array([x['n_ep'] for x in acc])
        print('   r_ab acc: ', np.round(np.percentile(r,[0,25,50,75,100]),3))
        print('   T acc ms: ', np.round(np.percentile(T,[0,25,50,75,100])))
        print('   cv acc:   ', np.round(np.percentile(cv,[0,25,50,75,100]),2))
        print('   n_ep acc: ', np.round(np.percentile(ne,[0,25,50,75,100])))
    if ok:
        r=np.array([x['r_ab'] for x in ok]); cv=np.array([x['cv'] for x in ok]);
        print('   basic-pass r_ab:', np.round(np.percentile(r,[10,50,90]),3), ' corr(cv,r)=%.2f'%np.corrcoef(cv,r)[0,1])
