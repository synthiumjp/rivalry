"""Pilot, part 2 (exploratory): window profile, gated and continuous delivery at 5%, 10% and 20% for the
same 10 configurations, 16 seeds, with the third test's code (run_chunk), to ask whether the profile
approaches the linear-response kernel as the increment shrinks."""
import sys, json, time
sys.path.insert(0, '../../li2017_test3')
import run_li2017_test3 as R3
R3.N_SEEDS = 16
from concurrent.futures import ProcessPoolExecutor
FR = (0.05, 0.1, 0.2)
CONDS = [(m, f, x) for f in FR for (m, x) in ([('win', w) for w in R3.WINDOWS] + [('late', None), ('gated', None), ('ungated', None)])]

def job(sub):
    return R3.run_chunk(sub, CONDS, 1.0, 120000.0, 'amp_pilot', R3.RANGES)

if __name__ == '__main__':
    D = json.load(open('../../li2017_test3/li2017_test3_results.json'))
    cfgs = D['accepted'][:10]
    t0 = time.time(); res = []
    with ProcessPoolExecutor(max_workers=2) as ex:
        for r in ex.map(job, [cfgs[:5], cfgs[5:]]):
            res.append(r); print('chunk done', time.time() - t0, flush=True)
    base = res[0][0] + res[1][0]; cond = {k: res[0][1][k] + res[1][1][k] for k in res[0][1]}
    json.dump(dict(base=base, cond=cond, T=res[0][2] + res[1][2], dL=res[0][3] + res[1][3]), open('amp_pilot.json', 'w'), default=float)
    print('done', time.time() - t0)
