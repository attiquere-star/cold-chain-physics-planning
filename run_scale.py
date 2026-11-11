"""Scalability: model size and solve time vs customers and horizon; relax-and-fix vs full MILP."""
import sys, os, json, time, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch; torch.set_num_threads(1)
from cc.config import cfg
from cc.generator import make_world
from cc.runner import *
from cc.simulator import ColdChainSim
from cc.planner import build_and_solve
OUT = sys.argv[1]; res = []
for C in [4, 8, 16]:
    LC = list(np.resize([150.0, 250.0, 350.0, 450.0], C)); MU = [list(x) for x in np.resize(np.array([[120, 80], [90, 70], [150, 60], [70, 90]]), (C, 2)) * 4 / C]
    for Hd in [5, 7, 10]:
        c = cfg(C=C, LC=LC, MU=MU, H_DAYS=Hd, CAP0=1000.0 * max(1, C // 4) if False else 1000.0)
        w = make_world(c, 1, n_days=800)
        for d in [732, 737, 742]:
            sim = ColdChainSim(c, w, d)
            pp = plan_params(c, w, None, d, 'oracle')
            for mode in ['agg', 'char']:
                for ni, tl in [(1, 60.0), (Hd, 60.0)]:
                    cc_ = dict(c); cc_['INT_DAYS'] = ni; cc_['TIME_LIMIT'] = tl; cc_['MIP_GAP'] = 0.005
                    dec, info = build_and_solve(cc_, sim, pp, mode)
                    r = dict(C=C, H=Hd, day=d, mode=mode, int_days=ni, n=info['n'], m=info['m'], nint=info['nint'],
                             time=info['time'], gap=info['gap'], obj=info['obj'], status=info['status'], fallback=info.get('fallback'))
                    res.append(r); print(r, flush=True)
json.dump(res, open(f'{OUT}/scale.json', 'w'), default=float)
