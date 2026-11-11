"""Robustness: key contrasts with the full-horizon MILP instead of relax-and-fix."""
import sys, os, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch; torch.set_num_threads(1)
from cc.config import cfg
from cc.generator import make_world
from cc.runner import *
seed = int(sys.argv[1]); meth = sys.argv[2]; OUT = sys.argv[3]; ND = 14
P = pickle.load(open(f'{OUT}/fc_{seed}.pkl', 'rb')); fc = P['fc']; th = P['theta']
c = cfg(INT_DAYS=7, TIME_LIMIT=15.0); w = make_world(c, seed)
spec = {'FTO-AGG': ('lstm', 'agg', None), 'DFL-AGG': ('lstm', 'agg', th['agg']),
        'FTO-CHAR': ('lstm', 'char', None), 'DFL-CHAR': ('lstm', 'char', th['char']),
        'ORACLE-AGG': ('oracle', 'agg', None), 'ORACLE-CHAR': ('oracle', 'char', None)}[meth]
res = {}
for tag, cc_ in {'full': c, 'rf': cfg()}.items():
    out = run_policy(cc_, w, fc, spec[0], spec[1], TEST_START, ND, theta=spec[2])
    res[tag] = out; print(seed, meth, tag, round(out['total_cost']), round(out['solve_time'], 2), flush=True)
json.dump(res, open(f'{OUT}/full_{seed}_{meth}.json', 'w'))
