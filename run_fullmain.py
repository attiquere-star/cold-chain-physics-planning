"""Main benchmark rerun with the full-horizon MILP (all integers kept integer), 42 test days."""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch; torch.set_num_threads(1)
from cc.config import cfg
from cc.generator import make_world
from cc.runner import *
seed = int(sys.argv[1]); meth = sys.argv[2]; OUT = sys.argv[3]
P = pickle.load(open(f'{OUT}/fc_{seed}.pkl', 'rb')); fc = P['fc']; th = P['theta']
c = cfg(INT_DAYS=7, TIME_LIMIT=30.0); w = make_world(c, seed)
spec = {'FTO-AGG': ('lstm', 'agg', None), 'DFL-AGG': ('lstm', 'agg', th['agg']),
        'FTO-CHAR': ('lstm', 'char', None), 'DFL-CHAR': ('lstm', 'char', th['char']),
        'ORACLE-AGG': ('oracle', 'agg', None), 'ORACLE-CHAR': ('oracle', 'char', None)}[meth]
out = run_policy(c, w, fc, spec[0], spec[1], TEST_START, TEST_DAYS, theta=spec[2])
print(seed, meth, round(out['total_cost']), round(out['solve_time'], 2), flush=True)
json.dump(out, open(f'{OUT}/fullmain_{seed}_{meth}.json', 'w'))
