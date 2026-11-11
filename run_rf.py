"""Closed-loop check: relax-and-fix (integer first day) vs full-horizon MILP, oracle inputs."""
import sys, os, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cc.config import cfg
from cc.generator import make_world
from cc.runner import *
seed = int(sys.argv[1]); OUT = sys.argv[2]; res = {}
for tag, over in {'relax_and_fix': {}, 'full_milp': dict(INT_DAYS=7, TIME_LIMIT=30.0, MIP_GAP=0.005)}.items():
    c = cfg(**over); w = make_world(c, seed)
    out = run_policy(c, w, None, 'oracle', 'char', TEST_START, 14)
    res[tag] = out; print(seed, tag, round(out['total_cost']), out['solve_time'], flush=True)
json.dump(res, open(f'{OUT}/rf_{seed}.json', 'w'))
