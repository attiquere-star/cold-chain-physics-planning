"""Stress scenarios from the thesis proposal (test period only); thetas from main run."""
import sys, os, json, pickle, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch; torch.set_num_threads(1)
from cc.config import cfg
from cc.generator import make_world, SCENARIOS
from cc.runner import *
seed = int(sys.argv[1]); OUT = sys.argv[2]; ND = 21
th = pickle.load(open(f'{OUT}/fc_{seed}.pkl', 'rb'))['theta']
res = {}
for sc in SCENARIOS[1:]:
    over = {}
    if sc == 'limited_vehicles': over = dict(NMAX0=0.25, NMAXC=0.25)
    if sc == 'high_carbon': over = dict(CARBON=0.30)
    c = cfg(**over); w = make_world(c, seed, scenario=sc)
    fc = prepare_forecasts(c, w, seed, epochs=40)
    res[sc] = {}
    for name, (src, mode, theta) in {'STATIC-AGG': ('static', 'agg', None), 'FTO-AGG': ('lstm', 'agg', None),
                                     'DFL-AGG': ('lstm', 'agg', th['agg']), 'FTO-CHAR': ('lstm', 'char', None),
                                     'DFL-CHAR': ('lstm', 'char', th['char']), 'ORACLE-CHAR': ('oracle', 'char', None)}.items():
        out = run_policy(c, w, fc, src, mode, TEST_START, ND, theta=theta)
        res[sc][name] = out; print(seed, sc, name, round(out['total_cost']), flush=True)
json.dump(res, open(f'{OUT}/scen_{seed}.json', 'w'))
