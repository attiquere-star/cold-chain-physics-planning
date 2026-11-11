"""One-way sensitivity: shelf life and long-haul distance. When does physics pay?"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch; torch.set_num_threads(1)
from cc.config import cfg
from cc.generator import make_world
from cc.runner import *
seed = int(sys.argv[1]); OUT = sys.argv[2]; ND = 21
settings = {'SL x0.5': dict(SL_DAYS=[3.5, 6.0]), 'SL x1.5': dict(SL_DAYS=[10.5, 18.0]),
            'L0 450km': dict(L0=450.0), 'L0 1350km': dict(L0=1350.0), 'base': {}}
res = {}
for nm, over in settings.items():
    c = cfg(**over); w = make_world(c, seed)
    fc = prepare_forecasts(c, w, seed, epochs=40)
    res[nm] = {}
    for name, (src, mode) in {'FTO-AGG': ('lstm', 'agg'), 'FTO-CHAR': ('lstm', 'char'),
                              'ORACLE-AGG': ('oracle', 'agg'), 'ORACLE-CHAR': ('oracle', 'char')}.items():
        out = run_policy(c, w, fc, src, mode, TEST_START, ND)
        res[nm][name] = out; print(seed, nm, name, round(out['total_cost']), flush=True)
json.dump(res, open(f'{OUT}/sens_{seed}.json', 'w'))
