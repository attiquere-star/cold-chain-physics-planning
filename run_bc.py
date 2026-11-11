"""Bias-correction baseline (reviewer-style check): scale LSTM forecasts by the
actual/forecast ratio measured on the same validation windows used for DFL, then
plan with FTO. Separates 'removing average bias' from 'tuning on decision cost'."""
import sys, os, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch; torch.set_num_threads(1)
from cc.config import cfg
from cc.generator import make_world
from cc.forecast import H
from cc.runner import *
import cc.runner as R
seed = int(sys.argv[1]); OUT = sys.argv[2]
c = cfg(); w = make_world(c, seed)
fc = pickle.load(open(f'{OUT}/fc_{seed}.pkl', 'rb'))['fc']
P = c['P']
orig = np.concatenate([np.arange(s, s + n) for s, n in R.DFL_WINDOWS])
i = orig - fc['origin0']
Fd = fc['dem'][i]                                       # (n, H, C, P)
Ad = np.stack([w['dem'][d:d + H] for d in orig])        # (n, H, C, P)
dem_mult = Ad.sum((0, 1, 2)) / Fd.sum((0, 1, 2))
Fv = fc['vel'][i]; Av = np.stack([w['v'][d:d + H] for d in orig])
vel_mult = float(Av.sum() / Fv.sum())
theta = np.r_[dem_mult, vel_mult, 1.0]
res = dict(seed=seed, theta_bc=theta.tolist(), eval={})
for name, mode in [('BC-AGG', 'agg'), ('BC-CHAR', 'char')]:
    out = run_policy(c, w, fc, 'lstm', mode, TEST_START, TEST_DAYS, theta=theta)
    res['eval'][name] = out; print(seed, name, round(out['total_cost']), flush=True)
json.dump(res, open(f'{OUT}/bc_{seed}.json', 'w'))
