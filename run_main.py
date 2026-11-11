"""Main experiment, one replication (seed) per call: forecasts, forecast accuracy,
decision-focused recalibration, and closed-loop evaluation of all methods."""
import sys, os, json, pickle, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch; torch.set_num_threads(1)
from cc.config import cfg
from cc.generator import make_world
from cc.forecast import dem_series, RidgeForecaster, snaive, H
from cc.runner import *
import cc.runner as R
FAST = os.environ.get('FAST') == '1'
if FAST:
    R.DFL_WINDOWS = [(650, 2)]
TEST_DAYS = 3 if FAST else TEST_DAYS

seed = int(sys.argv[1]); OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
c = cfg(); w = make_world(c, seed)
t0 = time.time()
fc = prepare_forecasts(c, w, seed, epochs=3 if FAST else 40)
res = dict(seed=seed)
# ---- forecast accuracy on test origins (demand)
y, promo = dem_series(w)
orig = np.arange(TEST_START, TEST_START + TEST_DAYS)
act = np.stack([y[d:d + H] for d in orig])
i0 = TEST_START - fc['origin0']
C, P = c['C'], c['P']
lstm = fc['dem'][i0:i0 + TEST_DAYS].reshape(TEST_DAYS, H, C * P)
rf = RidgeForecaster().fit(y, promo, w['t_amb'], w['t_fc'], TRAIN_END).predict(y, promo, w['t_amb'], w['t_fc'], orig)
sn = snaive(y, orig)
def metr(F, A):
    e = F - A
    return dict(RMSE=float(np.sqrt((e ** 2).mean())), MAE=float(np.abs(e).mean()),
                MAPE=float((np.abs(e) / A).mean() * 100), SMAPE=float((2 * np.abs(e) / (np.abs(F) + A)).mean() * 100),
                bias=float(e.mean() / A.mean() * 100))
res['acc'] = {'SNaive': metr(sn, act), 'Ridge': metr(rf, act), 'LSTM': metr(lstm, act)}
# 90% PI from validation multiplicative residuals of the LSTM
vorig = np.arange(TRAIN_END, TEST_START - H)
vf = fc['dem'][:len(vorig)].reshape(len(vorig), H, C * P); va = np.stack([y[d:d + H] for d in vorig])
r = va / np.maximum(vf, 1e-6); lo, hi = np.quantile(r, 0.05, axis=(0, 2)), np.quantile(r, 0.95, axis=(0, 2))
cov = ((act >= lstm * lo[None, :, None]) & (act <= lstm * hi[None, :, None])).mean()
res['acc']['LSTM']['PI90_coverage'] = float(cov * 100)
vact = np.stack([w['v'][d:d + H] for d in orig]); vF = fc['vel'][i0:i0 + TEST_DAYS]
res['acc_vel'] = metr(vF, vact)
res['t_forecast'] = time.time() - t0
# ---- decision-focused recalibration
th = {}
for mode in ['agg', 'char']:
    t = time.time(); theta, hist = dfl_tune(c, w, fc, mode, maxfev=3 if FAST else 25)
    th[mode] = theta.tolist(); res['dfl_%s_time' % mode] = time.time() - t
    res['dfl_%s_hist' % mode] = [(h[0].tolist(), h[1]) for h in hist]
res['theta'] = th
# ---- evaluation
methods = {
    'STATIC-AGG': ('static', 'agg', None, 1), 'FTO-AGG': ('lstm', 'agg', None, 1), 'DFL-AGG': ('lstm', 'agg', th['agg'], 1),
    'FTO-UPW': ('lstm', 'phy', None, 1), 'FTO-UPW-VIOL': ('lstm', 'phy', None, 2),
    'FTO-CHAR': ('lstm', 'char', None, 1), 'DFL-CHAR': ('lstm', 'char', th['char'], 1),
    'ORACLE-AGG': ('oracle', 'agg', None, 1), 'ORACLE-CHAR': ('oracle', 'char', None, 1)}
res['eval'] = {}
for name, (src, mode, theta, gf) in methods.items():
    t = time.time()
    out = run_policy(c, w, fc, src, mode, TEST_START, TEST_DAYS, theta=theta, grid_factor=gf)
    out['time'] = time.time() - t
    res['eval'][name] = out
    print(seed, name, round(out['total_cost']), round(out['time']), flush=True)
fc.pop('models', None)
fdump = dict(fc); pickle.dump(dict(fc=fdump, theta=th), open(f'{OUT}/fc_{seed}.pkl', 'wb'))
json.dump(res, open(f'{OUT}/main_{seed}.json', 'w'))
print('done', seed, round(time.time() - t0), flush=True)
