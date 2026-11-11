"""Forecast preparation, plan-parameter construction, rolling-horizon evaluation
and decision-focused recalibration."""
import time
import numpy as np
from scipy.optimize import minimize
from .config import loss_rate
from .forecast import LSTMForecaster, dem_series, H
from .simulator import ColdChainSim
from .planner import build_and_solve

TRAIN_END, TEST_START, TEST_DAYS, WARM = 610, 730, 42, 5
DFL_WINDOWS = [(650, 14), (690, 14)]

def prepare_forecasts(c, w, seed, epochs=40):
    """Fit LSTMs on days < TRAIN_END; return forecast arrays for all origins in [TRAIN_END, n-H]."""
    C, P = c['C'], c['P']; n = w['n_days']
    y, promo = dem_series(w)
    origins = np.arange(TRAIN_END, n - H + 1)
    fd = LSTMForecaster(epochs, seed).fit(y, promo, w['t_amb'], w['t_fc'], TRAIN_END)
    Fd = fd.predict(y, promo, w['t_amb'], w['t_fc'], origins)
    vz = np.zeros_like(w['v'])
    fv = LSTMForecaster(epochs, seed + 1).fit(w['v'], vz, w['t_amb'], w['t_fc'], TRAIN_END)
    Fv = fv.predict(w['v'], vz, w['t_amb'], w['t_fc'], origins)
    fc = dict(origin0=TRAIN_END, dem=Fd.reshape(len(origins), H, C, P), vel=Fv, models=(fd, fv))
    # static (historical) inputs
    dow = w['dow'][:TRAIN_END]
    fc['static_dem'] = np.stack([w['dem'][:TRAIN_END][dow == k].mean(0) for k in range(7)])
    fc['static_vel'] = w['v'][:TRAIN_END].mean(0); fc['static_tamb'] = w['t_amb'][:TRAIN_END].mean()
    return fc

def plan_params(c, w, fc, d, source, theta=None):
    SPD, K = c['SPD'], c['H_DAYS'] * c['SPD']; P = c['P']
    hd = np.arange(K) // SPD
    if source == 'oracle':
        dem = w['dem'][d + hd] / SPD; vel = w['v'][d + hd]
        km = w['k_mult'][d + hd][:, None]
        lam = np.stack([loss_rate(c, p, w['t_dc'][d + hd]) for p in range(P)], 1) * km
        xi = np.stack([loss_rate(c, p, w['t_rf'][d + hd]) for p in range(P)], 1) * km
        return dict(dem=dem, vel=vel, lam=lam, xi=xi, tamb=w['t_amb'][d + hd],
                    supcap=w['sup_cap_mult'][d:d + c['H_DAYS']])
    if source == 'static':
        dem = fc['static_dem'][(d + hd) % 7] / SPD
        vel = np.repeat(fc['static_vel'][None], K, 0); tamb = np.full(K, fc['static_tamb'])
    else:
        i = d - fc['origin0']
        dem = fc['dem'][i][hd] / SPD; vel = fc['vel'][i][hd]; tamb = w['t_fc'][d][hd]
    tdc = c['T_SET'] + 0.05 * (tamb - c['T_SET']); trf = c['T_SET'] + 0.12 * (tamb - c['T_SET'])
    lam = np.stack([loss_rate(c, p, tdc) for p in range(P)], 1)
    xi = np.stack([loss_rate(c, p, trf) for p in range(P)], 1)
    if theta is not None:
        dem = dem * np.asarray(theta[:P])[None, None, :]
        vel = vel * theta[P]; lam = lam * theta[P + 1]; xi = xi * theta[P + 1]
    supcap = np.repeat(w['sup_cap_mult'][d][None], c['H_DAYS'], 0)
    return dict(dem=dem, vel=vel, lam=lam, xi=xi, tamb=tamb, supcap=supcap)

def run_policy(c, w, fc, source, mode, start, ndays, theta=None, warm=WARM, grid_factor=1, pp_extra=None):
    sim = ColdChainSim(c, w, start - warm)
    infos = []; t0 = time.time(); fails = 0
    for d in range(start - warm, start + ndays):
        if d == start:
            sim.reset_accounts()
        pp = plan_params(c, w, fc, d, source, theta)
        if pp_extra: pp.update(pp_extra)
        dec, info = build_and_solve(c, sim, pp, mode, grid_factor)
        if dec is None:          # infeasible / failed plan: fall back to do-nothing for the day
            fails += 1
            dec = [dict(q=np.zeros(c['S']), A=np.zeros(c['S']), X=np.zeros(c['P']), N0=0,
                        z=np.zeros((c['C'], c['P'])), Nc=np.zeros(c['C'], int))] * c['SPD']
        if d >= start:
            infos.append(info)
        for k in range(c['SPD']):
            sim.step(dec[k])
    out = dict(cost=dict(sim.cost), kpi=dict(sim.kpi), fails=fails, wall=time.time() - t0,
               solve_time=float(np.mean([i['time'] for i in infos])) if infos else np.nan,
               solve_time_max=float(np.max([i['time'] for i in infos])) if infos else np.nan,
               gap=float(np.nanmean([i['gap'] if i['gap'] is not None else np.nan for i in infos])) if infos else np.nan,
               alpha0_max=float(np.max([i.get('alpha0_max', np.nan) for i in infos])) if infos else np.nan,
               fallbacks=int(sum(i.get('fallback', 0) for i in infos)))
    out['total_cost'] = sum(out['cost'].values())
    return out

def dfl_tune(c, w, fc, mode, maxfev=30):
    """Decision-focused recalibration: choose theta to minimise realised cost
    (equivalently regret) of the planning MILP on validation windows."""
    P = c['P']; hist = []
    def f(th):
        th = np.clip(th, 0.6, 1.6)
        tot = sum(run_policy(c, w, fc, 'lstm', mode, s, n, theta=th, warm=3)['total_cost'] for s, n in DFL_WINDOWS)
        hist.append((th.copy(), tot)); return tot
    x0 = np.ones(P + 2)
    res = minimize(f, x0, method='Nelder-Mead',
                   options=dict(maxfev=maxfev, initial_simplex=np.vstack([x0] + [x0 + 0.1 * np.eye(P + 2)[i] for i in range(P + 2)]),
                                xatol=0.01, fatol=1.0))
    best = min(hist, key=lambda t: t[1])
    return np.clip(best[0], 0.6, 1.6), hist
