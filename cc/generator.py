"""Synthetic but structured world: demand, promotions, weather, cold-room and
reefer temperatures, and route velocities. One call = one replication."""
import numpy as np

DOW_DEM = np.array([0.90, 0.92, 0.95, 1.00, 1.10, 1.20, 1.00])
DOW_VEL = np.array([0.92, 0.95, 0.95, 0.95, 0.90, 1.05, 1.08])

def ar1(rng, n, phi, sd):
    x = np.zeros(n); e = rng.normal(0, sd, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x

def make_world(c, seed, n_days=800, test_start=730, scenario='normal'):
    rng = np.random.default_rng(seed)
    C, P = c['C'], c['P']
    d = np.arange(n_days); dow = d % 7
    # weather
    t_amb = 18 + 10 * np.sin(2 * np.pi * (d - 100) / 365) + ar1(rng, n_days, 0.7, 1.8)
    heat = rng.random(n_days) < 0.02
    for i in np.where(heat)[0]:
        t_amb[i:i + 3] += 6
    h = np.arange(7)
    t_fc = np.zeros((n_days, 7))
    for i in range(n_days):
        idx = np.minimum(i + h, n_days - 1)
        t_fc[i] = t_amb[idx] + rng.normal(0, 0.8 + 0.3 * h)
    t_dc = c['T_SET'] + 0.05 * (t_amb - c['T_SET']) + rng.normal(0, 0.3, n_days)
    t_dc += np.where(rng.random(n_days) < 0.02, 5.0, 0.0)
    t_rf = c['T_SET'] + 0.12 * (t_amb - c['T_SET']) + rng.normal(0, 0.5, n_days)
    t_rf += np.where(rng.random(n_days) < 0.03, 6.0, 0.0)
    # velocities: route 0 long haul, routes 1..C regional
    v = np.zeros((n_days, C + 1))
    for e in range(C + 1):
        base = c['V0'] if e == 0 else c['VC']
        dis = np.where(rng.random(n_days) < 0.03, 0.55, 1.0)
        v[:, e] = base * DOW_VEL[dow] * np.exp(ar1(rng, n_days, 0.5, 0.06)) * dis
    # demand
    mu = np.array(c['MU'], float)
    promo = (rng.random((n_days, C, P)) < 0.06).astype(float)
    season = 1 + 0.12 * np.sin(2 * np.pi * (d - 150) / 365)
    trend = 1 + 0.05 * d / 365
    dem = np.zeros((n_days, C, P))
    for p in range(P):
        temp_eff = 1 + (0.012 * (t_amb - 18) if p == 0 else 0.0)
        for cc in range(C):
            noise = np.exp(rng.normal(0, 0.15, n_days) - 0.15 ** 2 / 2)
            dem[:, cc, p] = (mu[cc, p] * DOW_DEM[dow] * season * trend * temp_eff
                             * (1 + 0.45 * promo[:, cc, p]) * noise)
    w = dict(n_days=n_days, test_start=test_start, dow=dow, t_amb=t_amb, t_fc=t_fc,
             t_dc=t_dc, t_rf=t_rf, v=v, dem=dem, promo=promo,
             k_mult=np.ones(n_days), sup_cap_mult=np.ones((n_days, c['S'])))
    apply_scenario(w, scenario, rng)
    return w

SCENARIOS = ['normal', 'high_demand', 'supplier_disruption', 'transport_delay',
             'temperature_excursion', 'high_spoilage', 'limited_vehicles', 'high_carbon']

def apply_scenario(w, s, rng):
    ts = w['test_start']; sl = slice(ts, None)
    if s == 'high_demand':
        w['dem'][sl] *= 1.3
    elif s == 'supplier_disruption':
        w['sup_cap_mult'][ts + 5: ts + 19, 0] = 0.0   # cheapest supplier out for 14 days
        w['sup_cap_mult'][ts + 5: ts + 19, 1] = 0.5
    elif s == 'transport_delay':
        w['v'][sl] *= 0.7
    elif s == 'temperature_excursion':
        w['t_amb'][sl] += 8; w['t_fc'][sl] += 8
        w['t_dc'][sl] += 0.05 * 8 + 2.0; w['t_rf'][sl] += 0.12 * 8 + 3.0
    elif s == 'high_spoilage':
        w['k_mult'][sl] = 1.5
    # limited_vehicles and high_carbon are configuration changes (see run scripts)
