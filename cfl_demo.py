"""Numerical behaviour of the first-order upwind transport operator on the long-haul route
(L=900 km, v=480 km/day, dt=0.25 day; exact plug-flow delay 7.5 steps) for several Courant numbers."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cc.planner import route_operator
L, v, DT = 900.0, 480.0, 0.25
out = []
for J in [3, 4, 5, 6, 7, 8, 9, 10, 12]:
    a = v * DT / (L / J)
    M, o, _ = route_operator(v, 0.0, L, DT, J, 1)
    F = np.zeros(J); arr = []; minF = 0.0; maxmass = 0.0
    for k in range(60):
        x = F + (1000.0 if k == 0 else 0.0) * np.eye(J)[0]
        arr.append(o @ x); F = M @ x; minF = min(minF, F.min()); maxmass = max(maxmass, np.abs(F).sum())
    arr = np.array(arr); t = np.arange(60)
    mean = (arr * t).sum() / arr.sum() if abs(arr.sum()) > 1e-9 else np.nan
    sd = np.sqrt((arr * (t - mean) ** 2).sum() / arr.sum()) if abs(arr.sum()) > 1e-9 else np.nan
    early = arr[:7].sum() / 1000.0   # share believed to arrive before the true arrival step (7)
    out.append(dict(J=J, courant=a, mean_arrival_step=mean, sd_steps=sd, min_density=minF, max_abs_mass=maxmass,
                    total_arrived=arr.sum(), share_early=early, minarr=arr.min(), arrivals=arr[:20].tolist()))
    print(J, round(a, 2), 'mean', round(mean, 2), 'sd', round(sd, 2), 'minF', round(minF, 1), 'maxmass', round(maxmass), 'early', round(early, 3), 'minarr', round(arr.min(), 1))
json.dump(out, open(sys.argv[1] + '/cfl_demo.json', 'w'))
