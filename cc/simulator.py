"""'True' system: lot-based cold chain with plug-flow transport at realised
velocities, temperature-dependent first-order loss, shelf-life expiry, FEFO
picking, backlog with abandonment. Decisions come from a planner, one step at a time."""
import numpy as np
from .config import loss_rate

class ColdChainSim:
    def __init__(self, c, w, start_day):
        self.c, self.w = c, w
        C, P = c['C'], c['P']
        self.k = start_day * c['SPD']
        self.R = 2500.0; self.raw_pending = 0.0
        mu = np.array(c['MU'], float)
        tot = mu.sum(0)
        self.lh = []   # [p, qty, pos, age]
        for p in range(P):           # pipeline: about 1.5 days of supply spread on route
            for j in range(6):
                self.lh.append([p, tot[p] * 0.25 * 1.0, c['L0'] * (j + 0.5) / 6, 0.3 * j])
        self.dc = []   # [p, qty, age]
        for p in range(P):
            for a in range(4):
                self.dc.append([p, tot[p] * 0.5, 2.0 + a])
        self.rg = []   # [c, p, qty, pos, age]
        self.S = mu * 0.4; self.B = np.zeros((C, P))
        self.cost = {}; self.kpi = {}
        self.history = []; self.cap0 = 0.0; self.capc = np.zeros(C)

    def day(self):
        return self.k // self.c['SPD']

    def _add(self, d, key, val):
        d[key] = d.get(key, 0.0) + float(val)

    def step(self, dec):
        c, w = self.c, self.w; DT = c['DT']; day = self.day(); C, P = c['C'], c['P']
        cost, kpi = {}, {}
        # 3. decisions (raw material ordered last step becomes available now)
        self.R += self.raw_pending; self.raw_pending = 0.0
        if self.k % c['SPD'] == 0:
            self._add(cost, 'fixed_supplier', np.dot(c['F_SUP'], dec['A']))
        capm = w['sup_cap_mult'][day]
        q = np.minimum(np.maximum(dec['q'], 0), np.array(c['CAP_SUP']) * capm * dec['A'])
        self.raw_pending = q.sum()
        self._add(cost, 'procurement', np.dot(c['C_SUP'], q))
        if self.k % c['SPD'] == 0:      # daily truck commitment: cost now, capacity budget for the day
            self.cap0 = c['CAP0'] * dec['N0']; self.capc = c['CAPC'] * np.asarray(dec['Nc'], float)
            self._add(cost, 'transport', c['C_TRUCK0'] * dec['N0'])
            self._add(kpi, 'emis_truck', dec['N0'] * c['L0'] * c['EF0'])
            self._add(kpi, 'trucks', dec['N0'])
            for cc in range(C):
                self._add(cost, 'transport', (c['C_TRUCKC_FIX'] + c['C_TRUCKC_KM'] * c['LC'][cc]) * dec['Nc'][cc])
                self._add(kpi, 'emis_truck', dec['Nc'][cc] * c['LC'][cc] * c['EFC'])
                self._add(kpi, 'trucks', dec['Nc'][cc])
        X = np.maximum(dec['X'], 0)
        lim = min(self.R, c['PROD_CAP'], self.cap0)
        if X.sum() > lim + 1e-9:
            X = X * (lim / X.sum())
        self.R -= X.sum(); self.cap0 -= X.sum()
        for p in range(P):
            if X[p] > 1e-6:
                self.lh.append([p, X[p], 0.0, 0.0])
        self._add(cost, 'production', c['C_PROD'] * X.sum())
        z = np.maximum(dec['z'], 0).copy()
        for cc in range(C):
            capv = max(self.capc[cc], 0.0)
            if z[cc].sum() > capv + 1e-9:
                z[cc] *= capv / z[cc].sum()
        for p in range(P):
            lots = sorted([l for l in self.dc if l[0] == p], key=lambda l: -l[2])  # oldest first
            avail = sum(l[1] for l in lots); need = z[:, p].sum()
            if need > avail + 1e-9:
                z[:, p] *= avail / need if need > 0 else 0; need = avail
            taken, age_w = 0.0, 0.0
            for l in lots:
                if taken >= need - 1e-9: break
                t = min(l[1], need - taken); l[1] -= t; taken += t; age_w += t * l[2]
            a_bar = age_w / taken if taken > 0 else 0.0
            for cc in range(C):
                if z[cc, p] > 1e-6:
                    self.rg.append([cc, p, z[cc, p], 0.0, a_bar])
            self.dc = [l for l in self.dc if l[1] > 1e-9]
        self.capc -= z.sum(1)
        # 1. arrivals
        v = w['v'][day]
        keep = []
        for lot in self.lh:
            lot[2] += v[0] * DT
            if lot[2] >= c['L0']:
                self.dc.append([lot[0], lot[1], lot[3]])
            else:
                keep.append(lot)
        self.lh = keep
        keep = []
        for lot in self.rg:
            lot[3] += v[lot[0] + 1] * DT
            if lot[3] >= c['LC'][lot[0]]:
                self.S[lot[0], lot[1]] += lot[2]
            else:
                keep.append(lot)
        self.rg = keep
        # 2. demand
        dstep = w['dem'][day] / c['SPD']
        y = np.minimum(self.S, self.B + dstep)
        self.S -= y; self.B += dstep - y
        self._add(kpi, 'demand', dstep.sum()); self._add(kpi, 'served', y.sum())
        # 4. loss, ageing, expiry
        km = w['k_mult'][day]
        lam = [loss_rate(c, p, w['t_dc'][day]) * km for p in range(P)]
        xi = [loss_rate(c, p, w['t_rf'][day]) * km for p in range(P)]
        waste = 0.0; dc_units = 0.0; tr_units = 0.0
        for l in self.lh:
            s = l[1] * (1 - np.exp(-xi[l[0]] * DT)); l[1] -= s; waste += s; l[3] += DT; tr_units += l[1]
        for l in self.rg:
            s = l[2] * (1 - np.exp(-xi[l[1]] * DT)); l[2] -= s; waste += s; l[4] += DT; tr_units += l[2]
        keep = []
        for l in self.dc:
            s = l[1] * (1 - np.exp(-lam[l[0]] * DT)); l[1] -= s; waste += s; l[2] += DT
            if l[2] >= c['SL_DAYS'][l[0]] - 1e-9:
                waste += l[1]; self._add(kpi, 'expired', l[1])
            else:
                keep.append(l); dc_units += l[1]
        self.dc = keep
        s = self.S * (1 - np.exp(-c['KAPPA'] * DT)); self.S -= s; waste += s.sum()
        lost = self.B * (1 - np.exp(-c['RHO'] * DT)); self.B -= lost
        self._add(kpi, 'waste', waste); self._add(kpi, 'lost', lost.sum())
        self._add(cost, 'waste', c['DISPOSAL'] * waste)
        self._add(cost, 'lost_sales', c['PI_LOST'] * lost.sum())
        self._add(cost, 'backlog', c['PI_B'] * DT * self.B.sum())
        hold = (c['H_RAW'] * self.R + c['H_DC'] * dc_units + c['H_TR'] * tr_units
                + c['H_CUS'] * self.S.sum()) * DT
        self._add(cost, 'holding', hold)
        ta = w['t_amb'][day]
        kwh = (c['E_DC'] * (1 + 0.02 * (ta - 20)) * dc_units + c['E_TR'] * (1 + 0.03 * (ta - 20)) * tr_units) * DT
        self._add(kpi, 'kwh', kwh)
        self._add(cost, 'energy', c['P_KWH'] * kwh)
        emis = kpi.get('emis_truck', 0.0) + c['GRID_EF'] * kwh
        self._add(kpi, 'emissions', emis)
        self._add(cost, 'carbon', c['CARBON'] * emis)
        tv = (dc_units if w['t_dc'][day] > 8 else 0.0) + (tr_units if w['t_rf'][day] > 8 else 0.0)
        self._add(kpi, 'temp_violation_unitdays', tv * DT)
        self._add(kpi, 'backlog_unitdays', self.B.sum() * DT)
        for d_, src in ((self.cost, cost), (self.kpi, kpi)):
            for kk, vv in src.items():
                d_[kk] = d_.get(kk, 0.0) + vv
        self.k += 1
        return cost, kpi

    def reset_accounts(self):
        self.cost = {}; self.kpi = {}
