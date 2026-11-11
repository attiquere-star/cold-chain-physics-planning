"""Rolling-horizon planning MILPs.
mode='phy': upwind finite-volume transport on a CFL-checked grid + age-structured DC
            inventory with shelf-life expiry (the proposed physics-aware model).
mode='agg': conventional discrete-time model: fixed integer transport delay,
            single inventory pool with average loss, no ageing (benchmark)."""
import numpy as np
from .lp import Model

JMIN = 4

def cfl_grid(L, vmax, DT, factor=1, jmin=None):
    jmin = JMIN if jmin is None else jmin
    """CFL-compliant grid selection with sub-cycling: pick the smallest number of
    sub-steps n so that J = floor(n L / (vmax DT)) >= jmin cells; then the sub-step
    Courant number alpha = v (DT/n) / dx <= 1 for every forecast velocity v <= vmax.
    factor > 1 refines dx by that factor without more sub-steps (CFL violated)."""
    r = vmax * DT / L
    nsub = max(1, int(np.ceil(jmin * r)))
    J = max(1, int(np.floor(nsub / r)))
    J = J * factor
    return J, L / J, nsub

def route_operator(v, xi, L, DT, J, nsub):
    """Exact composition of nsub first-order upwind sub-steps (finite-volume, mass form).
    Returns M (J x J) and o (J,) such that, with injection u in cell 0 at the start of the step,
    F_next = M (F + u e0) and arrivals during the step = o . (F + u e0). Linear in F and u."""
    dx = L / J; dts = DT / nsub; a = v * dts / dx; d = 1.0 - a - xi * dts
    T = np.diag(np.full(J, d)) + np.diag(np.full(J - 1, a), -1)
    M = np.eye(J); o = np.zeros(J)
    for _ in range(nsub):
        o = o + a * M[J - 1]
        M = T @ M
    M[np.abs(M) < 1e-6] = 0.0; o[np.abs(o) < 1e-6] = 0.0
    return M, o, a


def char_arrivals(vel, xi, L, DT, K, pos0=0.0, k0=0):
    """Method of characteristics for the advection-reaction PDE with forecast velocity:
    a unit injected at step k0 (position pos0) arrives during step j; survival = exp(-sum xi dt).
    Equivalent to upwind at Courant number one on a characteristic-aligned grid (no numerical
    diffusion). Returns (j, survival per product, steps in transit)."""
    pos = pos0; j = k0; surv = np.ones(xi.shape[1])
    while True:
        v = vel[min(j, K - 1)]
        pos += v * DT
        if pos >= L - 1e-9:
            return j, surv, j - k0
        surv = surv * np.exp(-xi[min(j, K - 1)] * DT)
        j += 1

def build_and_solve(c, sim, pp, mode='phy', grid_factor=1, return_model=False):
    DT, SPD = c['DT'], c['SPD']; K = c['H_DAYS'] * SPD
    C, P, S = c['C'], c['P'], c['S']
    m = Model(); info_extra = {}
    dem, vel, lam, xi, tamb = pp['dem'], pp['vel'], pp['lam'], pp['xi'], pp['tamb']
    e_dc = c['E_DC'] * (1 + 0.02 * (tamb - 20)) * DT     # kWh per unit-step
    e_tr = c['E_TR'] * (1 + 0.03 * (tamb - 20)) * DT
    w_dc = c['H_DC'] * DT + (c['P_KWH'] + c['CARBON'] * c['GRID_EF']) * e_dc   # per unit-step at DC
    w_tr = c['H_TR'] * DT + (c['P_KWH'] + c['CARBON'] * c['GRID_EF']) * e_tr
    ndays = c['H_DAYS']
    # --- supply, raw, production, long-haul trucks
    A = m.var((S, ndays), 0, 1, integer=True)
    NI = c.get('INT_DAYS', ndays)   # relax-and-fix: integrality only on the first NI days
    for s in range(S):
        m.addcost(A[s], c['F_SUP'][s])
    q = m.var((S, K))
    for s in range(S):
        m.addcost(q[s], c['C_SUP'][s])
        for k in range(K):
            m.le([(q[s, k], 1), (A[s, k // SPD], -c['CAP_SUP'][s] * pp['supcap'][k // SPD, s])])
    R = m.var((K + 1,)); m.fix(R[0], sim.R); m.addcost(R[1:], c['H_RAW'] * DT)
    X = m.var((P, K)); m.addcost(X, c['C_PROD'])
    N0 = m.var((ndays,), 0, c['NMAX0'] * SPD, integer=True)     # daily truck dispatches
    m.addcost(N0, c['C_TRUCK0'] + c['CARBON'] * c['L0'] * c['EF0'])
    for k in range(K):
        qin = [(q[:, k - 1], -1)] if k >= 1 else []
        m.eq([(R[k + 1], 1), (R[k], -1), (X[:, k], 1)] + qin, sim.raw_pending if k == 0 else 0.0)
        m.le([(X[:, k], 1)], c['PROD_CAP'])
    for dd in range(ndays):
        m.le([(X[:, dd * SPD:(dd + 1) * SPD], 1), (N0[dd], -c['CAP0'])])
    # --- regional trucks & shipments
    z = m.var((C, P, K))
    Nc = m.var((C, ndays), 0, c['NMAXC'] * SPD, integer=True)
    for cc in range(C):
        m.addcost(Nc[cc], c['C_TRUCKC_FIX'] + c['C_TRUCKC_KM'] * c['LC'][cc] + c['CARBON'] * c['LC'][cc] * c['EFC'])
        for dd in range(ndays):
            m.le([(z[cc, :, dd * SPD:(dd + 1) * SPD], 1), (Nc[cc, dd], -c['CAPC'])])
    arrc = {}   # (cc,p,k) -> list of (idx, coef) or constant
    arr0 = {}
    waste_terms = []   # (idx, coef) of units lost, for disposal cost
    if mode == 'phy':
        vmax0 = vel[:, 0].max(); J0, dx0, ns0 = cfl_grid(c['L0'], vmax0, DT, grid_factor)
        info_extra['J0'] = J0; info_extra['nsub0'] = ns0
        lb_ = -np.inf if pp.get('free_sign') else 0.0
        F0 = m.var((P, J0, K + 1), lb=lb_)
        init = np.zeros((P, J0))
        for l in sim.lh:
            init[l[0], min(J0 - 1, int(l[2] // dx0))] += l[1]
        m.fix(F0[:, :, 0], init)
        amax = 0.0
        for p in range(P):
            for k in range(K):
                M, o, a = route_operator(vel[k, 0], xi[k, p], c['L0'], DT, J0, ns0); amax = max(amax, a)
                for j in range(J0):
                    nz = np.nonzero(M[j])[0]; m.eq([(F0[p, j, k + 1], 1), (F0[p, nz, k], -M[j, nz])] + ([(X[p, k], -M[j, 0])] if M[j, 0] != 0 else []))
                nz = np.nonzero(o)[0]; arr0[p, k] = [(F0[p, nz, k], o[nz])] + ([(X[p, k], o[0])] if o[0] != 0 else [])
            m.addcost(F0[p, :, 1:], np.repeat(w_tr[None, :], J0, 0) + c['DISPOSAL'] * xi[:, p][None, :] * DT)
            m.addcost(F0[p, :, K], -c['V_TERM'])
        info_extra['alpha0_max'] = amax
        # age-structured DC inventory
        tau0 = c['L0'] / vel[:, 0].mean()
        I, W = {}, {}
        for p in range(P):
            Am = int(round(c['SL_DAYS'][p] / DT))
            a_in = min(Am - 1, int(round(tau0 / DT)))
            I[p] = m.var((Am, K + 1)); W[p] = m.var((Am, K))
            init = np.zeros(Am)
            for l in sim.dc:
                if l[0] == p:
                    init[min(Am - 1, int(l[2] // DT))] += l[1]
            m.fix(I[p][:, 0], init)
            for k in range(K):
                g = 1 - lam[k, p] * DT
                for a in range(Am):
                    m.ge([(I[p][a, k], 1), (W[p][a, k], -1)], 0.0)
                    terms = [(I[p][a, k + 1], 1)]
                    if a >= 1:
                        terms += [(I[p][a - 1, k], -g), (W[p][a - 1, k], g)]
                    if a == a_in:
                        terms += [(v_, -co) for v_, co in arr0[p, k]]
                    m.eq(terms)
                # expiry of the oldest class: disposal cost
                m.addcost(I[p][Am - 1, k], c['DISPOSAL'] * g); m.addcost(W[p][Am - 1, k], -c['DISPOSAL'] * g)
                m.addcost(I[p][:, k + 1], w_dc[k] + c['DISPOSAL'] * lam[k, p] * DT)
                m.eq([(W[p][:, k], 1), (z[:, p, k], -1)])
            m.addcost(I[p][:, K], -c['V_TERM'])
        # regional routes
        for cc in range(C):
            vm = vel[:, cc + 1].max(); Jc, dxc, nsc = cfl_grid(c['LC'][cc], vm, DT, grid_factor)
            Fc = m.var((P, Jc, K + 1), lb=lb_)
            init = np.zeros((P, Jc))
            for l in sim.rg:
                if l[0] == cc:
                    init[l[1], min(Jc - 1, int(l[3] // dxc))] += l[2]
            m.fix(Fc[:, :, 0], init)
            amax = 0.0
            for p in range(P):
                for k in range(K):
                    M, o, a = route_operator(vel[k, cc + 1], xi[k, p], c['LC'][cc], DT, Jc, nsc); amax = max(amax, a)
                    for j in range(Jc):
                        nz = np.nonzero(M[j])[0]; m.eq([(Fc[p, j, k + 1], 1), (Fc[p, nz, k], -M[j, nz])] + ([(z[cc, p, k], -M[j, 0])] if M[j, 0] != 0 else []))
                    nz = np.nonzero(o)[0]; arrc[cc, p, k] = [(Fc[p, nz, k], o[nz])] + ([(z[cc, p, k], o[0])] if o[0] != 0 else [])
                m.addcost(Fc[p, :, 1:], np.repeat(w_tr[None, :], Jc, 0) + c['DISPOSAL'] * xi[:, p][None, :] * DT)
                m.addcost(Fc[p, :, K], -c['V_TERM'])
            info_extra['alpha_c_max_%d' % cc] = amax
    elif mode == 'char':
        # long haul: characteristic transport into age-structured DC
        contrib = {}   # (p, j, a_in) -> list of terms
        def addc(key, term):
            contrib.setdefault(key, []).append(term)
        for k in range(K):
            j, sv, st = char_arrivals(vel[:, 0], xi, c['L0'], DT, K, 0.0, k)
            for p in range(P):
                if j < K:
                    addc((p, j), (X[p, k], sv[p], st))
                else:
                    m.addcost(X[p, k], -c['V_TERM'] * sv[p])
                m.addcost(X[p, k], w_tr[min(k, K - 1)] * st + c['DISPOSAL'] * (1 - sv[p]))
        pipe_dc = {}
        for l in sim.lh:
            j, sv, st = char_arrivals(vel[:, 0], xi, c['L0'], DT, K, l[2], 0)
            if j < K:
                pipe_dc.setdefault((l[0], j), []).append((l[1] * sv[l[0]], st + l[3] / DT))
        I, W = {}, {}
        for p in range(P):
            Am = int(round(c['SL_DAYS'][p] / DT))
            I[p] = m.var((Am, K + 1)); W[p] = m.var((Am, K))
            init = np.zeros(Am)
            for l in sim.dc:
                if l[0] == p:
                    init[min(Am - 1, int(l[2] // DT))] += l[1]
            m.fix(I[p][:, 0], init)
            for k in range(K):
                g = 1 - lam[k, p] * DT
                byage = {}
                for (v_, sv, st) in contrib.get((p, k), []):
                    byage.setdefault(min(Am - 1, int(st)), []).append((v_, -sv))
                const = {}
                for (qty, age) in pipe_dc.get((p, k), []):
                    a_ = min(Am - 1, int(round(age))); const[a_] = const.get(a_, 0.0) + qty
                for a in range(Am):
                    m.ge([(I[p][a, k], 1), (W[p][a, k], -1)], 0.0)
                    terms = [(I[p][a, k + 1], 1)]
                    if a >= 1:
                        terms += [(I[p][a - 1, k], -g), (W[p][a - 1, k], g)]
                    terms += byage.get(a, [])
                    m.eq(terms, const.get(a, 0.0))
                m.addcost(I[p][Am - 1, k], c['DISPOSAL'] * g); m.addcost(W[p][Am - 1, k], -c['DISPOSAL'] * g)
                m.addcost(I[p][:, k + 1], w_dc[k] + c['DISPOSAL'] * lam[k, p] * DT)
                m.eq([(W[p][:, k], 1), (z[:, p, k], -1)])
            m.addcost(I[p][:, K], -c['V_TERM'])
        # regional routes: characteristic transport to customers
        for cc in range(C):
            for p in range(P):
                for k in range(K):
                    arrc[cc, p, k] = []
            for k in range(K):
                j, sv, st = char_arrivals(vel[:, cc + 1], xi, c['LC'][cc], DT, K, 0.0, k)
                for p in range(P):
                    if j < K:
                        arrc[cc, p, j].append((z[cc, p, k], sv[p]))
                    else:
                        m.addcost(z[cc, p, k], -c['V_TERM'] * sv[p])
                    m.addcost(z[cc, p, k], w_tr[k] * st + c['DISPOSAL'] * (1 - sv[p]))
            for l in sim.rg:
                if l[0] == cc:
                    j, sv, st = char_arrivals(vel[:, cc + 1], xi, c['LC'][cc], DT, K, l[3], 0)
                    if j < K:
                        arrc[cc, l[1], j].append(('const', l[2] * sv[l[1]]))
    else:   # aggregated benchmark
        v0bar = vel[:, 0].mean(); tau0 = max(1, int(round(c['L0'] / v0bar / DT)))
        xibar = xi.mean(0); lambar = lam.mean(0)
        ell0 = 1 - np.exp(-xibar * tau0 * DT)
        pipe0 = np.zeros((P, K))
        for l in sim.lh:
            kk = int(np.ceil((c['L0'] - l[2]) / (v0bar * DT))) - 1
            if 0 <= kk < K:
                pipe0[l[0], kk] += l[1]
            elif kk < 0:
                pipe0[l[0], 0] += l[1]
        I = m.var((P, K + 1)); W = m.var((P, K))
        m.fix(I[:, 0], [sum(l[1] for l in sim.dc if l[0] == p) for p in range(P)])
        for p in range(P):
            for k in range(K):
                g = 1 - lambar[p] * DT
                terms = [(I[p, k + 1], 1), (I[p, k], -g), (W[p, k], g)]
                if k - tau0 >= 0:
                    terms += [(X[p, k - tau0], -(1 - ell0[p]))]
                m.eq(terms, pipe0[p, k] * (1 - ell0[p] * 0.5))
                m.ge([(I[p, k], 1), (W[p, k], -1)], 0.0)
                m.eq([(W[p, k], 1), (z[:, p, k], -1)])
            m.addcost(I[p, 1:], w_dc + c['DISPOSAL'] * lambar[p] * DT)
            m.addcost(X[p], w_tr.mean() * tau0 + c['DISPOSAL'] * ell0[p])
            m.addcost(I[p, K], -c['V_TERM'])
            m.addcost(X[p, max(0, K - tau0):], -c['V_TERM'] * (1 - ell0[p]))
        for cc in range(C):
            vb = vel[:, cc + 1].mean(); tauc = max(1, int(round(c['LC'][cc] / vb / DT)))
            ellc = 1 - np.exp(-xibar * tauc * DT)
            pipe = np.zeros((P, K))
            for l in sim.rg:
                if l[0] == cc:
                    kk = int(np.ceil((c['LC'][cc] - l[3]) / (vb * DT))) - 1
                    pipe[l[1], min(max(kk, 0), K - 1)] += l[2] if kk < K else 0.0
            for p in range(P):
                for k in range(K):
                    arrc[cc, p, k] = [(z[cc, p, k - tauc], 1 - ellc[p])] if k - tauc >= 0 else []
                    if pipe[p, k] > 0:
                        arrc[cc, p, k] = arrc[cc, p, k] + [('const', pipe[p, k])]
                m.addcost(z[cc, p], w_tr.mean() * tauc + c['DISPOSAL'] * ellc[p])
                m.addcost(z[cc, p, max(0, K - tauc):], -c['V_TERM'] * (1 - ellc[p]))
    # --- customers: retail stock and backlog
    Sv = m.var((C, P, K + 1)); Bv = m.var((C, P, K + 1)); y = m.var((C, P, K))
    m.fix(Sv[:, :, 0], sim.S); m.fix(Bv[:, :, 0], sim.B)
    gk = 1 - c['KAPPA'] * DT; gb = 1 - c['RHO'] * DT
    for cc in range(C):
        for p in range(P):
            for k in range(K):
                const = 0.0; terms = [(Sv[cc, p, k + 1], 1), (Sv[cc, p, k], -gk), (y[cc, p, k], gk)]
                for v_, co in arrc[cc, p, k]:
                    if isinstance(v_, str): const += co * gk
                    else: terms.append((v_, -co * gk))
                m.eq(terms, const)
                m.eq([(Bv[cc, p, k + 1], 1), (Bv[cc, p, k], -gb), (y[cc, p, k], gb)], gb * dem[k, cc, p])
                # lost sales = rho*DT*(B + d - y)
                m.addcost(Bv[cc, p, k], c['PI_LOST'] * c['RHO'] * DT)
                m.addcost(y[cc, p, k], -c['PI_LOST'] * c['RHO'] * DT)
            m.addcost(Sv[cc, p, 1:], c['H_CUS'] * DT + c['DISPOSAL'] * c['KAPPA'] * DT)
            m.addcost(Bv[cc, p, 1:], c['PI_B'] * DT)
            m.addcost(Sv[cc, p, K], -c['V_TERM'])
            m.addcost(Bv[cc, p, K], c['PI_LOST'] * 0.5)
    for arr in (A, N0, Nc):
        for i in np.atleast_2d(arr)[..., NI:].ravel():
            m.integ[i] = 0
    x, info = m.solve(gap=c['MIP_GAP'], time_limit=c['TIME_LIMIT'])
    info.update(info_extra)
    if x is None and NI > 0:
        # no incumbent within the time limit: LP-relaxation fallback, integers rounded up below
        c2 = dict(c); c2['INT_DAYS'] = 0
        dec, info2 = build_and_solve(c2, sim, pp, mode, grid_factor, return_model)
        if info2 is not None:
            info2['fallback'] = 1; info2['time'] += info['time']
        return dec, info2
    if x is None:
        return None, info
    info['fallback'] = 0
    rd = lambda idx: np.round(x[idx], 6)
    dec = []
    for k in range(SPD):
        dec.append(dict(q=rd(q[:, k]), A=np.ceil(rd(A[:, 0]) - 1e-4), X=rd(X[:, k]), N0=int(np.ceil(x[N0[0]] - 1e-4)),
                        z=rd(z[:, :, k]), Nc=np.ceil(rd(Nc[:, 0]) - 1e-4).astype(int)))
    info['plan_obj'] = info['obj']
    if return_model:
        def ev(terms):
            t = 0.0
            for v_, co in terms:
                if isinstance(v_, str): t += co
                else: t += float(np.dot(np.atleast_1d(x[v_]), np.broadcast_to(co, np.shape(np.atleast_1d(v_)))))
            return t
        info['pred_cus_arr'] = np.array([sum(ev(arrc[cc, p, k]) for cc in range(C) for p in range(P)) for k in range(K)])
        if mode == 'phy':
            info['pred_dc_arr'] = np.array([sum(ev(arr0[p, k]) for p in range(P)) for k in range(K)])
        info['pred_y'] = x[y].sum((0, 1)); info['pred_S'] = x[Sv].sum((0, 1)); info['pred_B'] = x[Bv].sum((0, 1))
        info['all_dec'] = [dict(q=x[q[:, k]], A=np.ceil(x[A[:, k // SPD]] - 1e-6), X=x[X[:, k]], N0=int(np.ceil(x[N0[k // SPD]] - 1e-6)),
                                z=x[z[:, :, k]], Nc=np.ceil(x[Nc[:, k // SPD]] - 1e-6).astype(int)) for k in range(K)]
    return dec, info
