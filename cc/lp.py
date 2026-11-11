"""Tiny sparse MILP builder on top of scipy.optimize.milp (HiGHS)."""
import time
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

class Model:
    def __init__(self):
        self.n = 0; self.lb = []; self.ub = []; self.integ = []; self.cost = []
        self.rows, self.cols, self.vals = [], [], []; self.rlo = []; self.rhi = []; self.m = 0

    def var(self, shape, lb=0.0, ub=np.inf, integer=False, cost=0.0):
        size = int(np.prod(shape)) if shape else 1
        idx = np.arange(self.n, self.n + size).reshape(shape) if shape else self.n
        self.n += size
        self.lb += [lb] * size; self.ub += [ub] * size
        self.integ += [1 if integer else 0] * size; self.cost += [cost] * size
        return idx

    def fix(self, idx, val):
        for i, v in zip(np.atleast_1d(idx).ravel(), np.broadcast_to(val, np.shape(np.atleast_1d(idx))).ravel()):
            self.lb[i] = float(v); self.ub[i] = float(v)

    def setub(self, idx, val):
        for i, v in zip(np.atleast_1d(idx).ravel(), np.broadcast_to(val, np.shape(np.atleast_1d(idx))).ravel()):
            self.ub[i] = float(v)

    def addcost(self, idx, coef):
        for i, v in zip(np.atleast_1d(idx).ravel(), np.broadcast_to(coef, np.shape(np.atleast_1d(idx))).ravel()):
            self.cost[i] += float(v)

    def row(self, terms, lo, hi):
        """terms: list of (var index or array, coef or array)."""
        r = self.m
        for idx, coef in terms:
            idx = np.atleast_1d(idx).ravel()
            coef = np.broadcast_to(coef, idx.shape).ravel()
            self.rows += [r] * len(idx); self.cols += list(idx); self.vals += list(coef)
        self.rlo.append(lo); self.rhi.append(hi); self.m += 1

    def eq(self, terms, rhs=0.0): self.row(terms, rhs, rhs)
    def le(self, terms, rhs=0.0): self.row(terms, -np.inf, rhs)
    def ge(self, terms, rhs=0.0): self.row(terms, rhs, np.inf)

    def solve(self, gap=0.01, time_limit=20.0, relax=False):
        A = sp.csr_matrix((self.vals, (self.rows, self.cols)), shape=(self.m, self.n))
        t0 = time.time()
        res = milp(c=np.array(self.cost), constraints=LinearConstraint(A, self.rlo, self.rhi),
                   integrality=None if relax else np.array(self.integ),
                   bounds=Bounds(self.lb, self.ub),
                   options=dict(mip_rel_gap=gap, time_limit=time_limit, disp=False))
        info = dict(status=res.status, msg=res.message, time=time.time() - t0, obj=res.fun,
                    gap=getattr(res, 'mip_gap', None), n=self.n, m=self.m, nint=int(sum(self.integ)))
        return res.x, info
