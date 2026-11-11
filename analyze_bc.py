"""Exploratory analysis of the bias-correction baseline (BC-AGG, BC-CHAR)."""
import sys, json, glob, numpy as np, pandas as pd
from scipy.stats import wilcoxon
OUT = sys.argv[1] if len(sys.argv) > 1 else 'out'; A = OUT + '/analysis'
rng = np.random.default_rng(0)
def boot_ci(x, n=5000):
    m = rng.choice(x, (n, len(x))).mean(1); return np.percentile(m, [2.5, 97.5])
def holm(p):
    o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (len(p) - r) * p[i])); adj[i] = run
    return adj
M = pd.read_csv(A + '/main_runs.csv')
rows = []
for f in sorted(glob.glob(OUT + '/bc_*.json')):
    d = json.load(open(f))
    for name, o in d['eval'].items():
        k = o['kpi']
        rows.append(dict(seed=d['seed'], method=name, total_cost=o['total_cost'], fill=k['served'] / k['demand'] * 100,
                         waste_pct=k['waste'] / k['demand'] * 100, emissions_t=k['emissions'] / 1000,
                         theta=d['theta_bc']))
B = pd.DataFrame(rows); B.drop(columns='theta').to_csv(A + '/bc_runs.csv', index=False)
W = pd.concat([M[['seed', 'method', 'total_cost']], B[['seed', 'method', 'total_cost']]]).pivot(index='seed', columns='method', values='total_cost')
pairs = [('BC-CHAR', 'FTO-CHAR'), ('BC-AGG', 'FTO-AGG'), ('DFL-CHAR', 'BC-CHAR'), ('DFL-AGG', 'BC-AGG'), ('BC-CHAR', 'BC-AGG')]
out = []
for a, b in pairs:
    x, y = W[a].values, W[b].values; pct = (x - y) / y * 100
    lo, hi = boot_ci(pct)
    out.append(dict(contrast=f'{a} - {b}', diff=(x - y).mean(), pct=pct.mean(), lo=lo, hi=hi, wins=int((x < y).sum()), p=wilcoxon(x, y).pvalue))
C = pd.DataFrame(out); C['p_holm'] = holm(C.p.values); C.to_csv(A + '/bc_contrasts.csv', index=False)
print(C.round(3).to_string())
for m in ['BC-AGG', 'BC-CHAR']:
    print(m, 'mean', round(W[m].mean()), 'sd', round(W[m].std()))
th = np.array([json.load(open(f))['theta_bc'] for f in sorted(glob.glob(OUT + '/bc_*.json'))])
print('theta_bc mean', th.mean(0).round(3), 'sd', th.std(0, ddof=1).round(3))
# share of forecast penalty closed
for mdl in ['CHAR', 'AGG']:
    pen = W[f'FTO-{mdl}'] - W[f'ORACLE-{mdl}']
    print(mdl, 'penalty', round(pen.mean()), 'BC closes', round(((W[f'FTO-{mdl}'] - W[f'BC-{mdl}']).mean() / pen.mean()) * 100, 1), '% ; DFL closes', round(((W[f'FTO-{mdl}'] - W[f'DFL-{mdl}']).mean() / pen.mean()) * 100, 1), '%')
print(B.groupby('method')[['fill', 'waste_pct', 'emissions_t']].agg(['mean', 'std']).round(2))
