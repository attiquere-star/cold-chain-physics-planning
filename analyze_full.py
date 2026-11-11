import json, glob, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
OUT = sys.argv[1]
rows = []
for f in glob.glob(f'{OUT}/fullmain_*.json'):
    _, s, m = f.split('/')[-1][:-5].split('_', 2); o = json.load(open(f)); k = o['kpi']
    rows.append(dict(seed=int(s), method=m, cost=o['total_cost'], fill=k['served'] / k['demand'] * 100, solve=o['solve_time'], gap=o['gap'], fb=o['fallbacks']))
F = pd.DataFrame(rows)
M = pd.read_csv(f'{OUT}/analysis/main_runs.csv')
P = F.pivot(index='seed', columns='method', values='cost').dropna()
R = M.pivot(index='seed', columns='method', values='total_cost').loc[P.index]
print('n seeds complete', len(P))
print(F.groupby('method')[['cost', 'fill', 'solve', 'gap', 'fb']].mean().round(3).to_string())
def boot(x, n=5000):
    rng = np.random.default_rng(0); b = [rng.choice(x, len(x)).mean() for _ in range(n)]; return np.percentile(b, [2.5, 97.5])
res = []
for a, b, h in [('ORACLE-CHAR', 'ORACLE-AGG', 'H1'), ('FTO-CHAR', 'FTO-AGG', 'H3'), ('DFL-CHAR', 'FTO-CHAR', 'H4'), ('DFL-AGG', 'FTO-AGG', 'H4'),
                ('DFL-CHAR', 'DFL-AGG', '-'), ('DFL-CHAR', 'FTO-AGG', '-')]:
    for tag, T in (('full', P), ('rf', R)):
        d = (T[a] - T[b]) / T[b] * 100; lo, hi = boot(d.values)
        p = wilcoxon(d).pvalue if len(d) > 5 else np.nan
        res.append(dict(solver=tag, hyp=h, contrast=f'{a} - {b}', pct=d.mean(), lo=lo, hi=hi, wins=int((d < 0).sum()), n=len(d), p=p))
X = pd.DataFrame(res)
fam = X[X.solver == 'full'].copy(); ps = fam.p.values; o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
for r, i in enumerate(o): run = max(run, (m - r) * ps[i]); adj[i] = min(1, run)
X.loc[X.solver == 'full', 'p_holm'] = adj
ia = (P['FTO-CHAR'] - P['DFL-CHAR']) - (P['FTO-AGG'] - P['DFL-AGG'])
print(X.round(3).to_string()); print('interaction full', ia.mean().round(0), boot(ia.values).round(0), (ia > 0).sum(), wilcoxon(ia).pvalue if len(ia) > 5 else '')
both = (F.set_index(['seed', 'method']).cost / M.set_index(['seed', 'method']).total_cost.reindex(F.set_index(['seed', 'method']).index) - 1) * 100
print('full vs rf by method (%)', both.groupby(level=1).mean().round(2).to_dict())
F.to_csv(f'{OUT}/analysis/fullmain_runs.csv', index=False); X.to_csv(f'{OUT}/analysis/fullmain_contrasts.csv', index=False)
