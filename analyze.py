import sys, os, json, glob, numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = sys.argv[1]; A = OUT + '/analysis'; os.makedirs(A, exist_ok=True)
rng = np.random.default_rng(0)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#3d4a52',
                     'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'font.family': 'DejaVu Sans',
                     'axes.grid': True, 'grid.color': '#e3ecef', 'grid.linewidth': 0.6, 'savefig.dpi': 300})
NAVY, TEAL, LIGHT, PALE, GREEN, BRICK, SALMON = '#063D59', '#0B5C79', '#A8D8E6', '#D0EBEF', '#63B9A7', '#B96263', '#D88970'
from matplotlib.colors import LinearSegmentedColormap
DIV = LinearSegmentedColormap.from_list('ocean_div', [NAVY, TEAL, PALE, '#F7F5F2', '#F2D3C8', SALMON, BRICK])
def mcol(m):   # colour by information level; marker shape repeats the encoding
    return GREEN if m.startswith('DFL') else (BRICK if 'ORACLE' in m else (SALMON if m.startswith('BC') else TEAL))
def mmark(m):
    return 's' if m.startswith('DFL') else ('D' if 'ORACLE' in m else ('^' if m.startswith('BC') else 'o'))
def panel(ax, letter, title):
    ax.set_title(f'$\\bf{{{letter}}}$  {title}', fontsize=8.5, loc='left', color=NAVY, pad=6,
                 bbox=dict(boxstyle='square,pad=0.35', fc=PALE, ec='none'))
ORDER = ['STATIC-AGG', 'FTO-AGG', 'DFL-AGG', 'FTO-UPW', 'FTO-UPW-VIOL', 'FTO-CHAR', 'DFL-CHAR', 'ORACLE-AGG', 'ORACLE-CHAR']

def boot_ci(x, n=5000):
    x = np.asarray(x); b = rng.choice(x, (n, len(x))).mean(1); return np.percentile(b, [2.5, 97.5])
def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); m = len(p); run = 0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k]); adj[k] = min(1, run)
    return adj
def contrast(df, a, b, metric='total_cost'):
    x = df.loc[a, metric].values; y = df.loc[b, metric].values; d = x - y
    pct = d / y * 100
    try: p = wilcoxon(x, y).pvalue
    except ValueError: p = 1.0
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
    lo, hi = boot_ci(pct)
    return dict(contrast=f'{a} vs {b}', mean_diff=d.mean(), mean_pct=pct.mean(), ci_lo=lo, ci_hi=hi, wins=int((d < 0).sum()), n=len(d), p=p, dz=dz)

# ---------------- main
rows = []; acc = []; thetas = []
for f in sorted(glob.glob(OUT + '/main_*.json')):
    r = json.load(open(f)); s = r['seed']
    for m, e in r['eval'].items():
        k = e['kpi']
        rows.append(dict(seed=s, method=m, total_cost=e['total_cost'], fill=k['served'] / k['demand'] * 100,
                         lost=k['lost'], waste_pct=k['waste'] / k['demand'] * 100, expired=k.get('expired', 0.0),
                         emissions_t=k['emissions'] / 1000, kwh=k['kwh'], tv=k.get('temp_violation_unitdays', 0.0),
                         trucks=k['trucks'], solve=e['solve_time'], solve_max=e['solve_time_max'], fallbacks=e.get('fallbacks', 0),
                         **{'c_' + kk: vv for kk, vv in e['cost'].items()}))
    for mdl, mm in r['acc'].items():
        acc.append(dict(seed=s, model=mdl, **mm))
    acc.append(dict(seed=s, model='LSTM (route speed)', **r['acc_vel']))
    for mode, th in r['theta'].items():
        thetas.append(dict(seed=s, mode=mode, dem_p0=th[0], dem_p1=th[1], speed=th[2], loss=th[3]))
D = pd.DataFrame(rows); n_seeds = D.seed.nunique()
orc = D[D.method == 'ORACLE-CHAR'].set_index('seed').total_cost
D['regret_pct'] = D.apply(lambda r: (r.total_cost - orc[r.seed]) / orc[r.seed] * 100, axis=1)
D.to_csv(A + '/main_runs.csv', index=False)
agg = D.groupby('method').agg(['mean', 'std']).reindex(ORDER)
tab = pd.DataFrame({m: {'Total cost': f"{agg.loc[m, ('total_cost', 'mean')]:,.0f} ({agg.loc[m, ('total_cost', 'std')]:,.0f})",
                        'Regret vs oracle (%)': f"{agg.loc[m, ('regret_pct', 'mean')]:.1f} ({agg.loc[m, ('regret_pct', 'std')]:.1f})",
                        'Fill rate (%)': f"{agg.loc[m, ('fill', 'mean')]:.1f} ({agg.loc[m, ('fill', 'std')]:.1f})",
                        'Waste (% of demand)': f"{agg.loc[m, ('waste_pct', 'mean')]:.2f} ({agg.loc[m, ('waste_pct', 'std')]:.2f})",
                        'Emissions (t CO2e)': f"{agg.loc[m, ('emissions_t', 'mean')]:.1f} ({agg.loc[m, ('emissions_t', 'std')]:.1f})",
                        'Energy (kWh)': f"{agg.loc[m, ('kwh', 'mean')]:,.0f}",
                        'Temp. exposure (unit-days)': f"{agg.loc[m, ('tv', 'mean')]:,.0f}",
                        'Truck dispatches': f"{agg.loc[m, ('trucks', 'mean')]:.0f}",
                        'Mean solve time (s)': f"{agg.loc[m, ('solve', 'mean')]:.2f}",
                        'LP fallbacks per run': f"{agg.loc[m, ('fallbacks', 'mean')]:.1f}"} for m in ORDER}).T
tab.to_csv(A + '/table_main.csv')
W = D.set_index(['method', 'seed']).sort_index()
pairs = [('FTO-CHAR', 'FTO-AGG'), ('DFL-CHAR', 'FTO-CHAR'), ('DFL-AGG', 'FTO-AGG'), ('DFL-CHAR', 'DFL-AGG'), ('DFL-CHAR', 'FTO-AGG'),
         ('DFL-CHAR', 'STATIC-AGG'), ('FTO-UPW', 'FTO-CHAR'), ('FTO-UPW-VIOL', 'FTO-UPW'), ('ORACLE-CHAR', 'ORACLE-AGG')]
C = pd.DataFrame([contrast(W, a, b) for a, b in pairs]); C['p_holm'] = holm(C.p.values)
# interaction: (DFL-CHAR - FTO-CHAR) - (DFL-AGG - FTO-AGG)
inter = (W.loc['DFL-CHAR'].total_cost - W.loc['FTO-CHAR'].total_cost) - (W.loc['DFL-AGG'].total_cost - W.loc['FTO-AGG'].total_cost)
try: pint = wilcoxon(inter.values).pvalue
except ValueError: pint = 1.0
C = pd.concat([C, pd.DataFrame([dict(contrast='Interaction (DFL gain with CHAR minus with AGG)', mean_diff=inter.mean(), p=pint, n=len(inter))])])
fillc = pd.DataFrame([contrast(W, a, b, 'fill') for a, b in pairs])
C.to_csv(A + '/contrasts_cost.csv', index=False); fillc.to_csv(A + '/contrasts_fill.csv', index=False)
ACC = pd.DataFrame(acc).groupby('model').agg(['mean', 'std'])
ACC.to_csv(A + '/forecast_accuracy.csv')
TH = pd.DataFrame(thetas); TH.groupby('mode').agg(['mean', 'std', 'min', 'max']).to_csv(A + '/thetas.csv')
costcols = [c for c in D.columns if c.startswith('c_')]
D.groupby('method')[costcols].mean().reindex(ORDER).to_csv(A + '/cost_breakdown.csv')

# Figure 1: total cost by method (mean and 95% CI over replications)
fig, ax = plt.subplots(figsize=(6.6, 3.4))
order = [m for m in ORDER]
for i, m in enumerate(order):
    x = D[D.method == m].total_cost.values / 1000; lo, hi = boot_ci(x)
    col = mcol(m)
    vp = ax.violinplot([x], positions=[i], widths=0.75, showextrema=False)
    for b in vp['bodies']: b.set_facecolor(col); b.set_alpha(0.22); b.set_edgecolor('none')
    ax.scatter(np.full(len(x), i) + rng.uniform(-0.12, 0.12, len(x)), x, s=9, color=col, alpha=0.55, linewidths=0)
    ax.errorbar(i, x.mean(), yerr=[[x.mean() - lo], [hi - x.mean()]], fmt=mmark(m), color=col, mec='white', mew=0.8, ms=6.5, capsize=0, lw=2)
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=35, ha='right')
ax.set_ylabel('Total cost over 42-day test (thousands)'); ax.grid(axis='x', visible=False)
pass
fig.tight_layout(); fig.savefig(A + '/fig_cost_by_method.png'); fig.savefig(A + '/fig_cost_by_method.eps'); plt.close(fig)

# Figure 2: cost breakdown (stacked, a few groups)
B = D.groupby('method')[costcols].mean().reindex(ORDER) / 1000
groups = {'Procurement & production': ['c_procurement', 'c_production', 'c_fixed_supplier'], 'Transport': ['c_transport'],
          'Lost sales & backlog': ['c_lost_sales', 'c_backlog'], 'Holding, energy & carbon': ['c_holding', 'c_energy', 'c_carbon'], 'Waste disposal': ['c_waste']}
cols = [TEAL, GREEN, BRICK, LIGHT, SALMON]
fig, ax = plt.subplots(figsize=(6.6, 3.4)); left = np.zeros(len(ORDER))
for (g, cc), col in zip(groups.items(), cols):
    v = B[[x for x in cc if x in B.columns]].sum(1).values
    ax.barh(range(len(ORDER)), v, left=left, color=col, label=g, edgecolor='white', linewidth=1.5, height=0.7); left += v
ax.set_yticks(range(len(ORDER))); ax.set_yticklabels(ORDER); ax.invert_yaxis(); ax.set_xlabel('Mean cost (thousands)')
ax.legend(fontsize=7, frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.18)); ax.grid(axis='y', visible=False)
fig.tight_layout(); fig.savefig(A + '/fig_cost_breakdown.png'); fig.savefig(A + '/fig_cost_breakdown.eps'); plt.close(fig)

# Figure 3: CFL numerical demonstration
cd = json.load(open(OUT + '/cfl_demo.json'))
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.0))
for r_, col in zip([x for x in cd if x['J'] in (4, 6, 7)], [LIGHT, GREEN, NAVY]):
    axs[0].fill_between(range(20), np.array(r_['arrivals']) / 1000, color=col, alpha=0.18, lw=0)
    axs[0].plot(range(20), np.array(r_['arrivals']) / 1000, color=col, lw=2, marker='o', ms=3, label=f"Courant {r_['courant']:.2f}")
axs[0].axvline(7, color=BRICK, lw=1.5, ls='--'); axs[0].text(7.4, 0.45, 'true arrival\n(plug flow)', fontsize=7, color='#52514e')
axs[0].set_xlabel('Steps after dispatch (6 h)'); axs[0].set_ylabel('Share of shipment arriving'); axs[0].legend(fontsize=7, frameon=False)
panel(axs[0], 'a', 'CFL-compliant: stable but diffusive')
xs = [x['courant'] for x in cd]; ys = [max(1e-3, x['max_abs_mass'] / 1000) for x in cd]
axs[1].axvspan(1.0, max(xs) + 0.05, color='#F2D3C8', alpha=0.5, lw=0); axs[1].semilogy(xs, ys, 'o-', color=NAVY, mfc=LIGHT, mec=NAVY, lw=2, ms=5); axs[1].axvline(1.0, color=BRICK, lw=1.5, ls='--')
axs[1].set_xlabel('Courant number vΔt/Δx'); axs[1].set_ylabel('Max |mass on route| / shipped')
panel(axs[1], 'b', 'CFL violated (>1): unstable')
fig.tight_layout(); fig.savefig(A + '/fig_cfl.png'); fig.savefig(A + '/fig_cfl.eps'); plt.close(fig)

summary = dict(n_seeds=int(n_seeds))
# ---------------- scenarios
sc_rows = []
for f in sorted(glob.glob(OUT + '/scen_*.json')):
    s = int(f.split('_')[-1].split('.')[0]); r = json.load(open(f))
    for sc, mm in r.items():
        for m, e in mm.items():
            k = e['kpi']; sc_rows.append(dict(seed=s, scenario=sc, method=m, total_cost=e['total_cost'], fill=k['served'] / k['demand'] * 100,
                                              waste_pct=k['waste'] / k['demand'] * 100, emissions_t=k['emissions'] / 1000))
if sc_rows:
    S = pd.DataFrame(sc_rows); S.to_csv(A + '/scen_runs.csv', index=False)
    base = S[S.method == 'FTO-AGG'].set_index(['scenario', 'seed']).total_cost
    S['pct_vs_fto_agg'] = S.apply(lambda r: (r.total_cost - base[(r.scenario, r.seed)]) / base[(r.scenario, r.seed)] * 100, axis=1)
    P_ = S.groupby(['scenario', 'method']).pct_vs_fto_agg.mean().unstack()
    P_.to_csv(A + '/scen_pct_vs_ftoagg.csv'); S.groupby(['scenario', 'method'])[['total_cost', 'fill', 'waste_pct']].mean().to_csv(A + '/scen_means.csv')
    sp = []
    for sc in S.scenario.unique():
        T = S[S.scenario == sc].set_index(['method', 'seed']).sort_index()
        for a, b in [('DFL-CHAR', 'FTO-AGG'), ('FTO-CHAR', 'FTO-AGG'), ('DFL-CHAR', 'FTO-CHAR'), ('DFL-AGG', 'FTO-AGG')]:
            sp.append(dict(scenario=sc, **contrast(T, a, b)))
    SP = pd.DataFrame(sp); SP['p_holm'] = holm(SP.p.values); SP.to_csv(A + '/scen_contrasts.csv', index=False)
    meths = ['STATIC-AGG', 'DFL-AGG', 'FTO-CHAR', 'DFL-CHAR', 'ORACLE-CHAR']
    M = P_[meths]
    fig, ax = plt.subplots(figsize=(6.6, 3.2)); lim = np.nanmax(np.abs(M.values))
    im = ax.imshow(M.values, cmap=DIV, vmin=-lim, vmax=lim, aspect='auto')
    ax.set_xticks(range(len(meths))); ax.set_xticklabels(meths, rotation=25, ha='right'); ax.set_yticks(range(len(M))); ax.set_yticklabels([x.replace('_', ' ') for x in M.index])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f'{M.values[i, j]:+.1f}', ha='center', va='center', fontsize=7, color=('white' if abs(M.values[i, j]) > 5 else '#0b0b0b'))
    ax.grid(False); cb = fig.colorbar(im, ax=ax); cb.set_label('% cost vs FTO-AGG (blue = cheaper)')
    fig.tight_layout(); fig.savefig(A + '/fig_scenarios.png'); fig.savefig(A + '/fig_scenarios.eps'); plt.close(fig)
# ---------------- sensitivity
se = []
for f in sorted(glob.glob(OUT + '/sens_*.json')):
    s = int(f.split('_')[-1].split('.')[0]); r = json.load(open(f))
    for st, mm in r.items():
        for m, e in mm.items():
            se.append(dict(seed=s, setting=st, method=m, total_cost=e['total_cost'], fill=e['kpi']['served'] / e['kpi']['demand'] * 100,
                           waste_pct=e['kpi']['waste'] / e['kpi']['demand'] * 100, expired=e['kpi'].get('expired', 0.0)))
if se:
    E = pd.DataFrame(se); E.to_csv(A + '/sens_runs.csv', index=False)
    out = []
    for st in E.setting.unique():
        T = E[E.setting == st].set_index(['method', 'seed']).sort_index()
        for a, b in [('FTO-CHAR', 'FTO-AGG'), ('ORACLE-CHAR', 'ORACLE-AGG')]:
            out.append(dict(setting=st, **contrast(T, a, b)))
    SE = pd.DataFrame(out); SE['p_holm'] = holm(SE.p.values); SE.to_csv(A + '/sens_contrasts.csv', index=False)
    E.groupby(['setting', 'method'])[['total_cost', 'fill', 'waste_pct', 'expired']].mean().to_csv(A + '/sens_means.csv')
    order_s = ['SL x0.5', 'base', 'SL x1.5', 'L0 450km', 'L0 1350km']
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    for k_, (a, b, col, lab) in enumerate([('FTO-CHAR', 'FTO-AGG', TEAL, 'with LSTM forecasts'), ('ORACLE-CHAR', 'ORACLE-AGG', BRICK, 'with perfect information')]):
        sub = SE[SE.contrast == f'{a} vs {b}'].set_index('setting').reindex(order_s)
        x = np.arange(len(order_s)) + (k_ - 0.5) * 0.25
        ax.errorbar(x, sub.mean_pct, yerr=[sub.mean_pct - sub.ci_lo, sub.ci_hi - sub.mean_pct], fmt=('o' if k_ == 0 else 'D'), color=col, mec='white', mew=0.8, lw=2, ms=6, label=lab)
    ax.axhline(0, color='#52514e', lw=0.8); ax.set_xticks(range(len(order_s))); ax.set_xticklabels([{'SL x0.5':'Shelf life\n×0.5','SL x1.5':'Shelf life\n×1.5','L0 450km':'Long haul\n450 km','L0 1350km':'Long haul\n1,350 km','base':'Base'}.get(x, x) for x in order_s])
    ax.set_ylabel('% cost change, CHAR vs AGG\n(negative = physics model cheaper)'); ax.legend(fontsize=7, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2); ax.grid(axis='x', visible=False)
    fig.tight_layout(); fig.savefig(A + '/fig_sensitivity.png'); fig.savefig(A + '/fig_sensitivity.eps'); plt.close(fig)
# ---------------- scalability & relax-and-fix
if os.path.exists(OUT + '/scale.json'):
    Sc = pd.DataFrame(json.load(open(OUT + '/scale.json')))
    Sc.groupby(['mode', 'C', 'H', 'int_days'])[['n', 'm', 'nint', 'time', 'gap', 'obj']].mean().to_csv(A + '/scale_summary.csv')
rf = [json.load(open(f)) for f in sorted(glob.glob(OUT + '/rf_*.json'))]
if rf:
    pd.DataFrame([dict(relax_and_fix=r['relax_and_fix']['total_cost'], full_milp=r['full_milp']['total_cost'],
                       rf_time=r['relax_and_fix']['solve_time'], full_time=r['full_milp']['solve_time']) for r in rf]).to_csv(A + '/relax_and_fix_check.csv', index=False)
with open(A + '/summary.txt', 'w') as fh:
    fh.write(f"replications: {n_seeds}\n\nMAIN TABLE\n{tab.to_string()}\n\nCONTRASTS (cost)\n{C.round(4).to_string()}\n\nCONTRASTS (fill)\n{fillc.round(4).to_string()}\n\n")
    fh.write(f"FORECAST ACCURACY\n{ACC.round(2).to_string()}\n\nTHETAS\n{TH.groupby('mode').agg(['mean', 'std']).round(3).to_string()}\n")
print(open(A + '/summary.txt').read())

# ---------------- Figure: multi-criteria trade-offs (cost, service, waste, emissions)
TM = ['STATIC-AGG', 'FTO-AGG', 'DFL-AGG', 'BC-AGG', 'ORACLE-AGG', 'FTO-CHAR', 'DFL-CHAR', 'BC-CHAR', 'ORACLE-CHAR']
Tr = D[['seed', 'method', 'total_cost', 'fill', 'waste_pct', 'emissions_t']].copy()
if os.path.exists(A + '/bc_runs.csv'):
    Tr = pd.concat([Tr, pd.read_csv(A + '/bc_runs.csv')[['seed', 'method', 'total_cost', 'fill', 'waste_pct', 'emissions_t']]])
Tr = Tr[Tr.method.isin(TM)]
G = Tr.groupby('method').mean().reindex(TM).dropna()
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
def scatter_pair(ax, xc, yc, xl, yl):
    for m in G.index:
        sub = Tr[Tr.method == m]; col = mcol(m)
        ax.scatter(sub[xc], sub[yc], s=7, color=col, alpha=0.25, linewidths=0)
    for fam, ls in [('AGG', (0, (3, 2))), ('CHAR', '-')]:   # connect FTO -> DFL -> ORACLE within a model
        seq = [m for m in [f'FTO-{fam}', f'DFL-{fam}', f'ORACLE-{fam}'] if m in G.index]
        ax.plot(G.loc[seq, xc], G.loc[seq, yc], color=NAVY if fam == 'CHAR' else '#6f7f88', lw=1.3, ls=ls, zorder=2)
    for m in G.index:
        ax.scatter(G.loc[m, xc], G.loc[m, yc], s=46, marker=mmark(m), color=mcol(m), edgecolors=(NAVY if 'CHAR' in m else 'white'), linewidths=0.9, zorder=3)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
scatter_pair(axs[0], 'fill', 'total_cost', 'Fill rate (%)', 'Total cost')
axs[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f'{v/1000:.0f}k'))
panel(axs[0], 'a', 'Cost vs service')
scatter_pair(axs[1], 'waste_pct', 'emissions_t', 'Waste (% of demand)', 'Emissions (t CO$_2$e)')
panel(axs[1], 'b', 'Waste vs emissions')
# panel c: normalized KPI profile (0 = best, 1 = worst among shown methods) as a dot matrix
KP = ['total_cost', 'fill', 'waste_pct', 'emissions_t']; KL = ['Cost', 'Fill rate', 'Waste', 'Emissions']
show = [m for m in ['FTO-AGG', 'DFL-AGG', 'ORACLE-AGG', 'FTO-CHAR', 'BC-CHAR', 'DFL-CHAR', 'ORACLE-CHAR'] if m in G.index]
Z = G.loc[show, KP].copy(); Z['fill'] = -Z['fill']
Z = (Z - Z.min()) / (Z.max() - Z.min())
ax = axs[2]
for i, m in enumerate(show):
    for j, k in enumerate(KP):
        v = Z.loc[m, k]
        ax.scatter(j, i, s=40 + 260 * (1 - v), color=DIV(0.12 + 0.76 * v), edgecolors='white', linewidths=0.6)
ax.set_xticks(range(len(KP))); ax.set_xticklabels(KL, rotation=30, ha='right'); ax.set_yticks(range(len(show))); ax.set_yticklabels(show, fontsize=7)
ax.set_xlim(-0.6, len(KP) - 0.4); ax.set_ylim(len(show) - 0.5, -0.5); ax.grid(False)
for sp in ['left', 'bottom']: ax.spines[sp].set_visible(False)
ax.tick_params(length=0)
panel(ax, 'c', 'Relative performance')
from matplotlib.lines import Line2D
hand = [Line2D([], [], marker='o', ls='', color=TEAL, label='FTO / static'), Line2D([], [], marker='^', ls='', color=SALMON, label='Bias-corrected'),
        Line2D([], [], marker='s', ls='', color=GREEN, label='Decision-focused'), Line2D([], [], marker='D', ls='', color=BRICK, label='Perfect information'),
        Line2D([], [], color=NAVY, lw=1, label='Physics (CHAR): edged, solid'), Line2D([], [], color='#7d8b93', lw=1, ls=(0, (3, 2)), label='Aggregated: dashed')]
fig.legend(handles=hand, loc='lower center', ncol=3, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.11, 1, 1)); fig.savefig(A + '/fig_tradeoffs.png'); fig.savefig(A + '/fig_tradeoffs.eps'); plt.close(fig)
