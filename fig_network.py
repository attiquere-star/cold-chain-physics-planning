import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8})
fig, ax = plt.subplots(figsize=(6.8, 3.0), dpi=300); ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis('off')
INK='#063D59'; MUT='#3d4a52'; BLUE='#0B5C79'; ORANGE='#B96263'
def box(x, y, w, h, t, fc='#EAF5F7', ec=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc=fc, ec=ec, lw=1))
    ax.text(x + w/2, y + h/2, t, ha='center', va='center', color=INK)
def arr(x0, y0, x1, y1, col=INK, ls='-'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=9, color=col, lw=1.2, ls=ls))
for i, y in enumerate([3.4, 2.2, 1.0]):
    box(0.1, y, 1.2, 0.6, f'Supplier {i+1}'); arr(1.3, y + 0.3, 2.0, 2.2 + 0.3 + (y - 2.2) * 0.25)
ax.text(1.65, 3.95, 'procure $q_s$\nactivate $A_s$', ha='center', fontsize=7, color=MUT)
box(2.0, 1.9, 1.3, 1.2, 'Plant\nraw stock $R$\nproduce $X_p$')
# long-haul route as cells
x0 = 3.5
for j in range(5):
    ax.add_patch(plt.Rectangle((x0 + j * 0.42, 2.3), 0.4, 0.4, fc='#A8D8E6', ec=BLUE, lw=0.8))
arr(3.3, 2.5, 3.5, 2.5); arr(x0 + 5 * 0.42, 2.5, 5.85, 2.5)
ax.text(x0 + 1.05, 2.85, 'long-haul route, $N_0$ trucks/day\n$\\partial_t F + v\\,\\partial_x F = -\\xi(T)F$', ha='center', fontsize=7, color=INK)
ax.text(x0 + 1.05, 2.0, 'cells $\\Delta x$, CFL: $v\\Delta t/\\Delta x\\leq 1$', ha='center', fontsize=7, color=MUT)
# DC with age classes
ax.add_patch(FancyBboxPatch((5.85, 1.5), 1.3, 2.0, boxstyle='round,pad=0.02,rounding_size=0.08', fc='#EAF5F7', ec=INK, lw=1))
ax.text(6.5, 3.25, 'Cold DC', ha='center', color=INK)
for a in range(5):
    ax.add_patch(plt.Rectangle((6.0, 1.65 + a * 0.29), 1.0, 0.25, fc='#F2D3C8', ec=ORANGE, lw=0.8))
ax.text(6.5, 1.65 + 0.12, 'oldest: expiry', ha='center', va='center', fontsize=6, color=INK)
ax.text(6.5, 1.65 + 4 * 0.29 + 0.12, 'age 0', ha='center', va='center', fontsize=6, color=INK)
ax.text(6.5, 1.3, '$\\partial_t I + \\partial_a I = -\\lambda(T) I - w$', ha='center', fontsize=7, color=INK)
for i, y in enumerate([3.6, 2.6, 1.6, 0.6]):
    box(8.6, y, 1.3, 0.55, f'Customer {i+1}')
    arr(7.15, 2.5, 8.6, y + 0.27, col=BLUE)
ax.text(7.75, 3.95, 'regional routes\nship $z_{c,p}$, $N_c$ trucks', ha='center', fontsize=7, color=MUT)
ax.text(9.25, 0.2, 'stock $S$, backlog $B$,\ndemand $d$', ha='center', fontsize=7, color=MUT)
box(1.8, 0.1, 5.8, 0.6, 'LSTM forecasts of $d$, $v$, $T$  →  (recalibration $\\theta$)  →  daily MILP', fc='#ffffff', ec=MUT)
fig.tight_layout(); fig.savefig('out/analysis/fig_network.png'); fig.savefig('out/analysis/fig_network.eps'); print('ok')
