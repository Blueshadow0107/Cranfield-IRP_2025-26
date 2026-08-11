"""
Plot the trained phi field from rd_optimizer_compare_left.json.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rd_tjunc_router import build_phi, WALL, NX, NY, run_pattern

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

with open(os.path.join(FIG, 'rd_optimizer_compare_left.json')) as fh:
    data = json.load(fh)

best_name = min(data, key=lambda k: data[k]['best_f'])
print(f"best optimiser: {best_name}  soft_loss={data[best_name]['best_f']:.4f}  hard_loss={data[best_name]['hard_loss']:.4f}")

blocks = np.array(data[best_name]['best_x'])
phi = build_phi(blocks)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
im = ax.imshow(phi.T, origin='lower', vmin=0.010, vmax=0.040,
               cmap='viridis', extent=[0, NX, 0, NY])
wall_x, wall_y = np.where(WALL)
ax.scatter(wall_x, wall_y, c='k', s=0.5, alpha=0.4)
ax.set_title(f'trained phi ({best_name})')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im, ax=ax, fraction=0.046)

# probe trace
t, left, right = run_pattern(phi, True, False)
ax = axes[1]
ax.plot(t, left, label='left probe')
ax.plot(t, right, label='right probe')
ax.axhline(0.5, color='r', ls='--', lw=1, label='threshold')
ax.set_xlabel('time (t.u.)')
ax.set_ylabel('probe max u')
ax.set_title('probe response to input A')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG, 'rd_tjunc_trained.png'), dpi=150)
print(f'[saved] {FIG}/rd_tjunc_trained.png')
