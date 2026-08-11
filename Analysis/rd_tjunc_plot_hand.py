"""
Plot the hand-designed phi fields from rd_tjunc_hand_tests.json.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rd_tjunc_router import build_phi, WALL, NX, NY

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

with open(os.path.join(FIG, 'rd_tjunc_hand_tests.json')) as fh:
    data = json.load(fh)

names = ['uniform_phi0', 'left_block_high', 'right_block_high', 'top_block_high']
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes = axes.ravel()
for ax, name in zip(axes, names):
    blocks = data[name]['blocks']
    phi = build_phi(blocks)
    im = ax.imshow(phi.T, origin='lower', vmin=0.010, vmax=0.040,
                   cmap='viridis', extent=[0, NX, 0, NY])
    wall_x, wall_y = np.where(WALL)
    ax.scatter(wall_x, wall_y, c='k', s=0.5, alpha=0.4)
    ax.set_title(name.replace('_', ' '))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'rd_tjunc_hand_phi_maps.png'), dpi=150)
print(f'[saved] {FIG}/rd_tjunc_hand_phi_maps.png')
