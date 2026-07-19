#!/usr/bin/env python3
"""
fdtd_zk_2d_test_v1.py
=====================
Quick smoke test of the 2-D ZK FDTD solver.

Case 1: Air domain, line source at left boundary — check wave propagation.
Case 2: Uniform porous domain — check attenuation and slower phase velocity.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from fdtd_zk_2d_v1 import ZKFDTD2D


# =============================================================================
# PARAMETERS
# =============================================================================
NX, NY = 240, 120
LX = 0.60
DX = LX / NX
SRC_FREQ = 2000.0
SRC_AMP = 1.0
N_PERIODS = 12

x = np.linspace(0, LX, NX)
y = np.linspace(0, LX * NY / NX, NY)

# Source mask: a short line source centred in y
src_width_cells = 4
src_y0 = NY // 2 - src_width_cells // 2
src_y1 = src_y0 + src_width_cells
src_mask = np.zeros(NY, dtype=bool)
src_mask[src_y0:src_y1] = True

probe_i = int(0.45 / DX)
probe_j = NY // 2


def source_func(n, dt):
    return SRC_AMP * np.sin(2.0 * np.pi * SRC_FREQ * n * dt)


def run_case(material):
    solver = ZKFDTD2D(NX, NY, DX)
    if material == 'air':
        pass  # default air
    elif material == 'porous':
        solver.set_uniform_material(phi=0.5, sigma=5e3, ks=1.5)

    # Interior soft source at x = 0.05 m
    src_i = int(0.05 / DX)
    src_region = np.zeros((NX, NY), dtype=bool)
    y_idx = np.arange(NY)
    sigma_y = 2.0  # cells
    window = np.exp(-0.5 * ((y_idx - NY // 2) / sigma_y) ** 2)
    src_region[src_i, :] = window > 0.01
    src_window = window[window > 0.01]

    n_steps = int(N_PERIODS / (SRC_FREQ * solver.dt))
    probe_history = np.zeros((1, n_steps))
    field_history = []

    for n in range(n_steps):
        solver.step()
        src = source_func(n, solver.dt)
        solver.p[src_region] += src * np.tile(src_window, (src_window.size, 1)).T.ravel()[:solver.p[src_region].size]
        solver.apply_right_mur()
        # Also apply Mur at left since source is interior
        solver.p[0, :] = solver.p[1, :] + (solver.c0 * solver.dt - solver.dx) / (solver.c0 * solver.dt + solver.dx) * (solver.p[0, :] - solver.p[1, :])

        probe_history[0, n] = solver.p[probe_i, probe_j]
        if n % (n_steps // 4) == 0:
            field_history.append(solver.p.copy())

    return probe_history, field_history


# =============================================================================
# RUN
# =============================================================================
print("Running air case ...")
probe_air, fields_air = run_case('air')
print("Running porous case ...")
probe_por, fields_por = run_case('porous')

t = np.arange(len(probe_air[0])) * (LX / NX / 343.0 * 0.95)  # approximate dt

# =============================================================================
# PLOTS
# =============================================================================
fig_dir = Path(__file__).parent / 'figures'
fig_dir.mkdir(exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# pressure traces
axes[0, 0].plot(t * 1000, probe_air[0], label='air')
axes[0, 0].plot(t * 1000, probe_por[0], label='porous')
axes[0, 0].set_xlabel('t (ms)')
axes[0, 0].set_ylabel('p (Pa)')
axes[0, 0].set_title('Probe pressure')
axes[0, 0].legend()
axes[0, 0].grid(True)

# amplitude envelope
axes[0, 1].semilogy(t * 1000, np.abs(probe_air[0]) + 1e-12, label='air')
axes[0, 1].semilogy(t * 1000, np.abs(probe_por[0]) + 1e-12, label='porous')
axes[0, 1].set_xlabel('t (ms)')
axes[0, 1].set_ylabel('|p| (Pa)')
axes[0, 1].set_title('Amplitude envelope')
axes[0, 1].legend()
axes[0, 1].grid(True)

# final field snapshots
for col, (fields, title) in enumerate([(fields_air, 'air'), (fields_por, 'porous')]):
    ax = axes[1, col]
    im = ax.imshow(fields[-1].T, origin='lower', extent=[0, LX*1000, 0, y[-1]*1000],
                   cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax.set_title(f'Final p field: {title}')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    fig.colorbar(im, ax=ax)

# comparison line profile through centre
axes[1, 2].plot(x * 1000, fields_air[-1][:, NY//2], label='air')
axes[1, 2].plot(x * 1000, fields_por[-1][:, NY//2], label='porous')
axes[1, 2].set_xlabel('x (mm)')
axes[1, 2].set_ylabel('p (Pa)')
axes[1, 2].set_title('Centre-line profile')
axes[1, 2].legend()
axes[1, 2].grid(True)

plt.tight_layout()
plt.savefig(fig_dir / 'fdtd_zk_2d_test_v1.png', dpi=150)
print(f"\nSaved: {fig_dir / 'fdtd_zk_2d_test_v1.png'}")
