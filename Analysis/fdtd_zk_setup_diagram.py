#!/usr/bin/env python3
"""
fdtd_zk_setup_diagram.py
========================
Draw a schematic of the 2-D ZK MxV domain: sponge layers, design region,
source ports, output probes, plus a sample steady-state pressure field.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

from fdtd_zk_2d_v2 import ZKFDTD2D


# Same geometry as fdtd_zk_mxv_v2.py
L = 0.50
DX = 0.004
NX = int(L / DX)
NY = NX
FREQ = 2000.0
OMEGA = 2.0 * np.pi * FREQ
SRC_AMP = 1.0
SPONGE_WIDTH = 10
N_STEADY = 12
N_MEASURE = 3

SRC_X = 0.08
PROBE_X = 0.42
SRC_Y_OFFSETS = np.array([0.25, 0.75]) * L
PROBE_Y_OFFSETS = np.array([0.35, 0.65]) * L
DESIGN_X0, DESIGN_X1 = 0.15, 0.35
DESIGN_Y0, DESIGN_Y1 = 0.0, L


def build_source_masks():
    masks = []
    src_i = int(SRC_X / DX)
    sigma_j = 2.5
    j_idx = np.arange(NY)
    for y0 in SRC_Y_OFFSETS:
        src_j = int(y0 / DX)
        window = np.exp(-0.5 * ((j_idx - src_j) / sigma_j) ** 2)
        mask = np.zeros((NX, NY), dtype=bool)
        mask[src_i, :] = window > 0.01
        masks.append(mask)
    return masks


def get_sample_field(src_idx=0):
    """Run one source to steady state and return the pressure field."""
    solver = ZKFDTD2D(NX, NY, DX)
    solver.set_sponge_layer(SPONGE_WIDTH, sigma_max=2e4)
    src_masks = build_source_masks()

    dt = solver.dt
    n_measure_steps = int(np.round(N_MEASURE / (FREQ * dt)))
    n_total_steps = int(np.round(N_STEADY / (FREQ * dt))) + n_measure_steps

    for n in range(n_total_steps):
        t = n * dt
        src = SRC_AMP * np.cos(OMEGA * t)
        ramp_t = 2.0 / FREQ
        if t < ramp_t:
            src *= 0.5 * (1.0 - np.cos(np.pi * t / ramp_t))
        solver.step()
        solver.apply_soft_source(src, src_masks[src_idx])

    return solver.p


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- left panel: schematic layout ---
    ax = axes[0]
    ax.set_xlim(0, L * 1000)
    ax.set_ylim(0, L * 1000)
    ax.set_aspect('equal')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('Domain layout')

    # sponge layers
    sponge_mm = SPONGE_WIDTH * DX * 1000
    ax.add_patch(Rectangle((0, 0), L*1000, sponge_mm, color='lightgray', alpha=0.5, label='sponge'))
    ax.add_patch(Rectangle((0, L*1000 - sponge_mm), L*1000, sponge_mm, color='lightgray', alpha=0.5))
    ax.add_patch(Rectangle((0, 0), sponge_mm, L*1000, color='lightgray', alpha=0.5))
    ax.add_patch(Rectangle((L*1000 - sponge_mm, 0), sponge_mm, L*1000, color='lightgray', alpha=0.5))

    # design region
    ax.add_patch(Rectangle((DESIGN_X0*1000, DESIGN_Y0*1000),
                           (DESIGN_X1 - DESIGN_X0)*1000,
                           (DESIGN_Y1 - DESIGN_Y0)*1000,
                           edgecolor='green', facecolor='lightgreen', alpha=0.3,
                           linewidth=2, label='design region'))

    # sources
    for idx, y0 in enumerate(SRC_Y_OFFSETS):
        ax.plot(SRC_X*1000, y0*1000, 'o', color='blue', markersize=12,
                markeredgecolor='navy', label='source' if idx == 0 else '')
        ax.text(SRC_X*1000 - 20, y0*1000 + 15, f'S{idx}', color='blue', fontsize=11)

    # probes
    for idx, y0 in enumerate(PROBE_Y_OFFSETS):
        ax.plot(PROBE_X*1000, y0*1000, 's', color='red', markersize=12,
                markeredgecolor='darkred', label='probe' if idx == 0 else '')
        ax.text(PROBE_X*1000 + 15, y0*1000 + 15, f'P{idx}', color='red', fontsize=11)

    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- right panel: sample steady-state field ---
    p = get_sample_field(src_idx=0)
    vmax = np.max(np.abs(p))
    ax = axes[1]
    im = ax.imshow(p.T, origin='lower', extent=[0, L*1000, 0, L*1000],
                   cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(f'Sample steady-state field (source S0, f={FREQ/1000:.1f} kHz)')
    plt.colorbar(im, ax=ax, label='p (Pa)')

    # overlay source/probe positions
    for idx, y0 in enumerate(SRC_Y_OFFSETS):
        ax.plot(SRC_X*1000, y0*1000, 'o', color='blue', markersize=8,
                markeredgecolor='white')
    for idx, y0 in enumerate(PROBE_Y_OFFSETS):
        ax.plot(PROBE_X*1000, y0*1000, 's', color='red', markersize=8,
                markeredgecolor='white')

    plt.tight_layout()
    outdir = Path('figures')
    outdir.mkdir(exist_ok=True)
    outpath = outdir / 'fdtd_zk_setup_diagram.png'
    plt.savefig(outpath, dpi=150)
    print(f"Saved diagram to {outpath}")


if __name__ == '__main__':
    main()
