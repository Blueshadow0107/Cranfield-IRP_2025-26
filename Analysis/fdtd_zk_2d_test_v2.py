#!/usr/bin/env python3
"""
fdtd_zk_2d_test_v2.py
=====================
Smoke test for the 2-D Zwikker-Kosten FDTD solver v2.

A single Gaussian soft source is placed inside a square air domain and inside
a uniform porous domain.  We compare:
  - probe pressure time traces
  - amplitude envelopes
  - final pressure field snapshots
  - centre-line pressure profile

Expected behaviour:
  - air: fast, weakly attenuated wavefront
  - porous: slower, strongly attenuated wavefront
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from fdtd_zk_2d_v2 import ZKFDTD2D


# --- simulation parameters --------------------------------------------------
L = 0.60          # domain size (m)
DX = 0.002        # grid spacing (m)
NX = int(L / DX)
NY = NX
SRC_FREQ = 2000.0 # Hz
SRC_AMP = 1.0
N_PERIODS = 8
SPONGE_WIDTH = 15

# source: Gaussian line at x = 0.15 m, centred in y
src_i = int(0.15 / DX)
src_j = NY // 2
sigma_j = 3.0
j_idx = np.arange(NY)
window = np.exp(-0.5 * ((j_idx - src_j) / sigma_j) ** 2)
src_mask = np.zeros((NX, NY), dtype=bool)
src_mask[src_i, :] = window > 0.01
src_window = window[window > 0.01]

def source_func(n, dt):
    t = n * dt
    # Hanning-windowed tone burst
    T = N_PERIODS / SRC_FREQ
    if t >= T:
        return 0.0
    env = 0.5 * (1.0 - np.cos(2.0 * np.pi * t / T))
    return SRC_AMP * env * np.sin(2.0 * np.pi * SRC_FREQ * t)

# probe at x = 0.45 m, y = L/2
probe_i = int(0.45 / DX)
probe_j = NY // 2


def run_case(material):
    solver = ZKFDTD2D(NX, NY, DX)
    solver.set_sponge_layer(SPONGE_WIDTH, sigma_max=2e4)
    if material == 'air':
        pass
    elif material == 'porous':
        solver.set_uniform_material(phi=0.5, sigma=5e3, ks=1.5)

    n_steps = int((N_PERIODS + 4) / (SRC_FREQ * solver.dt))
    probe_history, field_history = solver.run(
        n_steps,
        source_func=lambda n: source_func(n, solver.dt),
        source_masks=[src_mask],
        record_interval=n_steps // 4,
        probe_coords=[(probe_i, probe_j)]
    )
    return probe_history, field_history, solver


def main():
    cases = {}
    for mat in ['air', 'porous']:
        print(f"Running {mat} case ...")
        probe_history, field_history, solver = run_case(mat)
        cases[mat] = (probe_history, field_history, solver)
        print(f"  max probe amplitude: {np.max(np.abs(probe_history)):.3e} Pa")
        print(f"  final field max    : {np.max(np.abs(field_history[-1])):.3e} Pa")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    t_air = np.arange(cases['air'][2].n_step) * cases['air'][2].dt * 1000
    t_porous = np.arange(cases['porous'][2].n_step) * cases['porous'][2].dt * 1000

    # time trace
    ax = axes[0, 0]
    ax.plot(t_air, cases['air'][0][0, :len(t_air)], label='air')
    ax.plot(t_porous, cases['porous'][0][0, :len(t_porous)], label='porous')
    ax.set_xlabel('t (ms)')
    ax.set_ylabel('p (Pa)')
    ax.set_title('Probe pressure')
    ax.legend()
    ax.grid(True)

    # envelope
    ax = axes[0, 1]
    ax.semilogy(t_air, np.abs(cases['air'][0][0, :len(t_air)]), label='air')
    ax.semilogy(t_porous, np.abs(cases['porous'][0][0, :len(t_porous)]), label='porous')
    ax.set_xlabel('t (ms)')
    ax.set_ylabel('|p| (Pa)')
    ax.set_title('Amplitude envelope')
    ax.legend()
    ax.grid(True)

    # final field snapshots
    for idx, mat in enumerate(['air', 'porous']):
        ax = axes[1, idx]
        field = cases[mat][1][-1]
        vmax = np.max(np.abs(field))
        if vmax == 0.0 or not np.isfinite(vmax):
            vmax = 1.0
        im = ax.imshow(field.T, origin='lower', extent=[0, L * 1000, 0, L * 1000],
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(f'Final p field: {mat}')
        plt.colorbar(im, ax=ax)

    # centre-line profile
    ax = axes[1, 2]
    x_mm = np.arange(NX) * DX * 1000
    for mat in ['air', 'porous']:
        profile = cases[mat][1][-1][:, NY // 2]
        ax.plot(x_mm, profile, label=mat)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('p (Pa)')
    ax.set_title('Centre-line profile')
    ax.legend()
    ax.grid(True)

    # hide unused top-right axes
    axes[0, 2].axis('off')

    plt.tight_layout()
    outdir = Path('figures')
    outdir.mkdir(exist_ok=True)
    outpath = outdir / 'fdtd_zk_2d_test_v2.png'
    plt.savefig(outpath, dpi=150)
    print(f"Saved figure to {outpath}")


if __name__ == '__main__':
    main()
