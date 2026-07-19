#!/usr/bin/env python3
"""
fdtd_zk_mxv_v2.py
=================
Faster 2-D ZK signed 2x2 MxV benchmark.

Same physics as v1, but with a coarser grid, a 3x3 design-parameter grid,
and a shorter steady-state measurement window so that differential evolution
finishes in a reasonable time.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import differential_evolution

from fdtd_zk_2d_v2 import ZKFDTD2D


# -----------------------------------------------------------------------------
# Coarse but adequate domain settings
# -----------------------------------------------------------------------------
L = 0.50          # domain size (m)
DX = 0.004        # grid spacing (m) -> ~43 cells/wavelength in air at 2 kHz
NX = int(L / DX)
NY = NX
C0 = 343.0
FREQ = 2000.0
OMEGA = 2.0 * np.pi * FREQ
SRC_AMP = 1.0
N_STEADY_PERIODS = 12
N_MEASURE_PERIODS = 3
SPONGE_WIDTH = 10

SRC_X = 0.08
PROBE_X = 0.42
SRC_Y_OFFSETS = np.array([0.25, 0.75]) * L
PROBE_Y_OFFSETS = np.array([0.35, 0.65]) * L

DESIGN_X0 = 0.15
DESIGN_X1 = 0.35
DESIGN_I0 = int(DESIGN_X0 / DX)
DESIGN_I1 = int(DESIGN_X1 / DX)
DESIGN_J0 = 0
DESIGN_J1 = NY

N_BLOCKS_X = 3
N_BLOCKS_Y = 3


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


def build_probe_coords():
    probe_i = int(PROBE_X / DX)
    coords = []
    for y0 in PROBE_Y_OFFSETS:
        probe_j = int(y0 / DX)
        coords.append((probe_i, probe_j))
    return coords


SOURCE_MASKS = build_source_masks()
PROBE_COORDS = build_probe_coords()


def design_to_material(solver, params):
    n = N_BLOCKS_X * N_BLOCKS_Y
    phi = params[0*n:1*n].reshape(N_BLOCKS_X, N_BLOCKS_Y)
    sigma = params[1*n:2*n].reshape(N_BLOCKS_X, N_BLOCKS_Y)
    ks = params[2*n:3*n].reshape(N_BLOCKS_X, N_BLOCKS_Y)
    solver.set_material_from_design(
        phi, sigma, ks,
        DESIGN_I0, DESIGN_I1, DESIGN_J0, DESIGN_J1,
        order=1
    )


def measure_complex_amplitudes(solver, n_steady, n_measure):
    dt = solver.dt
    n_measure_steps = int(np.round(n_measure / (FREQ * dt)))
    n_total_steps = int(np.round(n_steady / (FREQ * dt))) + n_measure_steps

    A = np.zeros((len(PROBE_COORDS), len(SOURCE_MASKS)), dtype=complex)

    for j_src, src_mask in enumerate(SOURCE_MASKS):
        solver.p[:] = 0.0
        solver.u[:] = 0.0
        solver.v[:] = 0.0
        solver.time = 0.0
        solver.n_step = 0

        probe_history = np.zeros((len(PROBE_COORDS), n_total_steps))

        for n in range(n_total_steps):
            t = n * dt
            src = SRC_AMP * np.cos(OMEGA * t)
            ramp_t = 2.0 / FREQ
            if t < ramp_t:
                src *= 0.5 * (1.0 - np.cos(np.pi * t / ramp_t))

            solver.step()
            solver.apply_soft_source(src, src_mask)

            for k, (i, j) in enumerate(PROBE_COORDS):
                probe_history[k, n] = solver.p[i, j]

        t_win = np.arange(n_measure_steps) * dt
        cos_ref = np.cos(OMEGA * t_win)
        sin_ref = np.sin(OMEGA * t_win)
        norm = 2.0 / n_measure_steps

        for k in range(len(PROBE_COORDS)):
            sig = probe_history[k, -n_measure_steps:]
            X = norm * np.sum(sig * cos_ref)
            Y = norm * np.sum(sig * sin_ref)
            A[k, j_src] = (X - 1j * Y) / SRC_AMP

    return A


def loss_function(params, W_target, return_details=False):
    solver = ZKFDTD2D(NX, NY, DX)
    solver.set_sponge_layer(SPONGE_WIDTH, sigma_max=2e4)
    design_to_material(solver, params)

    A = measure_complex_amplitudes(solver, N_STEADY_PERIODS, N_MEASURE_PERIODS)
    A_real = A.real

    mse = np.mean((A_real - W_target) ** 2)

    # small smoothness regularisation on porosity
    n = N_BLOCKS_X * N_BLOCKS_Y
    phi = params[0*n:1*n].reshape(N_BLOCKS_X, N_BLOCKS_Y)
    grad = np.sum(np.abs(np.diff(phi, axis=0))) + np.sum(np.abs(np.diff(phi, axis=1)))
    reg = 2e-4 * grad

    total = mse + reg

    if return_details:
        return total, A, A_real
    return total


def optimise(W_target, maxiter=15, popsize=4, workers=-1, seed=42, polish=False):
    n_params = 3 * N_BLOCKS_X * N_BLOCKS_Y
    bounds = [(0.0, 1.0)] * n_params

    result = differential_evolution(
        loss_function,
        bounds,
        args=(W_target,),
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-6,
        polish=polish,
        workers=workers,
        seed=seed,
        disp=True,
        updating='deferred' if workers != 1 else 'immediate',
    )
    return result


def plot_result(W_target, A_real, params, outname='fdtd_zk_mxv_v2_result.png'):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    n = N_BLOCKS_X * N_BLOCKS_Y
    phi = params[0*n:1*n].reshape(N_BLOCKS_X, N_BLOCKS_Y)
    sigma = params[1*n:2*n].reshape(N_BLOCKS_X, N_BLOCKS_Y)
    ks = params[2*n:3*n].reshape(N_BLOCKS_X, N_BLOCKS_Y)

    im0 = axes[0, 0].imshow(phi, origin='lower', cmap='viridis', vmin=0, vmax=1)
    axes[0, 0].set_title('phi (normalised)')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(sigma, origin='lower', cmap='inferno', vmin=0, vmax=1)
    axes[0, 1].set_title('sigma (normalised)')
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(ks, origin='lower', cmap='plasma', vmin=0, vmax=1)
    axes[0, 2].set_title('k_s (normalised)')
    plt.colorbar(im2, ax=axes[0, 2])

    solver = ZKFDTD2D(NX, NY, DX)
    solver.set_sponge_layer(SPONGE_WIDTH, sigma_max=2e4)
    design_to_material(solver, params)
    phi_phys = (solver.phi - 0.30) / 0.65
    im3 = axes[1, 0].imshow(phi_phys.T, origin='lower', extent=[0, L*1000, 0, L*1000],
                            cmap='viridis', vmin=0, vmax=1)
    axes[1, 0].set_title('phi physical (fine grid)')
    axes[1, 0].set_xlabel('x (mm)')
    axes[1, 0].set_ylabel('y (mm)')
    plt.colorbar(im3, ax=axes[1, 0])

    ax = axes[1, 1]
    mat = np.hstack([W_target, A_real])
    im = ax.matshow(mat, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title('Target vs Achieved')
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['T0', 'T1', 'A0', 'A1'])
    ax.set_yticks([0, 1])
    plt.colorbar(im, ax=ax)

    ax = axes[1, 2]
    err = A_real - W_target
    im = ax.matshow(err, cmap='RdBu_r')
    ax.set_title('Error')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    outdir = Path('figures')
    outdir.mkdir(exist_ok=True)
    outpath = outdir / outname
    plt.savefig(outpath, dpi=150)
    print(f"Saved result figure to {outpath}")


def main():
    W_target = np.array([
        [0.60, -0.40],
        [-0.20, 0.80]
    ])

    print("Target matrix:", flush=True)
    print(W_target, flush=True)
    print("\nStarting optimisation ...", flush=True)

    result = optimise(W_target, maxiter=20, popsize=4, workers=-1, seed=42)

    print("\nOptimisation finished.", flush=True)
    print(f"Best MSE: {result.fun:.6f}", flush=True)

    _, A, A_real = loss_function(result.x, W_target, return_details=True)
    print("\nAchieved real matrix A:", flush=True)
    print(A_real, flush=True)
    print("\nAchieved complex matrix A:", flush=True)
    print(A, flush=True)

    plot_result(W_target, A_real, result.x)

    return result


if __name__ == '__main__':
    main()
