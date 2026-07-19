#!/usr/bin/env python3
"""
fdtd_thermal_mxv_v1.py
========================
2-D acoustic MxV using temperature (hence sound speed) as the design field.

Design variables: heat-source strengths sigma_k in a grid of blocks.
Thermal physics:  steady-state Poisson equation  nabla^2 T = -sigma / D,
                  with Dirichlet T = T0 on all boundaries.
Acoustic physics: lossless FDTD on a staggered grid with spatially varying
                  sound speed c(x,y) = c0 * sqrt(T(x,y)/T0).

The effective real weight matrix is extracted by lock-in detection, exactly as
in fdtd_zk_mxv_v2.py.  Because temperature only changes phase velocity, the
medium can route/recombine waves by phase without absorbing energy.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import differential_evolution
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu


# -----------------------------------------------------------------------------
# Domain and physical constants
# -----------------------------------------------------------------------------
L = 0.50          # domain size (m)
DX = 0.004        # grid spacing (m)
NX = int(L / DX)
NY = NX
RHO0 = 1.225
T0 = 300.0        # ambient temperature (K)
C0 = 343.0        # sound speed at T0 (m/s)
T_MAX = 500.0     # maximum allowed temperature (K)
C_MAX = C0 * np.sqrt(T_MAX / T0)
D_THERMAL = 2.0e-5   # thermal diffusivity of air (m^2/s)
FREQ = 2000.0
OMEGA = 2.0 * np.pi * FREQ
SRC_AMP = 1.0
N_STEADY_PERIODS = 12
N_MEASURE_PERIODS = 3

# Source / probe geometry
SRC_X = 0.08
PROBE_X = 0.42
SRC_Y_OFFSETS = np.array([0.25, 0.75]) * L
PROBE_Y_OFFSETS = np.array([0.35, 0.65]) * L

# Heat-source design grid
N_BLOCKS_X = 4
N_BLOCKS_Y = 4
BLOCK_I0 = int(0.18 / DX)
BLOCK_I1 = int(0.38 / DX)
BLOCK_J0 = int(0.10 / DX)
BLOCK_J1 = int(0.90 / DX)

SIGMA_MIN = 0.0
SIGMA_MAX = 5.0e6   # W/m^3


# -----------------------------------------------------------------------------
# Thermal Poisson solver
# -----------------------------------------------------------------------------
def _build_poisson_matrix():
    """Build sparse Laplacian with Dirichlet BCs on all boundaries."""
    nx, ny = NX, NY
    n = nx * ny
    row_ind, col_ind, data = [], [], []

    def idx(i, j):
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            k = idx(i, j)
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                row_ind.append(k)
                col_ind.append(k)
                data.append(1.0)
                continue

            row_ind.append(k); col_ind.append(k); data.append(-4.0 / (DX * DX))
            row_ind.append(k); col_ind.append(idx(i + 1, j)); data.append(1.0 / (DX * DX))
            row_ind.append(k); col_ind.append(idx(i - 1, j)); data.append(1.0 / (DX * DX))
            row_ind.append(k); col_ind.append(idx(i, j + 1)); data.append(1.0 / (DX * DX))
            row_ind.append(k); col_ind.append(idx(i, j - 1)); data.append(1.0 / (DX * DX))

    A = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
    return splu(A)


_LU_POISSON = _build_poisson_matrix()


def solve_temperature(sigma_field):
    """Solve nabla^2 T = -sigma / D, Dirichlet T=T0 on boundaries."""
    b = -sigma_field.flatten() / D_THERMAL
    for i in range(NX):
        for j in range(NY):
            k = i * NY + j
            if i == 0 or i == NX - 1 or j == 0 or j == NY - 1:
                b[k] = T0
    T_flat = _LU_POISSON.solve(b)
    return T_flat.reshape(NX, NY)


def build_sigma_field(params):
    """Map block strengths to a full grid."""
    sigma = np.zeros((NX, NY))
    block_strengths = np.asarray(params).reshape(N_BLOCKS_X, N_BLOCKS_Y)
    block_strengths = SIGMA_MIN + (SIGMA_MAX - SIGMA_MIN) * np.clip(block_strengths, 0.0, 1.0)

    i_edges = np.linspace(BLOCK_I0, BLOCK_I1, N_BLOCKS_X + 1, dtype=int)
    j_edges = np.linspace(BLOCK_J0, BLOCK_J1, N_BLOCKS_Y + 1, dtype=int)

    for bi in range(N_BLOCKS_X):
        for bj in range(N_BLOCKS_Y):
            sigma[i_edges[bi]:i_edges[bi+1], j_edges[bj]:j_edges[bj+1]] = block_strengths[bi, bj]
    return sigma


def build_sound_speed(T):
    return C0 * np.sqrt(T / T0)


# -----------------------------------------------------------------------------
# Acoustic FDTD with spatially varying c
# -----------------------------------------------------------------------------
class ThermalFDTD2D:
    def __init__(self):
        self.p = np.zeros((NX, NY))
        self.u = np.zeros((NX + 1, NY))
        self.v = np.zeros((NX, NY + 1))
        self.c = np.ones((NX, NY)) * C0
        self.dt = 0.85 * DX / (C_MAX * np.sqrt(2.0))
        self.time = 0.0
        self.n_step = 0

    def set_sound_speed(self, c_field):
        self.c = c_field.copy()
        cmax = self.c.max()
        dt_safe = 0.85 * DX / (cmax * np.sqrt(2.0))
        if dt_safe < self.dt:
            self.dt = dt_safe

    def step(self):
        dt = self.dt
        dx = DX
        c = self.c

        p_old_left = self.p[0, :].copy()
        p_old_right = self.p[-1, :].copy()
        p_old_bottom = self.p[:, 0].copy()
        p_old_top = self.p[:, -1].copy()

        u_new = self.u.copy()
        v_new = self.v.copy()

        # velocity updates (rho0 = constant)
        dp_dx = (self.p[1:, :] - self.p[:-1, :]) / dx
        u_new[1:-1, :] = self.u[1:-1, :] - (dt / RHO0) * dp_dx

        dp_dy = (self.p[:, 1:] - self.p[:, :-1]) / dx
        v_new[:, 1:-1] = self.v[:, 1:-1] - (dt / RHO0) * dp_dy

        # pressure update with variable c^2
        du_dx = (u_new[1:, :] - u_new[:-1, :]) / dx
        dv_dy = (v_new[:, 1:] - v_new[:, :-1]) / dx
        p_new = self.p - dt * RHO0 * (self.c ** 2) * (du_dx + dv_dy)

        # Boundary conditions:
        #   top/bottom: hard wall (v = 0)
        #   left/right: characteristic (non-reflecting) using local sound speed
        v_new[:, 0] = 0.0
        v_new[:, -1] = 0.0
        u_new[0, :] = -p_new[0, :] / (RHO0 * c[0, :])
        u_new[-1, :] = p_new[-1, :] / (RHO0 * c[-1, :])

        self.p = p_new
        self.u = u_new
        self.v = v_new
        self.time += dt
        self.n_step += 1

    def apply_soft_source(self, src_value, src_mask):
        self.p[src_mask] += src_value


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
    sigma = build_sigma_field(params)
    T = solve_temperature(sigma)
    c = build_sound_speed(T)

    solver = ThermalFDTD2D()
    solver.set_sound_speed(c)

    A = measure_complex_amplitudes(solver, N_STEADY_PERIODS, N_MEASURE_PERIODS)
    A_real = A.real

    mse = np.mean((A_real - W_target) ** 2)

    # regularisation: penalise excessive heating
    heat_penalty = 1e-10 * np.mean(sigma ** 2)

    total = mse + heat_penalty

    if return_details:
        return total, A, A_real, T, c
    return total


def optimise(W_target, maxiter=20, popsize=4, workers=1, seed=42):
    n_params = N_BLOCKS_X * N_BLOCKS_Y
    bounds = [(0.0, 1.0)] * n_params

    result = differential_evolution(
        loss_function,
        bounds,
        args=(W_target,),
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-6,
        polish=False,
        workers=workers,
        seed=seed,
        disp=True,
    )
    return result


def plot_result(W_target, A_real, params, T, c, outname='fdtd_thermal_mxv_v1_result.png'):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    sigma = build_sigma_field(params)
    block_strengths = np.asarray(params).reshape(N_BLOCKS_X, N_BLOCKS_Y)

    im0 = axes[0, 0].imshow(block_strengths, origin='lower', cmap='hot', vmin=0, vmax=1)
    axes[0, 0].set_title('Heat source strength (normalised)')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(T.T, origin='lower', extent=[0, L*1000, 0, L*1000], cmap='hot')
    axes[0, 1].set_title('Temperature T (K)')
    axes[0, 1].set_xlabel('x (mm)')
    axes[0, 1].set_ylabel('y (mm)')
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(c.T, origin='lower', extent=[0, L*1000, 0, L*1000], cmap='viridis')
    axes[0, 2].set_title('Sound speed c (m/s)')
    axes[0, 2].set_xlabel('x (mm)')
    axes[0, 2].set_ylabel('y (mm)')
    plt.colorbar(im2, ax=axes[0, 2])

    # sample field from source 0
    solver = ThermalFDTD2D()
    solver.set_sound_speed(c)
    n_steps = int(np.round((N_STEADY_PERIODS + 1) / (FREQ * solver.dt)))
    src_mask = SOURCE_MASKS[0]
    for n in range(n_steps):
        t = n * solver.dt
        src = SRC_AMP * np.cos(OMEGA * t)
        solver.step()
        solver.apply_soft_source(src, src_mask)

    vmax = np.max(np.abs(solver.p))
    im3 = axes[1, 0].imshow(solver.p.T, origin='lower', extent=[0, L*1000, 0, L*1000],
                            cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1, 0].set_title('Sample field (source S0)')
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

    result = optimise(W_target, maxiter=20, popsize=4, workers=1, seed=42)

    print("\nOptimisation finished.", flush=True)
    print(f"Best MSE: {result.fun:.6f}", flush=True)

    _, A, A_real, T, c = loss_function(result.x, W_target, return_details=True)
    print("\nAchieved real matrix A:", flush=True)
    print(A_real, flush=True)
    print("\nAchieved complex matrix A:", flush=True)
    print(A, flush=True)

    plot_result(W_target, A_real, result.x, T, c)

    return result


if __name__ == '__main__':
    main()
