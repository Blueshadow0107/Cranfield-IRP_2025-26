#!/usr/bin/env python3
"""
acoustic_filter_fdtd_thermal.py
==============================
Acoustic FDTD with STEADY-STATE temperature solve.

Trainable parameters: heat source strengths sigma_k [W/m^3]
Thermal physics:      solve Poisson equation for T
Acoustic physics:     run FDTD with c(T)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

# =============================================================================
# Grid and physical parameters
# =============================================================================
NX, NY = 250, 50
LX, LY = 0.5, 0.1
DX = LX / NX
DY = LY / NY

RHO0 = 1.2
T0 = 300.0
C0 = 343.0
T_MAX = 500.0
C_MAX = C0 * np.sqrt(T_MAX / T0)

# Thermal parameters for air
D_THERMAL = 2.0e-5          # thermal diffusivity [m^2/s]
K_THERMAL = 0.026           # thermal conductivity [W/(m*K)]
# Heat source strength bounds
SIGMA_MIN = 0.0
SIGMA_MAX = 5.0e6           # W/m^3, generous upper bound

# CFL
CFL_FACTOR = 0.5
DT = CFL_FACTOR * DX / (C_MAX * np.sqrt(2))

print(f"[THERMAL] dx = {DX:.3e} m, dt = {DT:.3e} s")
print(f"[THERMAL] Thermal diffusivity D = {D_THERMAL:.3e} m^2/s")

# =============================================================================
# Heater block geometry
# =============================================================================
N_BLOCKS = 8
N_ROWS = 2
N_COLS = 4

BLOCK_W = NX // 12
BLOCK_H = NY // 3
GAP_X = NX // 10
GAP_Y = NY // 5
START_X = NX // 5
START_Y = NY // 5

BLOCK_BOUNDS = []
for row in range(N_ROWS):
    for col in range(N_COLS):
        x0 = START_X + col * (BLOCK_W + GAP_X)
        x1 = x0 + BLOCK_W
        y0 = START_Y + row * (BLOCK_H + GAP_Y)
        y1 = y0 + BLOCK_H
        BLOCK_BOUNDS.append((x0, x1, y0, y1))

print(f"[THERMAL] {N_ROWS}x{N_COLS} = {N_BLOCKS} blocks")

# =============================================================================
# Build Poisson matrix for steady-state heat equation
# =============================================================================
# We solve:  nabla^2 T = -sigma / D
# With Dirichlet BC: T = T0 on all boundaries

def build_poisson_matrix(nx, ny, dx, dy):
    """Build sparse Laplacian matrix with Dirichlet BCs."""
    n = nx * ny
    row_ind = []
    col_ind = []
    data = []

    def idx(i, j):
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            k = idx(i, j)
            # Boundary: Dirichlet T = T0
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                row_ind.append(k)
                col_ind.append(k)
                data.append(1.0)
                continue

            # Interior: 5-point Laplacian
            # (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1] - 4*T[i,j]) / dx^2
            row_ind.append(k)
            col_ind.append(k)
            data.append(-4.0 / (dx * dx))

            row_ind.append(k)
            col_ind.append(idx(i + 1, j))
            data.append(1.0 / (dx * dx))

            row_ind.append(k)
            col_ind.append(idx(i - 1, j))
            data.append(1.0 / (dx * dx))

            row_ind.append(k)
            col_ind.append(idx(i, j + 1))
            data.append(1.0 / (dx * dx))

            row_ind.append(k)
            col_ind.append(idx(i, j - 1))
            data.append(1.0 / (dx * dx))

    A = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
    return A


# Build and factorize Poisson matrix once
A_poisson = build_poisson_matrix(NX, NY, DX, DY)
lu_poisson = splu(A_poisson)
print("[THERMAL] Poisson matrix built and factorized")


# =============================================================================
# Thermal solver
# =============================================================================
def solve_temperature(sigma_field, nx=NX, ny=NY):
    """
    Solve nabla^2 T = -sigma / D with Dirichlet T=T0 on boundaries.

    Parameters
    ----------
    sigma_field : ndarray(nx, ny)
        Heat source field [W/m^3].

    Returns
    -------
    T : ndarray(nx, ny)
        Temperature field [K].
    """
    # RHS: -sigma / D
    b = -sigma_field.flatten() / D_THERMAL

    # Boundary conditions: T = T0
    for i in range(nx):
        for j in range(ny):
            k = i * ny + j
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                b[k] = T0

    # Solve
    T_flat = lu_poisson.solve(b)
    T = T_flat.reshape(nx, ny)
    return T


def build_sigma_field(sigma_blocks, nx=NX, ny=NY):
    """Build heat source field from block strengths."""
    sigma = np.zeros((nx, ny))
    for sk, (x0, x1, y0, y1) in zip(sigma_blocks, BLOCK_BOUNDS):
        sigma[x0:x1, y0:y1] = sk
    return sigma


def build_sound_speed(T):
    return C0 * np.sqrt(T / T0)


# =============================================================================
# Acoustic FDTD
# =============================================================================
SOURCE_CELLS = [3, 4, 5]
SOURCE_AMP = 1.0

_window = np.zeros(NX)
for idx, i in enumerate(SOURCE_CELLS):
    _window[i] = 0.5 * (1.0 - np.cos(np.pi * idx / (len(SOURCE_CELLS) - 1)))


def source_term(t, freq):
    return SOURCE_AMP * np.sin(2.0 * np.pi * freq * t)


def run_fdtd(c, freq, n_steps):
    p = np.zeros((NX, NY))
    u = np.zeros((NX + 1, NY))
    v = np.zeros((NX, NY + 1))
    c2 = c ** 2

    probe_p = np.zeros(n_steps)
    probe_u = np.zeros(n_steps)

    for n in range(n_steps):
        t = n * DT

        src = source_term(t, freq)
        for i in SOURCE_CELLS:
            p[i, :] += src * _window[i]

        u[1:NX, :] -= (DT / RHO0) * (p[1:NX, :] - p[0:NX - 1, :]) / DX
        v[:, 1:NY] -= (DT / RHO0) * (p[:, 1:NY] - p[:, 0:NY - 1]) / DY

        du_dx = (u[1:NX + 1, :] - u[0:NX, :]) / DX
        dv_dy = (v[:, 1:NY + 1] - v[:, 0:NY]) / DY
        p -= DT * RHO0 * c2 * (du_dx + dv_dy)

        v[:, 0] = 0.0
        v[:, NY] = 0.0
        u[0, :] = -p[0, :] / (RHO0 * c[0, :])
        u[NX, :] = p[NX - 1, :] / (RHO0 * c[NX - 1, :])

        probe_p[n] = p[NX - 1, NY // 2]
        probe_u[n] = u[NX, NY // 2]

    discard = int(0.8 * n_steps)
    intensity = probe_p * probe_u
    power = np.mean(intensity[discard:]) * LY
    return power


# =============================================================================
# Training
# =============================================================================
def evaluate_powers(sigma_blocks, f_pass=2000.0, f_block=8000.0):
    sigma = build_sigma_field(sigma_blocks)
    T = solve_temperature(sigma)
    c = build_sound_speed(T)
    n_steps = max(4000, int(6.0 * LX / (343.0 * DT)))
    p_pass = run_fdtd(c, f_pass, n_steps)
    p_block = run_fdtd(c, f_block, n_steps)
    return p_pass, p_block


def loss_fn(sigma_blocks, f_pass=2000.0, f_block=8000.0):
    p_pass, p_block = evaluate_powers(sigma_blocks, f_pass, f_block)
    ratio = p_pass / (p_block + 1e-12)
    return -ratio


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    fig_dir = Path(__file__).parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    F_PASS = 2000.0
    F_BLOCK = 8000.0

    # Baseline: no heat sources
    print("\n--- Baseline (no heating) ---")
    sigma_base = np.zeros(N_BLOCKS)
    p_pass_base, p_block_base = evaluate_powers(sigma_base, F_PASS, F_BLOCK)
    print(f"Pass ({F_PASS/1e3:.1f} kHz): {p_pass_base:.6f}")
    print(f"Block ({F_BLOCK/1e3:.1f} kHz): {p_block_base:.6f}")
    print(f"Ratio: {p_pass_base/max(p_block_base,1e-10):.2f}")

    # Visualize baseline temperature field
    T_base = solve_temperature(build_sigma_field(sigma_base))

    # Random search
    print(f"\n--- Random search (10 samples) ---")
    best_ratio = p_pass_base / max(p_block_base, 1e-10)
    best_sigma = sigma_base.copy()

    for trial in range(10):
        sigma_try = np.random.uniform(SIGMA_MIN, SIGMA_MAX, N_BLOCKS)
        p_pass, p_block = evaluate_powers(sigma_try, F_PASS, F_BLOCK)
        ratio = p_pass / max(p_block, 1e-10)
        if ratio > best_ratio:
            best_ratio = ratio
            best_sigma = sigma_try.copy()
            print(f"  Trial {trial}: ratio={ratio:.2f} (pass={p_pass:.6f}, block={p_block:.6f})")

    print(f"\nBest random ratio: {best_ratio:.2f}")

    # Optimization
    print(f"\n--- L-BFGS-B (15 iter max) ---")
    iteration = [0]

    def callback(sigma_blocks):
        iteration[0] += 1
        if iteration[0] % 3 == 0:
            p_pass, p_block = evaluate_powers(sigma_blocks, F_PASS, F_BLOCK)
            ratio = p_pass / max(p_block, 1e-10)
            print(f"  Iter {iteration[0]:3d}: ratio={ratio:.2f}")

    bounds = [(SIGMA_MIN, SIGMA_MAX) for _ in range(N_BLOCKS)]
    result = minimize(
        lambda s: loss_fn(s, F_PASS, F_BLOCK),
        best_sigma,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 15, 'ftol': 1e-6},
        callback=callback
    )

    sigma_opt = result.x
    p_pass_opt, p_block_opt = evaluate_powers(sigma_opt, F_PASS, F_BLOCK)
    ratio_opt = p_pass_opt / max(p_block_opt, 1e-10)
    ratio_base = p_pass_base / max(p_block_base, 1e-10)

    print(f"\n--- Results ---")
    print(f"Baseline:  ratio={ratio_base:.2f}")
    print(f"Optimized: ratio={ratio_opt:.2f}")
    print(f"Improvement: {ratio_opt/ratio_base:.2f}x")

    # Compute optimized temperature field
    sigma_opt_field = build_sigma_field(sigma_opt)
    T_opt = solve_temperature(sigma_opt_field)

    print(f"\nOptimized heat sources [W/m^3]:")
    for k, sk in enumerate(sigma_opt):
        print(f"  Block {k}: {sk:.3e}")

    print(f"\nResulting block temperatures [K]:")
    for k, (x0, x1, y0, y1) in enumerate(BLOCK_BOUNDS):
        T_avg = np.mean(T_opt[x0:x1, y0:y1])
        print(f"  Block {k}: {T_avg:.1f} K")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    im = ax.imshow(sigma_opt_field.T, origin='lower', cmap='hot',
                   extent=[0, LX*1000, 0, LY*1000])
    ax.set_title('Heat Source sigma [W/m^3]')
    fig.colorbar(im, ax=ax)

    ax = axes[1]
    im = ax.imshow(T_opt.T, origin='lower', cmap='hot',
                   extent=[0, LX*1000, 0, LY*1000])
    ax.set_title('Temperature T [K]')
    fig.colorbar(im, ax=ax)

    ax = axes[2]
    c_opt = build_sound_speed(T_opt)
    im = ax.imshow(c_opt.T, origin='lower', cmap='viridis',
                   extent=[0, LX*1000, 0, LY*1000])
    ax.set_title('Sound Speed c [m/s]')
    fig.colorbar(im, ax=ax)

    plt.suptitle('Thermal + Acoustic Filter', fontsize=14)
    plt.tight_layout()
    plt.savefig(fig_dir / "trained_filter_thermal.png", dpi=150)
    print(f"\nSaved: {fig_dir / 'trained_filter_thermal.png'}")
    plt.close()
