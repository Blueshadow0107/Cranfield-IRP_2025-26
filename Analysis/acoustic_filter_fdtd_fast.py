#!/usr/bin/env python3
"""Fast prototype version with coarser grid for quick training iteration."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize

# =============================================================================
# Physical and numerical parameters (FAST: coarser grid)
# =============================================================================
NX, NY = 250, 50            # coarser grid
LX, LY = 0.5, 0.1           # domain size [m]
DX = LX / NX                # 2.0e-3 m
DY = LY / NY                # 2.0e-3 m

RHO0 = 1.2
T0 = 300.0
C0 = 343.0
T_MAX = 500.0
C_MAX = C0 * np.sqrt(T_MAX / T0)

CFL_FACTOR = 0.5
DT = CFL_FACTOR * DX / (C_MAX * np.sqrt(2))
print(f"[FAST] dx = {DX:.3e} m,  dt = {DT:.3e} s,  c_max = {C_MAX:.1f} m/s")

# =============================================================================
# Heater blocks (same physical layout, adjusted for coarser grid)
# =============================================================================
N_BLOCKS = 8
N_ROWS = 2
N_COLS = 4

BLOCK_W = NX // 12          # ~20 cells
BLOCK_H = NY // 3           # ~16 cells
GAP_X = NX // 10            # ~25 cells
GAP_Y = NY // 5             # ~10 cells
START_X = NX // 5           # ~50 cells
START_Y = NY // 5           # ~10 cells

BLOCK_BOUNDS = []
for row in range(N_ROWS):
    for col in range(N_COLS):
        x0 = START_X + col * (BLOCK_W + GAP_X)
        x1 = x0 + BLOCK_W
        y0 = START_Y + row * (BLOCK_H + GAP_Y)
        y1 = y0 + BLOCK_H
        BLOCK_BOUNDS.append((x0, x1, y0, y1))

print(f"[FAST] Block layout: {N_ROWS}x{N_COLS} = {N_BLOCKS} blocks")
for k, (x0, x1, y0, y1) in enumerate(BLOCK_BOUNDS):
    print(f"  Block {k}: x=[{x0},{x1}], y=[{y0},{y1}]")


def build_temperature_field(T_blocks, nx=NX, ny=NY):
    T = np.full((nx, ny), T0)
    for Tk, (x0, x1, y0, y1) in zip(T_blocks, BLOCK_BOUNDS):
        T[x0:x1, y0:y1] = Tk
    return T


def build_sound_speed(T):
    return C0 * np.sqrt(T / T0)


# =============================================================================
# Source
# =============================================================================
SOURCE_CELLS = [3, 4, 5]
SOURCE_AMP = 1.0

_window = np.zeros(NX)
for idx, i in enumerate(SOURCE_CELLS):
    _window[i] = 0.5 * (1.0 - np.cos(np.pi * idx / (len(SOURCE_CELLS) - 1)))


def source_term(t, freq):
    return SOURCE_AMP * np.sin(2.0 * np.pi * freq * t)


# =============================================================================
# FDTD
# =============================================================================
def run_fdtd(c, freq, n_steps):
    p = np.zeros((NX, NY))
    u = np.zeros((NX + 1, NY))
    v = np.zeros((NX, NY + 1))
    c2 = c ** 2

    probe_p = np.zeros(n_steps)
    probe_u = np.zeros(n_steps)

    for n in range(n_steps):
        t = n * DT

        # Source
        src = source_term(t, freq)
        for i in SOURCE_CELLS:
            p[i, :] += src * _window[i]

        # Update u, v
        u[1:NX, :] -= (DT / RHO0) * (p[1:NX, :] - p[0:NX - 1, :]) / DX
        v[:, 1:NY] -= (DT / RHO0) * (p[:, 1:NY] - p[:, 0:NY - 1]) / DY

        # Update p
        du_dx = (u[1:NX + 1, :] - u[0:NX, :]) / DX
        dv_dy = (v[:, 1:NY + 1] - v[:, 0:NY]) / DY
        p -= DT * RHO0 * c2 * (du_dx + dv_dy)

        # BCs
        v[:, 0] = 0.0
        v[:, NY] = 0.0
        u[0, :] = -p[0, :] / (RHO0 * c[0, :])
        u[NX, :] = p[NX - 1, :] / (RHO0 * c[NX - 1, :])

        probe_p[n] = p[NX - 1, NY // 2]
        probe_u[n] = u[NX, NY // 2]

    # Time-averaged power
    discard = int(0.8 * n_steps)
    intensity = probe_p * probe_u
    power = np.mean(intensity[discard:]) * LY
    return power


# =============================================================================
# Training
# =============================================================================
def evaluate_powers(T_blocks, f_pass=2000.0, f_block=8000.0):
    T = build_temperature_field(T_blocks)
    c = build_sound_speed(T)
    n_steps = max(4000, int(6.0 * LX / (343.0 * DT)))
    p_pass = run_fdtd(c, f_pass, n_steps)
    p_block = run_fdtd(c, f_block, n_steps)
    return p_pass, p_block


def loss_fn(T_blocks, f_pass=2000.0, f_block=8000.0):
    p_pass, p_block = evaluate_powers(T_blocks, f_pass, f_block)
    ratio = p_pass / (p_block + 1e-12)
    return -ratio


if __name__ == "__main__":
    fig_dir = Path(__file__).parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    F_PASS = 2000.0
    F_BLOCK = 8000.0

    print(f"\n--- Baseline ---")
    T_base = np.full(N_BLOCKS, T0)
    p_pass_base, p_block_base = evaluate_powers(T_base, F_PASS, F_BLOCK)
    print(f"Pass ({F_PASS/1e3:.1f} kHz): {p_pass_base:.6f}")
    print(f"Block ({F_BLOCK/1e3:.1f} kHz): {p_block_base:.6f}")
    print(f"Ratio: {p_pass_base/max(p_block_base,1e-10):.2f}")

    print(f"\n--- Random search (10 samples) ---")
    best_ratio = p_pass_base / max(p_block_base, 1e-10)
    best_T = T_base.copy()

    for trial in range(5):
        T_try = np.random.uniform(T0, T_MAX, N_BLOCKS)
        p_pass, p_block = evaluate_powers(T_try, F_PASS, F_BLOCK)
        ratio = p_pass / max(p_block, 1e-10)
        if ratio > best_ratio:
            best_ratio = ratio
            best_T = T_try.copy()
            print(f"  Trial {trial}: ratio={ratio:.2f} (pass={p_pass:.6f}, block={p_block:.6f})")

    print(f"\nBest random ratio: {best_ratio:.2f}")

    print(f"\n--- L-BFGS-B (30 iter max) ---")
    iteration = [0]

    def callback(T_blocks):
        iteration[0] += 1
        if iteration[0] % 3 == 0:
            p_pass, p_block = evaluate_powers(T_blocks, F_PASS, F_BLOCK)
            ratio = p_pass / max(p_block, 1e-10)
            print(f"  Iter {iteration[0]:3d}: ratio={ratio:.2f}")

    bounds = [(T0, T_MAX) for _ in range(N_BLOCKS)]
    result = minimize(
        lambda T: loss_fn(T, F_PASS, F_BLOCK),
        best_T,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 15, 'ftol': 1e-6},
        callback=callback
    )

    T_opt = result.x
    p_pass_opt, p_block_opt = evaluate_powers(T_opt, F_PASS, F_BLOCK)
    ratio_opt = p_pass_opt / max(p_block_opt, 1e-10)
    ratio_base = p_pass_base / max(p_block_base, 1e-10)

    print(f"\n--- Results ---")
    print(f"Baseline:  ratio={ratio_base:.2f}")
    print(f"Optimized: ratio={ratio_opt:.2f}")
    print(f"Improvement: {ratio_opt/ratio_base:.2f}x")
    print(f"\nOptimized temperatures:")
    for k, Tk in enumerate(T_opt):
        print(f"  Block {k}: {Tk:.1f} K")

    # Plot
    T_opt_field = build_temperature_field(T_opt)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    im = ax.imshow(T_opt_field.T, origin='lower', cmap='hot',
                   extent=[0, LX*1000, 0, LY*1000])
    ax.set_title('Optimized Temperature [K]')
    fig.colorbar(im, ax=ax)

    ax = axes[1]
    ax.bar(['2 kHz\npass', '8 kHz\nblock'],
           [p_pass_opt, p_block_opt],
           color=['green', 'red'])
    ax.set_ylabel('Transmitted power')
    ax.set_title(f'Filter performance (ratio = {ratio_opt:.1f})')

    plt.tight_layout()
    plt.savefig(fig_dir / "trained_filter_fast.png", dpi=150)
    print(f"\nSaved: {fig_dir / 'trained_filter_fast.png'}")
    plt.close()
