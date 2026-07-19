"""
Classic 2D Gray-Scott pattern formation demo.

Shows how a small seed of autocatalyst V evolves into self-organised
spots/stripes.  This is the visual proof that reaction-diffusion can
act as a continuous, spatial computational medium.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
NX, NY = 256, 256
DX = 1.0
DT = 1.0
NSTEPS = 10000
PLOT_EVERY = 2500

Du = 0.16
Dv = 0.08
F = 0.035
K = 0.065


def laplacian(c):
    out = np.zeros_like(c)
    out[1:-1, 1:-1] = (
        c[:-2, 1:-1] + c[2:, 1:-1] +
        c[1:-1, :-2] + c[1:-1, 2:] -
        4 * c[1:-1, 1:-1]
    )
    # Neumann boundaries
    out[0, :] = out[1, :]
    out[-1, :] = out[-2, :]
    out[:, 0] = out[:, 1]
    out[:, -1] = out[:, -2]
    return out


def step(U, V):
    Lu = laplacian(U)
    Lv = laplacian(V)
    uvv = U * V * V
    U += DT * (Du * Lu - uvv + F * (1.0 - U))
    V += DT * (Dv * Lv + uvv - (F + K) * V)
    return U, V


def run():
    U = np.ones((NX, NY), dtype=float)
    V = np.zeros((NX, NY), dtype=float)

    # Seed a square region in the centre with V
    r = 20
    U[NX//2-r:NX//2+r, NY//2-r:NY//2+r] = 0.5
    V[NX//2-r:NX//2+r, NY//2-r:NY//2+r] = 0.25

    snapshots = []
    snapshot_times = []

    for n in range(NSTEPS):
        U, V = step(U, V)
        if (n + 1) % PLOT_EVERY == 0:
            snapshots.append(V.copy())
            snapshot_times.append((n + 1) * DT)

    return U, V, snapshots, snapshot_times


if __name__ == '__main__':
    U, V, snaps, times = run()

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    axes = axes.ravel()
    for ax, snap, t in zip(axes, snaps, times):
        ax.imshow(snap.T, origin='lower', cmap='inferno', vmin=0, vmax=0.5)
        ax.set_title(f't = {t:.0f}')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f'Gray-Scott pattern formation (F={F}, k={K})', fontsize=14)
    plt.tight_layout()
    plt.savefig('Analysis/gray_scott_pattern_formation.png', dpi=150)
    print('Saved Analysis/gray_scott_pattern_formation.png')

    # Also save final field alone for slides
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(V.T, origin='lower', cmap='inferno', vmin=0, vmax=0.5)
    ax.set_title(f'Gray-Scott spots at t={times[-1]:.0f}')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig('Analysis/gray_scott_final_field.png', dpi=150)
    print('Saved Analysis/gray_scott_final_field.png')
