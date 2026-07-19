"""
Quick parameter sweep for Barkley target waves.

Tests a small grid of (a, b, epsilon, radius, dt) values to find a
combination that nucleates circular target waves from a central excited
disk.  Saves a summary grid of final snapshots.
"""

import numpy as np
import matplotlib.pyplot as plt


def step(u, v, dt, dx, du, dv, a, b, eps):
    """One explicit Euler step for the Barkley model."""
    out_u = np.zeros_like(u)
    out_v = np.zeros_like(v)

    out_u[1:-1, 1:-1] = (
        u[:-2, 1:-1] + u[2:, 1:-1] +
        u[1:-1, :-2] + u[1:-1, 2:] -
        4 * u[1:-1, 1:-1]
    ) / dx**2
    out_v[1:-1, 1:-1] = (
        v[:-2, 1:-1] + v[2:, 1:-1] +
        v[1:-1, :-2] + v[1:-1, 2:] -
        4 * v[1:-1, 1:-1]
    ) / dx**2

    # Neumann boundaries
    out_u[0, :] = out_u[1, :]
    out_u[-1, :] = out_u[-2, :]
    out_u[:, 0] = out_u[:, 1]
    out_u[:, -1] = out_u[:, -2]
    out_v[0, :] = out_v[1, :]
    out_v[-1, :] = out_v[-2, :]
    out_v[:, 0] = out_v[:, 1]
    out_v[:, -1] = out_v[:, -2]

    uth = (v + b) / a
    u += dt * (du * out_u + (1.0 / eps) * u * (1.0 - u) * (u - uth))
    v += dt * (dv * out_v + u - v)

    np.clip(u, -0.1, 1.1, out=u)
    np.clip(v, -0.1, 1.1, out=v)
    return u, v


def run_target(a, b, eps, radius, dt=0.05, nx=256, ny=256, nsteps=8000):
    u = np.zeros((nx, ny), dtype=float)
    v = np.zeros((nx, ny), dtype=float)

    Y, X = np.mgrid[0:ny, 0:nx]
    cx, cy = nx // 2, ny // 2
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    u[r < radius] = 1.0

    for _ in range(nsteps):
        u, v = step(u, v, dt, 1.0, 1.0, 0.0, a, b, eps)

    return u, v


# Parameter grid to test
configs = [
    # (a, b, eps, radius, dt)
    (0.75, 0.01, 0.02, 30, 0.05),
    (0.75, 0.01, 0.02, 50, 0.05),
    (0.75, 0.01, 0.02, 80, 0.05),
    (0.75, 0.02, 0.05, 40, 0.05),
    (0.75, 0.06, 0.08, 40, 0.05),
    (0.75, 0.03, 0.05, 40, 0.05),
    (0.80, 0.02, 0.05, 40, 0.05),
    (0.70, 0.01, 0.02, 50, 0.05),
    (0.75, 0.01, 0.01, 40, 0.02),
    (0.75, 0.04, 0.08, 40, 0.05),
    (0.75, 0.02, 0.08, 60, 0.05),
    (0.60, 0.01, 0.02, 50, 0.05),
]

n = len(configs)
cols = 4
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
axes = axes.ravel()

results = []
for ax, (a, b, eps, radius, dt) in zip(axes, configs):
    u, v = run_target(a, b, eps, radius, dt=dt)
    ax.imshow(u.T, origin='lower', cmap='hot', vmin=0, vmax=1)
    ax.set_title(f'a={a} b={b} eps={eps} r={radius} dt={dt}\nmax u={u.max():.2f}', fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    results.append((a, b, eps, radius, dt, u.max(), u.mean()))

for ax in axes[n:]:
    ax.axis('off')

plt.suptitle('Barkley target-wave parameter sweep', fontsize=12)
plt.tight_layout()
plt.savefig('Analysis/barkley_target_sweep.png', dpi=150)
print('Saved Analysis/barkley_target_sweep.png')

print('\nSummary:')
print('a      b      eps    radius dt   max_u  mean_u')
for a, b, eps, radius, dt, mx, mn in results:
    print(f'{a:.2f}   {b:.2f}   {eps:.2f}   {radius:3d}    {dt:.2f}   {mx:.3f}  {mn:.3f}')
