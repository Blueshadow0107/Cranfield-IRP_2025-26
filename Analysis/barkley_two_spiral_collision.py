"""
Two-spiral collision test in the Barkley model.

Places two standard broken-wave spiral seeds in opposite corners of a
large domain and lets their outgoing spiral arms collide.  If the seeds
nucleate properly, the collision region in the centre should show
annihilation of counter-propagating wave fronts.
"""

import numpy as np
import matplotlib.pyplot as plt


NX, NY = 512, 512
DX = 1.0
DT = 0.05
NSTEPS = 10000
PLOT_EVERY = 2500

Du = 1.0
Dv = 0.0
A = 0.75
B = 0.01
EPS = 0.02


def laplacian(c):
    out = np.zeros_like(c)
    out[1:-1, 1:-1] = (
        c[:-2, 1:-1] + c[2:, 1:-1] +
        c[1:-1, :-2] + c[1:-1, 2:] -
        4 * c[1:-1, 1:-1]
    ) / DX**2
    out[0, :] = out[1, :]
    out[-1, :] = out[-2, :]
    out[:, 0] = out[:, 1]
    out[:, -1] = out[:, -2]
    return out


def step(u, v):
    Lu = laplacian(u)
    Lv = laplacian(v)
    u_threshold = (v + B) / A
    reaction = (1.0 / EPS) * u * (1.0 - u) * (u - u_threshold)
    u += DT * (Du * Lu + reaction)
    v += DT * (Dv * Lv + u - v)
    np.clip(u, -0.1, 1.1, out=u)
    np.clip(v, -0.1, 1.1, out=v)
    return u, v


def add_spiral_seed(u, v, cx, cy, size):
    """Add a standard Barkley broken-wave seed centred at (cx, cy)."""
    Y, X = np.mgrid[0:NY, 0:NX]
    local_x = X - cx
    local_y = Y - cy

    # Only affect a local box
    in_box = (np.abs(local_x) < size) & (np.abs(local_y) < size)

    # Diagonal half-plane within the box
    mask_wave = in_box & ((local_x + local_y) < 0)
    u[mask_wave] = 1.0

    # Quench a corner to break the wave tip
    mask_quench = (
        in_box &
        (local_x > size * 0.4) &
        (local_y > size * 0.4)
    )
    u[mask_quench] = 0.0
    v[mask_quench] = 0.5


u = np.zeros((NX, NY), dtype=float)
v = np.zeros((NX, NY), dtype=float)

# Two seeds in opposite corners
add_spiral_seed(u, v, 120, 120, 80)
add_spiral_seed(u, v, NX - 120, NY - 120, 80)

snapshots = []
times = []
for n in range(NSTEPS):
    u, v = step(u, v)
    if (n + 1) % PLOT_EVERY == 0:
        snapshots.append(u.copy())
        times.append((n + 1) * DT)
        print(f"t = {(n+1)*DT:.1f}, max u = {u.max():.3f}, mean u = {u.mean():.3f}")

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.ravel()
for ax, snap, t in zip(axes, snapshots, times):
    ax.imshow(snap.T, origin='lower', cmap='hot', vmin=0, vmax=1)
    ax.set_title(f't = {t:.1f}')
    ax.set_xticks([])
    ax.set_yticks([])
plt.suptitle('Barkley model: two spiral seeds and colliding arms', fontsize=13)
plt.tight_layout()
plt.savefig('Analysis/barkley_two_spiral_collision.png', dpi=120)
print('Saved Analysis/barkley_two_spiral_collision.png')
