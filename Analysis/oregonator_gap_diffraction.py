"""
Oregonator BZ wave passing through a gap in a no-flux wall.

A target-wave source on the left sends circular waves toward a wall
with a central gap.  The wave diffracts through the gap and spreads
out on the right side --- the prototype for channel routing.
"""

import numpy as np
import matplotlib.pyplot as plt


NX, NY = 256, 256
DX = 1.0
DT = 0.05
NSTEPS = 12000
PLOT_EVERY = 3000

Du = 1.0
Dv = 0.6
EPS = 0.05
Q = 0.002
F = 1.4
PHI = 0.0


def laplacian(c, mask):
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
    # No flux inside obstacle (mask = True means obstacle)
    out[mask] = 0.0
    return out


def step(u, v, mask):
    Lu = laplacian(u, mask)
    Lv = laplacian(v, mask)

    reaction = (1.0 / EPS) * (
        u - u**2 - (F * v + PHI) * (u - Q) / (u + Q)
    )

    u += DT * (Du * Lu + reaction)
    v += DT * (Dv * Lv + u - v)

    # Obstacle is inert
    u[mask] = 0.0
    v[mask] = 0.0

    np.clip(u, -0.1, 1.0, out=u)
    np.clip(v, -0.1, 1.5, out=v)
    return u, v


def make_wall_with_gap(nx, ny, wall_x, gap_half_width):
    """Return a boolean mask for a vertical wall with a central gap."""
    Y, X = np.mgrid[0:ny, 0:nx]
    mask = (np.abs(X - wall_x) < 3) & (np.abs(Y - ny // 2) >= gap_half_width)
    return mask


def run_gap_diffraction():
    u = np.zeros((NX, NY), dtype=float)
    v = np.zeros((NX, NY), dtype=float)

    wall_mask = make_wall_with_gap(NX, NY, NX // 2, 25)

    # Source on the left
    Y, X = np.mgrid[0:NY, 0:NX]
    r = np.sqrt((X - 50)**2 + (Y - NY // 2)**2)
    u[r < 25] = 0.8
    v[r < 25] = 0.2

    snapshots = []
    times = []
    for n in range(NSTEPS):
        u, v = step(u, v, wall_mask)
        if (n + 1) % PLOT_EVERY == 0:
            snapshots.append(u.copy())
            times.append((n + 1) * DT)
            print(f"t={(n+1)*DT:.1f}, max u={u.max():.3f}, mean u={u.mean():.3f}")

    return snapshots, times, wall_mask


def plot_series(snapshots, times, wall_mask, filename):
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    axes = axes.ravel()
    for ax, snap, t in zip(axes, snapshots, times):
        # Overlay wall in grey
        img = snap.T.copy()
        wall_img = wall_mask.T.copy()
        rgba = np.zeros((img.shape[0], img.shape[1], 4))
        cmap = plt.cm.hot
        norm = plt.Normalize(0, 1)
        rgba[:, :, :3] = cmap(norm(img))[:, :, :3]
        rgba[:, :, 3] = 1.0
        # grey obstacle
        rgba[wall_img, :3] = 0.4
        ax.imshow(rgba, origin='lower')
        ax.set_title(f't = {t:.1f}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle('Oregonator BZ: wave diffraction through a gap', fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f'Saved {filename}')


if __name__ == '__main__':
    snaps, times, wall = run_gap_diffraction()
    plot_series(snaps, times, wall, 'Analysis/oregonator_gap_diffraction.png')
