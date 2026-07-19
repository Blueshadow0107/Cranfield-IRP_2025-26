"""
Light-sensitive Oregonator model for BZ excitable media.

Equations (after Tyson & Fife scaling):
    du/dt = (1/eps) * [u - u**2 - (f*v + phi) * (u - q)/(u + q)] + Du * L u
    dv/dt = u - v + Dv * L v

u  ~ HBrO2  (activator)
v  ~ Ce(IV) / catalyst oxidation level
phi  ~ light intensity (suppresses excitability)
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
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
PHI = 0.0  # no light suppression


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

    reaction = (1.0 / EPS) * (
        u - u**2 - (F * v + PHI) * (u - Q) / (u + Q)
    )

    u += DT * (Du * Lu + reaction)
    v += DT * (Dv * Lv + u - v)

    np.clip(u, -0.1, 1.0, out=u)
    np.clip(v, -0.1, 1.5, out=v)
    return u, v


def run_target():
    u = np.zeros((NX, NY), dtype=float)
    v = np.zeros((NX, NY), dtype=float)

    # Central excited disk
    Y, X = np.mgrid[0:NY, 0:NX]
    r = np.sqrt((X - NX // 2)**2 + (Y - NY // 2)**2)
    mask = r < 25
    u[mask] = 0.8
    v[mask] = 0.2

    snapshots = []
    times = []
    for n in range(NSTEPS):
        u, v = step(u, v)
        if (n + 1) % PLOT_EVERY == 0:
            snapshots.append(u.copy())
            times.append((n + 1) * DT)
            print(f"target t={(n+1)*DT:.1f}, max u={u.max():.3f}, mean u={u.mean():.3f}")

    return snapshots, times


def run_collision():
    u = np.zeros((NX, NY), dtype=float)
    v = np.zeros((NX, NY), dtype=float)

    Y, X = np.mgrid[0:NY, 0:NX]
    r1 = np.sqrt((X - 70)**2 + (Y - NY // 2)**2)
    r2 = np.sqrt((X - (NX - 70))**2 + (Y - NY // 2)**2)
    u[r1 < 25] = 0.8
    u[r2 < 25] = 0.8
    v[(r1 < 25) | (r2 < 25)] = 0.2

    snapshots = []
    times = []
    for n in range(NSTEPS):
        u, v = step(u, v)
        if (n + 1) % PLOT_EVERY == 0:
            snapshots.append(u.copy())
            times.append((n + 1) * DT)
            print(f"collision t={(n+1)*DT:.1f}, max u={u.max():.3f}, mean u={u.mean():.3f}")

    return snapshots, times


def plot_series(snapshots, times, title, filename):
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    axes = axes.ravel()
    for ax, snap, t in zip(axes, snapshots, times):
        ax.imshow(snap.T, origin='lower', cmap='hot', vmin=0, vmax=1)
        ax.set_title(f't = {t:.1f}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f'Saved {filename}')


if __name__ == '__main__':
    print('Running Oregonator target wave demo...')
    snaps, times = run_target()
    plot_series(snaps, times,
                'Oregonator BZ: target waves from central seed',
                'Analysis/oregonator_target.png')

    print('\nRunning Oregonator collision demo...')
    snaps, times = run_collision()
    plot_series(snaps, times,
                'Oregonator BZ: counter-propagating target waves',
                'Analysis/oregonator_collision.png')
