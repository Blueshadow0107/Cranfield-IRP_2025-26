"""
Barkley model of an excitable medium.

This is a reduced two-variable model that captures the phenomenology of
the Belousov-Zhabotinsky (BZ) reaction: resting state, threshold,
excitation, and recovery.  It is much simpler than the full Oregonator
kinetics but reproduces the spiral waves that are the signature of
BZ-like excitable media.

Currently this script produces a single self-sustaining spiral wave.
Counter-propagating wave collision (the basis of BZ logic gates) is the
next step; the current parameter/initial-condition combinations tested
so far have not yielded usable planar collision figures.

References:
- Barkley, Physica D 1991.
- ibiblio.org/e-notes/webgl/gpu/waves/barkley.html
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
NX, NY = 256, 256
DX = 1.0
DT = 0.05
NSTEPS = 20000
PLOT_EVERY = 5000

Du = 1.0
Dv = 0.0      # inhibitor typically does not diffuse in Barkley
A = 0.75
B = 0.01
EPS = 0.02


def laplacian(c):
    """5-point Laplacian with Neumann boundaries."""
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

    du = Du * Lu + reaction
    dv = Dv * Lv + u - v

    u += DT * du
    v += DT * dv

    # Clip to keep numerics sane
    np.clip(u, -0.1, 1.1, out=u)
    np.clip(v, -0.1, 1.1, out=v)

    return u, v


def run_spiral():
    """Initialise a broken wave to nucleate a spiral wave."""
    u = np.zeros((NX, NY), dtype=float)
    v = np.zeros((NX, NY), dtype=float)

    # A broken planar wave: excited on bottom-left, quenched on top-right
    Y, X = np.mgrid[0:NY, 0:NX]
    mask_wave = (X + Y) < (NX + NY) // 2
    u[mask_wave] = 1.0

    # Quench a quadrant to break the wave tip
    mask_quench = (X > NX * 0.45) & (Y > NY * 0.55)
    u[mask_quench] = 0.0
    v[mask_quench] = 0.5

    snapshots = []
    times = []
    for n in range(NSTEPS):
        u, v = step(u, v)
        if (n + 1) % PLOT_EVERY == 0:
            snapshots.append(u.copy())
            times.append((n + 1) * DT)

    return u, v, snapshots, times


def plot_spiral(snapshots, times):
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    axes = axes.ravel()
    for ax, snap, t in zip(axes, snapshots, times):
        ax.imshow(snap.T, origin='lower', cmap='hot', vmin=0, vmax=1)
        ax.set_title(f't = {t:.1f}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle('Barkley model: spiral wave (BZ phenomenology)', fontsize=13)
    plt.tight_layout()
    plt.savefig('Analysis/barkley_spiral.png', dpi=150)
    print('Saved Analysis/barkley_spiral.png')


if __name__ == '__main__':
    print('Running spiral wave demo...')
    u, v, snaps, times = run_spiral()
    plot_spiral(snaps, times)
    print('Done.')
