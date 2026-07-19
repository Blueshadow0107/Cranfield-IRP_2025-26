"""
Oregonator BZ: timing as a control degree of freedom.

Two excited seeds are placed on the left and right.  The right seed is
fired at different delays relative to the left seed.  The final field
shows how timing changes the collision / routing outcome.
"""

import numpy as np
import matplotlib.pyplot as plt


NX, NY = 256, 256
DX = 1.0
DT = 0.05
Du = 1.0
Dv = 0.6
EPS = 0.05
Q = 0.002
F = 1.4
PHI = 0.0


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


def run(delay_steps, nsteps=12000):
    """Run with left pulse at t=0 and right pulse after delay_steps."""
    u = np.zeros((NX, NY), dtype=float)
    v = np.zeros((NX, NY), dtype=float)

    Y, X = np.mgrid[0:NY, 0:NX]
    left = np.sqrt((X - 60)**2 + (Y - NY // 2)**2) < 25
    right = np.sqrt((X - (NX - 60))**2 + (Y - NY // 2)**2) < 25

    for n in range(nsteps):
        if 0 <= n < 30:
            u[left] = 0.8
            v[left] = 0.2
        if delay_steps <= n < delay_steps + 30:
            u[right] = 0.8
            v[right] = 0.2
        u, v = step(u, v)

    return u, v


delays = [0, 300, 600, 900]  # in time steps
labels = [
    'simultaneous',
    'B delayed by 15 time units',
    'B delayed by 30 time units',
    'B delayed by 45 time units',
]

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.ravel()

for ax, delay, label in zip(axes, delays, labels):
    u, v = run(delay)
    ax.imshow(u.T, origin='lower', cmap='hot', vmin=0, vmax=1)
    ax.set_title(f'{label}\nmax u={u.max():.2f}, mean u={u.mean():.3f}')
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle('Oregonator BZ: effect of input timing on wave interaction', fontsize=13)
plt.tight_layout()
plt.savefig('Analysis/oregonator_timing_sweep.png', dpi=150)
print('Saved Analysis/oregonator_timing_sweep.png')
