"""
3D channel pulse transfer test (dark-spot Oregonator).

Geometry: 64x64x64 cubic domain with a straight 16x16 channel along x.
Input: one-shot dark-spot flash at the channel entrance (phi drops to 0.002
for 3.0 t.u.).
Output: JSON + PNG saved to Analysis/figures/rd_3d_transfer_channel.*.
"""

import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from rd_core_3d import RDSubstrate3D


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
F = 1.4375
EPS = 0.0501
Q = 0.002
DU = 1.0
DV = 0.0
PHI0 = 0.010
PHI_DARK = 0.002
T_FLASH = 3.0
DT = 0.05
DX = 1.0
N = 64
W = 16
W2 = W // 2
NSTEPS = 500
DURATION = int(T_FLASH / DT)
SPOT_R = 4

U_THRESH = 0.5
X_NEAR, X_MID, X_FAR = 16, 32, 48


def rest_u_star(f, phi):
    def F_u(u):
        return u - u**2 - (f * u + phi) * (u - Q) / (u + Q)
    ugrid = np.concatenate(([0.0], np.logspace(-7, np.log10(1.2), 4000)))
    vals = F_u(ugrid)
    for i in range(len(ugrid) - 1):
        if vals[i] > 0.0 and vals[i + 1] <= 0.0:
            return brentq(F_u, ugrid[i], ugrid[i + 1], xtol=1e-14, rtol=1e-12)
    raise RuntimeError('no rest state found')


def run_darkspot_3d(rd, spot_mask, nsteps, probes):
    """Run with a single dark-spot flash of length DURATION at phi=PHI_DARK."""
    series = {name: [] for name in probes}
    ts = []
    for step in range(nsteps):
        phi = np.full((rd.nx, rd.ny, rd.nz), PHI0, dtype=float)
        if step < DURATION:
            phi[spot_mask] = PHI_DARK
        rd.set_phi(phi)
        rd._step()
        rd.t += 1
        ts.append(rd.t * rd.dt)
        for name, mask in probes.items():
            series[name].append(float(rd.u[mask].mean()))
    out = {'t': np.array(ts, dtype=float)}
    for name in probes:
        out[name] = np.array(series[name], dtype=float)
    return out


def first_crossing(t, s, thresh=U_THRESH):
    idx = np.nonzero(s >= thresh)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    if i == 0:
        return float(t[0])
    s0, s1 = s[i - 1], s[i]
    frac = (thresh - s0) / (s1 - s0) if s1 > s0 else 0.0
    return float(t[i - 1] + frac * (t[i] - t[i - 1]))


def derive_speed(data):
    arrivals = {n: first_crossing(data['t'], data[n]) for n in ('near', 'mid', 'far')}
    pts = [(x, arrivals[n]) for n, x in
           (('near', X_NEAR), ('mid', X_MID), ('far', X_FAR))
           if arrivals[n] is not None]
    res = {'arrivals_tu': arrivals, 'speed_cells_per_tu': None}
    if len(pts) >= 2:
        xs = np.array([p[0] for p in pts], dtype=float)
        ts = np.array([p[1] for p in pts])
        inv_c, t0 = np.polyfit(xs, ts, 1)
        res['speed_cells_per_tu'] = float(1.0 / inv_c)
        res['linearity_max_resid_tu'] = float(
            np.abs(np.polyval([inv_c, t0], xs) - ts).max())
    return res


def build_channel():
    rd = RDSubstrate3D(nx=N, ny=N, nz=N, dx=DX, dt=DT,
                       f=F, eps=EPS, q=Q, Du=DU, Dv=DV, phi=PHI0)
    u_star = rest_u_star(F, PHI0)
    rd.set_rest(u_star, u_star)

    wall = np.ones((N, N, N), dtype=bool)
    cy = cz = N // 2
    wall[:, cy - W2:cy + W2, cz - W2:cz + W2] = False
    rd.set_walls(wall)

    rd.u[~wall] = u_star
    rd.v[~wall] = u_star
    return rd, wall, u_star, cy, cz


def strip_mask(x, wall):
    m = np.zeros((N, N, N), dtype=bool)
    cy = cz = N // 2
    m[x, cy - W2:cy + W2, cz - W2:cz + W2] = True
    return m & ~wall


def main():
    print(f'3D channel transfer: grid {N}x{N}x{N}, channel {W}x{W}')
    print(f'Parameters: f={F}, eps={EPS}, q={Q}, Du={DU}, Dv={DV}, phi0={PHI0}')
    print(f'Dark spot: phi={PHI_DARK}, T_flash={T_FLASH} t.u., radius={SPOT_R}')
    print(f'Diffusion number: {DU * DT / DX**2:.4f} (3D limit 1/6={1/6:.4f})')

    rd, wall, u_star, cy, cz = build_channel()

    # dark-spot at the entrance, centred inside the channel
    x0 = 8
    ii, jj, kk = np.mgrid[0:N, 0:N, 0:N]
    spot = ((ii - x0)**2 + (jj - cy)**2 + (kk - cz)**2) <= SPOT_R**2
    spot = spot & ~wall

    probes = {
        'near': strip_mask(X_NEAR, wall),
        'mid': strip_mask(X_MID, wall),
        'far': strip_mask(X_FAR, wall),
    }

    print(f'Running {NSTEPS} steps...')
    t_start = time.perf_counter()
    data = run_darkspot_3d(rd, spot, NSTEPS, probes)
    elapsed = time.perf_counter() - t_start
    print(f'Runtime: {elapsed:.1f} s')

    speed_info = derive_speed(data)
    peak = {n: float(data[n].max()) for n in probes}
    final = {n: float(data[n][-1]) for n in probes}

    print('\nProbe results:')
    for n in probes:
        arr = data[n]
        print(f'  {n}: peak={arr.max():.3f}, final={arr[-1]:.3f}, '
              f'arrival={speed_info["arrivals_tu"][n]}')
    print(f"Speed: {speed_info['speed_cells_per_tu']:.3f} cells/t.u.")

    # snapshot of mid-plane y=cy
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, step in zip(axes, [100, 250, 400]):
        if step <= NSTEPS:
            # replay not stored; show final only for steps <= current
            ax.imshow(rd.u[:, cy, :].T, origin='lower', cmap='inferno',
                      vmin=u_star, vmax=0.8)
            ax.set_title(f'final u, y={cy}')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
    fig.suptitle(f'3D channel pulse transfer (N={N}, W={W})')
    fig.tight_layout()

    figdir = Path('figures')
    figdir.mkdir(exist_ok=True)
    figpath = figdir / 'rd_3d_transfer_channel.png'
    fig.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f'Saved {figpath}')

    # time-series figure
    fig, ax = plt.subplots(figsize=(8, 4))
    for n in probes:
        ax.plot(data['t'], data[n], label=n)
    ax.axhline(U_THRESH, color='k', ls='--', lw=0.8)
    ax.set_xlabel('t (t.u.)')
    ax.set_ylabel('mean u')
    ax.set_title(f'3D channel probes (speed={speed_info["speed_cells_per_tu"]:.2f} cells/t.u.)')
    ax.legend()
    fig.tight_layout()
    ts_path = figdir / 'rd_3d_transfer_channel_series.png'
    fig.savefig(ts_path, dpi=150)
    plt.close(fig)
    print(f'Saved {ts_path}')

    results = {
        'test': '3D channel pulse transfer (dark-spot Oregonator)',
        'grid': [N, N, N],
        'channel_width': W,
        'dx': DX,
        'dt': DT,
        'nsteps': NSTEPS,
        'physical_time_tu': NSTEPS * DT,
        'runtime_s': elapsed,
        'parameters': {'f': F, 'eps': EPS, 'q': Q, 'Du': DU, 'Dv': DV,
                       'phi0': PHI0, 'phi_dark': PHI_DARK,
                       't_flash_tu': T_FLASH, 'spot_radius': SPOT_R},
        'rest_state_u_star': u_star,
        'diffusion_number': DU * DT / DX**2,
        'probes': {
            n: {'peak': peak[n], 'final': final[n],
                'arrival_tu': speed_info['arrivals_tu'][n]}
            for n in probes
        },
        'speed_cells_per_tu': speed_info['speed_cells_per_tu'],
        'linearity_max_resid_tu': speed_info.get('linearity_max_resid_tu'),
        'matches_2d': None,  # filled after logic if needed
    }

    jsonpath = figdir / 'rd_3d_transfer_channel.json'
    with open(jsonpath, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'Saved {jsonpath}')


if __name__ == '__main__':
    main()
