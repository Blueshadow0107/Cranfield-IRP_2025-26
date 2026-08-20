"""
One-way diode operator (out-of-regime geometry test).

Geometry: an asymmetric channel with a sawtooth / ramped wall on one side and
an abrupt step on the other.  The idea is that a pulse travelling in the
forward direction (ramp-first) keeps enough excitable medium ahead of it,
while a reverse pulse runs into the abrupt wall and its curvature pins or
blocks it.

This is an out-of-regime operator because simple asymmetric walls were not
sufficient in the original diode pilot; a more pronounced geometry is needed.

Equations: light-sensitive Oregonator, same scheme as rd_core.py.
"""

import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rd_core import RDSubstrate
from rd_darkspot_driver_multi import run_darkspot_multi


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
F = 1.4375
EPS = 0.05014844822490394
Q = 0.002
DU = 1.0
DV = 0.0
PHI0 = 0.010
PHI_DARK = 0.002
T_FLASH = 3.0
DT = 0.05
DX = 1.0
NX, NY = 200, 100
NSTEPS = 500
RUNTIME_S = NSTEPS * DT

U_STAR = 0.0030820999329396943
U_THRESH = 0.5

FIG = Path(__file__).parent / 'figures'
FIG.mkdir(exist_ok=True)


def build_geometry():
    """Return wall mask, forward/reverse spot masks, and probe masks."""
    wall = np.ones((NX, NY), bool)
    # Main channel body y in [40,60]
    wall[:, 40:60] = False

    # Insert an asymmetric obstacle in the middle
    # Forward direction: left -> right.  Ramp on left, step on right.
    # Reverse direction: right -> left.  Step on left, ramp on right.
    cx = 100
    for dx in range(-20, 21):
        x = cx + dx
        if x < 0 or x >= NX:
            continue
        if dx < 0:
            # ramp: gradually narrows the top half
            height = int(10 * (1 + dx / 20))  # 10 at dx=-20, 0 at dx=0
        else:
            # abrupt step
            height = 0
        wall[x, 40 + height:60] = True

    # Smooth the ramp corner to avoid artificial pinning
    for dx in range(-22, -17):
        x = cx + dx
        wall[x, 48:52] = False

    X, Y = np.mgrid[0:NX, 0:NY]
    spot_fwd = ((X - 25)**2 + (Y - 50)**2) <= 6**2
    spot_rev = ((X - 175)**2 + (Y - 50)**2) <= 6**2

    probe_fwd = np.zeros((NX, NY), bool)
    probe_fwd[170:176, 46:54] = True
    probe_rev = np.zeros((NX, NY), bool)
    probe_rev[24:30, 46:54] = True

    return wall, spot_fwd, spot_rev, probe_fwd, probe_rev


def run_case(direction):
    wall, spot_fwd, spot_rev, probe_fwd, probe_rev = build_geometry()
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, dx=DX, kinetics='oregonator',
                     f=F, eps=EPS, q=Q, Du=DU, Dv=DV,
                     clamp_rest=(U_STAR, U_STAR))
    rd.set_walls(wall)
    rd.set_phi(PHI0)
    rd.set_diffusion_tensor(DU, DU, 0.0)
    rd.u[:] = U_STAR
    rd.v[:] = U_STAR

    duration = int(T_FLASH / DT)
    if direction == 'forward':
        spots = [{'mask': spot_fwd, 'times': [0], 'duration': duration,
                  'phi_dark': PHI_DARK}]
        probe = probe_fwd
    else:
        spots = [{'mask': spot_rev, 'times': [0], 'duration': duration,
                  'phi_dark': PHI_DARK}]
        probe = probe_rev

    hist = run_darkspot_multi(rd, PHI0, spots, NSTEPS, probes={'out': probe})
    t = np.array(hist['t'])
    y = np.array(hist['out'])
    peak = float(y.max())
    crossings = np.where((y[:-1] < U_THRESH) & (y[1:] >= U_THRESH))[0]
    arrival = float(t[crossings[0]]) if len(crossings) else None
    return {'direction': direction, 'peak': peak, 'arrival_tu': arrival,
            'n_crossings': int(len(crossings))}


def main():
    t0 = time.time()
    fwd = run_case('forward')
    rev = run_case('reverse')
    runtime = time.time() - t0

    out = {
        'test': 'asymmetric diode junction (dark-spot Oregonator)',
        'grid': [NX, NY],
        'runtime_s': runtime,
        'parameters': {'f': F, 'eps': EPS, 'q': Q, 'Du': DU, 'Dv': DV,
                       'phi0': PHI0, 'phi_dark': PHI_DARK,
                       't_flash_tu': T_FLASH, 'u_thresh': U_THRESH},
        'forward': fwd,
        'reverse': rev,
        'extinction_ratio': fwd['peak'] / rev['peak'] if rev['peak'] > 0 else float('inf')
    }

    json_path = FIG / 'rd_operator_diode.json'
    with open(json_path, 'w') as fh:
        json.dump(out, fh, indent=2)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['forward', 'reverse'], [fwd['peak'], rev['peak']])
    ax.axhline(U_THRESH, color='k', ls='--', lw=0.8)
    ax.set_ylabel('Output peak u')
    ax.set_title('Asymmetric diode junction')
    fig.tight_layout()
    fig.savefig(FIG / 'rd_operator_diode.png', dpi=150)
    plt.close(fig)

    print(json.dumps(out, indent=2))
    print(f'Saved {json_path} and {FIG / "rd_operator_diode.png"}')


if __name__ == '__main__':
    main()
