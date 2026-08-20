"""
Merge / OR-with-timing operator (current regime).

Geometry: two horizontal input channels (A from the left, B from the right)
converge on a central vertical output stem.  A and B are dark-spot flashes
with a controllable relative delay.  The output probe sits in the vertical
stem above the junction.

Physical picture
----------------
A merging node should fire its output when either input arrives, unless the
two pulses collide head-on and annihilate.  By sweeping the delay we map the
transition from cancellation (coincident) to independent transmission
(offset).

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
NX, NY = 180, 140
NSTEPS = 600
RUNTIME_S = NSTEPS * DT

U_STAR = 0.0030820999329396943
U_THRESH = 0.2

FIG = Path(__file__).parent / 'figures'
FIG.mkdir(exist_ok=True)


def build_geometry():
    """Return wall mask and spot/probe masks."""
    wall = np.ones((NX, NY), bool)
    # Left input channel A: y in [64,80], x in [0,90]
    wall[:90, 64:80] = False
    # Right input channel B: y in [64,80], x in [90,180]
    wall[90:, 64:80] = False
    # Vertical output stem upward from junction
    wall[84:96, 20:64] = False

    # Round inner corners
    wall[84:88, 76:80] = False
    wall[92:96, 76:80] = False

    X, Y = np.mgrid[0:NX, 0:NY]
    spot_a = ((X - 25)**2 + (Y - 72)**2) <= 6**2
    spot_b = ((X - 155)**2 + (Y - 72)**2) <= 6**2

    probe = np.zeros((NX, NY), bool)
    probe[88:94, 30:36] = True

    return wall, spot_a, spot_b, probe


def run_case(delay_tu):
    wall, spot_a, spot_b, probe = build_geometry()
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, dx=DX, kinetics='oregonator',
                     f=F, eps=EPS, q=Q, Du=DU, Dv=DV,
                     clamp_rest=(U_STAR, U_STAR))
    rd.set_walls(wall)
    rd.set_phi(PHI0)
    rd.set_diffusion_tensor(DU, DU, 0.0)
    rd.u[:] = U_STAR
    rd.v[:] = U_STAR

    duration = int(T_FLASH / DT)
    t_a = 0
    t_b = int(delay_tu / DT)
    spots = [
        {'mask': spot_a, 'times': [t_a], 'duration': duration,
         'phi_dark': PHI_DARK},
        {'mask': spot_b, 'times': [t_b], 'duration': duration,
         'phi_dark': PHI_DARK}
    ]

    hist = run_darkspot_multi(rd, PHI0, spots, NSTEPS, probes={'out': probe})
    t = np.array(hist['t'])
    y = np.array(hist['out'])
    peak = float(y.max())
    crossings = np.where((y[:-1] < U_THRESH) & (y[1:] >= U_THRESH))[0]
    arrival = float(t[crossings[0]]) if len(crossings) else None
    return {'delay_tu': delay_tu, 'peak': peak, 'arrival_tu': arrival,
            'n_crossings': int(len(crossings))}


def main():
    delays = [-6.0, -3.0, -1.5, 0.0, 1.5, 3.0, 6.0]
    # Negative: A fires after B; positive: B fires after A
    results = []
    t0 = time.time()
    for d in delays:
        results.append(run_case(d))
    runtime = time.time() - t0

    out = {
        'test': 'merge / OR-with-timing (dark-spot Oregonator)',
        'grid': [NX, NY],
        'runtime_s': runtime,
        'parameters': {'f': F, 'eps': EPS, 'q': Q, 'Du': DU, 'Dv': DV,
                       'phi0': PHI0, 'phi_dark': PHI_DARK,
                       't_flash_tu': T_FLASH, 'u_thresh': U_THRESH},
        'delays': results
    }

    json_path = FIG / 'rd_operator_or_merge.json'
    with open(json_path, 'w') as fh:
        json.dump(out, fh, indent=2)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([r['delay_tu'] for r in results], [r['peak'] for r in results],
            'o-')
    ax.axhline(U_THRESH, color='k', ls='--', lw=0.8)
    ax.set_xlabel('Relative delay A - B (t.u.)')
    ax.set_ylabel('Output peak u')
    ax.set_title('Merge node: output vs input delay')
    fig.tight_layout()
    fig.savefig(FIG / 'rd_operator_or_merge.png', dpi=150)
    plt.close(fig)

    print(json.dumps(out, indent=2))
    print(f'Saved {json_path} and {FIG / "rd_operator_or_merge.png"}')


if __name__ == '__main__':
    main()
