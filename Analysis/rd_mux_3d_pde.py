"""
3D PDE validation of a 2-to-1 multiplexer.

Geometry: a horizontal output channel runs from x=40 to x=75 at y=40, z=40.
- Data A enters from the left along x and meets the output at x=50.
- Control S enters from the bottom along y and meets the A channel at (45,40,40)
  via a shorter path, so S arrives first and can make the junction refractory.
- Data B enters from the front along z and joins the output channel at x=60,
  downstream of the A/S junction, so B is unaffected by S.

The MUX truth table for one-hot data inputs:
  S=0, A=1, B=0 -> output fires (A passes)
  S=1, A=1, B=0 -> output silent (S blocks A)
  S=0, A=0, B=1 -> output fires (B passes)
  S=1, A=0, B=1 -> output fires (B passes)

Equations: 3D light-sensitive Oregonator, same as rd_core_3d.py.
"""

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
DT = 0.05
DX = 1.0
NX, NY, NZ = 80, 80, 80
NSTEPS = 500
DURATION = int(3.0 / DT)

U_STAR = 0.0030821
U_THRESH = 0.2

FIG = Path(__file__).parent / 'figures'
FIG.mkdir(exist_ok=True)


def build_geometry():
    """Return wall mask and port/probe masks."""
    wall = np.ones((NX, NY, NZ), bool)

    # Output channel: x in [40,75], y=38:42, z=38:42
    wall[40:75, 38:42, 38:42] = False

    # A input channel: x in [0,50], y=38:42, z=38:42
    wall[0:50, 38:42, 38:42] = False

    # S control channel: x=43:47, y in [0,40], z=38:42 (meets A channel)
    wall[43:47, 0:40, 38:42] = False

    # B input channel: joins output at x=60; path from z=0 to z=40 at x=58:62, y=38:42
    wall[58:62, 38:42, 0:40] = False
    # short connector to output
    wall[58:62, 38:42, 38:42] = False

    # Ports
    port_a = np.zeros((NX, NY, NZ), bool)
    port_a[5:10, 38:42, 38:42] = True

    port_s = np.zeros((NX, NY, NZ), bool)
    port_s[43:47, 5:10, 38:42] = True

    port_b = np.zeros((NX, NY, NZ), bool)
    port_b[58:62, 38:42, 5:10] = True

    # Output probe near x=70
    probe_o = np.zeros((NX, NY, NZ), bool)
    probe_o[68:72, 38:42, 38:42] = True

    return wall, port_a, port_s, port_b, probe_o


def run_pattern(s: bool, a: bool, b: bool):
    wall, port_a, port_s, port_b, probe_o = build_geometry()
    rd = RDSubstrate3D(nx=NX, ny=NY, nz=NZ, dx=DX, dt=DT,
                       f=F, eps=EPS, q=Q, Du=DU, Dv=DV,
                       phi=PHI0, clamp_rest=(U_STAR, U_STAR))
    rd.set_walls(wall)
    rd.set_phi(PHI0)
    rd.u[:] = U_STAR
    rd.v[:] = U_STAR

    rd.add_port('A', port_a)
    rd.add_port('S', port_s)
    rd.add_port('B', port_b)
    rd.add_probe('O', probe_o)

    if s:
        rd.fire('S', value=0.8, v_value=0.2, duration=DURATION)
    if a:
        rd.fire('A', value=0.8, v_value=0.2, duration=DURATION)
    if b:
        rd.fire('B', value=0.8, v_value=0.2, duration=DURATION)

    series = []
    for _ in range(NSTEPS):
        rd.run(1)
        series.append(float(rd.u[probe_o].mean()))

    t = np.arange(len(series)) * DT
    y = np.array(series)
    peak = float(y.max())
    crossings = np.where((y[:-1] < U_THRESH) & (y[1:] >= U_THRESH))[0]
    arrival = float(t[crossings[0]]) if len(crossings) else None
    return {'S': int(s), 'A': int(a), 'B': int(b),
            'peak': peak, 'arrival_tu': arrival,
            'n_crossings': int(len(crossings))}


def run_pattern_args(args):
    """Top-level wrapper so ProcessPoolExecutor can pickle the call."""
    return run_pattern(*args)


def main():
    patterns = [
        (False, True, False),   # select A
        (True, True, False),    # block A
        (False, False, True),   # select B
        (True, False, True),    # select B with control
    ]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run_pattern_args, patterns))
    runtime = time.time() - t0

    out = {
        'test': '2-to-1 multiplexer (3D PDE)',
        'grid': [NX, NY, NZ],
        'runtime_s': runtime,
        'parameters': {'f': F, 'eps': EPS, 'q': Q, 'Du': DU, 'Dv': DV,
                       'phi0': PHI0, 'u_thresh': U_THRESH},
        'truth_table': results
    }

    json_path = FIG / 'rd_mux_3d_pde.json'
    with open(json_path, 'w') as fh:
        json.dump(out, fh, indent=2)

    print(json.dumps(out, indent=2))
    print(f'Saved {json_path}')


if __name__ == '__main__':
    main()
