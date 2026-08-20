"""
Fan-out / splitter operator (current regime).

Geometry: a horizontal input channel meets a symmetric Y-junction; the two
output arms branch up and down.  A single dark-spot pulse is injected at the
input and the response is recorded on both output arms.

Physical picture
----------------
A pore that splits into two throats should pass a copy of the pulse into each
throat.  This measurement gives the transit time and amplitude on each branch
so the graph simulator can assign correct edge delays.

Equations: same light-sensitive Oregonator as rd_core.py,
    du/dt = (1/eps)[u - u^2 - (f*v + phi)*(u - q)/(u + q)] + div(D . grad u)
    dv/dt = u - v + Dv*laplacian(v)
with explicit operator-split integration and adaptive reaction subcycling.
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
NX, NY = 160, 160
NSTEPS = 500
RUNTIME_S = NSTEPS * DT

U_STAR = 0.0030820999329396943
U_THRESH = 0.2

FIG = Path(__file__).parent / 'figures'
FIG.mkdir(exist_ok=True)


def build_geometry():
    """Return wall mask, spot mask, and probe masks for up/down arms."""
    wall = np.ones((NX, NY), bool)
    # Horizontal input channel, y in [72,88]
    wall[:, 72:88] = False
    # Vertical split at x=80
    wall[80:100, 60:100] = False
    # Upward arm
    wall[80:96, 20:72] = False
    # Downward arm
    wall[80:96, 88:140] = False

    # Round the inner corner a little to avoid artificial pinning
    wall[80:84, 84:88] = False
    wall[80:84, 72:76] = False

    # Input dark spot near left end
    X, Y = np.mgrid[0:NX, 0:NY]
    spot = ((X - 20)**2 + (Y - 80)**2) <= 6**2

    # Output probes near arm tips
    probe_up = np.zeros((NX, NY), bool)
    probe_up[88:94, 30:36] = True
    probe_down = np.zeros((NX, NY), bool)
    probe_down[88:94, 132:138] = True

    return wall, spot, probe_up, probe_down


def main():
    wall, spot, probe_up, probe_down = build_geometry()

    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, dx=DX, kinetics='oregonator',
                     f=F, eps=EPS, q=Q, Du=DU, Dv=DV,
                     clamp_rest=(U_STAR, U_STAR))
    rd.set_walls(wall)
    rd.set_phi(PHI0)
    rd.set_diffusion_tensor(DU, DU, 0.0)
    rd.u[:] = U_STAR
    rd.v[:] = U_STAR

    duration = int(T_FLASH / DT)
    spots = [{'mask': spot, 'times': [0], 'duration': duration,
              'phi_dark': PHI_DARK}]

    t0 = time.time()
    hist = run_darkspot_multi(rd, PHI0, spots, NSTEPS,
                              probes={'up': probe_up, 'down': probe_down})
    runtime = time.time() - t0

    # Extract peaks and first crossings
    u_thresh = U_THRESH
    results = {}
    for name in ['up', 'down']:
        t = np.array(hist['t'])
        y = np.array(hist[name])
        peak = float(y.max())
        crossings = np.where((y[:-1] < u_thresh) & (y[1:] >= u_thresh))[0]
        arrival = float(t[crossings[0]]) if len(crossings) else None
        results[name] = {'peak': peak, 'arrival_tu': arrival,
                         'n_crossings': int(len(crossings))}

    out = {
        'test': 'fan-out splitter (dark-spot Oregonator)',
        'grid': [NX, NY],
        'runtime_s': runtime,
        'parameters': {'f': F, 'eps': EPS, 'q': Q, 'Du': DU, 'Dv': DV,
                       'phi0': PHI0, 'phi_dark': PHI_DARK,
                       't_flash_tu': T_FLASH},
        'outputs': results
    }

    json_path = FIG / 'rd_operator_fanout.json'
    with open(json_path, 'w') as fh:
        json.dump(out, fh, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, label in [('up', 'Up arm'), ('down', 'Down arm')]:
        ax.plot(hist['t'], hist[name], label=label)
    ax.axhline(u_thresh, color='k', ls='--', lw=0.8)
    ax.set_xlabel('Time (t.u.)')
    ax.set_ylabel('Mean activator u')
    ax.set_title('Fan-out splitter response')
    ax.legend()
    ax.set_xlim(0, RUNTIME_S)
    fig.tight_layout()
    fig.savefig(FIG / 'rd_operator_fanout.png', dpi=150)
    plt.close(fig)

    print(json.dumps(out, indent=2))
    print(f'Saved {json_path} and {FIG / "rd_operator_fanout.png"}')


if __name__ == '__main__':
    main()
