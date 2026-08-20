"""
Sub-excitable wave-fragment pilot (out-of-regime).

In the standard excitable regime a launched pulse expands as a full disk.  In a
sub-excitable ("sub-threshold") medium, small localized perturbations can travel
as stable wave fragments that preserve their shape.  These fragments can
collide and implement richer logic than simple annihilation.

This pilot lowers the background illumination phi to move toward the
sub-excitable border and tests whether a single dark-spot flash produces a
stable propagating fragment rather than an expanding target wave.  If a stable
fragment is found, two fragments are launched at an angle and their collision
outcome is recorded.

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
PHI0_STD = 0.010
PHI_DARK = 0.002
T_FLASH = 3.0
DT = 0.05
DX = 1.0
NX, NY = 200, 200
NSTEPS = 400
RUNTIME_S = NSTEPS * DT

U_THRESH = 0.5

FIG = Path(__file__).parent / 'figures'
FIG.mkdir(exist_ok=True)


def rest_state(phi):
    # Approximate rest state for the light-held Oregonator at given phi
    return 0.0030821


def test_fragment(phi0, label):
    """Launch a single dark spot and record whether a stable fragment forms."""
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, dx=DX, kinetics='oregonator',
                     f=F, eps=EPS, q=Q, Du=DU, Dv=DV,
                     clamp_rest=(rest_state(phi0), rest_state(phi0)))
    wall = np.zeros((NX, NY), bool)
    rd.set_walls(wall)
    rd.set_phi(phi0)
    rd.set_diffusion_tensor(DU, DU, 0.0)
    rd.u[:] = rest_state(phi0)
    rd.v[:] = rest_state(phi0)

    X, Y = np.mgrid[0:NX, 0:NY]
    spot = ((X - 50)**2 + (Y - 100)**2) <= 5**2

    duration = int(T_FLASH / DT)
    spots = [{'mask': spot, 'times': [0], 'duration': duration,
              'phi_dark': PHI_DARK}]

    # Track maximum extent in x as a proxy for expansion vs stable fragment
    hist = run_darkspot_multi(rd, phi0, spots, NSTEPS,
                              probes={'centre': (X - 100)**2 + (Y - 100)**2 <= 10**2})
    u_field = rd.u
    active = u_field > U_THRESH
    extent = int(active.sum())
    peak = float(u_field.max())
    return {'phi0': phi0, 'label': label, 'active_cells': extent,
            'peak_u': peak, 'rest_state': rest_state(phi0)}


def test_collision(phi0):
    """Launch two fragments at 90 deg and record the collision outcome."""
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, dx=DX, kinetics='oregonator',
                     f=F, eps=EPS, q=Q, Du=DU, Dv=DV,
                     clamp_rest=(rest_state(phi0), rest_state(phi0)))
    rd.set_walls(np.zeros((NX, NY), bool))
    rd.set_phi(phi0)
    rd.set_diffusion_tensor(DU, DU, 0.0)
    rd.u[:] = rest_state(phi0)
    rd.v[:] = rest_state(phi0)

    X, Y = np.mgrid[0:NX, 0:NY]
    spot_a = ((X - 50)**2 + (Y - 50)**2) <= 5**2
    spot_b = ((X - 50)**2 + (Y - 150)**2) <= 5**2

    duration = int(T_FLASH / DT)
    spots = [
        {'mask': spot_a, 'times': [0], 'duration': duration,
         'phi_dark': PHI_DARK},
        {'mask': spot_b, 'times': [0], 'duration': duration,
         'phi_dark': PHI_DARK}
    ]

    hist = run_darkspot_multi(rd, phi0, spots, NSTEPS,
                              probes={'centre': (X - 120)**2 + (Y - 100)**2 <= 12**2})
    y = np.array(hist['centre'])
    peak = float(y.max())
    crossings = np.where((y[:-1] < U_THRESH) & (y[1:] >= U_THRESH))[0]
    return {'phi0': phi0, 'centre_peak': peak, 'n_crossings': int(len(crossings))}


def main():
    t0 = time.time()
    phi_values = [0.010, 0.009, 0.008, 0.007, 0.006, 0.005]
    fragment_results = [test_fragment(phi, f'phi={phi}') for phi in phi_values]

    # Pick the phi with the smallest active extent that still propagates
    # as the best candidate for collision tests.
    best = min(fragment_results, key=lambda r: r['active_cells'])
    collision = test_collision(best['phi0'])

    runtime = time.time() - t0

    out = {
        'test': 'sub-excitable wave-fragment pilot',
        'grid': [NX, NY],
        'runtime_s': runtime,
        'parameters': {'f': F, 'eps': EPS, 'q': Q, 'Du': DU, 'Dv': DV,
                       'phi_dark': PHI_DARK, 't_flash_tu': T_FLASH,
                       'u_thresh': U_THRESH},
        'fragment_scan': fragment_results,
        'best_phi0': best['phi0'],
        'collision': collision
    }

    json_path = FIG / 'rd_operator_subexcitable_pilot.json'
    with open(json_path, 'w') as fh:
        json.dump(out, fh, indent=2)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([r['phi0'] for r in fragment_results],
            [r['active_cells'] for r in fragment_results], 'o-')
    ax.set_xlabel('Background phi0')
    ax.set_ylabel('Active cells at final time')
    ax.set_title('Sub-excitable fragment stability scan')
    fig.tight_layout()
    fig.savefig(FIG / 'rd_operator_subexcitable_pilot.png', dpi=150)
    plt.close(fig)

    print(json.dumps(out, indent=2))
    print(f'Saved {json_path} and {FIG / "rd_operator_subexcitable_pilot.png"}')


if __name__ == '__main__':
    main()
