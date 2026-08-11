"""
Dark-spot drive test: the physically honest f-I encoding.

A uniformly lit BZ gel (background phi0 = 0.010, excitable) with a small
DARK patch at the input end of the channel.  The patch is more excitable
than its surroundings; dimmed enough, it becomes a natural pacemaker,
emitting waves that travel to the probe.  Drive strength = patch darkness
phi_patch in [0, phi0].  No concentration clamps anywhere.

Geometry: 300x40, channel W=16 (matches rd_fi_native.py).  Fields
initialised at the lit-background rest state (u*, v*) everywhere -- the
physical picture is a gel equilibrated under uniform light, then the
pattern switched on.

Per level: 200 t.u., first 50 t.u. discarded, count 0.5-crossings at the
probe.  Control (phi_patch = phi0) must stay quiet.

Outputs: Analysis/figures/rd_darkspot.{json,png}
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rd_core import RDSubstrate

NX, NY = 300, 40
W2 = 8
CY = NY // 2
X_PROBE = 240
DT = 0.05
TU_DISCARD = 50.0
TU_RUN = 200.0
F, EPS, PHI0 = 1.4375, 0.05014844822490394, 0.010
U_STAR = 0.0030821
PATCH_C, PATCH_R = 25, 12
LEVELS = [0.010, 0.008, 0.006, 0.004, 0.002, 0.001, 0.0]
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')


def make_rd(phi_patch):
    wall = np.ones((NX, NY), bool)
    wall[:, CY - W2:CY + W2] = False
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, kinetics='oregonator',
                     f=F, eps=EPS, clamp_rest=(U_STAR, U_STAR))
    phi = np.full((NX, NY), PHI0)
    X, Y = np.mgrid[0:NX, 0:NY].astype(float)
    mask = ((X - PATCH_C) ** 2 + (Y - CY) ** 2) < PATCH_R ** 2
    phi[mask] = phi_patch
    rd.set_phi(phi)
    rd.set_walls(wall)
    rd.u[~wall] = U_STAR
    rd.v[~wall] = U_STAR
    return rd


def run_level(phi_patch):
    rd = make_rd(phi_patch)
    nsteps = int(TU_RUN / DT)
    ndiscard = int(TU_DISCARD / DT)
    series = []
    for n in range(nsteps):
        rd.run(1)
        if n >= ndiscard:
            series.append(float(rd.u[X_PROBE, CY]))
    series = np.array(series)
    idx = np.nonzero((series[:-1] <= 0.5) & (series[1:] > 0.5))[0]
    ncross = len(idx)
    rate = ncross / ((nsteps - ndiscard) * DT)
    isi = np.diff(idx) * DT if ncross > 1 else np.array([])
    return {'phi_patch': float(phi_patch), 'n_pulses': int(ncross),
            'rate': float(rate),
            'isi_mean': float(isi.mean()) if len(isi) else None,
            'isi_cv': float(isi.std() / isi.mean()) if len(isi) else None,
            'peak': float(series.max())}


def main():
    res = [run_level(p) for p in LEVELS]
    for r in res:
        print(f"phi={r['phi_patch']:.3f} pulses={r['n_pulses']:<4} "
              f"rate={r['rate']:.4f} isi_cv={r['isi_cv']} "
              f"peak={r['peak']:.3f}", flush=True)
    out = {'config': {'patch_centre_x': PATCH_C, 'patch_radius': PATCH_R,
                      'phi0': PHI0, 'probe_x': X_PROBE,
                      'tu_run': TU_RUN, 'tu_discard': TU_DISCARD,
                      'note': 'dark-spot drive, no clamps'},
           'levels': res}
    with open(os.path.join(FIG, 'rd_darkspot.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r['phi_patch'] for r in res], [r['rate'] for r in res], 'o-')
    ax.set_xlabel('patch light level phi (dark -> 0)')
    ax.set_ylabel('rate at probe (1/t.u.)')
    ax.set_title('dark-spot drive response')
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'rd_darkspot.png'), dpi=150)
    print('saved rd_darkspot.{json,png}')


if __name__ == '__main__':
    main()
