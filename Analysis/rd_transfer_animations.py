"""rd_transfer_animations.py -- export ParaView animations of the four
transfer-function experiments.

For each experiment we run the Oregonator-A case with the same geometry as the
quantitative scripts, but save the full u(x,y), v(x,y), and phi(x,y) fields at
regular intervals.  Each experiment produces a .pvd collection that can be
opened in ParaView; within ParaView toggle between the 'u', 'v', and 'phi'
scalars.

Outputs: Analysis/figures/transfer_animations/

Usage:
    cd Analysis
    ../.venv/bin/python rd_transfer_animations.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pyvista as pv

from rd_core import RDSubstrate
from scipy.optimize import brentq

FIG = Path(__file__).parent / 'figures' / 'transfer_animations'
FIG.mkdir(parents=True, exist_ok=True)

DT = 0.05
Q = 0.002
U_THRESH = 0.5
OREG_A = dict(f=1.4375, eps=0.05014844822490394, phi=0.010)


def rest_u_star(f, phi):
    def F(u):
        return u - u**2 - (f * u + phi) * (u - Q) / (u + Q)
    ugrid = np.concatenate(([0.0], np.logspace(-7, np.log10(1.2), 4000)))
    vals = F(ugrid)
    for i in range(len(ugrid) - 1):
        if vals[i] > 0.0 and vals[i + 1] <= 0.0:
            return brentq(F, ugrid[i], ugrid[i + 1], xtol=1e-14, rtol=1e-12)
    raise RuntimeError('no rest state found')


def make_oregonator(nx, ny, dt=DT):
    us = rest_u_star(OREG_A['f'], OREG_A['phi'])
    rd = RDSubstrate(nx=nx, ny=ny, dt=dt,
                     kinetics='oregonator', f=OREG_A['f'], eps=OREG_A['eps'],
                     clamp_rest=(us, us))
    rd.set_phi(OREG_A['phi'])
    rd.u[:] = us
    rd.v[:] = us
    return rd, us


def phi_array(rd):
    """Return phi as a 2D array (scalar phi is broadcast)."""
    if np.isscalar(rd.phi):
        return np.full_like(rd.u, float(rd.phi))
    return rd.phi.copy()


def write_vti(path, rd):
    """Write u, v, phi fields from rd to a .vti file using pyvista."""
    u = rd.u.astype(np.float32)
    v = rd.v.astype(np.float32)
    p = phi_array(rd).astype(np.float32)

    grid = pv.ImageData()
    grid.dimensions = (u.shape[0], u.shape[1], 1)
    grid.spacing = (1.0, 1.0, 1.0)
    grid.origin = (0.0, 0.0, 0.0)
    grid.point_data['u'] = u.T.ravel(order='C')
    grid.point_data['v'] = v.T.ravel(order='C')
    grid.point_data['phi'] = p.T.ravel(order='C')
    grid.save(str(path))


def write_pvd(path, frames):
    """Write a ParaView collection file. frames: list of (time, filename)."""
    from xml.etree.ElementTree import Element, SubElement, ElementTree
    root = Element('VTKFile', type='Collection', version='0.1',
                   byte_order='LittleEndian')
    coll = SubElement(root, 'Collection')
    for t, fname in frames:
        SubElement(coll, 'DataSet', timestep=str(float(t)), group='',
                   part='0', file=fname)
    ElementTree(root).write(str(path), encoding='UTF-8', xml_declaration=True)


def run_animation(name, rd, nsteps, stride=10, pre_steps=0):
    """Run rd for nsteps, saving a frame every stride steps."""
    case_dir = FIG / name
    case_dir.mkdir(parents=True, exist_ok=True)
    frames = []

    if pre_steps:
        rd.run(pre_steps)

    for s in range(0, nsteps, stride):
        steps = min(stride, nsteps - s)
        rd.run(steps)
        fname = f'frame_{s + steps:05d}.vti'
        write_vti(case_dir / fname, rd)
        frames.append(((s + steps) * rd.dt, fname))

    write_pvd(FIG / f'{name}.pvd',
              [(t, f'{name}/{f}') for t, f in frames])
    return len(frames)


# -----------------------------------------------------------------------------
# 1. Channel pulse transfer (W = 16)
# -----------------------------------------------------------------------------
def animate_channel():
    nx = ny = 256
    cy = ny // 2
    width = 16
    rd, us = make_oregonator(nx, ny)

    wall = np.ones((nx, ny), bool)
    j0 = cy - width // 2
    wall[:, j0:j0 + width] = False
    rd.set_walls(wall)
    rd.u[~wall] = us
    rd.v[~wall] = us

    port = np.zeros((nx, ny), bool)
    port[0:18, j0:j0 + width] = True
    rd.add_port('in', port)
    rd.fire('in', duration=30)

    return run_animation('channel_w16', rd, nsteps=1600, stride=10)


# -----------------------------------------------------------------------------
# 2. Frequency response (train of 6 pulses, period just above refractory)
# -----------------------------------------------------------------------------
def animate_frequency():
    nx, ny = 300, 48
    cy = ny // 2
    width = 16
    rd, us = make_oregonator(nx, ny)

    wall = np.ones((nx, ny), bool)
    wall[:, cy - width // 2:cy + width // 2] = False
    rd.set_walls(wall)
    rd.u[~wall] = us
    rd.v[~wall] = us

    port = np.zeros((nx, ny), bool)
    port[0:18, cy - width // 2:cy + width // 2] = True
    rd.add_port('in', port)

    # period ~ 70 steps (3.5 t.u.) -- close to refractory, some pulses drop
    period = 70
    times = [k * period for k in range(6)]
    rd.pulse_train('in', times, duration=30)
    nsteps = times[-1] + 30 + 800
    return run_animation('frequency_train', rd, nsteps=nsteps, stride=10)


# -----------------------------------------------------------------------------
# 3. Two-input collision logic: A only and A+B coincident
# -----------------------------------------------------------------------------
def animate_logic():
    nx, ny = 256, 200
    cy = ny // 2
    w = 20
    w2 = w // 2
    tj = 96
    rd, us = make_oregonator(nx, ny)

    wall = np.ones((nx, ny), bool)
    wall[:, cy - w2:cy + w2] = False
    wall[tj - w2:tj + w2, cy:172] = False
    rd.set_walls(wall)
    rd.u[~wall] = us
    rd.v[~wall] = us

    pA = np.zeros((nx, ny), bool)
    pA[25:43, cy - w2:cy + w2] = True
    pB = np.zeros((nx, ny), bool)
    pB[tj - w2:tj + w2, 140:158] = True
    rd.add_port('A', pA)
    rd.add_port('B', pB)

    results = {}
    for case, fire_b in [('A_only', False), ('A_and_B', True)]:
        rd2, _ = make_oregonator(nx, ny)
        rd2.set_walls(wall)
        rd2.u[~wall] = us
        rd2.v[~wall] = us
        rd2.add_port('A', pA)
        rd2.add_port('B', pB)
        rd2.pulse_train('A', [0], duration=30)
        if fire_b:
            rd2.pulse_train('B', [0], duration=30)
        n = run_animation(f'logic_{case}', rd2, nsteps=1000, stride=10)
        results[case] = n
    return results


# -----------------------------------------------------------------------------
# 4. Anisotropic routing (r = 4)
# -----------------------------------------------------------------------------
def animate_anisotropy():
    nx = ny = 256
    cx = cy = nx // 2
    r = 4
    rd, us = make_oregonator(nx, ny)

    th = np.deg2rad(0.0)
    c, s = np.cos(th), np.sin(th)
    Dxx = r * c**2 + 1.0 * s**2
    Dyy = r * s**2 + 1.0 * c**2
    Dxy = (r - 1.0) * s * c
    rd.set_diffusion_tensor(Dxx, Dyy, Dxy)
    rd.u[:] = us
    rd.v[:] = us

    X, Y = np.mgrid[0:nx, 0:ny]
    seed = (X - cx)**2 + (Y - cy)**2 <= 25**2
    rd.u[seed] = 0.8
    rd.v[seed] = 0.2

    return run_animation('anisotropic_r4', rd, nsteps=700, stride=10)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    summary = {}
    print('[anim] channel pulse transfer ...')
    summary['channel_w16'] = animate_channel()
    print('[anim] frequency response ...')
    summary['frequency_train'] = animate_frequency()
    print('[anim] collision logic ...')
    summary['logic'] = animate_logic()
    print('[anim] anisotropic routing ...')
    summary['anisotropic_r4'] = animate_anisotropy()

    with open(FIG / 'animation_summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2)

    print('\n[done] transfer animations saved to', FIG)
    for k, v in summary.items():
        print(f'  {k}: {v} frames')
    print('Open the .pvd files in ParaView and switch between u/v/phi scalars.')


if __name__ == '__main__':
    main()
