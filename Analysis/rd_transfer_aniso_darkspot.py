"""
rd_transfer_aniso_darkspot.py

Anisotropic routing check with a one-shot dark-spot stimulus, Oregonator only.
A circular dark patch at the centre emits a single pulse into a uniform
anisotropic medium (D_perp = 1, D_par = r, fast axis = x).  The front
extents along x and y are tracked to check c_par/c_perp ~ sqrt(r).

This mirrors rd_transfer_aniso_final2.py but uses the experimentally honest
dark-spot drive instead of a direct field seed.

Outputs:
    Analysis/figures/rd_transfer_aniso_darkspot.{png,json}
"""

import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from rd_core import RDSubstrate
from rd_darkspot_driver import run_darkspot

NX, NY = 256, 256
CX, CY = NX // 2, NY // 2
U_THRESH = 0.5
Q = 0.002
PHI0 = 0.010
PHI_DARK = 0.002
T_FLASH = 3.0
DT = 0.05
DURATION = int(T_FLASH / DT)
SPOT_R = 12

OREG_A = dict(f=1.4375, eps=0.05014844822490394, phi=PHI0)
CASES = [(1, 0.05, 900), (2, 0.05, 700), (4, 0.05, 500)]


def rest_u_star(f, phi):
    def F(u):
        return u - u**2 - (f * u + phi) * (u - Q) / (u + Q)
    ugrid = np.concatenate(([0.0], np.logspace(-7, np.log10(1.2), 4000)))
    vals = F(ugrid)
    for i in range(len(ugrid) - 1):
        if vals[i] > 0.0 and vals[i + 1] <= 0.0:
            return brentq(F, ugrid[i], ugrid[i + 1], xtol=1e-14, rtol=1e-12)
    raise RuntimeError('no rest state found')


U_STAR = rest_u_star(OREG_A['f'], OREG_A['phi'])


def disk_mask(cx, cy, r):
    X, Y = np.mgrid[0:NX, 0:NY]
    return (X - cx)**2 + (Y - cy)**2 <= r**2


def tensor_components(r, theta_deg):
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    return (r * c**2 + 1.0 * s**2, r * s**2 + 1.0 * c**2,
            (r - 1.0) * s * c)


def front_extents(u, thresh=U_THRESH):
    row = u[:, CY]
    col = u[CX, :]
    xs = np.nonzero(row > thresh)[0]
    ys = np.nonzero(col > thresh)[0]
    ex = (xs.max() - CX) if len(xs) else 0
    ey = (ys.max() - CY) if len(ys) else 0
    return ex, ey


def fit_speed(times, extents, lo=30, hi=100):
    sel = [(t, e) for t, e in zip(times, extents) if lo <= e <= hi]
    if len(sel) < 5:
        return None
    t = np.array([s[0] for s in sel])
    e = np.array([s[1] for s in sel])
    slope, _ = np.polyfit(t, e, 1)
    return float(slope)


def run_case(r, dt, nsteps):
    rd = RDSubstrate(nx=NX, ny=NY, dt=dt,
                     kinetics='oregonator',
                     f=OREG_A['f'], eps=OREG_A['eps'],
                     Du=1.0, Dv=0.0,
                     clamp_rest=(U_STAR, U_STAR))
    rd.set_phi(PHI0)
    Dxx, Dyy, Dxy = tensor_components(r, 0.0)
    rd.set_diffusion_tensor(Dxx, Dyy, Dxy)
    rd.u[:] = U_STAR
    rd.v[:] = U_STAR

    spot = disk_mask(CX, CY, SPOT_R)
    times, exs, eys = [], [], []
    chunk = 10
    done = 0
    # Manual chunked dark-spot run
    while done < nsteps:
        steps = min(chunk, nsteps - done)
        data = run_darkspot(rd, spot, PHI0, PHI_DARK, DURATION, steps, probes={})
        done += steps
        t = rd.t * dt
        ex, ey = front_extents(rd.u)
        times.append(t)
        exs.append(ex)
        eys.append(ey)
    c_par = fit_speed(times, exs)
    c_perp = fit_speed(times, eys)
    return c_par, c_perp, (times, exs, eys)


def control_run():
    rd = RDSubstrate(nx=NX, ny=NY, dt=0.05,
                     kinetics='oregonator',
                     f=OREG_A['f'], eps=OREG_A['eps'],
                     Du=1.0, Dv=0.0,
                     clamp_rest=(U_STAR, U_STAR))
    rd.set_phi(PHI0)
    Dxx, Dyy, Dxy = tensor_components(4, 0.0)
    rd.set_diffusion_tensor(Dxx, Dyy, Dxy)
    rd.u[:] = U_STAR
    rd.v[:] = U_STAR
    rd.run(2000)
    dev = float(np.abs(rd.u - U_STAR).max())
    quiet = dev < 0.01
    print(f'  control (r=4, no stimulus): max |u-u*| = {dev:.3e} -> '
          f'{"QUIET" if quiet else "SPONTANEOUS FIRING!"}')
    return {'rest_u': U_STAR, 'max_dev_100tu': dev, 'quiet': bool(quiet)}


def main():
    out = {'kinetics_label': 'Oregonator A (dark-spot drive)',
           'phi0': PHI0, 'phi_dark': PHI_DARK,
           't_flash_tu': T_FLASH, 'spot_radius': SPOT_R,
           'control': control_run(), 'cases': {}}
    tracks = {}
    print('--- Anisotropic routing, dark-spot drive ---')
    for r, dt, nsteps in CASES:
        c_par, c_perp, track = run_case(r, dt, nsteps)
        ratio = c_par / c_perp
        pred = np.sqrt(r)
        dev = abs(ratio - pred) / pred * 100
        out['cases'][str(r)] = {
            'dt': dt, 'c_par': c_par, 'c_perp': c_perp,
            'ratio': ratio, 'sqrt_r': float(pred),
            'deviation_pct': float(dev)}
        tracks[r] = track
        print(f'  r={r}: c_par={c_par:.3f}, c_perp={c_perp:.3f} '
              f'cells/t.u.; ratio={ratio:.3f} vs sqrt(r)={pred:.3f} '
              f'({dev:.2f}% off)', flush=True)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    times, exs, eys = tracks[4]
    axes[0].plot(times, exs, label='x extent (fast)')
    axes[0].plot(times, eys, label='y extent (slow)')
    axes[0].set_xlabel('t (t.u.)')
    axes[0].set_ylabel('front extent (cells)')
    axes[0].set_title('r = 4 front expansion')
    axes[0].legend()

    rs = sorted(float(k) for k in out['cases'])
    ratios = [out['cases'][str(int(r))]['ratio'] for r in rs]
    axes[1].plot(rs, ratios, 'o', color='tab:red', ms=8,
                 label='measured c_par/c_perp')
    rf = np.linspace(1, 4, 100)
    axes[1].plot(rf, np.sqrt(rf), 'k--', label='sqrt(r)')
    for r, ratio in zip(rs, ratios):
        c = out['cases'][str(int(r))]
        axes[1].annotate(f"{c['deviation_pct']:.1f}%", (r, ratio),
                         textcoords='offset points', xytext=(8, -12),
                         fontsize=9)
    axes[1].set_xlabel('anisotropy ratio r')
    axes[1].set_ylabel('c_par / c_perp')
    axes[1].set_title('sqrt(r) check (dark-spot drive)')
    axes[1].legend()
    fig.suptitle('TEST 4a: anisotropic routing -- dark-spot drive '
                 '(Oregonator A)', fontsize=13)
    fig.tight_layout()
    figpath = 'figures/rd_transfer_aniso_darkspot.png'
    fig.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f'Saved {figpath}')

    jsonpath = 'figures/rd_transfer_aniso_darkspot.json'
    with open(jsonpath, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f'Saved {jsonpath}')


if __name__ == '__main__':
    main()
