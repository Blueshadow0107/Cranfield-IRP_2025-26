"""
rd_diode_pilot.py -- Standalone one-way pulse diode for the light-sensitive
Oregonator BZ medium.

Tests three implementations of the same asymmetric transition:
  - geometry : physical walls form an abrupt expansion on one side and a
               gradual taper on the other
  - light    : straight physical channel, asymmetric phi barrier creates the
               one-way transition
  - hybrid   : physical walls + phi barrier combined

A single dark-spot flash (T_FLASH = 3.0 t.u.) is fired from one end of the
channel; the output probe at the opposite end records whether the pulse was
transmitted.

Expected behaviour (expansion-block diode):
  Forward  (left -> right) : wide -> gradual taper -> narrow  -> PASSES
  Reverse  (right -> left) : narrow -> abrupt expansion -> wide -> BLOCKS
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from rd_core import RDSubstrate
from rd_darkspot_driver import run_darkspot


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
NX, NY = 240, 100
DX, DT = 1.0, 0.05
NSTEPS = 1200

CY = NY // 2
W_CHAN = 20                     # straight channel half-width in cells
W_WIDE = 30                     # wide-section half-width
W_NARROW = 12                   # narrow-section half-width
X_TAPER_START = 90
X_STEP = 150                    # abrupt expansion faces reverse wave
X_END = 240

F = 1.4375
EPS = 0.05014844822490394
Q = 0.002
DU, DV = 1.0, 0.0
PHI0 = 0.010
PHI_DARK = 0.002
T_FLASH = 3.0
DURATION = int(T_FLASH / DT)
SPOT_R = 6
U_THRESH = 0.5

# Barrier strength for light/hybrid modes (must suppress propagation but not
# kill the forward pulse).  Determined empirically; non-propagating regime in
# the phi regime map begins around phi = 0.030.
PHI_BARRIER = 0.060


def rest_u_star(f, phi):
    def F(u):
        return u - u**2 - (f * u + phi) * (u - Q) / (u + Q)
    ugrid = np.concatenate(([0.0], np.logspace(-7, np.log10(1.2), 4000)))
    vals = F(ugrid)
    for i in range(len(ugrid) - 1):
        if vals[i] > 0.0 and vals[i + 1] <= 0.0:
            return brentq(F, ugrid[i], ugrid[i + 1], xtol=1e-14, rtol=1e-12)
    raise RuntimeError('no rest state found')


U_STAR = rest_u_star(F, PHI0)


def circle_mask(cx, cy, r):
    X, Y = np.mgrid[0:NX, 0:NY]
    return (X - cx)**2 + (Y - cy)**2 <= r**2


def y_top_wide(x):
    """Top wall y-coordinate in the wide section."""
    return CY + W_WIDE


def y_bot_wide(x):
    """Bottom wall y-coordinate in the wide section."""
    return CY - W_WIDE


def y_top_narrow(x):
    """Top wall y-coordinate in the narrow section."""
    return CY + W_NARROW


def y_bot_narrow(x):
    """Bottom wall y-coordinate in the narrow section."""
    return CY - W_NARROW


def y_top_taper(x):
    """Top wall in the taper: gradual from wide to narrow."""
    frac = (x - X_TAPER_START) / (X_STEP - X_TAPER_START)
    return CY + W_WIDE - frac * (W_WIDE - W_NARROW)


def y_bot_taper(x):
    """Bottom wall in the taper: gradual from wide to narrow."""
    frac = (x - X_TAPER_START) / (X_STEP - X_TAPER_START)
    return CY - W_WIDE + frac * (W_WIDE - W_NARROW)


def geometry_walls():
    """Asymmetric wall mask: gradual taper left of x=150, abrupt step right."""
    wall = np.ones((NX, NY), dtype=bool)
    X = np.arange(NX)
    for i in range(NX):
        x = X[i]
        if x < X_TAPER_START:
            y0, y1 = y_bot_wide(x), y_top_wide(x)
        elif x < X_STEP:
            y0, y1 = y_bot_taper(x), y_top_taper(x)
        else:
            y0, y1 = y_bot_narrow(x), y_top_narrow(x)
        j0 = max(0, int(np.floor(y0)))
        j1 = min(NY, int(np.ceil(y1)))
        wall[i, j0:j1] = False
    return wall


def straight_channel_walls():
    """Uniform-width channel for the light-only test."""
    wall = np.ones((NX, NY), dtype=bool)
    wall[:, CY - W_CHAN:CY + W_CHAN] = False
    return wall


def phi_from_geometry(wall, phi_channel=PHI0, phi_wall=PHI_BARRIER):
    """Map a wall geometry into a smooth phi field."""
    phi = np.full((NX, NY), phi_wall, dtype=float)
    phi[~wall] = phi_channel
    return phi


def make_substrate(mode):
    rd = RDSubstrate(nx=NX, ny=NY, dx=DX, dt=DT,
                     kinetics='oregonator',
                     f=F, eps=EPS, q=Q,
                     Du=DU, Dv=DV,
                     clamp_rest=(U_STAR, U_STAR))

    if mode == 'geometry':
        wall = geometry_walls()
        rd.set_walls(wall)
        rd.set_phi(PHI0)
    elif mode == 'light':
        wall = straight_channel_walls()
        rd.set_walls(wall)
        # phi barrier matches the geometry that would create the diode
        geom_wall = geometry_walls()
        rd.set_phi(phi_from_geometry(geom_wall))
    elif mode == 'hybrid':
        wall = geometry_walls()
        rd.set_walls(wall)
        geom_wall = geometry_walls()
        rd.set_phi(phi_from_geometry(geom_wall))
    else:
        raise ValueError(mode)

    rd.u[~rd.wall] = U_STAR
    rd.v[~rd.wall] = U_STAR
    return rd


def probe_mask(x, wall):
    """Vertical strip probe inside the channel."""
    m = np.zeros((NX, NY), bool)
    m[x, :] = True
    return m & ~wall


def input_spot(x, wall):
    """Circular dark spot at the channel entrance."""
    spot = circle_mask(x, CY, SPOT_R)
    return spot & ~wall


def first_crossing(series, thresh=U_THRESH):
    idx = np.nonzero(series >= thresh)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    if i == 0:
        return 0.0
    s0, s1 = series[i - 1], series[i]
    frac = (thresh - s0) / (s1 - s0) if s1 > s0 else 0.0
    return float(i - 1 + frac)


def run_direction(mode, direction):
    """Run one direction: 'forward' (left->right) or 'reverse' (right->left)."""
    rd = make_substrate(mode)

    if direction == 'forward':
        x_in = X_TAPER_START // 2
        x_out = X_STEP + (NX - X_STEP) // 2
    elif direction == 'reverse':
        x_in = X_STEP + (NX - X_STEP) // 2
        x_out = X_TAPER_START // 2
    else:
        raise ValueError(direction)

    spot = input_spot(x_in, rd.wall)
    out_probe = probe_mask(x_out, rd.wall)
    rd.add_probe('out', out_probe)

    data = run_darkspot(rd, spot, PHI0, PHI_DARK, DURATION, NSTEPS)

    peak = float(data['out'].max())
    crossing = first_crossing(data['out'])
    arrival_tu = crossing * DT if crossing is not None else None

    return {
        'mode': mode,
        'direction': direction,
        'peak': peak,
        'crossing_step': crossing,
        'arrival_tu': arrival_tu,
        't': data['t'][::4].tolist(),
        'out': data['out'][::4].tolist(),
    }


def plot_results(results, figpath):
    modes = ['geometry', 'light', 'hybrid']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, mode in zip(axes, modes):
        fwd = next(r for r in results if r['mode'] == mode and r['direction'] == 'forward')
        rev = next(r for r in results if r['mode'] == mode and r['direction'] == 'reverse')
        ax.plot(fwd['t'], fwd['out'], label='forward', lw=1.5)
        ax.plot(rev['t'], rev['out'], label='reverse', lw=1.5)
        ax.axhline(U_THRESH, color='k', ls='--', lw=0.8)
        ax.set_title(f'{mode}')
        ax.set_xlabel('t (t.u.)')
        ax.set_ylabel('mean u at output probe')
        ax.legend(fontsize=9)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle('One-way diode pilot -- forward vs reverse transmission',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f'Saved {figpath}')


def plot_geometry(figpath):
    """Snapshot of the geometry and phi field for reference."""
    rd_geom = make_substrate('geometry')
    rd_light = make_substrate('light')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.imshow(rd_geom.wall.T, origin='lower', cmap='Greys',
              extent=[0, NX, 0, NY], aspect='auto')
    ax.set_title('geometry mode: walls')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax = axes[1]
    ax.imshow(rd_light.phi.T, origin='lower', cmap='viridis',
              extent=[0, NX, 0, NY], aspect='auto', vmin=PHI0, vmax=PHI_BARRIER)
    ax.set_title('light mode: phi field')
    ax.set_xlabel('x')

    ax = axes[2]
    # overlay geometry walls on phi field for hybrid
    rd_hybrid = make_substrate('hybrid')
    im = ax.imshow(rd_hybrid.phi.T, origin='lower', cmap='viridis',
                   extent=[0, NX, 0, NY], aspect='auto', vmin=PHI0, vmax=PHI_BARRIER)
    ax.contour(rd_hybrid.wall.T, levels=[0.5], colors='w', linewidths=0.8,
               extent=[0, NX, 0, NY])
    ax.set_title('hybrid mode: walls + phi')
    ax.set_xlabel('x')
    fig.colorbar(im, ax=ax, label='phi')

    fig.tight_layout()
    fig.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f'Saved {figpath}')


def main():
    outdir = Path('figures')
    outdir.mkdir(exist_ok=True)

    results = []
    for mode in ['geometry', 'light', 'hybrid']:
        for direction in ['forward', 'reverse']:
            print(f'Running {mode} / {direction} ...', flush=True)
            r = run_direction(mode, direction)
            results.append(r)
            print(f"  peak={r['peak']:.3f}, arrival={r['arrival_tu']}", flush=True)

    # Summary metrics
    summary = {}
    for mode in ['geometry', 'light', 'hybrid']:
        fwd = next(r for r in results if r['mode'] == mode and r['direction'] == 'forward')
        rev = next(r for r in results if r['mode'] == mode and r['direction'] == 'reverse')
        summary[mode] = {
            'forward_peak': fwd['peak'],
            'reverse_peak': rev['peak'],
            'forward_arrival_tu': fwd['arrival_tu'],
            'reverse_arrival_tu': rev['arrival_tu'],
            'extinction_ratio': float(fwd['peak'] / rev['peak']) if rev['peak'] > 0 else float('inf'),
            'forward_transmitted': fwd['peak'] >= U_THRESH,
            'reverse_blocked': rev['peak'] < U_THRESH,
        }

    payload = {
        'parameters': {
            'nx': NX, 'ny': NY, 'dx': DX, 'dt': DT, 'nsteps': NSTEPS,
            'f': F, 'eps': EPS, 'q': Q, 'Du': DU, 'Dv': DV,
            'phi0': PHI0, 'phi_dark': PHI_DARK, 'phi_barrier': PHI_BARRIER,
            't_flash_tu': T_FLASH, 'spot_radius': SPOT_R,
            'x_taper_start': X_TAPER_START, 'x_step': X_STEP,
            'w_wide': W_WIDE, 'w_narrow': W_NARROW,
        },
        'summary': summary,
        'runs': results,
    }

    jsonpath = outdir / 'rd_diode_pilot.json'
    with open(jsonpath, 'w') as fh:
        json.dump(payload, fh, indent=2)
    print(f'Saved {jsonpath}')

    plot_results(results, outdir / 'rd_diode_pilot_series.png')
    plot_geometry(outdir / 'rd_diode_pilot_geometry.png')

    print('\nSummary:')
    for mode, s in summary.items():
        status = 'OK' if s['forward_transmitted'] and s['reverse_blocked'] else 'PARTIAL/FAIL'
        print(f"  {mode:10s}: fwd={s['forward_peak']:.3f}, rev={s['reverse_peak']:.3f}, "
              f"ratio={s['extinction_ratio']:.1f}  [{status}]")


if __name__ == '__main__':
    main()
