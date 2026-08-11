"""
rd_transfer_channel_darkspot.py

Channel pulse transfer with a one-shot dark-spot input, Oregonator only.
The spot is a circular patch at the left end of the channel where phi is
reduced from 0.010 to 0.002 for T_FLASH = 3.0 t.u. (60 steps).  This emits
exactly one pulse (per rd_flash_calib).

Geometry mirrors rd_transfer_channel_final2.py so results are directly
comparable to the clamped-port transfer function.

Outputs:
    Analysis/figures/rd_transfer_channel_darkspot.{png,json}
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
CY = NY // 2
X_NEAR, X_MID, X_FAR = 64, 128, 217
X_PORT = (0, 18)
U_THRESH = 0.5
NSTEPS = 1200
DT = 0.05
Q = 0.002
PHI0 = 0.010
PHI_DARK = 0.002
T_FLASH = 3.0                       # t.u. -> single pulse
DURATION = int(T_FLASH / DT)        # 60 steps
SPOT_R = 6

OREG_A = dict(f=1.4375, eps=0.05014844822490394, phi=PHI0)
WIDTHS = [60, 40, 24, 16, 12, 8, 4, 2, 1]
TRACE_WIDTHS = (60, 16, 8, 1)


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


def count_crossings(series, thresh=U_THRESH):
    above = series >= thresh
    return int(np.sum((~above[:-1]) & above[1:]))


def strip_mask(x, wall):
    m = np.zeros((NX, NY), bool)
    m[x, CY - 1:CY + 2] = True
    return m & ~wall


def circle_mask(cx, cy, r):
    X, Y = np.mgrid[0:NX, 0:NY]
    return (X - cx)**2 + (Y - cy)**2 <= r**2


def run_case(width=None):
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT,
                     kinetics='oregonator',
                     f=OREG_A['f'], eps=OREG_A['eps'],
                     Du=1.0, Dv=0.0,
                     clamp_rest=(U_STAR, U_STAR))
    wall = np.zeros((NX, NY), bool)
    if width is not None:
        wall[:] = True
        j0 = CY - width // 2
        j1 = j0 + width
        wall[:, j0:j1] = False
        rd.set_walls(wall)

    rd.u[~wall] = U_STAR
    rd.v[~wall] = U_STAR

    # dark spot centred on the left-edge channel opening
    if width is None:
        spot = circle_mask((X_PORT[0] + X_PORT[1]) // 2, CY, SPOT_R)
    else:
        spot = circle_mask((X_PORT[0] + X_PORT[1]) // 2, CY, SPOT_R)
        # keep only the part inside the channel
        channel = ~wall
        spot = spot & channel

    probes = {
        'near': strip_mask(X_NEAR, wall),
        'mid': strip_mask(X_MID, wall),
        'far': strip_mask(X_FAR, wall),
    }
    for name, mask in probes.items():
        rd.add_probe(name, mask)

    data = run_darkspot(rd, spot, PHI0, PHI_DARK, DURATION, NSTEPS, probes)
    data['_wall_leak'] = float(max(np.abs(rd.u[wall]).max(initial=0.0),
                                   np.abs(rd.v[wall]).max(initial=0.0)))
    data['_crossings'] = {n: count_crossings(data[n]) for n in probes}
    return data


def analyse(data, label):
    arr = {n: first_crossing(data[n]) for n in ('near', 'mid', 'far')}
    res = {'label': label,
           'peak_near': float(data['near'].max()),
           'peak_mid': float(data['mid'].max()),
           'peak_far': float(data['far'].max()),
           'arrivals_steps': arr,
           'crossings': data['_crossings'],
           'wall_leak': data['_wall_leak'],
           'transmitted': arr['far'] is not None}
    derive(res)
    return res


def derive(res):
    arr = res['arrivals_steps']
    pts = [(x, arr[n]) for n, x in
           (('near', X_NEAR), ('mid', X_MID), ('far', X_FAR))
           if arr[n] is not None]
    if len(pts) >= 2:
        xs = np.array([p[0] for p in pts], dtype=float)
        ts = np.array([p[1] for p in pts])
        inv_c, t0 = np.polyfit(xs, ts, 1)
        res['speed'] = float(1.0 / inv_c)
        res['speed_cells_per_tu'] = float(1.0 / inv_c / DT)
        res['linearity_max_resid_steps'] = float(
            np.abs(np.polyval([inv_c, t0], xs) - ts).max())
        if arr['mid'] is not None and arr['far'] is not None \
                and arr['far'] > arr['mid']:
            res['speed_midfar'] = float(
                (X_FAR - X_MID) / (arr['far'] - arr['mid']))
        else:
            res['speed_midfar'] = None
    else:
        res['speed'] = None
        res['speed_cells_per_tu'] = None
        res['speed_midfar'] = None
        res['linearity_max_resid_steps'] = None


def control_run():
    wall = np.ones((NX, NY), bool)
    wall[:, CY - 8:CY + 8] = False
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT,
                     kinetics='oregonator',
                     f=OREG_A['f'], eps=OREG_A['eps'],
                     Du=1.0, Dv=0.0,
                     clamp_rest=(U_STAR, U_STAR))
    rd.set_walls(wall)
    rd.u[~wall] = U_STAR
    rd.v[~wall] = U_STAR
    rd.set_phi(PHI0)
    rd.run(2000)
    dev = float(np.abs(rd.u[~wall] - U_STAR).max())
    quiet = dev < 0.01
    print(f'  control: max |u-u*| = {dev:.3e} -> '
          f'{"QUIET" if quiet else "SPONTANEOUS FIRING!"}')
    return {'rest_u': U_STAR, 'max_dev_100tu': dev, 'quiet': bool(quiet)}


def main():
    results = {'kinetics_label': 'Oregonator A (dark-spot drive)',
               'phi0': PHI0, 'phi_dark': PHI_DARK,
               't_flash_tu': T_FLASH, 'spot_radius': SPOT_R,
               'control': control_run()}

    print('Free-medium reference run...')
    free_data = run_case(width=None)
    free = analyse(free_data, 'free')
    results['free_medium'] = free
    print(f"  free: speed={free['speed_cells_per_tu']:.3f} cells/t.u., "
          f"peak_far={free['peak_far']:.3f}", flush=True)

    sweep, traces = [], {}
    for W in WIDTHS:
        data = run_case(width=W)
        r = analyse(data, f'W={W}')
        r['W'] = W
        sweep.append(r)
        if W in TRACE_WIDTHS:
            traces[str(W)] = {'t': data['t'][::4].tolist(),
                              'far': data['far'][::4].tolist()}
        if r['transmitted']:
            print(f"  W={W:3d}: speed={r['speed_cells_per_tu']:.3f} cells/t.u., "
                  f"peak_far={r['peak_far']:.3f}", flush=True)
        else:
            print(f"  W={W:3d}: BLOCKED (peak_far={r['peak_far']:.3f})",
                  flush=True)

    results['sweep'] = sweep
    results['traces'] = traces
    ok = [r['W'] for r in sweep if r['transmitted']]
    bad = [r['W'] for r in sweep if not r['transmitted']]
    results['block_threshold'] = {
        'min_width_transmitted': min(ok) if ok else None,
        'max_width_blocked': max(bad) if bad else None,
    }

    # Load clamped-port Oregonator data for comparison
    clamped_path = 'figures/rd_transfer_channel_final2.json'
    try:
        with open(clamped_path) as fh:
            clamped_all = json.load(fh)
        clamped = clamped_all['oregonator']
    except Exception as exc:
        print(f'Could not load clamped data ({exc}); comparison omitted.')
        clamped = None

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for W, d in traces.items():
        ax.plot(d['t'], d['far'], label=f'W={W}')
    ax.axhline(U_THRESH, color='k', ls='--', lw=0.8)
    ax.set_xlabel('t (t.u.)')
    ax.set_ylabel('mean u at far probe')
    ax.set_title('dark-spot: far-probe traces')
    ax.legend(fontsize=8)

    ax = axes[1]
    sp = [(r['W'], r['speed_cells_per_tu']) for r in sweep
          if r['speed_cells_per_tu'] is not None]
    ax.plot([s[0] for s in sp], [s[1] for s in sp], 'o-',
            color='tab:blue', ms=5, label='dark-spot')
    ax.axhline(free['speed_cells_per_tu'], color='tab:blue', ls='--',
               label=f"free ({free['speed_cells_per_tu']:.3f})")
    if clamped:
        csp = [(r['W'], r['speed_cells_per_tu']) for r in clamped['sweep']
               if r['speed_cells_per_tu'] is not None]
        ax.plot([s[0] for s in csp], [s[1] for s in csp], 's--',
                color='tab:orange', ms=5, label='clamped-port')
        ax.axhline(clamped['free_medium']['speed_cells_per_tu'],
                   color='tab:orange', ls='--',
                   label=f"clamped free ({clamped['free_medium']['speed_cells_per_tu']:.3f})")
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 4, 8, 16, 32, 60])
    ax.set_xticklabels([1, 2, 4, 8, 16, 32, 60])
    ax.set_xlabel('channel width W (cells)')
    ax.set_ylabel('speed (cells / t.u.)')
    ax.set_title('dark-spot vs clamped-port speed')
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot([r['W'] for r in sweep], [r['peak_far'] for r in sweep],
            'o-', color='tab:blue', ms=5, label='dark-spot')
    if clamped:
        ax.plot([r['W'] for r in clamped['sweep']],
                [r['peak_far'] for r in clamped['sweep']],
                's--', color='tab:orange', ms=5, label='clamped-port')
    ax.axhline(U_THRESH, color='k', ls='--', lw=0.8)
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 4, 8, 16, 32, 60])
    ax.set_xticklabels([1, 2, 4, 8, 16, 32, 60])
    ax.set_xlabel('channel width W (cells)')
    ax.set_ylabel('peak u at far probe')
    ax.set_title('transmission amplitude')
    ax.legend(fontsize=8)

    fig.suptitle('TEST 1: channel pulse transfer -- dark-spot vs clamped-port '
                 '(Oregonator A)', fontsize=13)
    fig.tight_layout()
    figpath = 'figures/rd_transfer_channel_darkspot.png'
    fig.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f'Saved {figpath}')

    jsonpath = 'figures/rd_transfer_channel_darkspot.json'
    with open(jsonpath, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'Saved {jsonpath}')


if __name__ == '__main__':
    main()
