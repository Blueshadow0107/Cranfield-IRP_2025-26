"""
TEST 1 (FINAL, de-hacked rd_core) -- Channel pulse transfer.

Adapted copy of rd_transfer_channel.py for the post-de-hack verification
numbers (the original script and its pre-de-hack outputs are untouched).
Differences vs the original:

  1. Kinetics: Barkley (a=0.75, b=0.01, eps=0.02, Du=1, Dv=0) and the
     validated EXCITABLE Oregonator Candidate A (f=1.4375, eps=0.0501,
     phi=0.010, q=0.002, Du=1, Dv=0.6; rest state u*=v*=0.0030821) --
     NOT the old oscillatory Oregonator baseline.  With phi>0 the state
     (0,0) is not rest, so all fields are initialised at (u*, v*) and a
     no-stimulus control verifies the medium stays quiet.
  2. Probes: 3-cell vertical strips (single x column) at x = 64/128/217
     instead of r=2 disks.  A planar channel front has no y-structure, so
     a single-column strip has zero dilution (the old disk diluted the
     Oregonator peak to ~0.54, marginally above threshold).
  3. Three probes instead of two: arrival-time vs x linearity is checked
     explicitly (least-squares slope + max residual), in addition to the
     old mid->far two-probe speed.

Protocol (unchanged physics): straight horizontal channel of width W cut
through a wall mask, one pulse fired at a left-edge port spanning the
channel, width sweep down to W=1.  Pre-de-hack finding to confirm/refute:
no geometric block at any W>=1 for either kinetics.

Usage:
    ../.venv/bin/python rd_transfer_channel_final.py [barkley|oregonator|all|collect]
Per-kinetics runs write partial JSON to a temp dir; 'all' runs both then
collects; 'collect' assembles the final outputs from the partials.

Outputs: Analysis/figures/rd_transfer_channel_final.{png,json}
"""

import json
import os
import sys
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from rd_core import RDSubstrate

NX, NY = 256, 256
CY = NY // 2
X_NEAR, X_MID, X_FAR = 64, 128, 217
X_PORT = (0, 18)         # port slab, flush to domain edge
U_THRESH = 0.5
NSTEPS = 2500
DURATION = 30
DT = 0.05
Q = 0.002

# Validated excitable Oregonator Candidate A
# (Notes/oregonator-excitable-regime-hunt-2026-07-20.md)
OREG_A = dict(f=1.4375, eps=0.05014844822490394, phi=0.010)

KINETICS = {
    'barkley': dict(kw=dict(kinetics='barkley', eps=0.02, Dv=0.0),
                    phi=0.0, label='Barkley (a=0.75, b=0.01, eps=0.02, Dv=0)'),
    'oregonator': dict(kw=dict(kinetics='oregonator', f=OREG_A['f'],
                               eps=OREG_A['eps']),
                       phi=OREG_A['phi'],
                       label='Oregonator A (f=1.4375, eps=0.0501, phi=0.010)'),
}

WIDTHS = [60, 48, 40, 32, 24, 20, 16, 14, 12, 10, 8, 6, 4, 3, 2, 1]
TRACE_WIDTHS = (60, 16, 8, 1)

PART_DIR = os.path.join(tempfile.gettempdir(), 'rd_channel_final_parts')


def rest_u_star(f, phi):
    """Smallest positive root of the u-nullcline with v = u (dv/dt = 0),
    identical to oregonator_regime_hunt.rest_state."""
    def F(u):
        return u - u**2 - (f * u + phi) * (u - Q) / (u + Q)
    ugrid = np.concatenate(([0.0], np.logspace(-7, np.log10(1.2), 4000)))
    vals = F(ugrid)
    for i in range(len(ugrid) - 1):
        if vals[i] > 0.0 and vals[i + 1] <= 0.0:
            return brentq(F, ugrid[i], ugrid[i + 1], xtol=1e-14, rtol=1e-12)
    raise RuntimeError('no rest state found')


def first_crossing(series, thresh=U_THRESH):
    """First index where series >= thresh, linearly interpolated, or None."""
    idx = np.nonzero(series >= thresh)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    if i == 0:
        return 0.0
    s0, s1 = series[i - 1], series[i]
    frac = (thresh - s0) / (s1 - s0) if s1 > s0 else 0.0
    return float(i - 1 + frac)


def strip_mask(x, wall):
    """3-cell vertical strip at column x, clipped to the channel interior."""
    m = np.zeros((NX, NY), bool)
    m[x, CY - 1:CY + 2] = True
    return m & ~wall


def run_case(kin, width=None, nsteps=NSTEPS):
    """width=None -> free medium (no walls). Returns probe data."""
    spec = KINETICS[kin]
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, **spec['kw'])
    if spec['phi']:
        rd.set_phi(spec['phi'])
    wall = np.zeros((NX, NY), bool)
    if width is not None:
        wall[:] = True
        j0 = CY - width // 2
        j1 = j0 + width               # exactly `width` rows
        wall[:, j0:j1] = False
        rd.set_walls(wall)
    if kin == 'oregonator':
        u_star = rest_u_star(spec['kw']['f'], spec['phi'])
        rd.u[~wall] = u_star          # (0,0) is NOT rest when phi > 0
        rd.v[~wall] = u_star
    port = np.zeros((NX, NY), bool)
    if width is None:
        port[X_PORT[0]:X_PORT[1], :] = True    # plane-wave slab
    else:
        port[X_PORT[0]:X_PORT[1],
             CY - width // 2:CY - width // 2 + width] = True
    rd.add_port('in', port)
    rd.add_probe('near', strip_mask(X_NEAR, wall))
    rd.add_probe('mid', strip_mask(X_MID, wall))
    rd.add_probe('far', strip_mask(X_FAR, wall))
    rd.fire('in', duration=DURATION)
    return rd.run(nsteps)


def analyse(data, label):
    arr = {n: first_crossing(data[n]) for n in ('near', 'mid', 'far')}
    res = {'label': label,
           'peak_near': float(data['near'].max()),
           'peak_mid': float(data['mid'].max()),
           'peak_far': float(data['far'].max()),
           'arrivals_steps': arr,
           'transmitted': arr['far'] is not None}
    derive(res)
    return res


def derive(res):
    """Fill speed / linearity-residual fields from res['arrivals_steps'].

    Fits t = (1/c) * x + t0 by least squares over the available probes, so
    the slope is steps/cell and the residual is directly in steps.  Kept
    separate from analyse() so collect() can re-derive these fields from
    the stored arrivals alone."""
    arr = res['arrivals_steps']
    pts = [(x, arr[n]) for n, x in
           (('near', X_NEAR), ('mid', X_MID), ('far', X_FAR))
           if arr[n] is not None]
    if len(pts) >= 2:
        xs = np.array([p[0] for p in pts], dtype=float)
        ts = np.array([p[1] for p in pts])
        inv_c, t0 = np.polyfit(xs, ts, 1)      # steps per cell
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


def control_run(kin):
    """No-stimulus control: the medium must stay at rest."""
    spec = KINETICS[kin]
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, **spec['kw'])
    if spec['phi']:
        rd.set_phi(spec['phi'])
    wall = np.ones((NX, NY), bool)
    wall[:, CY - 8:CY + 8] = False
    rd.set_walls(wall)
    rest = 0.0
    if kin == 'oregonator':
        rest = rest_u_star(spec['kw']['f'], spec['phi'])
        rd.u[~wall] = rest
        rd.v[~wall] = rest
    rd.run(2000)
    dev = float(np.abs(rd.u[~wall] - rest).max())
    quiet = dev < 0.01
    print(f'  control (no stimulus, 100 t.u.): max |u-u*| = {dev:.3e} '
          f'-> {"QUIET" if quiet else "SPONTANEOUS FIRING!"}')
    return {'rest_u': rest, 'max_dev_100tu': dev, 'quiet': bool(quiet)}


def sweep_kinetics(kin):
    results = {'kinetics_label': KINETICS[kin]['label'],
               'control': control_run(kin)}
    print(f'=== kinetics = {kin}: {KINETICS[kin]["label"]} ===')
    print('Free-medium reference run...')
    free = analyse(run_case(kin, width=None), 'free')
    results['free_medium'] = free
    print(f"  free: speed={free['speed']:.4f} cells/step "
          f"({free['speed_cells_per_tu']:.3f} cells/t.u.), "
          f"peak_far={free['peak_far']:.3f}, "
          f"linearity resid={free['linearity_max_resid_steps']:.3f} steps")

    sweep, traces = [], {}
    for W in WIDTHS:
        data = run_case(kin, width=W)
        r = analyse(data, f'W={W}')
        r['W'] = W
        sweep.append(r)
        if W in TRACE_WIDTHS:
            traces[str(W)] = {'t': data['t'][::4].tolist(),
                              'far': data['far'][::4].tolist()}
        if r['transmitted']:
            print(f"  W={W:3d}: speed={r['speed']:.4f} cells/step "
                  f"({r['speed_cells_per_tu']:.3f} cells/t.u.), "
                  f"peak_far={r['peak_far']:.3f}, "
                  f"resid={r['linearity_max_resid_steps']:.3f} steps",
                  flush=True)
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
    thr = results['block_threshold']
    print(f"Block threshold: transmits at W>={thr['min_width_transmitted']}, "
          f"blocked at W<={thr['max_width_blocked']}")

    c0 = free['speed_cells_per_tu']
    kappa_star = c0 / 1.0      # Du = 1
    results['eikonal_estimate'] = {
        'c0_cells_per_time_unit': c0,
        'kappa_star_per_cell': kappa_star,
        'critical_channel_width_cells': 2.0 / kappa_star,
    }
    print(f'Eikonal: kappa*={kappa_star:.2f}/cell -> '
          f'W_crit ~ {2.0/kappa_star:.2f} cells')
    return results


def collect():
    parts = {}
    for kin in KINETICS:
        p = os.path.join(PART_DIR, f'part_{kin}.json')
        with open(p) as fh:
            parts[kin] = json.load(fh)
    # re-derive speed / residual fields from the stored arrivals (single
    # source of truth) in case partials were written by an older analyse()
    for kin in KINETICS:
        derive(parts[kin]['free_medium'])
        for r in parts[kin]['sweep']:
            derive(r)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for row, kin in enumerate(('barkley', 'oregonator')):
        res = parts[kin]
        sweep, free, traces = res['sweep'], res['free_medium'], res['traces']
        thr = res['block_threshold']

        ax = axes[row, 0]
        for W, d in traces.items():
            ax.plot(d['t'], d['far'], label=f'W={W}')
        ax.axhline(U_THRESH, color='k', ls='--', lw=0.8)
        ax.set_xlabel('t (t.u.)')
        ax.set_ylabel('mean u at far probe')
        ax.set_title(f'{kin}: far-probe traces')
        ax.legend(fontsize=8)

        ax = axes[row, 1]
        sp = [(r['W'], r['speed_cells_per_tu']) for r in sweep
              if r['speed_cells_per_tu'] is not None]
        ax.plot([s[0] for s in sp], [s[1] for s in sp], 'o-', ms=4,
                label='channel (3-probe fit)')
        ax.axhline(free['speed_cells_per_tu'], color='r', ls='--',
                   label=f"free medium ({free['speed_cells_per_tu']:.3f})")
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 4, 8, 16, 32, 60])
        ax.set_xticklabels([1, 2, 4, 8, 16, 32, 60])
        ax.set_xlabel('channel width W (cells)')
        ax.set_ylabel('speed (cells / t.u.)')
        ax.set_title(f'{kin}: speed vs width')
        ax.legend(fontsize=8)

        ax = axes[row, 2]
        ax.plot([r['W'] for r in sweep], [r['peak_far'] for r in sweep],
                's-', ms=4, color='darkgreen')
        ax.axhline(U_THRESH, color='k', ls='--', lw=0.8, label='threshold')
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 4, 8, 16, 32, 60])
        ax.set_xticklabels([1, 2, 4, 8, 16, 32, 60])
        ax.set_xlabel('channel width W (cells)')
        ax.set_ylabel('peak u at far probe')
        ax.set_title(f'{kin}: transmission amplitude '
                     f"(transmits W>={thr['min_width_transmitted']})")
        ax.legend(fontsize=8)

    fig.suptitle('TEST 1 FINAL (de-hacked rd_core): pulse transfer through '
                 'a walled channel -- Barkley and Oregonator A', fontsize=13)
    fig.tight_layout()
    figpath = 'figures/rd_transfer_channel_final.png'
    fig.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f'Saved {figpath}')

    jsonpath = 'figures/rd_transfer_channel_final.json'
    with open(jsonpath, 'w') as fh:
        json.dump(parts, fh, indent=2)
    print(f'Saved {jsonpath}')


def main(which):
    if which == 'collect':
        collect()
        return
    kins = list(KINETICS) if which == 'all' else [which]
    os.makedirs(PART_DIR, exist_ok=True)
    for kin in kins:
        res = sweep_kinetics(kin)
        p = os.path.join(PART_DIR, f'part_{kin}.json')
        with open(p, 'w') as fh:
            json.dump(res, fh)
        print(f'wrote partial {p}', flush=True)
    if which == 'all':
        collect()


if __name__ == '__main__':
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else 'all'
    assert arg in ('barkley', 'oregonator', 'all', 'collect'), arg
    main(arg)
