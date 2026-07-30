"""
TEST 1 (FINAL2, clamp-rest fix) -- Channel pulse transfer.

Adapted copy of rd_transfer_channel_final.py (which is itself the
post-de-hack adaptation of rd_transfer_channel.py).  The ONLY physics
change vs the _final version: for the light-held Oregonator (phi>0,
rest state u*=v*=0.0030821) fired ports are now clamped to the TRUE
rest state (u*, u*) between pulses via the new `clamp_rest` constructor
argument of RDSubstrate, instead of (0, 0).  With the old (0, 0) clamp
the port sat below rest as a small sustained sink; the fix removes that
artifact.  Barkley is unchanged (rest state (0,0) = default clamp_rest).

Protocol (unchanged): straight horizontal channel of width W cut through
a wall mask, one pulse fired at a left-edge port spanning the channel,
width sweep down to W=1; free-medium reference; no-stimulus control;
3-probe arrival linearity.  Extra physics checks: wall cells remain
exactly at rest (leak-free walls) and the far probe crosses threshold
exactly once per single pulse (no pacemaking).

Usage:
    ../.venv/bin/python rd_transfer_channel_final2.py [barkley|oregonator|all]

Outputs: Analysis/figures/rd_transfer_channel_final2.{png,json}
"""

import json
import sys

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


def rest_state(kin):
    """Homogeneous rest (u_rest, v_rest) -- also the clamp_rest value."""
    if kin == 'oregonator':
        us = rest_u_star(KINETICS[kin]['kw']['f'], KINETICS[kin]['phi'])
        return us, us
    return 0.0, 0.0


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


def count_crossings(series, thresh=U_THRESH):
    above = series >= thresh
    return int(np.sum((~above[:-1]) & above[1:]))


def strip_mask(x, wall):
    """3-cell vertical strip at column x, clipped to the channel interior."""
    m = np.zeros((NX, NY), bool)
    m[x, CY - 1:CY + 2] = True
    return m & ~wall


def run_case(kin, width=None, nsteps=NSTEPS):
    """width=None -> free medium (no walls). Returns (probe data, rd)."""
    spec = KINETICS[kin]
    u_rest, v_rest = rest_state(kin)
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, clamp_rest=(u_rest, v_rest),
                     **spec['kw'])
    if spec['phi']:
        rd.set_phi(spec['phi'])
    wall = np.zeros((NX, NY), bool)
    if width is not None:
        wall[:] = True
        j0 = CY - width // 2
        j1 = j0 + width               # exactly `width` rows
        wall[:, j0:j1] = False
        rd.set_walls(wall)
    rd.u[~wall] = u_rest
    rd.v[~wall] = v_rest
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
    data = rd.run(nsteps)
    # physics checks: leak-free walls, exactly one crossing per pulse
    data['_wall_leak'] = float(max(np.abs(rd.u[wall]).max(initial=0.0),
                                   np.abs(rd.v[wall]).max(initial=0.0))) \
        if wall.any() else 0.0
    data['_crossings'] = {n: count_crossings(data[n])
                          for n in ('near', 'mid', 'far')}
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
    """Fill speed / linearity-residual fields from res['arrivals_steps']."""
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
    u_rest, v_rest = rest_state(kin)
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, clamp_rest=(u_rest, v_rest),
                     **spec['kw'])
    if spec['phi']:
        rd.set_phi(spec['phi'])
    wall = np.ones((NX, NY), bool)
    wall[:, CY - 8:CY + 8] = False
    rd.set_walls(wall)
    rd.u[~wall] = u_rest
    rd.v[~wall] = v_rest
    rd.run(2000)
    dev = float(np.abs(rd.u[~wall] - u_rest).max())
    quiet = dev < 0.01
    print(f'  control (no stimulus, 100 t.u.): max |u-u*| = {dev:.3e} '
          f'-> {"QUIET" if quiet else "SPONTANEOUS FIRING!"}')
    return {'rest_u': u_rest, 'max_dev_100tu': dev, 'quiet': bool(quiet)}


def sweep_kinetics(kin):
    results = {'kinetics_label': KINETICS[kin]['label'],
               'control': control_run(kin)}
    print(f'=== kinetics = {kin}: {KINETICS[kin]["label"]} ===')
    print('Free-medium reference run...')
    free_data = run_case(kin, width=None)
    free = analyse(free_data, 'free')
    assert free['crossings']['far'] == 1, \
        f"physics check failed: {free['crossings']['far']} far crossings"
    results['free_medium'] = free
    print(f"  free: speed={free['speed']:.4f} cells/step "
          f"({free['speed_cells_per_tu']:.3f} cells/t.u.), "
          f"peak_far={free['peak_far']:.3f}, "
          f"linearity resid={free['linearity_max_resid_steps']:.3f} steps, "
          f"crossings={free['crossings']}", flush=True)

    sweep, traces = [], {}
    for W in WIDTHS:
        data = run_case(kin, width=W)
        r = analyse(data, f'W={W}')
        r['W'] = W
        assert r['wall_leak'] == 0.0, f'wall leak at W={W}'
        if r['transmitted']:
            assert r['crossings']['far'] == 1, \
                f"W={W}: {r['crossings']['far']} far crossings (pacemaking?)"
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

    # channel-speed equality: widest channel vs free medium
    w60 = next(r for r in sweep if r['W'] == 60)
    results['channel_speed_equality'] = {
        'W': 60,
        'channel_speed': w60['speed_cells_per_tu'],
        'free_speed': free['speed_cells_per_tu'],
        'ratio': w60['speed_cells_per_tu'] / free['speed_cells_per_tu'],
    }

    c0 = free['speed_cells_per_tu']
    kappa_star = c0 / 1.0      # Du = 1
    results['eikonal_estimate'] = {
        'c0_cells_per_time_unit': c0,
        'kappa_star_per_cell': kappa_star,
        'critical_channel_width_cells': 2.0 / kappa_star,
    }
    print(f"channel/free speed ratio (W=60): "
          f"{results['channel_speed_equality']['ratio']:.4f}")
    return results


def collect(parts):
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

    fig.suptitle('TEST 1 FINAL2 (clamp-rest fix): pulse transfer through '
                 'a walled channel -- Barkley and Oregonator A', fontsize=13)
    fig.tight_layout()
    figpath = 'figures/rd_transfer_channel_final2.png'
    fig.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f'Saved {figpath}')

    jsonpath = 'figures/rd_transfer_channel_final2.json'
    with open(jsonpath, 'w') as fh:
        json.dump(parts, fh, indent=2)
    print(f'Saved {jsonpath}')


def main(which):
    kins = list(KINETICS) if which == 'all' else [which]
    parts = {}
    for kin in kins:
        parts[kin] = sweep_kinetics(kin)
    if which == 'all':
        collect(parts)
    else:
        p = f'figures/rd_transfer_channel_final2_{which}.json'
        with open(p, 'w') as fh:
            json.dump(parts[which], fh, indent=2)
        print(f'Saved {p}')


if __name__ == '__main__':
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else 'all'
    assert arg in ('barkley', 'oregonator', 'all'), arg
    main(arg)
