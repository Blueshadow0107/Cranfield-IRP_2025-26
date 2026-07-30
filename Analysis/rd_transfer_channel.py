"""
TEST 1 -- Channel pulse transfer.

A straight horizontal channel is cut through a wall mask (walls everywhere
except a channel of width W centred vertically).  One pulse is fired at a
left-edge port spanning the full channel width; probes sit at x = 0.5*NX
and x = 0.85*NX on the channel axis.

The sweep is run for TWO kinetics (see rd_core for definitions):
  - 'oregonator': the Tyson-Fife BZ baseline of oregonator_bz_demo.py.
    NOTE: at these parameters the rest state is an unstable node -- the
    medium is a relaxation oscillator, so a fired port's wake re-fires
    spontaneously (~every 70 steps).  Leading-edge measurements (first
    crossing, front speed) remain valid; probe traces show pulse trains.
  - 'barkley': genuinely excitable Barkley BZ phenomenology (stable rest
    state, single clean pulses).

Measured:
  - arrival delay at each probe (first crossing of u = 0.5),
  - wave speed from the two-probe delay difference,
  - peak amplitude at the far probe,
  - channel-width sweep down to the wave-block threshold.

Sanity: the wide-channel speed should approach the free-medium (no walls)
pulse speed, measured separately with a plane-wave slab port.

Outputs: Analysis/figures/rd_transfer_channel_{oregonator,barkley}.png/.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt

from rd_core import RDSubstrate

NX, NY = 256, 256
CY = NY // 2
X_MID = int(0.5 * NX)    # 128
X_FAR = int(0.85 * NX)   # 217
X_PORT = (0, 18)         # port slab, flush to domain edge (avoids leaving
                         # an un-clamped pocket that acts as a pacemaker)
U_THRESH = 0.5
NSTEPS = 2500
DURATION = 30

KINETICS = {
    'oregonator': dict(kinetics='oregonator'),                 # baseline
    'barkley': dict(kinetics='barkley', eps=0.02, Dv=0.0),     # excitable
}


def disk_mask(cx, cy, r):
    X, Y = np.mgrid[0:NX, 0:NY]   # X varies along axis 0 (x), Y along axis 1
    return (X - cx)**2 + (Y - cy)**2 <= r**2


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
    return (i - 1 + frac)


def run_case(kin, width=None, phi=0.0, nsteps=NSTEPS):
    """width=None -> free medium (no walls). Returns probe data."""
    rd = RDSubstrate(nx=NX, ny=NY, **KINETICS[kin])
    if phi:
        rd.set_phi(phi)
    wall = np.zeros((NX, NY), bool)
    if width is not None:
        wall[:] = True
        j0 = CY - width // 2
        j1 = j0 + width               # exactly `width` rows, odd W included
        wall[:, j0:j1] = False
        rd.set_walls(wall)
    port = np.zeros((NX, NY), bool)
    if width is None:
        port[X_PORT[0]:X_PORT[1], :] = True    # plane-wave slab
    else:
        port[X_PORT[0]:X_PORT[1], CY - width // 2:CY - width // 2 + width] = True
    rd.add_port('in', port)
    # small probes (r=2): the u-spike is only a few cells wide, so a large
    # probe dilutes the mean below threshold; clip to the channel interior
    rd.add_probe('mid', disk_mask(X_MID, CY, 2) & ~wall)
    rd.add_probe('far', disk_mask(X_FAR, CY, 2) & ~wall)
    rd.fire('in', duration=DURATION)
    return rd.run(nsteps)


def analyse(data, label):
    t_mid = first_crossing(data['mid'])
    t_far = first_crossing(data['far'])
    res = {'label': label,
           'peak_mid': float(data['mid'].max()),
           'peak_far': float(data['far'].max()),
           't_mid_steps': t_mid, 't_far_steps': t_far}
    if t_mid is not None and t_far is not None and t_far > t_mid:
        res['speed'] = (X_FAR - X_MID) / (t_far - t_mid)  # cells/step
    else:
        res['speed'] = None
    res['transmitted'] = t_far is not None
    return res


def sweep_kinetics(kin):
    results = {}
    print(f'=== kinetics = {kin} ===')
    print('Free-medium reference run...')
    data = run_case(kin, width=None)
    free = analyse(data, 'free')
    results['free_medium'] = free
    print(f"  free: speed={free['speed']:.4f} cells/step "
          f"({free['speed']/0.05:.3f} cells/time-unit), "
          f"peak_far={free['peak_far']:.3f}")

    widths = [60, 48, 40, 32, 24, 20, 16, 14, 12, 10, 8, 6, 4, 3, 2, 1]
    sweep = []
    traces = {}
    for W in widths:
        data = run_case(kin, width=W)
        r = analyse(data, f'W={W}')
        r['W'] = W
        sweep.append(r)
        if W in (60, 16, 8):
            traces[W] = data
        if r['transmitted']:
            print(f"  W={W:3d}: speed={r['speed']:.4f}, "
                  f"peak_far={r['peak_far']:.3f}, "
                  f"delay_mid={r['t_mid_steps']:.0f} steps")
        else:
            print(f"  W={W:3d}: BLOCKED (peak_far={r['peak_far']:.3f})")
    results['sweep'] = sweep

    ok = [r['W'] for r in sweep if r['transmitted']]
    bad = [r['W'] for r in sweep if not r['transmitted']]
    results['block_threshold'] = {
        'min_width_transmitted': min(ok) if ok else None,
        'max_width_blocked': max(bad) if bad else None,
    }
    thr = results['block_threshold']
    print(f"Block threshold: transmits at W>={thr['min_width_transmitted']}, "
          f"blocked at W<={thr['max_width_blocked']}")

    # eikonal estimate of the critical curvature for propagation failure:
    # c(kappa) = c0 - D*kappa = 0  ->  kappa* = c0 / Du;  W_crit ~ 2/kappa*
    c0 = free['speed'] / 0.05  # cells per time unit
    kappa_star = c0 / 1.0      # Du = 1
    results['eikonal_estimate'] = {
        'c0_cells_per_time_unit': c0,
        'kappa_star_per_cell': kappa_star,
        'critical_channel_width_cells': 2.0 / kappa_star,
    }
    print(f'Eikonal: kappa*={kappa_star:.2f}/cell -> '
          f'W_crit ~ {2.0/kappa_star:.2f} cells')
    return results, sweep, traces, free


def phi_scan_oregonator():
    """Light-suppression scan: with no geometric block available (critical
    width sub-grid at phi=0), locate the propagation-failure boundary in
    phi at fixed W=8 and in the free medium."""
    print('phi (light suppression) scan, oregonator, W=8 and free medium...')
    scan = []
    for phi in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]:
        pk_free = run_case('oregonator', width=None, phi=phi)['far'].max()
        pk_w8 = run_case('oregonator', width=8, phi=phi)['far'].max()
        scan.append({'phi': phi,
                     'peak_free': float(pk_free), 'peak_W8': float(pk_w8),
                     'free_transmits': bool(pk_free > U_THRESH),
                     'W8_transmits': bool(pk_w8 > U_THRESH)})
        print(f'  phi={phi:.2f}: free peak={pk_free:.3f}, '
              f'W=8 peak={pk_w8:.3f}')
    return scan


def plot_results(kin, sweep, traces, free, thr):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    for W, d in traces.items():
        ax.plot(d['t'], d['far'], label=f'W={W}')
    ax.axhline(U_THRESH, color='k', ls='--', lw=0.8)
    ax.set_xlabel('t'); ax.set_ylabel('mean u at far probe')
    ax.set_title('Far-probe traces'); ax.legend()

    ax = axes[1]
    sp = [(r['W'], r['speed']) for r in sweep if r['speed'] is not None]
    ax.plot([s[0] for s in sp], [s[1] for s in sp], 'o-', label='channel')
    ax.axhline(free['speed'], color='r', ls='--',
               label=f"free medium ({free['speed']:.3f})")
    ax.set_xlabel('channel width W (cells)')
    ax.set_ylabel('speed (cells/step)')
    ax.set_title('Wave speed vs channel width'); ax.legend()

    ax = axes[2]
    ax.plot([r['W'] for r in sweep], [r['peak_far'] for r in sweep], 's-',
            color='darkgreen')
    ax.axhline(U_THRESH, color='k', ls='--', lw=0.8, label='threshold')
    ax.set_xlabel('channel width W (cells)')
    ax.set_ylabel('peak u at far probe')
    ax.set_title('Transmission amplitude vs W'); ax.legend()

    plt.suptitle(f'TEST 1 ({kin}): pulse transfer through a walled channel '
                 f'(block: transmits W>={thr["min_width_transmitted"]}, '
                 f'blocked W<={thr["max_width_blocked"]})', fontsize=12)
    plt.tight_layout()
    figpath = f'figures/rd_transfer_channel_{kin}.png'
    plt.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f'Saved {figpath}')


def main():
    for kin in ('oregonator', 'barkley'):
        results, sweep, traces, free = sweep_kinetics(kin)
        if kin == 'oregonator':
            results['phi_scan'] = phi_scan_oregonator()
            results['note'] = (
                'Oregonator baseline is oscillatory (unstable rest state): '
                'port wake re-fires spontaneously, so traces are trains; '
                'all reported metrics use leading-edge first crossings.')
        plot_results(kin, sweep, traces, free,
                     results['block_threshold'])
        jsonpath = f'figures/rd_transfer_channel_{kin}.json'
        with open(jsonpath, 'w') as fh:
            json.dump(results, fh, indent=2)
        print(f'Saved {jsonpath}')


if __name__ == '__main__':
    main()
