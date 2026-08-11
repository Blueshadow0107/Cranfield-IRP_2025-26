"""
rd_light_walls_pilot.py

Compare physical no-flux walls (Case A) vs high-phi light barriers
(Case B) for confining a pulse in a straight 2D channel.

Case A: channel carved by a wall mask (true Neumann no-flux walls).
Case B: same outline, but no wall mask; the "wall" regions are set to
        phi=0.040, the regime-map non-propagating threshold, while the
        channel stays at phi0=0.010.

Input: one-shot circular dark-spot flash at the channel entrance
       (phi reduced from 0.010 to 0.002 for T_FLASH = 3.0 t.u.).

Outputs:
    Analysis/figures/rd_light_walls_pilot.{png,json}
"""

import json
import time

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
NX, NY = 256, 128
CY = NY // 2
CHANNEL_WIDTH = 16
X_PORT = (0, 18)
X_NEAR, X_MID, X_FAR = 64, 128, 217

DT = 0.05
Q = 0.002
PHI0 = 0.010          # channel / free-medium background
PHI_BARRIER = 0.040   # regime-map non-propagating threshold
PHI_DARK = 0.002
T_FLASH = 3.0
DURATION = int(T_FLASH / DT)
SPOT_R = 6

U_THRESH = 0.5
NSTEPS = 1200

OREG = dict(f=1.4375, eps=0.05014844822490394, phi=PHI0)


def rest_u_star(phi):
    """Homogeneous rest state of the light-held Oregonator."""
    def F(u):
        return u - u**2 - (OREG['f'] * u + phi) * (u - Q) / (u + Q)
    ugrid = np.concatenate(([0.0], np.logspace(-7, np.log10(1.2), 4000)))
    vals = F(ugrid)
    for i in range(len(ugrid) - 1):
        if vals[i] > 0.0 and vals[i + 1] <= 0.0:
            return brentq(F, ugrid[i], ugrid[i + 1], xtol=1e-14, rtol=1e-12)
    raise RuntimeError(f'no rest state found for phi={phi}')


U_STAR = rest_u_star(PHI0)
U_STAR_BARRIER = rest_u_star(PHI_BARRIER)


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


def channel_mask():
    """Return a bool mask that is True inside the channel."""
    m = np.zeros((NX, NY), bool)
    j0 = CY - CHANNEL_WIDTH // 2
    j1 = j0 + CHANNEL_WIDTH
    m[:, j0:j1] = True
    return m


def circle_mask(cx, cy, r):
    X, Y = np.mgrid[0:NX, 0:NY]
    return (X - cx)**2 + (Y - cy)**2 <= r**2


def probe_strip(x, inside):
    """Thin strip probe at x, restricted to inside mask."""
    m = np.zeros((NX, NY), bool)
    m[x, CY - 1:CY + 2] = True
    return m & inside


def make_spot(inside):
    """Dark spot centred on the left edge of the channel opening."""
    cx = (X_PORT[0] + X_PORT[1]) // 2
    spot = circle_mask(cx, CY, SPOT_R) & inside
    return spot


def run_case_a():
    """Physical walls."""
    print('  Case A: physical no-flux walls', flush=True)
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT,
                     kinetics='oregonator', f=OREG['f'], eps=OREG['eps'],
                     Du=1.0, Dv=0.0,
                     clamp_rest=(U_STAR, U_STAR))

    inside = channel_mask()
    wall = ~inside
    rd.set_walls(wall)

    rd.u[:] = U_STAR
    rd.v[:] = U_STAR
    # wall cells forced to 0 by set_walls, so rest state is only in channel

    probes = {
        'near': probe_strip(X_NEAR, inside),
        'mid': probe_strip(X_MID, inside),
        'far': probe_strip(X_FAR, inside),
    }
    for name, mask in probes.items():
        rd.add_probe(name, mask)

    spot = make_spot(inside)
    t0 = time.time()
    data = run_darkspot(rd, spot, PHI0, PHI_DARK, DURATION, NSTEPS, probes)
    runtime = time.time() - t0

    # confinement: max signal outside the channel (should be ~0 for true walls)
    outside = ~inside
    max_outside_u = float(rd.u[outside].max(initial=0.0))
    max_outside_v = float(rd.v[outside].max(initial=0.0))

    return {
        'rd': rd,
        'data': data,
        'inside': inside,
        'wall': wall,
        'runtime_s': runtime,
        'max_outside_u': max_outside_u,
        'max_outside_v': max_outside_v,
    }


def run_case_b():
    """High-phi barriers, no physical walls."""
    print('  Case B: high-phi barriers (no wall mask)', flush=True)
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT,
                     kinetics='oregonator', f=OREG['f'], eps=OREG['eps'],
                     Du=1.0, Dv=0.0,
                     clamp_rest=(U_STAR, U_STAR))

    inside = channel_mask()
    # no walls anywhere
    wall = np.zeros((NX, NY), bool)
    rd.set_walls(wall)

    # spatially varying phi: low in channel, high in barrier
    phi = np.full((NX, NY), PHI_BARRIER, dtype=float)
    phi[inside] = PHI0
    rd.set_phi(phi)

    # initialise entire domain at the channel rest state.
    # The barrier region will relax toward its own (higher) rest state.
    rd.u[:] = U_STAR
    rd.v[:] = U_STAR

    probes = {
        'near': probe_strip(X_NEAR, inside),
        'mid': probe_strip(X_MID, inside),
        'far': probe_strip(X_FAR, inside),
    }
    for name, mask in probes.items():
        rd.add_probe(name, mask)

    spot = make_spot(inside)
    t0 = time.time()
    data = run_darkspot(rd, spot, PHI0, PHI_DARK, DURATION, NSTEPS, probes)
    runtime = time.time() - t0

    outside = ~inside
    max_outside_u = float(rd.u[outside].max(initial=0.0))
    max_outside_v = float(rd.v[outside].max(initial=0.0))

    # also check whether barrier region has drifted to its own rest state
    return {
        'rd': rd,
        'data': data,
        'inside': inside,
        'wall': wall,
        'runtime_s': runtime,
        'max_outside_u': max_outside_u,
        'max_outside_v': max_outside_v,
        'mean_barrier_u_final': float(rd.u[outside].mean()),
        'mean_barrier_v_final': float(rd.v[outside].mean()),
    }


def analyse(data, inside):
    arr = {n: first_crossing(data[n]) for n in ('near', 'mid', 'far')}
    res = {
        'arrivals_steps': arr,
        'peak_near': float(data['near'].max()),
        'peak_mid': float(data['mid'].max()),
        'peak_far': float(data['far'].max()),
        'crossings': {n: count_crossings(data[n]) for n in ('near', 'mid', 'far')},
        'transmitted': arr['far'] is not None,
    }

    pts = [(x, arr[n]) for n, x in
           (('near', X_NEAR), ('mid', X_MID), ('far', X_FAR))
           if arr[n] is not None]
    if len(pts) >= 2:
        xs = np.array([p[0] for p in pts], dtype=float)
        ts = np.array([p[1] for p in pts])
        inv_c, t0 = np.polyfit(xs, ts, 1)
        res['speed_cells_per_tu'] = float(1.0 / inv_c / DT)
        res['linearity_max_resid_steps'] = float(
            np.abs(np.polyval([inv_c, t0], xs) - ts).max())
    else:
        res['speed_cells_per_tu'] = None
        res['linearity_max_resid_steps'] = None
    return res


def plot_state(rd, inside, ax, title, cmap='viridis'):
    im = ax.imshow(rd.u.T, origin='lower', cmap=cmap, vmin=0, vmax=0.8,
                   extent=[0, NX, 0, NY])
    # outline channel
    j0 = CY - CHANNEL_WIDTH // 2
    j1 = j0 + CHANNEL_WIDTH
    ax.plot([0, NX], [j0, j0], 'w--', lw=0.8)
    ax.plot([0, NX], [j1, j1], 'w--', lw=0.8)
    ax.set_title(title)
    ax.set_xlabel('x (cells)')
    ax.set_ylabel('y (cells)')
    return im


def main():
    results = {
        'params': {
            'nx': NX, 'ny': NY, 'channel_width': CHANNEL_WIDTH,
            'dt': DT, 'phi0': PHI0, 'phi_barrier': PHI_BARRIER,
            'phi_dark': PHI_DARK, 't_flash_tu': T_FLASH,
            'spot_radius': SPOT_R, 'nsteps': NSTEPS,
            'f': OREG['f'], 'eps': OREG['eps'], 'q': Q,
            'Du': 1.0, 'Dv': 0.0,
            'rest_u_star_channel': U_STAR,
            'rest_u_star_barrier': U_STAR_BARRIER,
        }
    }

    print('Running light-as-walls pilot...', flush=True)
    case_a = run_case_a()
    case_b = run_case_b()

    res_a = analyse(case_a['data'], case_a['inside'])
    res_b = analyse(case_b['data'], case_b['inside'])

    res_a.update({
        'label': 'physical walls',
        'runtime_s': case_a['runtime_s'],
        'max_outside_u': case_a['max_outside_u'],
        'max_outside_v': case_a['max_outside_v'],
    })
    res_b.update({
        'label': 'high-phi barrier',
        'runtime_s': case_b['runtime_s'],
        'max_outside_u': case_b['max_outside_u'],
        'max_outside_v': case_b['max_outside_v'],
        'mean_barrier_u_final': case_b['mean_barrier_u_final'],
        'mean_barrier_v_final': case_b['mean_barrier_v_final'],
    })

    results['case_a_physical_walls'] = res_a
    results['case_b_phi_barrier'] = res_b

    print(f"Case A: speed={res_a['speed_cells_per_tu']}, "
          f"peak_far={res_a['peak_far']:.3f}, "
          f"outside_u={res_a['max_outside_u']:.3f}", flush=True)
    print(f"Case B: speed={res_b['speed_cells_per_tu']}, "
          f"peak_far={res_b['peak_far']:.3f}, "
          f"outside_u={res_b['max_outside_u']:.3f}", flush=True)

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.2])

    def add_snapshot(grid_pos, rd_obj, inside, title):
        ax = fig.add_subplot(grid_pos)
        im = ax.imshow(rd_obj.u.T, origin='lower', cmap='hot',
                       vmin=0, vmax=0.8, extent=[0, NX, 0, NY])
        j0 = CY - CHANNEL_WIDTH // 2
        j1 = j0 + CHANNEL_WIDTH
        ax.plot([0, NX], [j0, j0], 'c--', lw=1.0)
        ax.plot([0, NX], [j1, j1], 'c--', lw=1.0)
        ax.set_title(title)
        ax.set_xlabel('x (cells)')
        ax.set_ylabel('y (cells)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return ax

    add_snapshot(gs[0, 0], case_a['rd'], case_a['inside'],
                 f'Case A: physical walls (t={NSTEPS*DT:.1f} t.u.)')
    add_snapshot(gs[0, 1], case_b['rd'], case_b['inside'],
                 f'Case B: high-phi barrier (t={NSTEPS*DT:.1f} t.u.)')
    # colorbar placeholder / phi map for case B
    ax = fig.add_subplot(gs[0, 2])
    phi_map = np.full((NX, NY), PHI_BARRIER)
    phi_map[case_b['inside']] = PHI0
    im = ax.imshow(phi_map.T, origin='lower', cmap='cividis',
                   vmin=PHI0, vmax=PHI_BARRIER, extent=[0, NX, 0, NY])
    ax.set_title('Case B: phi field')
    ax.set_xlabel('x (cells)')
    ax.set_ylabel('y (cells)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Far-probe traces
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(case_a['data']['t'], case_a['data']['far'], 'o-',
            color='tab:blue', ms=3, label='physical walls')
    ax.plot(case_b['data']['t'], case_b['data']['far'], 's-',
            color='tab:orange', ms=3, label='high-phi barrier')
    ax.axhline(U_THRESH, color='k', ls='--', lw=0.8)
    ax.set_xlabel('t (t.u.)')
    ax.set_ylabel('mean u at far probe')
    ax.set_title('Far-probe pulse traces')
    ax.legend()

    # Bar comparison
    ax = fig.add_subplot(gs[1, 1])
    labels = ['speed\n(cells/t.u.)', 'peak_far', 'max_outside_u']
    vals_a = [
        res_a['speed_cells_per_tu'] if res_a['speed_cells_per_tu'] else 0,
        res_a['peak_far'],
        res_a['max_outside_u'],
    ]
    vals_b = [
        res_b['speed_cells_per_tu'] if res_b['speed_cells_per_tu'] else 0,
        res_b['peak_far'],
        res_b['max_outside_u'],
    ]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, vals_a, width, label='physical walls', color='tab:blue')
    ax.bar(x + width/2, vals_b, width, label='high-phi barrier', color='tab:orange')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('value')
    ax.set_title('Comparison')
    ax.legend()

    # Case B all probes
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(case_b['data']['t'], case_b['data']['near'], label='near')
    ax.plot(case_b['data']['t'], case_b['data']['mid'], label='mid')
    ax.plot(case_b['data']['t'], case_b['data']['far'], label='far')
    ax.axhline(U_THRESH, color='k', ls='--', lw=0.8)
    ax.set_xlabel('t (t.u.)')
    ax.set_ylabel('mean u')
    ax.set_title('Case B: all probe traces')
    ax.legend()

    # Summary text
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')
    summary = (
        f"Grid: {NX}x{NY}, channel width {CHANNEL_WIDTH} cells\n"
        f"PHI0={PHI0}, PHI_BARRIER={PHI_BARRIER} (non-propagating threshold), "
        f"T_FLASH={T_FLASH} t.u., spot radius {SPOT_R}\n\n"
        f"Case A (physical walls):\n"
        f"  runtime {res_a['runtime_s']:.1f} s  |  "
        f"speed {res_a['speed_cells_per_tu']:.3f} cells/t.u.  |  "
        f"peak_far {res_a['peak_far']:.3f}  |  "
        f"max outside u {res_a['max_outside_u']:.3f}  |  "
        f"transmitted {res_a['transmitted']}\n\n"
        f"Case B (high-phi barrier):\n"
        f"  runtime {res_b['runtime_s']:.1f} s  |  "
        f"speed {res_b['speed_cells_per_tu']:.3f} cells/t.u.  |  "
        f"peak_far {res_b['peak_far']:.3f}  |  "
        f"max outside u {res_b['max_outside_u']:.3f}  |  "
        f"transmitted {res_b['transmitted']}\n"
        f"  final barrier mean u {res_b['mean_barrier_u_final']:.4f} "
        f"(barrier rest u*={U_STAR_BARRIER:.4f})"
    )
    ax.text(0.05, 0.5, summary, transform=ax.transAxes,
            verticalalignment='center', fontfamily='monospace', fontsize=10)

    fig.suptitle('Light-as-walls pilot: physical walls vs high-phi barriers',
                 fontsize=14)
    fig.tight_layout()
    figpath = 'figures/rd_light_walls_pilot.png'
    fig.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f'Saved {figpath}', flush=True)

    jsonpath = 'figures/rd_light_walls_pilot.json'
    with open(jsonpath, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'Saved {jsonpath}', flush=True)


if __name__ == '__main__':
    main()
