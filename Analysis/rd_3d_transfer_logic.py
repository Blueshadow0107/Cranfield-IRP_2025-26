"""
3D T-junction logic transfer test (dark-spot Oregonator).

Geometry: 48x48x48 domain.  A channel runs horizontally along x through the
centre; a B channel runs vertically along y and meets it at a T-junction.
Both channels have a 16x16 cross-section.

The four truth-table cases are run in parallel to keep wall time modest.
Outputs: JSON + PNG saved to Analysis/figures/rd_3d_transfer_logic.*.
"""

import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from rd_core_3d import RDSubstrate3D


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
F = 1.4375
EPS = 0.0501
Q = 0.002
DU = 1.0
DV = 0.0
PHI0 = 0.010
PHI_DARK = 0.002
T_FLASH = 3.0
DT = 0.05
DX = 1.0
NX, NY, NZ = 48, 48, 48
W = 16
W2 = W // 2
NSTEPS = 400
DURATION = int(T_FLASH / DT)
SPOT_R = 4
U_THRESH = 0.5
WIN_HALF = 2.5

CY = NY // 2            # 24
CZ = NZ // 2            # 24
TJ = NX // 2            # 24
X_PROBE = 40            # output probe x-coordinate
A_SOURCE_X = 6
B_SOURCE_Y = NY - 6


def rest_u_star(f, phi):
    def F_u(u):
        return u - u**2 - (f * u + phi) * (u - Q) / (u + Q)
    ugrid = np.concatenate(([0.0], np.logspace(-7, np.log10(1.2), 4000)))
    vals = F_u(ugrid)
    for i in range(len(ugrid) - 1):
        if vals[i] > 0.0 and vals[i + 1] <= 0.0:
            return brentq(F_u, ugrid[i], ugrid[i + 1], xtol=1e-14, rtol=1e-12)
    raise RuntimeError('no rest state found')


U_STAR = rest_u_star(F, PHI0)


def build_geometry():
    rd = RDSubstrate3D(nx=NX, ny=NY, nz=NZ, dx=DX, dt=DT,
                       f=F, eps=EPS, q=Q, Du=DU, Dv=DV, phi=PHI0)
    rd.set_rest(U_STAR, U_STAR)

    wall = np.ones((NX, NY, NZ), dtype=bool)
    # horizontal A channel
    wall[:, CY - W2:CY + W2, CZ - W2:CZ + W2] = False
    # vertical B channel joining from the top
    wall[TJ - W2:TJ + W2, :, CZ - W2:CZ + W2] = False
    rd.set_walls(wall)

    rd.u[~wall] = U_STAR
    rd.v[~wall] = U_STAR

    probe = np.zeros((NX, NY, NZ), dtype=bool)
    probe[X_PROBE, CY - W2:CY + W2, CZ - W2:CZ + W2] = True
    rd.add_probe('out', probe)
    return rd, wall


def make_spots(wall, fire_a, fire_b):
    ii, jj, kk = np.mgrid[0:NX, 0:NY, 0:NZ]
    spots = []
    if fire_a:
        ma = ((ii - A_SOURCE_X)**2 + (jj - CY)**2 + (kk - CZ)**2) <= SPOT_R**2
        ma = ma & ~wall
        spots.append({'mask': ma, 'times': [0],
                      'duration': DURATION, 'phi_dark': PHI_DARK})
    if fire_b:
        mb = ((ii - TJ)**2 + (jj - B_SOURCE_Y)**2 + (kk - CZ)**2) <= SPOT_R**2
        mb = mb & ~wall
        spots.append({'mask': mb, 'times': [0],
                      'duration': DURATION, 'phi_dark': PHI_DARK})
    return spots


def run_case(inputs):
    """Worker: run one truth-table case. `inputs` is (fire_a, fire_b)."""
    fire_a, fire_b = inputs
    rd, wall = build_geometry()
    spots = make_spots(wall, fire_a, fire_b)

    phi = np.full((NSTEPS, NX, NY, NZ), PHI0, dtype=float)
    for spot in spots:
        for t0 in spot['times']:
            phi[t0:t0 + DURATION, spot['mask']] = spot['phi_dark']

    series = []
    ts = []
    snapshots = {}
    snap_steps = [80, 160, 240, 320]
    for step in range(NSTEPS):
        rd.set_phi(phi[step])
        rd._step()
        rd.t += 1
        ts.append(rd.t * rd.dt)
        series.append(float(rd.u[rd.probes['out']].mean()))
        if step + 1 in snap_steps:
            snapshots[step + 1] = rd.u[:, :, CZ].copy()

    return {
        'fire_a': fire_a,
        'fire_b': fire_b,
        't': np.array(ts, dtype=float),
        'out': np.array(series, dtype=float),
        'snapshots': snapshots,
        'wall': wall,
        'max_u': float(rd.u.max()),
    }


def first_crossing(t, s, thresh=U_THRESH):
    idx = np.nonzero(s >= thresh)[0]
    return float(t[idx[0]]) if len(idx) else None


def window_peak(data, t0, win_half):
    m = (data['t'] >= t0 - win_half) & (data['t'] <= t0 + win_half)
    return float(data['out'][m].max()) if m.any() else 0.0


def main():
    print(f'3D T-junction logic: grid {NX}x{NY}x{NZ}, channel cross-section {W}x{W}')
    print(f'Parameters: f={F}, eps={EPS}, q={Q}, Du={DU}, Dv={DV}, phi0={PHI0}')
    print(f'Dark spot: phi={PHI_DARK}, T_flash={T_FLASH} t.u., radius={SPOT_R}')
    print(f'Rest state u* = {U_STAR:.6f}')
    print(f'Diffusion number: {DU * DT / DX**2:.4f} (3D limit 1/6={1/6:.4f})')
    print(f'Running 4 truth-table cases in parallel ({NSTEPS} steps each)...')

    t_start = time.perf_counter()
    cases = [(False, False), (True, False), (False, True), (True, True)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        results_list = list(pool.map(run_case, cases))
    elapsed = time.perf_counter() - t_start
    print(f'Total runtime: {elapsed:.1f} s')

    data = {f'{int(fa)}{int(fb)}': r for r in results_list
            for fa, fb in [(r['fire_a'], r['fire_b'])]}

    # control / quiet check
    quiet = data['00']['max_u']
    print(f'Control (00): max u = {quiet:.4f} -> '
          f'{"QUIET" if quiet < 0.1 else "SPONTANEOUS FIRING"}')

    tA = first_crossing(data['10']['t'], data['10']['out'])
    nA = int(np.sum((data['10']['out'][:-1] < U_THRESH) & (data['10']['out'][1:] >= U_THRESH)))
    print(f'A alone: arrival at probe t={tA:.2f} t.u., {nA} upward crossing(s)')

    truth = {k: window_peak(data[k], tA, WIN_HALF) if tA else 0.0
             for k in ('00', '10', '01', '11')}
    print('\nWindowed truth table (window +/-%.1f t.u. around tA=%.2f):' % (WIN_HALF, tA or 0))
    for k, v in truth.items():
        print(f'  ({k[0]},{k[1]}): peak={v:.3f}')

    true_peak = truth['10']
    false_peak = max(truth['00'], truth['01'], truth['11'])
    separation = true_peak / max(false_peak, 1e-3)
    threshold = 0.5 * (true_peak + false_peak)
    logic_ok = (truth['10'] > threshold and truth['00'] < threshold and
                truth['01'] < threshold and truth['11'] < threshold)
    print(f'Separation ratio = {separation:.1f}x, threshold={threshold:.3f}')
    print(f'Logic A AND (NOT B) recognised: {logic_ok}')

    # snapshots figure: A only vs A+B
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for row, key in enumerate(('10', '11')):
        r = data[key]
        wall = r['wall']
        for col, step in enumerate(sorted(r['snapshots'])):
            u = np.ma.masked_where(wall[:, :, CZ], r['snapshots'][step])
            ax = axes[row, col]
            ax.imshow(u.T, origin='lower', cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'{key}, t={step*DT:.1f}', fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel('A only')
    axes[1, 0].set_ylabel('A + B')
    fig.suptitle('3D T-junction inhibition dynamics (z mid-plane)', fontsize=13)
    fig.tight_layout()

    figdir = Path('figures')
    figdir.mkdir(exist_ok=True)
    snap_path = figdir / 'rd_3d_transfer_logic_snapshots.png'
    fig.savefig(snap_path, dpi=150)
    plt.close(fig)
    print(f'Saved {snap_path}')

    # main figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ['00', '10', '01', '11']
    vals = [truth[k] for k in labels]
    colors = ['gray', 'tab:blue', 'tab:red', 'tab:red']
    axes[0].bar(labels, vals, color=colors)
    axes[0].axhline(threshold, color='k', ls='--', lw=0.8,
                    label=f'threshold {threshold:.2f}')
    axes[0].set_xlabel('inputs (A,B)')
    axes[0].set_ylabel('windowed peak u at output')
    axes[0].set_title(f'A AND (NOT B) (separation {separation:.1f}x)')
    axes[0].legend()

    for key, color in [('10', 'tab:blue'), ('01', 'tab:red'), ('11', 'tab:purple')]:
        axes[1].plot(data[key]['t'], data[key]['out'],
                     label=f'({key[0]},{key[1]})', color=color)
    if tA:
        axes[1].axvline(tA, color='k', ls='--', lw=0.8)
    axes[1].axhline(threshold, color='k', ls='--', lw=0.8)
    axes[1].set_xlabel('t (t.u.)')
    axes[1].set_ylabel('mean u at output probe')
    axes[1].set_title('output-probe traces')
    axes[1].legend()
    fig.tight_layout()
    figpath = figdir / 'rd_3d_transfer_logic.png'
    fig.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f'Saved {figpath}')

    results = {
        'test': '3D T-junction A AND (NOT B) inhibition (dark-spot Oregonator)',
        'grid': [NX, NY, NZ],
        'channel_width': W,
        'dx': DX,
        'dt': DT,
        'nsteps': NSTEPS,
        'physical_time_tu': NSTEPS * DT,
        'runtime_s': elapsed,
        'parameters': {'f': F, 'eps': EPS, 'q': Q, 'Du': DU, 'Dv': DV,
                       'phi0': PHI0, 'phi_dark': PHI_DARK,
                       't_flash_tu': T_FLASH, 'spot_radius': SPOT_R},
        'rest_state_u_star': U_STAR,
        'diffusion_number': DU * DT / DX**2,
        'control': {'max_u': quiet, 'quiet': bool(quiet < 0.1)},
        'a_alone_arrival_tu': tA,
        'readout_window_half_tu': WIN_HALF,
        'truth_table_window_peak': truth,
        'separation_ratio': separation,
        'decision_threshold': threshold,
        'logic_recognised': logic_ok,
    }
    jsonpath = figdir / 'rd_3d_transfer_logic.json'
    with open(jsonpath, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'Saved {jsonpath}')


if __name__ == '__main__':
    main()
