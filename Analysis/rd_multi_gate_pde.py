"""
rd_multi_gate_pde.py -- PDE validation of the shared-control selective gate.

Geometry (2D, dark-spot Oregonator):
- Horizontal control channel A at y=YA, from x=5 to x=85.
- Vertical data channel B1 at x=XB1, from y=20 to y=125.
- Vertical data channel B2 at x=XB2, from y=20 to y=155.
- A input port at (XA_IN, YA); B1 at (XB1, YB1_IN); B2 at (XB2, YB2_IN).
- Output probes O1/O2 at the bottom of B1/B2 channels.

Function:
- A quiet: B1/B2 pulses travel to O1/O2.
- A fires: its pulse reaches the B1/B2 junctions first and makes them
  refractory, blocking B1/B2.

Runs all 8 input patterns in parallel.
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

from rd_core import RDSubstrate
from rd_darkspot_driver_multi import run_darkspot_multi


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
F = 1.4375
EPS = 0.05014844822490394  # match calibrated Oregonator A
Q = 0.002
DU = 1.0
DV = 0.0
PHI0 = 0.010
PHI_DARK = 0.002
T_FLASH = 3.0
DT = 0.05
DX = 1.0
NSTEPS = 800
DURATION = int(T_FLASH / DT)
U_THRESH = 0.5
WIN_HALF = 1.5

NX, NY = 100, 180
YA = 90
WA = 16               # A (control) channel width
WA2 = WA // 2
WB = 30               # B (data) channel width -- wider to favour straight-through
WB2 = WB // 2
# Shared control A enters near J1 so it reaches J1 first, then J2.
# Junctions are spaced far enough apart that a data pulse entering the A
# channel at J1 does not reach J2 before the local B2 pulse arrives.
XA_IN = 25
XB1 = 35
XB2 = 75
# Ports placed at the top of each vertical channel (channel entrance) so each
# dark spot emits a single downward pulse.  Distances chosen so A reaches
# each junction slightly before the corresponding B pulse.
YB1_IN = 115          # B1 channel top entrance, 25 cells above J1
YB2_IN = 150          # B2 channel top entrance, 60 cells above J2
SPOT_R = 6            # calibrated dark-spot radius


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
    rd = RDSubstrate(nx=NX, ny=NY, dx=DX, dt=DT,
                     f=F, eps=EPS, q=Q, Du=DU, Dv=DV,
                     clamp_rest=(U_STAR, U_STAR))
    rd.set_phi(PHI0)

    wall = np.ones((NX, NY), dtype=bool)
    # horizontal A channel (control line); starts left of J1 and continues
    # past J2 so the control pulse can reach both junctions and exit.
    wall[20:90, YA - WA2:YA + WA2] = False
    # vertical B1 channel, port at the top entrance
    wall[XB1 - WB2:XB1 + WB2, 20:YB1_IN + 5] = False
    # vertical B2 channel, port at the top entrance
    wall[XB2 - WB2:XB2 + WB2, 20:YB2_IN + 5] = False
    rd.set_walls(wall)

    rd.u[~wall] = U_STAR
    rd.v[~wall] = U_STAR

    # circular dark-spot ports, clipped to the channel interior
    ii, jj = np.mgrid[0:NX, 0:NY]
    a_port = ((ii - XA_IN)**2 + (jj - YA)**2) <= SPOT_R**2
    b1_port = ((ii - XB1)**2 + (jj - YB1_IN)**2) <= SPOT_R**2
    b2_port = ((ii - XB2)**2 + (jj - YB2_IN)**2) <= SPOT_R**2
    rd.add_port('A', a_port & ~wall)
    rd.add_port('B1', b1_port & ~wall)
    rd.add_port('B2', b2_port & ~wall)

    # thin horizontal strip probes just below each junction; a thin strip
    # matches the wave-front shape and gives a high mean-u signal.
    o1 = np.zeros((NX, NY), bool)
    o1[XB1 - WB2:XB1 + WB2, 49:52] = True
    o2 = np.zeros((NX, NY), bool)
    o2[XB2 - WB2:XB2 + WB2, 49:52] = True
    rd.add_probe('O1', o1 & ~wall)
    rd.add_probe('O2', o2 & ~wall)

    return rd, wall


def run_case(inputs):
    """Worker: run one (fire_a, fire_b1, fire_b2) case."""
    fire_a, fire_b1, fire_b2 = inputs
    rd, wall = build_geometry()

    spots = []
    if fire_a:
        spots.append(dict(mask=rd.ports['A'], times=[0],
                          duration=DURATION, phi_dark=PHI_DARK))
    if fire_b1:
        spots.append(dict(mask=rd.ports['B1'], times=[0],
                          duration=DURATION, phi_dark=PHI_DARK))
    if fire_b2:
        spots.append(dict(mask=rd.ports['B2'], times=[0],
                          duration=DURATION, phi_dark=PHI_DARK))

    probes = {'O1': rd.probes['O1'], 'O2': rd.probes['O2']}
    data = run_darkspot_multi(rd, PHI0, spots, NSTEPS, probes)

    # collect a few snapshots for visualisation
    snapshots = {}
    snap_steps = [100, 200, 300, 400]
    rd2, _ = build_geometry()
    schedule = np.full((max(snap_steps), NX, NY), PHI0, dtype=float)
    for spot in spots:
        for t0 in spot['times']:
            schedule[t0:t0 + spot['duration'], spot['mask']] = spot['phi_dark']
    for step in range(max(snap_steps)):
        rd2.set_phi(schedule[step])
        rd2._step()
        rd2.t += 1
        if rd2.t in snap_steps:
            snapshots[rd2.t] = rd2.u.copy()

    return {
        'fire_a': fire_a, 'fire_b1': fire_b1, 'fire_b2': fire_b2,
        't': data['t'],
        'O1': data['O1'],
        'O2': data['O2'],
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
    print(f'Shared-control gate PDE: grid {NX}x{NY}, A width {WA}, B width {WB}')
    print(f'Parameters: f={F}, eps={EPS}, q={Q}, Du={DU}, Dv={DV}, phi0={PHI0}')
    print(f'Dark spot: phi={PHI_DARK}, T_flash={T_FLASH} t.u.')
    print(f'Rest state u* = {U_STAR:.6f}')
    print(f'Running 8 input patterns in parallel ({NSTEPS} steps each)...')

    t_start = time.perf_counter()
    cases = [(a, b1, b2)
             for a in (0, 1)
             for b1 in (0, 1)
             for b2 in (0, 1)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        results_list = list(pool.map(run_case, cases))
    elapsed = time.perf_counter() - t_start
    print(f'Total runtime: {elapsed:.1f} s')

    data = {}
    for r in results_list:
        label = f'{r["fire_a"]}{r["fire_b1"]}{r["fire_b2"]}'
        data[label] = r

    # quiet check
    quiet = data['000']['max_u']
    print(f'Control (000): max u = {quiet:.4f} -> '
          f'{"QUIET" if quiet < 0.1 else "SPONTANEOUS FIRING"}')

    # reference arrival for A-only/B-only cases
    tA1 = first_crossing(data['100']['t'], data['100']['O1'])
    tA2 = first_crossing(data['100']['t'], data['100']['O2'])
    tB1 = first_crossing(data['010']['t'], data['010']['O1'])
    tB2 = first_crossing(data['001']['t'], data['001']['O2'])
    print(f'A-only: O1 arrival={tA1}, O2 arrival={tA2}')
    print(f'B1-only: O1 arrival={tB1}')
    print(f'B2-only: O2 arrival={tB2}')

    # windowed peaks around the expected data arrivals
    truth = {}
    for label, r in data.items():
        # use B1 arrival window for O1 and B2 arrival window for O2
        p1 = window_peak({'t': r['t'], 'out': r['O1']}, tB1, WIN_HALF) if tB1 else 0.0
        p2 = window_peak({'t': r['t'], 'out': r['O2']}, tB2, WIN_HALF) if tB2 else 0.0
        truth[label] = {'O1_peak': p1, 'O2_peak': p2}

    print('\nWindowed truth table:')
    for label, v in truth.items():
        print(f'  {label}: O1={v["O1_peak"]:.3f}, O2={v["O2_peak"]:.3f}')

    # decision logic: O1 fires when B1=1 and A=0; O2 fires when B2=1 and A=0
    o1_true_keys = ('010', '011')
    o1_false_keys = [k for k in truth if k not in o1_true_keys]
    o1_true_peak = max(truth[k]['O1_peak'] for k in o1_true_keys)
    o1_false_peak = max(truth[k]['O1_peak'] for k in o1_false_keys)
    o1_sep = o1_true_peak / max(o1_false_peak, 1e-3)
    o1_thr = 0.5 * (o1_true_peak + o1_false_peak)
    o1_ok = (all(truth[k]['O1_peak'] > o1_thr for k in o1_true_keys) and
             all(truth[k]['O1_peak'] < o1_thr for k in o1_false_keys))

    o2_true_keys = ('001', '011')
    o2_false_keys = [k for k in truth if k not in o2_true_keys]
    o2_true_peak = max(truth[k]['O2_peak'] for k in o2_true_keys)
    o2_false_peak = max(truth[k]['O2_peak'] for k in o2_false_keys)
    o2_sep = o2_true_peak / max(o2_false_peak, 1e-3)
    o2_thr = 0.5 * (o2_true_peak + o2_false_peak)
    o2_ok = (all(truth[k]['O2_peak'] > o2_thr for k in o2_true_keys) and
             all(truth[k]['O2_peak'] < o2_thr for k in o2_false_keys))

    print(f'\nO1 separation={o1_sep:.1f}x, recognised={o1_ok}')
    print(f'O2 separation={o2_sep:.1f}x, recognised={o2_ok}')

    # save snapshot figure for key cases
    figdir = Path('figures')
    figdir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for row, (a_key, b_key, title) in enumerate([
        ('010', '110', 'B1-only vs A+B1 (O1 should fire only without A)'),
        ('001', '101', 'B2-only vs A+B2 (O2 should fire only without A)'),
    ]):
        for col, key in enumerate([a_key, b_key]):
            r = data[key]
            u = np.ma.masked_where(r['wall'], r['snapshots'][200])
            ax = axes[row, col]
            ax.imshow(u.T, origin='lower', cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'{key}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        for col in range(2, 4):
            axes[row, col].axis('off')
    fig.suptitle('Shared-control gate PDE snapshots (t=10 t.u.)', fontsize=12)
    fig.tight_layout()
    snap_path = figdir / 'rd_multi_gate_pde_snapshots.png'
    fig.savefig(snap_path, dpi=150)
    plt.close(fig)
    print(f'Saved {snap_path}')

    # time-series figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {'010': 'tab:blue', '110': 'tab:red',
              '001': 'tab:green', '101': 'tab:purple'}
    for key in ['010', '110']:
        axes[0].plot(data[key]['t'], data[key]['O1'],
                     label=key, color=colors[key])
    if tB1:
        axes[0].axvline(tB1, color='k', ls='--', lw=0.8)
    axes[0].axhline(o1_thr, color='k', ls=':', lw=0.8)
    axes[0].set_xlabel('t (t.u.)')
    axes[0].set_ylabel('mean u at O1')
    axes[0].set_title('O1: B1-only vs A+B1')
    axes[0].legend()

    for key in ['001', '101']:
        axes[1].plot(data[key]['t'], data[key]['O2'],
                     label=key, color=colors[key])
    if tB2:
        axes[1].axvline(tB2, color='k', ls='--', lw=0.8)
    axes[1].axhline(o2_thr, color='k', ls=':', lw=0.8)
    axes[1].set_xlabel('t (t.u.)')
    axes[1].set_ylabel('mean u at O2')
    axes[1].set_title('O2: B2-only vs A+B2')
    axes[1].legend()
    fig.tight_layout()
    ts_path = figdir / 'rd_multi_gate_pde_series.png'
    fig.savefig(ts_path, dpi=150)
    plt.close(fig)
    print(f'Saved {ts_path}')

    results = {
        'test': 'shared-control selective gate (PDE)',
        'grid': [NX, NY],
        'channel_width_A': WA,
        'channel_width_B': WB,
        'geometry': {
            'YA': YA, 'XA_IN': XA_IN, 'XB1': XB1, 'XB2': XB2,
            'YB1_IN': YB1_IN, 'YB2_IN': YB2_IN,
        },
        'parameters': {'f': F, 'eps': EPS, 'q': Q, 'Du': DU, 'Dv': DV,
                       'phi0': PHI0, 'phi_dark': PHI_DARK,
                       't_flash_tu': T_FLASH},
        'runtime_s': elapsed,
        'control_max_u': quiet,
        'arrivals_tu': {
            'A_only_O1': tA1, 'A_only_O2': tA2,
            'B1_only_O1': tB1, 'B2_only_O2': tB2,
        },
        'truth_table_window_peak': truth,
        'O1': {'separation': o1_sep, 'threshold': o1_thr, 'recognised': o1_ok},
        'O2': {'separation': o2_sep, 'threshold': o2_thr, 'recognised': o2_ok},
    }
    jsonpath = figdir / 'rd_multi_gate_pde.json'
    with open(jsonpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved {jsonpath}')


if __name__ == '__main__':
    main()
