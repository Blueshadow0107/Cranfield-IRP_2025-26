"""
TEST 3 (dark-spot Oregonator) -- Two-input collision gate A AND (NOT B)
with windowed readout and a one-shot dark-spot input drive.

Mirrors rd_transfer_logic_oregonator.py but replaces the clamped ports
with circular dark spots at the channel entrances.  Each flash is
phi = 0.002 for 3.0 t.u. (60 steps) and emits one pulse.

Outputs: Analysis/figures/rd_transfer_logic_darkspot.{png,json},
         Analysis/figures/rd_transfer_logic_darkspot_snapshots.png
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from rd_core import RDSubstrate
from rd_darkspot_driver_multi import run_darkspot_multi

NX, NY = 256, 200
CY = NY // 2               # 100
TJ = 96                    # junction x
W = 20
W2 = W // 2
B_TOP = 172                # B channel top
X_PROBE = 240
U_THRESH = 0.5
DT = 0.05
Q = 0.002
DURATION = 60              # 3.0 t.u. flash

CAND = dict(f=1.4375, eps=0.05014844822490394, phi=0.010,
            label='A (f=1.4375, eps=0.0501, phi=0.010)',
            portA=(25, 43), portB=(140, 158), win_half=2.5)

PHI0 = 0.010
PHI_DARK = 0.002
SPOT_R = 6


def rest_u_star(f, phi):
    def F(u):
        return u - u**2 - (f * u + phi) * (u - Q) / (u + Q)
    ugrid = np.concatenate(([0.0], np.logspace(-7, np.log10(1.2), 4000)))
    vals = F(ugrid)
    for i in range(len(ugrid) - 1):
        if vals[i] > 0.0 and vals[i + 1] <= 0.0:
            return brentq(F, ugrid[i], ugrid[i + 1], xtol=1e-14, rtol=1e-12)
    raise RuntimeError('no rest state found')


def circle_mask(cx, cy, r):
    X, Y = np.mgrid[0:NX, 0:NY]
    return (X - cx)**2 + (Y - cy)**2 <= r**2


def make_rd(u_star):
    wall = np.ones((NX, NY), bool)
    wall[:, CY - W2:CY + W2] = False          # A channel -> output
    wall[TJ - W2:TJ + W2, CY:B_TOP] = False   # B channel joining from top
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, kinetics='oregonator',
                     eps=CAND['eps'], f=CAND['f'])
    rd.set_phi(PHI0)
    rd.set_walls(wall)
    rd.u[~wall] = u_star
    rd.v[~wall] = u_star
    probe = np.zeros((NX, NY), bool)
    probe[X_PROBE, CY - 1:CY + 2] = True
    rd.add_probe('out', probe)
    return rd, wall


def run_case(u_star, fire_a, fire_b, dB=0, nsteps=1000):
    rd, wall = make_rd(u_star)
    spots = []
    if fire_a:
        cxa = (CAND['portA'][0] + CAND['portA'][1]) // 2
        spot_a = circle_mask(cxa, CY, SPOT_R) & ~wall
        spots.append(dict(mask=spot_a, times=[0],
                          duration=DURATION, phi_dark=PHI_DARK))
    if fire_b:
        cyb = (CAND['portB'][0] + CAND['portB'][1]) // 2
        spot_b = circle_mask(TJ, cyb, SPOT_R) & ~wall
        spots.append(dict(mask=spot_b, times=[dB],
                          duration=DURATION, phi_dark=PHI_DARK))
    probe = np.zeros((NX, NY), bool)
    probe[X_PROBE, CY - 1:CY + 2] = True
    return run_darkspot_multi(rd, PHI0, spots, nsteps, {'out': probe})


def count_crossings(series, thresh=U_THRESH):
    above = series >= thresh
    return int(np.sum((~above[:-1]) & above[1:]))


def first_crossing(t, s, thresh=U_THRESH):
    idx = np.nonzero(s >= thresh)[0]
    return float(t[idx[0]]) if len(idx) else None


def window_peak(data, t0, win_half):
    m = (data['t'] >= t0 - win_half) & (data['t'] <= t0 + win_half)
    return float(data['out'][m].max())


def main():
    win_half = CAND['win_half']
    u_star = rest_u_star(CAND['f'], CAND['phi'])
    print(f'Candidate {CAND["label"]}: rest state u* = {u_star:.6f}')

    # --- no-stimulus control -------------------------------------------------
    rd, wall = make_rd(u_star)
    nsteps_ctrl = 1000
    ctrl_probe = np.zeros((NX, NY), bool)
    ctrl_probe[X_PROBE, CY - 1:CY + 2] = True
    run_darkspot_multi(rd, PHI0, [], nsteps_ctrl, {'out': ctrl_probe})
    quiet = float(rd.u.max())
    print(f'Control (no stimulus, 50 t.u.): max u = {quiet:.6f} '
          f'-> {"QUIET" if quiet < 0.1 else "SPONTANEOUS FIRING!"}')

    # --- calibrate the readout window on A's lone arrival --------------------
    d10 = run_case(u_star, True, False)
    tA = first_crossing(d10['t'], d10['out'])
    nA = count_crossings(d10['out'])
    print(f'A alone arrives at probe t = {tA:.2f} t.u. '
          f'({nA} crossing(s) over whole trace; window +/-{win_half} t.u.)')
    assert nA == 1, f'physics sanity failed: {nA} crossings for single pulse'

    # --- 0) naive readout: open junction computes OR, not XOR ----------------
    d01 = run_case(u_star, False, True)
    d11 = run_case(u_star, True, True, 0)
    naive = {'00': 0,
             '10': count_crossings(d10['out']),
             '01': count_crossings(d01['out']),
             '11': count_crossings(d11['out'])}
    print('Naive whole-trace readout (any crossing = 1): '
          + ', '.join(f'({k[0]},{k[1]})->{int(v > 0)}'
                      for k, v in naive.items())
          + '  -- OR, not XOR (annihilation not readable)')

    # --- truth table (windowed readout) --------------------------------------
    truth = {'00': 0.0}
    truth['10'] = window_peak(d10, tA, win_half)
    truth['01'] = window_peak(d01, tA, win_half)
    truth['11'] = window_peak(d11, tA, win_half)
    for k in ('10', '01', '11'):
        print(f'  ({k[0]},{k[1]}): window peak = {truth[k]:.3f}')

    true_peak = truth['10']
    false_peak = max(truth['00'], truth['01'], truth['11'])
    separation = true_peak / max(false_peak, 1e-3)
    thr = 0.5 * (true_peak + false_peak)
    print(f'Separation ratio (true / max false) = {separation:.1f} '
          f'(false peak = {false_peak:.3f})')

    # --- coincidence window for inhibition -----------------------------------
    print('Inhibitor timing sweep...')
    sweep = []
    for dB in [0, 20, 40, 60, 80, 100, 120, 160]:
        pk = window_peak(run_case(u_star, True, True, dB), tA, win_half)
        blocked = pk < thr
        sweep.append({'dB_steps': dB, 'window_peak': pk, 'blocked': blocked})
        print(f"  dB={dB:4d} steps: window peak={pk:.3f} -> "
              f"{'A INHIBITED' if blocked else 'A TRANSMITS'}", flush=True)
    blocked_ds = [s['dB_steps'] for s in sweep if s['blocked']]
    coincidence = max(blocked_ds) if blocked_ds else 0
    print(f'Inhibition (coincidence) window: dB < ~{coincidence} steps '
          f'({coincidence*DT:.1f} t.u.)')

    results = {
        'gate': 'A AND (NOT B) -- inhibition via head-on collision',
        'kinetics': f'oregonator candidate {CAND["label"]}, q={Q}, '
                    f'Du=1, Dv=0, dark-spot drive',
        'phi0': PHI0, 'phi_dark': PHI_DARK,
        't_flash_tu': DURATION * DT, 'spot_radius': SPOT_R,
        'rest_state_u_star': u_star,
        'control_max_u_50tu': quiet,
        'control_quiet': bool(quiet < 0.1),
        'geometry': (f'{NX}x{NY}, W={W}, TJ={TJ}, '
                     f'A port x{CAND["portA"]}, B port y{CAND["portB"]}, '
                     f'probe strip x={X_PROBE}'),
        'readout': f'windowed probe peak, window = {win_half} t.u. around '
                   f'A-alone arrival t={tA:.2f} t.u.',
        'naive_readout_crossings': naive,
        'naive_readout_note': ('whole-trace threshold readout gives OR, not '
                               'XOR: B diffracts into the output channel and '
                               'reaches the probe even when A is annihilated'),
        'truth_table': truth,
        'separation_ratio': separation,
        'decision_threshold': thr,
        'timing_sweep': sweep,
        'inhibition_window_steps': coincidence,
        'inhibition_window_time': coincidence * DT,
    }

    # --- snapshot figure (1,0) vs (1,1) --------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for row, (fa, fb, tag) in enumerate([(True, False, 'A only'),
                                         (True, True, 'A and B (coincident)')]):
        rd, wall = make_rd(u_star)
        spots = []
        if fa:
            cxa = (CAND['portA'][0] + CAND['portA'][1]) // 2
            spot_a = circle_mask(cxa, CY, SPOT_R) & ~wall
            spots.append(dict(mask=spot_a, times=[0],
                              duration=DURATION, phi_dark=PHI_DARK))
        if fb:
            cyb = (CAND['portB'][0] + CAND['portB'][1]) // 2
            spot_b = circle_mask(TJ, cyb, SPOT_R) & ~wall
            spots.append(dict(mask=spot_b, times=[0],
                              duration=DURATION, phi_dark=PHI_DARK))
        # burn-in snapshots by running explicitly
        nsteps_snap = max([150, 250, 400, 700])
        schedule = np.full((nsteps_snap, NX, NY), PHI0, dtype=float)
        for spot in spots:
            for t0 in spot['times']:
                schedule[t0:t0 + DURATION, spot['mask']] = PHI_DARK
        for col, ts in enumerate([150, 250, 400, 700]):
            for step in range(ts - rd.t):
                rd.set_phi(schedule[step])
                rd._step()
                rd.t += 1
            ax = axes[row, col]
            u = np.ma.masked_where(wall, rd.u)
            ax.imshow(u.T, origin='lower', cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'{tag}, step {ts}', fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('TEST 3 (dark-spot Oregonator A): inhibition gate dynamics',
                 fontsize=13)
    fig.tight_layout()
    snap_path = 'figures/rd_transfer_logic_darkspot_snapshots.png'
    fig.savefig(snap_path, dpi=150)
    plt.close(fig)
    print(f'Saved {snap_path}')

    # --- main figure ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    labels = ['00', '10', '01', '11']
    vals = [truth[k] for k in labels]
    colors = ['gray', 'tab:blue', 'tab:red', 'tab:red']
    ax.bar(labels, vals, color=colors)
    ax.axhline(thr, color='k', ls='--', lw=0.8,
               label=f'decision thr {thr:.2f}')
    ax.set_xlabel('inputs (A,B)')
    ax.set_ylabel('windowed peak u at output')
    ax.set_title(f'A AND (NOT B) gate (separation {separation:.0f}x)')
    ax.legend()

    ax = axes[1]
    ax.plot([s['dB_steps'] for s in sweep],
            [s['window_peak'] for s in sweep], 'o-')
    ax.axhline(thr, color='k', ls='--', lw=0.8)
    ax.set_xlabel('B firing delay dB (steps)')
    ax.set_ylabel('windowed peak u at output')
    ax.set_title(f'Inhibition window: dB < ~{coincidence} steps')

    plt.suptitle('TEST 3 (dark-spot Oregonator A): two-input collision '
                 'inhibition gate', fontsize=13)
    plt.tight_layout()
    figpath = 'figures/rd_transfer_logic_darkspot.png'
    plt.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f'Saved {figpath}')
    jsonpath = 'figures/rd_transfer_logic_darkspot.json'
    with open(jsonpath, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'Saved {jsonpath}')


if __name__ == '__main__':
    main()
