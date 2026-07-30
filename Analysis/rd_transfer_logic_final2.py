"""
TEST 3 (FINAL2, clamp-rest fix) -- Two-input collision gate: inhibition
(A AND NOT B) with windowed readout, BOTH kinetics.

Combines rd_transfer_logic.py (Barkley, OLD clipped solver -- numbers
superseded) and rd_transfer_logic_oregonator.py (post-de-hack but with
the (0,0)-clamp sink artifact) into one script on the fixed core.  Each
kinetics keeps its own validated geometry:

  barkley:    256x256, W=20, TJ=96, A port x[25,43), B port y[163,181),
              probe x=240, window +/-2.5 t.u.  (rd_transfer_logic.py)
  oregonator: 256x200, W=20, TJ=96, A port x[25,43), B port y[140,158),
              probe strip x=240, window +/-2.5 t.u.
              (rd_transfer_logic_oregonator.py, candidate A)

Clamp fix: Oregonator ports clamp to (u*, u*) between pulses
(clamp_rest), not (0, 0); Barkley keeps the (0, 0) default (= its rest).

The working primitive is the canonical INHIBITION gate A AND NOT B
(Steinbock/Kettunen/Showalter 1996; Toth & Showalter 1995): B's
left-going fragment collides head-on with A's wave inside A's arm and
annihilates it; the output is the probe peak inside a time window
centred on A's expected arrival.

Protocol (mirrors both predecessors):
  0. No-stimulus control (quiet); A-alone must give EXACTLY ONE probe
     crossing over the whole trace (no pacemaking).
  1. Naive whole-trace readout check (junction computes OR, not XOR).
  2. Windowed truth table: (0,0)->0, (1,0)->1, (0,1)->0, (1,1)->0.
  3. Inhibitor delay sweep dB -> coincidence window for inhibition.

Usage:
    ../.venv/bin/python rd_transfer_logic_final2.py [barkley|oregonator|all]

Outputs: Analysis/figures/rd_transfer_logic_final2_<kin>.{png,json},
         Analysis/figures/rd_transfer_logic_final2_snapshots_<kin>.png
"""

import json
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from rd_core import RDSubstrate

U_THRESH = 0.5
DURATION = 30
DT = 0.05
Q = 0.002
W = 20
W2 = W // 2
TJ = 96                  # junction x
X_PROBE = 240

OREG_A = dict(f=1.4375, eps=0.05014844822490394, phi=0.010)

KINETICS = {
    'barkley': dict(kw=dict(kinetics='barkley', eps=0.02, Dv=0.0),
                    phi=0.0, label='Barkley (a=0.75, b=0.01, eps=0.02, Dv=0)',
                    NX=256, NY=256, B_CH=(None, 200), portA=(25, 43),
                    portB=(163, 181), nsteps=1600, win_half=2.5,
                    snaps=(300, 500, 800, 1200),
                    sweep=(0, 20, 40, 60, 80, 100, 110, 120, 140, 160,
                           200, 260, 320)),
    'oregonator': dict(kw=dict(kinetics='oregonator', f=OREG_A['f'],
                               eps=OREG_A['eps']),
                       phi=OREG_A['phi'],
                       label='Oregonator A (f=1.4375, eps=0.0501, phi=0.010)',
                       NX=256, NY=200, B_CH=(None, 172), portA=(25, 43),
                       portB=(140, 158), nsteps=1000, win_half=2.5,
                       snaps=(150, 250, 400, 700),
                       sweep=(0, 20, 40, 60, 80, 100, 120, 160)),
}


def rest_u_star(f, phi):
    def F(u):
        return u - u**2 - (f * u + phi) * (u - Q) / (u + Q)
    ugrid = np.concatenate(([0.0], np.logspace(-7, np.log10(1.2), 4000)))
    vals = F(ugrid)
    for i in range(len(ugrid) - 1):
        if vals[i] > 0.0 and vals[i + 1] <= 0.0:
            return brentq(F, ugrid[i], ugrid[i + 1], xtol=1e-14, rtol=1e-12)
    raise RuntimeError('no rest state found')


def rest_state(kin):
    if kin == 'oregonator':
        us = rest_u_star(KINETICS[kin]['kw']['f'], KINETICS[kin]['phi'])
        return us, us
    return 0.0, 0.0


def make_rd(kin):
    spec = KINETICS[kin]
    NX, NY = spec['NX'], spec['NY']
    CY = NY // 2
    u_rest, v_rest = rest_state(kin)
    wall = np.ones((NX, NY), bool)
    wall[:, CY - W2:CY + W2] = False               # A channel -> output
    wall[TJ - W2:TJ + W2, CY:spec['B_CH'][1]] = False  # B channel from top
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, clamp_rest=(u_rest, v_rest),
                     **spec['kw'])
    if spec['phi']:
        rd.set_phi(spec['phi'])
    rd.set_walls(wall)
    rd.u[~wall] = u_rest
    rd.v[~wall] = v_rest
    pA = np.zeros((NX, NY), bool)
    pA[spec['portA'][0]:spec['portA'][1], CY - W2:CY + W2] = True
    pB = np.zeros((NX, NY), bool)
    pB[TJ - W2:TJ + W2, spec['portB'][0]:spec['portB'][1]] = True
    rd.add_port('A', pA)
    rd.add_port('B', pB)
    probe = np.zeros((NX, NY), bool)
    probe[X_PROBE, CY - 1:CY + 2] = True
    rd.add_probe('out', probe & ~wall)
    return rd, wall


def run_case(kin, fire_a, fire_b, dB=0, nsteps=None):
    spec = KINETICS[kin]
    rd, _ = make_rd(kin)
    if fire_a:
        rd.pulse_train('A', [0], duration=DURATION)
    if fire_b:
        rd.pulse_train('B', [dB], duration=DURATION)
    return rd.run(nsteps or spec['nsteps'])


def count_crossings(series, thresh=U_THRESH):
    above = series >= thresh
    return int(np.sum((~above[:-1]) & above[1:]))


def first_crossing(t, s, thresh=U_THRESH):
    idx = np.nonzero(s >= thresh)[0]
    return float(t[idx[0]]) if len(idx) else None


def window_peak(data, t0, win_half):
    m = (data['t'] >= t0 - win_half) & (data['t'] <= t0 + win_half)
    return float(data['out'][m].max())


def main(kin):
    spec = KINETICS[kin]
    win_half = spec['win_half']
    u_rest, _ = rest_state(kin)
    print(f'=== kinetics = {kin}: {spec["label"]} ===  rest u* = {u_rest:.6f}')

    # --- no-stimulus control --------------------------------------------------
    rd, wall = make_rd(kin)
    rd.run(1000)
    dev = float(np.abs(rd.u[~wall] - u_rest).max())
    leak = float(np.abs(rd.u[wall]).max())
    print(f'Control (no stimulus, 50 t.u.): max |u-u*| = {dev:.3e}, '
          f'wall leak = {leak:.1e} '
          f'-> {"QUIET" if dev < 0.01 and leak == 0.0 else "FAILED!"}')
    assert dev < 0.01 and leak == 0.0

    # --- calibrate the readout window on A's lone arrival ---------------------
    d10 = run_case(kin, True, False)
    tA = first_crossing(d10['t'], d10['out'])
    nA = count_crossings(d10['out'])
    print(f'A alone arrives at probe t = {tA:.2f} t.u. '
          f'({nA} crossing(s) over whole trace; window +/-{win_half} t.u.)')
    assert nA == 1, f'physics sanity failed: {nA} crossings for single pulse'

    # --- 0) naive readout: open junction computes OR, not XOR -----------------
    d01 = run_case(kin, False, True)
    d11 = run_case(kin, True, True, 0)
    naive = {'00': 0,
             '10': count_crossings(d10['out']),
             '01': count_crossings(d01['out']),
             '11': count_crossings(d11['out'])}
    print('Naive whole-trace readout (any crossing = 1): '
          + ', '.join(f'({k[0]},{k[1]})->{int(v > 0)}'
                      for k, v in naive.items())
          + '  -- OR, not XOR (annihilation not readable)')

    # --- truth table (windowed readout) ----------------------------------------
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

    # --- coincidence window for inhibition --------------------------------------
    print('Inhibitor timing sweep...')
    sweep = []
    for dB in spec['sweep']:
        pk = window_peak(run_case(kin, True, True, dB), tA, win_half)
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
        'kinetics': spec['label'] + ', de-hacked rd_core + clamp-rest fix',
        'rest_state_u_star': u_rest,
        'control_max_dev_50tu': dev,
        'control_quiet': bool(dev < 0.01),
        'single_pulse_crossings': nA,
        'geometry': (f"{spec['NX']}x{spec['NY']}, W={W}, TJ={TJ}, "
                     f"A port x{spec['portA']}, B port y{spec['portB']}, "
                     f"probe strip x={X_PROBE}"),
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

    # --- snapshot figure (1,0) vs (1,1) ------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for row, (fa, fb, tag) in enumerate([(True, False, 'A only'),
                                         (True, True, 'A and B (coincident)')]):
        rd, wall = make_rd(kin)
        rd.pulse_train('A', [0], duration=DURATION)
        if fb:
            rd.pulse_train('B', [0], duration=DURATION)
        for col, ts in enumerate(spec['snaps']):
            rd.run(ts - rd.t)
            ax = axes[row, col]
            u = np.ma.masked_where(wall, rd.u)
            ax.imshow(u.T, origin='lower', cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'{tag}, step {ts}', fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'TEST 3 FINAL2 ({kin}): inhibition gate dynamics',
                 fontsize=13)
    fig.tight_layout()
    snap_path = f'figures/rd_transfer_logic_final2_snapshots_{kin}.png'
    fig.savefig(snap_path, dpi=150)
    plt.close(fig)
    print(f'Saved {snap_path}')

    # --- main figure -------------------------------------------------------------
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

    plt.suptitle(f'TEST 3 FINAL2 ({kin}): two-input collision inhibition '
                 f'gate', fontsize=13)
    plt.tight_layout()
    figpath = f'figures/rd_transfer_logic_final2_{kin}.png'
    plt.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f'Saved {figpath}')
    jsonpath = f'figures/rd_transfer_logic_final2_{kin}.json'
    with open(jsonpath, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'Saved {jsonpath}')


if __name__ == '__main__':
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else 'all'
    assert arg in ('barkley', 'oregonator', 'all'), arg
    for k in (list(KINETICS) if arg == 'all' else [arg]):
        main(k)
