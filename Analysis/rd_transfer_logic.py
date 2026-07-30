"""
TEST 3 -- Two-input collision gate: inhibition (A AND NOT B) with
windowed readout (excitable Barkley BZ medium).

NOTE ON KINETICS: this test needs clean single pulses; the Oregonator
baseline is a relaxation oscillator (see TEST 2 header), so this test
uses the project's validated Barkley excitable model (a=0.75, b=0.01,
eps=0.02, Du=1, Dv=0).

WHY NOT THE NAIVE ANNIHILATION-XOR: the originally envisaged gate (two
input channels meeting head-on into a common output channel; lone pulses
transmit, coincident pair annihilates -> XOR) is NOT realisable in this
geometry class.  Empirically (T-junction, widened collision chamber,
constricted output gap, Y-junction, X-junction):
  * any open junction diffracts an incoming wave into ALL connected
    channels -- junctions compute OR, never XOR;
  * a head-on collision annihilates cleanly in a plain channel, but the
    annihilation cannot be READ OUT: any readout channel touching the
    collision zone is itself fired by the collision cusp (the contact
    line of two fronts meeting at an angle advances into fresh medium
    along the bisector of the front normals at c/sin(alpha) > c).
The working primitive (also the canonical one in the BZ literature,
e.g. Steinbock/Kettunen/Showalter 1996; Toth & Showalter 1995) is the
INHIBITION gate A AND NOT B:

Geometry (wall mask, W = 20 channels):
  - horizontal signal channel y = CY, port A at x in [25, 43],
  - vertical inhibitor channel x = TJ = 96, joining from the top
    (port B at y in [163, 181]); arm lengths A->T and B->T are matched
    (~60 / ~44 cells) so simultaneous firing intercepts,
  - output probe far right at x = 240 on the horizontal channel.

Protocol: A fired at step 0; B fired at step dB.  If B arrives at the
junction in time, its left-going fragment collides HEAD-ON with A's
wave inside A's arm and annihilates it (clean annihilation in a plain
channel, no readout channel at the collision point).  B's right-going
fragment reaches the probe EARLIER than A would; the gate output is the
probe peak inside a time window centred on A's expected arrival
(windowed readout, standard experimental practice).

Truth table: (0,0)->0, (1,0)->1, (0,1)->0, (1,1)->0  =  A AND (NOT B).
The delay sweep dB gives the coincidence window for inhibition.

Outputs: Analysis/figures/rd_transfer_logic.png,
         Analysis/figures/rd_transfer_logic_snapshots.png,
         Analysis/figures/rd_transfer_logic.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt

from rd_core import RDSubstrate

NX, NY = 256, 256
CY = NY // 2
TJ = 96                  # junction x
W = 20
W2 = W // 2
X_PROBE = 240
U_THRESH = 0.5
DURATION = 30
NSTEPS = 1600
WIN_HALF = 2.5           # readout window half-width (time units)
KIN = dict(kinetics='barkley', eps=0.02, Dv=0.0)


def make_rd():
    wall = np.ones((NX, NY), bool)
    wall[:, CY - W2:CY + W2] = False          # A channel -> output
    wall[TJ - W2:TJ + W2, CY:200] = False     # B channel joining from top
    rd = RDSubstrate(nx=NX, ny=NY, **KIN)
    rd.set_walls(wall)
    X, Y = np.mgrid[0:NX, 0:NY]
    pA = np.zeros((NX, NY), bool)
    pA[25:43, CY - W2:CY + W2] = True
    pB = np.zeros((NX, NY), bool)
    pB[TJ - W2:TJ + W2, 163:181] = True
    rd.add_port('A', pA)
    rd.add_port('B', pB)
    rd.add_probe('out', ((X - X_PROBE)**2 + (Y - CY)**2 <= 4) & ~wall)
    return rd, wall


def run_case(fire_a, fire_b, dB=0, nsteps=NSTEPS):
    rd, _ = make_rd()
    if fire_a:
        rd.pulse_train('A', [0], duration=DURATION)
    if fire_b:
        rd.pulse_train('B', [dB], duration=DURATION)
    return rd.run(nsteps)


def first_crossing(t, s, thresh=U_THRESH):
    idx = np.nonzero(s >= thresh)[0]
    return float(t[idx[0]]) if len(idx) else None


def window_peak(data, t0):
    m = (data['t'] >= t0 - WIN_HALF) & (data['t'] <= t0 + WIN_HALF)
    return float(data['out'][m].max())


def main():
    # --- calibrate the readout window on A's lone arrival ------------------
    d10 = run_case(True, False)
    tA = first_crossing(d10['t'], d10['out'])
    print(f'A alone arrives at probe t = {tA:.2f} t.u. '
          f'(window +/-{WIN_HALF} t.u.)')

    # --- truth table ---------------------------------------------------------
    truth = {'00': 0.0}
    truth['10'] = window_peak(d10, tA)
    truth['01'] = window_peak(run_case(False, True), tA)
    truth['11'] = window_peak(run_case(True, True, 0), tA)
    for k in ('10', '01', '11'):
        print(f'  ({k[0]},{k[1]}): window peak = {truth[k]:.3f}')

    true_peak = truth['10']
    false_peak = max(truth['00'], truth['01'], truth['11'])
    separation = true_peak / max(false_peak, 1e-3)
    thr = 0.5 * (true_peak + false_peak)
    print(f'Separation ratio (true / max false) = {separation:.1f} '
          f'(false peak = {false_peak:.3f})')

    # --- coincidence window for inhibition ------------------------------------
    print('Inhibitor timing sweep...')
    sweep = []
    for dB in [0, 20, 40, 60, 80, 100, 110, 120, 140, 160, 200, 260, 320]:
        pk = window_peak(run_case(True, True, dB), tA)
        blocked = pk < thr
        sweep.append({'dB_steps': dB, 'window_peak': pk, 'blocked': blocked})
        print(f"  dB={dB:4d} steps: window peak={pk:.3f} -> "
              f"{'A INHIBITED' if blocked else 'A TRANSMITS'}")
    blocked_ds = [s['dB_steps'] for s in sweep if s['blocked']]
    coincidence = max(blocked_ds) if blocked_ds else 0
    print(f'Inhibition (coincidence) window: dB < ~{coincidence} steps '
          f'({coincidence*0.05:.1f} t.u.)')

    results = {
        'gate': 'A AND (NOT B) -- inhibition via head-on collision',
        'kinetics': 'barkley (a=0.75, b=0.01, eps=0.02, Du=1, Dv=0)',
        'readout': f'windowed probe peak, window = {WIN_HALF} t.u. around '
                   f'A-alone arrival t={tA:.2f} t.u.',
        'truth_table': truth,
        'separation_ratio': separation,
        'decision_threshold': thr,
        'timing_sweep': sweep,
        'inhibition_window_steps': coincidence,
        'xor_note': ('naive annihilation-XOR is not realisable: junctions '
                     'diffract waves into all connected channels (OR), and '
                     'any readout at the collision zone is fired by the '
                     'collision cusp; see Notes/oregonator-oscillatory-'
                     'regime-2026-07-20.md and the script header'),
    }

    # --- snapshot figure (1,0) vs (1,1) ---------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for row, (fa, fb, tag) in enumerate([(True, False, 'A only'),
                                         (True, True, 'A and B (coincident)')]):
        rd, wall = make_rd()
        rd.pulse_train('A', [0], duration=DURATION)
        if fb:
            rd.pulse_train('B', [0], duration=DURATION)
        for col, ts in enumerate([300, 500, 800, 1200]):
            rd.run(ts - rd.t)
            ax = axes[row, col]
            u = np.ma.masked_where(wall, rd.u)
            ax.imshow(u.T, origin='lower', cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'{tag}, step {ts}', fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('TEST 3: inhibition gate dynamics', fontsize=13)
    fig.tight_layout()
    fig.savefig('figures/rd_transfer_logic_snapshots.png', dpi=150)
    plt.close(fig)
    print('Saved figures/rd_transfer_logic_snapshots.png')

    # --- main figure ------------------------------------------------------------
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

    plt.suptitle('TEST 3 (barkley): two-input collision inhibition gate',
                 fontsize=13)
    plt.tight_layout()
    figpath = 'figures/rd_transfer_logic.png'
    plt.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f'Saved {figpath}')
    jsonpath = 'figures/rd_transfer_logic.json'
    with open(jsonpath, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'Saved {jsonpath}')


if __name__ == '__main__':
    main()
