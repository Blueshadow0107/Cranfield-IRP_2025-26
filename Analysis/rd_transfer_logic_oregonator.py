"""
TEST 3 (Oregonator) -- Two-input collision gate: inhibition (A AND NOT B)
with windowed readout, excitable light-held Oregonator BZ medium.

Adapted copy of rd_transfer_logic.py (Barkley version) -- the Barkley
script is untouched.  Kinetics: validated EXCITABLE Oregonator regime
(Notes/oregonator-excitable-regime-hunt-2026-07-20.md):
    Candidate A (primary):   f=1.4375, eps=0.0501, phi=0.010, q=0.002
                             speed 6.21 cells/t.u.
    Candidate B (optional):  f=1.4375, eps=0.0199, phi=0.010
                             speed 11.70 cells/t.u.

CRITICAL DIFFERENCE vs Barkley: with phi>0 the state (0,0) is NOT rest;
all fields are initialised AT the rest state (u*, v*=u*) as in
oregonator_regime_hunt.py, and a no-stimulus control verifies quiet.

CHANNEL WIDTH: W=20 as in the Barkley version.  The Oregonator wavelength
(speed x refractory, measured in TEST 2) is printed in the results; the
regime hunt validated straight-channel propagation down to W=12, so W=20
has margin.  Geometry mirrors the Barkley script (same TJ, same ports,
same arm lengths) with NY reduced 256 -> 200 to cut compute; the B
channel and port are shifted up accordingly:
  - horizontal signal channel y = CY = 100, port A at x in [25, 43)
    (A arm to junction centre: 53 cells),
  - vertical inhibitor channel x = TJ = 96, joining from the top,
    port B at y in [140, 158) (B arm: 30 cells),
  - output probe strip at x = 240 on the horizontal channel.
At 6.21 cells/t.u.: A reaches the junction ~8.5 t.u. after firing, B
~4.8 t.u. after firing -> expected inhibition window ~3-4 t.u.; A arrives
at the probe ~31.7 t.u. after firing, B's right-going fragment ~8.5 t.u.
earlier (outside the +/-2.5 t.u. readout window for small dB).

CANDIDATE B GEOMETRY ADAPTATION (documented deviation): the A/B arrival
separation at the probe is (A_arm - B_arm)/c -- the junction-to-probe
leg cancels.  With the Barkley arms (53/30 cells -> 23-cell difference),
candidate B's speed (11.70 cells/t.u.) compresses this to ~2.0 t.u.,
INSIDE the +/-2.5 t.u. readout window: B's right-going fragment
contaminates A's window and the gate is unreadable (verified on a first
run: (0,1) window peak 0.747, separation 1.1x).  For candidate B the arm
difference is therefore enlarged to 61 cells (A port x[5,23) -> A arm 73;
B port y[122,140) -> B arm 12; separation ~5.2 t.u.) and the window
narrowed to +/-2.0 t.u.  The junction itself, channel widths and port
sizes are unchanged.

Protocol (mirrors the Barkley version):
  0. Naive readout check: threshold crossings over the WHOLE probe trace
     reproduce the literature result -- an open junction computes OR
     (B's fragment diffracts into the output channel), not XOR.
  1. Calibrate the readout window on A's lone arrival.
  2. Truth table with windowed readout: (0,0)->0, (1,0)->1, (0,1)->0,
     (1,1)->0  =  A AND (NOT B).
  3. Inhibitor delay sweep dB -> coincidence window for inhibition.

Usage:
    ../.venv/bin/python rd_transfer_logic_oregonator.py [A|B]

Outputs: Analysis/figures/rd_transfer_logic_oregonator{,_B}.{png,json},
         Analysis/figures/rd_transfer_logic_oregonator_snapshots{,_B}.png
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from rd_core import RDSubstrate

NX, NY = 256, 200
CY = NY // 2               # 100
TJ = 96                    # junction x
W = 20
W2 = W // 2
B_TOP = 172                # B channel top
X_PROBE = 240
U_THRESH = 0.5
DURATION = 30
NSTEPS = 1000              # 50 t.u.; A arrives at the probe ~32 t.u.
DT = 0.05
Q = 0.002

CANDIDATES = {
    'A': dict(f=1.4375, eps=0.05014844822490394, phi=0.010,
              label='A (f=1.4375, eps=0.0501, phi=0.010)',
              portA=(25, 43), portB=(140, 158), win_half=2.5),
    # arms enlarged to 73/12 cells (61-cell difference) so the A/B
    # arrival separation survives the higher speed -- see header
    'B': dict(f=1.4375, eps=0.01994079648317803, phi=0.010,
              label='B (f=1.4375, eps=0.0199, phi=0.010)',
              portA=(5, 23), portB=(122, 140), win_half=2.0),
}
SUFFIX = {'A': '', 'B': '_B'}


def rest_u_star(f, phi):
    """Smallest positive root of the u-nullcline with v = u (as in
    oregonator_regime_hunt.rest_state)."""
    def F(u):
        return u - u**2 - (f * u + phi) * (u - Q) / (u + Q)
    ugrid = np.concatenate(([0.0], np.logspace(-7, np.log10(1.2), 4000)))
    vals = F(ugrid)
    for i in range(len(ugrid) - 1):
        if vals[i] > 0.0 and vals[i + 1] <= 0.0:
            return brentq(F, ugrid[i], ugrid[i + 1], xtol=1e-14, rtol=1e-12)
    raise RuntimeError('no rest state found')


def make_rd(cand, u_star):
    wall = np.ones((NX, NY), bool)
    wall[:, CY - W2:CY + W2] = False          # A channel -> output
    wall[TJ - W2:TJ + W2, CY:B_TOP] = False   # B channel joining from top
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, kinetics='oregonator',
                     eps=cand['eps'], f=cand['f'])
    rd.set_phi(cand['phi'])
    rd.set_walls(wall)
    rd.u[~wall] = u_star          # (0,0) is NOT rest when phi > 0
    rd.v[~wall] = u_star
    pA = np.zeros((NX, NY), bool)
    pA[cand['portA'][0]:cand['portA'][1], CY - W2:CY + W2] = True
    pB = np.zeros((NX, NY), bool)
    pB[TJ - W2:TJ + W2, cand['portB'][0]:cand['portB'][1]] = True
    rd.add_port('A', pA)
    rd.add_port('B', pB)
    probe = np.zeros((NX, NY), bool)
    probe[X_PROBE, CY - 1:CY + 2] = True
    rd.add_probe('out', probe)
    return rd, wall


def run_case(cand, u_star, fire_a, fire_b, dB=0, nsteps=NSTEPS):
    rd, _ = make_rd(cand, u_star)
    if fire_a:
        rd.pulse_train('A', [0], duration=DURATION)
    if fire_b:
        rd.pulse_train('B', [dB], duration=DURATION)
    return rd.run(nsteps)


def count_crossings(series, thresh=U_THRESH):
    above = series >= thresh
    return int(np.sum((~above[:-1]) & above[1:]))


def first_crossing(t, s, thresh=U_THRESH):
    idx = np.nonzero(s >= thresh)[0]
    return float(t[idx[0]]) if len(idx) else None


def window_peak(data, t0, win_half):
    m = (data['t'] >= t0 - win_half) & (data['t'] <= t0 + win_half)
    return float(data['out'][m].max())


def main(cand_key):
    cand = CANDIDATES[cand_key]
    win_half = cand['win_half']
    u_star = rest_u_star(cand['f'], cand['phi'])
    sfx = SUFFIX[cand_key]
    print(f'Candidate {cand["label"]}: rest state u* = {u_star:.6f}')

    # --- no-stimulus control: medium must stay at rest ----------------------
    rd, _ = make_rd(cand, u_star)
    rd.run(1000)
    quiet = float(rd.u.max())
    print(f'Control (no stimulus, 50 t.u.): max u = {quiet:.6f} '
          f'-> {"QUIET" if quiet < 0.1 else "SPONTANEOUS FIRING!"}')

    # --- calibrate the readout window on A's lone arrival --------------------
    d10 = run_case(cand, u_star, True, False)
    tA = first_crossing(d10['t'], d10['out'])
    nA = count_crossings(d10['out'])
    print(f'A alone arrives at probe t = {tA:.2f} t.u. '
          f'({nA} crossing(s) over whole trace; window +/-{win_half} t.u.)')
    assert nA == 1, f'physics sanity failed: {nA} crossings for single pulse'

    # --- 0) naive readout: open junction computes OR, not XOR -----------------
    d01 = run_case(cand, u_star, False, True)
    d11 = run_case(cand, u_star, True, True, 0)
    naive = {'00': 0,
             '10': count_crossings(d10['out']),
             '01': count_crossings(d01['out']),
             '11': count_crossings(d11['out'])}
    print('Naive whole-trace readout (any crossing = 1): '
          + ', '.join(f'({k[0]},{k[1]})->{int(v > 0)}'
                      for k, v in naive.items())
          + '  -- OR, not XOR (annihilation not readable)')

    # --- truth table (windowed readout) ---------------------------------------
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
    for dB in [0, 20, 40, 60, 80, 100, 120, 160]:
        pk = window_peak(run_case(cand, u_star, True, True, dB), tA,
                         win_half)
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
        'kinetics': f'oregonator candidate {cand["label"]}, q=0.002, '
                    f'Du=1, Dv=0.6, de-hacked rd_core',
        'rest_state_u_star': u_star,
        'control_max_u_50tu': quiet,
        'control_quiet': bool(quiet < 0.1),
        'geometry': (f'{NX}x{NY}, W={W}, TJ={TJ}, '
                     f'A port x{cand["portA"]}, B port y{cand["portB"]}, '
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

    # --- snapshot figure (1,0) vs (1,1) ---------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for row, (fa, fb, tag) in enumerate([(True, False, 'A only'),
                                         (True, True, 'A and B (coincident)')]):
        rd, wall = make_rd(cand, u_star)
        rd.pulse_train('A', [0], duration=DURATION)
        if fb:
            rd.pulse_train('B', [0], duration=DURATION)
        for col, ts in enumerate([150, 250, 400, 700]):
            rd.run(ts - rd.t)
            ax = axes[row, col]
            u = np.ma.masked_where(wall, rd.u)
            ax.imshow(u.T, origin='lower', cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'{tag}, step {ts}', fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'TEST 3 (oregonator {cand_key}): inhibition gate dynamics',
                 fontsize=13)
    fig.tight_layout()
    snap_path = f'figures/rd_transfer_logic_oregonator_snapshots{sfx}.png'
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

    plt.suptitle(f'TEST 3 (oregonator {cand_key}): two-input collision '
                 f'inhibition gate', fontsize=13)
    plt.tight_layout()
    figpath = f'figures/rd_transfer_logic_oregonator{sfx}.png'
    plt.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f'Saved {figpath}')
    jsonpath = f'figures/rd_transfer_logic_oregonator{sfx}.json'
    with open(jsonpath, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'Saved {jsonpath}')


if __name__ == '__main__':
    key = sys.argv[1].upper() if len(sys.argv) > 1 else 'A'
    assert key in CANDIDATES, f'unknown candidate {key!r}'
    main(key)
