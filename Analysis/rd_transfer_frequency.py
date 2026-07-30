"""
TEST 2 -- Refractory frequency response (excitable Barkley BZ medium).

NOTE ON KINETICS: this test requires a genuinely excitable medium (single
pulses on demand).  The Oregonator baseline (eps=0.05, q=0.002, f=1.4) is
a relaxation OSCILLATOR -- its rest state is an unstable node (verified
numerically: eigenvalues +10.8, +0.67), so a fired port's wake re-fires
spontaneously and no clean two-pulse/frequency protocol is possible.
This test therefore uses the project's validated Barkley excitable model
(barkley_bz_demo.py: a=0.75, b=0.01, eps=0.02, Du=1, Dv=0).

A wide channel (W = 32, well above the Test-1 block threshold) carries
pulse trains from a left-edge port to a far downstream probe.

Protocol:
  1. Absolute refractory period: fire two pulses S steps apart; the
     second survives iff it reaches the far probe.  Scan S to find the
     minimum surviving separation.
  2. Trains of N pulses at periods spanning well below to well above the
     refractory period; count transmitted pulses (u = 0.5 crossings at
     the far probe).

Output: f_out vs f_in (expect a slope-1 region then saturation/staircase),
maximum following rate and refractory period in JSON.

Outputs: Analysis/figures/rd_transfer_frequency.png (+ .json).
"""

import json
import numpy as np
import matplotlib.pyplot as plt

from rd_core import RDSubstrate

NX, NY = 256, 256
CY = NY // 2
WIDTH = 32
X_FAR = int(0.8 * NX)     # 204
X_PORT = (0, 18)
U_THRESH = 0.5
DURATION = 30
DT = 0.05
KIN = dict(kinetics='barkley', eps=0.02, Dv=0.0)


def disk_mask(cx, cy, r):
    X, Y = np.mgrid[0:NX, 0:NY]
    return (X - cx)**2 + (Y - cy)**2 <= r**2


def make_rd():
    rd = RDSubstrate(nx=NX, ny=NY, **KIN)
    wall = np.ones((NX, NY), bool)
    wall[:, CY - WIDTH // 2:CY + WIDTH // 2] = False
    rd.set_walls(wall)
    port = np.zeros((NX, NY), bool)
    port[X_PORT[0]:X_PORT[1], CY - WIDTH // 2:CY + WIDTH // 2] = True
    rd.add_port('in', port)
    rd.add_probe('far', disk_mask(X_FAR, CY, 2))
    return rd


def count_crossings(series, thresh=U_THRESH):
    """Number of upward threshold crossings."""
    above = series >= thresh
    return int(np.sum((~above[:-1]) & above[1:]))


def two_pulse(separation, nsteps=2200):
    """Fire two pulses `separation` steps apart; return far-probe series."""
    rd = make_rd()
    rd.pulse_train('in', [0, separation], duration=DURATION)
    return rd.run(nsteps)


def pulse_train_run(period, n_pulses, tail=1400):
    times = [k * period for k in range(n_pulses)]
    nsteps = times[-1] + DURATION + tail
    rd = make_rd()
    rd.pulse_train('in', times, duration=DURATION)
    data = rd.run(nsteps)
    return data, nsteps


def main():
    # --- 1) refractory period ---------------------------------------------
    print('Two-pulse refractory scan (barkley excitable medium)...')
    survivors = {}
    for S in [200, 150, 125, 100, 90, 80, 70, 60, 50, 40, 30]:
        data = two_pulse(S)
        n_cross = count_crossings(data['far'])
        survivors[S] = n_cross
        print(f'  S={S:4d} steps ({S*DT:5.1f} t.u.): {n_cross} pulses arrived')
    surviving = sorted(S for S, n in survivors.items() if n >= 2)
    refractory = surviving[0] if surviving else None
    # refine in steps of 2 between largest failing and smallest surviving
    failing = [S for S, n in survivors.items() if n < 2]
    if refractory and failing:
        for S in range(max(failing) + 2, refractory, 2):
            data = two_pulse(S)
            n_cross = count_crossings(data['far'])
            survivors[S] = n_cross
            print(f'  S={S:4d} steps (refine): {n_cross} pulses arrived')
            if n_cross >= 2:
                refractory = S
                break
    print(f'Refractory period ~ {refractory} steps '
          f'({refractory*DT:.1f} time units)')

    # --- 2) frequency response --------------------------------------------
    periods = sorted({int(round(refractory * x))
                      for x in [0.45, 0.6, 0.75, 0.9, 1.0, 1.15, 1.35,
                                1.6, 2.0, 2.5]})
    n_pulses = 6
    rows = []
    for P in periods:
        data, nsteps = pulse_train_run(P, n_pulses)
        n_out = count_crossings(data['far'])
        duration_t = (n_pulses - 1) * P * DT
        f_in = 1.0 / (P * DT)
        # sustained output-rate estimate: (n_out-1) intervals spanning the
        # train duration -> equals f_in exactly at 1:1 transmission
        f_out = max(n_out - 1, 0) / (duration_t + 1e-12)
        rows.append({'period_steps': P, 'fired': n_pulses,
                     'transmitted': n_out, 'f_in': f_in, 'f_out': f_out})
        print(f'  period={P:4d} steps: fired {n_pulses}, '
              f'transmitted {n_out}, f_in={f_in:.4f}, f_out~{f_out:.4f}')

    # max following rate: highest f_in at which transmission is 1:1
    one_to_one = [r for r in rows if r['transmitted'] >= r['fired']]
    f_max = max((r['f_in'] for r in one_to_one),
                default=max(r['f_out'] for r in rows))
    f_sat = max(r['f_out'] for r in rows)
    print(f'Max 1:1 following rate = {f_max:.4f} /t.u. '
          f'(period {1.0/f_max:.1f} t.u.); saturated output rate '
          f'= {f_sat:.4f} /t.u.')

    results = {
        'kinetics': 'barkley (a=0.75, b=0.01, eps=0.02, Du=1, Dv=0)',
        'refractory_period_steps': refractory,
        'refractory_period_time': refractory * DT,
        'two_pulse_scan': {str(k): v for k, v in sorted(survivors.items())},
        'train_results': rows,
        'max_following_rate': f_max,
        'saturated_output_rate': f_sat,
    }

    # --- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot([r['f_in'] for r in rows], [r['f_out'] for r in rows], 'o-',
            label='measured')
    lim = max(r['f_in'] for r in rows) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', lw=0.8, label='1:1 line')
    ax.axvline(1.0 / (refractory * DT), color='r', ls=':',
               label=f'1/refractory ({1.0/(refractory*DT):.3f})')
    ax.set_xlabel('input rate f_in (pulses / time unit)')
    ax.set_ylabel('output rate f_out')
    ax.set_title('Frequency response'); ax.legend()

    ax = axes[1]
    ax.plot([r['period_steps'] for r in rows],
            [r['transmitted'] for r in rows], 's-', color='darkgreen')
    ax.axhline(n_pulses, color='k', ls='--', lw=0.8,
               label=f'fired ({n_pulses})')
    ax.axvline(refractory, color='r', ls=':',
               label=f'refractory ({refractory} steps)')
    ax.set_xlabel('input period (steps)')
    ax.set_ylabel('pulses transmitted')
    ax.set_title('Transmission count vs period'); ax.legend()

    plt.suptitle('TEST 2 (barkley): refractory-limited frequency response',
                 fontsize=13)
    plt.tight_layout()
    figpath = 'figures/rd_transfer_frequency.png'
    plt.savefig(figpath, dpi=150)
    print(f'Saved {figpath}')
    jsonpath = 'figures/rd_transfer_frequency.json'
    with open(jsonpath, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'Saved {jsonpath}')


if __name__ == '__main__':
    main()
