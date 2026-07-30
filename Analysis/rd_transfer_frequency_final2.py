"""
TEST 2 (FINAL2, clamp-rest fix) -- Refractory frequency response,
BOTH kinetics (Barkley and excitable Oregonator Candidate A).

Combines rd_transfer_frequency.py (Barkley, OLD clipped solver -- numbers
superseded) and rd_transfer_frequency_oregonator.py (post-de-hack but
with the (0,0)-clamp sink artifact) into one script on the fixed core.

DOCUMENTED GEOMETRY UNIFICATION: both kinetics now use the 300x48 strip,
W=16 channel, 3-cell far-probe strip at x=240 (the Oregonator _final
geometry).  The old Barkley run used 256x256, W=32, r=2 disk probe at
x=204; the strip geometry removes probe dilution and cuts compute.
W=16 is well above the Test-1 block threshold (W>=1 transmits) for both
kinetics, so the channel is optically 'wide' and the measured refractory
period is a medium property, not a geometric one.

Clamp fix: the Oregonator port is clamped to (u*, u*) between pulses
(clamp_rest), not (0, 0); Barkley keeps the (0, 0) default (= its rest).

Protocol (mirrors both predecessors):
  0. No-stimulus control (medium stays at rest); single-pulse run must
     give EXACTLY ONE probe threshold crossing (no pacemaking).
  1. Absolute refractory period: two pulses S steps apart; the second
     survives iff it reaches the far probe.  Coarse scan then refinement.
  2. Trains of 6 pulses at periods spanning below/above the refractory
     period; count transmitted pulses (u = 0.5 crossings at far probe).

Usage:
    ../.venv/bin/python rd_transfer_frequency_final2.py [barkley|oregonator|all]

Outputs: Analysis/figures/rd_transfer_frequency_final2_<kin>.{png,json}
"""

import json
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from rd_core import RDSubstrate

NX, NY = 300, 48
CY = NY // 2
WIDTH = 16
X_FAR = 240
X_PORT = (0, 18)
U_THRESH = 0.5
DURATION = 30
DT = 0.05
Q = 0.002
TAIL = 1400          # steps of quiet tail after the last pulse

OREG_A = dict(f=1.4375, eps=0.05014844822490394, phi=0.010)

KINETICS = {
    'barkley': dict(kw=dict(kinetics='barkley', eps=0.02, Dv=0.0),
                    phi=0.0, label='Barkley (a=0.75, b=0.01, eps=0.02, Dv=0)'),
    'oregonator': dict(kw=dict(kinetics='oregonator', f=OREG_A['f'],
                               eps=OREG_A['eps']),
                       phi=OREG_A['phi'],
                       label='Oregonator A (f=1.4375, eps=0.0501, phi=0.010)'),
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
    u_rest, v_rest = rest_state(kin)
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, clamp_rest=(u_rest, v_rest),
                     **spec['kw'])
    if spec['phi']:
        rd.set_phi(spec['phi'])
    wall = np.ones((NX, NY), bool)
    wall[:, CY - WIDTH // 2:CY + WIDTH // 2] = False
    rd.set_walls(wall)
    rd.u[~wall] = u_rest
    rd.v[~wall] = v_rest
    port = np.zeros((NX, NY), bool)
    port[X_PORT[0]:X_PORT[1], CY - WIDTH // 2:CY + WIDTH // 2] = True
    rd.add_port('in', port)
    probe = np.zeros((NX, NY), bool)
    probe[X_FAR, CY - 1:CY + 2] = True
    rd.add_probe('far', probe)
    return rd, wall


def count_crossings(series, thresh=U_THRESH):
    above = series >= thresh
    return int(np.sum((~above[:-1]) & above[1:]))


def two_pulse(kin, separation, tail=TAIL):
    rd, _ = make_rd(kin)
    rd.pulse_train('in', [0, separation], duration=DURATION)
    return rd.run(separation + DURATION + tail)


def pulse_train_run(kin, period, n_pulses, tail=TAIL):
    times = [k * period for k in range(n_pulses)]
    nsteps = times[-1] + DURATION + tail
    rd, _ = make_rd(kin)
    rd.pulse_train('in', times, duration=DURATION)
    return rd.run(nsteps)


def main(kin):
    spec = KINETICS[kin]
    u_rest, _ = rest_state(kin)
    print(f'=== kinetics = {kin}: {spec["label"]} ===  rest u* = {u_rest:.6f}')

    # --- 0) controls ------------------------------------------------------
    rd, wall = make_rd(kin)
    rd.run(2000)
    dev = float(np.abs(rd.u[~wall] - u_rest).max())
    leak = float(np.abs(rd.u[wall]).max())
    print(f'Control (no stimulus, 100 t.u.): max |u-u*| = {dev:.3e}, '
          f'wall leak = {leak:.1e} '
          f'-> {"QUIET" if dev < 0.01 and leak == 0.0 else "FAILED!"}')
    assert dev < 0.01 and leak == 0.0

    rd1, _ = make_rd(kin)
    rd1.pulse_train('in', [0], duration=DURATION)
    n1 = count_crossings(rd1.run(DURATION + TAIL)['far'])
    print(f'Single-pulse check: {n1} crossing(s) at far probe')
    assert n1 == 1, f'physics check failed: {n1} crossings for single pulse'

    # --- 1) refractory period: coarse scan then refinement -----------------
    print('Two-pulse refractory scan (coarse)...')
    survivors = {}
    for S_tu in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]:
        S = int(round(S_tu / DT))
        n = count_crossings(two_pulse(kin, S)['far'])
        survivors[S] = n
        print(f'  S={S:4d} steps ({S*DT:5.1f} t.u.): {n} pulses arrived',
              flush=True)
    surviving = sorted(S for S, n in survivors.items() if n >= 2)
    failing = sorted(S for S, n in survivors.items() if n < 2)
    if not surviving:
        raise RuntimeError('no separation transmitted two pulses -- '
                           'extend the coarse scan')
    refractory = surviving[0]
    if failing and max(failing) < refractory:
        lo, hi = max(failing), refractory
        step = max(2, (hi - lo) // 8)
        for S in range(lo + step, hi, step):
            n = count_crossings(two_pulse(kin, S)['far'])
            survivors[S] = n
            print(f'  S={S:4d} steps ({S*DT:5.1f} t.u., refine): '
                  f'{n} pulses arrived', flush=True)
            if n >= 2:
                refractory = S
                break
    print(f'Refractory period ~ {refractory} steps '
          f'({refractory*DT:.2f} time units)')

    # --- 2) frequency response ----------------------------------------------
    periods = sorted({int(round(refractory * x))
                      for x in [0.45, 0.6, 0.75, 0.9, 1.0, 1.15, 1.35,
                                1.6, 2.0, 2.5]})
    n_pulses = 6
    rows = []
    for P in periods:
        data = pulse_train_run(kin, P, n_pulses)
        n_out = count_crossings(data['far'])
        duration_t = (n_pulses - 1) * P * DT
        f_in = 1.0 / (P * DT)
        # sustained output-rate estimate: (n_out-1) intervals spanning the
        # train duration -> equals f_in exactly at 1:1 transmission
        f_out = max(n_out - 1, 0) / (duration_t + 1e-12)
        rows.append({'period_steps': P, 'fired': n_pulses,
                     'transmitted': n_out, 'f_in': f_in, 'f_out': f_out})
        print(f'  period={P:4d} steps: fired {n_pulses}, '
              f'transmitted {n_out}, f_in={f_in:.4f}, f_out~{f_out:.4f}',
              flush=True)

    one_to_one = [r for r in rows if r['transmitted'] >= r['fired']]
    f_max = max((r['f_in'] for r in one_to_one),
                default=max(r['f_out'] for r in rows))
    f_sat = max(r['f_out'] for r in rows)
    print(f'Max 1:1 following rate = {f_max:.4f} /t.u. '
          f'(period {1.0/f_max:.1f} t.u.); saturated output rate '
          f'= {f_sat:.4f} /t.u.')

    results = {
        'kinetics': spec['label'] + ', de-hacked rd_core + clamp-rest fix',
        'rest_state_u_star': u_rest,
        'control_max_dev_100tu': dev,
        'control_quiet': bool(dev < 0.01),
        'single_pulse_crossings': n1,
        'geometry': f'{NX}x{NY} channel, W={WIDTH}, far probe strip x={X_FAR}',
        'refractory_period_steps': refractory,
        'refractory_period_time': refractory * DT,
        'two_pulse_scan': {str(k): v for k, v in sorted(survivors.items())},
        'train_results': rows,
        'max_following_rate': f_max,
        'saturated_output_rate': f_sat,
        'slope1_region': bool(one_to_one),
    }

    # --- figure --------------------------------------------------------------
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

    plt.suptitle(f'TEST 2 FINAL2 ({kin}): refractory-limited frequency '
                 f'response', fontsize=13)
    plt.tight_layout()
    figpath = f'figures/rd_transfer_frequency_final2_{kin}.png'
    plt.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f'Saved {figpath}')
    jsonpath = f'figures/rd_transfer_frequency_final2_{kin}.json'
    with open(jsonpath, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'Saved {jsonpath}')


if __name__ == '__main__':
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else 'all'
    assert arg in ('barkley', 'oregonator', 'all'), arg
    for k in (list(KINETICS) if arg == 'all' else [arg]):
        main(k)
