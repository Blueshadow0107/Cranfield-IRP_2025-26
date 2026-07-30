"""
TEST 2 (Oregonator) -- Refractory frequency response, excitable light-held
Oregonator BZ medium.

Adapted copy of rd_transfer_frequency.py (Barkley version) -- the Barkley
script is untouched.  Kinetics are now the validated EXCITABLE Oregonator
regime from the (f, eps, phi) scan
(Notes/oregonator-excitable-regime-hunt-2026-07-20.md):

    Candidate A (primary):   f=1.4375, eps=0.0501, phi=0.010, q=0.002
                             speed 6.21 cells/t.u. (300x40, W=12 channel)
    Candidate B (secondary): f=1.4375, eps=0.0199, phi=0.010, q=0.002
                             speed 11.70 cells/t.u.

CRITICAL DIFFERENCE vs Barkley: with phi>0 the state (0,0) is NOT rest
(du/dt(0) = phi/eps > 0; a (0,0)-initialised channel ignites
synchronously).  All fields are initialised AT the rest state
(u*, v*=u*) computed from the u-nullcline, exactly as in
oregonator_regime_hunt.py.  A no-stimulus control verifies the medium
stays at rest.

Protocol (mirrors the Barkley version):
  1. Absolute refractory period: two pulses S steps apart; the second
     survives iff it reaches the far probe.  Coarse scan then refinement.
  2. Trains of N pulses at periods spanning below/above the refractory
     period; count transmitted pulses (u = 0.5 crossings at far probe).

Geometry: 300x48 walled channel, W=16 (regime hunt validated W=12;
Oregonator wavelength = speed x refractory is checked and printed).
Far probe: 3-cell strip at x=240 (tight strip instead of the Barkley
r=2 disk -- the disk dilutes the peak to 0.536, marginally above 0.5).

Usage:
    ../.venv/bin/python rd_transfer_frequency_oregonator.py [A|B]

Outputs: Analysis/figures/rd_transfer_frequency_oregonator{,_B}.{png,json}
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

CANDIDATES = {
    'A': dict(f=1.4375, eps=0.05014844822490394, phi=0.010,
              label='A (f=1.4375, eps=0.0501, phi=0.010)'),
    'B': dict(f=1.4375, eps=0.01994079648317803, phi=0.010,
              label='B (f=1.4375, eps=0.0199, phi=0.010)'),
}
SUFFIX = {'A': '', 'B': '_B'}


def rest_u_star(f, phi):
    """Smallest positive root of the u-nullcline with v = u (dv/dt = 0),
    identical to oregonator_regime_hunt.rest_state."""
    def F(u):
        return u - u**2 - (f * u + phi) * (u - Q) / (u + Q)
    ugrid = np.concatenate(([0.0], np.logspace(-7, np.log10(1.2), 4000)))
    vals = F(ugrid)
    for i in range(len(ugrid) - 1):
        if vals[i] > 0.0 and vals[i + 1] <= 0.0:
            return brentq(F, ugrid[i], ugrid[i + 1], xtol=1e-14, rtol=1e-12)
    raise RuntimeError('no rest state found')


def make_rd(cand, u_star):
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, kinetics='oregonator',
                     eps=cand['eps'], f=cand['f'])
    rd.set_phi(cand['phi'])
    wall = np.ones((NX, NY), bool)
    wall[:, CY - WIDTH // 2:CY + WIDTH // 2] = False
    rd.set_walls(wall)
    rd.u[~wall] = u_star          # (0,0) is NOT rest when phi > 0
    rd.v[~wall] = u_star
    port = np.zeros((NX, NY), bool)
    port[X_PORT[0]:X_PORT[1], CY - WIDTH // 2:CY + WIDTH // 2] = True
    rd.add_port('in', port)
    probe = np.zeros((NX, NY), bool)
    probe[X_FAR, CY - 1:CY + 2] = True
    rd.add_probe('far', probe)
    return rd


def count_crossings(series, thresh=U_THRESH):
    """Number of upward threshold crossings."""
    above = series >= thresh
    return int(np.sum((~above[:-1]) & above[1:]))


def two_pulse(cand, u_star, separation, tail=1000):
    """Fire two pulses `separation` steps apart; return far-probe series."""
    rd = make_rd(cand, u_star)
    rd.pulse_train('in', [0, separation], duration=DURATION)
    return rd.run(separation + DURATION + tail)


def pulse_train_run(cand, u_star, period, n_pulses, tail=1000):
    times = [k * period for k in range(n_pulses)]
    nsteps = times[-1] + DURATION + tail
    rd = make_rd(cand, u_star)
    rd.pulse_train('in', times, duration=DURATION)
    return rd.run(nsteps)


def main(cand_key):
    cand = CANDIDATES[cand_key]
    u_star = rest_u_star(cand['f'], cand['phi'])
    print(f'Candidate {cand["label"]}: rest state u* = {u_star:.6f}')

    # --- 0) no-stimulus control: medium must stay at rest -------------------
    rd = make_rd(cand, u_star)
    rd.run(2000)
    quiet = float(rd.u.max())
    print(f'Control (no stimulus, 100 t.u.): max u = {quiet:.6f} '
          f'-> {"QUIET" if quiet < 0.1 else "SPONTANEOUS FIRING!"}')

    # --- 1) refractory period: coarse scan then refinement ------------------
    print('Two-pulse refractory scan (coarse)...')
    survivors = {}
    for S_tu in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]:
        S = int(round(S_tu / DT))
        n = count_crossings(two_pulse(cand, u_star, S)['far'])
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
        # bisect-ish refinement in steps of max(2, gap//8)
        lo, hi = max(failing), refractory
        step = max(2, (hi - lo) // 8)
        for S in range(lo + step, hi, step):
            n = count_crossings(two_pulse(cand, u_star, S)['far'])
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
        data = pulse_train_run(cand, u_star, P, n_pulses)
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
        'kinetics': f'oregonator candidate {cand["label"]}, q=0.002, '
                    f'Du=1, Dv=0.6, de-hacked rd_core',
        'rest_state_u_star': u_star,
        'control_max_u_100tu': quiet,
        'control_quiet': bool(quiet < 0.1),
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

    plt.suptitle(f'TEST 2 (oregonator {cand_key}): refractory-limited '
                 f'frequency response', fontsize=13)
    plt.tight_layout()
    sfx = SUFFIX[cand_key]
    figpath = f'figures/rd_transfer_frequency_oregonator{sfx}.png'
    plt.savefig(figpath, dpi=150)
    print(f'Saved {figpath}')
    jsonpath = f'figures/rd_transfer_frequency_oregonator{sfx}.json'
    with open(jsonpath, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'Saved {jsonpath}')


if __name__ == '__main__':
    key = sys.argv[1].upper() if len(sys.argv) > 1 else 'A'
    assert key in CANDIDATES, f'unknown candidate {key!r}'
    main(key)
