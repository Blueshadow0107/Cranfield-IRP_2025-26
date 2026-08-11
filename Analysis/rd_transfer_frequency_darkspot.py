"""
TEST 2 (dark-spot Oregonator) -- Refractory frequency response with a
one-shot dark-spot input drive.

Mirrors rd_transfer_frequency_oregonator.py but replaces the clamped port
with a circular dark spot: phi is reduced from 0.010 to 0.002 for 3.0 t.u.
(60 steps) at the channel entrance.  Per rd_flash_calib, this emits exactly
one pulse per flash.

Outputs: Analysis/figures/rd_transfer_frequency_darkspot.{png,json}
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from rd_core import RDSubstrate
from rd_darkspot_driver_multi import run_darkspot_multi

NX, NY = 300, 48
CY = NY // 2
WIDTH = 16
X_FAR = 240
X_PORT = (0, 18)
U_THRESH = 0.5
DT = 0.05
Q = 0.002

CAND = dict(f=1.4375, eps=0.05014844822490394, phi=0.010,
            label='A (f=1.4375, eps=0.0501, phi=0.010)')

PHI0 = 0.010
PHI_DARK = 0.002
T_FLASH = 3.0                       # single-pulse flash
DURATION = int(T_FLASH / DT)        # 60 steps
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
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, kinetics='oregonator',
                     eps=CAND['eps'], f=CAND['f'])
    rd.set_phi(PHI0)
    wall = np.ones((NX, NY), bool)
    wall[:, CY - WIDTH // 2:CY + WIDTH // 2] = False
    rd.set_walls(wall)
    rd.u[~wall] = u_star
    rd.v[~wall] = u_star
    return rd, wall


def count_crossings(series, thresh=U_THRESH):
    above = series >= thresh
    return int(np.sum((~above[:-1]) & above[1:]))


def two_pulse(u_star, separation, tail=1400):
    """Two flashes `separation` steps apart; return far-probe series."""
    rd, wall = make_rd(u_star)
    spot = circle_mask((X_PORT[0] + X_PORT[1]) // 2, CY, SPOT_R)
    spot = spot & ~wall
    nsteps = separation + DURATION + tail
    spots = [dict(mask=spot, times=[0, separation],
                  duration=DURATION, phi_dark=PHI_DARK)]
    probe = np.zeros((NX, NY), bool)
    probe[X_FAR, CY - 1:CY + 2] = True
    probes = {'far': probe}
    return run_darkspot_multi(rd, PHI0, spots, nsteps, probes)


def pulse_train_run(u_star, period, n_pulses, tail=1400):
    times = [k * period for k in range(n_pulses)]
    nsteps = times[-1] + DURATION + tail
    rd, wall = make_rd(u_star)
    spot = circle_mask((X_PORT[0] + X_PORT[1]) // 2, CY, SPOT_R)
    spot = spot & ~wall
    spots = [dict(mask=spot, times=times,
                  duration=DURATION, phi_dark=PHI_DARK)]
    probe = np.zeros((NX, NY), bool)
    probe[X_FAR, CY - 1:CY + 2] = True
    probes = {'far': probe}
    return run_darkspot_multi(rd, PHI0, spots, nsteps, probes)


def main():
    u_star = rest_u_star(CAND['f'], CAND['phi'])
    print(f'Candidate {CAND["label"]}: rest state u* = {u_star:.6f}')

    # --- 0) no-stimulus control ---------------------------------------------
    rd, wall = make_rd(u_star)
    nsteps_ctrl = 2000
    spots = []
    probe = np.zeros((NX, NY), bool)
    probe[X_FAR, CY - 1:CY + 2] = True
    probes = {'far': probe}
    data = run_darkspot_multi(rd, PHI0, spots, nsteps_ctrl, probes)
    quiet = float(data['far'].max())
    print(f'Control (no stimulus, 100 t.u.): max u = {quiet:.6f} '
          f'-> {"QUIET" if quiet < 0.1 else "SPONTANEOUS FIRING!"}')

    # --- 1) refractory period: coarse scan then refinement ------------------
    print('Two-pulse refractory scan (coarse)...')
    survivors = {}
    for S_tu in [3, 4, 6, 8, 12, 16, 24, 32]:
        S = int(round(S_tu / DT))
        S = max(S, DURATION)          # flashes must not overlap
        n = count_crossings(two_pulse(u_star, S)['far'])
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
            S = max(S, DURATION)
            n = count_crossings(two_pulse(u_star, S)['far'])
            survivors[S] = n
            print(f'  S={S:4d} steps ({S*DT:5.1f} t.u., refine): '
                  f'{n} pulses arrived', flush=True)
            if n >= 2:
                refractory = S
                break
    print(f'Refractory period ~ {refractory} steps '
          f'({refractory*DT:.2f} time units)')

    # --- 2) frequency response ----------------------------------------------
    # With a 3.0 t.u. flash the start-to-start period must exceed the flash
    # duration or successive flashes merge into one long pacemaker flash.
    # Explicit non-overlapping periods are used; the lower bound is therefore
    # the input encoding, not the medium refractory period.
    periods = [80, 100, 120, 160, 240, 320, 480, 640]
    n_pulses = 6
    rows = []
    for P in periods:
        data = pulse_train_run(u_star, P, n_pulses)
        n_out = count_crossings(data['far'])
        duration_t = (n_pulses - 1) * P * DT
        f_in = 1.0 / (P * DT)
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
        'kinetics': f'oregonator candidate {CAND["label"]}, q={Q}, '
                    f'Du=1, Dv=0, dark-spot drive',
        'phi0': PHI0, 'phi_dark': PHI_DARK,
        't_flash_tu': T_FLASH, 'spot_radius': SPOT_R,
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

    plt.suptitle('TEST 2 (dark-spot Oregonator A): refractory-limited '
                 'frequency response', fontsize=13)
    plt.tight_layout()
    figpath = 'figures/rd_transfer_frequency_darkspot.png'
    plt.savefig(figpath, dpi=150)
    print(f'Saved {figpath}')
    jsonpath = 'figures/rd_transfer_frequency_darkspot.json'
    with open(jsonpath, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'Saved {jsonpath}')


if __name__ == '__main__':
    main()
