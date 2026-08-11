"""
Frequency-response characterisation of the T-junction router.

Geometry: same fixed-wall T-junction as rd_tjunc_router.py.
A static phi barrier is placed on the right branch (the hand-designed
right_block_high field), which routes a single low-frequency pulse to
the left probe.

Input: pulse train at the base of the stem, period varied from well above
the refractory period down to the following limit.
Output: peak response at left and right probes.

The expectation is that low-frequency trains route cleanly to left,
while high-frequency trains are attenuated/blocked because pulses enter
the junction before the medium has recovered, and the phi barrier then
prevents any transmission.
"""
import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rd_core import RDSubstrate
from rd_tjunc_router import (
    build_phi, WALL, NX, NY, A_MASK, U_STAR, F, EPS, DT, PHI0, DARK,
    L_MASK, R_MASK, U_THRESH
)

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG, exist_ok=True)

# Use the right_block_high field from hand tests: blocks right branch
right_block = np.full((4, 4), PHI0)
right_block[2:, :] = 0.040  # right half high phi
phi = build_phi(right_block.flatten())

# Pulse train parameters
T_FLASH = 4.0
NPULSES = 6
PERIODS = [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 16.0, 20.0, 30.0]
TU_AFTER = 40.0


def run_train(period):
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, kinetics='oregonator',
                     f=F, eps=EPS, clamp_rest=(U_STAR, U_STAR))
    rd.set_walls(WALL)
    rd.set_phi(phi)
    rd.u[:] = U_STAR
    rd.v[:] = U_STAR

    # dark-spot pulse train
    phi_flash = phi.copy()
    phi_flash[A_MASK] = DARK
    n_flash = int(T_FLASH / DT)
    n_period = int(period / DT)
    n_after = int(TU_AFTER / DT)
    nsteps = NPULSES * n_period + n_after

    left = np.empty(nsteps)
    right = np.empty(nsteps)
    flash_on = np.zeros(nsteps, bool)
    for p in range(NPULSES):
        start = p * n_period
        flash_on[start:start + n_flash] = True

    for n in range(nsteps):
        rd.set_phi(phi_flash if flash_on[n] else phi)
        rd.run(1)
        left[n] = rd.u[L_MASK].max() if L_MASK.any() else 0.0
        right[n] = rd.u[R_MASK].max() if R_MASK.any() else 0.0

    t = np.arange(nsteps) * DT
    return t, left, right


def main():
    results = []
    for period in PERIODS:
        print(f'period={period:.1f} t.u. ...', flush=True)
        t, left, right = run_train(period)
        lpeak, rpeak = left.max(), right.max()
        lpresent = lpeak >= U_THRESH
        rpresent = rpeak >= U_THRESH
        # count crossings
        lcross = int(np.sum((left[:-1] < U_THRESH) & (left[1:] >= U_THRESH)))
        rcross = int(np.sum((right[:-1] < U_THRESH) & (right[1:] >= U_THRESH)))
        results.append({
            'period_tu': float(period),
            'left_peak': float(lpeak),
            'right_peak': float(rpeak),
            'left_present': bool(lpresent),
            'right_present': bool(rpresent),
            'left_crossings': lcross,
            'right_crossings': rcross
        })
        print(f"  L peak={lpeak:.3f} crossings={lcross}  R peak={rpeak:.3f} crossings={rcross}", flush=True)

    with open(os.path.join(FIG, 'rd_tjunc_frequency.json'), 'w') as fh:
        json.dump({'phi': 'right_block_high', 'results': results}, fh, indent=2)

    # Plot
    periods = np.array([r['period_tu'] for r in results])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot(periods, [r['left_peak'] for r in results], 'o-', label='left probe')
    ax.plot(periods, [r['right_peak'] for r in results], 's-', label='right probe')
    ax.axhline(U_THRESH, color='r', ls='--', lw=1)
    ax.set_xlabel('input pulse period (t.u.)')
    ax.set_ylabel('peak u')
    ax.set_title('T-junction frequency response with right-branch block')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(periods, [r['left_crossings'] for r in results], 'o-', label='left crossings')
    ax.plot(periods, [r['right_crossings'] for r in results], 's-', label='right crossings')
    ax.set_xlabel('input pulse period (t.u.)')
    ax.set_ylabel('number of threshold crossings')
    ax.set_title('threshold crossings vs input period')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'rd_tjunc_frequency.png'), dpi=150)
    print(f'[saved] {FIG}/rd_tjunc_frequency.png')


if __name__ == '__main__':
    main()
