"""
Two-spot authority map (timing knob) in FREE MEDIUM -- no wall mask.

Two dark spots (radius R) in an open no-flux box with phi0 background.

Part 1 -- summation: both spots at phi=0.005 (each below the firing
threshold measured in rd_darkspot.py: onset between phi 0.006 and 0.004).
Cases: A alone, B alone, A+B simultaneously.  Question: does coincident
drive sum into firing?  (amplitude-free graded drive / AND-like primitive)

Part 2 -- delay kernel: both spots at phi=0.002 (both fire alone).
B's first firing is delayed by dB relative to A; sweep dB.  Probe at the
midpoint records the pattern (rate + peak pattern factor) vs delay --
the timing-kernel of the two-spot input.

Outputs: Analysis/figures/rd_twospot.{json,png}
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rd_core import RDSubstrate

NX, NY = 160, 160
CY = NY // 2
SPOT_R = 12
A_POS = (50, 80)
B_POS = (110, 80)
PROBE_POS = (80, 80)
DT = 0.05
TU_DISCARD = 40.0
TU_RUN = 120.0
F, EPS, PHI0 = 1.4375, 0.05014844822490394, 0.010
U_STAR = 0.0030821
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

SUM_PHI = 0.005
KERN_PHI = 0.002
DELAYS_TU = [0, 1, 2, 3, 4, 6]


def spot_mask(pos):
    X, Y = np.mgrid[0:NX, 0:NY].astype(float)
    return ((X - pos[0]) ** 2 + (Y - pos[1]) ** 2) < SPOT_R ** 2


def make_rd(spots):
    """spots: dict name -> phi value (dark spots); phi0 elsewhere."""
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, kinetics='oregonator',
                     f=F, eps=EPS, clamp_rest=(U_STAR, U_STAR))
    phi = np.full((NX, NY), PHI0)
    for val, pos in spots:
        phi[spot_mask(pos)] = val
    rd.set_phi(phi)
    rd.u[:] = U_STAR
    rd.v[:] = U_STAR
    probe = np.zeros((NX, NY), bool)
    probe[PROBE_POS[0] - 1:PROBE_POS[0] + 2,
          PROBE_POS[1] - 1:PROBE_POS[1] + 2] = True
    rd.add_probe('mid', probe)
    return rd


def measure(rd, tu=TU_RUN):
    nsteps = int(tu / DT)
    ndiscard = int(TU_DISCARD / DT)
    series = []
    for n in range(nsteps):
        rd.run(1)
        if n >= ndiscard:
            series.append(float(rd.u[PROBE_POS]))
    series = np.array(series)
    idx = np.nonzero((series[:-1] <= 0.5) & (series[1:] > 0.5))[0]
    ncross = len(idx)
    return {'n_pulses': int(ncross),
            'rate': ncross / ((nsteps - ndiscard) * DT),
            'peak': float(series.max())}


def part1():
    out = {}
    for tag, spots in [('A_only', [(SUM_PHI, A_POS)]),
                       ('B_only', [(SUM_PHI, B_POS)]),
                       ('A_plus_B', [(SUM_PHI, A_POS), (SUM_PHI, B_POS)])]:
        rd = make_rd(spots)
        out[tag] = measure(rd)
        print(f"[sum] {tag}: pulses={out[tag]['n_pulses']} "
              f"rate={out[tag]['rate']:.4f} peak={out[tag]['peak']:.3f}",
              flush=True)
    return out


def part2():
    out = {}
    for d_tu in DELAYS_TU:
        d_steps = int(d_tu / DT)
        rd = make_rd([(KERN_PHI, A_POS), (KERN_PHI, B_POS)])
        # B starts LIT (patch held at phi0); darkens at n == d_steps
        phi = np.full((NX, NY), PHI0)
        phi[spot_mask(A_POS)] = KERN_PHI
        rd.set_phi(phi)
        nsteps = int(TU_RUN / DT)
        ndiscard = int(TU_DISCARD / DT)
        series = []
        for n in range(nsteps):
            if n == d_steps:
                phi2 = np.full((NX, NY), PHI0)
                phi2[spot_mask(A_POS)] = KERN_PHI
                phi2[spot_mask(B_POS)] = KERN_PHI
                rd.set_phi(phi2)
            rd.run(1)
            if n >= ndiscard:
                series.append(float(rd.u[PROBE_POS]))
        series = np.array(series)
        idx = np.nonzero((series[:-1] <= 0.5) & (series[1:] > 0.5))[0]
        out[d_tu] = {'n_pulses': len(idx),
                     'rate': len(idx) / ((nsteps - ndiscard) * DT),
                     'peak': float(series.max())}
        print(f"[kernel] dB={d_tu} t.u.: pulses={out[d_tu]['n_pulses']} "
              f"rate={out[d_tu]['rate']:.4f} peak={out[d_tu]['peak']:.3f}",
              flush=True)
    return out


if __name__ == '__main__':
    print('--- part 1: summation (both spots phi={}) ---'.format(SUM_PHI))
    s = part1()
    print('--- part 2: delay kernel (both spots phi={}) ---'.format(KERN_PHI))
    k = part2()
    out = {'config': {'box': [NX, NY], 'spot_r': SPOT_R, 'A': A_POS,
                      'B': B_POS, 'probe': PROBE_POS, 'phi0': PHI0,
                      'sum_phi': SUM_PHI, 'kernel_phi': KERN_PHI,
                      'note': 'free medium, no walls; dark-spot inputs'},
           'summation': s, 'delay_kernel': k}
    with open(os.path.join(FIG, 'rd_twospot.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    print('saved rd_twospot.json')
