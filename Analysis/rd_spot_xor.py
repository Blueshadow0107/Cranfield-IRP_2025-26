"""
rd_spot_xor.py -- PROTOCOL STAGE ONLY (baseline test, no training).

Free-medium 200x200, Oregonator-A, phi0=0.010.  Two one-shot dark spots
A(40,60), B(40,100) r=12; probe arc P1(140,40), P2(140,80), P3(140,120).

Step 1: calibrate T_flash -- single flash of A at several durations,
        check exactly one wave emits and reaches the arc.
Step 2: run the four input patterns at uniform phi0, record per-probe
        signatures (first arrival, window peak, crossings).

Outputs: Analysis/figures/rd_spot_xor_protocol.{json,png}
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rd_core import RDSubstrate

NX, NY = 200, 200
SPOT_R = 12
A_POS, B_POS = (40, 80), (40, 120)
PROBES = {'PC': (95, 100),
          'PA1': (82, 26), 'PA2': (114, 58), 'PA3': (125, 100),
          'PA4': (114, 142), 'PA5': (82, 174)}
DT = 0.05
TU_RUN = 100.0
F, EPS, PHI0 = 1.4375, 0.05014844822490394, 0.010
U_STAR = 0.0030821
DARK = 0.002
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
U_THRESH = 0.5


def spot_mask(pos):
    X, Y = np.mgrid[0:NX, 0:NY].astype(float)
    return ((X - pos[0]) ** 2 + (Y - pos[1]) ** 2) < SPOT_R ** 2


def make_rd():
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, kinetics='oregonator',
                     f=F, eps=EPS, clamp_rest=(U_STAR, U_STAR))
    rd.set_phi(np.full((NX, NY), PHI0))
    rd.u[:] = U_STAR
    rd.v[:] = U_STAR
    return rd


def flash(rd, pos, t_start, t_dur, dark=DARK):
    """Darken the spot's patch from t_start for t_dur time units."""
    phi = np.full((NX, NY), PHI0)
    phi[spot_mask(pos)] = dark
    rd.set_phi(phi)
    rd.run(int(t_dur / DT))
    rd.set_phi(np.full((NX, NY), PHI0))


def run_pattern(fire_a, fire_b, t_flash):
    rd = make_rd()
    if fire_a:
        flash(rd, A_POS, 0, t_flash)
    if fire_b:
        flash(rd, B_POS, 0, t_flash)
    series = {name: [] for name in PROBES}
    for _ in range(int(TU_RUN / DT)):
        rd.run(1)
        for name, pos in PROBES.items():
            series[name].append(float(rd.u[pos]))
    sigs, traces = {}, {}
    for name in PROBES:
        s = np.array(series[name])
        idx = np.nonzero((s[:-1] <= U_THRESH) & (s[1:] > U_THRESH))[0]
        sigs[name] = {
            'first_arrival': float(idx[0] * DT) if len(idx) else None,
            'n_crossings': len(idx), 'peak': float(s.max())}
        traces[name] = s
    return sigs, traces


def main():
    # --- step 1: T_flash calibration (A only) -------------------------------
    print('--- T_flash calibration (A only) ---')
    calib = {}
    for t_flash in (4.0, 8.0, 12.0):
        rd = make_rd()
        flash(rd, A_POS, 0, t_flash)
        # count waves emitted: crossings at a point just right of the spot
        series = []
        for _ in range(int(TU_RUN / DT)):
            rd.run(1)
            series.append(float(rd.u[60, 60]))
        series = np.array(series)
        n = int(np.sum((series[:-1] <= U_THRESH) & (series[1:] > U_THRESH)))
        calib[t_flash] = n
        print(f'  T_flash={t_flash} t.u. -> {n} wave(s) emitted', flush=True)
    t_flash = max((t for t, n in calib.items() if n == 1), default=8.0)
    print(f'  chosen T_flash = {t_flash} t.u.')

    # --- step 2: four patterns at uniform phi0 -------------------------------
    print('--- baseline patterns (uniform phi0) ---')
    results, all_traces = {}, {}
    for tag, fa, fb in [('00', False, False), ('10', True, False),
                        ('01', False, True), ('11', True, True)]:
        sigs, traces = run_pattern(fa, fb, t_flash)
        results[tag] = sigs
        all_traces[tag] = traces
        line = ' | '.join(
            f"{k}: arr={sigs[k]['first_arrival']} n={sigs[k]['n_crossings']} "
            f"pk={sigs[k]['peak']:.3f}" for k in PROBES)
        print(f'  ({tag}): {line}', flush=True)

    out = {'config': {'box': [NX, NY], 'A': A_POS, 'B': B_POS,
                      'probes': PROBES, 'spot_r': SPOT_R, 'phi0': PHI0,
                      'dark': DARK, 't_flash': t_flash,
                      'tu_run': TU_RUN, 'u_thresh': U_THRESH,
           'design_region_xy': ((55, 135), (20, 180))},
           'calibration': calib, 'baseline': results}
    with open(os.path.join(FIG, 'rd_spot_xor_protocol.json'), 'w') as fh:
        json.dump(out, fh, indent=2)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (tag, traces) in zip(axes.ravel(), all_traces.items()):
        for name, series in traces.items():
            ax.plot(np.arange(len(series)) * DT, series, label=name)
        ax.set_title(f'pattern ({tag})')
        ax.set_ylim(-0.05, 1.0)
        ax.legend(fontsize=8)
    axes[1, 0].set_xlabel('t (t.u.)')
    axes[1, 1].set_xlabel('t (t.u.)')
    plt.suptitle('rd_spot_xor baseline probe traces (uniform phi0)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'rd_spot_xor_protocol.png'), dpi=150)
    print('saved rd_spot_xor_protocol.{json,png}')


if __name__ == '__main__':
    main()
