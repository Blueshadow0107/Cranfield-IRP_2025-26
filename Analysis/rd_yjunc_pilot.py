"""rd_yjunc_pilot.py -- Y-junction collision-XOR pilot (dark-spot sources).

Validates the Test-5 geometry before the full transfer-test rerun:
two input arms merge symmetrically into a single output stem. Lone pulse
from either arm diffracts at the merge and propagates down the stem
(probe fires). Coincident pulses annihilate at the merge -- the seam
sweeps across the stem mouth and nothing propagates past it (probe dark
beyond the seam, but a probe AT the seam still sees the arrival spike).

Geometry (distance-to-segment channels, no stair-stepped angles):
  domain 256x256, channel half-width HW=10 (W=20 as the T test),
  junction J=(128,120), arm ends A=(60,210) B=(196,210),
  stem from J down to (128,20).

Sources: dark slabs (phi 0.010 -> 0.002 for T_FLASH=3.0 t.u., the
single-pulse window from rd_flash_calib) across each arm near its top
end (first 18 cells of the arm). Flashes are concurrent: A dark on
[0, fl), B dark on [offset, offset+fl). Probe readout: max over mask.

Runs: A only, B only, A+B coincident, A+B with B offset +5 t.u.
Probes: 'seam' just below the merge, 'stem' 50 cells further down.

Outputs: figures/rd_yjunc_pilot.{png,json}
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rd_core import RDSubstrate

NX = NY = 256
HW = 10                       # channel half-width (W = 20)
J = (128, 120)                # junction
A_END, B_END = (60, 210), (196, 210)
STEM_END = (128, 20)
SRC_LEN = 18                  # dark slab length along arm
DT = 0.05
T_FLASH = 3.0                 # t.u. (60 steps; single-pulse window per
                              # rd_flash_calib: <=4 t.u. -> exactly 1 pulse)
DARK = 0.002
PHI0 = 0.010
U_STAR = 0.0030821
U_THRESH = 0.5
T_TOTAL = 45.0                # t.u.
NSTEPS = int(T_TOTAL / DT)
OFFSET_B = int(5.0 / DT)      # B delay for the offset run (steps)
SNAPS = (150, 320, 420, 700)  # snapshot steps

F, EPS = 1.4375, 0.05014844822490394

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG, exist_ok=True)


def seg_mask(p0, p1, half_w):
    """Cells within half_w (perpendicular) of segment p0->p1."""
    X, Y = np.mgrid[0:NX, 0:NY].astype(float)
    d = np.array(p1, float) - np.array(p0, float)
    t = np.clip(((X - p0[0]) * d[0] + (Y - p0[1]) * d[1])
                / (d @ d), 0.0, 1.0)
    cx = p0[0] + t * d[0]
    cy = p0[1] + t * d[1]
    return ((X - cx) ** 2 + (Y - cy) ** 2) <= half_w ** 2


ARM_A = seg_mask(J, A_END, HW)
ARM_B = seg_mask(J, B_END, HW)
STEM = seg_mask(J, STEM_END, HW)
CHANNEL = ARM_A | ARM_B | STEM
WALL = ~CHANNEL


def src_mask(arm, end):
    """Dark slab: first SRC_LEN cells of the arm measured from its end."""
    X, Y = np.mgrid[0:NX, 0:NY].astype(float)
    d = np.array(J, float) - np.array(end, float)
    t = ((X - end[0]) * d[0] + (Y - end[1]) * d[1]) / (d @ d)
    return arm & (t * np.sqrt(d @ d) <= SRC_LEN)


SRC_A = src_mask(ARM_A, A_END)
SRC_B = src_mask(ARM_B, B_END)

PROBE_SEAM = np.zeros((NX, NY), bool)
PROBE_SEAM[J[0] - 1:J[0] + 2, J[1] - 8:J[1] - 2] = True
PROBE_SEAM &= STEM
PROBE_STEM = np.zeros((NX, NY), bool)
PROBE_STEM[J[0] - 1:J[0] + 2, J[1] - 53:J[1] - 47] = True
PROBE_STEM &= STEM


def make_rd():
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, kinetics='oregonator',
                     f=F, eps=EPS, clamp_rest=(U_STAR, U_STAR))
    rd.set_phi(np.full((NX, NY), PHI0))
    rd.set_walls(WALL)
    rd.u[CHANNEL] = U_STAR
    rd.v[CHANNEL] = U_STAR
    rd.add_probe('seam', PROBE_SEAM)
    rd.add_probe('stem', PROBE_STEM)
    return rd


def run_case_probes(fire_a, fire_b, offset_b=0):
    """Concurrent flashes: A dark during [0, fl), B dark during
    [offset_b, offset_b+fl). Probe readout is max over the mask (a strip
    mean dilutes the 2-3-cell front below threshold)."""
    rd = make_rd()
    fl = int(T_FLASH / DT)
    base = np.full((NX, NY), PHI0)
    series = {n: [] for n in ('seam', 'stem')}
    snaps = {}
    for step in range(NSTEPS):
        phi = base.copy()
        if fire_a and step < fl:
            phi[SRC_A] = DARK
        if fire_b and offset_b <= step < offset_b + fl:
            phi[SRC_B] = DARK
        rd.set_phi(phi)
        rd.run(1)
        series['seam'].append(float(rd.u[PROBE_SEAM].max()))
        series['stem'].append(float(rd.u[PROBE_STEM].max()))
        if step + 1 in SNAPS:
            snaps[step + 1] = rd.u.copy()
    for s in SNAPS:
        if s not in snaps:
            snaps[s] = np.zeros((NX, NY))
    return rd, snaps, {k: np.array(v) for k, v in series.items()}


CASES = [('A only', True, False, 0),
         ('B only', False, True, 0),
         ('A+B coincident', True, True, 0),
         ('A+B offset +5 t.u.', True, True, OFFSET_B)]


def main():
    results = {}
    all_snaps = {}
    all_series = {}
    for name, fa, fb, off in CASES:
        print(f'[run] {name} ...', flush=True)
        rd, snaps, series = run_case_probes(fa, fb, off)
        all_snaps[name] = snaps
        all_series[name] = series
        results[name] = {
            'seam_peak': float(series['seam'].max()),
            'stem_peak': float(series['stem'].max()),
            'stem_crossings': int(np.sum((series['stem'][:-1] < U_THRESH)
                                         & (series['stem'][1:] >= U_THRESH))),
            'stem_fired': bool(series['stem'].max() >= U_THRESH),
        }
        print(f"        seam peak {results[name]['seam_peak']:.3f}, "
              f"stem peak {results[name]['stem_peak']:.3f}, "
              f"stem crossings {results[name]['stem_crossings']}",
              flush=True)

    # ---- figure: snapshot rows for 3 cases + probe traces ----
    show = ['A only', 'A+B coincident', 'A+B offset +5 t.u.']
    fig, axes = plt.subplots(4, len(SNAPS) + 1, figsize=(16, 11))
    wall_rgba = np.ones((NX, NY, 4))
    wall_rgba[WALL, :3] = 0.55
    for row, name in enumerate(show):
        for col, s in enumerate(SNAPS):
            ax = axes[row, col]
            u = all_snaps[name][s].T
            ax.imshow(u, origin='lower', cmap='viridis', vmin=0, vmax=1)
            ax.imshow(np.ma.masked_where(~WALL.T, WALL.T), origin='lower',
                      cmap=matplotlib.colors.ListedColormap(['#8b98a8']),
                      alpha=0.9)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f't = {s * DT:.0f} t.u.', fontsize=9)
            if col == 0:
                ax.set_ylabel(name, fontsize=9)
        # last column: probe traces for this case
        ax = axes[row, -1]
        t = np.arange(len(all_series[name]['stem'])) * DT
        ax.plot(t, all_series[name]['seam'], label='seam', lw=1)
        ax.plot(t, all_series[name]['stem'], label='stem', lw=1)
        ax.axhline(U_THRESH, color='r', ls=':', lw=0.8)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc='upper right')
        ax.set_title('probe traces', fontsize=9)
        ax.tick_params(labelsize=7)
    # bottom row: all four cases' stem traces overlaid in first slot
    for col in range(len(SNAPS) + 1):
        axes[3, col].axis('off')
    ax = axes[3, 0]
    ax.axis('on')
    for name, _, _, _ in CASES:
        t = np.arange(len(all_series[name]['stem'])) * DT
        ax.plot(t, all_series[name]['stem'], label=name, lw=1)
    ax.axhline(U_THRESH, color='r', ls=':', lw=0.8)
    ax.set_title('stem probe, all cases', fontsize=9)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)
    plt.suptitle('Y-junction collision pilot (Oregonator A, dark-slab '
                 'sources, W=20): lone pulses enter the stem, coincident '
                 'pulses annihilate at the merge', fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(os.path.join(FIG, 'rd_yjunc_pilot.png'), dpi=140)

    out = {'geometry': {'nx': NX, 'ny': NY, 'hw': HW, 'junction': J,
                        'a_end': A_END, 'b_end': B_END, 'stem_end': STEM_END},
           'source': {'dark': DARK, 'phi0': PHI0, 't_flash_tu': T_FLASH,
                      'src_len': SRC_LEN},
           'kinetics': 'oregonator_A', 'dt': DT, 't_total_tu': T_TOTAL,
           'offset_b_tu': OFFSET_B * DT, 'cases': results}
    with open(os.path.join(FIG, 'rd_yjunc_pilot.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    print('[done] figures/rd_yjunc_pilot.{png,json}', flush=True)


if __name__ == '__main__':
    main()
