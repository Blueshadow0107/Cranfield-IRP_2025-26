"""
rd_xor_chamber_sweep.py -- sweep chamber radius for collision-XOR.

Reuses the geometry and physics of rd_xor_chamber_pde.py and runs a small
parameter sweep to find a chamber size that suppresses the 11 leak.

Outputs:
    Analysis/figures/rd_xor_chamber_sweep.json
    Analysis/figures/rd_xor_chamber_sweep.png
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rd_core import RDSubstrate

# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------
NX = NY = 256
DT = 0.05
F = 1.4375
EPS = 0.05014844822490394
PHI0 = 0.010
DARK = 0.002
U_STAR = 0.0030821
U_THRESH = 0.5
T_FLASH = 3.0
FLASH_STEPS = int(T_FLASH / DT)
T_TOTAL = 60.0
NSTEPS = int(T_TOTAL / DT)

HW = 10
A_END = (60, 210)
B_END = (196, 210)
STEM_END = (128, 20)
SRC_LEN = 18
CHAMBER_C = (128, 120)

PROBE_OUT = np.zeros((NX, NY), bool)
PROBE_OUT[CHAMBER_C[0] - 2:CHAMBER_C[0] + 3, 50:56] = True

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG, exist_ok=True)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def seg_mask(p0, p1, half_w):
    X, Y = np.mgrid[0:NX, 0:NY].astype(float)
    d = np.array(p1, float) - np.array(p0, float)
    t = np.clip(((X - p0[0]) * d[0] + (Y - p0[1]) * d[1]) / (d @ d), 0.0, 1.0)
    cx = p0[0] + t * d[0]
    cy = p0[1] + t * d[1]
    return ((X - cx) ** 2 + (Y - cy) ** 2) <= half_w ** 2


def disk_mask(c, r):
    X, Y = np.mgrid[0:NX, 0:NY].astype(float)
    return ((X - c[0]) ** 2 + (Y - c[1]) ** 2) <= r ** 2


def src_mask(arm, end):
    X, Y = np.mgrid[0:NX, 0:NY].astype(float)
    d = np.array(CHAMBER_C, float) - np.array(end, float)
    t = ((X - end[0]) * d[0] + (Y - end[1]) * d[1]) / (d @ d)
    return arm & (t * np.sqrt(d @ d) <= SRC_LEN)


def first_crossing(series, dt, thresh=U_THRESH):
    idx = np.nonzero(series >= thresh)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    if i == 0:
        return 0.0
    s0, s1 = series[i - 1], series[i]
    frac = (thresh - s0) / (s1 - s0) if s1 > s0 else 0.0
    return (i - 1 + frac) * dt


def build_geometry(radius):
    arm_a = seg_mask(CHAMBER_C, A_END, HW)
    arm_b = seg_mask(CHAMBER_C, B_END, HW)
    stem = seg_mask(CHAMBER_C, STEM_END, HW)
    chamber = disk_mask(CHAMBER_C, radius)
    channel = arm_a | arm_b | stem | chamber
    wall = ~channel
    src_a = src_mask(arm_a, A_END)
    src_b = src_mask(arm_b, B_END)
    probe_out = PROBE_OUT & stem
    return wall, src_a, src_b, probe_out, channel


def run_pattern(radius, fire_a, fire_b):
    wall, src_a, src_b, probe_out, channel = build_geometry(radius)
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, kinetics='oregonator',
                     f=F, eps=EPS, clamp_rest=(U_STAR, U_STAR))
    rd.set_phi(np.full((NX, NY), PHI0))
    rd.set_walls(wall)
    rd.u[channel] = U_STAR
    rd.v[channel] = U_STAR
    rd.add_probe('out', probe_out)

    base = np.full((NX, NY), PHI0)
    out_series = []
    for step in range(NSTEPS):
        phi = base.copy()
        if fire_a and step < FLASH_STEPS:
            phi[src_a] = DARK
        if fire_b and step < FLASH_STEPS:
            phi[src_b] = DARK
        rd.set_phi(phi)
        rd.run(1)
        out_series.append(float(rd.u[probe_out].max()))

    out = np.array(out_series)
    return {
        'out_arrival_tu': first_crossing(out, DT),
        'out_peak': float(out.max()),
        'out_fired': bool(out.max() >= U_THRESH),
    }


def run_radius(radius):
    print(f'[radius {radius}] ...', flush=True)
    patterns = [
        ('00', False, False),
        ('01', False, True),
        ('10', True, False),
        ('11', True, True),
    ]
    res = {}
    for label, fa, fb in patterns:
        res[label] = run_pattern(radius, fa, fb)
    xor_quality = (res['01']['out_fired'] and res['10']['out_fired']
                   and not res['11']['out_fired'] and not res['00']['out_fired'])
    print(f"    01/10 fire, 11 fires={res['11']['out_fired']}, "
          f"XOR clean={xor_quality}", flush=True)
    return res


def main():
    radii = [25, 35, 45, 55]
    results = {r: run_radius(r) for r in radii}

    # summary plot
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(radii))
    width = 0.2
    patterns = ['00', '01', '10', '11']
    for i, pat in enumerate(patterns):
        peaks = [results[r][pat]['out_peak'] for r in radii]
        ax.bar(x + i * width, peaks, width, label=pat)
    ax.axhline(U_THRESH, color='r', ls='--', lw=1, label='threshold')
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([str(r) for r in radii])
    ax.set_xlabel('chamber radius (cells)')
    ax.set_ylabel('output probe peak u')
    ax.set_title('Collision-XOR output peak vs chamber radius')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'rd_xor_chamber_sweep.png'), dpi=140)

    out = {
        'radii': radii,
        'physics': {'f': F, 'eps': EPS, 'phi0': PHI0, 'phi_dark': DARK,
                    'u_star': U_STAR, 'dt': DT, 't_flash_tu': T_FLASH,
                    't_total_tu': T_TOTAL, 'hw': HW},
        'results': results,
    }
    with open(os.path.join(FIG, 'rd_xor_chamber_sweep.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    print('[done] figures/rd_xor_chamber_sweep.{png,json}', flush=True)


if __name__ == '__main__':
    main()
