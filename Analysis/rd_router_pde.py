"""
rd_router_pde.py -- PDE validation of a 2-to-1 priority router / pulse arbiter.

Geometry: two input arms of different lengths merge into a single output stem.
Because arm A is shorter than arm B, A reaches the junction first and makes it
refractory; a coincident B pulse is blocked.  This matches the graph model in
rd_router_graph.py.

Outputs:
    Analysis/figures/rd_router_pde.{png,json}
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rd_core import RDSubstrate

# ---------------------------------------------------------------------------
# Physics and domain
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
T_TOTAL = 70.0
NSTEPS = int(T_TOTAL / DT)

HW = 10                      # channel half-width
JUNCTION = (50, 128)
A_END = (50, 68)             # arm A length 60 cells
B_END = (50, 198)            # arm B length 70 cells
STEM_END = (200, 128)
SRC_LEN = 18

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG, exist_ok=True)


# ---------------------------------------------------------------------------
# Geometry masks
# ---------------------------------------------------------------------------
def seg_mask(p0, p1, half_w):
    X, Y = np.mgrid[0:NX, 0:NY].astype(float)
    d = np.array(p1, float) - np.array(p0, float)
    t = np.clip(((X - p0[0]) * d[0] + (Y - p0[1]) * d[1]) / (d @ d), 0.0, 1.0)
    cx = p0[0] + t * d[0]
    cy = p0[1] + t * d[1]
    return ((X - cx) ** 2 + (Y - cy) ** 2) <= half_w ** 2


ARM_A = seg_mask(JUNCTION, A_END, HW)
ARM_B = seg_mask(JUNCTION, B_END, HW)
STEM = seg_mask(JUNCTION, STEM_END, HW)
CHANNEL = ARM_A | ARM_B | STEM
WALL = ~CHANNEL


def src_mask(arm, end):
    X, Y = np.mgrid[0:NX, 0:NY].astype(float)
    d = np.array(JUNCTION, float) - np.array(end, float)
    t = ((X - end[0]) * d[0] + (Y - end[1]) * d[1]) / (d @ d)
    return arm & (t * np.sqrt(d @ d) <= SRC_LEN)


SRC_A = src_mask(ARM_A, A_END)
SRC_B = src_mask(ARM_B, B_END)

PROBE_OUT = np.zeros((NX, NY), bool)
PROBE_OUT[180:183, 126:130] = True
PROBE_OUT &= STEM

SNAPS = (150, 400, 700, 1000, 1300)


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------
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


def make_rd():
    rd = RDSubstrate(nx=NX, ny=NY, dt=DT, kinetics='oregonator',
                     f=F, eps=EPS, clamp_rest=(U_STAR, U_STAR))
    rd.set_phi(np.full((NX, NY), PHI0))
    rd.set_walls(WALL)
    rd.u[CHANNEL] = U_STAR
    rd.v[CHANNEL] = U_STAR
    rd.add_probe('out', PROBE_OUT)
    return rd


def run_case(fire_a, fire_b):
    rd = make_rd()
    base = np.full((NX, NY), PHI0)
    series = {'out': []}
    snaps = {}
    for step in range(NSTEPS):
        phi = base.copy()
        if fire_a and step < FLASH_STEPS:
            phi[SRC_A] = DARK
        if fire_b and step < FLASH_STEPS:
            phi[SRC_B] = DARK
        rd.set_phi(phi)
        rd.run(1)
        series['out'].append(float(rd.u[PROBE_OUT].max()))
        if step + 1 in SNAPS:
            snaps[step + 1] = rd.u.copy()
    for s in SNAPS:
        if s not in snaps:
            snaps[s] = np.zeros((NX, NY))
    t = np.arange(len(series['out'])) * DT
    return snaps, np.array(series['out']), t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cases = [
        ('00', False, False),
        ('01', False, True),
        ('10', True, False),
        ('11', True, True),
    ]

    results = {}
    all_series = {}
    all_snaps = {}

    for label, fa, fb in cases:
        print(f'[run] {label} ...', flush=True)
        snaps, out, t = run_case(fa, fb)
        all_series[label] = out
        all_snaps[label] = snaps

        arrival = first_crossing(out, DT)
        peak = float(out.max())
        results[label] = {
            'out_arrival_tu': arrival,
            'out_peak': peak,
            'out_count': int(np.sum((out[:-1] < U_THRESH) & (out[1:] >= U_THRESH))),
            'out_fired': arrival is not None,
        }
        print(f"        peak {peak:.3f}, arrival {arrival}, count {results[label]['out_count']}",
              flush=True)

    # ---- figure ----
    fig, axes = plt.subplots(len(cases), len(SNAPS) + 1, figsize=(16, 10))
    wall_rgba = np.ones((NX, NY, 4))
    wall_rgba[WALL, :3] = 0.55

    for row, (label, _, _) in enumerate(cases):
        for col, s in enumerate(SNAPS):
            ax = axes[row, col]
            u = all_snaps[label][s].T
            ax.imshow(u, origin='lower', cmap='viridis', vmin=0, vmax=1)
            ax.imshow(np.ma.masked_where(~WALL.T, WALL.T), origin='lower',
                      cmap=matplotlib.colors.ListedColormap(['#8b98a8']),
                      alpha=0.9)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f't = {s * DT:.1f} t.u.', fontsize=9)
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
        ax = axes[row, -1]
        ax.plot(t, all_series[label], lw=1)
        ax.axhline(U_THRESH, color='r', ls=':', lw=0.8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title('out probe', fontsize=9)
        ax.tick_params(labelsize=7)

    plt.suptitle('2-to-1 priority router (Oregonator A, dark-spot inputs)',
                 fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(os.path.join(FIG, 'rd_router_pde.png'), dpi=140)

    out = {
        'geometry': {
            'nx': NX, 'ny': NY, 'hw': HW,
            'junction': JUNCTION, 'a_end': A_END, 'b_end': B_END,
            'stem_end': STEM_END, 'src_len': SRC_LEN,
        },
        'physics': {
            'f': F, 'eps': EPS, 'phi0': PHI0, 'phi_dark': DARK,
            'u_star': U_STAR, 'dt': DT, 't_flash_tu': T_FLASH,
            't_total_tu': T_TOTAL,
        },
        'cases': results,
    }
    with open(os.path.join(FIG, 'rd_router_pde.json'), 'w') as fh:
        json.dump(out, fh, indent=2)

    print('[done] figures/rd_router_pde.{png,json}', flush=True)


if __name__ == '__main__':
    main()
