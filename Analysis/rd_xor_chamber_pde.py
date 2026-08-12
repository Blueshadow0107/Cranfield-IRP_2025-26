"""
rd_xor_chamber_pde.py -- PDE validation of collision-XOR with a chamber.

Geometry: two input arms merge into a circular chamber, then a single output
stem.  Inputs are dark-spot flashes (phi 0.010 -> 0.002 for 3.0 t.u.) at the
ends of each arm.  Walls are physical no-flux masks, not high-phi regions.

The goal is to measure output arrival times for patterns 00, 01, 10, 11 and
map them onto the event-driven graph simulator in rd_graph_sim.py.

Outputs:
    Analysis/figures/rd_xor_chamber_pde.{png,json}
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
T_FLASH = 3.0                 # t.u. -> single pulse
FLASH_STEPS = int(T_FLASH / DT)
T_TOTAL = 60.0                # t.u.
NSTEPS = int(T_TOTAL / DT)

# Geometry
HW = 10                       # channel half-width -> W = 20
CHAMBER_C = (128, 120)        # chamber centre
CHAMBER_R = 25                # chamber radius
A_END = (60, 210)
B_END = (196, 210)
STEM_END = (128, 20)
SRC_LEN = 18                  # dark slab length along each arm

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG, exist_ok=True)


# ---------------------------------------------------------------------------
# Geometry masks
# ---------------------------------------------------------------------------
def seg_mask(p0, p1, half_w):
    """Cells within half_w (perpendicular) of segment p0->p1."""
    X, Y = np.mgrid[0:NX, 0:NY].astype(float)
    d = np.array(p1, float) - np.array(p0, float)
    t = np.clip(((X - p0[0]) * d[0] + (Y - p0[1]) * d[1]) / (d @ d), 0.0, 1.0)
    cx = p0[0] + t * d[0]
    cy = p0[1] + t * d[1]
    return ((X - cx) ** 2 + (Y - cy) ** 2) <= half_w ** 2


def disk_mask(c, r):
    X, Y = np.mgrid[0:NX, 0:NY].astype(float)
    return ((X - c[0]) ** 2 + (Y - c[1]) ** 2) <= r ** 2


ARM_A = seg_mask(CHAMBER_C, A_END, HW)
ARM_B = seg_mask(CHAMBER_C, B_END, HW)
STEM = seg_mask(CHAMBER_C, STEM_END, HW)
CHAMBER = disk_mask(CHAMBER_C, CHAMBER_R)
CHANNEL = ARM_A | ARM_B | STEM | CHAMBER
WALL = ~CHANNEL


def src_mask(arm, end):
    """Dark slab: first SRC_LEN cells of the arm measured from its end."""
    X, Y = np.mgrid[0:NX, 0:NY].astype(float)
    d = np.array(CHAMBER_C, float) - np.array(end, float)
    t = ((X - end[0]) * d[0] + (Y - end[1]) * d[1]) / (d @ d)
    return arm & (t * np.sqrt(d @ d) <= SRC_LEN)


SRC_A = src_mask(ARM_A, A_END)
SRC_B = src_mask(ARM_B, B_END)

# Probes: chamber and output stem
PROBE_CHAMBER = np.zeros((NX, NY), bool)
PROBE_CHAMBER[CHAMBER_C[0] - 5:CHAMBER_C[0] + 6,
               CHAMBER_C[1] - 5:CHAMBER_C[1] + 6] = True
PROBE_CHAMBER &= CHAMBER

PROBE_OUT = np.zeros((NX, NY), bool)
PROBE_OUT[CHAMBER_C[0] - 2:CHAMBER_C[0] + 3, 50:56] = True
PROBE_OUT &= STEM

SNAPS = (100, 250, 400, 600, 900)


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
    rd.add_probe('chamber', PROBE_CHAMBER)
    rd.add_probe('out', PROBE_OUT)
    return rd


def run_case(fire_a, fire_b, offset_b=0):
    """Run one input pattern.  B flash starts at offset_b steps."""
    rd = make_rd()
    base = np.full((NX, NY), PHI0)
    series = {n: [] for n in ('chamber', 'out')}
    snaps = {}
    for step in range(NSTEPS):
        phi = base.copy()
        if fire_a and step < FLASH_STEPS:
            phi[SRC_A] = DARK
        if fire_b and offset_b <= step < offset_b + FLASH_STEPS:
            phi[SRC_B] = DARK
        rd.set_phi(phi)
        rd.run(1)
        series['chamber'].append(float(rd.u[PROBE_CHAMBER].max()))
        series['out'].append(float(rd.u[PROBE_OUT].max()))
        if step + 1 in SNAPS:
            snaps[step + 1] = rd.u.copy()
    for s in SNAPS:
        if s not in snaps:
            snaps[s] = np.zeros((NX, NY))
    t = np.arange(len(series['out'])) * DT
    return snaps, {k: np.array(v) for k, v in series.items()}, t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cases = [
        ('00', False, False, 0),
        ('01', False, True, 0),
        ('10', True, False, 0),
        ('11', True, True, 0),
    ]

    results = {}
    all_series = {}
    all_snaps = {}

    for label, fa, fb, off in cases:
        print(f'[run] {label} ...', flush=True)
        snaps, series, t = run_case(fa, fb, off)
        all_series[label] = series
        all_snaps[label] = snaps

        out_cross = first_crossing(series['out'], DT)
        chamber_cross = first_crossing(series['chamber'], DT)
        out_peak = float(series['out'].max())
        chamber_peak = float(series['chamber'].max())

        results[label] = {
            'out_arrival_tu': out_cross,
            'chamber_arrival_tu': chamber_cross,
            'out_peak': out_peak,
            'chamber_peak': chamber_peak,
            'out_fired': out_cross is not None,
        }
        print(f"        chamber peak {chamber_peak:.3f}, out peak {out_peak:.3f}, "
              f"out arrival {out_cross}", flush=True)

    # ---- figure ----
    fig, axes = plt.subplots(len(cases), len(SNAPS) + 1,
                             figsize=(16, 10))
    if len(cases) == 1:
        axes = axes.reshape(1, -1)
    wall_rgba = np.ones((NX, NY, 4))
    wall_rgba[WALL, :3] = 0.55

    for row, (label, _, _, _) in enumerate(cases):
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
        # trace panel
        ax = axes[row, -1]
        ax.plot(t, all_series[label]['chamber'], label='chamber', lw=1)
        ax.plot(t, all_series[label]['out'], label='out', lw=1)
        ax.axhline(U_THRESH, color='r', ls=':', lw=0.8)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc='upper right')
        ax.set_title('probe traces', fontsize=9)
        ax.tick_params(labelsize=7)

    plt.suptitle('Collision-XOR with chamber (Oregonator A, dark-spot inputs)',
                 fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(os.path.join(FIG, 'rd_xor_chamber_pde.png'), dpi=140)

    out = {
        'geometry': {
            'nx': NX, 'ny': NY, 'hw': HW,
            'chamber_c': CHAMBER_C, 'chamber_r': CHAMBER_R,
            'a_end': A_END, 'b_end': B_END, 'stem_end': STEM_END,
            'src_len': SRC_LEN,
        },
        'physics': {
            'f': F, 'eps': EPS, 'phi0': PHI0, 'phi_dark': DARK,
            'u_star': U_STAR, 'dt': DT, 't_flash_tu': T_FLASH,
            't_total_tu': T_TOTAL,
        },
        'cases': results,
    }
    with open(os.path.join(FIG, 'rd_xor_chamber_pde.json'), 'w') as fh:
        json.dump(out, fh, indent=2)

    print('[done] figures/rd_xor_chamber_pde.{png,json}', flush=True)


if __name__ == '__main__':
    main()
