"""
Wave Propagation Snapshot Visualiser
====================================
Captures the 2D pressure field at multiple time instants and arranges
them as a comic-strip figure.  Useful for verifying pulse shape, BC
behaviour, and interference patterns.

Run:  python3 wave_snapshot.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

# ---------------------------------------------------------------------------
# Physics & grid (same as fdtd_modular.py)
# ---------------------------------------------------------------------------
NX, NY = 400, 50
DX = 0.002
C0 = 343.0
RHO0 = 1.225
T0 = 300.0
D_THERMAL = 2e-5

# State
c_field = None
c2 = None
c_max = None
dt = None
p = None
u = None
v = None

BC_LEFT = 'absorbing'
BC_RIGHT = 'absorbing'
BC_TOP = 'hard_wall'
BC_BOTTOM = 'hard_wall'

SOURCE_TYPE = 'pulse'
PULSE_TYPE = 'gaussian'
SRC_I0, SRC_I1 = 3, 6
SRC_AMP = 1.0
SRC_FREQ = 5000.0
PULSE_T0 = 0.0005
PULSE_TAU = 0.0001

# Pulse starts at PULSE_T0 = 0.5 ms, exits domain by ~2.3 ms
SNAPSHOT_TIMES = [0.0006, 0.0010, 0.0014, 0.0018, 0.0022, 0.0026]  # seconds

FIG_DIR = Path(__file__).parent / 'figures'
FIG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Solver functions (copied inline so this file is standalone)
# ---------------------------------------------------------------------------
def set_c_field(new_c_field):
    global c_field, c2, c_max, dt
    c_field = new_c_field.astype(float)
    c2 = c_field ** 2
    c_max = float(c_field.max())
    dt = 0.9 * DX / (c_max * np.sqrt(2))


def reset_state():
    global p, u, v
    p = np.zeros((NX, NY))
    u = np.zeros((NX + 1, NY))
    v = np.zeros((NX, NY + 1))


def apply_boundary_conditions():
    global u, v
    if BC_LEFT == 'absorbing':
        u[0, :] = -p[0, :] / (RHO0 * c_field[0, :])
    elif BC_LEFT == 'hard_wall':
        u[0, :] = 0.0
    elif BC_LEFT == 'pressure_release':
        u[0, :] -= (dt / RHO0) * 2.0 * p[0, :] / DX

    if BC_RIGHT == 'absorbing':
        u[NX, :] = p[NX - 1, :] / (RHO0 * c_field[NX - 1, :])
    elif BC_RIGHT == 'hard_wall':
        u[NX, :] = 0.0
    elif BC_RIGHT == 'pressure_release':
        u[NX, :] += (dt / RHO0) * 2.0 * p[NX - 1, :] / DX

    if BC_BOTTOM == 'absorbing':
        v[:, 0] = -p[:, 0] / (RHO0 * c_field[:, 0])
    elif BC_BOTTOM == 'hard_wall':
        v[:, 0] = 0.0
    elif BC_BOTTOM == 'pressure_release':
        v[:, 0] -= (dt / RHO0) * 2.0 * p[:, 0] / DX

    if BC_TOP == 'absorbing':
        v[:, NY] = p[:, NY - 1] / (RHO0 * c_field[:, NY - 1])
    elif BC_TOP == 'hard_wall':
        v[:, NY] = 0.0
    elif BC_TOP == 'pressure_release':
        v[:, NY] += (dt / RHO0) * 2.0 * p[:, NY - 1] / DX


def source_amplitude(t):
    if SOURCE_TYPE == 'none':
        return 0.0
    elif SOURCE_TYPE == 'continuous_sine':
        return SRC_AMP * np.sin(2.0 * np.pi * SRC_FREQ * t)
    elif SOURCE_TYPE == 'pulse':
        if PULSE_TYPE == 'gaussian':
            tau = 0.0001
        elif PULSE_TYPE == 'broadband':
            tau = 0.0005
        env = np.exp(-((t - PULSE_T0) / tau) ** 2)
        return SRC_AMP * env * np.sin(2.0 * np.pi * SRC_FREQ * (t - PULSE_T0))
    return 0.0


def step(t):
    global p
    u[1:NX, :] -= (dt / RHO0) * (p[1:NX, :] - p[0:NX - 1, :]) / DX
    v[:, 1:NY] -= (dt / RHO0) * (p[:, 1:NY] - p[:, 0:NY - 1]) / DX
    apply_boundary_conditions()
    du_dx = (u[1:NX + 1, :] - u[0:NX, :]) / DX
    dv_dy = (v[:, 1:NY + 1] - v[:, 0:NY]) / DX
    p -= dt * RHO0 * c2 * (du_dx + dv_dy)
    p[SRC_I0:SRC_I1, :] += source_amplitude(t)


# ---------------------------------------------------------------------------
# Run simulation and capture snapshots
# ---------------------------------------------------------------------------
def run_and_capture(snapshot_times):
    snapshots = {}
    targets = sorted(snapshot_times)
    idx = 0
    n_steps = int(targets[-1] / dt) + 1

    for n in range(n_steps):
        t = n * dt
        step(t)
        if idx < len(targets) and t >= targets[idx]:
            snapshots[targets[idx]] = p.copy()
            idx += 1
    return snapshots


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_snapshots(snapshots):
    n = len(snapshots)
    cols = 3
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 4.2, rows * 2.5))
    gs = fig.add_gridspec(rows, cols, hspace=0.35, wspace=0.25)

    x_mm = np.arange(NX) * DX * 1000  # mm
    y_mm = np.arange(NY) * DX * 1000
    X, Y = np.meshgrid(x_mm, y_mm, indexing='ij')

    # Symmetric colour scale across all snapshots
    all_vals = np.concatenate([s.ravel() for s in snapshots.values()])
    vmax = np.max(np.abs(all_vals))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    axes = []
    for idx, (t_snap, field) in enumerate(snapshots.items()):
        ax = fig.add_subplot(gs[idx // cols, idx % cols])
        im = ax.pcolormesh(X, Y, field, shading='auto', cmap='RdBu_r', norm=norm)
        ax.set_aspect('equal')
        ax.set_xlim(0, NX * DX * 1000)
        ax.set_ylim(0, NY * DX * 1000)
        ax.set_title(f't = {t_snap * 1000:.1f} ms')
        ax.set_xlabel('x (mm)')
        if idx % cols == 0:
            ax.set_ylabel('y (mm)')
        else:
            ax.set_ylabel('')
        axes.append(ax)

    fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.75, label='Pressure p')
    fig.suptitle('Acoustic Pulse Propagation (p-field)', fontsize=12, y=1.02)

    outfile = FIG_DIR / 'wave_snapshots.png'
    fig.savefig(outfile, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outfile}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Initialising wave snapshot run...")
    set_c_field(np.full((NX, NY), C0))
    reset_state()

    print(f"  dt = {dt * 1e6:.2f} µs, {int(SNAPSHOT_TIMES[-1] / dt)} steps total")
    print("  Capturing snapshots...")

    snapshots = run_and_capture(SNAPSHOT_TIMES)
    plot_snapshots(snapshots)

    print("Done.")
