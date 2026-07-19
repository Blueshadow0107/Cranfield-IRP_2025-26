"""
Visualise all 6 validation tests as pressure-field snapshots.
Run: python3 visualize_tests.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

# ---------------------------------------------------------------------------
# Physics & grid
# ---------------------------------------------------------------------------
NX, NY = 400, 50
DX = 0.002
C0 = 343.0
RHO0 = 1.225

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

SOURCE_TYPE = 'none'
PULSE_TYPE = 'gaussian'
SRC_I0, SRC_I1 = 3, 6
SRC_AMP = 1.0
SRC_FREQ = 5000.0
PULSE_T0 = 0.0005
PULSE_TAU = 0.0001

FIG_DIR = Path(__file__).parent / 'figures'
FIG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Solver functions
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
# Run with snapshot capture
# ---------------------------------------------------------------------------
def run_with_snapshots(duration, snapshot_times):
    snapshots = {}
    targets = sorted(snapshot_times)
    idx = 0
    n_steps = int(duration / dt) + 1

    for n in range(n_steps):
        t = n * dt
        step(t)
        if idx < len(targets) and t >= targets[idx]:
            snapshots[targets[idx]] = p.copy()
            idx += 1
    return snapshots


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------
def plot_test_snapshots(title, snapshots, filename):
    n = len(snapshots)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 4.0, rows * 2.3))
    gs = fig.add_gridspec(rows, cols, hspace=0.35, wspace=0.25)

    x_mm = np.arange(NX) * DX * 1000
    y_mm = np.arange(NY) * DX * 1000
    X, Y = np.meshgrid(x_mm, y_mm, indexing='ij')

    all_vals = np.concatenate([s.ravel() for s in snapshots.values()])
    vmax = np.max(np.abs(all_vals)) if len(all_vals) > 0 else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    axes = []
    for idx, (t_snap, field) in enumerate(snapshots.items()):
        ax = fig.add_subplot(gs[idx // cols, idx % cols])
        im = ax.pcolormesh(X, Y, field, shading='auto', cmap='RdBu_r', norm=norm)
        ax.set_aspect('equal')
        ax.set_xlim(0, NX * DX * 1000)
        ax.set_ylim(0, NY * DX * 1000)
        ax.set_title(f't = {t_snap * 1000:.2f} ms')
        ax.set_xlabel('x (mm)')
        if idx % cols == 0:
            ax.set_ylabel('y (mm)')
        axes.append(ax)

    fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.75, label='Pressure p')
    fig.suptitle(title, fontsize=11, y=1.02)

    outfile = FIG_DIR / filename
    fig.savefig(outfile, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outfile}")


# ---------------------------------------------------------------------------
# Test 01: Pulse transit
# ---------------------------------------------------------------------------
def viz_test_01():
    print("\nVisualising Test 01: Pulse Transit")
    global BC_LEFT, BC_RIGHT, BC_TOP, BC_BOTTOM, SOURCE_TYPE, PULSE_TYPE, PULSE_T0, SRC_I0, SRC_I1
    BC_LEFT = BC_RIGHT = BC_TOP = BC_BOTTOM = 'absorbing'
    SOURCE_TYPE = 'pulse'
    PULSE_TYPE = 'gaussian'
    PULSE_T0 = 0.0005
    SRC_I0, SRC_I1 = 3, 6

    reset_state()
    snaps = run_with_snapshots(0.020, [0.0006, 0.0010, 0.0014, 0.0018, 0.0022, 0.0028, 0.00032])
    plot_test_snapshots('Test 01: Pulse Transit (absorbing BCs)', snaps, 'viz_test_01.png')


# ---------------------------------------------------------------------------
# Test 02: Hard wall reflection
# ---------------------------------------------------------------------------
def viz_test_02():
    print("\nVisualising Test 02: Hard Wall Reflection")
    global BC_LEFT, BC_RIGHT, BC_TOP, BC_BOTTOM, SOURCE_TYPE, SRC_FREQ, SRC_AMP, SRC_I0, SRC_I1

    freq, amp = 5000.0, 1.0
    src_i0, src_i1 = 197, 203
    SRC_AMP = amp
    SRC_I0, SRC_I1 = src_i0, src_i1
    SOURCE_TYPE = 'continuous_sine'
    SRC_FREQ = freq
    BC_TOP = BC_BOTTOM = 'hard_wall'

    # Run A: hard wall
    BC_LEFT = 'hard_wall'
    BC_RIGHT = 'absorbing'
    reset_state()
    # Run to steady state then capture
    n = int(0.010 / dt)
    discard = int(0.6 * n)
    for step_n in range(n):
        step(step_n * dt)
    snap_hard = p.copy()

    # Run B: absorbing
    BC_LEFT = 'absorbing'
    reset_state()
    for step_n in range(n):
        step(step_n * dt)
    snap_abs = p.copy()

    fig = plt.figure(figsize=(9, 3.5))
    gs = fig.add_gridspec(1, 2, wspace=0.25)

    x_mm = np.arange(NX) * DX * 1000
    y_mm = np.arange(NY) * DX * 1000
    X, Y = np.meshgrid(x_mm, y_mm, indexing='ij')

    vmax = max(np.max(np.abs(snap_hard)), np.max(np.abs(snap_abs)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.pcolormesh(X, Y, snap_hard, shading='auto', cmap='RdBu_r', norm=norm)
    ax0.set_aspect('equal')
    ax0.set_title('Hard wall (left)')
    ax0.set_xlabel('x (mm)')
    ax0.set_ylabel('y (mm)')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.pcolormesh(X, Y, snap_abs, shading='auto', cmap='RdBu_r', norm=norm)
    ax1.set_aspect('equal')
    ax1.set_title('Absorbing (left)')
    ax1.set_xlabel('x (mm)')

    fig.colorbar(im, ax=[ax0, ax1], orientation='vertical', shrink=0.8, label='Pressure p')
    fig.suptitle('Test 02: Hard Wall vs Absorbing (steady state)', fontsize=11, y=1.05)

    outfile = FIG_DIR / 'viz_test_02.png'
    fig.savefig(outfile, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outfile}")


# ---------------------------------------------------------------------------
# Test 03: Pressure-release resonator
# ---------------------------------------------------------------------------
def viz_test_03():
    print("\nVisualising Test 03: Pressure-Release Boundary")
    global BC_LEFT, BC_RIGHT, BC_TOP, BC_BOTTOM, SOURCE_TYPE
    BC_LEFT = 'hard_wall'
    BC_RIGHT = 'pressure_release'
    BC_TOP = BC_BOTTOM = 'hard_wall'
    SOURCE_TYPE = 'none'

    reset_state()
    x_idx = np.arange(NX)
    y_idx = np.arange(NY)
    Xg, Yg = np.meshgrid(x_idx, y_idx, indexing='ij')
    p[:, :] = np.exp(-((Xg - NX // 2)**2 + (Yg - NY // 2)**2) / (2 * 5**2))

    snaps = run_with_snapshots(0.050, [0.001, 0.005, 0.010, 0.020, 0.035, 0.04])
    plot_test_snapshots('Test 03: Quarter-Wave Ring-Down (p=0 at right)', snaps, 'viz_test_03.png')


# ---------------------------------------------------------------------------
# Test 04: Eigenfrequencies
# ---------------------------------------------------------------------------
def viz_test_04():
    print("\nVisualising Test 04: Eigenfrequencies")
    global BC_LEFT, BC_RIGHT, BC_TOP, BC_BOTTOM, SOURCE_TYPE
    BC_LEFT = BC_RIGHT = BC_TOP = BC_BOTTOM = 'hard_wall'
    SOURCE_TYPE = 'none'

    reset_state()
    x_idx = np.arange(NX)
    y_idx = np.arange(NY)
    Xg, Yg = np.meshgrid(x_idx, y_idx, indexing='ij')
    p[:, :] = np.exp(-((Xg - NX // 2)**2 + (Yg - NY // 2)**2) / (2 * 5**2))

    snaps = run_with_snapshots(0.050, [0.001, 0.005, 0.010, 0.020, 0.035, 0.045])
    plot_test_snapshots('Test 04: Closed Cavity Ring-Down (all hard walls)', snaps, 'viz_test_04.png')


# ---------------------------------------------------------------------------
# Test 05: Energy conservation
# ---------------------------------------------------------------------------
def viz_test_05():
    print("\nVisualising Test 05: Energy Conservation")
    global BC_LEFT, BC_RIGHT, BC_TOP, BC_BOTTOM, SOURCE_TYPE
    BC_LEFT = BC_RIGHT = BC_TOP = BC_BOTTOM = 'hard_wall'
    SOURCE_TYPE = 'none'

    reset_state()
    x_idx = np.arange(NX)
    y_idx = np.arange(NY)
    Xg, Yg = np.meshgrid(x_idx, y_idx, indexing='ij')
    p[:, :] = np.exp(-((Xg - NX // 2)**2 + (Yg - NY // 2)**2) / (2 * 5**2))

    snaps = run_with_snapshots(0.020, [0.000, 0.002, 0.005, 0.010, 0.015, 0.02, 0.025])
    plot_test_snapshots('Test 05: Energy Conservation (Gaussian blob oscillating)', snaps, 'viz_test_05.png')


# ---------------------------------------------------------------------------
# Test 06: Phase velocity
# ---------------------------------------------------------------------------
def viz_test_06():
    print("\nVisualising Test 06: Phase Velocity")
    global BC_LEFT, BC_RIGHT, BC_TOP, BC_BOTTOM, SOURCE_TYPE, SRC_FREQ, SRC_I0, SRC_I1
    BC_LEFT = BC_RIGHT = 'absorbing'
    BC_TOP = BC_BOTTOM = 'hard_wall'
    SOURCE_TYPE = 'continuous_sine'
    SRC_I0, SRC_I1 = 3, 6

    freqs_test = [2000.0, 5000.0, 10000.0]
    fig = plt.figure(figsize=(12, 3.5))
    gs = fig.add_gridspec(1, 3, wspace=0.25)

    x_mm = np.arange(NX) * DX * 1000
    y_mm = np.arange(NY) * DX * 1000
    X, Y = np.meshgrid(x_mm, y_mm, indexing='ij')

    axes = []
    for idx, freq in enumerate(freqs_test):
        SRC_FREQ = freq
        reset_state()
        n = int(0.015 / dt)
        discard = int(0.6 * n)
        for step_n in range(n):
            step(step_n * dt)

        ax = fig.add_subplot(gs[0, idx])
        vmax = np.max(np.abs(p))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.pcolormesh(X, Y, p, shading='auto', cmap='RdBu_r', norm=norm)
        ax.set_aspect('equal')
        ax.set_title(f'f = {freq/1000:.1f} kHz')
        ax.set_xlabel('x (mm)')
        if idx == 0:
            ax.set_ylabel('y (mm)')
        axes.append(ax)

    fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.8, label='Pressure p')
    fig.suptitle('Test 06: Steady-State Sine Waves (wavelength comparison)', fontsize=11, y=1.05)

    outfile = FIG_DIR / 'viz_test_06.png'
    fig.savefig(outfile, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outfile}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Initialising solver for test visualisations...")
    set_c_field(np.full((NX, NY), C0))

    viz_test_01()
    viz_test_02()
    viz_test_03()
    viz_test_04()
    viz_test_05()
    viz_test_06()

    print("\nAll test visualisations saved to figures/")
