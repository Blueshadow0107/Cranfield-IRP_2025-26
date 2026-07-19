"""
Visualise Bragg diffraction / reflection from a periodic thermal grating.
Shows: incident wave, reflected wave, transmitted wave, and standing-wave buildup.
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

# Bragg design for 8 kHz
F_BLOCK = 8000.0
LAMBDA_BLOCK = C0 / F_BLOCK  # 42.875 mm
D_BRAGG = LAMBDA_BLOCK / 2.0   # 21.4375 mm

STRIP_WIDTH = 0.010
N_PERIODS = 20
START_X = 0.060

T_COLD = 300.0
T_HOT = 600.0
C_COLD = C0
C_HOT = C0 * np.sqrt(T_HOT / T_COLD)

FIG_DIR = Path(__file__).parent / 'figures'
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Build Bragg grating c-field
# ---------------------------------------------------------------------------
def build_bragg_field():
    c = np.ones((NX, NY)) * C_COLD
    period_cells = int(D_BRAGG / DX)
    width_cells = int(STRIP_WIDTH / DX)
    start_i = int(START_X / DX)
    for n in range(N_PERIODS):
        i_s = start_i + n * period_cells
        i_e = i_s + width_cells
        if i_e < NX:
            c[i_s:i_e, :] = C_HOT
    return c


# ---------------------------------------------------------------------------
# FDTD solver (inline for standalone)
# ---------------------------------------------------------------------------
def run_bragg(c_field, source_freq, duration, snapshot_times, src_type='pulse'):
    c2 = c_field ** 2
    c_max = float(c_field.max())
    dt = 0.9 * DX / (c_max * np.sqrt(2))

    p = np.zeros((NX, NY))
    u = np.zeros((NX + 1, NY))
    v = np.zeros((NX, NY + 1))

    # Absorbing on left, hard wall on top/bottom, absorbing on right
    def apply_bc():
        u[0, :] = -p[0, :] / (RHO0 * c_field[0, :])
        u[NX, :] = p[NX - 1, :] / (RHO0 * c_field[NX - 1, :])
        v[:, 0] = -p[:, 0] / (RHO0 * c_field[:, 0])
        v[:, NY] = p[:, NY - 1] / (RHO0 * c_field[:, NY - 1])

    def src_amp(t):
        if src_type == 'pulse':
            tau = 0.0001
            env = np.exp(-((t - 0.0005) / tau) ** 2)
            return np.sin(2 * np.pi * source_freq * (t - 0.0005)) * env
        else:
            return np.sin(2 * np.pi * source_freq * t)

    snapshots = {}
    targets = sorted(snapshot_times)
    t_idx = 0
    n_steps = int(duration / dt) + 1

    for n in range(n_steps):
        t = n * dt
        u[1:NX, :] -= (dt / RHO0) * (p[1:NX, :] - p[0:NX - 1, :]) / DX
        v[:, 1:NY] -= (dt / RHO0) * (p[:, 1:NY] - p[:, 0:NY - 1]) / DX
        apply_bc()
        du = (u[1:NX + 1, :] - u[0:NX, :]) / DX
        dv = (v[:, 1:NY + 1] - v[:, 0:NY]) / DX
        p -= dt * RHO0 * c2 * (du + dv)
        p[3:6, :] += src_amp(t)

        if t_idx < len(targets) and t >= targets[t_idx]:
            snapshots[targets[t_idx]] = p.copy()
            t_idx += 1

    return snapshots, dt


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_bragg(title, snapshots, filename, c_field=None):
    n = len(snapshots)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 4.2, rows * 2.6))
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

        # Overlay grating boundaries
        if c_field is not None:
            ax.contour(X, Y, c_field, levels=[(C_COLD + C_HOT) / 2],
                       colors='lime', linewidths=0.8, alpha=0.7)

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
# Main: compare on-resonance (8 kHz) vs off-resonance (5 kHz)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Building Bragg grating...")
    c_field = build_bragg_field()

    period_cells = int(D_BRAGG / DX)
    width_cells = int(STRIP_WIDTH / DX)
    start_i = int(START_X / DX)
    print(f"  Period: {D_BRAGG*1000:.2f} mm ({period_cells} cells)")
    print(f"  Strips: {N_PERIODS} periods, width {STRIP_WIDTH*1000:.0f} mm")
    print(f"  c_hot = {C_HOT:.1f} m/s, c_cold = {C_COLD:.1f} m/s")
    print(f"  Reflection per interface: {abs((C_HOT - C_COLD)/(C_HOT + C_COLD)):.3f}")

    # Snapshot times tuned for pulse transit + reflection dynamics
    snap_times = [0.0006, 0.0012, 0.0018, 0.0024, 0.0030]

    # --- Off-resonance: 5 kHz (should mostly transmit) ---
    print("\nRunning 5 kHz (off-resonance)...")
    snaps_5k, dt = run_bragg(c_field, 5000.0, 0.020, snap_times, src_type='pulse')
    plot_bragg('Bragg: 5 kHz pulse (off-resonance, mostly transmits)',
               snaps_5k, 'bragg_5kHz.png', c_field)

    # --- On-resonance: 8 kHz (should strongly reflect) ---
    print("\nRunning 8 kHz (on-resonance)...")
    snaps_8k, _ = run_bragg(c_field, 8000.0, 0.020, snap_times, src_type='pulse')
    plot_bragg('Bragg: 8 kHz pulse (on-resonance, strong reflection)',
               snaps_8k, 'bragg_8kHz.png', c_field)

    # --- Steady-state 8 kHz to show standing wave ---
    print("\nRunning 8 kHz continuous (steady-state standing wave)...")
    snaps_8k_ss, _ = run_bragg(c_field, 8000.0, 0.020,
                                [0.003, 0.006, 0.009, 0.012, 0.015],
                                src_type='continuous')
    plot_bragg('Bragg: 8 kHz continuous (standing wave in front of grating)',
               snaps_8k_ss, 'bragg_8kHz_steady.png', c_field)

    # --- Layout figure ---
    fig, ax = plt.subplots(figsize=(12, 2.5))
    im = ax.imshow(c_field.T, origin='lower', aspect='auto',
                   extent=[0, NX*DX*1000, 0, NY*DX*1000], cmap='coolwarm',
                   vmin=C_COLD, vmax=C_HOT)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(f'Bragg Grating Layout: {N_PERIODS} periods, d={D_BRAGG*1000:.1f} mm')
    fig.colorbar(im, ax=ax, label='c (m/s)')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'bragg_layout.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'bragg_layout.png'}")

    print("\nAll Bragg visualisations saved.")
