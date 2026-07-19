#!/usr/bin/env python3
"""
acoustic_filter_fdtd.py
=======================
2D first-order acoustic FDTD solver with spatially varying sound speed
c(x,y) from a temperature field. The temperature field is parameterized by
8 rectangular heater blocks (2 rows x 4 columns).

Governing equations (staggered leapfrog):
    dp/dt = -rho0 * c(x,y)^2 * (du/dx + dv/dy) + s(x,y,t)
    du/dt = -(1/rho0) * dp/dx
    dv/dt = -(1/rho0) * dp/dy

Grid (staggered Yee):
    p[i,j]   : cell centers,     i=0..nx-1,  j=0..ny-1
    u[i,j]   : x-faces,          i=0..nx,    j=0..ny-1
    v[i,j]   : y-faces,          i=0..nx-1,  j=0..ny

Boundary conditions:
    Left/Right : absorbing (characteristic)
    Top/Bottom : rigid wall (v=0, Neumann for p)

Source:
    Soft additive pressure source in strip near left boundary.
    Sinusoidal at specified frequency.

Output:
    Time-averaged acoustic intensity at right boundary outlet.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# Physical and numerical parameters
# =============================================================================
NX, NY = 500, 100           # grid cells
LX, LY = 0.5, 0.1           # domain size [m]
DX = LX / NX                # 1.0e-3 m
DY = LY / NY                # 1.0e-3 m

RHO0 = 1.2                  # air density [kg/m^3]
T0 = 300.0                  # reference temperature [K]
C0 = 343.0                  # sound speed at T0 [m/s]
T_MAX = 500.0               # max heater temp [K]
C_MAX = C0 * np.sqrt(T_MAX / T0)

# CFL stability: dt = 0.5 * dx / (c_max * sqrt(2))
CFL_FACTOR = 0.5
DT = CFL_FACTOR * DX / (C_MAX * np.sqrt(2))
print(f"dx = {DX:.3e} m,  dt = {DT:.3e} s,  c0 = {C0} m/s,  c_max = {C_MAX:.1f} m/s")
print(f"CFL = {C_MAX * DT / DX:.4f}  (limit = {1/np.sqrt(2):.4f})")

# =============================================================================
# Heater block geometry (8 blocks: 2 rows x 4 columns)
# =============================================================================
N_BLOCKS = 8
N_ROWS = 2
N_COLS = 4

# Block dimensions in cells
BLOCK_W = 30                # width in x
BLOCK_H = 20                # height in y
GAP_X = 40                  # horizontal gap between blocks
GAP_Y = 20                  # vertical gap between rows
START_X = 100               # starting x cell index
START_Y = 20                # starting y cell index (bottom row)

# Build block masks: list of (x0, x1, y0, y1) for each block
BLOCK_BOUNDS = []
for row in range(N_ROWS):
    for col in range(N_COLS):
        x0 = START_X + col * (BLOCK_W + GAP_X)
        x1 = x0 + BLOCK_W
        y0 = START_Y + row * (BLOCK_H + GAP_Y)
        y1 = y0 + BLOCK_H
        BLOCK_BOUNDS.append((x0, x1, y0, y1))

print(f"\nBlock layout: {N_ROWS} rows x {N_COLS} cols = {N_BLOCKS} blocks")
for k, (x0, x1, y0, y1) in enumerate(BLOCK_BOUNDS):
    print(f"  Block {k}: x=[{x0},{x1}], y=[{y0},{y1}]")


def build_temperature_field(T_blocks, nx=NX, ny=NY):
    """
    Build temperature field T(x,y) from block temperatures.

    Parameters
    ----------
    T_blocks : ndarray(N_BLOCKS,)
        Temperature of each heater block [K].

    Returns
    -------
    T : ndarray(nx, ny)
        Temperature field.
    """
    T = np.full((nx, ny), T0)
    for Tk, (x0, x1, y0, y1) in zip(T_blocks, BLOCK_BOUNDS):
        T[x0:x1, y0:y1] = Tk
    return T


def build_sound_speed(T):
    """Compute sound speed field from temperature field."""
    return C0 * np.sqrt(T / T0)


# =============================================================================
# Source
# =============================================================================
# Soft source strip near left boundary (cells 3,4,5)
SOURCE_CELLS = [3, 4, 5]
SOURCE_AMP = 1.0

# Smooth window: cosine bell from 1 at i=3 to 0 at i=5
_window = np.zeros(NX)
for idx, i in enumerate(SOURCE_CELLS):
    _window[i] = 0.5 * (1.0 - np.cos(np.pi * idx / (len(SOURCE_CELLS) - 1)))


def source_term(t, freq):
    """Return source amplitude at time t for frequency freq."""
    return SOURCE_AMP * np.sin(2.0 * np.pi * freq * t)


# =============================================================================
# FDTD core update
# =============================================================================
def run_fdtd(c, freq, n_steps, record_interval=None):
    """
    Run FDTD simulation.

    Parameters
    ----------
    c : ndarray(NX, NY)
        Sound speed field [m/s].
    freq : float
        Source frequency [Hz].
    n_steps : int
        Number of time steps.
    record_interval : int, optional
        If given, save full field every N steps for animation.

    Returns
    -------
    power : float
        Time-averaged transmitted power at outlet.
    probe_p : ndarray
        Pressure history at outlet center.
    probe_u : ndarray
        u-velocity history at outlet face.
    snapshots : list
        Full p-field snapshots if record_interval is set.
    """
    # Fields
    p = np.zeros((NX, NY))
    u = np.zeros((NX + 1, NY))
    v = np.zeros((NX, NY + 1))

    # Precompute c^2
    c2 = c ** 2

    # Recording
    probe_p = np.zeros(n_steps)
    probe_u = np.zeros(n_steps)
    snapshots = []

    # Time stepping (staggered leapfrog)
    for n in range(n_steps):
        t = n * DT

        # --- Inject soft source into pressure ---
        src = source_term(t, freq)
        for i in SOURCE_CELLS:
            p[i, :] += src * _window[i]

        # --- Update u and v (half-step) ---
        # du/dt = -(1/rho0) * dp/dx
        # dp/dx at u[i,j] = (p[i,j] - p[i-1,j]) / dx
        u[1:NX, :] -= (DT / RHO0) * (p[1:NX, :] - p[0:NX - 1, :]) / DX

        # dv/dt = -(1/rho0) * dp/dy
        # dp/dy at v[i,j] = (p[i,j] - p[i,j-1]) / dy
        v[:, 1:NY] -= (DT / RHO0) * (p[:, 1:NY] - p[:, 0:NY - 1]) / DY

        # --- Update p (integer step) ---
        # du/dx at p[i,j] = (u[i+1,j] - u[i,j]) / dx
        # dv/dy at p[i,j] = (v[i,j+1] - v[i,j]) / dy
        du_dx = (u[1:NX + 1, :] - u[0:NX, :]) / DX
        dv_dy = (v[:, 1:NY + 1] - v[:, 0:NY]) / DY
        p -= DT * RHO0 * c2 * (du_dx + dv_dy)

        # --- Boundary conditions ---
        # Top/Bottom: rigid wall (v = 0)
        v[:, 0] = 0.0
        v[:, NY] = 0.0

        # Left absorbing: u[0,j] = -p[0,j] / (rho0 * c[0,j])
        u[0, :] = -p[0, :] / (RHO0 * c[0, :])

        # Right absorbing: u[nx,j] = p[nx-1,j] / (rho0 * c[nx-1,j])
        u[NX, :] = p[NX - 1, :] / (RHO0 * c[NX - 1, :])

        # --- Record probe ---
        probe_p[n] = p[NX - 1, NY // 2]
        probe_u[n] = u[NX, NY // 2]

        if record_interval and n % record_interval == 0:
            snapshots.append(p.copy())

    # --- Compute transmitted power ---
    # Time-average intensity <p*u> at outlet over last 20% of simulation
    # (after transient has died out)
    discard = int(0.8 * n_steps)
    probe_intensity = probe_p * probe_u   # instantaneous intensity at center
    power_center = np.mean(probe_intensity[discard:])

    # Multiply by outlet area (height * 1 in z) to get total power
    # This assumes intensity is roughly uniform across y; for blocked
    # frequencies this is an approximation.
    power = power_center * LY

    return power, probe_p, probe_u, snapshots


# =============================================================================
# Transfer function evaluation
# =============================================================================
def evaluate_transfer(T_blocks, f1=2000.0, f2=5000.0):
    """
    Evaluate frequency filter performance.

    Parameters
    ----------
    T_blocks : ndarray(N_BLOCKS,)
        Heater block temperatures [K].
    f1, f2 : float
        Target frequencies [Hz].

    Returns
    -------
    power1, power2 : float
        Transmitted power at f1 and f2.
    """
    T = build_temperature_field(T_blocks)
    c = build_sound_speed(T)

    # Determine n_steps from lower frequency (needs more steps)
    period1 = 1.0 / f1
    transit = LX / C0
    n_steps = int(10.0 * transit / DT)  # ~10 transit times
    n_steps = max(n_steps, int(10.0 * period1 / DT))  # at least 10 periods

    power1, _, _, _ = run_fdtd(c, f1, n_steps)
    power2, _, _, _ = run_fdtd(c, f2, n_steps)

    return power1, power2


# =============================================================================
# Training loss
# =============================================================================
def loss_fn(T_blocks, target1=1.0, target2=0.0):
    """Training loss for frequency filter."""
    p1, p2 = evaluate_transfer(T_blocks)
    return (p1 - target1) ** 2 + (p2 - target2) ** 2


# =============================================================================
# Visualization
# =============================================================================
def plot_fields(p, u, v, T, c, freq, save_path=None):
    """Plot current fields."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Pressure
    ax = axes[0, 0]
    im = ax.imshow(p.T, origin='lower', cmap='RdBu_r',
                   vmin=-np.abs(p).max(), vmax=np.abs(p).max(),
                   extent=[0, LX * 1000, 0, LY * 1000])
    ax.set_title('Pressure p [Pa]')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    fig.colorbar(im, ax=ax)

    # u-velocity
    ax = axes[0, 1]
    im = ax.imshow(u.T, origin='lower', cmap='RdBu_r',
                   vmin=-np.abs(u).max(), vmax=np.abs(u).max(),
                   extent=[0, LX * 1000, 0, LY * 1000])
    ax.set_title('u-velocity [m/s]')
    fig.colorbar(im, ax=ax)

    # v-velocity
    ax = axes[0, 2]
    im = ax.imshow(v.T, origin='lower', cmap='RdBu_r',
                   vmin=-np.abs(v).max(), vmax=np.abs(v).max(),
                   extent=[0, LX * 1000, 0, LY * 1000])
    ax.set_title('v-velocity [m/s]')
    fig.colorbar(im, ax=ax)

    # Temperature
    ax = axes[1, 0]
    im = ax.imshow(T.T, origin='lower', cmap='hot',
                   extent=[0, LX * 1000, 0, LY * 1000])
    ax.set_title('Temperature [K]')
    fig.colorbar(im, ax=ax)

    # Sound speed
    ax = axes[1, 1]
    im = ax.imshow(c.T, origin='lower', cmap='viridis',
                   extent=[0, LX * 1000, 0, LY * 1000])
    ax.set_title('Sound speed [m/s]')
    fig.colorbar(im, ax=ax)

    # Intensity
    intensity = p * u[0:NX, :]
    ax = axes[1, 2]
    im = ax.imshow(intensity.T, origin='lower', cmap='coolwarm',
                   vmin=-np.abs(intensity).max(), vmax=np.abs(intensity).max(),
                   extent=[0, LX * 1000, 0, LY * 1000])
    ax.set_title('Intensity p*u [W/m^2]')
    fig.colorbar(im, ax=ax)

    plt.suptitle(f'f = {freq/1000:.1f} kHz', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# =============================================================================
# Main: test with uniform temperature (baseline)
# =============================================================================
if __name__ == "__main__":
    fig_dir = Path(__file__).parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("BASELINE: Uniform temperature (no heater blocks)")
    print("=" * 60)

    # Uniform temperature
    T_blocks_base = np.full(N_BLOCKS, T0)
    T_base = build_temperature_field(T_blocks_base)
    c_base = build_sound_speed(T_base)

    # Run at 2 kHz with snapshots
    n_steps = int(10.0 * LX / (C0 * DT))
    n_steps = max(n_steps, int(10.0 / (2000.0 * DT)))
    print(f"Running {n_steps} steps...")

    power_2k, probe_p, probe_u, snaps = run_fdtd(
        c_base, 2000.0, n_steps, record_interval=n_steps // 10
    )
    print(f"Transmitted power at 2 kHz (uniform T): {power_2k:.6f}")

    # Plot final state
    p_final = snaps[-1] if snaps else None
    if p_final is not None:
        # Reconstruct u, v from last step for visualization
        # (simplified: just plot what we have)
        u_vis = np.zeros((NX + 1, NY))
        v_vis = np.zeros((NX, NY + 1))
        plot_fields(p_final, u_vis, v_vis, T_base, c_base, 2000.0,
                    save_path=fig_dir / "baseline_2kHz.png")

    # Probe time trace
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    t = np.arange(len(probe_p)) * DT * 1000  # ms

    axes[0].plot(t, probe_p)
    axes[0].set_ylabel('Pressure [Pa]')
    axes[0].set_title('Probe at outlet: pressure')
    axes[0].axvline(x=t[int(0.6 * len(t))], color='r', linestyle='--',
                    label='transient discarded')
    axes[0].legend()

    axes[1].plot(t, probe_u)
    axes[1].set_ylabel('u-velocity [m/s]')
    axes[1].set_xlabel('Time [ms]')
    axes[1].set_title('Probe at outlet: u-velocity')
    axes[1].axvline(x=t[int(0.6 * len(t))], color='r', linestyle='--')

    plt.tight_layout()
    plt.savefig(fig_dir / "baseline_probe_trace.png", dpi=150)
    print(f"Saved: {fig_dir / 'baseline_probe_trace.png'}")
    plt.close()

    print("\nDone. Next: run with non-uniform T_blocks to see filtering.")
