"""
acoustic_filter_with_thermal.py

Integrated 2D acoustic FDTD with trainable temperature field.
- Thermal: steady-state Poisson solve with 4 heater strips
- Acoustic: staggered leapfrog with variable sound speed
- Goal: frequency-selective transmission (pass 2 kHz, block 8 kHz)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# =============================================================================
# PHYSICAL PARAMETERS
# =============================================================================
# Grid
NX, NY = 250, 50          # cells
DX = 0.002                # m (2 mm)
DY = DX                   # uniform
Lx = NX * DX              # 0.5 m
Ly = NY * DY              # 0.1 m

# Air at 300 K
RHO0 = 1.225              # kg/m^3
C0 = 343.0                # m/s
T0 = 300.0                # K
D_THERMAL = 2e-5          # m^2/s (thermal diffusivity of air)

# Heater strips
STRIP_POSITIONS = np.array([0.100, 0.150, 0.200, 0.250])  # m from left
STRIP_WIDTH = 0.020        # m (20 mm)

# Source (distributed over 3 cells for smoother plane wave)
I_SRC_START = 3
I_SRC_END = 6              # cells i=3,4,5 get source injection
SRC_AMP = 1.0

# Measurement: spatial average over last ~200mm (excludes boundary layer)
I_OUT_START = NX - 100     # start averaging 200mm from end
I_OUT_END = NX - 10        # stop 20mm from boundary

# Time-stepping
SIM_DURATION = 0.010       # s (10 ms physical time)
START_AVG_FRACTION = 0.6   # start averaging after 60% of steps

# =============================================================================
# THERMAL SOLVER: ∇²T = -σ/D  with  Dirichlet T=T0 on all boundaries
# =============================================================================

def build_heat_source(sigma_amps, nx=NX, ny=NY, dx=DX, positions=STRIP_POSITIONS, width=STRIP_WIDTH):
    """
    Build 2D heat source field σ(x,y) from 4 strip amplitudes.
    Each strip is a vertical bar of uniform σ, full height.
    """
    sigma = np.zeros((nx, ny))
    half_w_cells = int(width / (2 * dx))
    for k, amp in enumerate(sigma_amps):
        i_c = int(positions[k] / dx)
        i_s = max(0, i_c - half_w_cells)
        i_e = min(nx, i_c + half_w_cells)
        sigma[i_s:i_e, :] = amp
    return sigma


def solve_temperature(sigma, dx=DX, D=D_THERMAL, T0=T0):
    """
    Solve ∇²T = -sigma/D  on uniform grid with Dirichlet T=T0 on all boundaries.
    Uses sparse direct solver (exact, fast for 12k unknowns).
    """
    nx, ny = sigma.shape
    N = nx * ny

    # Build sparse Laplacian matrix with Dirichlet BCs
    rows, cols, data = [], [], []
    for i in range(nx):
        for j in range(ny):
            k = i * ny + j
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                # Boundary: Dirichlet T = T0
                rows.append(k)
                cols.append(k)
                data.append(1.0)
            else:
                # Interior: 5-point Laplacian stencil
                rows.append(k); cols.append(k); data.append(-4.0)               # center
                rows.append(k); cols.append((i + 1) * ny + j); data.append(1.0)  # i+1,j
                rows.append(k); cols.append((i - 1) * ny + j); data.append(1.0)  # i-1,j
                rows.append(k); cols.append(i * ny + (j + 1)); data.append(1.0)  # i,j+1
                rows.append(k); cols.append(i * ny + (j - 1)); data.append(1.0)  # i,j-1

    A = csr_matrix((data, (rows, cols)), shape=(N, N))

    # Build RHS
    b = np.zeros(N)
    for i in range(nx):
        for j in range(ny):
            k = i * ny + j
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                b[k] = T0
            else:
                b[k] = -sigma[i, j] * dx**2 / D

    T_flat = spsolve(A, b)
    return T_flat.reshape((nx, ny))


# =============================================================================
# ACOUSTIC FDTD: first-order linear acoustics with variable sound speed
# =============================================================================

def run_fdtd(c_field, frequency, duration=SIM_DURATION, rho0=RHO0, dx=DX,
             i_src_start=I_SRC_START, i_src_end=I_SRC_END,
             i_out_start=I_OUT_START, i_out_end=I_OUT_END, src_amp=SRC_AMP):
    """
    2D staggered leapfrog acoustic FDTD.
    Returns time-averaged transmitted power at outlet slice.
    """
    nx, ny = c_field.shape

    # CFL stability: dt < dx / (c_max * sqrt(2))
    c_max = float(c_field.max())
    dt = 0.9 * dx / (c_max * np.sqrt(2))
    n_steps = int(duration / dt)
    start_avg = int(n_steps * START_AVG_FRACTION)

    omega = 2.0 * np.pi * frequency

    # State arrays
    p = np.zeros((nx, ny))
    u = np.zeros((nx + 1, ny))      # x-velocity on vertical faces
    v = np.zeros((nx, ny + 1))      # y-velocity on horizontal faces

    p_out_history = []

    for n in range(n_steps):
        t = n * dt

        # --- Update velocities (interior) ---
        u[1:nx, :] -= (dt / rho0) * (p[1:nx, :] - p[0:nx - 1, :]) / dx
        v[:, 1:ny] -= (dt / rho0) * (p[:, 1:ny] - p[:, 0:ny - 1]) / dx

        # --- Absorbing boundary conditions on velocities ---
        u[0, :]  = -p[0, :] / (rho0 * c_field[0, :])
        u[nx, :] =  p[nx - 1, :] / (rho0 * c_field[nx - 1, :])
        v[:, 0]  = -p[:, 0] / (rho0 * c_field[:, 0])
        v[:, ny] =  p[:, ny - 1] / (rho0 * c_field[:, ny - 1])

        # --- Update pressure (interior, uses boundary velocities) ---
        p -= dt * rho0 * c_field**2 * (
            (u[1:nx + 1, :] - u[0:nx, :]) / dx +
            (v[:, 1:ny + 1] - v[:, 0:ny]) / dx
        )

        # --- Source injection (plane wave, all y) ---
        p[i_src_start:i_src_end, :] += src_amp * np.sin(omega * t)

        # --- Record outlet pressure for averaging ---
        if n >= start_avg:
            p_out_history.append(p[i_out_start:i_out_end, :].copy())

    # Time-averaged power ∝ <p^2> over outlet region (spatial + temporal average)
    p_out = np.array(p_out_history)  # shape: (n_avg_time, n_avg_x, ny)
    power = np.mean(p_out**2)
    return power, dt, n_steps


# =============================================================================
# DEMO
# =============================================================================

def demo():
    print("=" * 60)
    print("ACOUSTIC FILTER WITH THERMAL FIELD")
    print("=" * 60)
    print(f"Domain: {Lx*1000:.0f} mm x {Ly*1000:.0f} mm  ({NX}x{NY} cells)")
    print(f"dx = {DX*1000:.1f} mm")
    print(f"Base sound speed c0 = {C0:.1f} m/s")
    print()

    # -------------------------------------------------------------------------
    # Case 1: Uniform temperature (no heaters) — baseline
    # -------------------------------------------------------------------------
    print("CASE 1: Uniform temperature (no heaters)")
    sigma_uniform = np.zeros((NX, NY))
    T_uniform = solve_temperature(sigma_uniform)
    c_uniform = C0 * np.sqrt(T_uniform / T0)

    print(f"  T range: {T_uniform.min():.1f} K - {T_uniform.max():.1f} K")
    print(f"  c range: {c_uniform.min():.1f} - {c_uniform.max():.1f} m/s")

    p2k_u, dt2, ns2 = run_fdtd(c_uniform, 2000.0)
    p8k_u, dt8, ns8 = run_fdtd(c_uniform, 8000.0)
    ratio_u = p2k_u / (p8k_u + 1e-30)

    print(f"  Power @ 2 kHz: {p2k_u:.6e}")
    print(f"  Power @ 8 kHz: {p8k_u:.6e}")
    print(f"  Ratio (2k/8k): {ratio_u:.3f}")
    print()

    # -------------------------------------------------------------------------
    # Case 2: With heater strips
    # -------------------------------------------------------------------------
    print("CASE 2: Heater strips active")
    sigma_amps = np.array([5.0, 8.0, 6.0, 3.0])  # W/m^3 (demo values)
    sigma_heated = build_heat_source(sigma_amps)
    T_heated = solve_temperature(sigma_heated)
    c_heated = C0 * np.sqrt(T_heated / T0)

    print(f"  T range: {T_heated.min():.1f} K - {T_heated.max():.1f} K")
    print(f"  c range: {c_heated.min():.1f} - {c_heated.max():.1f} m/s")

    p2k_h, dt2, ns2 = run_fdtd(c_heated, 2000.0)
    p8k_h, dt8, ns8 = run_fdtd(c_heated, 8000.0)
    ratio_h = p2k_h / (p8k_h + 1e-30)

    print(f"  Power @ 2 kHz: {p2k_h:.6e}")
    print(f"  Power @ 8 kHz: {p8k_h:.6e}")
    print(f"  Ratio (2k/8k): {ratio_h:.3f}")
    print()

    # -------------------------------------------------------------------------
    # Plot temperature and sound speed fields
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    ax = axes[0, 0]
    im = ax.imshow(T_heated.T, origin='lower', aspect='auto',
                   extent=[0, Lx*1000, 0, Ly*1000], cmap='hot')
    ax.set_title('Temperature T (K)')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    im = ax.imshow(c_heated.T, origin='lower', aspect='auto',
                   extent=[0, Lx*1000, 0, Ly*1000], cmap='viridis')
    ax.set_title('Sound speed c (m/s)')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    plt.colorbar(im, ax=ax)

    ax = axes[1, 0]
    ax.bar(['2 kHz\n(uniform)', '8 kHz\n(uniform)', '2 kHz\n(heated)', '8 kHz\n(heated)'],
           [p2k_u, p8k_u, p2k_h, p8k_h], color=['blue', 'red', 'blue', 'red'],
           alpha=0.7)
    ax.set_ylabel('Transmitted power')
    ax.set_title('Power comparison')

    ax = axes[1, 1]
    ax.bar(['Uniform', 'Heated'], [ratio_u, ratio_h], color=['gray', 'orange'])
    ax.set_ylabel('Ratio (2 kHz / 8 kHz)')
    ax.set_title('Filter selectivity')
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='no filtering')
    ax.legend()

    plt.tight_layout()
    plt.savefig('thermal_filter_demo.png', dpi=150)
    print("Saved figure: thermal_filter_demo.png")
    plt.show()


if __name__ == '__main__':
    demo()
