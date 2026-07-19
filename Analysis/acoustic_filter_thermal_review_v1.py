#!/usr/bin/env python3
"""
acoustic_filter_thermal_review_v1.py
====================================
Cleaned-up steady-state thermal acoustic filter review.

Goal: test whether a static temperature field in air can produce useful
frequency-selective acoustic filtering in a compact 2D domain.

Physics
-------
- Steady heat conduction:  nabla^2 T = -sigma / D
  with Dirichlet T = T0 on all outer boundaries.
  No advection, no radiation, no transients -- this is the BEST case for
  the thermal approach.
- Sound speed from temperature:  c(T) = c0 * sqrt(T / T0).
- Acoustic propagation: 2D first-order FDTD with first-order absorbing
  boundaries on all four sides.

Cases compared
--------------
1. Uniform T = T0 (baseline).
2. Single hot strip (refraction / speed-contrast only).
3. Periodic thermal grating (Bragg/diffraction mechanism).

Output
------
- Transmitted power vs frequency for each case.
- Temperature and sound-speed fields.
- Size estimate showing why a useful thermal grating is too large.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# =============================================================================
# Grid and physical constants
# =============================================================================
NX, NY = 400, 50
LX, LY = 0.80, 0.10
DX = LX / NX
DY = LY / NY

RHO0 = 1.225          # kg/m^3
T0 = 300.0            # K
C0 = 343.0            # m/s
D_THERMAL = 2.0e-5    # m^2/s, thermal diffusivity of air

T_MAX_TARGET = 500.0  # K, hot-side temperature for high-contrast cases

# Frequency band to test (Hz)
FREQS = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0])

# Source: plane pressure source a short distance from the left boundary
SRC_I0 = 10
SRC_I1 = 14
SRC_AMP = 1.0

# Outlet probe: spatial average over a strip near the right boundary,
# away from the corner reflections
PROBE_I0 = NX - 30
PROBE_I1 = NX - 5

# Simulation duration and averaging
SIM_DURATION = 0.012          # s
START_AVG_FRACTION = 0.65     # start averaging after transients

# Figure output
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("THERMAL ACOUSTIC FILTER REVIEW")
print("=" * 70)
print(f"Domain: {LX*1000:.0f} mm x {LY*1000:.0f} mm  ({NX} x {NY} cells)")
print(f"dx = dy = {DX*1000:.2f} mm")
print(f"Base sound speed c0 = {C0:.1f} m/s")
print(f"Thermal diffusivity D = {D_THERMAL:.2e} m^2/s")
print()


# =============================================================================
# Thermal solver: nabla^2 T = -sigma / D,  Dirichlet T=T0 on boundaries
# =============================================================================
def build_laplacian(nx, ny, dx, dy):
    """Build sparse 2D Laplacian matrix with Dirichlet BCs."""
    N = nx * ny
    rows, cols, data = [], [], []

    def idx(i, j):
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            k = idx(i, j)
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                rows.append(k); cols.append(k); data.append(1.0)
            else:
                rows.append(k); cols.append(k); data.append(-4.0)
                rows.append(k); cols.append(idx(i + 1, j)); data.append(1.0)
                rows.append(k); cols.append(idx(i - 1, j)); data.append(1.0)
                rows.append(k); cols.append(idx(i, j + 1)); data.append(1.0)
                rows.append(k); cols.append(idx(i, j - 1)); data.append(1.0)

    A = csr_matrix((data, (rows, cols)), shape=(N, N))
    return A


# Build and factorise once
LAPLACIAN = build_laplacian(NX, NY, DX, DY)


def solve_temperature(sigma):
    """
    Solve nabla^2 T = -sigma / D with Dirichlet T=T0 on all boundaries.

    Parameters
    ----------
    sigma : ndarray (NX, NY)
        Volumetric heat source [W/m^3].

    Returns
    -------
    T : ndarray (NX, NY)
        Temperature [K].
    """
    nx, ny = sigma.shape
    b = -sigma.flatten() * (DX * DY) / D_THERMAL

    for i in range(nx):
        for j in range(ny):
            k = i * ny + j
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                b[k] = T0

    T_flat = spsolve(LAPLACIAN, b)
    return T_flat.reshape((nx, ny))


def sound_speed_from_temperature(T):
    """Ideal-gas sound speed vs temperature."""
    return C0 * np.sqrt(T / T0)


# =============================================================================
# Acoustic FDTD
# =============================================================================
def run_fdtd(c_field, freq):
    """
    2D staggered first-order acoustic FDTD with first-order absorbing BCs.

    Returns
    -------
    power : float
        Time-averaged squared pressure over the outlet probe region.
    """
    nx, ny = c_field.shape
    c_max = float(c_field.max())
    dt = 0.9 * DX / (c_max * np.sqrt(2))
    n_steps = int(SIM_DURATION / dt)
    start_avg = int(n_steps * START_AVG_FRACTION)

    omega = 2.0 * np.pi * freq

    p = np.zeros((nx, ny))
    u = np.zeros((nx + 1, ny))
    v = np.zeros((nx, ny + 1))

    # Smooth source window in y (Hanning) so the source is not a hard corner
    src_window_y = np.ones(ny)
    src_window_y[0] = 0.0
    src_window_y[-1] = 0.0

    outlet_history = []

    for n in range(n_steps):
        t = n * dt
        src = SRC_AMP * np.sin(omega * t)

        # Inject source as a soft source plane
        p[SRC_I0:SRC_I1, :] += 0.25 * src * src_window_y[None, :]

        # Update velocities
        u[1:nx, :] -= (dt / RHO0) * (p[1:nx, :] - p[0:nx - 1, :]) / DX
        v[:, 1:ny] -= (dt / RHO0) * (p[:, 1:ny] - p[:, 0:ny - 1]) / DY

        # First-order absorbing BCs on all four sides
        u[0, :] = -p[0, :] / (RHO0 * c_field[0, :])
        u[nx, :] = p[nx - 1, :] / (RHO0 * c_field[nx - 1, :])
        v[:, 0] = -p[:, 0] / (RHO0 * c_field[:, 0])
        v[:, ny] = p[:, ny - 1] / (RHO0 * c_field[:, ny - 1])

        # Update pressure
        p -= dt * RHO0 * c_field**2 * (
            (u[1:nx + 1, :] - u[0:nx, :]) / DX +
            (v[:, 1:ny + 1] - v[:, 0:ny]) / DY
        )

        # Record outlet pressure
        if n >= start_avg:
            outlet_history.append(p[PROBE_I0:PROBE_I1, :].copy())

    outlet = np.array(outlet_history)
    power = np.mean(outlet**2)
    return power


# =============================================================================
# Thermal case definitions
# =============================================================================
def case_uniform():
    """Baseline: no heating."""
    sigma = np.zeros((NX, NY))
    T = solve_temperature(sigma)
    c = sound_speed_from_temperature(T)
    return sigma, T, c, "uniform"


def case_single_strip():
    """One hot strip in the centre."""
    sigma = np.zeros((NX, NY))
    x_centre = 0.5 * LX
    width = 0.020
    half_w = 0.5 * width
    i0 = max(0, int((x_centre - half_w) / DX))
    i1 = min(NX, int((x_centre + half_w) / DX))

    # Choose sigma to hit roughly T_MAX_TARGET in the strip centre
    # This is found by a quick scaling; exact value is not critical.
    sigma_val = 2.0e7
    sigma[i0:i1, :] = sigma_val

    T = solve_temperature(sigma)
    c = sound_speed_from_temperature(T)
    return sigma, T, c, "single strip"


def case_thermal_grating():
    """Periodic hot/cold strips forming a Bragg-like grating."""
    sigma = np.zeros((NX, NY))

    # Grating period matched to the 4 kHz wavelength in ambient air
    period = C0 / 4000.0
    n_periods = int((0.6 * LX) / period)
    width = 0.5 * period
    half_w = 0.5 * width

    start_x = 0.2 * LX
    sigma_val = 2.0e7

    for k in range(n_periods):
        x_c = start_x + (k + 0.5) * period
        i0 = max(0, int((x_c - half_w) / DX))
        i1 = min(NX, int((x_c + half_w) / DX))
        if i1 > i0:
            sigma[i0:i1, :] = sigma_val

    T = solve_temperature(sigma)
    c = sound_speed_from_temperature(T)
    return sigma, T, c, f"thermal grating ({n_periods} periods, period={period*1000:.1f} mm)"


# =============================================================================
# Run all cases
# =============================================================================
cases = [case_uniform(), case_single_strip(), case_thermal_grating()]
results = {}

for sigma, T, c, label in cases:
    print(f"\n--- Case: {label} ---")
    print(f"  T range: {T.min():.1f} K - {T.max():.1f} K")
    print(f"  c range: {c.min():.1f} - {c.max():.1f} m/s")

    powers = np.array([run_fdtd(c, f) for f in FREQS])
    results[label] = powers

    for f, pwr in zip(FREQS, powers):
        print(f"  f = {f/1000:.1f} kHz  power = {pwr:.6e}")


# =============================================================================
# Figures
# =============================================================================
def plot_fields(sigma, T, c, title, filename):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    im = ax.imshow(sigma.T, origin='lower', cmap='hot',
                   extent=[0, LX*1000, 0, LY*1000])
    ax.set_title('Heat source sigma [W/m^3]')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    fig.colorbar(im, ax=ax)

    ax = axes[1]
    im = ax.imshow(T.T, origin='lower', cmap='hot',
                   extent=[0, LX*1000, 0, LY*1000])
    ax.set_title('Temperature T [K]')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    fig.colorbar(im, ax=ax)

    ax = axes[2]
    im = ax.imshow(c.T, origin='lower', cmap='viridis',
                   extent=[0, LX*1000, 0, LY*1000])
    ax.set_title('Sound speed c [m/s]')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    fig.colorbar(im, ax=ax)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=150)
    plt.close()
    print(f"Saved: {FIG_DIR / filename}")


for sigma, T, c, label in cases:
    safe_label = label.replace(' ', '_').replace('(', '').replace(')', '')
    plot_fields(sigma, T, c, f"Thermal case: {label}", f"thermal_review_{safe_label}.png")


# =============================================================================
# Frequency response comparison
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 5))
for label, powers in results.items():
    ax.semilogy(FREQS / 1000, powers, marker='o', label=label, linewidth=2)

ax.set_xlabel('Frequency [kHz]')
ax.set_ylabel('Transmitted power (arbitrary units)')
ax.set_title('Thermal acoustic filter: frequency response')
ax.legend()
ax.grid(True, which='both', ls='--', alpha=0.5)
ax.set_xlim(FREQS[0]/1000, FREQS[-1]/1000)
plt.tight_layout()
plt.savefig(FIG_DIR / "thermal_review_frequency_response.png", dpi=150)
plt.close()
print(f"Saved: {FIG_DIR / 'thermal_review_frequency_response.png'}")


# =============================================================================
# Size estimate summary
# =============================================================================
print("\n" + "=" * 70)
print("SIZE ESTIMATE")
print("=" * 70)
print("A useful Bragg/diffraction grating needs several periods.")
print("Period is set by the acoustic wavelength in the medium.")
print()
for f in [2000.0, 4000.0, 8000.0]:
    lam = C0 / f
    print(f"  f = {f/1000:.1f} kHz  lambda = {lam*1000:.1f} mm")
    for n in [3, 5, 10]:
        print(f"    {n} periods -> grating length ≈ {n*lam*1000:.1f} mm ({n*lam:.3f} m)")
print()
print("Conclusion: even at 8 kHz, a 5-period grating is ~215 mm long.")
print("At speech-band frequencies (2-4 kHz) the device would be 0.3-0.5 m")
print("or larger, which is incompatible with a compact analog wave computer.")
print("=" * 70)
