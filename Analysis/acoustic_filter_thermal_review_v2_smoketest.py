#!/usr/bin/env python3
"""
acoustic_filter_thermal_review_v2_smoketest.py
==============================================
Lightweight smoke test of the steady-state thermal acoustic filter idea.

Goal: quickly check whether a static temperature field in air can produce
useful frequency-selective acoustic filtering.

Grid is coarser and frequencies are fewer than v1, so it finishes in a
reasonable time without stressing the system.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# =============================================================================
# Grid and physical constants (coarse for speed)
# =============================================================================
NX, NY = 200, 40
LX, LY = 0.80, 0.10
DX = LX / NX
DY = LY / NY

RHO0 = 1.225
T0 = 300.0
C0 = 343.0
D_THERMAL = 2.0e-5

FREQS = np.array([2000.0, 4000.0, 8000.0])

SRC_I0 = 6
SRC_I1 = 10
SRC_AMP = 1.0

PROBE_I0 = NX - 20
PROBE_I1 = NX - 4

SIM_DURATION = 0.008
START_AVG_FRACTION = 0.6

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("THERMAL ACOUSTIC FILTER -- SMOKE TEST")
print("=" * 60)
print(f"Domain: {LX*1000:.0f} mm x {LY*1000:.0f} mm  ({NX} x {NY} cells)")
print(f"dx = dy = {DX*1000:.2f} mm")
print(f"Base sound speed c0 = {C0:.1f} m/s")
print()


# =============================================================================
# Thermal solver
# =============================================================================
def build_laplacian(nx, ny):
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

    return csr_matrix((data, (rows, cols)), shape=(N, N))


LAPLACIAN = build_laplacian(NX, NY)


def solve_temperature(sigma):
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
    return C0 * np.sqrt(T / T0)


# =============================================================================
# Acoustic FDTD
# =============================================================================
def run_fdtd(c_field, freq, label=""):
    nx, ny = c_field.shape
    c_max = float(c_field.max())
    dt = 0.9 * DX / (c_max * np.sqrt(2))
    n_steps = int(SIM_DURATION / dt)
    start_avg = int(n_steps * START_AVG_FRACTION)
    omega = 2.0 * np.pi * freq

    p = np.zeros((nx, ny))
    u = np.zeros((nx + 1, ny))
    v = np.zeros((nx, ny + 1))

    # Window the source in y so it is not a hard corner
    src_window_y = np.ones(ny)
    src_window_y[0] = 0.0
    src_window_y[-1] = 0.0

    outlet_history = []

    for n in range(n_steps):
        t = n * dt
        src = SRC_AMP * np.sin(omega * t)
        p[SRC_I0:SRC_I1, :] += 0.25 * src * src_window_y[None, :]

        u[1:nx, :] -= (dt / RHO0) * (p[1:nx, :] - p[0:nx - 1, :]) / DX
        v[:, 1:ny] -= (dt / RHO0) * (p[:, 1:ny] - p[:, 0:ny - 1]) / DY

        u[0, :] = -p[0, :] / (RHO0 * c_field[0, :])
        u[nx, :] = p[nx - 1, :] / (RHO0 * c_field[nx - 1, :])
        v[:, 0] = -p[:, 0] / (RHO0 * c_field[:, 0])
        v[:, ny] = p[:, ny - 1] / (RHO0 * c_field[:, ny - 1])

        p -= dt * RHO0 * c_field**2 * (
            (u[1:nx + 1, :] - u[0:nx, :]) / DX +
            (v[:, 1:ny + 1] - v[:, 0:ny]) / DY
        )

        if n >= start_avg:
            outlet_history.append(p[PROBE_I0:PROBE_I1, :].copy())

    outlet = np.array(outlet_history)
    power = np.mean(outlet**2)
    print(f"    {label} f={freq/1000:.1f} kHz  power={power:.6e}  (n_steps={n_steps})")
    return power


# =============================================================================
# Thermal cases
# =============================================================================
def case_uniform():
    sigma = np.zeros((NX, NY))
    T = solve_temperature(sigma)
    c = sound_speed_from_temperature(T)
    return sigma, T, c, "uniform"


def case_single_strip():
    sigma = np.zeros((NX, NY))
    x_centre = 0.5 * LX
    width = 0.020
    half_w = 0.5 * width
    i0 = max(0, int((x_centre - half_w) / DX))
    i1 = min(NX, int((x_centre + half_w) / DX))
    sigma[i0:i1, :] = 2.0e7
    T = solve_temperature(sigma)
    c = sound_speed_from_temperature(T)
    return sigma, T, c, "single strip"


def case_thermal_grating():
    sigma = np.zeros((NX, NY))
    period = C0 / 8000.0
    n_periods = 6
    width = 0.5 * period
    half_w = 0.5 * width
    start_x = 0.25 * LX
    sigma_val = 2.0e7

    for k in range(n_periods):
        x_c = start_x + (k + 0.5) * period
        i0 = max(0, int((x_c - half_w) / DX))
        i1 = min(NX, int((x_c + half_w) / DX))
        if i1 > i0:
            sigma[i0:i1, :] = sigma_val

    T = solve_temperature(sigma)
    c = sound_speed_from_temperature(T)
    return sigma, T, c, f"thermal grating ({n_periods}p, {period*1000:.1f}mm)"


# =============================================================================
# Run smoke test
# =============================================================================
cases = [case_uniform(), case_single_strip(), case_thermal_grating()]
results = {}

for sigma, T, c, label in cases:
    print(f"\n--- Case: {label} ---")
    print(f"  T range: {T.min():.1f} K - {T.max():.1f} K")
    print(f"  c range: {c.min():.1f} - {c.max():.1f} m/s")
    powers = np.array([run_fdtd(c, f, label) for f in FREQS])
    results[label] = powers


# =============================================================================
# Plot frequency response
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 5))
for label, powers in results.items():
    ax.semilogy(FREQS / 1000, powers, marker='o', label=label, linewidth=2)

ax.set_xlabel('Frequency [kHz]')
ax.set_ylabel('Transmitted power')
ax.set_title('Thermal acoustic filter smoke test')
ax.legend()
ax.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig(FIG_DIR / "thermal_smoketest_frequency_response.png", dpi=150)
plt.close()
print(f"\nSaved: {FIG_DIR / 'thermal_smoketest_frequency_response.png'}")


# =============================================================================
# Plot fields for each case
# =============================================================================
def plot_fields(sigma, T, c, title, filename):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))

    ax = axes[0]
    im = ax.imshow(sigma.T, origin='lower', cmap='hot',
                   extent=[0, LX*1000, 0, LY*1000])
    ax.set_title('Heat source')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    fig.colorbar(im, ax=ax)

    ax = axes[1]
    im = ax.imshow(T.T, origin='lower', cmap='hot',
                   extent=[0, LX*1000, 0, LY*1000])
    ax.set_title('Temperature [K]')
    ax.set_xlabel('x [mm]')
    fig.colorbar(im, ax=ax)

    ax = axes[2]
    im = ax.imshow(c.T, origin='lower', cmap='viridis',
                   extent=[0, LX*1000, 0, LY*1000])
    ax.set_title('Sound speed [m/s]')
    ax.set_xlabel('x [mm]')
    fig.colorbar(im, ax=ax)

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=150)
    plt.close()
    print(f"Saved: {FIG_DIR / filename}")


for sigma, T, c, label in cases:
    safe_label = label.replace(' ', '_').replace('(', '').replace(')', '')
    plot_fields(sigma, T, c, f"Thermal case: {label}", f"thermal_smoketest_{safe_label}.png")


# =============================================================================
# Size estimate
# =============================================================================
print("\n" + "=" * 60)
print("SIZE ESTIMATE")
print("=" * 60)
for f in [2000.0, 4000.0, 8000.0]:
    lam = C0 / f
    print(f"f = {f/1000:.1f} kHz  lambda = {lam*1000:.1f} mm")
    for n in [3, 5, 10]:
        print(f"  {n} periods -> grating length ≈ {n*lam*1000:.1f} mm")
print("=" * 60)
