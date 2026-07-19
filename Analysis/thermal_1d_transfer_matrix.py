#!/usr/bin/env python3
"""
thermal_1d_transfer_matrix.py
=============================
1D acoustic transfer-matrix calculation for a stack of thermally heated air layers.

Purpose: quickly test whether a temperature field can produce useful
frequency-selective acoustic filtering, and estimate the physical size
needed if the mechanism is a Bragg/diffraction grating.

Physics
-------
Each layer is treated as a uniform fluid with:
  c(T) = c0 * sqrt(T / T0)
  rho(T) = rho0 * (T0 / T)        (ideal gas at constant pressure)
  Z(T) = rho(T) * c(T) = Z0 / sqrt(T / T0)
  k(T) = omega / c(T)

The acoustic transfer matrix for a layer of thickness L is:
  [ p_out ]   [ cos(kL)       j Z sin(kL) ] [ p_in ]
  [ u_out ] = [ j/Z sin(kL)   cos(kL)     ] [ u_in ]

For a stack, multiply the layer matrices. The transmission coefficient
(pressure amplitude ratio) into a matched load is:
  T = 2 Z0 / (A Z0 + B + C Z0^2 + D Z0)
where [A B; C D] is the total system matrix and Z0 is the ambient impedance.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# Physical constants
# =============================================================================
T0 = 300.0          # K, ambient temperature
C0 = 343.0          # m/s, ambient sound speed
RHO0 = 1.225        # kg/m^3, ambient density
Z0 = RHO0 * C0      # Pa.s/m, ambient characteristic impedance

# Hot-side temperature
T_HOT = 500.0       # K

# Frequency band (Hz)
FREQS = np.linspace(1000.0, 10000.0, 901)
OMEGAS = 2.0 * np.pi * FREQS

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def fluid_props(T):
    """Return (c, rho, Z, k/omega) for air at temperature T [K]."""
    c = C0 * np.sqrt(T / T0)
    rho = RHO0 * (T0 / T)
    Z = rho * c
    return c, rho, Z


def layer_matrix(omega, T, L):
    """Acoustic transfer matrix for a single fluid layer."""
    c, rho, Z = fluid_props(T)
    k = omega / c
    arg = k * L
    cos_a = np.cos(arg)
    sin_a = np.sin(arg)

    A = cos_a
    B = 1j * Z * sin_a
    C = 1j * sin_a / Z
    D = cos_a
    return np.array([[A, B], [C, D]])


def transmission(stack_T, stack_L, freqs=FREQS):
    """
    Compute pressure transmission coefficient magnitude for a layer stack.

    Parameters
    ----------
    stack_T : list or ndarray
        Temperature of each layer [K].
    stack_L : list or ndarray
        Thickness of each layer [m].

    Returns
    -------
    T_mag : ndarray
        |p_transmitted / p_incident| at each frequency.
    """
    stack_T = np.asarray(stack_T)
    stack_L = np.asarray(stack_L)
    n_layers = len(stack_T)

    T_mag = np.zeros_like(freqs, dtype=float)

    for i_f, omega in enumerate(OMEGAS):
        M_total = np.eye(2, dtype=complex)
        for i in range(n_layers):
            M_layer = layer_matrix(omega, stack_T[i], stack_L[i])
            M_total = M_layer @ M_total

        A, B, C, D = M_total[0, 0], M_total[0, 1], M_total[1, 0], M_total[1, 1]
        T_complex = 2.0 * Z0 / (A * Z0 + B + C * Z0**2 + D * Z0)
        T_mag[i_f] = np.abs(T_complex)

    return T_mag


# =============================================================================
# Case 1: single hot layer
# =============================================================================
L_single = 0.05     # 50 mm hot layer
T_single = [T0, T_HOT, T0]
L_single_stack = [0.0, L_single, 0.0]  # outer layers are half-spaces (L=0 => identity)

T_mag_single = transmission(T_single, L_single_stack)

# =============================================================================
# Case 2: periodic thermal grating (quarter-wave Bragg stack)
# =============================================================================
# For a stop band, each layer must be a quarter wavelength thick at the
# target frequency, so reflections from every interface add in phase.
# Period = lambda_hot/4 + lambda_cold/4 = (c_hot + c_cold) / (4 f_target).
c_hot, rho_hot, Z_hot = fluid_props(T_HOT)
c_cold, rho_cold, Z_cold = fluid_props(T0)
n_periods = 20

# Tuned to 8 kHz
f_target_8k = 8000.0
L_hot_8k = c_hot / (4.0 * f_target_8k)
L_cold_8k = c_cold / (4.0 * f_target_8k)
period_8k = L_hot_8k + L_cold_8k

stack_T_grating_8k = []
stack_L_grating_8k = []
for _ in range(n_periods):
    stack_T_grating_8k.extend([T_HOT, T0])
    stack_L_grating_8k.extend([L_hot_8k, L_cold_8k])

T_mag_grating_8k = transmission(stack_T_grating_8k, stack_L_grating_8k)
grating_length_8k = n_periods * period_8k

# =============================================================================
# Case 3: periodic grating tuned to 4 kHz (longer period, larger device)
# =============================================================================
f_target_4k = 4000.0
L_hot_4k = c_hot / (4.0 * f_target_4k)
L_cold_4k = c_cold / (4.0 * f_target_4k)
period_4k = L_hot_4k + L_cold_4k

stack_T_grating_4k = []
stack_L_grating_4k = []
for _ in range(n_periods):
    stack_T_grating_4k.extend([T_HOT, T0])
    stack_L_grating_4k.extend([L_hot_4k, L_cold_4k])

T_mag_grating_4k = transmission(stack_T_grating_4k, stack_L_grating_4k)
grating_length_4k = n_periods * period_4k

# =============================================================================
# Plot transmission spectra
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(FREQS / 1000, T_mag_single, label=f"single hot layer ({L_single*1000:.0f} mm)", linewidth=2)
ax.plot(FREQS / 1000, T_mag_grating_8k, label=f"{n_periods}-period grating tuned to 8 kHz (L={grating_length_8k*1000:.1f} mm)", linewidth=2)
ax.plot(FREQS / 1000, T_mag_grating_4k, label=f"{n_periods}-period grating tuned to 4 kHz (L={grating_length_4k*1000:.1f} mm)", linewidth=2)

ax.set_xlabel("Frequency [kHz]")
ax.set_ylabel("|p_t / p_i|")
ax.set_title("1D acoustic transmission through thermally heated air layers")
ax.legend()
ax.grid(True, ls="--", alpha=0.5)
ax.set_xlim(FREQS[0] / 1000, FREQS[-1] / 1000)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(FIG_DIR / "thermal_1d_transmission.png", dpi=150)
plt.close()
print(f"Saved: {FIG_DIR / 'thermal_1d_transmission.png'}")

# =============================================================================
# Print summary
# =============================================================================
print("=" * 60)
print("1D THERMAL TRANSFER MATRIX RESULTS")
print("=" * 60)
print(f"Ambient: T0 = {T0:.0f} K, c0 = {C0:.0f} m/s, Z0 = {Z0:.2f} Pa.s/m")
print(f"Hot side: T_hot = {T_HOT:.0f} K, c_hot = {c_hot:.0f} m/s")
print()
print("Case 1: single hot layer")
print(f"  Thickness = {L_single*1000:.0f} mm")
print(f"  Transmission at 2 kHz: {T_mag_single[np.argmin(np.abs(FREQS - 2000))]:.4f}")
print(f"  Transmission at 4 kHz: {T_mag_single[np.argmin(np.abs(FREQS - 4000))]:.4f}")
print(f"  Transmission at 8 kHz: {T_mag_single[np.argmin(np.abs(FREQS - 8000))]:.4f}")
print()
print(f"Case 2: thermal grating ({n_periods} periods, quarter-wave hot/cold stack)")
print(f"  Tuned to 8 kHz: period = {period_8k*1000:.1f} mm, total length = {grating_length_8k*1000:.1f} mm")
print(f"    Transmission at 8 kHz: {T_mag_grating_8k[np.argmin(np.abs(FREQS - 8000))]:.4f}")
print(f"  Tuned to 4 kHz: period = {period_4k*1000:.1f} mm, total length = {grating_length_4k*1000:.1f} mm")
print(f"    Transmission at 4 kHz: {T_mag_grating_4k[np.argmin(np.abs(FREQS - 4000))]:.4f}")
print()
print("Conclusion: a single hot layer is basically transparent.")
print("A useful stop band needs a multi-period grating whose period is")
print("comparable to the acoustic wavelength. At speech-band frequencies")
print("this makes the device several hundred millimetres long.")
print("=" * 60)
