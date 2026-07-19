#!/usr/bin/env python3
"""
fdtd_zk_uniform_1d_v1.py
=========================
Clean 1-D validation of the Zwikker-Kosten (ZK) equivalent-fluid model in a
uniform porous domain (no air-porous interfaces).

A continuous sinusoidal source is injected through a transparent left boundary.
Two probes separated by a known distance record the steady-state pressure.
The amplitude ratio and phase difference are compared against the analytical
ZK complex wavenumber.

The analytical wavenumber is:

    k(omega) = (omega / c_zk) * sqrt(1 - j * sigma / (omega * rho_eff))

with c_zk = sqrt(p0 / (k_s * rho0)) and rho_eff = k_s * rho0 / phi.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
RHO0 = 1.225          # kg/m^3, ambient air density
C0   = 343.0          # m/s, ambient sound speed
P0   = 101325.0       # Pa, ambient pressure

# =============================================================================
# POROUS MEDIUM PARAMETERS
# =============================================================================
PHI       = 0.50      # porosity
K_S       = 1.50      # structure factor / tortuosity
SIGMA     = 5.0e3     # Pa s / m^2, static flow resistivity (moderate damping)

RHO_EFF = K_S * RHO0 / PHI
K_EFF   = P0 / PHI
C_ZK    = np.sqrt(K_EFF / RHO_EFF)
Z_AIR   = RHO0 * C0

print(f"ZK parameters: phi={PHI}, k_s={K_S}, sigma={SIGMA:.2e}")
print(f"  rho_eff = {RHO_EFF:.3f} kg/m^3")
print(f"  K_eff   = {K_EFF:.1f} Pa")
print(f"  c_zk    = {C_ZK:.1f} m/s")

# =============================================================================
# DOMAIN / GRID
# =============================================================================
LX       = 2.0        # m
NX       = 1200
DX       = LX / NX
DT       = 0.95 * DX / C0

x = np.linspace(0.0, LX, NX)

# Uniform porous arrays
phi_arr     = np.full(NX, PHI)
ks_arr      = np.full(NX, K_S)
sigma_arr   = np.full(NX, SIGMA)
rho_eff_arr = ks_arr * RHO0 / phi_arr
K_eff_arr   = P0 / phi_arr

# Probes (close separation so phase difference stays within [-pi, pi])
PROBE1_X = 0.3
PROBE2_X = 0.35
probe1_i = int(PROBE1_X / DX)
probe2_i = int(PROBE2_X / DX)
D_PROBES = PROBE2_X - PROBE1_X

# =============================================================================
# ANALYTICAL ZK WAVENUMBER
# =============================================================================
def zk_wavenumber(omega):
    """Complex wavenumber; fields vary as exp(j(omega t - k x))."""
    return (omega / C_ZK) * np.sqrt(1.0 - 1j * SIGMA / (omega * RHO_EFF))


# =============================================================================
# FDTD RUNNER
# =============================================================================
def run_fdtd(freq, n_periods=25):
    """Run ZK FDTD at one frequency; return probe time histories."""
    omega = 2.0 * np.pi * freq

    def source_value(n):
        return np.sin(omega * n * DT)

    p = np.zeros(NX)
    v = np.zeros(NX + 1)

    hist1 = []
    hist2 = []

    n_steps = int(n_periods / (freq * DT))

    for n in range(n_steps):
        v_new = v.copy()

        # velocity update
        dp_dx = (p[1:] - p[:-1]) / DX
        v_new[1:-1] = v[1:-1] + DT * (-dp_dx - sigma_arr[1:] * v[1:-1]) / rho_eff_arr[1:]

        # pressure update
        dv_dx = (v_new[1:] - v_new[:-1]) / DX
        p_new = p - DT * K_eff_arr * dv_dx

        # transparent source at left boundary
        src = source_value(n)
        p_new[0] = src
        v_new[0] = src / Z_AIR

        # Mur ABC at right boundary
        p_new[-1] = p[-2] + (C0 * DT - DX) / (C0 * DT + DX) * (p_new[-2] - p[-1])

        p, v = p_new, v_new
        hist1.append(p[probe1_i])
        hist2.append(p[probe2_i])

    return np.array(hist1), np.array(hist2), n_steps


# =============================================================================
# LOCK-IN EXTRACTION
# =============================================================================
def lockin(hist, freq, n_steps, n_avg_periods=10):
    """Return complex amplitude via quadrature demodulation."""
    t = np.arange(n_steps) * DT
    omega = 2.0 * np.pi * freq
    start = int((len(t) - n_avg_periods / (freq * DT)))
    if start < 0:
        start = 0
    t_win = t[start:]
    s_win = hist[start:]
    c = np.cos(omega * t_win)
    s = np.sin(omega * t_win)
    A = 2.0 * np.mean(s_win * c)
    B = 2.0 * np.mean(s_win * s)
    return A + 1j * B


# =============================================================================
# SWEEP OVER FREQUENCIES
# =============================================================================
freqs = np.linspace(200.0, 3000.0, 15)

fdtd_amp_ratio = []
fdtd_phase_diff = []
ana_amp_ratio = []
ana_phase_diff = []

print("\n" + "=" * 60)
print("Frequency sweep")
print("=" * 60)

for f in freqs:
    h1, h2, n_steps = run_fdtd(f, n_periods=30)
    c1 = lockin(h1, f, n_steps)
    c2 = lockin(h2, f, n_steps)

    omega = 2.0 * np.pi * f
    k = zk_wavenumber(omega)
    alpha = -np.imag(k)            # attenuation constant (Np/m)
    k_r = np.real(k)

    fdtd_amp_ratio.append(np.abs(c2) / np.abs(c1))
    fdtd_phase_diff.append(np.angle(c2 / c1))

    ana_amp_ratio.append(np.exp(-alpha * D_PROBES))
    # Lock-in convention maps a physical lag to a positive phase, so the
    # measured phase difference is +k_r * D.
    ana_phase_diff.append(k_r * D_PROBES)

    print(f"f={f:6.1f} Hz | FDTD amp={fdtd_amp_ratio[-1]:.4f} ana={ana_amp_ratio[-1]:.4f} | "
          f"FDTD phase={fdtd_phase_diff[-1]:6.3f} ana={ana_phase_diff[-1]:6.3f}")

fdtd_amp_ratio = np.array(fdtd_amp_ratio)
fdtd_phase_diff = np.unwrap(np.array(fdtd_phase_diff))
ana_amp_ratio = np.array(ana_amp_ratio)
ana_phase_diff = np.unwrap(np.array(ana_phase_diff))

# =============================================================================
# PLOTS
# =============================================================================
fig_dir = Path(__file__).parent / 'figures'
fig_dir.mkdir(exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Amplitude ratio
axes[0, 0].semilogy(freqs, fdtd_amp_ratio, 'o-', label='FDTD')
axes[0, 0].semilogy(freqs, ana_amp_ratio, 's--', label='Analytical ZK')
axes[0, 0].set_xlabel('Frequency (Hz)')
axes[0, 0].set_ylabel('Amplitude ratio |p2|/|p1|')
axes[0, 0].set_title(f'Amplitude decay over {D_PROBES:.2f} m')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Phase difference
axes[0, 1].plot(freqs, fdtd_phase_diff, 'o-', label='FDTD')
axes[0, 1].plot(freqs, ana_phase_diff, 's--', label='Analytical ZK')
axes[0, 1].set_xlabel('Frequency (Hz)')
axes[0, 1].set_ylabel('Phase difference arg(p2/p1) (rad)')
axes[0, 1].set_title('Phase accumulation between probes')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Phase velocity: c_phase = omega / k_r, k_r = phase_diff / D
fdtd_k_r = fdtd_phase_diff / D_PROBES
ana_k_r = ana_phase_diff / D_PROBES
fdtd_c_phase = 2.0 * np.pi * freqs / fdtd_k_r
ana_c_phase = 2.0 * np.pi * freqs / ana_k_r
axes[1, 0].plot(freqs, fdtd_c_phase, 'o-', label='FDTD')
axes[1, 0].plot(freqs, ana_c_phase, 's--', label='Analytical ZK')
axes[1, 0].axhline(C_ZK, color='red', linestyle='--', label='high-freq ZK limit')
axes[1, 0].set_xlabel('Frequency (Hz)')
axes[1, 0].set_ylabel('Phase velocity (m/s)')
axes[1, 0].set_title('ZK phase velocity vs frequency')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Error
amp_err_pct = 100.0 * np.abs(fdtd_amp_ratio - ana_amp_ratio) / ana_amp_ratio
phase_err_deg = 180.0 / np.pi * np.abs(fdtd_phase_diff - ana_phase_diff)
axes[1, 1].semilogy(freqs, amp_err_pct, 'o-', label='Amplitude')
axes[1, 1].plot(freqs, phase_err_deg, 's-', label='Phase')
axes[1, 1].set_xlabel('Frequency (Hz)')
axes[1, 1].set_ylabel('Error')
axes[1, 1].set_title('FDTD vs analytical error')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig(fig_dir / 'fdtd_zk_uniform_1d_v1.png', dpi=150)
print(f"\nSaved: {fig_dir / 'fdtd_zk_uniform_1d_v1.png'}")
