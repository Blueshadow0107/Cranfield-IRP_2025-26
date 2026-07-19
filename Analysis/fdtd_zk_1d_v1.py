#!/usr/bin/env python3
"""
fdtd_zk_1d_v1.py
================
One-dimensional FDTD validation of the Zwikker-Kosten (ZK) equivalent-fluid
model for sound propagation in a rigid-frame porous medium.

The ZK time-domain equations are:

    rho_eff * dv/dt + sigma * v = - dp/dx
    phi/p0 * dp/dt            = - dv/dx

where:
    rho_eff = k_s * rho0 / phi
    k_s     = structure factor / high-frequency tortuosity
    sigma   = static flow resistivity
    phi     = porosity
    p0      = ambient pressure

These give a complex frequency-domain wavenumber

    k(omega) = (omega / c_zk) * sqrt(1 + j * sigma / (omega * rho_eff))

with c_zk = sqrt(p0 / (k_s * rho0)).  The square root produces attenuation and
a frequency-dependent phase velocity.

Validation:
    A continuous sinusoidal pressure source is applied at the left boundary of
    a long duct.  Two simulations are run:
      (a) air everywhere (reference)
      (b) air with a uniform porous slab in the middle
    The transmitted complex amplitude is compared against the analytical
    transfer-matrix prediction for the air-porous-air slab.
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
SIGMA     = 1.0e4     # Pa s / m^2, static flow resistivity

RHO_EFF = K_S * RHO0 / PHI
K_EFF   = P0 / PHI
C_ZK    = np.sqrt(K_EFF / RHO_EFF)

print(f"ZK parameters: phi={PHI}, k_s={K_S}, sigma={SIGMA:.2e}")
print(f"  rho_eff = {RHO_EFF:.3f} kg/m^3")
print(f"  K_eff   = {K_EFF:.1f} Pa")
print(f"  c_zk    = {C_ZK:.1f} m/s")

# =============================================================================
# DOMAIN / GRID
# =============================================================================
LX       = 6.0        # m  (long enough for reflections to arrive late)
NX       = 3000
DX       = LX / NX
DT       = 0.95 * DX / C0

x = np.linspace(0.0, LX, NX)

# Slab geometry
SLAB_X0  = 2.0
SLAB_X1  = 2.5
SLAB_D   = SLAB_X1 - SLAB_X0
i0 = int(SLAB_X0 / DX)
i1 = int(SLAB_X1 / DX)

# Probe locations (in air, clear of slab)
PROBE_IN_X  = 1.0
PROBE_OUT_X = 3.5
probe_in_i  = int(PROBE_IN_X / DX)
probe_out_i = int(PROBE_OUT_X / DX)

# =============================================================================
# SOURCE
# =============================================================================
SRC_FREQ  = 1000.0     # Hz
SRC_AMP   = 1.0


def source_value(n):
    t = n * DT
    return SRC_AMP * np.sin(2.0 * np.pi * SRC_FREQ * t)


# =============================================================================
# FDTD UPDATE
# =============================================================================
def make_materials(has_slab):
    if has_slab:
        is_porous = np.zeros(NX, dtype=bool)
        is_porous[i0:i1] = True
    else:
        is_porous = np.zeros(NX, dtype=bool)

    phi_arr     = np.where(is_porous, PHI, 1.0)
    ks_arr      = np.where(is_porous, K_S, 1.0)
    sigma_arr   = np.where(is_porous, SIGMA, 0.0)
    rho_eff_arr = ks_arr * RHO0 / phi_arr
    # Air: adiabatic bulk modulus rho0*c0^2; porous ZK: isothermal p0/phi.
    K_eff_arr = np.where(is_porous, P0 / phi_arr, RHO0 * C0**2)
    return rho_eff_arr, K_eff_arr, sigma_arr


def run_fdtd(has_slab, n_steps):
    """Run 1-D ZK FDTD.  Returns probe histories and full pressure field."""
    rho_eff_arr, K_eff_arr, sigma_arr = make_materials(has_slab)

    p = np.zeros(NX)
    v = np.zeros(NX + 1)

    history_in  = []
    history_out = []

    for n in range(n_steps):
        v_new = v.copy()

        # velocity update on interior faces 1..NX-1
        dp_dx = (p[1:] - p[:-1]) / DX
        v_new[1:-1] = v[1:-1] + DT * (-dp_dx - sigma_arr[1:] * v[1:-1]) / rho_eff_arr[1:]

        # pressure update on centres 0..NX-1
        dv_dx = (v_new[1:] - v_new[:-1]) / DX
        p_new = p - DT * K_eff_arr * dv_dx

        # transparent source at left boundary:
        # incoming characteristic p + Z_air*v = 2*source(t)
        # outgoing characteristic p - Z_air*v = 0
        # => p = source(t), v = source(t)/Z_air
        Z_AIR = RHO0 * C0
        p_new[0] = source_value(n)
        v_new[0] = source_value(n) / Z_AIR

        # first-order Mur ABC at right boundary
        p_new[-1] = p[-2] + (C0 * DT - DX) / (C0 * DT + DX) * (p_new[-2] - p[-1])

        p, v = p_new, v_new

        history_in.append(p[probe_in_i])
        history_out.append(p[probe_out_i])

    return np.array(history_in), np.array(history_out), p


# =============================================================================
# ANALYTICAL TRANSFER MATRIX
# =============================================================================
def zk_wavenumber(omega):
    """Complex wavenumber in the ZK porous medium.

    Sign convention: physical fields vary as exp(j(omega t - k x)), so a
    positive attenuation constant corresponds to Im(k) < 0.
    """
    return (omega / C_ZK) * np.sqrt(1.0 - 1j * SIGMA / (omega * RHO_EFF))


def zk_impedance(omega):
    """Complex characteristic impedance in the ZK porous medium."""
    k = zk_wavenumber(omega)
    return (omega * RHO_EFF - 1j * SIGMA) / k


def layer_matrix(k, Z, d):
    kd = k * d
    return np.array([[np.cos(kd), 1j * Z * np.sin(kd)],
                     [1j / Z * np.sin(kd), np.cos(kd)]])


def _transfer_rightgoing(omega, x_out, x_in=0.0, is_air=False):
    """
    Complex pressure ratio p(x_out)/p(x_in) for a pure right-going wave
    injected at x_in.  If is_air is True, the whole domain is air.
    """
    Z_air = RHO0 * C0
    k_air = omega / C0
    if is_air:
        k_por, Z_por = k_air, Z_air
    else:
        k_por = zk_wavenumber(omega)
        Z_por = zk_impedance(omega)

    d_left  = max(SLAB_X0 - x_in, 0.0)
    d_right = max(x_out - SLAB_X1, 0.0)

    T = layer_matrix(k_air, Z_air, d_left) @ \
        layer_matrix(k_por, Z_por, SLAB_D) @ \
        layer_matrix(k_air, Z_air, d_right)

    A, B = T[0, 0], T[0, 1]
    # For pure right-going input at x_in and right-going output at x_out:
    # p_in = (A + B/Z_air) p_out
    return 1.0 / (A + B / Z_air)


def analytical_transmission(omega, x_out=PROBE_OUT_X):
    """
    Ratio of pressure at x_out with the slab to pressure at x_out in air only,
    for the same pure right-going source at x=0.
    """
    T_slab = _transfer_rightgoing(omega, x_out, 0.0, is_air=False)
    T_air  = _transfer_rightgoing(omega, x_out, 0.0, is_air=True)
    return T_slab / T_air


# =============================================================================
# RUN SIMULATIONS
# =============================================================================
n_periods = 20
n_steps = int(n_periods / (SRC_FREQ * DT))
print(f"Running {n_steps} steps ({n_periods} periods) ...")

hist_in_ref,  hist_out_ref,  p_ref  = run_fdtd(False, n_steps)
hist_in_slab, hist_out_slab, p_slab = run_fdtd(True,  n_steps)

t = np.arange(n_steps) * DT

# =============================================================================
# EXTRACT COMPLEX AMPLITUDES (lock-in on last 5 periods)
# =============================================================================
lock_start = int((n_periods - 5) / (SRC_FREQ * DT))
lock_t = t[lock_start:]
lock_ref = hist_out_ref[lock_start:]
lock_slab = hist_out_slab[lock_start:]

omega = 2.0 * np.pi * SRC_FREQ
cos_ref = np.cos(omega * lock_t)
sin_ref = np.sin(omega * lock_t)

A_ref  = 2.0 * np.mean(lock_ref * cos_ref)
B_ref  = 2.0 * np.mean(lock_ref * sin_ref)
A_slab = 2.0 * np.mean(lock_slab * cos_ref)
B_slab = 2.0 * np.mean(lock_slab * sin_ref)

amp_ref  = np.sqrt(A_ref**2 + B_ref**2)
amp_slab = np.sqrt(A_slab**2 + B_slab**2)
phase_ref  = np.angle(A_ref + 1j * B_ref)
phase_slab = np.angle(A_slab + 1j * B_slab)

fdtd_trans = (amp_slab / amp_ref) * np.exp(1j * (phase_slab - phase_ref))
fdtd_amp   = np.abs(fdtd_trans)
fdtd_phase = np.angle(fdtd_trans)

ana_trans  = analytical_transmission(omega, PROBE_OUT_X)
ana_amp    = np.abs(ana_trans)
ana_phase  = np.angle(ana_trans)

print("\n" + "=" * 60)
print(f"Frequency = {SRC_FREQ:.1f} Hz")
print(f"FDTD transmission:  amp={fdtd_amp:.4f}, phase={fdtd_phase:.4f} rad")
print(f"Analytical TMM:     amp={ana_amp:.4f}, phase={ana_phase:.4f} rad")
if ana_amp > 1e-12:
    print(f"Amplitude error:    {100*np.abs(fdtd_amp - ana_amp)/ana_amp:.2f}%")
print(f"Phase error:        {180/np.pi * np.abs(np.angle(np.exp(1j*(fdtd_phase - ana_phase)))):.2f} deg")

# =============================================================================
# PLOTS
# =============================================================================
fig_dir = Path(__file__).parent / 'figures'
fig_dir.mkdir(exist_ok=True)

fig, axes = plt.subplots(3, 1, figsize=(10, 9))

axes[0].plot(t * 1000, hist_out_ref,  label='air reference', alpha=0.7)
axes[0].plot(t * 1000, hist_out_slab, label='with slab', alpha=0.7)
axes[0].axvspan(SLAB_X0 / C0 * 1000, SLAB_X1 / C0 * 1000, alpha=0.1, color='gray')
axes[0].set_ylabel('p (Pa)')
axes[0].set_title('ZK 1-D slab: continuous-wave transmission')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(x, p_ref,  label='air reference', alpha=0.7)
axes[1].plot(x, p_slab, label='with slab', alpha=0.7)
axes[1].axvspan(SLAB_X0, SLAB_X1, alpha=0.1, color='gray')
axes[1].set_xlabel('x (m)')
axes[1].set_ylabel('p (Pa)')
axes[1].set_title('Final pressure field snapshot')
axes[1].legend()
axes[1].grid(True)

# dispersion curve
freqs = np.logspace(1, 4, 200)
omega_vec = 2.0 * np.pi * freqs
k_vec = zk_wavenumber(omega_vec)
c_phase = omega_vec / np.real(k_vec)
axes[2].semilogx(freqs, c_phase, label='ZK phase velocity')
axes[2].axhline(C0, color='gray', linestyle='--', label='air')
axes[2].axhline(C_ZK, color='red', linestyle='--', label='high-freq ZK limit')
axes[2].set_xlabel('Frequency (Hz)')
axes[2].set_ylabel('Phase velocity (m/s)')
axes[2].set_title('ZK dispersion: phase velocity vs frequency')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig(fig_dir / 'fdtd_zk_1d_v1.png', dpi=150)
print(f"\nSaved: {fig_dir / 'fdtd_zk_1d_v1.png'}")
