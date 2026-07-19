"""
1-D JCA transfer-matrix validation.
Analytical transmission of a uniform porous slab vs numerical Helmholtz solve.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

import jca_helmholtz_2d_v1 as jca


freq = 5000.0
omega = 2.0 * np.pi * freq

phi_slab, sigma_slab, alpha_inf_slab = 0.60, 1e4, 1.5
L_slab = 0.050

phi_air, sigma_air, alpha_inf_air = 1.0, 0.0, 1.0

L_air_left = 0.100
L_air_right = 0.200
L_total = L_air_left + L_slab + L_air_right

Nx = 400
dx = L_total / (Nx - 1)
x = np.linspace(0, L_total, Nx)


def make_materials(Nx):
    phi = np.full(Nx, phi_air)
    sigma = np.full(Nx, sigma_air)
    alpha_inf = np.full(Nx, alpha_inf_air)
    i1 = int(L_air_left / dx)
    i2 = int((L_air_left + L_slab) / dx)
    phi[i1:i2] = phi_slab
    sigma[i1:i2] = sigma_slab
    alpha_inf[i1:i2] = alpha_inf_slab
    return phi, sigma, alpha_inf


phi, sigma, alpha_inf = make_materials(Nx)
rho_tilde, K_tilde = jca.jca_properties(omega, phi, sigma, alpha_inf)


# analytical transfer-matrix transmission
rho1, K1 = jca.jca_properties(omega, np.array([phi_air]), np.array([sigma_air]), np.array([alpha_inf_air]))
rho2, K2 = jca.jca_properties(omega, np.array([phi_slab]), np.array([sigma_slab]), np.array([alpha_inf_slab]))

Z1 = np.sqrt(rho1[0] * K1[0])
Z2 = np.sqrt(rho2[0] * K2[0])
k1 = omega * np.sqrt(rho1[0] / K1[0])
k2 = omega * np.sqrt(rho2[0] / K2[0])

T_analytical = 2.0 / (2.0 * np.cos(k2 * L_slab)
                      + 1j * (Z2/Z1 + Z1/Z2) * np.sin(k2 * L_slab))

print("=" * 60)
print("1-D TRANSFER-MATRIX VALIDATION")
print("=" * 60)
print(f"f            = {freq:.1f} Hz")
print(f"slab         = phi={phi_slab}, sigma={sigma_slab:.2e}, alpha_inf={alpha_inf_slab}")
print(f"L_slab       = {L_slab*1000:.1f} mm")
print(f"Z1 (air)     = {Z1:.4f}")
print(f"Z2 (slab)    = {Z2:.4f}")
print(f"k1 (air)     = {k1:.2f}  -> lambda1 = {2*np.pi/k1.real*1000:.2f} mm")
print(f"k2 (slab)    = {k2:.2f}  -> lambda2 = {2*np.pi/k2.real*1000:.2f} mm")
print(f"|T_analytical| = {np.abs(T_analytical):.6f}")
print(f"arg(T_analytical) = {np.angle(T_analytical)*180/np.pi:.2f} deg")


# numerical 1-D Helmholtz solve
def solve_1d(omega, rho_tilde, K_tilde, dx):
    Nx = len(rho_tilde)
    inv_rho = 1.0 / rho_tilde
    k = omega * np.sqrt(rho_tilde / K_tilde)

    row, col, data = [], [], []
    b = np.zeros(Nx, dtype=complex)

    for i in range(Nx):
        if i == 0:
            row.append(i); col.append(i); data.append(1.0)
            b[i] = 1.0
            continue

        if i == Nx - 1:
            row.append(i); col.append(i); data.append(1.0/dx + 1j*k[i])
            row.append(i); col.append(i - 1); data.append(-1.0/dx)
            continue

        irhox_minus = 0.5 * (inv_rho[i - 1] + inv_rho[i])
        irhox_plus  = 0.5 * (inv_rho[i] + inv_rho[i + 1])

        diag = -(irhox_minus + irhox_plus) / dx**2 + omega**2 / K_tilde[i]
        row.append(i); col.append(i); data.append(diag)
        row.append(i); col.append(i - 1); data.append(irhox_minus / dx**2)
        row.append(i); col.append(i + 1); data.append(irhox_plus  / dx**2)

    A = sp.csr_matrix((data, (row, col)), shape=(Nx, Nx), dtype=complex)
    return spla.spsolve(A, b)


p_num = solve_1d(omega, rho_tilde, K_tilde, dx)


# extract right-going travelling-wave amplitudes
def extract_amplitude(p, x_loc, k):
    i0 = len(x_loc) // 3
    i1 = 2 * len(x_loc) // 3
    e0p = np.exp(-1j * k * x_loc[i0])
    e0m = np.exp( 1j * k * x_loc[i0])
    e1p = np.exp(-1j * k * x_loc[i1])
    e1m = np.exp( 1j * k * x_loc[i1])
    det = e0p * e1m - e0m * e1p
    return (p[i0] * e1m - p[i1] * e0m) / det


i_left  = (x > 0.010) & (x < L_air_left - 0.010)
i_right = (x > L_air_left + L_slab + 0.010) & (x < L_total - 0.010)

A_inc = extract_amplitude(p_num[i_left],  x[i_left],  k1)
A_trn = extract_amplitude(p_num[i_right], x[i_right] - (L_air_left + L_slab), k1)

# account for attenuation/propagation from x=0 to slab entrance
T_numerical = A_trn / (A_inc * np.exp(-1j * k1 * L_air_left))

print(f"\n|A_incident|  = {np.abs(A_inc):.6f}")
print(f"|A_transmitted| = {np.abs(A_trn):.6f}")
print(f"|T_numerical|   = {np.abs(T_numerical):.6f}")
print(f"arg(T_numerical) = {np.angle(T_numerical)*180/np.pi:.2f} deg")
print(f"\nerror in |T| = {abs(abs(T_numerical) - abs(T_analytical))/abs(T_analytical)*100:.2f}%")
print(f"error in phase = {abs(np.angle(T_numerical) - np.angle(T_analytical))*180/np.pi:.2f} deg")


fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

ax = axes[0]
ax.plot(x * 1000, np.abs(p_num), 'b-', label='|p| numerical')
ax.axvspan(L_air_left*1000, (L_air_left+L_slab)*1000, color='gray', alpha=0.2, label='slab')
ax.set_ylabel('|p|')
ax.set_title('1-D JCA slab transmission')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(x * 1000, np.angle(p_num) * 180 / np.pi, 'r-', label='arg(p) numerical')
ax.axvspan(L_air_left*1000, (L_air_left+L_slab)*1000, color='gray', alpha=0.2)
ax.set_xlabel('x [mm]')
ax.set_ylabel('phase [deg]')
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('validate_jca_1d.png', dpi=150)
print("\nsaved validate_jca_1d.png")
