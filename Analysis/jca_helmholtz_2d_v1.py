"""
JCA Helmholtz 2D — Frequency-domain acoustic solver for rigid-frame porous media.

Governing equation:
    div((1/rho_tilde) grad p) + (omega^2 / K_tilde) * p = S * delta(x - x_s)

Boundary conditions:
    - left/right: Dirichlet p = 0 with a quadratic sponge layer next to it
    - top/bottom: hard wall (Neumann dp/dy = 0)

Design variables per block:
    phi       — porosity
    sigma     — static flow resistivity [Pa s / m^2]
    alpha_inf — high-frequency tortuosity

Fixed JCA parameters:
    Lambda  = 30e-6 m
    Lambda' = 60e-6 m
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# --- physical constants (air at 300 K) ----------------------------------------
RHO0  = 1.225          # kg/m^3
C0    = 343.0          # m/s
P0    = 1.01325e5      # Pa
GAMMA = 1.4
ETA   = 1.84e-5        # dynamic viscosity [Pa s]
PR    = 0.71           # Prandtl number

# fixed JCA characteristic lengths
LAMBDA  = 30e-6
LAMBDA_ = 60e-6


# --- JCA effective properties -------------------------------------------------
def jca_properties(omega, phi, sigma, alpha_inf):
    """
    Return complex effective density rho_tilde and bulk modulus K_tilde.
    Inputs may be scalars or 2-D arrays. Shapes must broadcast.
    """
    phi       = np.asarray(phi, dtype=float)
    sigma     = np.asarray(sigma, dtype=float)
    alpha_inf = np.asarray(alpha_inf, dtype=float)

    # viscous dynamic density (Johnson-Champoux-Allard)
    rho_infty = RHO0 * alpha_inf / phi
    rho_tilde = rho_infty.astype(complex)

    sigma_pos = sigma > 0.0
    if np.any(sigma_pos):
        sp = sigma[sigma_pos]
        ph = phi[sigma_pos]
        ai = alpha_inf[sigma_pos]
        visc_term = sp * ph / (1j * omega * RHO0 * ai)
        sq_arg = 1.0 + 1j * (4.0 * ai**2 * ETA * RHO0 * omega) / (sp**2 * ph**2 * LAMBDA**2)
        rho_tilde[sigma_pos] = (RHO0 * ai / ph) * (1.0 + visc_term * np.sqrt(sq_arg))

    # thermal dynamic bulk modulus (Champoux-Allard)
    a  = (8.0 * ETA) / (1j * omega * RHO0 * PR * LAMBDA_**2)
    b  = 1.0 + 1j * (omega * RHO0 * PR * LAMBDA_**2) / (16.0 * ETA)
    denom = 1.0 + a * np.sqrt(b)
    bracket = GAMMA - (GAMMA - 1.0) / denom
    K_tilde = (GAMMA * P0 / phi) / bracket

    return rho_tilde, K_tilde


# --- sponge absorbing layer ---------------------------------------------------
def build_sponge(Nx, Ny, sponge_width, sigma_max):
    """Quadratic sponge profile on left/right edges."""
    sponge = np.zeros((Nx, Ny), dtype=float)
    if sponge_width <= 0:
        return sponge

    w = float(sponge_width)
    for i in range(1, sponge_width + 1):
        d = (w - i) / w
        sponge[i, :] = sigma_max * d * d

    for i in range(Nx - 1 - sponge_width, Nx - 1):
        d = (i - (Nx - 1 - sponge_width)) / w
        sponge[i, :] = sigma_max * d * d

    return sponge


# --- matrix assembly ----------------------------------------------------------
def build_matrix(omega, rho_tilde, K_tilde, dx, sponge=None):
    """Build sparse Helmholtz matrix with Robin (left/right) and Neumann (top/bottom) BCs."""
    Nx, Ny = rho_tilde.shape
    N = Nx * Ny
    dx2 = dx * dx

    inv_rho = 1.0 / rho_tilde
    diag_helm = omega**2 / K_tilde
    k0 = omega * np.sqrt(rho_tilde / K_tilde)

    row, col, data = [], [], []

    for i in range(Nx):
        for j in range(Ny):
            idx = i * Ny + j

            # Robin radiation on left domain edge
            if i == 0:
                row.append(idx); col.append(idx); data.append(-1.0 / dx - 1j * k0[i, j])
                row.append(idx); col.append(idx + Ny); data.append(1.0 / dx)
                continue

            # Robin radiation on right domain edge
            if i == Nx - 1:
                row.append(idx); col.append(idx); data.append(1.0 / dx + 1j * k0[i, j])
                row.append(idx); col.append(idx - Ny); data.append(-1.0 / dx)
                continue

            coeff = 0.0 + 0.0j

            # x-derivative (variable 1/rho)
            irhx_minus = 0.5 * (inv_rho[i - 1, j] + inv_rho[i, j])
            irhx_plus  = 0.5 * (inv_rho[i, j] + inv_rho[i + 1, j])

            coeff += -(irhx_minus + irhx_plus) / dx2
            row.append(idx); col.append((i - 1) * Ny + j); data.append(irhx_minus / dx2)
            row.append(idx); col.append((i + 1) * Ny + j); data.append(irhx_plus  / dx2)

            # y-derivative with hard-wall Neumann on top/bottom
            if j == 0:
                irhy = 0.5 * (inv_rho[i, j] + inv_rho[i, j + 1])
                coeff += -2.0 * irhy / dx2
                row.append(idx); col.append(i * Ny + (j + 1)); data.append(2.0 * irhy / dx2)
            elif j == Ny - 1:
                irhy = 0.5 * (inv_rho[i, j - 1] + inv_rho[i, j])
                coeff += -2.0 * irhy / dx2
                row.append(idx); col.append(i * Ny + (j - 1)); data.append(2.0 * irhy / dx2)
            else:
                irhy_minus = 0.5 * (inv_rho[i, j - 1] + inv_rho[i, j])
                irhy_plus  = 0.5 * (inv_rho[i, j] + inv_rho[i, j + 1])
                coeff += -(irhy_minus + irhy_plus) / dx2
                row.append(idx); col.append(i * Ny + (j - 1)); data.append(irhy_minus / dx2)
                row.append(idx); col.append(i * Ny + (j + 1)); data.append(irhy_plus  / dx2)

            # Helmholtz diagonal with sponge damping
            coeff += diag_helm[i, j]
            row.append(idx); col.append(idx); data.append(coeff)

    A = sp.csr_matrix((data, (row, col)), shape=(N, N), dtype=complex)
    return A


def solve_helmholtz(omega, phi, sigma, alpha_inf, dx,
                    src_pos, src_amp=1.0):
    """
    Solve the 2-D JCA Helmholtz equation on a uniform grid.

    Parameters
    ----------
    omega : float
        Angular frequency [rad/s].
    phi, sigma, alpha_inf : 2-D ndarray of shape (Nx, Ny)
        Porous material fields.
    dx : float
        Grid spacing [m].
    src_pos : tuple (i, j)
        Source grid index.
    src_amp : complex
        Source amplitude.
    Returns
    -------
    p : ndarray of shape (Nx, Ny)
        Complex pressure field.
    """
    rho_tilde, K_tilde = jca_properties(omega, phi, sigma, alpha_inf)

    A = build_matrix(omega, rho_tilde, K_tilde, dx)

    Nx, Ny = phi.shape
    N = Nx * Ny
    b = np.zeros(N, dtype=complex)

    # hard point source
    src_idx = src_pos[0] * Ny + src_pos[1]
    A = A.tolil()
    A[src_idx, :] = 0.0
    A[src_idx, src_idx] = 1.0
    A = A.tocsr()
    b[src_idx] = src_amp

    p_flat = spla.spsolve(A, b)
    return p_flat.reshape((Nx, Ny))


# --- probes / post-processing -------------------------------------------------
def probe_value(p, pos):
    """Complex pressure at a single grid point."""
    return p[pos[0], pos[1]]


def probe_energy(p, corner, size):
    """Integrated squared pressure magnitude over a rectangular window."""
    i0, j0 = corner
    ni, nj = size
    return np.sum(np.abs(p[i0:i0+ni, j0:j0+nj])**2)


# --- sanity checks + demo -----------------------------------------------------
def measure_wavelength(p, dx, axis=0):
    """Crude wavelength estimate from zero-crossing spacing of Re(p)."""
    line = p[:, p.shape[1] // 2].real if axis == 0 else p[p.shape[0] // 2, :].real
    zeros = []
    for k in range(len(line) - 1):
        if line[k] == 0.0:
            zeros.append(float(k))
        elif line[k] * line[k + 1] < 0.0:
            zeros.append(k + line[k] / (line[k] - line[k + 1]))
    if len(zeros) < 4:
        return None
    spacings = np.diff(zeros[2:-2])
    return 2.0 * np.mean(spacings) * dx


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    freq = 5000.0
    omega = 2.0 * np.pi * freq
    Lx, Ly = 0.400, 0.100

    print("=" * 60)
    print("SANITY CHECK 1: JCA properties for phi=1, sigma=0, alpha_inf=1")
    print("=" * 60)
    rho_t, K_t = jca_properties(omega, 1.0, 0.0, 1.0)
    c_t = np.sqrt(K_t / rho_t)
    print(f"  rho_tilde = {rho_t:.4f}  (expected {RHO0:.4f})")
    print(f"  K_tilde   = {K_t:.4f}")
    print(f"  c_tilde   = {c_t:.2f}")
    print("  note: finite Lambda' makes K_tilde differ from gamma*P0")

    print("\n" + "=" * 60)
    print("SANITY CHECK 2: wavelength in JCA waveguide")
    print("=" * 60)
    dx = 0.002
    Nx, Ny = int(Lx / dx), int(Ly / dx)
    phi = np.ones((Nx, Ny))
    sigma = np.zeros((Nx, Ny))
    alpha_inf = np.ones((Nx, Ny))
    src_pos = (int(0.100 / dx), Ny // 2)

    p = solve_helmholtz(omega, phi, sigma, alpha_inf, dx, src_pos)
    lam_num = measure_wavelength(p, dx)
    lam_th = c_t.real / freq
    print(f"  numerical wavelength: {lam_num*1000:.2f} mm")
    print(f"  JCA wavelength:       {lam_th*1000:.2f} mm")
    print(f"  error: {abs(lam_num - lam_th)/lam_th*100:.2f}%")

    print("\n" + "=" * 60)
    print("SANITY CHECK 3: grid convergence")
    print("=" * 60)
    probe_x, probe_y = 0.350, Ly / 2
    results = []
    for dx in [0.002, 0.001]:
        Nx, Ny = int(Lx / dx), int(Ly / dx)
        phi = np.ones((Nx, Ny))
        sigma = np.zeros((Nx, Ny))
        alpha_inf = np.ones((Nx, Ny))
        src_pos = (int(0.100 / dx), Ny // 2)
        p = solve_helmholtz(omega, phi, sigma, alpha_inf, dx, src_pos)
        ip, jp = int(probe_x / dx), int(probe_y / dx)
        results.append((dx, np.abs(p[ip, jp])))
        print(f"  dx={dx*1000:.1f} mm  |p| at probe = {results[-1][1]:.6e}")
    print(f"  relative change = {abs(results[1][1] - results[0][1]) / results[0][1] * 100:.2f}%")

    print("\n" + "=" * 60)
    print("DEMO: waveguide with porous insert")
    print("=" * 60)
    dx = 0.002
    Nx, Ny = int(Lx / dx), int(Ly / dx)
    phi = np.ones((Nx, Ny))
    sigma = np.zeros((Nx, Ny))
    alpha_inf = np.ones((Nx, Ny))
    phi[80:120, 15:35] = 0.60
    sigma[80:120, 15:35] = 1e4
    src_pos = (int(0.100 / dx), Ny // 2)
    p = solve_helmholtz(omega, phi, sigma, alpha_inf, dx, src_pos)
    A = probe_energy(p, (170, 30), (5, 8))
    B = probe_energy(p, (170, 12), (5, 8))
    print(f"  probe A energy: {A:.6e}")
    print(f"  probe B energy: {B:.6e}")
    print(f"  fraction to A: {A / (A + B + 1e-30):.4f}")

    fig, ax = plt.subplots(figsize=(10, 3))
    vmax = np.abs(p).max()
    im = ax.imshow(p.T.real, origin='lower', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax,
                   extent=[0, Nx*dx*1000, 0, Ny*dx*1000])
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title(f'Re(p) at {freq/1000:.1f} kHz')
    fig.colorbar(im, ax=ax, label='Pressure [Pa]')
    fig.tight_layout()
    fig.savefig('jca_helmholtz_demo.png', dpi=150)
    print('  saved jca_helmholtz_demo.png')
