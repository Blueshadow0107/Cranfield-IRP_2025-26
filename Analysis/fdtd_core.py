"""
fdtd_core.py

Master 2D acoustic FDTD solver with selectable boundary conditions,
source types, and optional thermal field solver.

Grid: cell-centered pressure p, face-normal velocities u, v (Yee staggered)
  p[i,j] at x=(i+0.5)*dx, y=(j+0.5)*dy     shape: (nx, ny)
  u[i,j] at x=i*dx, y=(j+0.5)*dy            shape: (nx+1, ny)
  v[i,j] at x=(i+0.5)*dx, y=j*dy            shape: (nx, ny+1)

Boundary conditions (per face):
  'absorbing'       : impedance match  u = +/- p/(rho0*c)   (Engquist-Majda / Mur)
  'hard_wall'       : zero normal velocity  u = 0
  'pressure_release': p = 0 at face  ->  u -= (dt/rho0)*2*p/dx

Sources:
  'continuous_sine' : sinusoidal plane wave
  'gaussian_pulse'  : narrow Gaussian envelope + carrier
  'broadband_pulse' : wide Gaussian envelope + carrier (for FFT/eigenfreq)
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


# =============================================================================
# THERMAL SOLVER
# =============================================================================

def solve_temperature(sigma, dx, D, T0):
    """
    Solve steady-state thermal Poisson equation:  nabla^2 T = -sigma/D
    with Dirichlet T = T0 on all boundaries.

    Parameters
    ----------
    sigma : ndarray, shape (nx, ny)
        Heat source field [W/m^3].
    dx : float
        Grid spacing [m].
    D : float
        Thermal diffusivity [m^2/s].
    T0 : float
        Boundary temperature [K].

    Returns
    -------
    T : ndarray, shape (nx, ny)
        Temperature field [K].
    """
    nx, ny = sigma.shape
    N = nx * ny
    rows, cols, data = [], [], []

    for i in range(nx):
        for j in range(ny):
            k = i * ny + j
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                rows.append(k); cols.append(k); data.append(1.0)
            else:
                rows.append(k); cols.append(k); data.append(-4.0)
                rows.append(k); cols.append((i + 1) * ny + j); data.append(1.0)
                rows.append(k); cols.append((i - 1) * ny + j); data.append(1.0)
                rows.append(k); cols.append(i * ny + (j + 1)); data.append(1.0)
                rows.append(k); cols.append(i * ny + (j - 1)); data.append(1.0)

    A = csr_matrix((data, (rows, cols)), shape=(N, N))
    b = np.zeros(N)
    for i in range(nx):
        for j in range(ny):
            k = i * ny + j
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                b[k] = T0
            else:
                b[k] = -sigma[i, j] * dx ** 2 / D

    T_flat = spsolve(A, b)
    return T_flat.reshape((nx, ny))


def build_c_field_from_temperature(T, c0, T0):
    """Convert temperature field to sound speed: c = c0 * sqrt(T/T0)."""
    return c0 * np.sqrt(T / T0)


def build_heat_source_strips(nx, ny, dx, sigma_amps, positions, width):
    """
    Build 2D heat source with vertical strip heaters.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions.
    dx : float
        Cell size [m].
    sigma_amps : array_like
        Heat source amplitude per strip [W/m^3].
    positions : array_like
        x-position of each strip centre [m].
    width : float
        Strip width [m].

    Returns
    -------
    sigma : ndarray, shape (nx, ny)
    """
    sigma = np.zeros((nx, ny))
    half_w_cells = int(width / (2 * dx))
    for amp, pos in zip(sigma_amps, positions):
        i_c = int(pos / dx)
        i_s = max(0, i_c - half_w_cells)
        i_e = min(nx, i_c + half_w_cells)
        sigma[i_s:i_e, :] = amp
    return sigma


# =============================================================================
# ACOUSTIC FDTD SOLVER
# =============================================================================

class FDTDSolver:
    """
    2D staggered leapfrog acoustic FDTD solver.
    """

    def __init__(self, nx, ny, dx, c0=343.0, rho0=1.225, T0=300.0):
        """
        Parameters
        ----------
        nx, ny : int
            Number of cells in x and y.
        dx : float
            Cell size [m].  (Assumes dx == dy.)
        c0, rho0, T0 : float
            Reference sound speed [m/s], density [kg/m^3], temperature [K].
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dx
        self.c0 = c0
        self.rho0 = rho0
        self.T0 = T0

        # State arrays
        self.p = np.zeros((nx, ny))
        self.u = np.zeros((nx + 1, ny))
        self.v = np.zeros((nx, ny + 1))

        # Sound speed field (must be set before running)
        self.c_field = None
        self.c2 = None

        # Time stepping
        self.dt = None
        self.n_steps = 0
        self.time = 0.0

        # Boundary conditions
        self.bc = {
            'left': 'absorbing',
            'right': 'absorbing',
            'top': 'absorbing',
            'bottom': 'absorbing',
        }

        # Source configuration
        self.source = None

        # Recording
        self.record_history = []
        self.time_history = []

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def set_c_field(self, c_field):
        """Set sound speed field and compute derived quantities."""
        if c_field.shape != (self.nx, self.ny):
            raise ValueError(f"c_field shape {c_field.shape} does not match grid ({self.nx},{self.ny})")
        self.c_field = c_field.astype(float)
        self.c2 = self.c_field ** 2
        c_max = float(self.c_field.max())
        self.dt = 0.9 * self.dx / (c_max * np.sqrt(2))

    def set_bc(self, left=None, right=None, top=None, bottom=None):
        """Set boundary condition per face.  Options: 'absorbing', 'hard_wall', 'pressure_release'."""
        if left is not None:
            self.bc['left'] = left
        if right is not None:
            self.bc['right'] = right
        if top is not None:
            self.bc['top'] = top
        if bottom is not None:
            self.bc['bottom'] = bottom

    def set_source(self, source_type, i_start, i_end, **kwargs):
        """
        Configure acoustic source.

        Parameters
        ----------
        source_type : str
            'continuous_sine', 'gaussian_pulse', or 'broadband_pulse'.
        i_start, i_end : int
            x-cell range for source injection (inclusive start, exclusive end).
        kwargs :
            'continuous_sine'  ->  amplitude, frequency
            'gaussian_pulse'   ->  amplitude, f0, tau, t0
            'broadband_pulse'  ->  amplitude, f0, tau, t0
        """
        self.source = {
            'type': source_type,
            'i_start': i_start,
            'i_end': i_end,
            'params': kwargs,
        }

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _source_amplitude(self, t):
        """Compute source amplitude at time t."""
        if self.source is None:
            return 0.0
        typ = self.source['type']
        prm = self.source['params']

        if typ == 'continuous_sine':
            return prm['amplitude'] * np.sin(2.0 * np.pi * prm['frequency'] * t)

        elif typ in ('gaussian_pulse', 'broadband_pulse'):
            amp = prm['amplitude']
            f0 = prm['f0']
            tau = prm['tau']
            t0 = prm['t0']
            envelope = np.exp(-((t - t0) / tau) ** 2)
            return amp * envelope * np.sin(2.0 * np.pi * f0 * (t - t0))

        else:
            raise ValueError(f"Unknown source type: {typ}")

    def _apply_bc(self):
        """Apply boundary conditions to velocity fields."""
        nx, ny = self.nx, self.ny
        dt, rho0 = self.dt, self.rho0
        dx, dy = self.dx, self.dy
        p, u, v = self.p, self.u, self.v
        c = self.c_field

        # --- Left face (x = 0) ---
        bc = self.bc['left']
        if bc == 'absorbing':
            u[0, :] = -p[0, :] / (rho0 * c[0, :])
        elif bc == 'hard_wall':
            u[0, :] = 0.0
        elif bc == 'pressure_release':
            u[0, :] -= (dt / rho0) * 2.0 * p[0, :] / dx
        else:
            raise ValueError(f"Unknown BC: {bc}")

        # --- Right face (x = L) ---
        bc = self.bc['right']
        if bc == 'absorbing':
            u[nx, :] = p[nx - 1, :] / (rho0 * c[nx - 1, :])
        elif bc == 'hard_wall':
            u[nx, :] = 0.0
        elif bc == 'pressure_release':
            # Note: += because ∂p/∂x is negative at right boundary
            u[nx, :] += (dt / rho0) * 2.0 * p[nx - 1, :] / dx
        else:
            raise ValueError(f"Unknown BC: {bc}")

        # --- Bottom face (y = 0) ---
        bc = self.bc['bottom']
        if bc == 'absorbing':
            v[:, 0] = -p[:, 0] / (rho0 * c[:, 0])
        elif bc == 'hard_wall':
            v[:, 0] = 0.0
        elif bc == 'pressure_release':
            v[:, 0] -= (dt / rho0) * 2.0 * p[:, 0] / dy
        else:
            raise ValueError(f"Unknown BC: {bc}")

        # --- Top face (y = L) ---
        bc = self.bc['top']
        if bc == 'absorbing':
            v[:, ny] = p[:, ny - 1] / (rho0 * c[:, ny - 1])
        elif bc == 'hard_wall':
            v[:, ny] = 0.0
        elif bc == 'pressure_release':
            # Note: += because ∂p/∂y is negative at top boundary
            v[:, ny] += (dt / rho0) * 2.0 * p[:, ny - 1] / dy
        else:
            raise ValueError(f"Unknown BC: {bc}")

    # -------------------------------------------------------------------------
    # Time stepping
    # -------------------------------------------------------------------------

    def step(self):
        """Advance one leapfrog step."""
        if self.c_field is None:
            raise RuntimeError("Sound speed field not set. Call set_c_field() first.")

        nx, ny = self.nx, self.ny
        dt, rho0 = self.dt, self.rho0
        dx, dy = self.dx, self.dy
        p, u, v = self.p, self.u, self.v
        c2 = self.c2

        # --- Update interior velocities ---
        u[1:nx, :] -= (dt / rho0) * (p[1:nx, :] - p[0:nx - 1, :]) / dx
        v[:, 1:ny] -= (dt / rho0) * (p[:, 1:ny] - p[:, 0:ny - 1]) / dy

        # --- Apply boundary conditions ---
        self._apply_bc()

        # --- Update pressure ---
        du_dx = (u[1:nx + 1, :] - u[0:nx, :]) / dx
        dv_dy = (v[:, 1:ny + 1] - v[:, 0:ny]) / dy
        p -= dt * rho0 * c2 * (du_dx + dv_dy)

        # --- Source injection ---
        if self.source is not None:
            src = self._source_amplitude(self.time)
            i0, i1 = self.source['i_start'], self.source['i_end']
            p[i0:i1, :] += src

        self.time += dt
        self.n_steps += 1

    def run(self, n_steps, record_interval=None, probe_coords=None):
        """
        Run simulation for n_steps.

        Parameters
        ----------
        n_steps : int
        record_interval : int or None
            If set, record full p-field every `record_interval` steps.
        probe_coords : tuple or None
            (i, j) cell index to record p at every step.

        Returns
        -------
        probe : ndarray or None
            Time series at probe location.
        snapshots : list
            List of (step, p_field) tuples if record_interval is set.
        """
        probe = [] if probe_coords is not None else None
        snapshots = [] if record_interval is not None else None

        for _ in range(n_steps):
            self.step()

            if probe_coords is not None:
                i, j = probe_coords
                probe.append(float(self.p[i, j]))

            if record_interval is not None and self.n_steps % record_interval == 0:
                snapshots.append((self.n_steps, self.p.copy()))

        if probe is not None:
            probe = np.array(probe)
        return probe, snapshots

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def reset_state(self):
        """Zero all state arrays and counters."""
        self.p.fill(0.0)
        self.u.fill(0.0)
        self.v.fill(0.0)
        self.time = 0.0
        self.n_steps = 0
        self.record_history = []
        self.time_history = []

    def compute_energy(self):
        """
        Compute total acoustic energy:  E = sum( p^2/(2*rho0*c^2) + 0.5*rho0*(u^2+v^2) )
        Returns kinetic, potential, and total energy.
        """
        nx, ny = self.nx, self.ny
        dt, rho0 = self.dt, self.rho0
        dx = self.dx
        p, u, v = self.p, self.u, self.v
        c2 = self.c2

        # Potential energy:  p^2 / (2 * rho0 * c^2)  averaged over cell volume
        e_pot = np.sum(p ** 2 / (2.0 * rho0 * c2)) * dx * dx

        # Kinetic energy:  0.5 * rho0 * (u^2 + v^2)
        # u lives on vertical faces, v on horizontal faces.
        # Average u^2 and v^2 to cell centers for consistency.
        u_avg = 0.5 * (u[0:nx, :] ** 2 + u[1:nx + 1, :] ** 2)
        v_avg = 0.5 * (v[:, 0:ny] ** 2 + v[:, 1:ny + 1] ** 2)
        e_kin = 0.5 * rho0 * np.sum(u_avg + v_avg) * dx * dx

        return e_kin, e_pot, e_kin + e_pot

    def get_courant_number(self):
        """Return maximum Courant number: c_max * dt / dx."""
        if self.c_field is None:
            return None
        return self.c_field.max() * self.dt / self.dx
