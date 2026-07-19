#!/usr/bin/env python3
"""
fdtd_cylinder_array_scatter_v1.py
==================================
Acoustic scattering from a 2D array of hard-wall cylinders using the
existing FDTDSolver from fdtd_core.py.

Parameter sweep: cylinder diameter D and centre-to-centre spacing S.
Measure transmitted acoustic power downstream of the array.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import the validated FDTD core
sys.path.insert(0, str(Path(__file__).parent))
from fdtd_core import FDTDSolver


# =============================================================================
# OBSTACLE-AWARE FDTD SOLVER
# =============================================================================

class ObstacleFDTDSolver(FDTDSolver):
    """
    Extend FDTDSolver with hard-wall cylinder obstacles.

    Obstacles are represented by a cell-centred boolean mask. After each
    leapfrog step, all face-normal velocities adjacent to an obstacle cell
    are zeroed, enforcing zero normal velocity at the obstacle boundary
    (no-slip / hard-wall condition).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obstacle_mask = np.zeros((self.nx, self.ny), dtype=bool)

    def add_cylinder(self, cx, cy, radius):
        """
        Add a circular hard-wall obstacle centred at (cx, cy) with given radius.
        All arguments in metres.
        """
        x = (np.arange(self.nx) + 0.5) * self.dx
        y = (np.arange(self.ny) + 0.5) * self.dy
        X, Y = np.meshgrid(x, y, indexing='ij')
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        self.obstacle_mask |= (dist <= radius)

    def step(self):
        """Advance one leapfrog step with hard-wall obstacle enforcement."""
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

        # --- Enforce no-slip on obstacle faces BEFORE pressure update ---
        if np.any(self.obstacle_mask):
            self.u[:-1, :][self.obstacle_mask] = 0.0
            self.u[1:, :][self.obstacle_mask] = 0.0
            self.v[:, :-1][self.obstacle_mask] = 0.0
            self.v[:, 1:][self.obstacle_mask] = 0.0

        # --- Apply domain boundary conditions ---
        self._apply_bc()

        # --- Update pressure using velocities that already respect obstacles ---
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


# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# Grid
NX, NY = 400, 100
LX = 1.0
DX = LX / NX
DY = DX

# Air at 300 K
RHO0 = 1.225
C0 = 343.0

# Source: continuous sine at 2 kHz
SRC_FREQ = 2000.0
SRC_AMP = 1.0
SRC_CELLS = [3, 4, 5]

# Probes
PROBE_X = [0.75, 0.90]  # downstream of cylinder array
PROBE_Y_CENTRE = 0.5 * NY * DY

# Duration (cover ~20 source periods)
DURATION = 20.0 / SRC_FREQ

# Cylinder array parameters to sweep
DIAMETERS = np.array([0.02, 0.04, 0.06, 0.08])  # m
SPACINGS = np.array([0.10, 0.15, 0.20, 0.25])    # centre-to-centre, m
N_CYLINDERS = 5
CYLINDER_Y = 0.5 * NY * DY  # centre of channel


def build_array(cx_start, spacing, diameter, y_centre):
    """Return list of (cx, cy, radius) for a row of cylinders."""
    radius = diameter / 2.0
    cylinders = []
    for k in range(N_CYLINDERS):
        cx = cx_start + k * spacing
        cylinders.append((cx, y_centre, radius))
    return cylinders


def run_scattering(diameter, spacing, cx_start=0.35):
    """Run one scattering simulation. Return transmitted power metric."""
    solver = ObstacleFDTDSolver(NX, NY, DX, c0=C0, rho0=RHO0)

    # Uniform sound speed
    c_field = np.full((NX, NY), C0)
    solver.set_c_field(c_field)

    # Absorbing top/bottom, source on left, absorbing outlet
    solver.set_bc(left='hard_wall', right='absorbing',
                  top='absorbing', bottom='absorbing')

    # Add cylinders
    for cx, cy, r in build_array(cx_start, spacing, diameter, CYLINDER_Y):
        solver.add_cylinder(cx, cy, r)

    # Continuous sine source near left wall
    solver.set_source('continuous_sine', SRC_CELLS[0], SRC_CELLS[-1] + 1,
                      amplitude=SRC_AMP, frequency=SRC_FREQ)

    n_steps = int(DURATION / solver.dt)
    discard = int(0.5 * n_steps)

    probe_i = [int(x / DX) for x in PROBE_X]
    probe_j = int(PROBE_Y_CENTRE / DY)

    p_probe = {x: [] for x in PROBE_X}

    for n in range(n_steps):
        solver.step()
        for x, i in zip(PROBE_X, probe_i):
            p_probe[x].append(float(solver.p[i, probe_j]))

    # Steady-state RMS pressure as transmission metric
    results = {}
    for x in PROBE_X:
        p_ss = np.array(p_probe[x][discard:])
        results[x] = np.sqrt(np.mean(p_ss ** 2))

    return results, solver


def main():
    fig_dir = Path(__file__).parent / 'figures'
    fig_dir.mkdir(exist_ok=True)

    print("Cylinder-array acoustic scattering sweep")
    print(f"Grid: {NX}x{NY}, dx={DX:.4f} m, source f={SRC_FREQ/1e3:.1f} kHz")
    print(f"Sweeping diameters {DIAMETERS} m and spacings {SPACINGS} m")
    print()

    transmission = np.zeros((len(DIAMETERS), len(SPACINGS)))

    for i, D in enumerate(DIAMETERS):
        for j, S in enumerate(SPACINGS):
            print(f"  Running D={D:.2f} m, S={S:.2f} m ...", end=' ', flush=True)
            res, solver = run_scattering(D, S)
            transmission[i, j] = res[PROBE_X[-1]]
            print(f"probe power = {transmission[i, j]:.4f}")

    # Normalise by no-obstacle reference
    print("  Running reference (no cylinders) ...", end=' ', flush=True)
    ref_res, _ = run_scattering(0.0, 0.0)
    ref_power = ref_res[PROBE_X[-1]]
    print(f"ref power = {ref_power:.4f}")

    transmission_db = 20.0 * np.log10(transmission / ref_power + 1e-12)

    # Plot sweep
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    im = ax.imshow(transmission_db, origin='lower', aspect='auto', cmap='viridis',
                   extent=[SPACINGS[0] - 0.5 * (SPACINGS[1] - SPACINGS[0]),
                           SPACINGS[-1] + 0.5 * (SPACINGS[1] - SPACINGS[0]),
                           DIAMETERS[0] - 0.5 * (DIAMETERS[1] - DIAMETERS[0]),
                           DIAMETERS[-1] + 0.5 * (DIAMETERS[1] - DIAMETERS[0])])
    ax.set_xlabel('Spacing S (m)')
    ax.set_ylabel('Diameter D (m)')
    ax.set_title('Transmission loss (dB) downstream of cylinder array')
    fig.colorbar(im, ax=ax, label='dB')

    ax = axes[1]
    for i, D in enumerate(DIAMETERS):
        ax.plot(SPACINGS, transmission_db[i, :], 'o-', label=f'D={D:.2f} m')
    ax.set_xlabel('Spacing S (m)')
    ax.set_ylabel('Transmission loss (dB)')
    ax.set_title('Spacing sweep per diameter')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / 'cylinder_array_sweep_v1.png', dpi=150)
    print(f"\nSaved: {fig_dir / 'cylinder_array_sweep_v1.png'}")

    # Save a snapshot of one configuration
    D_snap = DIAMETERS[2]
    S_snap = SPACINGS[2]
    _, solver_snap = run_scattering(D_snap, S_snap)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(solver_snap.p.T, origin='lower', cmap='RdBu_r',
                   extent=[0, LX * 1000, 0, NY * DY * 1000],
                   vmin=-0.5, vmax=0.5)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(f'Pressure field snapshot: D={D_snap:.2f} m, S={S_snap:.2f} m')
    fig.colorbar(im, ax=ax, label='p')
    plt.tight_layout()
    plt.savefig(fig_dir / 'cylinder_array_snapshot_v1.png', dpi=150)
    print(f"Saved: {fig_dir / 'cylinder_array_snapshot_v1.png'}")


if __name__ == '__main__':
    main()
