#!/usr/bin/env python3
"""
fdtd_mxv_porous_v1.py
=====================
Trainable acoustic MxV benchmark using a continuous porous-medium sound-speed
map instead of discrete hard-wall cylinders.

The design variables are a coarse grid of sound-speed modulation values in the
central region of the domain.  These are smoothly interpolated onto the FDTD
grid to create a continuous c_field(x,y).  Differential evolution tunes the
coarse grid so that the measured transmission matrix matches a target matrix.

Physical picture:
    c_field(x,y) = c0 * (1 - alpha(x,y))

where alpha(x,y) in [0, alpha_max] represents the local porosity-induced
reduction of sound speed.  alpha = 0 is free air; alpha = alpha_max is the
slowest (most porous) region.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.optimize import differential_evolution
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).parent))
from fdtd_core import FDTDSolver


# =============================================================================
# PHYSICAL / NUMERICAL PARAMETERS
# =============================================================================

NX, NY = 240, 120          # grid cells
LX = 0.60                  # domain length [m]
DX = LX / NX               # cell size [m]
DY = DX

RHO0 = 1.225               # air density [kg/m^3]
C0 = 343.0                 # free-air sound speed [m/s]
ALPHA_MAX = 0.50           # max sound-speed reduction: c_min = c0*(1-alpha_max)

SRC_FREQ = 2000.0          # source frequency [Hz]
SRC_AMP = 1.0              # source amplitude [Pa]
SRC_WIDTH_CELLS = 3        # vertical extent of source port

# Probe ports: centres in metres
PROBE_POSITIONS = {
    0: {"x": 0.45, "y": 0.10},
    1: {"x": 0.45, "y": 0.22},
}

SOURCE_POSITIONS = {
    0: {"y": 0.10},
    1: {"y": 0.22},
}

N_PERIODS = 12
DISCARD_FRACTION = 0.3

# Continuous-map design space
CONTROL_GRID = (4, 3)      # coarse control grid inside the design region
DESIGN_X_RANGE = (0.12, 0.40)  # region where porosity can vary
DESIGN_Y_RANGE = (0.05, 0.27)

# Optimisation control
DE_POPSIZE = 4
DE_MAXITER = 6
DE_TOL = 1e-4
DE_SEED = 42

# Target matrix (diagonal routing)
W_TARGET = np.array([[0.8, 0.2],
                     [0.2, 0.8]])


# =============================================================================
# MxV SOLVER WRAPPER
# =============================================================================

class PorousMxVFDTD:
    """
    Wrap FDTDSolver for a 2x2 MxV experiment with a continuous c_field map.
    """

    def __init__(self, nx=NX, ny=NY, dx=DX, c0=C0, rho0=RHO0):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dx
        self.c0 = c0
        self.rho0 = rho0
        self.freq = SRC_FREQ

        self.probe_i = []
        self.probe_j = []
        for idx in sorted(PROBE_POSITIONS.keys()):
            pos = PROBE_POSITIONS[idx]
            self.probe_i.append(int(pos["x"] / dx))
            self.probe_j.append(int(pos["y"] / dx))

        self.source_y = {}
        for idx in sorted(SOURCE_POSITIONS.keys()):
            self.source_y[idx] = int(SOURCE_POSITIONS[idx]["y"] / dx)

        self.probe_radius_cells = int(0.025 / dx)

        # Design region in grid indices
        self.design_i0 = int(DESIGN_X_RANGE[0] / dx)
        self.design_i1 = int(DESIGN_X_RANGE[1] / dx)
        self.design_j0 = int(DESIGN_Y_RANGE[0] / dx)
        self.design_j1 = int(DESIGN_Y_RANGE[1] / dx)

    def _build_c_field(self, alpha_design):
        """
        Convert a coarse alpha design grid into a full FDTD c_field.

        Parameters
        ----------
        alpha_design : ndarray, shape CONTROL_GRID
            Values in [0, 1] representing relative porosity/slow-down.

        Returns
        -------
        c_field : ndarray, shape (nx, ny)
        """
        # Smoothly interpolate the coarse design grid to the design-region size
        design_ni = self.design_i1 - self.design_i0
        design_nj = self.design_j1 - self.design_j0

        # zoom factors: scipy.ndimage.zoom expects (new/old)
        zoom_factors = (design_ni / alpha_design.shape[0],
                        design_nj / alpha_design.shape[1])
        alpha_region = zoom(alpha_design, zoom_factors, order=3)

        # Ensure bounds after interpolation
        alpha_region = np.clip(alpha_region, 0.0, 1.0)

        c_field = np.full((self.nx, self.ny), self.c0)
        c_field[self.design_i0:self.design_i1,
                self.design_j0:self.design_j1] = self.c0 * (1.0 - ALPHA_MAX * alpha_region)

        return c_field

    def _build_solver(self, alpha_design):
        """Create solver with the continuous sound-speed map."""
        solver = FDTDSolver(self.nx, self.ny, self.dx,
                            c0=self.c0, rho0=self.rho0)
        c_field = self._build_c_field(alpha_design)
        solver.set_c_field(c_field)

        solver.set_bc(left='hard_wall', right='absorbing',
                      top='absorbing', bottom='absorbing')

        return solver

    def _source_slice(self, source_index):
        """Return (i_start, i_end, j_start, j_end) for a source port."""
        yc = self.source_y[source_index]
        j_start = max(0, yc - SRC_WIDTH_CELLS // 2)
        j_end = min(self.ny, yc + SRC_WIDTH_CELLS // 2 + 1)
        return 3, 6, j_start, j_end

    def _probe_intensity(self, solver):
        """Measure time-averaged squared pressure at each probe disc."""
        intensities = np.zeros(len(self.probe_i))
        for k, (ic, jc) in enumerate(zip(self.probe_i, self.probe_j)):
            r = self.probe_radius_cells
            i0, i1 = max(0, ic - r), min(self.nx, ic + r + 1)
            j0, j1 = max(0, jc - r), min(self.ny, jc + r + 1)
            region = solver.p[i0:i1, j0:j1]
            intensities[k] = np.mean(region ** 2)
        return intensities

    def measure_matrix(self, alpha_design):
        """
        Fire each source alone and build the measured intensity matrix.
        """
        n_sources = len(self.source_y)
        n_probes = len(self.probe_i)
        W_raw = np.zeros((n_probes, n_sources))

        for s in range(n_sources):
            solver = self._build_solver(alpha_design)
            i0, i1, j0, j1 = self._source_slice(s)

            src_mask = np.zeros((self.nx, self.ny), dtype=bool)
            src_mask[i0:i1, j0:j1] = True

            n_steps = int(N_PERIODS / (self.freq * solver.dt))
            n_discard = int(DISCARD_FRACTION * n_steps)

            probe_history = np.zeros((n_probes, n_steps - n_discard))

            for n in range(n_steps):
                src = SRC_AMP * np.sin(2.0 * np.pi * self.freq * solver.time)
                solver.p[src_mask] += src
                solver.step()

                if n >= n_discard:
                    probe_history[:, n - n_discard] = self._probe_intensity(solver)

            W_raw[:, s] = np.mean(probe_history, axis=1)

        W = np.zeros_like(W_raw)
        for s in range(n_sources):
            col_max = W_raw[:, s].max()
            if col_max > 1e-12:
                W[:, s] = W_raw[:, s] / col_max

        return W


# =============================================================================
# OPTIMISATION
# =============================================================================

def params_to_alpha_design(params):
    """Flat parameter vector -> coarse alpha design grid."""
    return np.array(params).reshape(CONTROL_GRID)


def build_bounds():
    """Bounds for each control value: alpha in [0, 1]."""
    return [(0.0, 1.0)] * (CONTROL_GRID[0] * CONTROL_GRID[1])


def objective(params):
    """Mean squared error between measured and target matrices."""
    alpha_design = params_to_alpha_design(params)
    mxv = PorousMxVFDTD()
    W = mxv.measure_matrix(alpha_design)
    loss = float(np.mean((W - W_TARGET) ** 2))

    objective.n_evals += 1
    if loss < objective.best_loss:
        objective.best_loss = loss
        objective.best_W = W.copy()
        objective.best_params = params.copy()
    if objective.n_evals % 10 == 0:
        print(f"  eval {objective.n_evals:4d}: loss={loss:.4f}  best={objective.best_loss:.4f}")

    return loss


objective.n_evals = 0
objective.best_loss = np.inf
objective.best_W = None
objective.best_params = None


def main():
    fig_dir = Path(__file__).parent / 'figures'
    fig_dir.mkdir(exist_ok=True)

    print("2x2 porous-medium acoustic MxV benchmark")
    print(f"Grid: {NX}x{NY}, dx={DX*1000:.2f} mm, f={SRC_FREQ/1e3:.1f} kHz")
    print(f"Control grid: {CONTROL_GRID} over design region")
    print(f"Target matrix:\n{W_TARGET}\n")

    bounds = build_bounds()
    result = differential_evolution(
        objective,
        bounds,
        maxiter=DE_MAXITER,
        popsize=DE_POPSIZE,
        tol=DE_TOL,
        seed=DE_SEED,
        polish=False,
        workers=1,
        updating='immediate',
    )

    best_alpha = params_to_alpha_design(result.x)
    best_W = objective.best_W
    best_loss = objective.best_loss

    print("\n" + "="*60)
    print("Best alpha design grid:")
    print(best_alpha)
    print(f"\nBest loss: {best_loss:.4f}")
    print("Measured matrix:")
    print(best_W)
    print("Target matrix:")
    print(W_TARGET)

    # -------------------------------------------------------------------------
    # Plot 1: measured vs target matrix as heatmaps
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(W_TARGET, origin='lower', vmin=0, vmax=1, cmap='viridis')
    axes[0].set_title('Target matrix')
    axes[0].set_xlabel('Source index')
    axes[0].set_ylabel('Probe index')
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f'{W_TARGET[i, j]:.2f}',
                         ha='center', va='center', color='white')

    im1 = axes[1].imshow(best_W, origin='lower', vmin=0, vmax=1, cmap='viridis')
    axes[1].set_title('Measured matrix (best)')
    axes[1].set_xlabel('Source index')
    axes[1].set_ylabel('Probe index')
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f'{best_W[i, j]:.2f}',
                         ha='center', va='center', color='white')

    fig.colorbar(im1, ax=axes, label='Normalised intensity')
    plt.tight_layout()
    plt.savefig(fig_dir / 'mxv_porous_matrix_v1.png', dpi=150)
    print(f"\nSaved: {fig_dir / 'mxv_porous_matrix_v1.png'}")

    # -------------------------------------------------------------------------
    # Plot 2: best sound-speed map and fields
    # -------------------------------------------------------------------------
    mxv = PorousMxVFDTD()
    c_field = mxv._build_c_field(best_alpha)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Sound-speed map
    ax = axes[0, 0]
    im = ax.imshow(c_field.T, origin='lower', cmap='viridis',
                   extent=[0, LX*1000, 0, NY*DX*1000])
    ax.set_title('Optimised sound-speed map')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    fig.colorbar(im, ax=ax, label='c (m/s)')

    # Alpha map derived from c_field
    ax = axes[0, 1]
    alpha_full = (C0 - c_field) / (ALPHA_MAX * C0)
    alpha_full = np.clip(alpha_full, 0.0, 1.0)
    im = ax.imshow(alpha_full.T, origin='lower', cmap='Reds',
                   extent=[0, LX*1000, 0, NY*DX*1000])
    ax.set_title('Alpha field (interpolated)')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    fig.colorbar(im, ax=ax, label='alpha')

    # Pressure fields for each source
    for s in range(2):
        solver = mxv._build_solver(best_alpha)
        i0, i1, j0, j1 = mxv._source_slice(s)
        src_mask = np.zeros((NX, NY), dtype=bool)
        src_mask[i0:i1, j0:j1] = True

        n_steps = int(N_PERIODS / (SRC_FREQ * solver.dt))
        for _ in range(n_steps):
            src = SRC_AMP * np.sin(2.0 * np.pi * SRC_FREQ * solver.time)
            solver.p[src_mask] += src
            solver.step()

        ax = axes[1, s]
        im = ax.imshow(solver.p.T, origin='lower', cmap='RdBu_r',
                       extent=[0, LX*1000, 0, NY*DX*1000],
                       vmin=-0.6, vmax=0.6)
        ax.set_title(f'Source {s} active')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        fig.colorbar(im, ax=ax, label='p (Pa)')

    plt.tight_layout()
    plt.savefig(fig_dir / 'mxv_porous_field_v1.png', dpi=150)
    print(f"Saved: {fig_dir / 'mxv_porous_field_v1.png'}")


if __name__ == '__main__':
    main()
