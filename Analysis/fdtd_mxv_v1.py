#!/usr/bin/env python3
"""
fdtd_mxv_v1.py
==============
First trainable acoustic matrix-vector multiplication benchmark.

A 2x2 analog MxV layer: two source ports on the left, two probe ports on
the right, and a small number of hard-wall cylinders in between.  The
obstacle geometry is tuned by random search so that the measured
transmission matrix matches a target matrix.

Target example (diagonal routing):
    W_target = [[0.8, 0.2],
                [0.2, 0.8]]

Meaning:
    - input 1 should reach output 1 strongly and output 2 weakly
    - input 2 should reach output 2 strongly and output 1 weakly

The mapping from physical simulation to matrix is:
    W_measured[i, j] = normalised steady-state intensity at probe i
                       when source j fires alone

Because the medium is passive, all entries lie in [0, 1].
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from fdtd_cylinder_array_scatter_v1 import ObstacleFDTDSolver


# =============================================================================
# PHYSICAL / NUMERICAL PARAMETERS
# =============================================================================

NX, NY = 240, 120          # grid cells
LX = 0.60                  # domain length [m]
DX = LX / NX               # cell size [m]
DY = DX

RHO0 = 1.225               # air density [kg/m^3]
C0 = 343.0                 # sound speed [m/s]

SRC_FREQ = 2000.0          # source frequency [Hz]
SRC_AMP = 1.0              # source amplitude [Pa]
SRC_WIDTH_CELLS = 3        # vertical extent of source port

# Probe ports: centres in metres (must lie inside NY*DY = 0.30 m)
PROBE_POSITIONS = {
    0: {"x": 0.45, "y": 0.10},   # top output
    1: {"x": 0.45, "y": 0.22},   # bottom output
}

SOURCE_POSITIONS = {
    0: {"y": 0.10},              # top input
    1: {"y": 0.22},              # bottom input
}

N_PERIODS = 12               # number of source periods to simulate
DISCARD_FRACTION = 0.3       # discard initial transient before measuring

# Design space for random search
N_TRIALS = 40
N_OBSTACLES = 2
OBSTACLE_RADIUS = 0.025      # m
OBSTACLE_X_RANGE = (0.12, 0.32)
OBSTACLE_Y_RANGE = (0.05, 0.27)

# Target matrix (diagonal routing)
W_TARGET = np.array([[0.8, 0.2],
                     [0.2, 0.8]])

# Optional seed for reproducibility
RNG_SEED = 42


# =============================================================================
# MxV SOLVER WRAPPER
# =============================================================================

class MxVFDTD:
    """
    Wrap ObstacleFDTDSolver for a 2x2 MxV experiment.
    """

    def __init__(self, nx=NX, ny=NY, dx=DX, c0=C0, rho0=RHO0):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dx
        self.c0 = c0
        self.rho0 = rho0
        self.freq = SRC_FREQ

        # Probe coordinates in grid indices
        self.probe_i = []
        self.probe_j = []
        for idx in sorted(PROBE_POSITIONS.keys()):
            pos = PROBE_POSITIONS[idx]
            self.probe_i.append(int(pos["x"] / dx))
            self.probe_j.append(int(pos["y"] / dx))

        # Source y-centres in grid indices
        self.source_y = {}
        for idx in sorted(SOURCE_POSITIONS.keys()):
            self.source_y[idx] = int(SOURCE_POSITIONS[idx]["y"] / dx)

        self.probe_radius_cells = int(0.025 / dx)

    def _build_solver(self, cylinders):
        """Create solver with given cylinder list and uniform sound speed."""
        solver = ObstacleFDTDSolver(self.nx, self.ny, self.dx,
                                    c0=self.c0, rho0=self.rho0)
        c_field = np.full((self.nx, self.ny), self.c0)
        solver.set_c_field(c_field)

        # Domain: hard wall on left (source plane), absorbing elsewhere
        solver.set_bc(left='hard_wall', right='absorbing',
                      top='absorbing', bottom='absorbing')

        for cx, cy, r in cylinders:
            solver.add_cylinder(cx, cy, r)

        return solver

    def _source_slice(self, source_index):
        """Return (i_start, i_end, j_start, j_end) for a source port."""
        yc = self.source_y[source_index]
        j_start = max(0, yc - SRC_WIDTH_CELLS // 2)
        j_end = min(self.ny, yc + SRC_WIDTH_CELLS // 2 + 1)
        return 3, 6, j_start, j_end

    def _probe_intensity(self, solver):
        """
        Measure time-averaged squared pressure over a small disc around
        each probe point.  Return array of length n_probes.
        """
        intensities = np.zeros(len(self.probe_i))
        for k, (ic, jc) in enumerate(zip(self.probe_i, self.probe_j)):
            r = self.probe_radius_cells
            i0, i1 = max(0, ic - r), min(self.nx, ic + r + 1)
            j0, j1 = max(0, jc - r), min(self.ny, jc + r + 1)
            region = solver.p[i0:i1, j0:j1]
            intensities[k] = np.mean(region ** 2)
        return intensities

    def measure_matrix(self, cylinders, verbose=False):
        """
        Fire each source alone and build the measured intensity matrix.

        Returns
        -------
        W : ndarray, shape (n_probes, n_sources)
            Normalised intensities.  Column j is the response to source j.
        """
        n_sources = len(self.source_y)
        n_probes = len(self.probe_i)
        W_raw = np.zeros((n_probes, n_sources))

        for s in range(n_sources):
            solver = self._build_solver(cylinders)
            i0, i1, j0, j1 = self._source_slice(s)

            # Source mask: localised pressure injection at the chosen port.
            src_mask = np.zeros((self.nx, self.ny), dtype=bool)
            src_mask[i0:i1, j0:j1] = True

            n_steps = int(N_PERIODS / (self.freq * solver.dt))
            n_discard = int(DISCARD_FRACTION * n_steps)

            probe_history = np.zeros((n_probes, n_steps - n_discard))

            for n in range(n_steps):
                # Manual sinusoidal source injection at the port
                src = SRC_AMP * np.sin(2.0 * np.pi * self.freq * solver.time)
                solver.p[src_mask] += src

                solver.step()

                if n >= n_discard:
                    probe_history[:, n - n_discard] = self._probe_intensity(solver)

            # Steady-state mean intensity for this source
            W_raw[:, s] = np.mean(probe_history, axis=1)

            if verbose:
                print(f"  Source {s}: raw probe intensities {W_raw[:, s]}")

        # Normalise each column by the maximum intensity in that column so
        # that the strongest probe response for each source is 1.0.
        W = np.zeros_like(W_raw)
        for s in range(n_sources):
            col_max = W_raw[:, s].max()
            if col_max > 1e-12:
                W[:, s] = W_raw[:, s] / col_max

        return W

    def loss(self, cylinders):
        """Mean squared error between measured and target matrices."""
        W = self.measure_matrix(cylinders)
        return np.mean((W - W_TARGET) ** 2), W


# =============================================================================
# RANDOM SEARCH
# =============================================================================

def sample_cylinders(rng):
    """Sample a random obstacle configuration."""
    cylinders = []
    for _ in range(N_OBSTACLES):
        cx = rng.uniform(*OBSTACLE_X_RANGE)
        cy = rng.uniform(*OBSTACLE_Y_RANGE)
        cylinders.append((cx, cy, OBSTACLE_RADIUS))
    return cylinders


def main():
    fig_dir = Path(__file__).parent / 'figures'
    fig_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)
    mxv = MxVFDTD()

    print("2x2 acoustic MxV benchmark")
    print(f"Grid: {NX}x{NY}, dx={DX*1000:.2f} mm, f={SRC_FREQ/1e3:.1f} kHz")
    print(f"Target matrix:\n{W_TARGET}")
    print(f"Running {N_TRIALS} random obstacle configurations...\n")

    best_loss = np.inf
    best_cylinders = None
    best_W = None
    history = []

    for trial in range(N_TRIALS):
        cylinders = sample_cylinders(rng)
        loss, W = mxv.loss(cylinders)
        history.append(loss)

        print(f"Trial {trial+1:3d}/{N_TRIALS}: loss={loss:.4f}")

        if loss < best_loss:
            best_loss = loss
            best_cylinders = cylinders
            best_W = W.copy()
            print(f"  *** new best ***")

    print("\n" + "="*60)
    print("Best configuration:")
    for k, (cx, cy, r) in enumerate(best_cylinders):
        print(f"  Cylinder {k+1}: cx={cx:.3f} m, cy={cy:.3f} m, r={r:.3f} m")
    print(f"\nBest loss: {best_loss:.4f}")
    print("Measured matrix:")
    print(best_W)
    print("Target matrix:")
    print(W_TARGET)

    # -------------------------------------------------------------------------
    # Plot 1: convergence of random search
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(history)+1), history, 'o-', alpha=0.6, label='trial loss')
    ax.axhline(best_loss, color='r', linestyle='--', label=f'best loss = {best_loss:.4f}')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Mean squared error')
    ax.set_title('Random-search convergence for 2x2 acoustic MxV')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / 'mxv_random_search_v1.png', dpi=150)
    print(f"\nSaved: {fig_dir / 'mxv_random_search_v1.png'}")

    # -------------------------------------------------------------------------
    # Plot 2: measured vs target matrix as heatmaps
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    vmax = max(W_TARGET.max(), best_W.max())
    im0 = axes[0].imshow(W_TARGET, origin='lower', vmin=0, vmax=1,
                         cmap='viridis')
    axes[0].set_title('Target matrix')
    axes[0].set_xlabel('Source index')
    axes[0].set_ylabel('Probe index')
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f'{W_TARGET[i, j]:.2f}',
                         ha='center', va='center', color='white')

    im1 = axes[1].imshow(best_W, origin='lower', vmin=0, vmax=1,
                         cmap='viridis')
    axes[1].set_title('Measured matrix (best)')
    axes[1].set_xlabel('Source index')
    axes[1].set_ylabel('Probe index')
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f'{best_W[i, j]:.2f}',
                         ha='center', va='center', color='white')

    fig.colorbar(im1, ax=axes, label='Normalised intensity')
    plt.tight_layout()
    plt.savefig(fig_dir / 'mxv_matrix_comparison_v1.png', dpi=150)
    print(f"Saved: {fig_dir / 'mxv_matrix_comparison_v1.png'}")

    # -------------------------------------------------------------------------
    # Plot 3: pressure field for best geometry, each source
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    for s in range(2):
        cylinders = best_cylinders
        solver = mxv._build_solver(cylinders)
        i0, i1, j0, j1 = mxv._source_slice(s)
        src_mask = np.zeros((NX, NY), dtype=bool)
        src_mask[i0:i1, j0:j1] = True
        n_steps = int(N_PERIODS / (SRC_FREQ * solver.dt))
        for _ in range(n_steps):
            src = SRC_AMP * np.sin(2.0 * np.pi * SRC_FREQ * solver.time)
            solver.p[src_mask] += src
            solver.step()

        ax = axes[s]
        im = ax.imshow(solver.p.T, origin='lower', cmap='RdBu_r',
                       extent=[0, LX*1000, 0, NY*DX*1000],
                       vmin=-0.6, vmax=0.6)
        ax.set_title(f'Source {s} active')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        for cx, cy, r in cylinders:
            circle = plt.Circle((cx*1000, cy*1000), r*1000,
                                color='black', fill=True, alpha=0.7)
            ax.add_patch(circle)
        fig.colorbar(im, ax=ax, label='p (Pa)')

    plt.tight_layout()
    plt.savefig(fig_dir / 'mxv_best_field_v1.png', dpi=150)
    print(f"Saved: {fig_dir / 'mxv_best_field_v1.png'}")


if __name__ == '__main__':
    main()
