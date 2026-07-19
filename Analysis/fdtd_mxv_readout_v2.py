#!/usr/bin/env python3
"""
fdtd_mxv_readout_v1.py
========================
Acoustic MxV layer + nonlinear readout for classification.

The physical layer is a continuous porous-medium sound-speed map that performs
a linear matrix-vector multiplication via wave interference.  A digital ReLU
readout is applied to the probe intensities, turning the device into a single
neural-network layer.

Tasks demonstrated:
  - 2-class classifier: input [1,0] -> class A, input [0,1] -> class B
  - Optional scaling test to 3x3 routing (set N_PORTS = 3)

The readout is:
    y_neural = ReLU(y_physical - threshold)

where threshold is a trainable scalar.
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
# CONFIGURATION
# =============================================================================

N_PORTS = 3                  # number of input/output ports (try 2 or 3)
TASK = 'classifier'          # 'classifier' or 'routing'

# Domain
NX, NY = 280, 160            # slightly larger domain for 3 ports
LX = 0.70
DX = LX / NX

RHO0 = 1.225
C0 = 343.0
ALPHA_MAX = 0.50
SRC_FREQ = 2000.0
SRC_AMP = 1.0
SRC_WIDTH_CELLS = 3
N_PERIODS = 12
DISCARD_FRACTION = 0.3

# Porous design region
CONTROL_GRID = (6, 4)
DESIGN_X_RANGE = (0.15, 0.50)
DESIGN_Y_RANGE = (0.06, 0.34)

# Optimisation
DE_POPSIZE = 5
DE_MAXITER = 10
DE_TOL = 1e-4
DE_SEED = 42


# =============================================================================
# PORT LAYOUTS
# =============================================================================

def get_port_positions(n_ports):
    """Return y-positions (in metres) for n_ports evenly spaced in the domain."""
    y_min, y_max = 0.06, 0.26
    return np.linspace(y_min, y_max, n_ports)


PORT_Y = get_port_positions(N_PORTS)
PROBE_X = 0.45


# =============================================================================
# TARGETS
# =============================================================================

def get_targets(n_ports, task):
    """Return input vectors and target output vectors."""
    if task == 'classifier':
        # One-hot inputs, one-hot outputs
        inputs = [np.eye(n_ports)[k] for k in range(n_ports)]
        targets = [np.eye(n_ports)[k] for k in range(n_ports)]
        return inputs, targets

    elif task == 'routing':
        # Diagonal routing: input k -> output k strongly
        W = np.eye(n_ports) * 0.8 + np.ones((n_ports, n_ports)) * 0.2 / n_ports
        inputs = [np.eye(n_ports)[k] for k in range(n_ports)]
        targets = [W @ inp for inp in inputs]
        return inputs, targets

    else:
        raise ValueError(f"Unknown task: {task}")


INPUTS, TARGETS = get_targets(N_PORTS, TASK)


# =============================================================================
# FDTD WRAPPER
# =============================================================================

class ReadoutMxVFDTD:
    """
    MxV physical layer with N input and N output ports.
    """

    def __init__(self, n_ports=N_PORTS):
        self.n_ports = n_ports
        self.nx = NX
        self.ny = NY
        self.dx = DX
        self.dy = DX
        self.c0 = C0
        self.rho0 = RHO0
        self.freq = SRC_FREQ

        self.probe_i = int(PROBE_X / DX)
        self.probe_j = [int(y / DX) for y in PORT_Y]
        self.source_j = [int(y / DX) for y in PORT_Y]
        self.probe_radius_cells = int(0.025 / DX)

        self.design_i0 = int(DESIGN_X_RANGE[0] / DX)
        self.design_i1 = int(DESIGN_X_RANGE[1] / DX)
        self.design_j0 = int(DESIGN_Y_RANGE[0] / DX)
        self.design_j1 = int(DESIGN_Y_RANGE[1] / DX)

    def _build_c_field(self, alpha_design):
        design_ni = self.design_i1 - self.design_i0
        design_nj = self.design_j1 - self.design_j0
        zoom_factors = (design_ni / alpha_design.shape[0],
                        design_nj / alpha_design.shape[1])
        alpha_region = zoom(alpha_design, zoom_factors, order=3)
        alpha_region = np.clip(alpha_region, 0.0, 1.0)

        c_field = np.full((self.nx, self.ny), self.c0)
        c_field[self.design_i0:self.design_i1,
                self.design_j0:self.design_j1] = self.c0 * (1.0 - ALPHA_MAX * alpha_region)
        return c_field

    def _build_solver(self, alpha_design):
        solver = FDTDSolver(self.nx, self.ny, self.dx, c0=self.c0, rho0=self.rho0)
        solver.set_c_field(self._build_c_field(alpha_design))
        solver.set_bc(left='hard_wall', right='absorbing',
                      top='absorbing', bottom='absorbing')
        return solver

    def _source_slice(self, source_index):
        yc = self.source_j[source_index]
        j_start = max(0, yc - SRC_WIDTH_CELLS // 2)
        j_end = min(self.ny, yc + SRC_WIDTH_CELLS // 2 + 1)
        return 3, 6, j_start, j_end

    def _probe_intensity(self, solver):
        intensities = np.zeros(self.n_ports)
        for k, jc in enumerate(self.probe_j):
            r = self.probe_radius_cells
            i0, i1 = max(0, self.probe_i - r), min(self.nx, self.probe_i + r + 1)
            j0, j1 = max(0, jc - r), min(self.ny, jc + r + 1)
            intensities[k] = np.mean(solver.p[i0:i1, j0:j1] ** 2)
        return intensities

    def measure_matrix(self, alpha_design):
        """
        Fire each source alone and return normalised intensity matrix W.
        W[i,j] = normalised intensity at probe i when source j fires.
        """
        W_raw = np.zeros((self.n_ports, self.n_ports))

        for s in range(self.n_ports):
            solver = self._build_solver(alpha_design)
            i0, i1, j0, j1 = self._source_slice(s)
            src_mask = np.zeros((self.nx, self.ny), dtype=bool)
            src_mask[i0:i1, j0:j1] = True

            n_steps = int(N_PERIODS / (self.freq * solver.dt))
            n_discard = int(DISCARD_FRACTION * n_steps)
            probe_history = np.zeros((self.n_ports, n_steps - n_discard))

            for n in range(n_steps):
                src = SRC_AMP * np.sin(2.0 * np.pi * self.freq * solver.time)
                solver.p[src_mask] += src
                solver.step()

                if n >= n_discard:
                    probe_history[:, n - n_discard] = self._probe_intensity(solver)

            W_raw[:, s] = np.mean(probe_history, axis=1)

        W = np.zeros_like(W_raw)
        for s in range(self.n_ports):
            col_max = W_raw[:, s].max()
            if col_max > 1e-12:
                W[:, s] = W_raw[:, s] / col_max

        return W


# =============================================================================
# READOUT LAYER
# =============================================================================

def relu_readout(y_physical, threshold):
    """ReLU readout: y = max(y_physical - threshold, 0)."""
    return np.maximum(y_physical - threshold, 0.0)


def evaluate_loss(alpha_design, threshold, return_details=False):
    """
    Evaluate classification/routing loss over all input-target pairs.

    Parameters
    ----------
    alpha_design : ndarray, shape CONTROL_GRID
    threshold : float
    return_details : bool
        If True, return predictions and targets.

    Returns
    -------
    loss : float
    details : dict (optional)
    """
    mxv = ReadoutMxVFDTD(N_PORTS)
    W = mxv.measure_matrix(alpha_design)

    total_loss = 0.0
    predictions = []
    linear_outputs = []

    for inp, target in zip(INPUTS, TARGETS):
        y_linear = W @ inp
        y_neural = relu_readout(y_linear, threshold)
        total_loss += np.mean((y_neural - target) ** 2)
        predictions.append(y_neural)
        linear_outputs.append(y_linear)

    loss = total_loss / len(INPUTS)

    if return_details:
        return loss, {
            'W': W,
            'predictions': predictions,
            'linear_outputs': linear_outputs,
            'targets': TARGETS,
        }
    return loss


# =============================================================================
# OPTIMISATION
# =============================================================================

def params_to_design(params):
    n_alpha = CONTROL_GRID[0] * CONTROL_GRID[1]
    alpha = np.array(params[:n_alpha]).reshape(CONTROL_GRID)
    threshold = params[n_alpha]
    return alpha, threshold


def build_bounds():
    n_alpha = CONTROL_GRID[0] * CONTROL_GRID[1]
    bounds = [(0.0, 1.0)] * n_alpha
    bounds.append((0.0, 0.9))   # threshold range
    return bounds


def objective(params):
    alpha, threshold = params_to_design(params)
    loss = evaluate_loss(alpha, threshold)

    objective.n_evals += 1
    if loss < objective.best_loss:
        objective.best_loss = loss
        objective.best_params = params.copy()
    if objective.n_evals % 10 == 0:
        print(f"  eval {objective.n_evals:4d}: loss={loss:.4f}  best={objective.best_loss:.4f}")

    return loss


objective.n_evals = 0
objective.best_loss = np.inf
objective.best_params = None


# =============================================================================
# MAIN
# =============================================================================

def main():
    fig_dir = Path(__file__).parent / 'figures'
    fig_dir.mkdir(exist_ok=True)

    print(f"Acoustic MxV + ReLU readout: {N_PORTS}x{N_PORTS} {TASK}")
    print(f"Grid: {NX}x{NY}, dx={DX*1000:.2f} mm, f={SRC_FREQ/1e3:.1f} kHz")
    print(f"Ports at y = {PORT_Y*1000} mm")
    print(f"Control grid: {CONTROL_GRID}\n")

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

    best_alpha, best_threshold = params_to_design(result.x)
    best_loss, details = evaluate_loss(best_alpha, best_threshold, return_details=True)

    print("\n" + "="*60)
    print(f"Best threshold: {best_threshold:.3f}")
    print("Measured physical matrix W:")
    print(details['W'])
    print("\nLinear outputs -> ReLU outputs -> Targets:")
    for k, (lin, pred, tgt) in enumerate(zip(details['linear_outputs'],
                                              details['predictions'],
                                              details['targets'])):
        print(f"Input {k}: {lin.round(3)} -> {pred.round(3)}  (target {tgt.round(3)})")
    print(f"\nBest loss: {best_loss:.4f}")

    # Classification accuracy
    if TASK == 'classifier':
        correct = 0
        for pred, tgt in zip(details['predictions'], details['targets']):
            if np.argmax(pred) == np.argmax(tgt):
                correct += 1
        accuracy = correct / len(details['predictions'])
        print(f"Classification accuracy: {accuracy*100:.1f}%")

    # -------------------------------------------------------------------------
    # Plot 1: physical matrix
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(details['W'], origin='lower', vmin=0, vmax=1, cmap='viridis')
    ax.set_title('Measured physical matrix W')
    ax.set_xlabel('Source index')
    ax.set_ylabel('Probe index')
    ax.set_xticks(range(N_PORTS))
    ax.set_yticks(range(N_PORTS))
    for i in range(N_PORTS):
        for j in range(N_PORTS):
            ax.text(j, i, f'{details["W"][i, j]:.2f}',
                    ha='center', va='center', color='white')
    fig.colorbar(im, ax=ax, label='Normalised intensity')
    plt.tight_layout()
    plt.savefig(fig_dir / f'mxv_readout_matrix_{N_PORTS}x{N_PORTS}_v1.png', dpi=150)
    print(f"\nSaved: {fig_dir / f'mxv_readout_matrix_{N_PORTS}x{N_PORTS}_v1.png'}")

    # -------------------------------------------------------------------------
    # Plot 2: sound-speed map and fields
    # -------------------------------------------------------------------------
    mxv = ReadoutMxVFDTD(N_PORTS)
    c_field = mxv._build_c_field(best_alpha)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    im = ax.imshow(c_field.T, origin='lower', cmap='viridis',
                   extent=[0, LX*1000, 0, NY*DX*1000])
    ax.set_title('Optimised sound-speed map')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    fig.colorbar(im, ax=ax, label='c (m/s)')

    ax = axes[0, 1]
    alpha_full = (C0 - c_field) / (ALPHA_MAX * C0)
    alpha_full = np.clip(alpha_full, 0.0, 1.0)
    im = ax.imshow(alpha_full.T, origin='lower', cmap='Reds',
                   extent=[0, LX*1000, 0, NY*DX*1000])
    ax.set_title('Alpha field')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    fig.colorbar(im, ax=ax, label='alpha')

    for s in range(min(N_PORTS, 2)):
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
    plt.savefig(fig_dir / f'mxv_readout_field_{N_PORTS}x{N_PORTS}_v1.png', dpi=150)
    print(f"Saved: {fig_dir / f'mxv_readout_field_{N_PORTS}x{N_PORTS}_v1.png'}")


if __name__ == '__main__':
    main()
