#!/usr/bin/env python3
"""
================================================================================
POROUS ROUTER — Blockwise Porosity Optimisation (Verification Script)
================================================================================

Uses fdtd_porous.py to learn a spatial porosity pattern in a central design
region that routes a continuous 5 kHz acoustic wave toward a target output port.

This is a *verification* script: the goal is to confirm that the porous solver
and the objective function behave sensibly before running a full optimiser.

Workflow:
    1. Baseline uniform porosity (phi = 1 everywhere).
    2. Two hand-designed vertical gradients.
    3. Short random search over 16 blockwise porosity values.
    4. Plot the best design found and its pressure field.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from time import perf_counter

import fdtd_porous as fdtd


# =============================================================================
# CONFIGURATION
# =============================================================================

# Physical domain constants (must match fdtd_porous.py)
NX, NY = fdtd.NX, fdtd.NY
DX = fdtd.DX
C0 = fdtd.C0
RHO0 = fdtd.RHO0

# Source
FREQ = 5000.0                     # Hz
DURATION = 0.008                  # total simulation time [s]
RMS_WINDOW = 0.002                # final time window used for RMS readout [s]

# Design region (indices)
DESIGN_X0, DESIGN_X1 = 140, 220   # 80 cells wide
DESIGN_Y0, DESIGN_Y1 = 10, 40     # 30 cells tall
N_BLOCKS_X = 8
N_BLOCKS_Y = 2
N_BLOCKS = N_BLOCKS_X * N_BLOCKS_Y

# Readout windows (point probes at window centres)
PROBE_TARGET = (370, 37)          # top-right
PROBE_OTHER = (370, 12)           # bottom-right

# Porosity bounds
PHI_MIN = 0.40
PHI_MAX = 1.00
ALPHA_INF = 1.0
SIGMA = 0.0                       # pure phase-speed/impedance contrast first

# Random search
N_RANDOM = 30
SEED = 42

# Output
FIG_DIR = Path(__file__).parent / 'figures'
FIG_DIR.mkdir(exist_ok=True)


# =============================================================================
# SOLVER SETUP
# =============================================================================

fdtd.BC_LEFT = 'absorbing'
fdtd.BC_RIGHT = 'absorbing'
fdtd.BC_TOP = 'hard_wall'
fdtd.BC_BOTTOM = 'hard_wall'

fdtd.SOURCE_TYPE = 'continuous_sine'
fdtd.SRC_FREQ = FREQ
fdtd.SRC_AMP = 1.0
fdtd.SRC_I0, fdtd.SRC_I1 = 3, 6


# =============================================================================
# DESIGN-REGION MAPPING
# =============================================================================

def params_to_phi(params):
    """Map 1D parameter vector to full (NX, NY) porosity field."""
    params = np.asarray(params).clip(PHI_MIN, PHI_MAX)
    phi = np.ones((NX, NY), dtype=float)

    dx = (DESIGN_X1 - DESIGN_X0) // N_BLOCKS_X
    dy = (DESIGN_Y1 - DESIGN_Y0) // N_BLOCKS_Y

    k = 0
    for iy in range(N_BLOCKS_Y):
        y0 = DESIGN_Y0 + iy * dy
        y1 = y0 + dy
        for ix in range(N_BLOCKS_X):
            x0 = DESIGN_X0 + ix * dx
            x1 = x0 + dx
            phi[x0:x1, y0:y1] = params[k]
            k += 1

    return phi


def phi_to_block_image(phi):
    """Return a block-averaged image of the design region for plotting."""
    dx = (DESIGN_X1 - DESIGN_X0) // N_BLOCKS_X
    dy = (DESIGN_Y1 - DESIGN_Y0) // N_BLOCKS_Y
    img = np.zeros((N_BLOCKS_X, N_BLOCKS_Y))
    for iy in range(N_BLOCKS_Y):
        y0 = DESIGN_Y0 + iy * dy
        y1 = y0 + dy
        for ix in range(N_BLOCKS_X):
            x0 = DESIGN_X0 + ix * dx
            x1 = x0 + dx
            img[ix, iy] = phi[x0:x1, y0:y1].mean()
    return img


# =============================================================================
# FDTD EVALUATION
# =============================================================================

def run_fdtd_two_probes(phi):
    """
    Set porosity field, reset state, run FDTD, and return RMS pressures at the
    target and other probes measured over the final RMS_WINDOW seconds.
    """
    # Set porous parameters and recompute derived quantities (including dt)
    fdtd.set_porous_params(
        phi,
        np.full_like(phi, ALPHA_INF),
        np.full_like(phi, SIGMA)
    )
    fdtd.reset_state()

    dt = fdtd.dt
    n_total = int(DURATION / dt)
    n_rms = int(RMS_WINDOW / dt)

    target_series = []
    other_series = []

    for n in range(n_total):
        fdtd.step(n * dt)
        if n >= n_total - n_rms:
            target_series.append(float(fdtd.p[PROBE_TARGET]))
            other_series.append(float(fdtd.p[PROBE_OTHER]))

    p_target = np.sqrt(np.mean(np.asarray(target_series) ** 2))
    p_other = np.sqrt(np.mean(np.asarray(other_series) ** 2))
    return p_target, p_other


def fom(p_target, p_other):
    """Figure of merit: 0 = all energy to other, 1 = all energy to target."""
    eps = 1e-12
    return (p_target ** 2) / (p_target ** 2 + p_other ** 2 + eps)


def evaluate(params, verbose=False):
    """Evaluate a design parameter vector. Returns FOM and probe pressures."""
    phi = params_to_phi(params)
    t0 = perf_counter()
    p_target, p_other = run_fdtd_two_probes(phi)
    elapsed = perf_counter() - t0
    score = fom(p_target, p_other)

    if verbose:
        print(f"  FOM={score:.4f}  P_target={p_target:.4e}  P_other={p_other:.4e}  "
              f"dt={fdtd.dt*1e6:.2f} µs  time={elapsed:.2f}s")

    return score, p_target, p_other


# =============================================================================
# BASELINE AND SANITY CHECKS
# =============================================================================

def run_baseline():
    """Uniform free-air equivalent porosity."""
    print("\n" + "=" * 60)
    print("BASELINE: uniform phi = 1.0")
    print("=" * 60)
    params = np.ones(N_BLOCKS)
    score, p_t, p_o = evaluate(params, verbose=True)
    print(f"  Expected FOM ~ 0.5 (symmetric). Got {score:.4f}")
    return params, score, p_t, p_o


def run_handcrafted():
    """Two deterministic patterns to verify the FOM moves off baseline."""
    print("\n" + "=" * 60)
    print("HAND-CRAFTED PATTERNS")
    print("=" * 60)

    # Pattern A: denser (lower phi) on top half -> should slow top wave,
    # bend energy downward?  Actually lower phi -> higher c, so top channel
    # is faster.  We just want to see a *different* FOM.
    params_top_heavy = np.ones(N_BLOCKS)
    params_top_heavy[N_BLOCKS_X:] = PHI_MIN  # bottom row dense

    # Pattern B: denser on bottom half
    params_bottom_heavy = np.ones(N_BLOCKS)
    params_bottom_heavy[:N_BLOCKS_X] = PHI_MIN  # top row dense

    results = []
    for name, params in [("top-row dense", params_top_heavy),
                         ("bottom-row dense", params_bottom_heavy)]:
        score, p_t, p_o = evaluate(params, verbose=True)
        results.append((name, params, score, p_t, p_o))
        print(f"  {name}: FOM={score:.4f}")

    return results


# =============================================================================
# RANDOM SEARCH
# =============================================================================

def run_random_search(n_samples=N_RANDOM):
    """Sample random blockwise porosity designs and keep the best."""
    print("\n" + "=" * 60)
    print(f"RANDOM SEARCH: {n_samples} samples")
    print("=" * 60)

    rng = np.random.default_rng(SEED)
    best_score = -1.0
    best_params = None
    history = []

    for i in range(n_samples):
        params = rng.uniform(PHI_MIN, PHI_MAX, N_BLOCKS)
        score, p_t, p_o = evaluate(params)
        history.append((score, params.copy(), p_t, p_o))
        if score > best_score:
            best_score = score
            best_params = params.copy()
        print(f"  Sample {i+1:3d}/{n_samples}: FOM={score:.4f}  "
              f"P_t={p_t:.4e}  P_o={p_o:.4e}")

    history.sort(key=lambda x: x[0], reverse=True)
    return best_params, best_score, history


# =============================================================================
# PLOTTING
# =============================================================================

def plot_design(phi, title, filename):
    """Plot porosity design field."""
    fig, ax = plt.subplots(figsize=(10, 3))
    extent = [0, NX * DX * 1000, 0, NY * DX * 1000]
    im = ax.imshow(phi.T, origin='lower', cmap='viridis_r',
                   vmin=PHI_MIN, vmax=1.0, extent=extent)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='Porosity φ')
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_pressure(phi, title, filename):
    """Run one more FDTD evaluation and plot the final pressure snapshot."""
    run_fdtd_two_probes(phi)  # leaves fdtd.p populated
    p = fdtd.p

    fig, ax = plt.subplots(figsize=(10, 3))
    extent = [0, NX * DX * 1000, 0, NY * DX * 1000]
    vmax = np.abs(p).max()
    im = ax.imshow(p.T, origin='lower', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, extent=extent)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='Pressure [Pa]')
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_history(history, baseline_score, filename):
    """Bar plot of FOMs from random search, with baseline marked."""
    scores = [h[0] for h in history]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(baseline_score, color='r', ls='--', label='Baseline')
    ax.bar(range(1, len(scores) + 1), sorted(scores, reverse=True),
           color='steelblue')
    ax.set_xlabel('Sample rank')
    ax.set_ylabel('FOM (target / total)')
    ax.set_title('Random-search FOM distribution')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("Porous router verification")
    print(f"  Domain: {NX}x{NY}, DX={DX*1000:.2f} mm")
    print(f"  Design region: x=[{DESIGN_X0},{DESIGN_X1}], y=[{DESIGN_Y0},{DESIGN_Y1}]")
    print(f"  Blocks: {N_BLOCKS_X}x{N_BLOCKS_Y} = {N_BLOCKS}")
    print(f"  Frequency: {FREQ/1000:.1f} kHz, Duration: {DURATION*1000:.1f} ms")

    # Baseline
    base_params, base_score, base_pt, base_po = run_baseline()

    # Hand-crafted sanity checks
    handcrafted = run_handcrafted()

    # Random search
    best_params, best_score, history = run_random_search()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Baseline FOM: {base_score:.4f}")
    for name, _, score, _, _ in handcrafted:
        print(f"  Hand-crafted ({name}): {score:.4f}")
    print(f"  Random-search best FOM: {best_score:.4f}")
    print(f"  Improvement over baseline: {(best_score - base_score)*100:.2f} pp")

    # Plot best design
    best_phi = params_to_phi(best_params)
    plot_design(best_phi, f'Best random design (FOM={best_score:.3f})',
                FIG_DIR / 'porous_router_best_design.png')
    plot_pressure(best_phi, f'Pressure field — best design (FOM={best_score:.3f})',
                  FIG_DIR / 'porous_router_best_pressure.png')
    plot_history(history, base_score,
                 FIG_DIR / 'porous_router_random_history.png')

    print("\nFigures saved to:", FIG_DIR)
