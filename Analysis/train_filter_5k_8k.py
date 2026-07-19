"""
train_filter_5k_8k.py

Optimize 4 heater strip amplitudes to maximize power ratio:
    ratio = power_5kHz / power_8kHz

Uses thermal solver + acoustic FDTD from acoustic_filter_with_thermal.py.
Phase 1: optimize amplitudes only (fixed positions and widths).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from acoustic_filter_with_thermal import (
    build_heat_source, solve_temperature, run_fdtd,
    NX, NY, DX, C0, T0, D_THERMAL, STRIP_POSITIONS, STRIP_WIDTH
)

# =============================================================================
# OPTIMIZATION SETTINGS
# =============================================================================
F_PASS = 5000.0   # Hz — frequency to pass
F_BLOCK = 8000.0  # Hz — frequency to block

# Bounds for heat source amplitudes (W/m^3)
# Empirically: sigma ~ 5-10 gives Tmax ~ 400-500 K
SIGMA_MIN = 0.0
SIGMA_MAX = 15.0

# Number of random search samples
N_RANDOM = 20

# L-BFGS-B settings
MAX_ITER = 100

# =============================================================================
# LOSS FUNCTION
# =============================================================================

def evaluate_ratio(sigma_amps, verbose=False):
    """
    Evaluate the power ratio for given heater amplitudes.
    Returns: ratio (higher is better), power_pass, power_block
    """
    # Build heat source and solve temperature
    sigma = build_heat_source(sigma_amps)
    T = solve_temperature(sigma)
    c_field = C0 * np.sqrt(T / T0)

    # Run acoustic FDTD at both frequencies
    p_pass, dt_pass, ns_pass = run_fdtd(c_field, F_PASS)
    p_block, dt_block, ns_block = run_fdtd(c_field, F_BLOCK)

    ratio = p_pass / (p_block + 1e-30)

    if verbose:
        print(f"  sigma = {sigma_amps}")
        print(f"  Tmax = {T.max():.1f} K, cmax = {c_field.max():.1f} m/s")
        print(f"  power_5kHz = {p_pass:.4e}, power_8kHz = {p_block:.4e}")
        print(f"  ratio = {ratio:.3f}")

    return ratio, p_pass, p_block


def loss_fn(sigma_amps):
    """Negative ratio for minimization."""
    ratio, _, _ = evaluate_ratio(sigma_amps)
    return -ratio


# =============================================================================
# TRAINING
# =============================================================================

def train():
    print("=" * 60)
    print("TRAINING: 5 kHz PASS / 8 kHz BLOCK")
    print("=" * 60)
    print(f"Random search: {N_RANDOM} samples")
    print(f"L-BFGS-B max iterations: {MAX_ITER}")
    print(f"Sigma bounds: [{SIGMA_MIN}, {SIGMA_MAX}] W/m^3")
    print()

    # -------------------------------------------------------------------------
    # Baseline: uniform temperature (no heaters)
    # -------------------------------------------------------------------------
    print("BASELINE (no heaters):")
    baseline_ratio, baseline_pass, baseline_block = evaluate_ratio(
        np.zeros(4), verbose=True
    )
    print()

    # -------------------------------------------------------------------------
    # Random search
    # -------------------------------------------------------------------------
    print("RANDOM SEARCH...")
    best_ratio = -1.0
    best_sigma = None
    best_pass = 0.0
    best_block = 0.0

    history = []  # list of (sigma_amps, ratio)

    for i in range(N_RANDOM):
        sigma_trial = np.random.uniform(SIGMA_MIN, SIGMA_MAX, 4)
        ratio, p_pass, p_block = evaluate_ratio(sigma_trial)
        history.append((sigma_trial.copy(), ratio))

        print(f"  [{i+1:2d}/{N_RANDOM}] ratio = {ratio:.3f}, sigma = {sigma_trial}")

        if ratio > best_ratio:
            best_ratio = ratio
            best_sigma = sigma_trial.copy()
            best_pass = p_pass
            best_block = p_block

    print(f"\nBest from random search: ratio = {best_ratio:.3f}")
    print(f"  sigma = {best_sigma}")
    print()

    # -------------------------------------------------------------------------
    # L-BFGS-B refinement
    # -------------------------------------------------------------------------
    print("L-BFGS-B OPTIMIZATION...")
    bounds = [(SIGMA_MIN, SIGMA_MAX)] * 4

    result = minimize(
        loss_fn,
        x0=best_sigma,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': MAX_ITER, 'disp': True}
    )

    opt_sigma = result.x
    opt_ratio, opt_pass, opt_block = evaluate_ratio(opt_sigma, verbose=True)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Baseline ratio:     {baseline_ratio:.3f}")
    print(f"Best random ratio:  {best_ratio:.3f}")
    print(f"Optimized ratio:    {opt_ratio:.3f}")
    print()
    print(f"Optimized sigma:    {opt_sigma}")
    print(f"Power @ 5 kHz:      {opt_pass:.4e}")
    print(f"Power @ 8 kHz:      {opt_block:.4e}")
    print()

    # -------------------------------------------------------------------------
    # Visualize optimized temperature field
    # -------------------------------------------------------------------------
    sigma_opt = build_heat_source(opt_sigma)
    T_opt = solve_temperature(sigma_opt)
    c_opt = C0 * np.sqrt(T_opt / T0)

    fig, axes = plt.subplots(2, 3, figsize=(14, 6))

    # Temperature
    ax = axes[0, 0]
    im = ax.imshow(T_opt.T, origin='lower', aspect='auto',
                   extent=[0, NX*DX*1000, 0, NY*DX*1000], cmap='hot')
    ax.set_title(f'Optimized T (K)\nmax={T_opt.max():.1f} K')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    plt.colorbar(im, ax=ax)

    # Sound speed
    ax = axes[0, 1]
    im = ax.imshow(c_opt.T, origin='lower', aspect='auto',
                   extent=[0, NX*DX*1000, 0, NY*DX*1000], cmap='viridis')
    ax.set_title(f'Optimized c (m/s)\nmax={c_opt.max():.1f} m/s')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    plt.colorbar(im, ax=ax)

    # Strip amplitudes
    ax = axes[0, 2]
    x_pos = STRIP_POSITIONS * 1000
    ax.bar(range(4), opt_sigma, color='orange', edgecolor='black')
    ax.set_xticks(range(4))
    ax.set_xticklabels([f'{x:.0f} mm' for x in x_pos])
    ax.set_ylabel('Heat source σ (W/m³)')
    ax.set_title('Optimized strip amplitudes')

    # Power comparison
    ax = axes[1, 0]
    labels = ['5 kHz\n(baseline)', '8 kHz\n(baseline)',
              '5 kHz\n(optimized)', '8 kHz\n(optimized)']
    values = [baseline_pass, baseline_block, opt_pass, opt_block]
    colors = ['blue', 'red', 'blue', 'red']
    ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Transmitted power')
    ax.set_title('Power comparison')

    # Ratio comparison
    ax = axes[1, 1]
    ax.bar(['Baseline', 'Best random', 'Optimized'],
           [baseline_ratio, best_ratio, opt_ratio],
           color=['gray', 'lightblue', 'green'],
           edgecolor='black')
    ax.set_ylabel('Ratio (5 kHz / 8 kHz)')
    ax.set_title('Filter selectivity')
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='unity')
    ax.legend()

    # Random search history
    ax = axes[1, 2]
    ratios = [r for _, r in history]
    ax.plot(range(1, len(ratios)+1), ratios, 'o-', color='purple', alpha=0.6)
    ax.axhline(y=baseline_ratio, color='gray', linestyle='--', label='baseline')
    ax.axhline(y=best_ratio, color='blue', linestyle='--', label='best random')
    ax.axhline(y=opt_ratio, color='green', linestyle='--', label='optimized')
    ax.set_xlabel('Random sample')
    ax.set_ylabel('Ratio')
    ax.set_title('Random search history')
    ax.legend()

    plt.tight_layout()
    plt.savefig('training_results_5k_8k.png', dpi=150)
    print("Saved figure: training_results_5k_8k.png")
    plt.show()

    # Save data
    np.savez('training_results_5k_8k.npz',
             baseline_ratio=baseline_ratio,
             baseline_pass=baseline_pass,
             baseline_block=baseline_block,
             best_random_ratio=best_ratio,
             best_random_sigma=best_sigma,
             optimized_ratio=opt_ratio,
             optimized_sigma=opt_sigma,
             optimized_pass=opt_pass,
             optimized_block=opt_block,
             random_history=np.array([r for _, r in history]))
    print("Saved data: training_results_5k_8k.npz")


if __name__ == '__main__':
    train()
