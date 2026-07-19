"""
train_filter_12blocks.py

Optimize 12 heater blocks (4 strips x 3 vertical segments).
Each strip at fixed x-position, split into 3 horizontal segments.
Goal: maximize power ratio = power_5kHz / power_8kHz.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from acoustic_filter_with_thermal import (
    solve_temperature, run_fdtd,
    NX, NY, DX, C0, T0, D_THERMAL, STRIP_POSITIONS, STRIP_WIDTH
)

# =============================================================================
# SETTINGS
# =============================================================================
F_PASS = 5000.0
F_BLOCK = 8000.0
SIGMA_MIN = 0.0
SIGMA_MAX = 15.0
N_RANDOM = 30
MAX_ITER = 100
N_SEGMENTS = 3  # vertical segments per strip


def build_heat_source_12blocks(sigma_amps, nx=NX, ny=NY, dx=DX,
                                positions=STRIP_POSITIONS, width=STRIP_WIDTH,
                                n_segments=N_SEGMENTS):
    """
    Build heat source from 12 block amplitudes.
    sigma_amps: array of 12 values, ordered as:
        [strip1_seg1, strip1_seg2, strip1_seg3,
         strip2_seg1, strip2_seg2, strip2_seg3,
         ...]
    """
    sigma = np.zeros((nx, ny))
    half_w_cells = int(width / (2 * dx))
    seg_height = ny // n_segments  # ~16 or 17 cells
    
    idx = 0
    for k, x_k in enumerate(positions):
        i_c = int(x_k / dx)
        i_s = max(0, i_c - half_w_cells)
        i_e = min(nx, i_c + half_w_cells)
        
        for seg in range(n_segments):
            j_s = seg * seg_height
            j_e = min(ny, (seg + 1) * seg_height)
            sigma[i_s:i_e, j_s:j_e] = sigma_amps[idx]
            idx += 1
    
    return sigma


def evaluate_ratio(sigma_amps, verbose=False):
    sigma = build_heat_source_12blocks(sigma_amps)
    T = solve_temperature(sigma)
    c_field = C0 * np.sqrt(T / T0)
    
    p_pass, _, _ = run_fdtd(c_field, F_PASS)
    p_block, _, _ = run_fdtd(c_field, F_BLOCK)
    
    ratio = p_pass / (p_block + 1e-30)
    
    if verbose:
        print(f"  Tmax = {T.max():.1f} K")
        print(f"  power_5k = {p_pass:.4e}, power_8k = {p_block:.4e}")
        print(f"  ratio = {ratio:.3f}")
    
    return ratio, p_pass, p_block


def loss_fn(sigma_amps):
    ratio, _, _ = evaluate_ratio(sigma_amps)
    return -ratio


def train():
    print("=" * 60)
    print("TRAINING: 12 BLOCKS (4 strips x 3 segments)")
    print("=" * 60)
    print(f"Random search: {N_RANDOM} samples")
    print(f"Total parameters: {4 * N_SEGMENTS}")
    print()
    
    # Baseline
    print("BASELINE (no heaters):")
    baseline_ratio, baseline_pass, baseline_block = evaluate_ratio(
        np.zeros(4 * N_SEGMENTS), verbose=True
    )
    print()
    
    # Random search
    print("RANDOM SEARCH...")
    best_ratio = -1.0
    best_sigma = None
    history = []
    
    for i in range(N_RANDOM):
        sigma_trial = np.random.uniform(SIGMA_MIN, SIGMA_MAX, 4 * N_SEGMENTS)
        ratio, p_pass, p_block = evaluate_ratio(sigma_trial)
        history.append((sigma_trial.copy(), ratio))
        print(f"  [{i+1:2d}/{N_RANDOM}] ratio = {ratio:.3f}")
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_sigma = sigma_trial.copy()
    
    print(f"\nBest random: ratio = {best_ratio:.3f}")
    print()
    
    # L-BFGS-B
    print("L-BFGS-B OPTIMIZATION...")
    bounds = [(SIGMA_MIN, SIGMA_MAX)] * (4 * N_SEGMENTS)
    result = minimize(loss_fn, x0=best_sigma, method='L-BFGS-B',
                      bounds=bounds, options={'maxiter': MAX_ITER, 'disp': True})
    
    opt_sigma = result.x
    opt_ratio, opt_pass, opt_block = evaluate_ratio(opt_sigma, verbose=True)
    
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Baseline:   {baseline_ratio:.3f}")
    print(f"Best random: {best_ratio:.3f}")
    print(f"Optimized:   {opt_ratio:.3f}")
    print(f"\nOptimized sigma:\n{opt_sigma.reshape((4, N_SEGMENTS))}")
    
    # Visualize
    sigma_opt = build_heat_source_12blocks(opt_sigma)
    T_opt = solve_temperature(sigma_opt)
    c_opt = C0 * np.sqrt(T_opt / T0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    ax = axes[0]
    im = ax.imshow(T_opt.T, origin='lower', aspect='auto',
                   extent=[0, NX*DX*1000, 0, NY*DX*1000], cmap='hot')
    ax.set_title(f'Optimized T (K)\nmax={T_opt.max():.1f} K')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1]
    im = ax.imshow(c_opt.T, origin='lower', aspect='auto',
                   extent=[0, NX*DX*1000, 0, NY*DX*1000], cmap='viridis')
    ax.set_title(f'Optimized c (m/s)\nmax={c_opt.max():.1f} m/s')
    ax.set_xlabel('x (mm)')
    plt.colorbar(im, ax=ax)
    
    ax = axes[2]
    ax.bar(['Baseline', 'Best random', 'Optimized'],
           [baseline_ratio, best_ratio, opt_ratio],
           color=['gray', 'lightblue', 'green'], edgecolor='black')
    ax.set_ylabel('Ratio (5 kHz / 8 kHz)')
    ax.set_title('Filter selectivity')
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('training_results_12blocks.png', dpi=150)
    print("Saved: training_results_12blocks.png")
    plt.show()
    
    np.savez('training_results_12blocks.npz',
             baseline_ratio=baseline_ratio, baseline_pass=baseline_pass,
             baseline_block=baseline_block, best_random_ratio=best_ratio,
             optimized_ratio=opt_ratio, optimized_sigma=opt_sigma,
             optimized_pass=opt_pass, optimized_block=opt_block)
    print("Saved: training_results_12blocks.npz")


if __name__ == '__main__':
    train()
