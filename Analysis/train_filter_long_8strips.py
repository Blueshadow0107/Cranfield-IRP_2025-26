"""
train_filter_long_8strips.py

Longer domain (800mm) with 8 strips x 3 segments = 24 blocks.
Goal: accumulate enough phase shift for strong frequency filtering.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from acoustic_filter_with_thermal import solve_temperature, run_fdtd, C0, T0

# =============================================================================
# NEW GEOMETRY
# =============================================================================
NX, NY = 400, 50       # 800mm x 100mm
DX = 0.002

# 8 strips spaced 80mm apart, starting at 80mm
STRIP_POSITIONS = np.array([0.080, 0.160, 0.240, 0.320,
                             0.400, 0.480, 0.560, 0.640])
STRIP_WIDTH = 0.020
N_SEGMENTS = 3

# Measurement: spatial average over last ~200mm
I_OUT_START = NX - 100
I_OUT_END = NX - 10
I_SRC_START = 3
I_SRC_END = 6

# =============================================================================
# SETTINGS
# =============================================================================
F_PASS = 5000.0
F_BLOCK = 8000.0
SIGMA_MIN = 0.0
SIGMA_MAX = 15.0
N_RANDOM = 20
MAX_ITER = 100


def build_heat_source(sigma_amps, nx=NX, ny=NY, dx=DX):
    sigma = np.zeros((nx, ny))
    half_w = int(STRIP_WIDTH / (2 * dx))
    seg_h = ny // N_SEGMENTS
    idx = 0
    for x_k in STRIP_POSITIONS:
        i_c = int(x_k / dx)
        i_s = max(0, i_c - half_w)
        i_e = min(nx, i_c + half_w)
        for seg in range(N_SEGMENTS):
            j_s = seg * seg_h
            j_e = min(ny, (seg + 1) * seg_h)
            sigma[i_s:i_e, j_s:j_e] = sigma_amps[idx]
            idx += 1
    return sigma


def evaluate_ratio(sigma_amps, verbose=False):
    sigma = build_heat_source(sigma_amps)
    T = solve_temperature(sigma, dx=DX, D=2e-5, T0=T0)
    c_field = C0 * np.sqrt(T / T0)
    
    p_pass, _, _ = run_fdtd(c_field, F_PASS, i_src_start=I_SRC_START,
                             i_src_end=I_SRC_END, i_out_start=I_OUT_START,
                             i_out_end=I_OUT_END)
    p_block, _, _ = run_fdtd(c_field, F_BLOCK, i_src_start=I_SRC_START,
                              i_src_end=I_SRC_END, i_out_start=I_OUT_START,
                              i_out_end=I_OUT_END)
    
    ratio = p_pass / (p_block + 1e-30)
    if verbose:
        print(f"  Tmax={T.max():.1f}K, cmax={c_field.max():.1f}m/s")
        print(f"  5k={p_pass:.4e}, 8k={p_block:.4e}, ratio={ratio:.3f}")
    return ratio, p_pass, p_block


def loss_fn(sigma_amps):
    ratio, _, _ = evaluate_ratio(sigma_amps)
    return -ratio


def train():
    print("=" * 60)
    print("LONG CHANNEL: 8 STRIPS x 3 SEGMENTS = 24 BLOCKS")
    print(f"Domain: {NX*DX*1000:.0f}mm x {NY*DX*1000:.0f}mm ({NX}x{NY})")
    print("=" * 60)
    print(f"Random search: {N_RANDOM}, L-BFGS-B maxiter: {MAX_ITER}")
    print()
    
    # Baseline
    print("BASELINE:")
    base_r, base_p, base_b = evaluate_ratio(np.zeros(8*N_SEGMENTS), verbose=True)
    print()
    
    # Random search
    print("RANDOM SEARCH...")
    best_r = -1.0
    best_s = None
    history = []
    for i in range(N_RANDOM):
        s = np.random.uniform(SIGMA_MIN, SIGMA_MAX, 8*N_SEGMENTS)
        r, p5, p8 = evaluate_ratio(s)
        history.append((s.copy(), r))
        print(f"  [{i+1:2d}/{N_RANDOM}] ratio={r:.3f}")
        if r > best_r:
            best_r = r
            best_s = s.copy()
    print(f"\nBest random: ratio={best_r:.3f}\n")
    
    # L-BFGS-B
    print("L-BFGS-B...")
    bounds = [(SIGMA_MIN, SIGMA_MAX)] * (8*N_SEGMENTS)
    res = minimize(loss_fn, x0=best_s, method='L-BFGS-B',
                   bounds=bounds, options={'maxiter': MAX_ITER})
    opt_s = res.x
    opt_r, opt_p, opt_b = evaluate_ratio(opt_s, verbose=True)
    
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Baseline:    {base_r:.3f}")
    print(f"Best random: {best_r:.3f}")
    print(f"Optimized:   {opt_r:.3f}")
    print(f"\nOptimized sigma (8x3):\n{opt_s.reshape((8, N_SEGMENTS))}")
    
    # Plot
    sigma_opt = build_heat_source(opt_s)
    T_opt = solve_temperature(sigma_opt, dx=DX)
    c_opt = C0 * np.sqrt(T_opt / T0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    im = ax.imshow(T_opt.T, origin='lower', aspect='auto',
                   extent=[0, NX*DX*1000, 0, NY*DX*1000], cmap='hot')
    ax.set_title(f'T (K), max={T_opt.max():.1f}')
    ax.set_xlabel('x (mm)')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1]
    im = ax.imshow(c_opt.T, origin='lower', aspect='auto',
                   extent=[0, NX*DX*1000, 0, NY*DX*1000], cmap='viridis')
    ax.set_title(f'c (m/s), max={c_opt.max():.1f}')
    ax.set_xlabel('x (mm)')
    plt.colorbar(im, ax=ax)
    
    ax = axes[2]
    ax.bar(['Baseline', 'Best random', 'Optimized'],
           [base_r, best_r, opt_r], color=['gray', 'lightblue', 'green'],
           edgecolor='black')
    ax.set_ylabel('Ratio (5kHz/8kHz)')
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('training_long_8strips.png', dpi=150)
    print("Saved: training_long_8strips.png")
    plt.show()
    
    np.savez('training_long_8strips.npz',
             baseline_ratio=base_r, optimized_ratio=opt_r,
             optimized_sigma=opt_s)
    print("Saved: training_long_8strips.npz")


if __name__ == '__main__':
    train()
