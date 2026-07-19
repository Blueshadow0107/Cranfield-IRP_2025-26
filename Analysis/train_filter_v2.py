#!/usr/bin/env python3
"""Train temperature field with ratio objective and wider frequency spread."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from acoustic_filter_fdtd import (
    build_temperature_field, build_sound_speed, run_fdtd,
    N_BLOCKS, T0, T_MAX, LX, DT
)

fig_dir = Path(__file__).parent / "figures"
fig_dir.mkdir(exist_ok=True)

# =============================================================================
# Configuration
# =============================================================================
F_PASS = 2000.0      # frequency to pass [Hz]
F_BLOCK = 8000.0     # frequency to block [Hz]

n_steps = int(8.0 * LX / (343.0 * DT))  # 8 transit times
n_steps = max(n_steps, int(8.0 / (F_PASS * DT)))

print(f"Grid: dx={1e3:.1f} mm, dt={DT:.3e} s")
print(f"Frequencies: pass={F_PASS/1e3:.1f} kHz, block={F_BLOCK/1e3:.1f} kHz")
print(f"Wavelengths: lambda_pass={343/F_PASS*1e3:.1f} mm, lambda_block={343/F_BLOCK*1e3:.1f} mm")
print(f"FDTD steps per run: {n_steps}")

# =============================================================================
# Evaluation
# =============================================================================
def evaluate_powers(T_blocks):
    """Return (power_pass, power_block) for given temperatures."""
    T = build_temperature_field(T_blocks)
    c = build_sound_speed(T)
    p_pass, _, _, _ = run_fdtd(c, F_PASS, n_steps)
    p_block, _, _, _ = run_fdtd(c, F_BLOCK, n_steps)
    return p_pass, p_block


def loss_fn(T_blocks):
    """Ratio objective: maximize power_pass / power_block."""
    p_pass, p_block = evaluate_powers(T_blocks)
    ratio = p_pass / (p_block + 1e-12)
    return -ratio


# =============================================================================
# Baseline
# =============================================================================
print("\n--- Baseline (uniform temperature) ---")
T_base = np.full(N_BLOCKS, T0)
p_pass_base, p_block_base = evaluate_powers(T_base)
print(f"Pass ({F_PASS/1e3:.1f} kHz): {p_pass_base:.6f}")
print(f"Block ({F_BLOCK/1e3:.1f} kHz): {p_block_base:.6f}")
print(f"Ratio: {p_pass_base/max(p_block_base,1e-10):.2f}")

# =============================================================================
# Random search for good initialization
# =============================================================================
print("\n--- Random search (20 samples) ---")
best_ratio = p_pass_base / max(p_block_base, 1e-10)
best_T = T_base.copy()

for trial in range(10):
    T_try = np.random.uniform(T0, T_MAX, N_BLOCKS)
    p_pass, p_block = evaluate_powers(T_try)
    ratio = p_pass / max(p_block, 1e-10)
    if ratio > best_ratio:
        best_ratio = ratio
        best_T = T_try.copy()
        print(f"  Trial {trial:2d}: ratio={ratio:.2f}  (pass={p_pass:.6f}, block={p_block:.6f})")

print(f"Best random ratio: {best_ratio:.2f}")

# =============================================================================
# Gradient-based optimization
# =============================================================================
print("\n--- L-BFGS-B optimization ---")
iteration = [0]

def callback(T_blocks):
    iteration[0] += 1
    if iteration[0] % 3 == 0:
        p_pass, p_block = evaluate_powers(T_blocks)
        ratio = p_pass / max(p_block, 1e-10)
        print(f"  Iter {iteration[0]:3d}: ratio={ratio:.2f}, "
              f"pass={p_pass:.6f}, block={p_block:.6f}")

bounds = [(T0, T_MAX) for _ in range(N_BLOCKS)]

result = minimize(
    loss_fn,
    best_T,
    method='L-BFGS-B',
    bounds=bounds,
    options={'maxiter': 30, 'ftol': 1e-6},
    callback=callback
)

T_opt = result.x
print(f"\nOptimization success: {result.success}")

# =============================================================================
# Final evaluation
# =============================================================================
print("\n--- Final Results ---")
p_pass_opt, p_block_opt = evaluate_powers(T_opt)
ratio_opt = p_pass_opt / max(p_block_opt, 1e-10)
ratio_base = p_pass_base / max(p_block_base, 1e-10)

print(f"Baseline:  pass={p_pass_base:.6f}, block={p_block_base:.6f}, ratio={ratio_base:.2f}")
print(f"Optimized: pass={p_pass_opt:.6f}, block={p_block_opt:.6f}, ratio={ratio_opt:.2f}")
print(f"Improvement: {ratio_opt/ratio_base:.2f}x")

print(f"\nOptimized temperatures:")
for k, Tk in enumerate(T_opt):
    print(f"  Block {k}: {Tk:.1f} K")

# =============================================================================
# Plot
# =============================================================================
T_opt_field = build_temperature_field(T_opt)
c_opt = build_sound_speed(T_opt_field)

_, _, _, snaps_pass = run_fdtd(c_opt, F_PASS, n_steps, record_interval=n_steps-1)
_, _, _, snaps_block = run_fdtd(c_opt, F_BLOCK, n_steps, record_interval=n_steps-1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
im = ax.imshow(T_opt_field.T, origin='lower', cmap='hot',
               extent=[0, LX*1000, 0, 0.1*1000])
ax.set_title('Optimized Temperature [K]')
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
fig.colorbar(im, ax=ax)

ax = axes[1]
p = snaps_pass[-1]
im = ax.imshow(p.T, origin='lower', cmap='RdBu_r',
               vmin=-np.abs(p).max(), vmax=np.abs(p).max(),
               extent=[0, LX*1000, 0, 0.1*1000])
ax.set_title(f'Pass: {F_PASS/1e3:.1f} kHz\nPower={p_pass_opt:.5f}')
fig.colorbar(im, ax=ax)

ax = axes[2]
p = snaps_block[-1]
im = ax.imshow(p.T, origin='lower', cmap='RdBu_r',
               vmin=-np.abs(p).max(), vmax=np.abs(p).max(),
               extent=[0, LX*1000, 0, 0.1*1000])
ax.set_title(f'Block: {F_BLOCK/1e3:.1f} kHz\nPower={p_block_opt:.5f}')
fig.colorbar(im, ax=ax)

plt.suptitle(f'Frequency Filter (ratio = {ratio_opt:.1f})', fontsize=14)
plt.tight_layout()
out_path = fig_dir / "trained_filter_v2.png"
plt.savefig(out_path, dpi=150)
print(f"\nSaved: {out_path}")
plt.close()
