#!/usr/bin/env python3
"""Train temperature field to act as a frequency filter."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from acoustic_filter_fdtd import (
    build_temperature_field, build_sound_speed, run_fdtd,
    N_BLOCKS, T0, T_MAX, LX, LY, DT
)

fig_dir = Path(__file__).parent / "figures"
fig_dir.mkdir(exist_ok=True)

# =============================================================================
# Baseline: uniform temperature
# =============================================================================
print("Computing baseline (uniform temperature)...")
T_base = np.full(N_BLOCKS, T0)
c_base = build_sound_speed(build_temperature_field(T_base))

n_steps = int(10.0 * LX / (343.0 * DT))
n_steps = max(n_steps, int(10.0 / (2000.0 * DT)))

power_2k_base, _, _, _ = run_fdtd(c_base, 2000.0, n_steps)
power_5k_base, _, _, _ = run_fdtd(c_base, 5000.0, n_steps)

print(f"Baseline power: 2kHz={power_2k_base:.6f}, 5kHz={power_5k_base:.6f}")

# =============================================================================
# Training objective
# =============================================================================
# Goal: maximize power at 2kHz, minimize power at 5kHz
# We normalize by baseline to make targets dimensionless
TARGET_1 = 2.0 * power_2k_base   # want 2x baseline at 2kHz
TARGET_2 = 0.1 * power_5k_base   # want 0.1x baseline at 5kHz

print(f"Targets: 2kHz={TARGET_1:.6f}, 5kHz={TARGET_2:.6f}")


def loss_fn(T_blocks):
    """Compute training loss."""
    T = build_temperature_field(T_blocks)
    c = build_sound_speed(T)

    p1, _, _, _ = run_fdtd(c, 2000.0, n_steps)
    p2, _, _, _ = run_fdtd(c, 5000.0, n_steps)

    loss = (p1 - TARGET_1) ** 2 + (p2 - TARGET_2) ** 2
    return loss


# =============================================================================
# Optimization
# =============================================================================
print("\nStarting optimization...")
print("(Each iteration = 2 FDTD runs, ~1-2 seconds)")

# Initial guess: all blocks at T0 + 50K
T0_init = np.full(N_BLOCKS, T0 + 50.0)

# Bounds: each block between ambient and max
bounds = [(T0, T_MAX) for _ in range(N_BLOCKS)]

# Progress callback
iteration = [0]

def callback(T_blocks):
    iteration[0] += 1
    if iteration[0] % 5 == 0:
        p1, p2 = evaluate_powers(T_blocks)
        print(f"  Iter {iteration[0]:3d}: 2kHz={p1:.6f}, 5kHz={p2:.6f}, "
              f"ratio={p1/max(p2,1e-10):.2f}, loss={loss_fn(T_blocks):.6e}")


def evaluate_powers(T_blocks):
    T = build_temperature_field(T_blocks)
    c = build_sound_speed(T)
    p1, _, _, _ = run_fdtd(c, 2000.0, n_steps)
    p2, _, _, _ = run_fdtd(c, 5000.0, n_steps)
    return p1, p2


result = minimize(
    loss_fn,
    T0_init,
    method='L-BFGS-B',
    bounds=bounds,
    options={'maxiter': 100, 'disp': True},
    callback=callback
)

T_opt = result.x
print(f"\nOptimization finished: {result.success}")
print(f"Final loss: {result.fun:.6e}")
print(f"Final temperatures:")
for k, Tk in enumerate(T_opt):
    print(f"  Block {k}: {Tk:.1f} K")

# =============================================================================
# Evaluate and plot
# =============================================================================
p1_opt, p2_opt = evaluate_powers(T_opt)
print(f"\nOptimized power: 2kHz={p1_opt:.6f}, 5kHz={p2_opt:.6f}")
print(f"Ratio (2k/5k): {p1_opt/max(p2_opt,1e-10):.2f}")
print(f"Improvement over baseline ratio: "
      f"{(p1_opt/p2_opt)/(power_2k_base/power_5k_base):.2f}x")

# Plot
T_opt_field = build_temperature_field(T_opt)
c_opt = build_sound_speed(T_opt_field)

_, _, _, snaps_2k = run_fdtd(c_opt, 2000.0, n_steps, record_interval=n_steps-1)
_, _, _, snaps_5k = run_fdtd(c_opt, 5000.0, n_steps, record_interval=n_steps-1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
im = ax.imshow(T_opt_field.T, origin='lower', cmap='hot',
               extent=[0, LX*1000, 0, LY*1000])
ax.set_title('Optimized Temperature [K]')
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
fig.colorbar(im, ax=ax)

ax = axes[1]
p2 = snaps_2k[-1]
im = ax.imshow(p2.T, origin='lower', cmap='RdBu_r',
               vmin=-np.abs(p2).max(), vmax=np.abs(p2).max(),
               extent=[0, LX*1000, 0, LY*1000])
ax.set_title(f'2 kHz (pass)\nPower = {p1_opt:.4f}')
fig.colorbar(im, ax=ax)

ax = axes[2]
p5 = snaps_5k[-1]
im = ax.imshow(p5.T, origin='lower', cmap='RdBu_r',
               vmin=-np.abs(p5).max(), vmax=np.abs(p5).max(),
               extent=[0, LX*1000, 0, LY*1000])
ax.set_title(f'5 kHz (block)\nPower = {p2_opt:.4f}')
fig.colorbar(im, ax=ax)

plt.tight_layout()
out_path = fig_dir / "trained_filter.png"
plt.savefig(out_path, dpi=150)
print(f"\nSaved: {out_path}")
plt.close()
