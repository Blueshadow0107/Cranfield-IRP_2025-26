#!/usr/bin/env python3
"""Test random temperature pattern to verify filtering behavior."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from acoustic_filter_fdtd import (
    build_temperature_field, build_sound_speed, run_fdtd,
    N_BLOCKS, T0, T_MAX, LX, LY, DT, NX, NY
)

fig_dir = Path(__file__).parent / "figures"
fig_dir.mkdir(exist_ok=True)

# Random temperatures: some hot, some cold
np.random.seed(42)
T_blocks = np.random.uniform(T0, T_MAX, N_BLOCKS)
print("Random block temperatures:")
for k, Tk in enumerate(T_blocks):
    print(f"  Block {k}: {Tk:.1f} K")

T = build_temperature_field(T_blocks)
c = build_sound_speed(T)

# Run at both frequencies
n_steps = int(10.0 * LX / (343.0 * DT))
n_steps = max(n_steps, int(10.0 / (2000.0 * DT)))

print(f"\nRunning {n_steps} steps per frequency...")

power_2k, probe_p_2k, probe_u_2k, snaps_2k = run_fdtd(c, 2000.0, n_steps, record_interval=n_steps-1)
power_5k, probe_p_5k, probe_u_5k, snaps_5k = run_fdtd(c, 5000.0, n_steps, record_interval=n_steps-1)

print(f"\nTransmitted power:")
print(f"  2 kHz: {power_2k:.6f}")
print(f"  5 kHz: {power_5k:.6f}")
print(f"  Ratio (2k/5k): {power_2k / max(abs(power_5k), 1e-10):.2f}")

# Plot temperature field and pressure fields at both frequencies
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
im = ax.imshow(T.T, origin='lower', cmap='hot', extent=[0, LX*1000, 0, LY*1000])
ax.set_title('Temperature field [K]')
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
fig.colorbar(im, ax=ax)

ax = axes[1]
p2 = snaps_2k[-1]
im = ax.imshow(p2.T, origin='lower', cmap='RdBu_r',
               vmin=-np.abs(p2).max(), vmax=np.abs(p2).max(),
               extent=[0, LX*1000, 0, LY*1000])
ax.set_title(f'Pressure at 2 kHz\nPower = {power_2k:.4f}')
fig.colorbar(im, ax=ax)

ax = axes[2]
p5 = snaps_5k[-1]
im = ax.imshow(p5.T, origin='lower', cmap='RdBu_r',
               vmin=-np.abs(p5).max(), vmax=np.abs(p5).max(),
               extent=[0, LX*1000, 0, LY*1000])
ax.set_title(f'Pressure at 5 kHz\nPower = {power_5k:.4f}')
fig.colorbar(im, ax=ax)

plt.tight_layout()
out_path = fig_dir / "random_blocks_test.png"
plt.savefig(out_path, dpi=150)
print(f"\nSaved: {out_path}")
plt.close()
