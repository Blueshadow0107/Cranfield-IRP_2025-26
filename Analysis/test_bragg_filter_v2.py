"""
test_bragg_filter_v2.py

Bragg grating with corrected period d=24mm (12 cells).
Target: block 8 kHz, pass 5 kHz.
"""

import numpy as np
import matplotlib.pyplot as plt
from acoustic_filter_with_thermal import run_fdtd

NX, NY = 400, 50
DX = 0.002
C0 = 343.0

# Bragg design
F_BLOCK = 8000.0
LAMBDA_BLOCK = C0 / F_BLOCK  # 42.875 mm

# CORRECTED: d = 24 mm (12 cells)
PERIOD_MM = 24.0
STRIP_WIDTH = 0.010
N_PERIODS = 20
START_X = 0.060

T_HOT = 600.0
C_HOT = C0 * np.sqrt(T_HOT / 300.0)  # 485 m/s
C_COLD = C0

# Build c-field
period_cells = int(PERIOD_MM / DX)    # 12 cells
width_cells = int(STRIP_WIDTH / DX)   # 5 cells
start_i = int(START_X / DX)

c_field = np.ones((NX, NY)) * C_COLD

for n in range(N_PERIODS):
    i_start = start_i + n * period_cells
    i_end = i_start + width_cells
    if i_end < NX:
        c_field[i_start:i_end, :] = C_HOT

# Effective refractive index check
n_hot = C_COLD / C_HOT
n_eff = (width_cells * n_hot + (period_cells - width_cells) * 1.0) / period_cells
optical_period = n_eff * PERIOD_MM

print(f"Period: {PERIOD_MM} mm ({period_cells} cells)")
print(f"Strip width: {STRIP_WIDTH*1000:.0f} mm ({width_cells} cells)")
print(f"Gap: {(PERIOD_MM - STRIP_WIDTH)*1000:.0f} mm ({period_cells - width_cells} cells)")
print(f"n_hot = {n_hot:.3f}, n_eff = {n_eff:.3f}")
print(f"Optical period = {optical_period:.2f} mm")
print(f"Target lambda/2 = {LAMBDA_BLOCK/2*1000:.2f} mm")
print(f"Mismatch = {abs(optical_period - LAMBDA_BLOCK/2*1000):.2f} mm")
print(f"r = {(C_HOT/C_COLD - 1)/(C_HOT/C_COLD + 1):.3f}")

# Visualize
plt.figure(figsize=(12, 3))
plt.imshow(c_field.T, origin='lower', aspect='auto',
           extent=[0, NX*DX*1000, 0, NY*DX*1000], cmap='coolwarm')
plt.colorbar(label='c (m/s)')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title(f'Bragg grating v2: {N_PERIODS} periods, d={PERIOD_MM}mm')
plt.tight_layout()
plt.savefig('bragg_grating_v2.png', dpi=150)
plt.close()

# Run FDTD
SIM_DURATION = 0.020
I_SRC_START = 3
I_SRC_END = 6
I_OUT_START = NX - 100
I_OUT_END = NX - 10

print(f"\nRunning FDTD (duration = {SIM_DURATION*1000:.0f} ms)...")

for f in [5000.0, 8000.0]:
    print(f"\nf = {f:.0f} Hz:")
    p, dt, n_steps = run_fdtd(c_field, f, duration=SIM_DURATION,
                               i_src_start=I_SRC_START, i_src_end=I_SRC_END,
                               i_out_start=I_OUT_START, i_out_end=I_OUT_END)
    print(f"  dt = {dt:.2e}, n_steps = {n_steps}")
    print(f"  Transmitted power = {p:.4e}")

# Ratio
p5, _, _ = run_fdtd(c_field, 5000.0, duration=SIM_DURATION,
                     i_src_start=I_SRC_START, i_src_end=I_SRC_END,
                     i_out_start=I_OUT_START, i_out_end=I_OUT_END)
p8, _, _ = run_fdtd(c_field, 8000.0, duration=SIM_DURATION,
                     i_src_start=I_SRC_START, i_src_end=I_SRC_END,
                     i_out_start=I_OUT_START, i_out_end=I_OUT_END)

print(f"\n{'='*50}")
print(f"RATIO (5kHz / 8kHz) = {p5/(p8+1e-30):.3f}")
print(f"{'='*50}")
