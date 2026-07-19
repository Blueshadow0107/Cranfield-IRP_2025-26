"""
test_bragg_filter_v3.py

Bragg grating with corrected units (meters, not mm).
"""

import numpy as np
import matplotlib.pyplot as plt
from acoustic_filter_with_thermal import run_fdtd

NX, NY = 400, 50
DX = 0.002  # 2 mm in meters
C0 = 343.0

F_BLOCK = 8000.0
LAMBDA_BLOCK = C0 / F_BLOCK  # 42.875 mm = 0.042875 m

# CORRECTED UNITS: everything in meters
D_BRAGG = 0.024      # 24 mm period
STRIP_WIDTH = 0.010  # 10 mm hot strip
N_PERIODS = 20
START_X = 0.060      # 60 mm

T_HOT = 600.0
C_HOT = C0 * np.sqrt(T_HOT / 300.0)
C_COLD = C0

period_cells = int(D_BRAGG / DX)      # 12 cells
width_cells = int(STRIP_WIDTH / DX)   # 5 cells
start_i = int(START_X / DX)

c_field = np.ones((NX, NY)) * C_COLD

for n in range(N_PERIODS):
    i_start = start_i + n * period_cells
    i_end = i_start + width_cells
    if i_end < NX:
        c_field[i_start:i_end, :] = C_HOT

n_hot = C_COLD / C_HOT
n_eff = (width_cells * n_hot + (period_cells - width_cells) * 1.0) / period_cells
optical_period_mm = n_eff * D_BRAGG * 1000

print(f"Period: {D_BRAGG*1000:.1f} mm ({period_cells} cells)")
print(f"Strip: {STRIP_WIDTH*1000:.1f} mm ({width_cells} cells)")
print(f"Gap: {(D_BRAGG - STRIP_WIDTH)*1000:.1f} mm ({period_cells - width_cells} cells)")
print(f"n_eff = {n_eff:.3f}")
print(f"Optical period = {optical_period_mm:.2f} mm")
print(f"Target lambda/2 = {LAMBDA_BLOCK/2*1000:.2f} mm")

# Count actual strips placed
n_strips = np.sum(np.any(c_field != C_COLD, axis=1))
print(f"Actual strips placed: {n_strips}")

# Run FDTD
SIM_DURATION = 0.020
for f in [5000.0, 8000.0]:
    p, dt, ns = run_fdtd(c_field, f, duration=SIM_DURATION,
                         i_src_start=3, i_src_end=6,
                         i_out_start=NX-100, i_out_end=NX-10)
    print(f"f={f:.0f}Hz: power={p:.4e}")

p5, _, _ = run_fdtd(c_field, 5000.0, duration=SIM_DURATION,
                     i_src_start=3, i_src_end=6,
                     i_out_start=NX-100, i_out_end=NX-10)
p8, _, _ = run_fdtd(c_field, 8000.0, duration=SIM_DURATION,
                     i_src_start=3, i_src_end=6,
                     i_out_start=NX-100, i_out_end=NX-10)
print(f"\nRATIO (5k/8k) = {p5/(p8+1e-30):.3f}")
