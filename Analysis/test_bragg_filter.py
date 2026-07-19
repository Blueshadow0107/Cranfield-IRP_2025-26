"""
test_bragg_filter.py

Periodic temperature strips as a Bragg grating.
Period tuned to lambda/2 for 8 kHz → strong reflection at 8 kHz.
5 kHz is off-resonance → should transmit better.
"""

import numpy as np
import matplotlib.pyplot as plt
from acoustic_filter_with_thermal import run_fdtd

# =============================================================================
# PARAMETERS
# =============================================================================
NX, NY = 400, 50
DX = 0.002
C0 = 343.0
RHO0 = 1.225

# Bragg design for 8 kHz
F_BLOCK = 8000.0
LAMBDA_BLOCK = C0 / F_BLOCK  # 42.875 mm
D_BRAGG = LAMBDA_BLOCK / 2.0   # 21.4375 mm

# Strip geometry
STRIP_WIDTH = 0.010            # 10 mm hot strip
N_PERIODS = 20
START_X = 0.060                # first strip starts at 60 mm

# Temperature / sound speed
T_COLD = 300.0
T_HOT = 600.0
C_COLD = C0
C_HOT = C0 * np.sqrt(T_HOT / T_COLD)  # 485 m/s

# Build c-field directly (no thermal solve for this test)
c_field = np.ones((NX, NY)) * C_COLD

period_cells = int(D_BRAGG / DX)      # ~10.7 → 11 cells
width_cells = int(STRIP_WIDTH / DX)   # 5 cells
start_i = int(START_X / DX)           # i=30

print(f"Bragg period: {D_BRAGG*1000:.2f} mm ({period_cells} cells)")
print(f"Strip width: {STRIP_WIDTH*1000:.1f} mm ({width_cells} cells)")
print(f"Gap: {(D_BRAGG - STRIP_WIDTH)*1000:.2f} mm ({period_cells - width_cells} cells)")
print(f"c_hot = {C_HOT:.1f} m/s, c_cold = {C_COLD:.1f} m/s")
print(f"Impedance ratio Z_hot/Z_cold = {C_HOT/C_COLD:.3f}")
print(f"Reflection per interface r = {(C_HOT/C_COLD - 1)/(C_HOT/C_COLD + 1):.3f}")

for n in range(N_PERIODS):
    i_start = start_i + n * period_cells
    i_end = i_start + width_cells
    if i_end < NX:
        c_field[i_start:i_end, :] = C_HOT

# Visual check
plt.figure(figsize=(12, 3))
plt.imshow(c_field.T, origin='lower', aspect='auto',
           extent=[0, NX*DX*1000, 0, NY*DX*1000], cmap='coolwarm')
plt.colorbar(label='c (m/s)')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title(f'Bragg grating: {N_PERIODS} periods, d={D_BRAGG*1000:.1f}mm')
plt.axvline(x=START_X*1000, color='g', linestyle='--', label='grating start')
plt.legend()
plt.tight_layout()
plt.savefig('bragg_grating_layout.png', dpi=150)
plt.close()

# =============================================================================
# RUN FDTD
# =============================================================================
SIM_DURATION = 0.020  # 20 ms
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

print("\nDone.")
