"""
Test 01: Pulse Transit Time

Validates wave propagation speed by measuring pulse arrival time at outlet.

Setup:
  - Uniform medium, absorbing on all sides
  - Gaussian pulse injected at left
  - Probe at outlet (NX-4, NY//2)

Theory:
  Transit time = probe_distance / c0

PASS criterion: |t_measured - t_theory| / t_theory < 3%.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from fdtd_core import FDTDSolver

# =============================================================================
# PARAMETERS
# =============================================================================
NX, NY = 400, 50
DX = 0.002
C0 = 343.0
RHO0 = 1.225

SIM_DURATION = 0.020  # 20 ms

SRC_I0, SRC_I1 = 3, 6
PULSE_T0 = 0.0005
PULSE_TAU = 0.0001
PULSE_F0 = 6000.0
PULSE_AMP = 1.0

PROBE_I = NX - 4
PROBE_J = NY // 2

# =============================================================================
# RUN
# =============================================================================
print("=" * 60)
print("TEST 01: Pulse Transit Time")
print("=" * 60)

solver = FDTDSolver(NX, NY, DX, c0=C0, rho0=RHO0)
solver.set_c_field(np.full((NX, NY), C0))
solver.set_bc(left='absorbing', right='absorbing',
              top='absorbing', bottom='absorbing')
solver.set_source('gaussian_pulse', SRC_I0, SRC_I1,
                  amplitude=PULSE_AMP, f0=PULSE_F0, tau=PULSE_TAU, t0=PULSE_T0)

n_steps = int(SIM_DURATION / solver.dt)
probe, _ = solver.run(n_steps, probe_coords=(PROBE_I, PROBE_J))
t = np.arange(len(probe)) * solver.dt

# =============================================================================
# ANALYSIS
# =============================================================================
peak_idx = np.argmax(np.abs(probe))
t_measured = t[peak_idx]
# Transit from source centre (cell ~4.5) to probe (cell PROBE_I)
src_centre = (SRC_I0 + SRC_I1) / 2.0
t_theory = (PROBE_I - src_centre) * DX / C0 + PULSE_T0
error = abs(t_measured - t_theory) / t_theory * 100
status = "PASS" if error < 3.0 else "FAIL"

print(f"Probe distance: {PROBE_I * DX * 1000:.1f} mm")
print(f"Transit time (theory): {t_theory * 1000:.3f} ms")
print(f"Transit time (measured): {t_measured * 1000:.3f} ms")
print(f"Error: {error:.2f}%  ->  {status}")
print()

# =============================================================================
# PLOT
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t * 1000, probe, 'b-', linewidth=0.8)
ax.axvline(x=t_theory * 1000, color='g', linestyle='--', label=f'theory: {t_theory*1000:.3f} ms')
ax.axvline(x=t_measured * 1000, color='r', linestyle='--', label=f'measured: {t_measured*1000:.3f} ms')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Pressure p')
ax.set_title('Test 01: Pulse arrival at outlet')
ax.grid(True, alpha=0.3)
ax.legend()

fig_path = Path(__file__).parent / 'figures' / 'test_01_pulse_arrival.png'
fig_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_path, dpi=150)
print(f"Saved figure: {fig_path}")
plt.show()
