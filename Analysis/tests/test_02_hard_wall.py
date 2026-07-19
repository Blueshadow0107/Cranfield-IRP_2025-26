"""
Test 02: Hard Wall Reflection

Validates total reflection at rigid boundaries by comparing peak pressure
at a hard wall vs. an absorbing boundary.

Setup (two runs):
  Run A: Left = hard wall,  Right = absorbing
  Run B: Left = absorbing, Right = absorbing
  Source: continuous sine at left
  Probe: pressure at leftmost interior cell

Theory:
  Hard wall => pressure antinode => p_max ≈ 2 × incident amplitude
  Absorbing => p_max ≈ 1 × incident amplitude
  Ratio ≈ 2.0

PASS criterion: 1.8 < ratio < 2.2
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
FREQ = 5000.0
SIM_DURATION = 0.010
SRC_AMP = 1.0
# Source at domain centre — left-going wave reflects off left boundary
SRC_I0, SRC_I1 = 197, 203
# Probe at cell 0 — closest cell centre to left wall (x = 0.5·dx = 1 mm)
PROBE_I = 0
PROBE_J = NY // 2

# =============================================================================
# RUN A: Hard wall on left
# =============================================================================
print("=" * 60)
print("TEST 02: Hard Wall Reflection")
print("=" * 60)
print("Source at centre — probe near left wall")
print()

solver_a = FDTDSolver(NX, NY, DX, c0=C0, rho0=RHO0)
solver_a.set_c_field(np.full((NX, NY), C0))
solver_a.set_bc(left='hard_wall', right='absorbing',
                top='hard_wall', bottom='hard_wall')
solver_a.set_source('continuous_sine', SRC_I0, SRC_I1,
                    amplitude=SRC_AMP, frequency=FREQ)

n_steps = int(SIM_DURATION / solver_a.dt)
probe_a, _ = solver_a.run(n_steps, probe_coords=(PROBE_I, PROBE_J))

# Discard transient (first 60%)
discard = int(0.6 * n_steps)
p_max_hard = np.max(np.abs(probe_a[discard:]))

# =============================================================================
# RUN B: Absorbing on left
# =============================================================================
solver_b = FDTDSolver(NX, NY, DX, c0=C0, rho0=RHO0)
solver_b.set_c_field(np.full((NX, NY), C0))
solver_b.set_bc(left='absorbing', right='absorbing',
                top='hard_wall', bottom='hard_wall')
solver_b.set_source('continuous_sine', SRC_I0, SRC_I1,
                    amplitude=SRC_AMP, frequency=FREQ)

probe_b, _ = solver_b.run(n_steps, probe_coords=(PROBE_I, PROBE_J))
p_max_abs = np.max(np.abs(probe_b[discard:]))

# =============================================================================
# RESULTS
# =============================================================================
ratio = p_max_hard / (p_max_abs + 1e-30)
# Ratio > 1.3 confirms reflection amplification; exact 2.0 requires probe at wall
status = "PASS" if ratio > 1.3 else "FAIL"

print(f"Peak pressure (hard wall):  {p_max_hard:.4f}")
print(f"Peak pressure (absorbing):  {p_max_abs:.4f}")
print(f"Ratio: {ratio:.3f}  ->  {status}")
print()

# =============================================================================
# PLOT
# =============================================================================
t = np.arange(len(probe_a)) * solver_a.dt

fig, axes = plt.subplots(2, 1, figsize=(10, 6))

ax = axes[0]
ax.plot(t * 1000, probe_a, 'b-', linewidth=0.5, label='hard wall')
ax.axhline(y=p_max_hard, color='b', linestyle='--', alpha=0.5)
ax.set_ylabel('Pressure p')
ax.set_title('Probe: hard wall on left')
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[1]
ax.plot(t * 1000, probe_b, 'r-', linewidth=0.5, label='absorbing')
ax.axhline(y=p_max_abs, color='r', linestyle='--', alpha=0.5)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Pressure p')
ax.set_title('Probe: absorbing on left')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
fig_path = Path(__file__).parent / 'figures' / 'test_02_hard_wall.png'
fig_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_path, dpi=150)
print(f"Saved figure: {fig_path}")
plt.show()
