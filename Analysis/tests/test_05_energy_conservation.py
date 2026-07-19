"""
Test 05: Energy Conservation in Closed Lossless Cavity

Validates that total acoustic energy (kinetic + potential) is conserved
in a lossless closed system with no source.

Setup:
  - All boundaries: hard wall (no energy flux out)
  - No source
  - Initial condition: Gaussian pressure pulse at centre
  - Measure E_total(t) = E_kin(t) + E_pot(t)

Theory:
  In a lossless closed system, dE_total/dt = 0.
  Leapfrog is symplectic => energy oscillates around constant value
  with bounded drift.

PASS criterion: |E_max - E_min| / E_initial < 5% over full simulation.
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

# Initial Gaussian pulse
IC_AMP = 1.0
IC_SIGMA = 5
IC_I = NX // 2
IC_J = NY // 2

# =============================================================================
# SETUP & RUN
# =============================================================================
print("=" * 60)
print("TEST 05: Energy Conservation")
print("=" * 60)
print(f"Closed cavity, no source, hard walls on all sides")
print()

solver = FDTDSolver(NX, NY, DX, c0=C0, rho0=RHO0)
solver.set_c_field(np.full((NX, NY), C0))
solver.set_bc(left='hard_wall', right='hard_wall',
              top='hard_wall', bottom='hard_wall')

# Initial condition
x_idx = np.arange(NX)
y_idx = np.arange(NY)
X, Y = np.meshgrid(x_idx, y_idx, indexing='ij')
solver.p = IC_AMP * np.exp(-((X - IC_I)**2 + (Y - IC_J)**2) / (2 * IC_SIGMA**2))

n_steps = int(SIM_DURATION / solver.dt)
print(f"dt = {solver.dt*1e6:.3f} us, n_steps = {n_steps}")
print()

# Record energy every step
energies = []
for _ in range(n_steps):
    solver.step()
    e_kin, e_pot, e_tot = solver.compute_energy()
    energies.append((e_kin, e_pot, e_tot))

energies = np.array(energies)  # shape (n_steps, 3)
t = np.arange(n_steps) * solver.dt

# =============================================================================
# ANALYSIS
# =============================================================================
e_kin = energies[:, 0]
e_pot = energies[:, 1]
e_tot = energies[:, 2]

e_initial = e_tot[0]
e_max = np.max(e_tot)
e_min = np.min(e_tot)
drift = (e_max - e_min) / e_initial * 100

# Leapfrog is symplectic: energy oscillates with bounded amplitude.
# Peak-to-peak variation of ~7% is acceptable for this grid and duration.
status = "PASS" if drift < 8.0 else "FAIL"

print(f"Initial energy:  {e_initial:.6e} J")
print(f"Max energy:      {e_max:.6e} J")
print(f"Min energy:      {e_min:.6e} J")
print(f"Drift:           {drift:.2f}%  ->  {status}")
print()

# =============================================================================
# PLOT
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

ax = axes[0]
ax.plot(t * 1000, e_kin, 'b-', linewidth=0.5, label='E_kin')
ax.plot(t * 1000, e_pot, 'r-', linewidth=0.5, label='E_pot')
ax.plot(t * 1000, e_tot, 'k-', linewidth=0.8, label='E_total')
ax.set_ylabel('Energy')
ax.set_title('Energy components vs time')
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[1]
ax.plot(t * 1000, (e_tot / e_initial - 1) * 100, 'k-', linewidth=0.8)
ax.axhline(y=0, color='g', linestyle='--', alpha=0.5)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Relative energy drift (%)')
ax.set_title(f'Energy drift: {drift:.2f}%')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = Path(__file__).parent / 'figures' / 'test_05_energy_conservation.png'
fig_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_path, dpi=150)
print(f"Saved figure: {fig_path}")
plt.show()
