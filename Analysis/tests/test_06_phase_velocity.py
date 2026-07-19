"""
Test 06: Phase Velocity Dispersion

Validates that the numerically computed wavelength matches theory
at multiple frequencies.

Setup:
  - Uniform medium, absorbing on left/right, hard wall top/bottom
  - Continuous sine source at left
  - After transient, measure spatial pressure profile
  - Find wavelength by peak-to-peak distance

Theory:
  lambda = c / f

PASS criterion: |lambda_num - lambda_theory| / lambda_theory < 3%
  for each tested frequency.
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

SIM_DURATION = 0.015  # 15 ms
SRC_AMP = 1.0
SRC_I0, SRC_I1 = 3, 6

# Frequencies to test
FREQS = [2000.0, 5000.0, 8000.0, 10000.0]

# =============================================================================
# RUN & ANALYSE
# =============================================================================
print("=" * 60)
print("TEST 06: Phase Velocity Dispersion")
print("=" * 60)
print(f"Grid: {NX}x{NY}, dx={DX*1000:.1f} mm")
print()

results = []

for freq in FREQS:
    solver = FDTDSolver(NX, NY, DX, c0=C0, rho0=RHO0)
    solver.set_c_field(np.full((NX, NY), C0))
    solver.set_bc(left='absorbing', right='absorbing',
                  top='hard_wall', bottom='hard_wall')
    solver.set_source('continuous_sine', SRC_I0, SRC_I1,
                      amplitude=SRC_AMP, frequency=freq)

    n_steps = int(SIM_DURATION / solver.dt)
    discard = int(0.6 * n_steps)

    # Run and record spatial profiles after transient
    profiles = []
    for n in range(n_steps):
        solver.step()
        if n >= discard:
            profiles.append(solver.p[:, NY // 2].copy())

    # Time-average to suppress temporal oscillations
    profile = np.mean(profiles, axis=0)

    # Find peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(profile, height=np.max(profile)*0.1, distance=5)
    peak_x = peaks * DX

    # Compute wavelength from adjacent peaks
    if len(peak_x) >= 2:
        wavelengths = np.diff(peak_x)
        lambda_num = np.mean(wavelengths)
    else:
        lambda_num = np.nan

    lambda_theory = C0 / freq
    err = abs(lambda_num - lambda_theory) / lambda_theory * 100 if not np.isnan(lambda_num) else np.nan
    status = "PASS" if err < 3.0 else "FAIL" if not np.isnan(err) else "FAIL"

    results.append((freq, lambda_theory, lambda_num, err, status, profile, peaks))
    print(f"f = {freq/1000:.1f} kHz:  theory={lambda_theory*1000:.3f} mm,  num={lambda_num*1000:.3f} mm,  err={err:.2f}%  ->  {status}")

print()

# =============================================================================
# PLOT
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, (freq, _, _, _, _, profile, peaks) in zip(axes, results):
    x = np.arange(NX) * DX * 1000
    ax.plot(x, profile, 'b-', linewidth=0.8)
    ax.plot(peaks * DX * 1000, profile[peaks], 'ro', markersize=4)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('p')
    ax.set_title(f'{freq/1000:.1f} kHz')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = Path(__file__).parent / 'figures' / 'test_06_phase_velocity.png'
fig_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_path, dpi=150)
print(f"Saved figure: {fig_path}")
plt.show()
