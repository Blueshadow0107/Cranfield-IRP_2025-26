"""
Test 04: Eigenfrequencies of Closed Cavity

Validates resonant modes by comparing FFT peaks against theoretical
eigenfrequencies of a hard-walled rectangular cavity.

Setup:
  - All boundaries: hard wall
  - Initial condition: Gaussian pressure pulse at centre
  - No source during run
  - Probe: pressure at cell (5, NY//2)

Theory (2D hard-walled cavity):
  f_{n,m} = (c/2) * sqrt( (n/Lx)^2 + (m/Ly)^2 )
  where n, m = 0, 1, 2, ...  (but n=m=0 is excluded)

For Lx=0.8 m, Ly=0.1 m, c=343 m/s:
  n=1,m=0: 214.4 Hz
  n=2,m=0: 428.8 Hz
  n=0,m=1: 1715.0 Hz
  n=3,m=0: 643.1 Hz
  ...

Note: A centred Gaussian excites even modes strongly, odd modes weakly
(due to symmetry).

PASS criterion: |f_num - f_theory| / f_theory < 3% for first 5 detectable peaks.
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
Lx = NX * DX
Ly = NY * DX

SIM_DURATION = 0.050

# Initial Gaussian pulse (centred)
IC_AMP = 1.0
IC_SIGMA = 5  # cells (10 mm)
IC_I = NX // 2
IC_J = NY // 2

PROBE_I = 5
PROBE_J = NY // 2

# =============================================================================
# SETUP & RUN
# =============================================================================
print("=" * 60)
print("TEST 04: Eigenfrequencies of Closed Cavity")
print("=" * 60)
print(f"Cavity: {Lx*1000:.0f} mm x {Ly*1000:.0f} mm")
print(f"All boundaries: hard wall")
print()

solver = FDTDSolver(NX, NY, DX, c0=C0, rho0=RHO0)
solver.set_c_field(np.full((NX, NY), C0))
solver.set_bc(left='hard_wall', right='hard_wall',
              top='hard_wall', bottom='hard_wall')

# Initial condition: Gaussian pressure pulse
x_idx = np.arange(NX)
y_idx = np.arange(NY)
X, Y = np.meshgrid(x_idx, y_idx, indexing='ij')
solver.p = IC_AMP * np.exp(-((X - IC_I)**2 + (Y - IC_J)**2) / (2 * IC_SIGMA**2))

n_steps = int(SIM_DURATION / solver.dt)
print(f"dt = {solver.dt*1e6:.3f} us, n_steps = {n_steps}")
print()

probe, _ = solver.run(n_steps, probe_coords=(PROBE_I, PROBE_J))

# =============================================================================
# ANALYSIS
# =============================================================================
window = np.hanning(len(probe))
probe_w = probe * window

freqs = np.fft.rfftfreq(len(probe_w), solver.dt)
spectrum = np.abs(np.fft.rfft(probe_w))

# Find peaks
from scipy.signal import find_peaks
peaks, _ = find_peaks(spectrum, height=np.max(spectrum)*0.005, distance=30)
peak_freqs = freqs[peaks]
peak_amps = spectrum[peaks]

# Keep first 10 peaks below 4 kHz
mask = peak_freqs < 4000
peak_freqs = peak_freqs[mask][:10]
peak_amps = peak_amps[mask][:10]

# Theory: compute all modes up to 4 kHz
modes = []
for n in range(0, 20):
    for m in range(0, 20):
        if n == 0 and m == 0:
            continue
        f_th = 0.5 * C0 * np.sqrt((n / Lx)**2 + (m / Ly)**2)
        if f_th < 4000:
            modes.append((f_th, n, m))

modes.sort()
f_theory = np.array([m[0] for m in modes])

# Match peaks to theory
matches = []
for f_num in peak_freqs:
    idx = np.argmin(np.abs(f_theory - f_num))
    f_th = f_theory[idx]
    err = abs(f_num - f_th) / f_th * 100
    matches.append((f_num, f_th, err))

# =============================================================================
# RESULTS
# =============================================================================
print("-" * 60)
print("Peak frequency comparison")
print("-" * 60)
print(f"{'Numeric (Hz)':>14}  {'Theory (Hz)':>14}  {'Error %':>10}  {'Status'}")
print("-" * 60)

all_pass = True
for f_num, f_th, err in matches[:8]:
    status = "PASS" if err < 3.0 else "FAIL"
    if err >= 3.0:
        all_pass = False
    print(f"{f_num:14.2f}  {f_th:14.2f}  {err:10.2f}  {status}")

print("-" * 60)
print(f"Overall: {'PASS' if all_pass else 'FAIL'}")
print()

# =============================================================================
# PLOT
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

ax = axes[0]
ax.plot(np.arange(len(probe)) * solver.dt * 1000, probe, 'b-', linewidth=0.5)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Pressure p')
ax.set_title('Ring-down signal (closed cavity)')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.semilogy(freqs, spectrum, 'b-', linewidth=0.5, alpha=0.7)
ax.plot(peak_freqs, peak_amps, 'ro', markersize=5, label='detected peaks')
for f_th, n, m in modes[:15]:
    ax.axvline(x=f_th, color='g', linestyle='--', alpha=0.3)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('|P(f)|')
ax.set_title('FFT: cavity eigenfrequencies (green dashed = theory)')
ax.set_xlim(0, 4000)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
fig_path = Path(__file__).parent / 'figures' / 'test_04_eigenfrequencies.png'
fig_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_path, dpi=150)
print(f"Saved figure: {fig_path}")
plt.show()
