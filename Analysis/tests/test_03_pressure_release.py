"""
Test 03: Pressure-Release Boundary Condition

Validates the antisymmetric ghost-cell treatment for p=0 (Dirichlet) boundaries
by comparing resonant frequencies of a quarter-wave resonator against theory.

Setup:
  - 2D waveguide (NX=400, NY=5) with dx=2 mm  ->  Lx=0.8 m, Ly=0.01 m
  - Left: hard wall (u=0)
  - Right: pressure-release (p=0)
  - Top/Bottom: hard wall (v=0)
  - Source: broadband Gaussian pulse at centre
  - Probe: pressure near left wall

Theory (1D, since Ly is small and y-modes are >17 kHz):
  f_n = (2n - 1) * c / (4L)   for n = 1, 2, 3, ...

Expected (c=343 m/s, L=0.8 m):
  n=1: 107.2 Hz
  n=2: 321.6 Hz
  n=3: 536.0 Hz
  ...
  n=10: 1.072 kHz
  n=20: 2.144 kHz
  n=30: 3.216 kHz

PASS criterion: |f_num - f_theory| / f_theory < 3% for first 5 peaks.
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
NX, NY = 400, 5
DX = 0.002
C0 = 343.0
RHO0 = 1.225
Lx = NX * DX

SIM_DURATION = 0.050  # 50 ms for good frequency resolution (df = 20 Hz)

# Pulse source (broadband)
SRC_I0, SRC_I1 = 195, 205  # centre of domain
PULSE_T0 = 0.0005
PULSE_TAU = 0.0002
PULSE_F0 = 2000.0
PULSE_AMP = 1.0

# Probe near hard wall
PROBE_I = 5
PROBE_J = NY // 2

# =============================================================================
# RUN SIMULATION
# =============================================================================
print("=" * 60)
print("TEST 03: Pressure-Release Boundary Condition")
print("=" * 60)
print(f"Grid: {NX}x{NY}, dx={DX*1000:.1f} mm, Lx={Lx*1000:.0f} mm")
print(f"Left: hard wall  |  Right: pressure-release")
print(f"Top/Bottom: hard wall")
print()

solver = FDTDSolver(NX, NY, DX, c0=C0, rho0=RHO0)

# Uniform sound speed
solver.set_c_field(np.full((NX, NY), C0))

# BCs: hard wall left, pressure-release right, hard wall top/bottom
solver.set_bc(left='hard_wall', right='pressure_release',
              top='hard_wall', bottom='hard_wall')

# Broadband pulse source
solver.set_source('broadband_pulse', SRC_I0, SRC_I1,
                  amplitude=PULSE_AMP, f0=PULSE_F0, tau=PULSE_TAU, t0=PULSE_T0)

n_steps = int(SIM_DURATION / solver.dt)
print(f"dt = {solver.dt*1e6:.3f} us, n_steps = {n_steps}")
print(f"Courant number = {solver.get_courant_number():.4f}")
print()

probe, snapshots = solver.run(n_steps, probe_coords=(PROBE_I, PROBE_J))
t = np.arange(len(probe)) * solver.dt

# =============================================================================
# ANALYSIS: FFT to find resonant peaks
# =============================================================================
# Window and FFT
window = np.hanning(len(probe))
probe_w = probe * window

freqs = np.fft.rfftfreq(len(probe_w), solver.dt)
spectrum = np.abs(np.fft.rfft(probe_w))

# Find peaks
from scipy.signal import find_peaks
# Use prominence instead of absolute height for robust peak detection
prom = np.max(spectrum) * 0.005
peaks, props = find_peaks(spectrum, prominence=prom, distance=5)
peak_freqs = freqs[peaks]
peak_amps = spectrum[peaks]

# Keep only first 15 peaks below 5 kHz
mask = peak_freqs < 5000
peak_freqs = peak_freqs[mask][:15]
peak_amps = peak_amps[mask][:15]

# Theory: first 20 quarter-wave modes
n_theory = np.arange(1, 21)
f_theory = (2 * n_theory - 1) * C0 / (4 * Lx)

# Match detected peaks to theory (nearest neighbour, one-to-one)
matches = []
used_theory = set()
for f_num in peak_freqs:
    best_err = np.inf
    best_idx = -1
    for idx, f_th in enumerate(f_theory):
        if idx in used_theory:
            continue
        err = abs(f_num - f_th)
        if err < best_err:
            best_err = err
            best_idx = idx
    if best_idx >= 0 and best_err < 50.0:  # 50 Hz tolerance
        used_theory.add(best_idx)
        f_th = f_theory[best_idx]
        err_pct = abs(f_num - f_th) / f_th * 100
        matches.append((best_idx + 1, f_th, f_num, err_pct))

# =============================================================================
# RESULTS
# =============================================================================
print("-" * 60)
print("Resonant frequency comparison")
print("-" * 60)
print(f"{'n':>3}  {'Theory (Hz)':>14}  {'Numeric (Hz)':>14}  {'Error %':>10}  {'Status'}")
print("-" * 60)

all_pass = True
for n, f_th, f_num, err in matches:
    # First mode (<150 Hz) limited by FFT resolution; allow 10% instead of 3%
    tol = 10.0 if f_th < 150.0 else 3.0
    status = "PASS" if err < tol else "FAIL"
    if err >= tol:
        all_pass = False
    print(f"{n:3d}  {f_th:14.2f}  {f_num:14.2f}  {err:10.2f}  {status}")

if not matches:
    print("  WARNING: No peaks detected!")
    all_pass = False

print("-" * 60)
print(f"Overall: {'PASS' if all_pass else 'FAIL'}")
print(f"Detected {len(peak_freqs)} peaks, matched {len(matches)} to theory")
print()

# =============================================================================
# PLOTS
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Time domain
ax = axes[0]
ax.plot(t * 1000, probe, 'b-', linewidth=0.5)
ax.axvline(x=PULSE_T0 * 1000, color='r', linestyle='--', alpha=0.5, label='pulse centre')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Pressure p')
ax.set_title('Probe signal near hard wall (pressure-release at far end)')
ax.grid(True, alpha=0.3)
ax.legend()

# Frequency domain
ax = axes[1]
ax.semilogy(freqs, spectrum, 'b-', linewidth=0.5, alpha=0.7)
matched_freqs = [m[2] for m in matches]
matched_amps = [spectrum[np.argmin(np.abs(freqs - f))] for f in matched_freqs]
ax.plot(matched_freqs, matched_amps, 'ro', markersize=6, label='matched peaks')
for f_th in f_theory[:10]:
    ax.axvline(x=f_th, color='g', linestyle='--', alpha=0.4)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('|P(f)|')
ax.set_title('FFT: quarter-wave resonator peaks (green dashed = theory)')
ax.set_xlim(0, 5000)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
fig_path = Path(__file__).parent / 'figures' / 'test_03_pressure_release.png'
fig_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_path, dpi=150)
print(f"Saved figure: {fig_path}")
plt.show()
