"""
Step 2: Absorbing Boundary Conditions (sponge layer)
======================================================

Add a damping layer near domain boundaries to absorb outgoing waves.
Test by injecting a pulse and checking reflection amplitude.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Parameters (same as Step 1)
# ---------------------------------------------------------------------------
NX, NY = 100, 100
DX = 1e-4
C = 1500.0
CFL = 0.5
DT = CFL * DX / (C * np.sqrt(2))
coef = (C * DT / DX) ** 2

# Sponge layer parameters
SPONGE_WIDTH = 15            # Cells wide
SPONGE_DAMP = 0.3            # Damping coefficient per step

def build_sponge_mask(nx, ny, width, damp):
    """
    Build a damping mask that is 0 in the interior and increases
    linearly to `damp` at the boundaries.
    """
    mask = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            # Distance to nearest boundary
            d = min(i, j, nx - 1 - i, ny - 1 - j)
            if d < width:
                mask[i, j] = damp * (1 - d / width) ** 2
    return mask

sponge = build_sponge_mask(NX, NY, SPONGE_WIDTH, SPONGE_DAMP)

# ---------------------------------------------------------------------------
# Source (Gaussian pulse)
# ---------------------------------------------------------------------------
SRC_X, SRC_Y = NX // 2, NY // 2
PULSE_WIDTH = 20 * DT
PULSE_DELAY = 50 * DT
SRC_AMP = 1.0

def source_pulse(t):
    return SRC_AMP * np.exp(-((t - PULSE_DELAY) / PULSE_WIDTH) ** 2)

# ---------------------------------------------------------------------------
# Run with and without sponge
# ---------------------------------------------------------------------------
NT = 600

# Probe near boundary (where reflections would arrive)
PROBE_X, PROBE_Y = 5, NY // 2

def run_simulation(use_sponge):
    p_prev = np.zeros((NX, NY))
    p_curr = np.zeros((NX, NY))
    p_next = np.zeros((NX, NY))
    probe_trace = []
    max_p = []

    for n in range(NT):
        t = n * DT
        p_curr[SRC_X, SRC_Y] += source_pulse(t)

        laplacian = (
            np.roll(p_curr, 1, axis=0) + np.roll(p_curr, -1, axis=0) +
            np.roll(p_curr, 1, axis=1) + np.roll(p_curr, -1, axis=1) -
            4 * p_curr
        )

        p_next[:] = 2 * p_curr - p_prev + coef * laplacian

        if use_sponge:
            # Apply sponge damping to velocity-like term
            p_next -= sponge * (p_next - p_prev)

        p_prev, p_curr, p_next = p_curr, p_next, p_prev
        probe_trace.append(p_curr[PROBE_X, PROBE_Y])
        max_p.append(np.abs(p_curr).max())

    return np.array(probe_trace), np.array(max_p)

print("Running without sponge...")
trace_no_sponge, max_no = run_simulation(use_sponge=False)
print("Running with sponge...")
trace_sponge, max_sp = run_simulation(use_sponge=True)

# ---------------------------------------------------------------------------
# Analysis: quantify reflection
# ---------------------------------------------------------------------------
t = np.arange(NT) * DT * 1e6  # µs

# Direct arrival time from source to probe
dist = np.sqrt((SRC_X - PROBE_X)**2 + (SRC_Y - PROBE_Y)**2) * DX
arrival = dist / C * 1e6       # µs
# Reflection arrival time (source → far wall → probe)
refl_dist = ((NX - 1 - PROBE_X) + (NX - 1 - SRC_X)) * DX
refl_arrival = refl_dist / C * 1e6 + arrival

print(f"\nDirect arrival at probe: ~{arrival:.2f} µs")
print(f"Reflected arrival at probe: ~{refl_arrival:.2f} µs")

# Find peak before and after reflection
before_mask = t < refl_arrival - 1
after_mask = t > refl_arrival + 1

peak_before_no = np.max(np.abs(trace_no_sponge[before_mask])) if np.any(before_mask) else 0
peak_after_no = np.max(np.abs(trace_no_sponge[after_mask])) if np.any(after_mask) else 0
peak_before_sp = np.max(np.abs(trace_sponge[before_mask])) if np.any(before_mask) else 0
peak_after_sp = np.max(np.abs(trace_sponge[after_mask])) if np.any(after_mask) else 0

refl_ratio_no = peak_after_no / peak_before_no if peak_before_no > 0 else 0
refl_ratio_sp = peak_after_sp / peak_before_sp if peak_before_sp > 0 else 0

print(f"\nWithout sponge: reflected peak / direct peak = {refl_ratio_no:.3f}")
print(f"With sponge:    reflected peak / direct peak = {refl_ratio_sp:.3f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig_dir = Path(__file__).parent / "figures"
fig_dir.mkdir(exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Probe traces
ax = axes[0, 0]
ax.plot(t, trace_no_sponge, 'b-', alpha=0.7, label='No sponge')
ax.plot(t, trace_sponge, 'r-', alpha=0.7, label='With sponge')
ax.axvline(refl_arrival, color='k', linestyle='--', alpha=0.5, label='Expected reflection')
ax.set_xlabel('t [µs]')
ax.set_ylabel('p [Pa]')
ax.set_title(f'Probe at boundary (x={PROBE_X*DX*1e3:.1f} mm)')
ax.legend()
ax.grid(True)

# Zoomed probe traces
ax = axes[0, 1]
ax.plot(t, trace_no_sponge, 'b-', alpha=0.7, label='No sponge')
ax.plot(t, trace_sponge, 'r-', alpha=0.7, label='With sponge')
ax.axvline(refl_arrival, color='k', linestyle='--', alpha=0.5)
ax.set_xlim(refl_arrival - 2, t[-1])
ax.set_xlabel('t [µs]')
ax.set_ylabel('p [Pa]')
ax.set_title('Zoom: reflection region')
ax.legend()
ax.grid(True)

# Sponge mask
ax = axes[1, 0]
im = ax.imshow(sponge.T, origin='lower', cmap='YlOrRd',
               extent=[0, NX*DX*1e3, 0, NY*DX*1e3])
ax.set_title('Sponge damping mask')
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
fig.colorbar(im, ax=ax, fraction=0.046)

# Max amplitude over time
ax = axes[1, 1]
ax.plot(t, max_no, 'b-', alpha=0.7, label='No sponge')
ax.plot(t, max_sp, 'r-', alpha=0.7, label='With sponge')
ax.set_xlabel('t [µs]')
ax.set_ylabel('max |p| [Pa]')
ax.set_title('Global max amplitude')
ax.legend()
ax.grid(True)

plt.suptitle('Step 2: Absorbing Boundary Conditions', fontsize=14)
plt.tight_layout()
out_path = fig_dir / 'step2_abc_test.png'
plt.savefig(out_path, dpi=150)
print(f"\nSaved figure: {out_path}")
plt.close()

print("\n--- Step 2 Complete ---")
print("Checks:")
print(f"  - Reflection without sponge: {refl_ratio_no:.3f}  (should be significant)")
print(f"  - Reflection with sponge:    {refl_ratio_sp:.3f}  (should be < 0.05)")
if refl_ratio_sp < 0.05:
    print("  - Sponge effective: ✓")
else:
    print("  - Sponge may need tuning (increase SPONGE_DAMP or SPONGE_WIDTH)")
