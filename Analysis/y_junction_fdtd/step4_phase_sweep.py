"""
Y-Junction Acoustic Logic Unit — 2D FDTD Step 4
================================================

Phase sweep demonstration.

For each phase difference φ between the two inlet sources:
    - Reset fields to zero
    - Run to steady state (1700 steps)
    - Measure outlet pressure over last 3 periods

Outputs:
    - Amplitude vs phase curve (3 metrics)
    - Side-by-side pressure fields for φ = 0° and 180°
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# Reuse Step 3 parameters and geometry
# ============================================================================
DX = 5e-5
C = 1500.0
FREQ = 1.0e6
LAMBDA = C / FREQ
CFL = 0.5
DT = CFL * DX / (C * np.sqrt(2))
coef = (C * DT / DX) ** 2

LAMBDA_CELLS = int(round(LAMBDA / DX))
W_CHANNEL = int(round(0.3 * LAMBDA_CELLS))
NX, NY = 250, 200
L_SECTION = NX // 3
X_BEND = L_SECTION
X_MERGE = 2 * L_SECTION
Y_MERGE = NY // 2
INLET_ANGLE_RAD = np.deg2rad(30)
Y_OFFSET = L_SECTION * np.tan(INLET_ANGLE_RAD)
Y_INLET_A = Y_MERGE + Y_OFFSET
Y_INLET_B = Y_MERGE - Y_OFFSET

SRC_AMP = 1.0
STEPS_PER_PERIOD = int(round(1.0 / FREQ / DT))
NT = 1700                      # ~20 periods to reach steady state
MEASURE_STEPS = 3 * STEPS_PER_PERIOD   # Last 3 periods for measurement

print(f"Steps per period = {STEPS_PER_PERIOD}")
print(f"Total steps per run = {NT}")
print(f"Measurement window = last {MEASURE_STEPS} steps ({MEASURE_STEPS/STEPS_PER_PERIOD:.1f} periods)")

# Sources and probe
srcA_x, srcA_y = 0, int(round(Y_INLET_A))
srcB_x, srcB_y = 0, int(round(Y_INLET_B))
PROBE_X = X_MERGE + LAMBDA_CELLS
PROBE_Y = Y_MERGE

# ============================================================================
# Geometry builder (copy from Step 3)
# ============================================================================

def point_segment_distance(px, py, x1, y1, x2, y2):
    dx_seg = x2 - x1
    dy_seg = y2 - y1
    if dx_seg == 0 and dy_seg == 0:
        return np.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx_seg + (py - y1) * dy_seg) /
                     (dx_seg**2 + dy_seg**2)))
    proj_x = x1 + t * dx_seg
    proj_y = y1 + t * dy_seg
    return np.hypot(px - proj_x, py - proj_y)


def build_y_junction_mask(nx, ny, x_bend, x_merge, y_merge, w, y_inlet_a, y_inlet_b):
    mask = np.ones((nx, ny), dtype=int)
    segments = [
        (0, y_inlet_a, x_bend, y_inlet_a),
        (x_bend, y_inlet_a, x_merge, y_merge),
        (0, y_inlet_b, x_bend, y_inlet_b),
        (x_bend, y_inlet_b, x_merge, y_merge),
        (x_merge, y_merge, nx - 1, y_merge),
    ]
    for i in range(nx):
        for j in range(ny):
            d_min = min(point_segment_distance(i, j, *seg) for seg in segments)
            if d_min < w / 2:
                mask[i, j] = 0
    return mask


bc_mask = build_y_junction_mask(NX, NY, X_BEND, X_MERGE, Y_MERGE,
                                W_CHANNEL, Y_INLET_A, Y_INLET_B)

# ============================================================================
# Laplacian with internal wall reflection (Fix 1)
# ============================================================================

def compute_laplacian(p, mask):
    left = np.zeros_like(p)
    left[1:, :] = p[:-1, :]
    wall_left = np.zeros_like(mask, dtype=bool)
    wall_left[1:, :] = (mask[:-1, :] == 1)
    left[wall_left] = p[wall_left]
    left[0, :] = np.where(mask[0, :] == 1, p[0, :], left[0, :])

    right = np.zeros_like(p)
    right[:-1, :] = p[1:, :]
    wall_right = np.zeros_like(mask, dtype=bool)
    wall_right[:-1, :] = (mask[1:, :] == 1)
    right[wall_right] = p[wall_right]
    right[-1, :] = np.where(mask[-1, :] == 1, p[-1, :], right[-1, :])

    bottom = np.zeros_like(p)
    bottom[:, 1:] = p[:, :-1]
    wall_bottom = np.zeros_like(mask, dtype=bool)
    wall_bottom[:, 1:] = (mask[:, :-1] == 1)
    bottom[wall_bottom] = p[wall_bottom]
    bottom[:, 0] = np.where(mask[:, 0] == 1, p[:, 0], bottom[:, 0])

    top = np.zeros_like(p)
    top[:, :-1] = p[:, 1:]
    wall_top = np.zeros_like(mask, dtype=bool)
    wall_top[:, :-1] = (mask[:, 1:] == 1)
    top[wall_top] = p[wall_top]
    top[:, -1] = np.where(mask[:, -1] == 1, p[:, -1], top[:, -1])

    return left + right + bottom + top - 4 * p


# ============================================================================
# Single FDTD run function
# ============================================================================

def run_fdtd(phase_b, nt, store_snapshot=False):
    """
    Run FDTD with given phase difference.
    Returns probe_trace and optionally the final pressure field.
    """
    p_prev = np.zeros((NX, NY))
    p_curr = np.zeros((NX, NY))
    p_next = np.zeros((NX, NY))
    probe_trace = []
    final_field = None

    for n in range(nt):
        t = n * DT
        p_curr[srcA_x, srcA_y] = SRC_AMP * np.sin(2 * np.pi * FREQ * t)
        p_curr[srcB_x, srcB_y] = SRC_AMP * np.sin(2 * np.pi * FREQ * t + phase_b)

        laplacian = compute_laplacian(p_curr, bc_mask)
        p_next[:] = 2 * p_curr - p_prev + coef * laplacian
        p_next[bc_mask == 1] = p_curr[bc_mask == 1]
        p_prev, p_curr, p_next = p_curr, p_next, p_prev

        probe_trace.append(p_curr[PROBE_X, PROBE_Y])

    if store_snapshot:
        final_field = p_curr.copy()

    return np.array(probe_trace), final_field


# ============================================================================
# Phase sweep
# ============================================================================
PHASE_DEGREES = np.arange(0, 361, 30)   # 0°, 30°, ..., 360°
PHASE_RADIANS = np.deg2rad(PHASE_DEGREES)

results_peak = []
results_rms = []
results_pkpk = []
snapshot_fields = {}   # Store fields for selected phases

print(f"\nRunning phase sweep: {len(PHASE_DEGREES)} cases")
print("=" * 50)

for phi_deg, phi_rad in zip(PHASE_DEGREES, PHASE_RADIANS):
    probe, field = run_fdtd(phi_rad, NT, store_snapshot=True)

    # Extract measurement window (last 3 periods)
    window = probe[-MEASURE_STEPS:]

    # Three metrics
    peak = np.max(np.abs(window))
    rms = np.sqrt(np.mean(window**2))
    pkpk = np.max(window) - np.min(window)

    results_peak.append(peak)
    results_rms.append(rms)
    results_pkpk.append(pkpk)

    # Store field for selected phases
    if phi_deg in [0, 90, 180, 270]:
        snapshot_fields[phi_deg] = field

    print(f"  φ = {phi_deg:3d}°  peak = {peak:.4f}  RMS = {rms:.4f}  Pk-Pk = {pkpk:.4f}")

print("=" * 50)

# Convert to arrays
results_peak = np.array(results_peak)
results_rms = np.array(results_rms)
results_pkpk = np.array(results_pkpk)

# ============================================================================
# Visualisation
# ============================================================================
fig_dir = Path(__file__).parent / "figures"
fig_dir.mkdir(exist_ok=True)

# --- Figure 1: Amplitude vs Phase ---
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(PHASE_DEGREES, results_peak, 'o-', label='Peak |p|', color='C0')
ax.plot(PHASE_DEGREES, results_rms, 's-', label='RMS p', color='C1')
ax.plot(PHASE_DEGREES, results_pkpk, '^-', label='Peak-to-peak', color='C2')

ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
ax.axvline(180, color='gray', linestyle='--', alpha=0.3)
ax.text(0, ax.get_ylim()[1]*0.95, 'Constructive\n(AND/OR)', ha='center', fontsize=9)
ax.text(180, ax.get_ylim()[1]*0.95, 'Destructive\n(XOR)', ha='center', fontsize=9)

ax.set_xlabel('Phase difference φ [degrees]')
ax.set_ylabel('Outlet pressure [Pa]')
ax.set_title('Logic Gate Performance: Outlet Pressure vs Input Phase')
ax.set_xticks(PHASE_DEGREES)
ax.legend()
ax.grid(True)
plt.tight_layout()
out_path = fig_dir / 'step4_phase_sweep.png'
plt.savefig(out_path, dpi=150)
print(f"\nSaved figure: {out_path}")
plt.close()

# --- Figure 2: Side-by-side pressure fields ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
vmax = max(np.abs(f).max() for f in snapshot_fields.values())

for ax, (phi_deg, field) in zip(axes.flatten(), snapshot_fields.items()):
    im = ax.imshow(field.T, origin='lower', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax,
                   extent=[0, NX*DX*1e3, 0, NY*DX*1e3])
    ax.contour(bc_mask.T, levels=[0.5], colors='k', linewidths=0.5,
               extent=[0, NX*DX*1e3, 0, NY*DX*1e3])
    ax.set_title(f'φ = {phi_deg}°')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    fig.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('Step 4: Pressure Field at Selected Phase Differences', fontsize=14)
plt.tight_layout()
out_path = fig_dir / 'step4_snapshots.png'
plt.savefig(out_path, dpi=150)
print(f"Saved figure: {out_path}")
plt.close()

# --- Figure 3: Logic contrast summary ---
logic_high = results_rms[0]      # φ = 0°
logic_low = results_rms[6]       # φ = 180°
contrast = (logic_high - logic_low) / (logic_high + logic_low)

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(['Logic "1"\n(φ = 0°)', 'Logic "0"\n(φ = 180°)'],
       [logic_high, logic_low],
       color=['C0', 'C3'])
ax.axhline((logic_high + logic_low)/2, color='k', linestyle='--',
           label=f'Threshold = {(logic_high+logic_low)/2:.4f} Pa')
ax.set_ylabel('RMS pressure [Pa]')
ax.set_title(f'Logic Contrast Ratio = {contrast:.3f}')
ax.legend()
plt.tight_layout()
out_path = fig_dir / 'step4_logic_contrast.png'
plt.savefig(out_path, dpi=150)
print(f"Saved figure: {out_path}")
plt.close()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 50)
print("STEP 4 COMPLETE")
print("=" * 50)
print(f"Logic HIGH (φ = 0°):   RMS = {logic_high:.4f} Pa")
print(f"Logic LOW  (φ = 180°): RMS = {logic_low:.4f} Pa")
print(f"Contrast ratio:        {contrast:.3f}")
print(f"Threshold:             {(logic_high + logic_low)/2:.4f} Pa")
print("=" * 50)
