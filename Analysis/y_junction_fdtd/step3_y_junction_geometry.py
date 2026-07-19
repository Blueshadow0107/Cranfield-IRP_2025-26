"""
Y-Junction Acoustic Logic Unit — 2D FDTD Step 3
================================================

Three-section geometry:
    1. Two straight horizontal inlets from left edge
    2. Angled merge region (±30°)
    3. Straight outlet to right edge

All three sections have equal horizontal extent (~83 cells).

Fixes applied:
    - Internal walls reflect in Laplacian (ghost-cell Neumann)
    - Wall cells frozen after each time step
    - Hard source overwrite (no accumulation)

Grid: dx = 50 µm, c = 1500 m/s, f = 1 MHz → λ = 1.5 mm = 30 cells
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# Physical and numerical parameters
# ============================================================================
DX = 5e-5                    # Grid spacing [m] = 50 µm
C = 1500.0                   # Wave speed [m/s] (water)
FREQ = 1.0e6                 # Source frequency [Hz]
LAMBDA = C / FREQ            # Wavelength [m]

CFL = 0.5                    # CFL safety factor
DT = CFL * DX / (C * np.sqrt(2))
coef = (C * DT / DX) ** 2

# Stability check
dt_max = DX / (C * np.sqrt(2))
assert DT <= dt_max, f"CFL violation: DT={DT:.3e} > dt_max={dt_max:.3e}"

# ============================================================================
# Geometry parameters (in cells)
# ============================================================================
LAMBDA_CELLS = int(round(LAMBDA / DX))   # Should be 30
W_CHANNEL = int(round(0.3 * LAMBDA_CELLS))   # 0.3λ ≈ 9 cells

# Domain size
NX = 250                     # 12.5 mm
NY = 200                     # 10 mm

# Three sections with equal horizontal extent
L_SECTION = NX // 3          # ~83 cells each
X_BEND = L_SECTION           # Where horizontal inlets become angled
X_MERGE = 2 * L_SECTION      # Where angled inlets meet / outlet begins
Y_MERGE = NY // 2            # Domain centre vertically

# Angled section geometry
INLET_ANGLE_DEG = 30
INLET_ANGLE_RAD = np.deg2rad(INLET_ANGLE_DEG)
L_ANGLED = L_SECTION / np.cos(INLET_ANGLE_RAD)  # Actual segment length
Y_OFFSET = L_SECTION * np.tan(INLET_ANGLE_RAD)   # Vertical drop/rise

# Inlet centerline y-positions
Y_INLET_A = Y_MERGE + Y_OFFSET
Y_INLET_B = Y_MERGE - Y_OFFSET

print("=" * 60)
print("STEP 3: Y-Junction Geometry Setup")
print("=" * 60)
print(f"dx = {DX*1e6:.1f} µm,  dt = {DT*1e9:.3f} ns,  c = {C} m/s")
print(f"Frequency = {FREQ/1e6:.2f} MHz,  λ = {LAMBDA*1e3:.2f} mm = {LAMBDA_CELLS} cells")
print(f"CFL = {CFL:.3f}  (stability limit = {1/np.sqrt(2):.3f})")
print(f"Domain: {NX} x {NY} = {NX*DX*1e3:.2f} mm x {NY*DX*1e3:.2f} mm")
print(f"Channel width = 0.3λ = {W_CHANNEL} cells = {W_CHANNEL*DX*1e6:.0f} µm")
print(f"Section length (horizontal) = {L_SECTION} cells = {L_SECTION*DX*1e3:.2f} mm")
print(f"Merge point = ({X_MERGE}, {Y_MERGE})")
print(f"Inlet A center y = {Y_INLET_A:.1f}")
print(f"Inlet B center y = {Y_INLET_B:.1f}")
print("=" * 60)

# ============================================================================
# Geometry builder: distance-to-segment
# ============================================================================

def point_segment_distance(px, py, x1, y1, x2, y2):
    """Euclidean distance from point (px, py) to segment (x1,y1)-(x2,y2)."""
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
    """
    Build bc_mask for three-section Y-junction.
    Returns mask where 0 = interior, 1 = wall.
    """
    mask = np.ones((nx, ny), dtype=int)

    # 5 segments: 2 horizontal inlets, 2 angled inlets, 1 outlet
    segments = [
        # Inlet A horizontal
        (0, y_inlet_a, x_bend, y_inlet_a),
        # Inlet A angled
        (x_bend, y_inlet_a, x_merge, y_merge),
        # Inlet B horizontal
        (0, y_inlet_b, x_bend, y_inlet_b),
        # Inlet B angled
        (x_bend, y_inlet_b, x_merge, y_merge),
        # Outlet
        (x_merge, y_merge, nx - 1, y_merge),
    ]

    for i in range(nx):
        for j in range(ny):
            d_min = min(point_segment_distance(i, j, *seg) for seg in segments)
            if d_min < w / 2:
                mask[i, j] = 0

    return mask


# Build geometry
bc_mask = build_y_junction_mask(NX, NY, X_BEND, X_MERGE, Y_MERGE,
                                W_CHANNEL, Y_INLET_A, Y_INLET_B)

print(f"Interior cells: {np.sum(bc_mask == 0)}")
print(f"Wall cells:     {np.sum(bc_mask == 1)}")

# ============================================================================
# Visualise geometry
# ============================================================================
fig_dir = Path(__file__).parent / "figures"
fig_dir.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(bc_mask.T, origin='lower', cmap='gray_r',
               extent=[0, NX*DX*1e3, 0, NY*DX*1e3],
               vmin=0, vmax=1)
ax.plot(X_MERGE*DX*1e3, Y_MERGE*DX*1e3, 'ro', markersize=8, label='Merge point')
ax.axvline(X_BEND*DX*1e3, color='g', linestyle='--', alpha=0.5, label='Bend start')
ax.axvline(X_MERGE*DX*1e3, color='b', linestyle='--', alpha=0.5, label='Merge start')
ax.set_title('Y-Junction Geometry (Step 3)')
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
ax.legend()
plt.tight_layout()
out_path = fig_dir / 'step3_geometry.png'
plt.savefig(out_path, dpi=150)
print(f"Saved geometry figure: {out_path}")
plt.close()

# ============================================================================
# FDTD field arrays and source parameters
# ============================================================================
p_prev = np.zeros((NX, NY))
p_curr = np.zeros((NX, NY))
p_next = np.zeros((NX, NY))

SRC_AMP = 1.0
PHASE_A = 0.0
PHASE_B = 0.0

# Sources at left edge of horizontal inlets
srcA_x, srcA_y = 0, int(round(Y_INLET_A))
srcB_x, srcB_y = 0, int(round(Y_INLET_B))

# Probe in outlet, 1λ downstream of merge
PROBE_X = X_MERGE + LAMBDA_CELLS
PROBE_Y = Y_MERGE

print(f"Source A at ({srcA_x}, {srcA_y})")
print(f"Source B at ({srcB_x}, {srcB_y})")
print(f"Probe at ({PROBE_X}, {PROBE_Y})")

# ============================================================================
# Laplacian with internal wall reflection (Fix 1)
# ============================================================================

def compute_laplacian(p, mask):
    """
    Compute ∇²p with Neumann reflection at internal and external walls.
    """
    # LEFT neighbor (i-1)
    left = np.zeros_like(p)
    left[1:, :] = p[:-1, :]
    wall_left = np.zeros_like(mask, dtype=bool)
    wall_left[1:, :] = (mask[:-1, :] == 1)
    left[wall_left] = p[wall_left]
    left[0, :] = np.where(mask[0, :] == 1, p[0, :], left[0, :])

    # RIGHT neighbor (i+1)
    right = np.zeros_like(p)
    right[:-1, :] = p[1:, :]
    wall_right = np.zeros_like(mask, dtype=bool)
    wall_right[:-1, :] = (mask[1:, :] == 1)
    right[wall_right] = p[wall_right]
    right[-1, :] = np.where(mask[-1, :] == 1, p[-1, :], right[-1, :])

    # BOTTOM neighbor (j-1)
    bottom = np.zeros_like(p)
    bottom[:, 1:] = p[:, :-1]
    wall_bottom = np.zeros_like(mask, dtype=bool)
    wall_bottom[:, 1:] = (mask[:, :-1] == 1)
    bottom[wall_bottom] = p[wall_bottom]
    bottom[:, 0] = np.where(mask[:, 0] == 1, p[:, 0], bottom[:, 0])

    # TOP neighbor (j+1)
    top = np.zeros_like(p)
    top[:, :-1] = p[:, 1:]
    wall_top = np.zeros_like(mask, dtype=bool)
    wall_top[:, :-1] = (mask[:, 1:] == 1)
    top[wall_top] = p[wall_top]
    top[:, -1] = np.where(mask[:, -1] == 1, p[:, -1], top[:, -1])

    return left + right + bottom + top - 4 * p


# ============================================================================
# Time stepping
# ============================================================================
NT = 800
T_PERIOD = 1.0 / FREQ
STEPS_PER_PERIOD = int(round(T_PERIOD / DT))

print(f"Steps per period = {STEPS_PER_PERIOD}")

snapshot_interval = STEPS_PER_PERIOD
snapshots = []
times = []
probe_trace = []

print(f"\nRunning {NT} time steps...")
for n in range(NT):
    t = n * DT

    # Inject hard sinusoidal sources (Fix 3: overwrite, not accumulate)
    p_curr[srcA_x, srcA_y] = SRC_AMP * np.sin(2 * np.pi * FREQ * t + PHASE_A)
    p_curr[srcB_x, srcB_y] = SRC_AMP * np.sin(2 * np.pi * FREQ * t + PHASE_B)

    # Compute Laplacian
    laplacian = compute_laplacian(p_curr, bc_mask)

    # Time update (leapfrog)
    p_next[:] = 2 * p_curr - p_prev + coef * laplacian

    # Freeze wall cells (Fix 2)
    p_next[bc_mask == 1] = p_curr[bc_mask == 1]

    # Swap arrays
    p_prev, p_curr, p_next = p_curr, p_next, p_prev

    # Record probe
    probe_trace.append(p_curr[PROBE_X, PROBE_Y])

    # Save snapshot
    if n % snapshot_interval == 0:
        snapshots.append(p_curr.copy())
        times.append(t)

    if n % (2 * STEPS_PER_PERIOD) == 0:
        print(f"  Step {n:4d}/{NT}  t = {t*1e6:.2f} µs  "
              f"max |p| = {np.abs(p_curr).max():.4f}")

print("Done.\n")

# ============================================================================
# Visualisation
# ============================================================================
n_snaps = len(snapshots)
ncols = 5
nrows = (n_snaps + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
if nrows == 1:
    axes = axes.reshape(1, -1)
axes = axes.flatten()

vmax = max(np.abs(s).max() for s in snapshots) if snapshots else 1.0

for i, (snap, t) in enumerate(zip(snapshots, times)):
    ax = axes[i]
    im = ax.imshow(snap.T, origin='lower', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax,
                   extent=[0, NX*DX*1e3, 0, NY*DX*1e3])
    ax.contour(bc_mask.T, levels=[0.5], colors='k', linewidths=0.5,
               extent=[0, NX*DX*1e3, 0, NY*DX*1e3])
    ax.set_title(f't = {t*1e6:.1f} µs ({t/T_PERIOD:.1f} T)')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    fig.colorbar(im, ax=ax, fraction=0.046)

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.suptitle('Step 3: Y-Junction Wave Propagation (φ = 0°)', fontsize=14)
plt.tight_layout()
out_path = fig_dir / 'step3_y_junction.png'
plt.savefig(out_path, dpi=150)
print(f"Saved figure: {out_path}")
plt.close()

# Probe trace
fig, ax = plt.subplots(figsize=(10, 4))
t_probe = np.arange(len(probe_trace)) * DT * 1e6
ax.plot(t_probe, probe_trace, 'b-')
ax.set_xlabel('t [µs]')
ax.set_ylabel('p [Pa]')
ax.set_title(f'Probe at outlet ({PROBE_X*DX*1e3:.2f} mm, {PROBE_Y*DX*1e3:.2f} mm)')
ax.grid(True)
plt.tight_layout()
out_path = fig_dir / 'step3_probe.png'
plt.savefig(out_path, dpi=150)
print(f"Saved figure: {out_path}")
plt.close()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("STEP 3 COMPLETE")
print("=" * 60)
print(f"  Grid:           {NX} x {NY}")
print(f"  dx:             {DX*1e6:.1f} µm")
print(f"  dt:             {DT*1e9:.3f} ns")
print(f"  Frequency:      {FREQ/1e6:.2f} MHz")
print(f"  Wavelength:     {LAMBDA*1e3:.2f} mm = {LAMBDA_CELLS} cells")
print(f"  Channel width:  {W_CHANNEL} cells = {W_CHANNEL*DX*1e6:.0f} µm")
print(f"  Section length: {L_SECTION} cells = {L_SECTION*DX*1e3:.2f} mm")
print(f"  Merge angle:    ±{INLET_ANGLE_DEG}° (60° total)")
print(f"  Phase A:        {PHASE_A:.2f} rad")
print(f"  Phase B:        {PHASE_B:.2f} rad")
print(f"  Figures saved:  {fig_dir}/")
print("=" * 60)
