"""
Y-Junction Acoustic Logic Unit — 2D FDTD Prototype
====================================================

Step 1: Empty-domain wave propagation with boundary mask system.

Implements the 2D scalar acoustic wave equation:
    ∂²p/∂t² = c² (∂²p/∂x² + ∂²p/∂y²)

using second-order central differences in space and time (leapfrog).
Boundary conditions are handled via a bc_mask array for flexibility.

Boundary mask values:
    0 = interior         (full 5-point stencil)
    1 = Neumann (rigid)  (∂p/∂n = 0, ghost cell = boundary cell)
    2 = Dirichlet (free) (p = 0, ghost cell = -boundary cell)  [placeholder]
    3 = absorbing        (outgoing wave, no reflection)          [placeholder]

Physical parameters (water-like, scaled for fast prototyping):
    - Domain: 100 x 100 grid points
    - Grid spacing dx = 100 µm
    - Wave speed c = 1500 m/s (water)
    - Time step dt set by CFL = 0.5

Author: MSc IRP 2025-26
Date: 2026-05-22
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# Physical and numerical parameters
# ============================================================================
NX, NY = 100, 100           # Grid size
DX = 1e-4                   # Grid spacing [m] = 100 µm
C = 1500.0                  # Wave speed [m/s] (water)
CFL = 0.5                   # CFL safety factor (< 1/√2 for stability)
DT = CFL * DX / (C * np.sqrt(2))  # Time step [s]

# Stability check
dt_max = DX / (C * np.sqrt(2))
assert DT <= dt_max, f"CFL violation: DT={DT:.3e} > dt_max={dt_max:.3e}"
print(f"dx = {DX:.3e} m,  dt = {DT:.3e} s,  c = {C} m/s")
print(f"CFL = {C*DT/DX:.4f}  (stability limit = {1/np.sqrt(2):.4f})")

# ============================================================================
# Boundary condition mask
# ============================================================================
# bc_mask[i, j] encodes the boundary condition type at each cell.
# For now: all domain edges are Neumann (rigid wall), everything else interior.

bc_mask = np.zeros((NX, NY), dtype=int)   # 0 = interior by default
bc_mask[0, :] = 1                          # left edge   → Neumann
bc_mask[-1, :] = 1                         # right edge  → Neumann
bc_mask[:, 0] = 1                          # bottom edge → Neumann
bc_mask[:, -1] = 1                         # top edge    → Neumann

# Corners are already covered by the edge assignments above.
# In future steps, internal walls (Y-junction geometry) will also set mask=1.

print(f"Boundary cells: {np.sum(bc_mask == 1)}  (Neumann)")
print(f"Interior cells: {np.sum(bc_mask == 0)}")

# ============================================================================
# Laplacian with boundary-aware stencil
# ============================================================================

def compute_laplacian(p, mask):
    """
    Compute the spatial Laplacian ∇²p with boundary-aware stencils.

    For interior cells (mask=0): standard 5-point stencil
        ∇²p ≈ p[i-1,j] + p[i+1,j] + p[i,j-1] + p[i,j+1] - 4·p[i,j]

    For Neumann boundary cells (mask=1):
        The ghost cell outside the domain equals the boundary cell itself
        (∂p/∂n = 0  →  p_ghost = p_boundary).
        This modifies the coefficient on the diagonal.
        - Edge cell:    ∇²p ≈ ... - 3·p[i,j]
        - Corner cell:  ∇²p ≈ ... - 2·p[i,j]

    Parameters
    ----------
    p : ndarray (NX, NY)
        Pressure field at current time step.
    mask : ndarray (NX, NY), int
        Boundary condition mask (0=interior, 1=Neumann, 2=Dirichlet, ...).

    Returns
    -------
    lap : ndarray (NX, NY)
        Laplacian of p.
    """
    nx, ny = p.shape
    lap = np.zeros_like(p)

    # ------------------------------------------------------------------
    # i-1 (left) contribution
    # ------------------------------------------------------------------
    left = np.zeros_like(p)
    left[1:, :] = p[:-1, :]                       # interior neighbors
    # Boundary: ghost cell value depends on BC type
    # Neumann (mask=1):  p_ghost = p[0, :]       (reflect)
    left[0, :] = np.where(mask[0, :] == 1, p[0, :], left[0, :])

    # ------------------------------------------------------------------
    # i+1 (right) contribution
    # ------------------------------------------------------------------
    right = np.zeros_like(p)
    right[:-1, :] = p[1:, :]                      # interior neighbors
    right[-1, :] = np.where(mask[-1, :] == 1, p[-1, :], right[-1, :])

    # ------------------------------------------------------------------
    # j-1 (bottom) contribution
    # ------------------------------------------------------------------
    bottom = np.zeros_like(p)
    bottom[:, 1:] = p[:, :-1]                     # interior neighbors
    bottom[:, 0] = np.where(mask[:, 0] == 1, p[:, 0], bottom[:, 0])

    # ------------------------------------------------------------------
    # j+1 (top) contribution
    # ------------------------------------------------------------------
    top = np.zeros_like(p)
    top[:, :-1] = p[:, 1:]                        # interior neighbors
    top[:, -1] = np.where(mask[:, -1] == 1, p[:, -1], top[:, -1])

    # ------------------------------------------------------------------
    # Assemble Laplacian
    # ------------------------------------------------------------------
    lap = left + right + bottom + top - 4 * p

    # NOTE: For Dirichlet (mask=2), ghost = -p_boundary, which would change
    # the coefficient to -5 at edges and -6 at corners. Add handling here
    # when Dirichlet regions are needed.

    return lap

# ============================================================================
# Source parameters (Gaussian pulse)
# ============================================================================
SRC_X, SRC_Y = NX // 2, NY // 2   # Source at domain centre
PULSE_WIDTH = 20 * DT             # Gaussian width [s]
PULSE_DELAY = 50 * DT             # Time delay before peak [s]
SRC_AMP = 1.0                     # Peak amplitude [Pa]

def source_pulse(t):
    """Gaussian pressure pulse."""
    return SRC_AMP * np.exp(-((t - PULSE_DELAY) / PULSE_WIDTH) ** 2)

# ============================================================================
# FDTD update coefficient
# ============================================================================
# p^{n+1} = 2*p^n - p^{n-1} + (c*dt/dx)^2 * Laplacian(p^n)
coef = (C * DT / DX) ** 2

# ============================================================================
# Field arrays
# ============================================================================
p_prev = np.zeros((NX, NY))   # p^{n-1}
p_curr = np.zeros((NX, NY))   # p^n
p_next = np.zeros((NX, NY))   # p^{n+1}

# ============================================================================
# Time stepping
# ============================================================================
NT = 400                      # Total time steps
snapshot_interval = 50        # Save a figure every N steps
snapshots = []
times = []

print(f"\nRunning {NT} time steps...")
for n in range(NT):
    t = n * DT

    # Inject source (hard source: overwrite the cell)
    p_curr[SRC_X, SRC_Y] += source_pulse(t)

    # Compute Laplacian with boundary-aware stencil
    laplacian = compute_laplacian(p_curr, bc_mask)

    # Time update (leapfrog)
    p_next[:] = 2 * p_curr - p_prev + coef * laplacian

    # Swap arrays for next step
    p_prev, p_curr, p_next = p_curr, p_next, p_prev

    # Save snapshot
    if n % snapshot_interval == 0:
        snapshots.append(p_curr.copy())
        times.append(t)

    # Progress
    if n % 100 == 0:
        print(f"  Step {n:4d}/{NT}  t = {t*1e6:.2f} µs  max |p| = {np.abs(p_curr).max():.4f}")

print("Done.\n")

# ============================================================================
# Visualisation
# ============================================================================
fig_dir = Path(__file__).parent / "figures"
fig_dir.mkdir(exist_ok=True)

n_snaps = len(snapshots)
fig, axes = plt.subplots(2, (n_snaps + 1) // 2, figsize=(14, 6))
axes = axes.flatten()

vmax = max(np.abs(s).max() for s in snapshots)

for i, (snap, t) in enumerate(zip(snapshots, times)):
    ax = axes[i]
    im = ax.imshow(snap.T, origin='lower', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, extent=[0, NX*DX*1e3, 0, NY*DX*1e3])
    ax.set_title(f't = {t*1e6:.1f} µs')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    fig.colorbar(im, ax=ax, fraction=0.046)

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.suptitle('Step 1: Empty-domain wave propagation (Neumann BCs)', fontsize=14)
plt.tight_layout()
out_path = fig_dir / 'step1_empty_domain.png'
plt.savefig(out_path, dpi=150)
print(f"Saved figure: {out_path}")
plt.close()

# ============================================================================
# Debug: Verify 1/√r decay in 2D
# ============================================================================
final = snapshots[-1]
r = np.arange(1, min(NX, NY) // 2)
p_radial = []
for ri in r:
    ix = SRC_X + ri
    iy = SRC_Y + ri
    if ix < NX and iy < NY:
        p_radial.append(np.abs(final[ix, iy]))
    else:
        break

r = r[:len(p_radial)]
p_radial = np.array(p_radial)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(r * DX * 1e3, p_radial, 'b-', label='|p(r)|')
if len(p_radial) > 1:
    theory = p_radial[0] * np.sqrt(r[0]) / np.sqrt(r)
    ax.plot(r * DX * 1e3, theory, 'r--', label=r'$1/\sqrt{r}$ theory')
ax.set_xlabel('r [mm]')
ax.set_ylabel('|p| [Pa]')
ax.set_title('Amplitude decay verification (2D)')
ax.legend()
ax.grid(True)
plt.tight_layout()
out_path = fig_dir / 'step1_decay_check.png'
plt.savefig(out_path, dpi=150)
print(f"Saved figure: {out_path}")
plt.close()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*60)
print("STEP 1 COMPLETE")
print("="*60)
print(f"  Grid:           {NX} x {NY}")
print(f"  dx:             {DX*1e6:.1f} µm")
print(f"  dt:             {DT*1e9:.3f} ns")
print(f"  CFL:            {CFL:.3f}  (< {1/np.sqrt(2):.3f} ✓)")
print(f"  Boundary:       Neumann (∂p/∂n = 0) on all edges")
print(f"  Boundary cells: {np.sum(bc_mask == 1)}")
print(f"  Interior cells: {np.sum(bc_mask == 0)}")
print(f"  Figures saved:  {fig_dir}/")
print("="*60)
