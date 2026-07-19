#!/usr/bin/env python3
"""
================================================================================
FDTD POROUS V2 — Volume-Velocity Formulation (Scaffold / Learning Version)
================================================================================

This is a corrected scaffold for acoustic FDTD in a rigid-framed porous medium.

Key change from fdtd_porous.py:
    The primary velocity variable is now the macroscopic volume velocity
        V = phi * u
    where u is the pore-fluid particle velocity.  V is continuous across
    porosity jumps, so the divergence in the continuity equation is clean:

        dp/dt + rho0 * c0^2 * div(V) = 0

    The momentum equation becomes

        (rho0 * alpha_inf / phi) * dV/dt + sigma * V = -grad(p)

Why this fixes the interface problem:
    At a sharp boundary between two porous media, pressure and normal volume
    velocity must be continuous.  Because V lives at cell faces on the staggered
    grid, continuity of V_n is built in.  Using u instead would force continuity
    of pore velocity, which is wrong.

Scaffold contents:
    1. Grid and physical constants
    2. State variables and parameter fields
    3. Setup helpers
    4. Boundary conditions (hard wall, absorbing, pressure release)
    5. Time-stepping loop
    6. Energy diagnostic
    7. Two simple validation tests

TODO markers show where you can extend the code.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =============================================================================
# SECTION 1: GRID AND PHYSICAL CONSTANTS
# =============================================================================

NX, NY = 400, 50          # number of cells
DX = 0.002                # cell size [m]

RHO0 = 1.225              # fluid density [kg/m^3]
C0 = 343.0                # sound speed in free air [m/s]
K0 = RHO0 * C0 ** 2       # fluid bulk modulus [Pa]


# =============================================================================
# SECTION 2: STATE VARIABLES
# =============================================================================

# Porous-medium parameter fields (set by setup functions below)
phi = None                # porosity, 0 < phi <= 1
alpha_inf = None          # tortuosity, >= 1
sigma = None              # static flow resistivity [Pa*s/m^2]

# Coefficients used in the updates (recomputed when parameters change)
c_eff = None              # effective sound speed [m/s]
Z_V = None                # characteristic impedance p/V for volume velocity
a_coeff = None            # phi / (rho0 * alpha_inf)  [m^3/kg]
b_coeff = None            # sigma * phi * dt / (rho0 * alpha_inf)  [dimensionless]
dt = None

# Acoustic fields
p = None                  # pressure at cell centres, shape (NX, NY)
Vx = None                 # x-volume-velocity at x-faces, shape (NX+1, NY)
Vy = None                 # y-volume-velocity at y-faces, shape (NX, NY+1)


# =============================================================================
# SECTION 3: SETUP HELPERS
# =============================================================================

def _update_derived():
    """Recompute all derived quantities and a stable time step."""
    global c_eff, Z_V, a_coeff, b_coeff, dt

    # Effective sound speed for the ZK rigid-frame model
    c_eff = C0 / np.sqrt(phi * alpha_inf)

    # Characteristic impedance using the volume-velocity definition V = phi*u.
    # Derivation: for a plane wave p = Z_u * u with Z_u = rho0*c0*sqrt(alpha/phi),
    # and V = phi*u, so Z_V = p/V = Z_u / phi = rho0*c0*sqrt(alpha) / phi^(3/2).
    Z_V = RHO0 * C0 * np.sqrt(alpha_inf) / (phi ** 1.5)

    # CFL condition: dt <= DX / (c_max * sqrt(2)) in 2D
    c_max = float(c_eff.max())
    dt = 0.9 * DX / (c_max * np.sqrt(2))

    # Momentum-equation coefficients for the current dt.
    # Semi-implicit update:
    #   V_new = (V_old - dt * a_coeff * grad(p)) / (1 + b_coeff)
    a_coeff = phi / (RHO0 * alpha_inf)
    b_coeff = sigma * phi * dt / (RHO0 * alpha_inf)

    print(f"  [setup] c_max = {c_max:.1f} m/s, dt = {dt*1e6:.3f} µs")


def set_uniform_params(phi_val, alpha_val, sigma_val):
    """Set the same porous parameters everywhere."""
    global phi, alpha_inf, sigma
    phi = np.full((NX, NY), float(phi_val))
    alpha_inf = np.full((NX, NY), float(alpha_val))
    sigma = np.full((NX, NY), float(sigma_val))
    _update_derived()


def set_free_air():
    """Free air: phi = 1, alpha_inf = 1, sigma = 0."""
    set_uniform_params(1.0, 1.0, 0.0)


def apply_region(mask, phi_val, alpha_val, sigma_val):
    """Apply porous parameters to a masked sub-region."""
    global phi, alpha_inf, sigma
    phi = phi.copy()
    alpha_inf = alpha_inf.copy()
    sigma = sigma.copy()
    phi[mask] = float(phi_val)
    alpha_inf[mask] = float(alpha_val)
    sigma[mask] = float(sigma_val)
    _update_derived()


def reset_fields():
    """Zero all acoustic fields."""
    global p, Vx, Vy
    p = np.zeros((NX, NY))
    Vx = np.zeros((NX + 1, NY))
    Vy = np.zeros((NX, NY + 1))


# =============================================================================
# SECTION 4: BOUNDARY CONDITIONS
# =============================================================================

# Default: waveguide with hard walls top/bottom and absorbing ends
BC_LEFT = 'absorbing'
BC_RIGHT = 'absorbing'
BC_BOTTOM = 'hard_wall'
BC_TOP = 'hard_wall'


def apply_boundary_conditions():
    """Apply BCs to the volume-velocity fields."""
    global Vx, Vy

    # Left face (Vx[0, :])
    if BC_LEFT == 'absorbing':
        # Outgoing wave travels left (-x): p = -Z_V * Vx  =>  Vx = -p / Z_V
        Vx[0, :] = -p[0, :] / Z_V[0, :]
    elif BC_LEFT == 'hard_wall':
        Vx[0, :] = 0.0
    elif BC_LEFT == 'pressure_release':
        # p = 0 at the boundary; ghost-cell pressure is -p[0], so the gradient
        # across the boundary face is 2*p[0]/DX.  The coefficient a_coeff
        # converts pressure gradient into volume-velocity acceleration.
        Vx[0, :] -= dt * a_coeff[0, :] * 2.0 * p[0, :] / DX

    # Right face (Vx[NX, :])
    if BC_RIGHT == 'absorbing':
        # Outgoing wave travels right (+x): p = +Z_V * Vx  =>  Vx = +p / Z_V
        Vx[NX, :] = p[NX - 1, :] / Z_V[NX - 1, :]
    elif BC_RIGHT == 'hard_wall':
        Vx[NX, :] = 0.0
    elif BC_RIGHT == 'pressure_release':
        Vx[NX, :] += dt * a_coeff[NX - 1, :] * 2.0 * p[NX - 1, :] / DX

    # Bottom face (Vy[:, 0])
    if BC_BOTTOM == 'absorbing':
        # Outgoing wave travels down (-y): p = -Z_V * Vy  =>  Vy = -p / Z_V
        Vy[:, 0] = -p[:, 0] / Z_V[:, 0]
    elif BC_BOTTOM == 'hard_wall':
        Vy[:, 0] = 0.0
    elif BC_BOTTOM == 'pressure_release':
        Vy[:, 0] -= dt * a_coeff[:, 0] * 2.0 * p[:, 0] / DX

    # Top face (Vy[:, NY])
    if BC_TOP == 'absorbing':
        # Outgoing wave travels up (+y): p = +Z_V * Vy  =>  Vy = +p / Z_V
        Vy[:, NY] = p[:, NY - 1] / Z_V[:, NY - 1]
    elif BC_TOP == 'hard_wall':
        Vy[:, NY] = 0.0
    elif BC_TOP == 'pressure_release':
        Vy[:, NY] += dt * a_coeff[:, NY - 1] * 2.0 * p[:, NY - 1] / DX


# =============================================================================
# SECTION 5: SOURCE
# =============================================================================

SOURCE_TYPE = 'continuous_sine'     # 'continuous_sine' | 'pulse' | 'none'
SRC_AMP = 1.0                       # pressure amplitude [Pa]
SRC_FREQ = 5000.0                   # Hz
SRC_I0, SRC_I1 = 20, 23             # source strip in x, moved away from left BC

PULSE_T0 = 0.0005
PULSE_TAU = 0.0001
PULSE_F0 = 6000.0


def source_value(t):
    """Pressure source amplitude at time t."""
    if SOURCE_TYPE == 'none':
        return 0.0
    elif SOURCE_TYPE == 'continuous_sine':
        return SRC_AMP * np.sin(2.0 * np.pi * SRC_FREQ * t)
    elif SOURCE_TYPE == 'pulse':
        env = np.exp(-((t - PULSE_T0) / PULSE_TAU) ** 2)
        return SRC_AMP * env * np.sin(2.0 * np.pi * PULSE_F0 * (t - PULSE_T0))
    return 0.0


# =============================================================================
# SECTION 6: TIME STEPPING
# =============================================================================

def step(t):
    """One leapfrog step with the volume-velocity formulation."""
    global p

    # --- Update volume velocities ---
    # Momentum: (rho0*alpha/phi) dV/dt + sigma*V = -grad(p)
    # Semi-implicit: damping evaluated at new time, pressure at old time.
    dpdx = (p[1:NX, :] - p[0:NX - 1, :]) / DX
    dpdy = (p[:, 1:NY] - p[:, 0:NY - 1]) / DX

    Vx[1:NX, :] = (
        Vx[1:NX, :]
        - dt * a_coeff[1:NX, :] * dpdx
    ) / (1.0 + b_coeff[1:NX, :])

    Vy[:, 1:NY] = (
        Vy[:, 1:NY]
        - dt * a_coeff[:, 1:NY] * dpdy
    ) / (1.0 + b_coeff[:, 1:NY])

    # Apply boundary conditions
    apply_boundary_conditions()

    # --- Update pressure ---
    # Continuity: dp/dt + rho0*c0^2 * div(V) = 0
    dVx_dx = (Vx[1:NX + 1, :] - Vx[0:NX, :]) / DX
    dVy_dy = (Vy[:, 1:NY + 1] - Vy[:, 0:NY]) / DX
    p -= dt * K0 * (dVx_dx + dVy_dy)

    # --- Inject source ---
    p[SRC_I0:SRC_I1, :] += source_value(t)


# =============================================================================
# SECTION 7: RUNNING AND DIAGNOSTICS
# =============================================================================

def run(duration, probe_coords=None):
    """Run for `duration` seconds. Optionally record a single probe."""
    n_steps = int(duration / dt)
    probe = [] if probe_coords is not None else None
    for n in range(n_steps):
        step(n * dt)
        if probe_coords is not None:
            probe.append(float(p[probe_coords]))
    return np.array(probe) if probe is not None else None


def compute_energy():
    """Total acoustic energy using the volume-velocity formulation."""
    # Potential energy per bulk volume: p^2 / (2 * K0)
    e_pot = np.sum(p ** 2 / (2.0 * K0)) * DX ** 2

    # Kinetic energy per bulk volume: 0.5 * (rho0*alpha/phi) * |V|^2
    Vx_avg = 0.5 * (Vx[0:NX, :] ** 2 + Vx[1:NX + 1, :] ** 2)
    Vy_avg = 0.5 * (Vy[:, 0:NY] ** 2 + Vy[:, 1:NY + 1] ** 2)
    rho_macro = RHO0 * alpha_inf / phi
    e_kin = 0.5 * np.sum(rho_macro * (Vx_avg + Vy_avg)) * DX ** 2

    return e_kin, e_pot, e_kin + e_pot


def rms_probe(probe, window_seconds):
    """RMS amplitude over the last `window_seconds` of a probe trace."""
    n = int(window_seconds / dt)
    if n <= 0 or len(probe) == 0:
        return 0.0
    tail = probe[-n:]
    return np.sqrt(np.mean(tail ** 2))


# =============================================================================
# SECTION 8: PLOTTING HELPER
# =============================================================================

FIG_DIR = Path(__file__).parent / 'figures'
FIG_DIR.mkdir(exist_ok=True)


def plot_fields(title, filename):
    """Plot pressure field and porosity field side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    extent = [0, NX * DX * 1000, 0, NY * DX * 1000]

    im0 = axes[0].imshow(phi.T, origin='lower', cmap='viridis_r',
                         vmin=0.3, vmax=1.0, extent=extent)
    axes[0].set_title('Porosity φ')
    axes[0].set_xlabel('x [mm]')
    axes[0].set_ylabel('y [mm]')
    fig.colorbar(im0, ax=axes[0])

    vmax = np.abs(p).max()
    im1 = axes[1].imshow(p.T, origin='lower', cmap='RdBu_r',
                         vmin=-vmax, vmax=vmax, extent=extent)
    axes[1].set_title(title)
    axes[1].set_xlabel('x [mm]')
    axes[1].set_ylabel('y [mm]')
    fig.colorbar(im1, ax=axes[1], label='p [Pa]')

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  Saved figure: {filename}")


# =============================================================================
# SECTION 9: VALIDATION TESTS
# =============================================================================

def test_free_air_baseline():
    """TEST V2-01: free-air pulse transit time."""
    print("\nTEST V2-01: Free-air baseline")
    set_free_air()
    reset_fields()

    global SOURCE_TYPE, SRC_FREQ
    SOURCE_TYPE = 'pulse'
    SRC_FREQ = PULSE_F0

    probe_i = NX - 4
    probe = run(0.020, probe_coords=(probe_i, NY // 2))
    t = np.arange(len(probe)) * dt

    t_peak = t[np.argmax(np.abs(probe))]
    src_centre = 0.5 * (SRC_I0 + SRC_I1)
    t_theory = (probe_i - src_centre) * DX / C0 + PULSE_T0
    err = abs(t_peak - t_theory) / t_theory * 100

    print(f"  Measured transit: {t_peak*1000:.3f} ms")
    print(f"  Theoretical:      {t_theory*1000:.3f} ms")
    print(f"  Error:            {err:.2f}%  ->  {'PASS' if err < 3.0 else 'FAIL'}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t * 1000, probe, 'b-', lw=0.8)
    ax.axvline(t_theory * 1000, color='g', ls='--', label='theory')
    ax.axvline(t_peak * 1000, color='r', ls='--', label='measured')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('p [Pa]')
    ax.set_title('V2-01: Free-air pulse transit')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / 'v2_test_01_free_air.png', dpi=150)
    plt.close(fig)


def test_porosity_step_reflection():
    """
    TEST V2-02: 1D reflection/transmission at a single porosity step.

    Analytic pressure reflection coefficient for a step from medium 1 to 2:

        R = (Z_V2 - Z_V1) / (Z_V2 + Z_V1)

    where Z_V = rho0*c0*sqrt(alpha) / phi^(3/2).

    Domain is made narrow in y to approximate 1D propagation.
    """
    print("\nTEST V2-02: Porosity-step reflection/transmission")

    # TODO(student): try different phi1/phi2 values and check the analytic R
    phi1, alpha1 = 1.0, 1.0
    phi2, alpha2 = 0.5, 1.0
    sigma_val = 0.0

    set_uniform_params(phi1, alpha1, sigma_val)
    mask = np.zeros((NX, NY), dtype=bool)
    mask[NX // 2:, :] = True
    apply_region(mask, phi2, alpha2, sigma_val)
    reset_fields()

    global SOURCE_TYPE, SRC_FREQ
    SOURCE_TYPE = 'continuous_sine'
    SRC_FREQ = 5000.0

    # Source on the left side of the step
    global SRC_I0, SRC_I1
    SRC_I0, SRC_I1 = 20, 23

    global BC_BOTTOM, BC_TOP, BC_LEFT, BC_RIGHT
    BC_BOTTOM = 'hard_wall'
    BC_TOP = 'hard_wall'
    BC_LEFT = 'absorbing'
    BC_RIGHT = 'absorbing'

    # Run until steady state
    run(0.015)

    # Measure amplitudes just left and right of the step
    refl_probe = p[NX // 2 - 20, NY // 2]
    trans_probe = p[NX // 2 + 20, NY // 2]

    # Analytic reflection coefficient
    Z1 = RHO0 * C0 * np.sqrt(alpha1) / (phi1 ** 1.5)
    Z2 = RHO0 * C0 * np.sqrt(alpha2) / (phi2 ** 1.5)
    R_analytic = (Z2 - Z1) / (Z2 + Z1)

    print(f"  Z1 = {Z1:.3f}, Z2 = {Z2:.3f}")
    print(f"  Analytic |R| = {abs(R_analytic):.3f}")
    print(f"  Measured amplitudes: reflected {abs(refl_probe):.4f}, "
          f"transmitted {abs(trans_probe):.4f}")
    print("  TODO: extract a clean |R| from the standing-wave envelope "
          "and compare to the analytic value.")

    plot_fields('V2-02: Porosity step pressure field',
                FIG_DIR / 'v2_test_02_step.png')


# =============================================================================
# SECTION 10: MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Volume-velocity porous FDTD scaffold")
    print("=" * 60)

    test_free_air_baseline()
    test_porosity_step_reflection()

    print("\nDone. Inspect the figures and the TODO markers to extend the solver.")
