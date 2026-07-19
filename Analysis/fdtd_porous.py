"""
================================================================================
FDTD POROUS — First-Order Acoustic FDTD with Zwikker-Kosten Porous Media
================================================================================

This solver extends the modular first-order acoustic FDTD to rigid-framed
porous media using the Zwikker-Kosten (ZK) equivalent-fluid model.

Governing time-domain equations (ZK):

    rho0 * alpha_inf * du/dt + sigma * phi * u = -grad(p)     (momentum)
    dp/dt + (rho0 * c0^2 / phi) * div(u) = 0                   (continuity)

where:
    phi       — porosity            (0 < phi <= 1)
    alpha_inf — tortuosity          (alpha_inf >= 1)
    sigma     — static flow resistivity [Pa·s/m²]

Effective sound speed:

    c_eff = c0 / sqrt(phi * alpha_inf)

Effective impedance:

    Z_eff = rho_eff_real * c_eff = rho0 * c0 * sqrt(alpha_inf / phi)

Order of operations:
    1. set_porous_params(...) or set_c_field(...) + set_porous_region(...)
    2. Configure BCs and source
    3. reset_state()
    4. run_simulation(...)

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks


# =============================================================================
# SECTION 1: PHYSICAL CONSTANTS
# =============================================================================

# Grid
NX, NY = 400, 50
DX = 0.002                      # m

# Air at 300 K
RHO0 = 1.225                    # kg/m³
C0 = 343.0                      # m/s


# =============================================================================
# SECTION 2: STATE VARIABLES (initialised by setup functions and reset_state)
# =============================================================================

# Sound speed field and derived quantities
c_field = None
c2 = None
c_max = None
dt = None

# Porous-medium parameters (default = free air)
phi = None
alpha_inf = None
sigma = None
rho_eff_real = None          # rho0 * alpha_inf
K_eff = None                 # rho0 * c0^2 / phi
Z_eff = None                 # rho_eff_real * c_eff
damping_factor = None        # 1 / (1 + sigma*phi*dt / rho_eff_real)

# Acoustic fields
p = None
u = None
v = None

# Boundary conditions (mutable — tests change these)
BC_LEFT = 'absorbing'
BC_RIGHT = 'absorbing'
BC_TOP = 'hard_wall'
BC_BOTTOM = 'hard_wall'

# Source configuration (mutable)
SOURCE_TYPE = 'continuous_sine'
SRC_I0, SRC_I1 = 3, 6
SRC_AMP = 1.0
SRC_FREQ = 5000.0
PULSE_T0 = 0.0005
PULSE_TAU = 0.0001
PULSE_F0 = 6000.0


# =============================================================================
# SECTION 3: SOLVER SETUP
# =============================================================================

def _update_porous_derived():
    """Recompute derived porous quantities, sound speed, and dt."""
    global c_field, c2, c_max, dt
    global rho_eff_real, K_eff, Z_eff, damping_factor

    rho_eff_real = RHO0 * alpha_inf
    K_eff = RHO0 * C0 ** 2 / phi
    c_field = C0 / np.sqrt(phi * alpha_inf)
    c2 = c_field ** 2
    Z_eff = rho_eff_real * c_field
    c_max = float(c_field.max())
    dt = 0.9 * DX / (c_max * np.sqrt(2))
    damping_factor = 1.0 / (1.0 + sigma * phi * dt / rho_eff_real)


def set_c_field(new_c_field):
    """
    Update sound speed field directly and reset porous parameters to free air.
    Use this for non-porous inhomogeneous media or for baseline comparisons.
    """
    global phi, alpha_inf, sigma
    c_field_local = new_c_field.astype(float)
    phi = np.ones_like(c_field_local)
    alpha_inf = np.ones_like(c_field_local)
    sigma = np.zeros_like(c_field_local)
    _update_porous_derived()
    # Override derived c_field with user-supplied values
    global c_field, c2, c_max, Z_eff
    c_field = c_field_local
    c2 = c_field ** 2
    Z_eff = RHO0 * c_field
    c_max = float(c_field.max())
    dt = 0.9 * DX / (c_max * np.sqrt(2))
    damping_factor = 1.0 / (1.0 + sigma * phi * dt / rho_eff_real)
    print(f"  [set_c_field] c_max = {c_max:.1f} m/s, dt = {dt*1e6:.3f} µs")


def set_porous_params(phi_in, alpha_inf_in, sigma_in):
    """
    Set Zwikker-Kosten porous parameters over the whole domain.

    Parameters
    ----------
    phi_in : array_like, shape (NX, NY)
        Porosity, 0 < phi <= 1.
    alpha_inf_in : array_like, shape (NX, NY)
        Tortuosity, alpha_inf >= 1.
    sigma_in : array_like, shape (NX, NY)
        Static flow resistivity [Pa·s/m²], sigma >= 0.
    """
    global phi, alpha_inf, sigma
    phi = phi_in.astype(float)
    alpha_inf = alpha_inf_in.astype(float)
    sigma = sigma_in.astype(float)
    _update_porous_derived()
    print(f"  [set_porous_params] c_max = {c_max:.1f} m/s, dt = {dt*1e6:.3f} µs")


def set_porous_region(mask, phi_val, alpha_inf_val, sigma_val):
    """
    Apply porous parameters to a sub-region defined by a boolean mask.
    Outside the mask, parameters remain unchanged.

    Parameters
    ----------
    mask : ndarray of bool, shape (NX, NY)
        True where porous parameters should be applied.
    phi_val, alpha_inf_val, sigma_val : float
        Porous parameter values for the masked region.
    """
    global phi, alpha_inf, sigma
    phi = phi.copy()
    alpha_inf = alpha_inf.copy()
    sigma = sigma.copy()
    phi[mask] = float(phi_val)
    alpha_inf[mask] = float(alpha_inf_val)
    sigma[mask] = float(sigma_val)
    _update_porous_derived()
    print(f"  [set_porous_region] c_max = {c_max:.1f} m/s, dt = {dt*1e6:.3f} µs")


def reset_state():
    """Zero all acoustic fields. Call between tests."""
    global p, u, v
    p = np.zeros((NX, NY))
    u = np.zeros((NX + 1, NY))
    v = np.zeros((NX, NY + 1))


# =============================================================================
# SECTION 4: BOUNDARY CONDITIONS
# =============================================================================

def apply_boundary_conditions():
    """Apply BCs to velocity fields using effective impedance Z_eff."""
    global u, v

    # Left face
    if BC_LEFT == 'absorbing':
        u[0, :] = -p[0, :] / Z_eff[0, :]
    elif BC_LEFT == 'hard_wall':
        u[0, :] = 0.0
    elif BC_LEFT == 'pressure_release':
        u[0, :] -= (dt / RHO0) * 2.0 * p[0, :] / DX

    # Right face
    if BC_RIGHT == 'absorbing':
        u[NX, :] = p[NX - 1, :] / Z_eff[NX - 1, :]
    elif BC_RIGHT == 'hard_wall':
        u[NX, :] = 0.0
    elif BC_RIGHT == 'pressure_release':
        u[NX, :] += (dt / RHO0) * 2.0 * p[NX - 1, :] / DX

    # Bottom face
    if BC_BOTTOM == 'absorbing':
        v[:, 0] = -p[:, 0] / Z_eff[:, 0]
    elif BC_BOTTOM == 'hard_wall':
        v[:, 0] = 0.0
    elif BC_BOTTOM == 'pressure_release':
        v[:, 0] -= (dt / RHO0) * 2.0 * p[:, 0] / DX

    # Top face
    if BC_TOP == 'absorbing':
        v[:, NY] = p[:, NY - 1] / Z_eff[:, NY - 1]
    elif BC_TOP == 'hard_wall':
        v[:, NY] = 0.0
    elif BC_TOP == 'pressure_release':
        v[:, NY] += (dt / RHO0) * 2.0 * p[:, NY - 1] / DX


# =============================================================================
# SECTION 5: SOURCE
# =============================================================================

PULSE_TYPE = 'gaussian'  # 'gaussian' | 'broadband'


def source_amplitude(t):
    """Compute source amplitude at time t."""
    if SOURCE_TYPE == 'none':
        return 0.0
    elif SOURCE_TYPE == 'continuous_sine':
        return SRC_AMP * np.sin(2.0 * np.pi * SRC_FREQ * t)
    elif SOURCE_TYPE == 'pulse':
        if PULSE_TYPE == 'gaussian':
            tau = 0.0001
        elif PULSE_TYPE == 'broadband':
            tau = 0.0005
        env = np.exp(-((t - PULSE_T0) / tau) ** 2)
        return SRC_AMP * env * np.sin(2.0 * np.pi * SRC_FREQ * (t - PULSE_T0))
    return 0.0


# =============================================================================
# SECTION 6: TIME STEPPING (Zwikker-Kosten)
# =============================================================================

def step(t):
    """One leapfrog step with ZK porous terms."""
    global p

    # Update interior velocities with implicit damping
    dpdx = (p[1:NX, :] - p[0:NX - 1, :]) / DX
    dpdy = (p[:, 1:NY] - p[:, 0:NY - 1]) / DX

    u[1:NX, :] = damping_factor[1:NX, :] * (
        u[1:NX, :] - (dt / rho_eff_real[1:NX, :]) * dpdx
    )
    v[:, 1:NY] = damping_factor[:, 1:NY] * (
        v[:, 1:NY] - (dt / rho_eff_real[:, 1:NY]) * dpdy
    )

    # Apply BCs
    apply_boundary_conditions()

    # Update pressure using effective bulk modulus
    du_dx = (u[1:NX + 1, :] - u[0:NX, :]) / DX
    dv_dy = (v[:, 1:NY + 1] - v[:, 0:NY]) / DX
    p -= dt * K_eff * (du_dx + dv_dy)

    # Source injection
    p[SRC_I0:SRC_I1, :] += source_amplitude(t)


# =============================================================================
# SECTION 7: REUSABLE UTILITIES
# =============================================================================

def run_simulation(duration, probe_coords=None):
    """Run FDTD loop for `duration` seconds. Returns probe history."""
    n_steps = int(duration / dt)
    probe = []
    for n in range(n_steps):
        step(n * dt)
        if probe_coords is not None:
            probe.append(float(p[probe_coords[0], probe_coords[1]]))
    return np.array(probe)


def run_test(bcs, duration, probe_coords, src_type='none', src_freq=5000.0):
    """Configure BCs + source, run, return probe. Caller must reset first."""
    global BC_LEFT, BC_RIGHT, BC_TOP, BC_BOTTOM, SOURCE_TYPE, SRC_FREQ
    BC_LEFT, BC_RIGHT, BC_TOP, BC_BOTTOM = bcs
    SOURCE_TYPE, SRC_FREQ = src_type, src_freq
    return run_simulation(duration, probe_coords)


def compute_energy():
    """Compute total acoustic energy using effective density and bulk modulus."""
    e_pot = np.sum(p ** 2 / (2.0 * K_eff)) * DX ** 2
    u_avg = 0.5 * (u[0:NX, :] ** 2 + u[1:NX + 1, :] ** 2)
    v_avg = 0.5 * (v[:, 0:NY] ** 2 + v[:, 1:NY + 1] ** 2)
    e_kin = 0.5 * np.sum(rho_eff_real * (u_avg + v_avg)) * DX ** 2
    return e_kin, e_pot, e_kin + e_pot


def compute_fft(probe):
    """Compute windowed FFT. Returns (freqs, spectrum)."""
    window = np.hanning(len(probe))
    freqs = np.fft.rfftfreq(len(probe), dt)
    spectrum = np.abs(np.fft.rfft(probe * window))
    return freqs, spectrum


def find_fft_peaks(probe, height_ratio=0.005, distance=5):
    """FFT + find peak frequencies. Returns (peak_freqs, peak_amps)."""
    freqs, spectrum = compute_fft(probe)
    prom = np.max(spectrum) * height_ratio
    peaks, _ = find_peaks(spectrum, prominence=prom, distance=distance)
    return freqs[peaks], spectrum[peaks]


# =============================================================================
# SECTION 8: PLOTTING
# =============================================================================

FIG_DIR = Path(__file__).parent / 'figures'
FIG_DIR.mkdir(exist_ok=True)


def plot_probe(t, probe, title, filename):
    """Standard figure: time-domain + FFT. Saves and closes."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    axes[0].plot(t * 1000, probe, 'b-', lw=0.5)
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Pressure p')
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    freqs, spectrum = compute_fft(probe)
    axes[1].semilogy(freqs / 1000, spectrum, 'b-', lw=0.5)
    axes[1].set_xlabel('Frequency (kHz)')
    axes[1].set_ylabel('|P(f)|')
    axes[1].set_title('FFT')
    axes[1].set_xlim(0, 12)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


# =============================================================================
# SECTION 9: ZK VALIDATION TESTS
# =============================================================================

def test_free_air_baseline():
    """TEST ZK-01: Confirm free-air behaviour matches fdtd_modular.py."""
    print("\nZK-TEST 01: Free-Air Baseline")

    set_porous_params(
        np.ones((NX, NY)),
        np.ones((NX, NY)),
        np.zeros((NX, NY))
    )
    reset_state()

    global PULSE_TYPE, PULSE_T0
    PULSE_TYPE = 'gaussian'
    PULSE_T0 = 0.0005

    probe = run_test(
        ('absorbing', 'absorbing', 'hard_wall', 'hard_wall'),
        0.020, (NX - 4, NY // 2), src_type='pulse'
    )
    t = np.arange(len(probe)) * dt
    t_meas = t[np.argmax(np.abs(probe))]
    t_theory = (NX - 4 - 4.5) * DX / C0 + PULSE_T0
    err = abs(t_meas - t_theory) / t_theory * 100
    status = "PASS" if err < 3.0 else "FAIL"
    print(f"  Transit error: {err:.2f}%  ->  {status}")

    plot_probe(t, probe, 'ZK-01: Free-air baseline', FIG_DIR / 'zk_test_01.png')
    return status


def test_slowdown_only():
    """TEST ZK-02: Wave slowdown without damping (sigma = 0)."""
    print("\nZK-TEST 02: Slowdown Only")

    phi_val = 0.5
    alpha_val = 2.5
    c_eff_expected = C0 / np.sqrt(phi_val * alpha_val)

    set_porous_params(
        np.ones((NX, NY)) * phi_val,
        np.ones((NX, NY)) * alpha_val,
        np.zeros((NX, NY))
    )
    reset_state()

    global PULSE_TYPE, PULSE_T0
    PULSE_TYPE = 'gaussian'
    PULSE_T0 = 0.0005

    probe = run_test(
        ('absorbing', 'absorbing', 'hard_wall', 'hard_wall'),
        0.030, (NX - 4, NY // 2), src_type='pulse'
    )
    t = np.arange(len(probe)) * dt
    t_meas = t[np.argmax(np.abs(probe))]
    t_theory = (NX - 4 - 4.5) * DX / c_eff_expected + PULSE_T0
    err = abs(t_meas - t_theory) / t_theory * 100
    status = "PASS" if err < 5.0 else "FAIL"
    print(f"  c_eff expected = {c_eff_expected:.1f} m/s")
    print(f"  Transit error: {err:.2f}%  ->  {status}")

    plot_probe(t, probe, f'ZK-02: Slowdown (phi={phi_val}, alpha={alpha_val})',
               FIG_DIR / 'zk_test_02.png')
    return status


def test_damping_only():
    """TEST ZK-03: Exponential decay in a damped uniform medium."""
    print("\nZK-TEST 03: Damping Only")

    sigma_val = 50000.0  # Pa·s/m²
    set_porous_params(
        np.ones((NX, NY)),
        np.ones((NX, NY)),
        np.ones((NX, NY)) * sigma_val
    )
    reset_state()

    global PULSE_TYPE, PULSE_T0, SRC_FREQ
    PULSE_TYPE = 'gaussian'
    PULSE_T0 = 0.0005
    SRC_FREQ = 5000.0

    probe = run_test(
        ('absorbing', 'absorbing', 'hard_wall', 'hard_wall'),
        0.020, (NX - 4, NY // 2), src_type='pulse'
    )
    t = np.arange(len(probe)) * dt

    # Expected decay rate for low-frequency ZK plane wave
    gamma = sigma_val / (2.0 * RHO0 * C0)
    expected_decay = np.exp(-gamma * C0 * (t - PULSE_T0))
    expected_decay[t < PULSE_T0] = 1.0

    # Compare peak envelope roughly
    peaks, _ = find_peaks(np.abs(probe), height=np.max(np.abs(probe)) * 0.05, distance=10)
    if len(peaks) >= 2:
        measured_decays = np.abs(probe[peaks])
        predicted_decays = np.exp(-gamma * C0 * (t[peaks] - PULSE_T0))
        ratios = measured_decays / (predicted_decays + 1e-30)
        err = np.std(ratios) / np.mean(ratios) * 100
    else:
        err = 0.0

    status = "PASS" if err < 30.0 else "FAIL"
    print(f"  Decay rate gamma = {gamma:.1f} 1/s")
    print(f"  Envelope std/mean: {err:.1f}%  ->  {status}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t * 1000, np.abs(probe), 'b-', lw=0.8, label='|p|')
    ax.plot(t * 1000, np.max(np.abs(probe)) * expected_decay / np.max(expected_decay),
            'r--', lw=1.0, label='Expected decay shape')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('|p|')
    ax.set_title('ZK-03: Damping only')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / 'zk_test_03.png', dpi=150)
    plt.close(fig)
    return status


def test_porous_slab():
    """TEST ZK-04: Pulse through a porous slab in free air."""
    print("\nZK-TEST 04: Porous Slab")

    # Free-air background
    phi_bg = np.ones((NX, NY))
    alpha_bg = np.ones((NX, NY))
    sigma_bg = np.zeros((NX, NY))
    set_porous_params(phi_bg, alpha_bg, sigma_bg)

    # Porous slab in the centre (choose phi*alpha > 1 so c_eff < c0)
    x_start, x_end = NX // 2 - 30, NX // 2 + 30
    slab_mask = np.zeros((NX, NY), dtype=bool)
    slab_mask[x_start:x_end, :] = True
    phi_slab = 0.6
    alpha_slab = 2.0
    sigma_slab = 30000.0
    set_porous_region(slab_mask, phi_slab, alpha_slab, sigma_slab)

    c_eff_slab = C0 / np.sqrt(phi_slab * alpha_slab)
    print(f"  Slab c_eff = {c_eff_slab:.1f} m/s")

    reset_state()

    global PULSE_TYPE, PULSE_T0
    PULSE_TYPE = 'gaussian'
    PULSE_T0 = 0.0005

    probe = run_test(
        ('absorbing', 'absorbing', 'hard_wall', 'hard_wall'),
        0.035, (NX - 4, NY // 2), src_type='pulse'
    )
    t_slab = np.arange(len(probe)) * dt

    # Compare against a reference free-air run
    reset_state()
    set_porous_params(phi_bg, alpha_bg, sigma_bg)
    probe_ref = run_test(
        ('absorbing', 'absorbing', 'hard_wall', 'hard_wall'),
        0.030, (NX - 4, NY // 2), src_type='pulse'
    )
    t_ref = np.arange(len(probe_ref)) * dt

    amp_slab = np.max(np.abs(probe))
    amp_ref = np.max(np.abs(probe_ref))
    attenuation_db = 20.0 * np.log10(amp_slab / (amp_ref + 1e-30))
    print(f"  Transmitted amplitude: {amp_slab:.4f} vs reference {amp_ref:.4f}")
    print(f"  Attenuation: {attenuation_db:.2f} dB")

    status = "PASS" if attenuation_db < -1.0 else "FAIL"
    print(f"  ->  {status}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_slab * 1000, probe, 'b-', lw=0.8, label='With slab')
    ax.plot(t_ref * 1000, probe_ref, 'r--', lw=0.8, label='Free air')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Pressure p')
    ax.set_title('ZK-04: Pulse through porous slab')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / 'zk_test_04.png', dpi=150)
    plt.close(fig)
    return status


def test_energy_decay_in_porous_region():
    """TEST ZK-05: Energy decreases inside a damped region."""
    print("\nZK-TEST 05: Energy Decay in Porous Region")

    # Damped region in the centre, free air elsewhere
    set_porous_params(np.ones((NX, NY)), np.ones((NX, NY)), np.zeros((NX, NY)))
    x_start, x_end = NX // 2 - 40, NX // 2 + 40
    damp_mask = np.zeros((NX, NY), dtype=bool)
    damp_mask[x_start:x_end, :] = True
    set_porous_region(damp_mask, 1.0, 1.0, 80000.0)

    reset_state()
    BC_LEFT = BC_RIGHT = BC_TOP = BC_BOTTOM = 'hard_wall'
    SOURCE_TYPE = 'none'

    # Initial Gaussian pressure pulse centred in the damped region
    x_idx = np.arange(NX)
    y_idx = np.arange(NY)
    X, Y = np.meshgrid(x_idx, y_idx, indexing='ij')
    p[:, :] = np.exp(-((X - NX // 2)**2 + (Y - NY // 2)**2) / (2 * 5**2))

    n = int(0.020 / dt)
    energies = []
    for step_n in range(n):
        step(step_n * dt)
        energies.append(compute_energy()[2])
    energies = np.array(energies)

    # Energy should drop significantly due to damping
    drop = (energies[0] - energies[-1]) / energies[0] * 100
    status = "PASS" if drop > 20.0 else "FAIL"
    print(f"  Energy drop: {drop:.1f}%  ->  {status}")

    fig, ax = plt.subplots(figsize=(10, 4))
    t = np.arange(n) * dt
    ax.plot(t * 1000, energies / energies[0], 'k-', lw=0.8)
    ax.axvspan(x_start * DX * 1000, x_end * DX * 1000, alpha=0.2, color='red',
               label='Damped region')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('E / E0')
    ax.set_title('ZK-05: Energy decay in damped region')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / 'zk_test_05.png', dpi=150)
    plt.close(fig)
    return status


# =============================================================================
# SECTION 10: MAIN — Run Selected Tests
# =============================================================================

TESTS_TO_RUN = [
    'free_air_baseline',
    'slowdown_only',
    'damping_only',
    'porous_slab',
    'energy_decay_in_porous_region',
]

if __name__ == '__main__':
    print("Initialising porous FDTD solver...")
    # Default to free air
    set_porous_params(np.ones((NX, NY)), np.ones((NX, NY)), np.zeros((NX, NY)))

    results = {}
    for name in TESTS_TO_RUN:
        if name == 'free_air_baseline':
            results[name] = test_free_air_baseline()
        elif name == 'slowdown_only':
            results[name] = test_slowdown_only()
        elif name == 'damping_only':
            results[name] = test_damping_only()
        elif name == 'porous_slab':
            results[name] = test_porous_slab()
        elif name == 'energy_decay_in_porous_region':
            results[name] = test_energy_decay_in_porous_region()
        else:
            print(f"Unknown test: {name}")

    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    for name, status in results.items():
        print(f"  {name:35s}  {status}")
    total_pass = sum(1 for s in results.values() if s == "PASS")
    total_fail = sum(1 for s in results.values() if s == "FAIL")
    print(f"TOTAL: {total_pass} PASS, {total_fail} FAIL")
