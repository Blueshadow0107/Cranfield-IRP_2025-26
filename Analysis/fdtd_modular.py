"""
================================================================================
FDTD MODULAR — Clean Function-Based Design
================================================================================

Order of operations for any run:
    1. Build c_field (uniform OR thermal)
    2. set_c_field(c_field)  ← MUST call this before running
    3. Configure BCs and source
    4. reset_state()
    5. run_simulation(...)

This file is organised bottom-up: physics first, then tests.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
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
T0 = 300.0                      # K
D_THERMAL = 2e-5                # m²/s


# =============================================================================
# SECTION 2: STATE VARIABLES (initialised by set_c_field and reset_state)
# =============================================================================

# Sound speed field and derived quantities
c_field = None
c2 = None
c_max = None
dt = None

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
# SECTION 3: THERMAL SOLVER
# =============================================================================

def build_heat_source_strips(sigma_amps, positions, width):
    """
    Build 2D heat source field σ(x,y) with vertical strip heaters.
    Each strip is full height in y, centred at 'positions' [m] in x.
    """
    sigma = np.zeros((NX, NY))
    half_w_cells = int(width / (2 * DX))
    for amp, pos in zip(sigma_amps, positions):
        i_c = int(pos / DX)
        i_s = max(0, i_c - half_w_cells)
        i_e = min(NX, i_c + half_w_cells)
        sigma[i_s:i_e, :] = amp
    return sigma


def solve_temperature(sigma):
    """
    Solve steady-state thermal Poisson: ∇²T = -σ/D
    with Dirichlet T = T0 on all boundaries.
    """
    nx, ny = sigma.shape
    N = nx * ny

    rows, cols, data = [], [], []
    for i in range(nx):
        for j in range(ny):
            k = i * ny + j
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                # Boundary: T = T0
                rows.append(k); cols.append(k); data.append(1.0)
            else:
                # Interior: 5-point Laplacian
                rows.append(k); cols.append(k); data.append(-4.0)
                rows.append(k); cols.append((i + 1) * ny + j); data.append(1.0)
                rows.append(k); cols.append((i - 1) * ny + j); data.append(1.0)
                rows.append(k); cols.append(i * ny + (j + 1)); data.append(1.0)
                rows.append(k); cols.append(i * ny + (j - 1)); data.append(1.0)

    A = csr_matrix((data, (rows, cols)), shape=(N, N))

    b = np.zeros(N)
    for i in range(nx):
        for j in range(ny):
            k = i * ny + j
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                b[k] = T0
            else:
                b[k] = -sigma[i, j] * DX ** 2 / D_THERMAL

    T_flat = spsolve(A, b)
    return T_flat.reshape((nx, ny))


def build_c_field_thermal(sigma_amps, positions, width):
    """
    Full thermal pipeline:
        heat source → temperature → sound speed
    Returns c_field [m/s].
    """
    sigma = build_heat_source_strips(sigma_amps, positions, width)
    T = solve_temperature(sigma)
    return C0 * np.sqrt(T / T0)


# =============================================================================
# SECTION 4: SOLVER SETUP
# =============================================================================

def set_c_field(new_c_field):
    """
    Update sound speed field and recompute c2, c_max, dt.
    MUST be called every time the c-field changes.
    """
    global c_field, c2, c_max, dt
    c_field = new_c_field.astype(float)
    c2 = c_field ** 2
    c_max = float(c_field.max())
    dt = 0.9 * DX / (c_max * np.sqrt(2))
    print(f"  c_max = {c_max:.1f} m/s, dt = {dt*1e6:.3f} µs, Courant = {c_max*dt/DX:.4f}")


def reset_state():
    """Zero all acoustic fields. Call between tests."""
    global p, u, v
    p = np.zeros((NX, NY))
    u = np.zeros((NX + 1, NY))
    v = np.zeros((NX, NY + 1))


def apply_boundary_conditions():
    """Apply BCs to velocity fields."""
    global u, v

    # Left face
    if BC_LEFT == 'absorbing':
        u[0, :] = -p[0, :] / (RHO0 * c_field[0, :])
    elif BC_LEFT == 'hard_wall':
        u[0, :] = 0.0
    elif BC_LEFT == 'pressure_release':
        u[0, :] -= (dt / RHO0) * 2.0 * p[0, :] / DX

    # Right face
    if BC_RIGHT == 'absorbing':
        u[NX, :] = p[NX - 1, :] / (RHO0 * c_field[NX - 1, :])
    elif BC_RIGHT == 'hard_wall':
        u[NX, :] = 0.0
    elif BC_RIGHT == 'pressure_release':
        u[NX, :] += (dt / RHO0) * 2.0 * p[NX - 1, :] / DX

    # Bottom face
    if BC_BOTTOM == 'absorbing':
        v[:, 0] = -p[:, 0] / (RHO0 * c_field[:, 0])
    elif BC_BOTTOM == 'hard_wall':
        v[:, 0] = 0.0
    elif BC_BOTTOM == 'pressure_release':
        v[:, 0] -= (dt / RHO0) * 2.0 * p[:, 0] / DX

    # Top face
    if BC_TOP == 'absorbing':
        v[:, NY] = p[:, NY - 1] / (RHO0 * c_field[:, NY - 1])
    elif BC_TOP == 'hard_wall':
        v[:, NY] = 0.0
    elif BC_TOP == 'pressure_release':
        v[:, NY] += (dt / RHO0) * 2.0 * p[:, NY - 1] / DX


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
# SECTION 5: TIME STEPPING
# =============================================================================

def step(t):
    """One leapfrog step."""
    global p

    # Update interior velocities
    u[1:NX, :] -= (dt / RHO0) * (p[1:NX, :] - p[0:NX - 1, :]) / DX
    v[:, 1:NY] -= (dt / RHO0) * (p[:, 1:NY] - p[:, 0:NY - 1]) / DX

    # Apply BCs
    apply_boundary_conditions()

    # Update pressure
    du_dx = (u[1:NX + 1, :] - u[0:NX, :]) / DX
    dv_dy = (v[:, 1:NY + 1] - v[:, 0:NY]) / DX
    p -= dt * RHO0 * c2 * (du_dx + dv_dy)

    # Source injection
    p[SRC_I0:SRC_I1, :] += source_amplitude(t)


# =============================================================================
# SECTION 6: REUSABLE UTILITIES
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
    """Compute total acoustic energy: E = E_kin + E_pot."""
    e_pot = np.sum(p ** 2 / (2.0 * RHO0 * c2)) * DX ** 2
    u_avg = 0.5 * (u[0:NX, :] ** 2 + u[1:NX + 1, :] ** 2)
    v_avg = 0.5 * (v[:, 0:NY] ** 2 + v[:, 1:NY + 1] ** 2)
    e_kin = 0.5 * RHO0 * np.sum(u_avg + v_avg) * DX ** 2
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
# SECTION 7: PLOTTING
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
# SECTION 8: TEST FUNCTIONS
# =============================================================================

def test_pulse_arrival():
    """TEST 01: Measure pulse transit time."""
    global PULSE_TYPE, PULSE_T0

    print("\nTEST 01: Pulse Transit Time")
    PULSE_TYPE = 'gaussian'
    PULSE_T0 = 0.0005

    reset_state()
    probe = run_test(
        ('absorbing', 'absorbing', 'hard_wall', 'hard_wall'),
        0.020, (NX - 4, NY // 2), src_type='pulse'
    )
    t = np.arange(len(probe)) * dt

    t_meas = t[np.argmax(np.abs(probe))]
    t_theory = (NX - 4 - 4.5) * DX / C0 + PULSE_T0
    err = abs(t_meas - t_theory) / t_theory * 100
    status = "PASS" if err < 3.0 else "FAIL"
    print(f"  Error: {err:.2f}%  ->  {status}")

    plot_probe(t, probe, 'Test 01: Pulse arrival', FIG_DIR / 'test_01.png')
    return status


def test_hard_wall():
    """TEST 02: Compare peak pressure at hard wall vs. absorbing."""
    global SRC_AMP, SRC_I0, SRC_I1

    print("\nTEST 02: Hard Wall Reflection")
    freq, amp = 5000.0, 1.0
    src_i0, src_i1 = 197, 203
    probe_i = 0
    SRC_AMP = amp
    SRC_I0, SRC_I1 = src_i0, src_i1

    reset_state()
    probe_a = run_test(
        ('hard_wall', 'absorbing', 'hard_wall', 'hard_wall'),
        0.010, (probe_i, NY // 2), src_type='continuous_sine', src_freq=freq
    )
    p_hard = np.max(np.abs(probe_a[int(0.6 * len(probe_a)):]))

    reset_state()
    probe_b = run_test(
        ('absorbing', 'absorbing', 'hard_wall', 'hard_wall'),
        0.010, (probe_i, NY // 2), src_type='continuous_sine', src_freq=freq
    )
    p_abs = np.max(np.abs(probe_b[int(0.6 * len(probe_b)):]))

    ratio = p_hard / (p_abs + 1e-30)
    status = "PASS" if ratio > 1.3 else "FAIL"
    print(f"  Ratio: {ratio:.3f}  ->  {status}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    t = np.arange(len(probe_a)) * dt
    axes[0].plot(t * 1000, probe_a, 'b-', lw=0.5)
    axes[0].set_title('Hard wall'); axes[0].grid(True, alpha=0.3)
    axes[1].plot(t * 1000, probe_b, 'r-', lw=0.5)
    axes[1].set_title('Absorbing'); axes[1].set_xlabel('Time (ms)')
    axes[1].grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / 'test_02.png', dpi=150)
    plt.close(fig)
    return status


def test_pressure_release():
    """TEST 03: Quarter-wave resonator."""
    print("\nTEST 03: Pressure-Release Boundary")

    reset_state()
    x_idx = np.arange(NX)
    y_idx = np.arange(NY)
    X, Y = np.meshgrid(x_idx, y_idx, indexing='ij')
    p[:, :] = np.exp(-((X - NX // 2)**2 + (Y - NY // 2)**2) / (2 * 5**2))

    probe = run_test(
        ('hard_wall', 'pressure_release', 'hard_wall', 'hard_wall'),
        0.050, (5, NY // 2), src_type='none'
    )
    peak_freqs, _ = find_fft_peaks(probe, height_ratio=0.005, distance=5)
    peak_freqs = peak_freqs[peak_freqs < 5000][:15]

    f_theory = (2 * np.arange(1, 21) - 1) * C0 / (4 * NX * DX)

    matches = []
    used = set()
    for f_num in peak_freqs:
        best_err, best_idx = np.inf, -1
        for idx, f_th in enumerate(f_theory):
            if idx in used: continue
            err = abs(f_num - f_th)
            if err < best_err: best_err, best_idx = err, idx
        if best_idx >= 0 and best_err < 50.0:
            used.add(best_idx)
            err_pct = abs(f_num - f_theory[best_idx]) / f_theory[best_idx] * 100
            matches.append((best_idx + 1, f_theory[best_idx], f_num, err_pct))

    all_pass = True
    for n_val, f_th, f_num, err in matches[:10]:
        tol = 10.0 if f_th < 150.0 else 3.0
        if err >= tol: all_pass = False
    status = "PASS" if all_pass else "FAIL"
    print(f"  Matched {len(matches)} peaks  ->  {status}")

    plot_probe(np.arange(len(probe)) * dt, probe, 'Test 03: Ring-down', FIG_DIR / 'test_03.png')
    return status


def test_eigenfrequencies():
    """TEST 04: Closed cavity eigenmodes."""
    print("\nTEST 04: Eigenfrequencies")

    reset_state()
    x_idx = np.arange(NX)
    y_idx = np.arange(NY)
    X, Y = np.meshgrid(x_idx, y_idx, indexing='ij')
    p[:, :] = np.exp(-((X - NX // 2)**2 + (Y - NY // 2)**2) / (2 * 5**2))

    probe = run_test(
        ('hard_wall', 'hard_wall', 'hard_wall', 'hard_wall'),
        0.050, (5, NY // 2), src_type='none'
    )
    peak_freqs, _ = find_fft_peaks(probe, height_ratio=0.005, distance=30)
    peak_freqs = peak_freqs[peak_freqs < 4000][:10]

    Lx, Ly = NX * DX, NY * DX
    modes = []
    for n_mode in range(20):
        for m in range(20):
            if n_mode == 0 and m == 0: continue
            f_th = 0.5 * C0 * np.sqrt((n_mode / Lx)**2 + (m / Ly)**2)
            if f_th < 4000: modes.append(f_th)
    modes.sort()
    f_theory = np.array(modes)

    matches = []
    used = set()
    for f_num in peak_freqs:
        best_err, best_idx = np.inf, -1
        for idx, f_th in enumerate(f_theory):
            if idx in used: continue
            err = abs(f_num - f_th)
            if err < best_err: best_err, best_idx = err, idx
        if best_idx >= 0 and best_err < 100.0:
            used.add(best_idx)
            matches.append((f_num, f_theory[best_idx], abs(f_num - f_theory[best_idx]) / f_theory[best_idx] * 100))

    all_pass = all(err < 3.0 for _, _, err in matches[:8])
    status = "PASS" if all_pass else "FAIL"
    print(f"  Matched {len(matches)} modes  ->  {status}")

    plot_probe(np.arange(len(probe)) * dt, probe, 'Test 04: Ring-down', FIG_DIR / 'test_04.png')
    return status


def test_energy_conservation():
    """TEST 05: Energy drift in closed lossless cavity."""
    global BC_LEFT, BC_RIGHT, BC_TOP, BC_BOTTOM, SOURCE_TYPE, p

    print("\nTEST 05: Energy Conservation")
    BC_LEFT = BC_RIGHT = BC_TOP = BC_BOTTOM = 'hard_wall'
    SOURCE_TYPE = 'none'

    reset_state()
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

    drift = (np.max(energies) - np.min(energies)) / energies[0] * 100
    status = "PASS" if drift < 8.0 else "FAIL"
    print(f"  Drift: {drift:.2f}%  ->  {status}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    t = np.arange(n) * dt
    axes[0].plot(t * 1000, energies, 'k-', lw=0.8)
    axes[0].set_ylabel('Energy'); axes[0].set_title('Total energy'); axes[0].grid(True, alpha=0.3)
    axes[1].plot(t * 1000, (energies / energies[0] - 1) * 100, 'k-', lw=0.8)
    axes[1].axhline(0, color='g', ls='--', alpha=0.5)
    axes[1].set_xlabel('Time (ms)'); axes[1].set_ylabel('Drift (%)')
    axes[1].set_title(f'Drift: {drift:.2f}%'); axes[1].grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / 'test_05.png', dpi=150)
    plt.close(fig)
    return status


def test_phase_velocity():
    """TEST 06: Wavelength measurement at multiple frequencies."""
    global BC_LEFT, BC_RIGHT, BC_TOP, BC_BOTTOM, SOURCE_TYPE, SRC_FREQ, SRC_I0, SRC_I1

    print("\nTEST 06: Phase Velocity")
    BC_LEFT = BC_RIGHT = 'absorbing'
    BC_TOP = BC_BOTTOM = 'hard_wall'
    SOURCE_TYPE = 'continuous_sine'
    SRC_I0, SRC_I1 = 3, 6
    freqs_test = [2000.0, 5000.0, 8000.0, 10000.0]

    results = []
    for freq in freqs_test:
        SRC_FREQ = freq
        reset_state()
        n = int(0.015 / dt)
        discard = int(0.6 * n)
        profiles = []
        for step_n in range(n):
            step(step_n * dt)
            if step_n >= discard:
                profiles.append(p[:, NY // 2].copy())

        profile = np.mean(profiles, axis=0)
        peaks, _ = find_peaks(profile, height=np.max(profile) * 0.1, distance=5)
        peak_x = peaks * DX
        if len(peak_x) >= 2:
            lambda_num = np.mean(np.diff(peak_x))
        else:
            lambda_num = np.nan

        lambda_theory = C0 / freq
        err = abs(lambda_num - lambda_theory) / lambda_theory * 100 if not np.isnan(lambda_num) else np.nan
        status = "PASS" if err < 3.0 else "FAIL"
        results.append((freq, lambda_theory, lambda_num, err, status))
        print(f"  f = {freq/1000:.1f} kHz: err = {err:.2f}%  ->  {status}")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, (freq, _, _, _, _) in zip(axes, results):
        SRC_FREQ = freq
        reset_state()
        n = int(0.015 / dt)
        discard = int(0.6 * n)
        profiles = []
        for step_n in range(n):
            step(step_n * dt)
            if step_n >= discard:
                profiles.append(p[:, NY // 2].copy())
        profile = np.mean(profiles, axis=0)
        peaks, _ = find_peaks(profile, height=np.max(profile) * 0.1, distance=5)
        x = np.arange(NX) * DX * 1000
        ax.plot(x, profile, 'b-', lw=0.8)
        ax.plot(peaks * DX * 1000, profile[peaks], 'ro', ms=4)
        ax.set_xlabel('x (mm)'); ax.set_ylabel('p')
        ax.set_title(f'{freq/1000:.1f} kHz')
        ax.grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / 'test_06.png', dpi=150)
    plt.close(fig)

    all_pass = all(r[4] == "PASS" for r in results)
    return "PASS" if all_pass else "FAIL"


# =============================================================================
# SECTION 9: MAIN — Run Selected Tests
# =============================================================================

TESTS_TO_RUN = [
    'pulse_arrival',
    'hard_wall',
    'pressure_release',
    'eigenfrequencies',
    'energy_conservation',
    'phase_velocity',
]

if __name__ == '__main__':
    # Initialise with uniform sound speed
    print("Initialising solver...")
    set_c_field(np.full((NX, NY), C0))

    results = {}
    for name in TESTS_TO_RUN:
        if name == 'pulse_arrival':
            results[name] = test_pulse_arrival()
        elif name == 'hard_wall':
            results[name] = test_hard_wall()
        elif name == 'pressure_release':
            results[name] = test_pressure_release()
        elif name == 'eigenfrequencies':
            results[name] = test_eigenfrequencies()
        elif name == 'energy_conservation':
            results[name] = test_energy_conservation()
        elif name == 'phase_velocity':
            results[name] = test_phase_velocity()
        else:
            print(f"Unknown test: {name}")

    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    for name, status in results.items():
        print(f"  {name:25s}  {status}")
    total_pass = sum(1 for s in results.values() if s == "PASS")
    total_fail = sum(1 for s in results.values() if s == "FAIL")
    print(f"TOTAL: {total_pass} PASS, {total_fail} FAIL")
