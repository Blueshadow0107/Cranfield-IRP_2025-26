"""
================================================================================
FDTD MASTER MONOLITH — Educational Version
================================================================================

This is a single-file, heavily-commented 2D acoustic FDTD solver.
It contains EVERYTHING in one place so you can read top-to-bottom and
understand exactly what each line does.

The code is intentionally verbose. Production code would split this into
modules and use classes — this version is for learning.

PHYSICS
-------
First-order linear acoustics in 2D:
    ∂u/∂t = -(1/ρ₀) ∂p/∂x
    ∂v/∂t = -(1/ρ₀) ∂p/∂y
    ∂p/∂t = -ρ₀ c² (∂u/∂x + ∂v/∂y)

GRID (Yee Staggered)
--------------------
    p[i,j]  — pressure at cell centres     shape: (nx, ny)
    u[i,j]  — x-velocity at vertical faces  shape: (nx+1, ny)
    v[i,j]  — y-velocity at horizontal faces shape: (nx, ny+1)

    x:    0      0.5·dx     dx      1.5·dx    2·dx  ...  L-0.5·dx    L
          |         |        |         |        |   ...      |        |
         u[0]     p[0]     u[1]     p[1]     u[2]  ...    p[nx-1]  u[nx]
        face    centre    face    centre    face   ...    centre   face

TIME STEPPING (Leapfrog)
------------------------
    1. Update u, v using OLD pressure
    2. Apply boundary conditions to u, v
    3. Update p using NEW u, v
    4. Add source injection

STABILITY (CFL)
---------------
    dt < dx / (c_max · √2)    for 2D

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# =============================================================================
# SECTION 1: PHYSICAL PARAMETERS (edit these to change the problem)
# =============================================================================

# Grid dimensions (number of CELLS, not points)
NX = 400          # cells in x
NY = 50           # cells in y
DX = 0.002        # cell size [m]  (2 mm)
DY = DX           # uniform square cells

# Domain size
LX = NX * DX      # 0.8 m
LY = NY * DY      # 0.1 m

# Air at 300 K
RHO0 = 1.225      # density [kg/m³]
C0 = 343.0        # sound speed [m/s]
T0 = 300.0        # temperature [K]

# Thermal diffusivity of air (only needed if you use thermal strips)
D_THERMAL = 2e-5  # m²/s

# =============================================================================
# SECTION 2: SOUND SPEED FIELD (uniform, or with thermal strips)
# =============================================================================

def build_c_field_uniform():
    """Simplest case: constant sound speed everywhere."""
    return np.full((NX, NY), C0)


def solve_temperature(sigma, dx=DX, D=D_THERMAL, T0=T0):
    """
    Solve steady-state thermal Poisson: ∇²T = -σ/D
    with Dirichlet T = T0 on all boundaries.

    This is a LINEAR SYSTEM solved with sparse direct solver.
    For each cell (i,j), we write one equation:
        - Interior: T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1] - 4T[i,j] = -σ·dx²/D
        - Boundary: T[i,j] = T0

    We flatten T[i,j] into a 1D vector and build a sparse matrix.
    """
    nx, ny = sigma.shape
    N = nx * ny

    # Build sparse Laplacian matrix
    rows, cols, data = [], [], []
    for i in range(nx):
        for j in range(ny):
            k = i * ny + j  # flatten 2D index (i,j) -> 1D index k

            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                # BOUNDARY CELL: Dirichlet T = T0
                rows.append(k)
                cols.append(k)
                data.append(1.0)
            else:
                # INTERIOR CELL: 5-point Laplacian stencil
                # center: -4
                rows.append(k); cols.append(k); data.append(-4.0)
                # neighbours: +1 each
                rows.append(k); cols.append((i + 1) * ny + j); data.append(1.0)
                rows.append(k); cols.append((i - 1) * ny + j); data.append(1.0)
                rows.append(k); cols.append(i * ny + (j + 1)); data.append(1.0)
                rows.append(k); cols.append(i * ny + (j - 1)); data.append(1.0)

    A = csr_matrix((data, (rows, cols)), shape=(N, N))

    # Build right-hand side vector
    b = np.zeros(N)
    for i in range(nx):
        for j in range(ny):
            k = i * ny + j
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                b[k] = T0
            else:
                b[k] = -sigma[i, j] * dx ** 2 / D

    # Solve A·T = b
    T_flat = spsolve(A, b)
    return T_flat.reshape((nx, ny))


def build_heat_source_strips(sigma_amps, positions, width):
    """
    Build heat source field σ(x,y) with vertical strip heaters.
    Each strip is full height in y, centred at 'positions' in x.
    """
    sigma = np.zeros((NX, NY))
    half_w_cells = int(width / (2 * DX))
    for amp, pos in zip(sigma_amps, positions):
        i_c = int(pos / DX)
        i_s = max(0, i_c - half_w_cells)
        i_e = min(NX, i_c + half_w_cells)
        sigma[i_s:i_e, :] = amp
    return sigma


def build_c_field_from_thermal(sigma_amps, positions, width):
    """Thermal strips -> temperature field -> sound speed field."""
    sigma = build_heat_source_strips(sigma_amps, positions, width)
    T = solve_temperature(sigma)
    return C0 * np.sqrt(T / T0)


# =============================================================================
# SECTION 3: STATE INITIALISATION
# =============================================================================

# Sound speed field (choose ONE):
c_field = build_c_field_uniform()         # uniform medium
# c_field = build_c_field_from_thermal(...)  # with thermal strips

# Pre-compute c² for efficiency
c2 = c_field ** 2

# State arrays (all zero initially)
p = np.zeros((NX, NY))        # pressure at cell centres
u = np.zeros((NX + 1, NY))    # x-velocity at vertical faces
v = np.zeros((NX, NY + 1))    # y-velocity at horizontal faces

# Time step from CFL condition
c_max = float(c_field.max())
dt = 0.9 * DX / (c_max * np.sqrt(2))
print(f"dt = {dt*1e6:.3f} µs, Courant = {c_max*dt/DX:.4f}")


# =============================================================================
# SECTION 4: BOUNDARY CONDITIONS (choose per face)
# =============================================================================
# Options: 'absorbing', 'hard_wall', 'pressure_release'

BC_LEFT = 'absorbing'
BC_RIGHT = 'absorbing'
BC_BOTTOM = 'hard_wall'
BC_TOP = 'hard_wall'


def apply_boundary_conditions():
    """
    Apply boundary conditions to the velocity fields.
    This is called AFTER the interior velocity update and BEFORE the pressure update.
    """
    global u, v

    # --- LEFT face (x = 0) ---
    if BC_LEFT == 'absorbing':
        # Impedance match: u = -p/(ρ₀c)  (Engquist-Majda / Mur ABC)
        # Negative sign: wave traveling left (exiting domain)
        u[0, :] = -p[0, :] / (RHO0 * c_field[0, :])

    elif BC_LEFT == 'hard_wall':
        # Rigid boundary: zero normal velocity
        u[0, :] = 0.0

    elif BC_LEFT == 'pressure_release':
        # p = 0 at boundary face.
        # Ghost cell: p_ghost = -p[0] (antisymmetric reflection)
        # Velocity update: u[0] -= (dt/ρ₀) * (p[0] - p_ghost) / dx
        #                = (dt/ρ₀) * 2p[0] / dx
        u[0, :] -= (dt / RHO0) * 2.0 * p[0, :] / DX

    # --- RIGHT face (x = L) ---
    if BC_RIGHT == 'absorbing':
        # Wave traveling right (exiting domain)
        u[NX, :] = p[NX - 1, :] / (RHO0 * c_field[NX - 1, :])

    elif BC_RIGHT == 'hard_wall':
        u[NX, :] = 0.0

    elif BC_RIGHT == 'pressure_release':
        # Note: += here because ∂p/∂x is NEGATIVE at right boundary
        # (0 - p[nx-1]) / (0.5·dx) = -2p[nx-1]/dx
        # ∂u/∂t = -(1/ρ₀)∂p/∂x = +(2/ρ₀dx)p[nx-1]
        u[NX, :] += (dt / RHO0) * 2.0 * p[NX - 1, :] / DX

    # --- BOTTOM face (y = 0) ---
    if BC_BOTTOM == 'absorbing':
        v[:, 0] = -p[:, 0] / (RHO0 * c_field[:, 0])

    elif BC_BOTTOM == 'hard_wall':
        v[:, 0] = 0.0

    elif BC_BOTTOM == 'pressure_release':
        v[:, 0] -= (dt / RHO0) * 2.0 * p[:, 0] / DY

    # --- TOP face (y = L) ---
    if BC_TOP == 'absorbing':
        v[:, NY] = p[:, NY - 1] / (RHO0 * c_field[:, NY - 1])

    elif BC_TOP == 'hard_wall':
        v[:, NY] = 0.0

    elif BC_TOP == 'pressure_release':
        # Note: += for same reason as right boundary
        v[:, NY] += (dt / RHO0) * 2.0 * p[:, NY - 1] / DY


# =============================================================================
# SECTION 5: SOURCE INJECTION (choose type and parameters)
# =============================================================================

SOURCE_TYPE = 'continuous_sine'   # options: 'continuous_sine', 'gaussian_pulse', 'broadband_pulse'
SRC_I0, SRC_I1 = 3, 6             # x-cells where source is injected

# Continuous sine parameters
SRC_FREQ = 5000.0                 # Hz
SRC_AMP = 1.0

# Gaussian pulse parameters
PULSE_T0 = 0.0005                 # pulse centre time [s]
PULSE_TAU = 0.0001                # pulse width [s]
PULSE_F0 = 6000.0                 # carrier frequency [Hz]


def source_amplitude(t):
    """Compute source amplitude at time t."""
    if SOURCE_TYPE == 'continuous_sine':
        return SRC_AMP * np.sin(2.0 * np.pi * SRC_FREQ * t)

    elif SOURCE_TYPE in ('gaussian_pulse', 'broadband_pulse'):
        envelope = np.exp(-((t - PULSE_T0) / PULSE_TAU) ** 2)
        return SRC_AMP * envelope * np.sin(2.0 * np.pi * PULSE_F0 * (t - PULSE_T0))

    else:
        return 0.0


# =============================================================================
# SECTION 6: CORE TIME-STEPPING LOOP
# =============================================================================

def step(t):
    """Advance one leapfrog step."""
    global p, u, v

    # --- 6.1: Update INTERIOR velocities ---
    # u[i] lives between p[i-1] and p[i]
    # ∂u/∂t = -(1/ρ₀) ∂p/∂x  ->  centred difference
    u[1:NX, :] -= (dt / RHO0) * (p[1:NX, :] - p[0:NX - 1, :]) / DX

    # v[j] lives between p[:,j-1] and p[:,j]
    v[:, 1:NY] -= (dt / RHO0) * (p[:, 1:NY] - p[:, 0:NY - 1]) / DY

    # --- 6.2: Apply boundary conditions ---
    apply_boundary_conditions()

    # --- 6.3: Update pressure ---
    # p[i,j] uses u[i+1] - u[i] for x-flux and v[j+1] - v[j] for y-flux
    du_dx = (u[1:NX + 1, :] - u[0:NX, :]) / DX
    dv_dy = (v[:, 1:NY + 1] - v[:, 0:NY]) / DY
    p -= dt * RHO0 * c2 * (du_dx + dv_dy)

    # --- 6.4: Source injection ---
    src = source_amplitude(t)
    p[SRC_I0:SRC_I1, :] += src


# =============================================================================
# SECTION 7: STATE RESET HELPER
# =============================================================================

def reset_state():
    """Reallocate and zero all arrays. Call this between tests."""
    global p, u, v, time, n_steps
    p = np.zeros((NX, NY))
    u = np.zeros((NX + 1, NY))
    v = np.zeros((NX, NY + 1))
    time = 0.0
    n_steps = 0


# =============================================================================
# SECTION 8: VALIDATION TESTS
# =============================================================================
# Each test:
#   1. Sets BCs, source, probe location
#   2. Runs the simulation loop
#   3. Analyses and prints PASS/FAIL
#   4. Saves a figure

from scipy.signal import find_peaks

FIG_DIR = Path(__file__).parent / 'figures'
FIG_DIR.mkdir(exist_ok=True)


def test_pulse_arrival():
    """TEST 01: Measure pulse transit time."""
    global BC_LEFT, BC_RIGHT, BC_TOP, BC_BOTTOM, SOURCE_TYPE
    global SRC_I0, SRC_I1, PULSE_T0, PULSE_TAU, PULSE_F0

    print("\n" + "=" * 60)
    print("TEST 01: Pulse Transit Time")
    print("=" * 60)

    # Configure
    BC_LEFT = 'absorbing'; BC_RIGHT = 'absorbing'
    BC_TOP = 'absorbing'; BC_BOTTOM = 'absorbing'
    SOURCE_TYPE = 'gaussian_pulse'
    SRC_I0, SRC_I1 = 3, 6
    PULSE_T0, PULSE_TAU, PULSE_F0 = 0.0005, 0.0001, 6000.0

    probe_i, probe_j = NX - 4, NY // 2
    src_centre = (SRC_I0 + SRC_I1) / 2.0

    # Run
    reset_state()
    duration = 0.020
    n = int(duration / dt)
    probe_history = []
    for step_n in range(n):
        step(step_n * dt)
        probe_history.append(float(p[probe_i, probe_j]))
    probe_history = np.array(probe_history)
    t_history = np.arange(n) * dt

    # Analyse
    peak_idx = np.argmax(np.abs(probe_history))
    t_meas = t_history[peak_idx]
    t_theory = (probe_i - src_centre) * DX / C0 + PULSE_T0
    err = abs(t_meas - t_theory) / t_theory * 100
    status = "PASS" if err < 3.0 else "FAIL"
    print(f"Transit (theory): {t_theory*1000:.3f} ms")
    print(f"Transit (measured): {t_meas*1000:.3f} ms")
    print(f"Error: {err:.2f}%  ->  {status}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_history * 1000, probe_history, 'b-', lw=0.8)
    ax.axvline(t_theory * 1000, color='g', ls='--', label=f'theory: {t_theory*1000:.3f} ms')
    ax.axvline(t_meas * 1000, color='r', ls='--', label=f'measured: {t_meas*1000:.3f} ms')
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('p'); ax.set_title('Test 01: Pulse arrival')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / 'test_01_pulse_arrival.png', dpi=150)
    plt.close(fig)
    return status


def test_hard_wall():
    """TEST 02: Compare peak pressure at hard wall vs. absorbing."""
    global BC_LEFT, BC_RIGHT, BC_TOP, BC_BOTTOM, SOURCE_TYPE, SRC_FREQ, SRC_AMP
    global SRC_I0, SRC_I1

    print("\n" + "=" * 60)
    print("TEST 02: Hard Wall Reflection")
    print("=" * 60)

    freq, amp = 5000.0, 1.0
    src_i0, src_i1 = 197, 203
    probe_i, probe_j = 0, NY // 2

    # --- Run A: hard wall on left ---
    BC_LEFT = 'hard_wall'; BC_RIGHT = 'absorbing'
    BC_TOP = 'hard_wall'; BC_BOTTOM = 'hard_wall'
    SOURCE_TYPE = 'continuous_sine'
    SRC_FREQ, SRC_AMP = freq, amp
    SRC_I0, SRC_I1 = src_i0, src_i1

    reset_state()
    duration = 0.010
    n = int(duration / dt)
    probe_a = []
    for step_n in range(n):
        step(step_n * dt)
        probe_a.append(float(p[probe_i, probe_j]))
    probe_a = np.array(probe_a)
    p_max_hard = np.max(np.abs(probe_a[int(0.6 * n):]))

    # --- Run B: absorbing on left ---
    BC_LEFT = 'absorbing'

    reset_state()
    probe_b = []
    for step_n in range(n):
        step(step_n * dt)
        probe_b.append(float(p[probe_i, probe_j]))
    probe_b = np.array(probe_b)
    p_max_abs = np.max(np.abs(probe_b[int(0.6 * n):]))

    ratio = p_max_hard / (p_max_abs + 1e-30)
    status = "PASS" if ratio > 1.3 else "FAIL"
    print(f"Peak (hard wall): {p_max_hard:.4f}")
    print(f"Peak (absorbing): {p_max_abs:.4f}")
    print(f"Ratio: {ratio:.3f}  ->  {status}")

    t = np.arange(n) * dt
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(t * 1000, probe_a, 'b-', lw=0.5); axes[0].set_title('Hard wall'); axes[0].grid(True, alpha=0.3)
    axes[1].plot(t * 1000, probe_b, 'r-', lw=0.5); axes[1].set_title('Absorbing'); axes[1].set_xlabel('Time (ms)')
    axes[1].grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / 'test_02_hard_wall.png', dpi=150)
    plt.close(fig)
    return status


def test_pressure_release():
    """TEST 03: Quarter-wave resonator with pressure-release boundary."""
    global BC_LEFT, BC_RIGHT, BC_TOP, BC_BOTTOM

    print("\n" + "=" * 60)
    print("TEST 03: Pressure-Release Boundary")
    print("=" * 60)

    BC_LEFT = 'hard_wall'; BC_RIGHT = 'pressure_release'
    BC_TOP = 'hard_wall'; BC_BOTTOM = 'hard_wall'

    reset_state()

    # Initial Gaussian pulse (no source during run)
    x_idx = np.arange(NX)
    y_idx = np.arange(NY)
    X, Y = np.meshgrid(x_idx, y_idx, indexing='ij')
    p[:, :] = 1.0 * np.exp(-((X - NX // 2)**2 + (Y - NY // 2)**2) / (2 * 5**2))

    duration = 0.050
    n = int(duration / dt)
    probe_history = []
    for step_n in range(n):
        step(step_n * dt)
        probe_history.append(float(p[5, NY // 2]))

    probe_history = np.array(probe_history)

    # FFT
    window = np.hanning(len(probe_history))
    freqs = np.fft.rfftfreq(len(probe_history), dt)
    spectrum = np.abs(np.fft.rfft(probe_history * window))

    prom = np.max(spectrum) * 0.005
    peaks, _ = find_peaks(spectrum, prominence=prom, distance=5)
    peak_freqs = freqs[peaks]
    peak_freqs = peak_freqs[peak_freqs < 5000][:15]

    f_theory = (2 * np.arange(1, 21) - 1) * C0 / (4 * NX * DX)

    matches = []
    used = set()
    for f_num in peak_freqs:
        best_err = np.inf; best_idx = -1
        for idx, f_th in enumerate(f_theory):
            if idx in used: continue
            err = abs(f_num - f_th)
            if err < best_err: best_err = err; best_idx = idx
        if best_idx >= 0 and best_err < 50.0:
            used.add(best_idx)
            err_pct = abs(f_num - f_theory[best_idx]) / f_theory[best_idx] * 100
            matches.append((best_idx + 1, f_theory[best_idx], f_num, err_pct))

    all_pass = True
    print(f"{'n':>3}  {'Theory (Hz)':>14}  {'Numeric (Hz)':>14}  {'Error %':>10}  {'Status'}")
    print("-" * 60)
    for n_val, f_th, f_num, err in matches[:10]:
        tol = 10.0 if f_th < 150.0 else 3.0
        st = "PASS" if err < tol else "FAIL"
        if err >= tol: all_pass = False
        print(f"{n_val:3d}  {f_th:14.2f}  {f_num:14.2f}  {err:10.2f}  {st}")
    print("-" * 60)
    status = "PASS" if all_pass else "FAIL"
    print(f"Overall: {status}")

    # Plot
    t = np.arange(len(probe_history)) * dt
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(t * 1000, probe_history, 'b-', lw=0.5)
    axes[0].set_xlabel('Time (ms)'); axes[0].set_ylabel('p'); axes[0].set_title('Ring-down')
    axes[0].grid(True, alpha=0.3)
    axes[1].semilogy(freqs, spectrum, 'b-', lw=0.5, alpha=0.7)
    matched_freqs = [m[2] for m in matches]
    matched_amps = [spectrum[np.argmin(np.abs(freqs - f))] for f in matched_freqs]
    axes[1].plot(matched_freqs, matched_amps, 'ro', ms=6, label='matched')
    for f_th in f_theory[:10]: axes[1].axvline(f_th, color='g', ls='--', alpha=0.4)
    axes[1].set_xlabel('Frequency (Hz)'); axes[1].set_ylabel('|P(f)|')
    axes[1].set_title('FFT: quarter-wave peaks')
    axes[1].set_xlim(0, 5000); axes[1].grid(True, alpha=0.3); axes[1].legend()
    fig.savefig(FIG_DIR / 'test_03_pressure_release.png', dpi=150)
    plt.close(fig)
    return status


def test_eigenfrequencies():
    """TEST 04: Closed cavity eigenmodes."""
    global BC_LEFT, BC_RIGHT, BC_TOP, BC_BOTTOM, p, u, v

    print("\n" + "=" * 60)
    print("TEST 04: Eigenfrequencies")
    print("=" * 60)

    BC_LEFT = 'hard_wall'; BC_RIGHT = 'hard_wall'
    BC_TOP = 'hard_wall'; BC_BOTTOM = 'hard_wall'

    reset_state()

    # Initial Gaussian pulse
    x_idx = np.arange(NX)
    y_idx = np.arange(NY)
    X, Y = np.meshgrid(x_idx, y_idx, indexing='ij')
    p[:, :] = 1.0 * np.exp(-((X - NX // 2)**2 + (Y - NY // 2)**2) / (2 * 5**2))

    duration = 0.050
    n = int(duration / dt)
    probe_history = []
    for step_n in range(n):
        t = step_n * dt
        u[1:NX, :] -= (dt / RHO0) * (p[1:NX, :] - p[0:NX-1, :]) / DX
        v[:, 1:NY] -= (dt / RHO0) * (p[:, 1:NY] - p[:, 0:NY-1]) / DY
        apply_boundary_conditions()
        du_dx = (u[1:NX+1, :] - u[0:NX, :]) / DX
        dv_dy = (v[:, 1:NY+1] - v[:, 0:NY]) / DY
        p -= dt * RHO0 * c2 * (du_dx + dv_dy)
        probe_history.append(float(p[5, NY // 2]))

    probe_history = np.array(probe_history)
    window = np.hanning(len(probe_history))
    freqs = np.fft.rfftfreq(len(probe_history), dt)
    spectrum = np.abs(np.fft.rfft(probe_history * window))

    peaks, _ = find_peaks(spectrum, height=np.max(spectrum) * 0.005, distance=30)
    peak_freqs = freqs[peaks]
    peak_freqs = peak_freqs[peak_freqs < 4000][:10]

    modes = []
    for n_mode in range(20):
        for m in range(20):
            if n_mode == 0 and m == 0: continue
            f_th = 0.5 * C0 * np.sqrt((n_mode / LX)**2 + (m / LY)**2)
            if f_th < 4000: modes.append((f_th, n_mode, m))
    modes.sort()
    f_theory = np.array([m[0] for m in modes])

    matches = []
    used = set()
    for f_num in peak_freqs:
        best_err = np.inf; best_idx = -1
        for idx, f_th in enumerate(f_theory):
            if idx in used: continue
            err = abs(f_num - f_th)
            if err < best_err: best_err = err; best_idx = idx
        if best_idx >= 0 and best_err < 100.0:
            used.add(best_idx)
            err_pct = abs(f_num - f_theory[best_idx]) / f_theory[best_idx] * 100
            matches.append((f_num, f_theory[best_idx], err_pct))

    all_pass = True
    print(f"{'Numeric (Hz)':>14}  {'Theory (Hz)':>14}  {'Error %':>10}  {'Status'}")
    print("-" * 60)
    for f_num, f_th, err in matches[:8]:
        st = "PASS" if err < 3.0 else "FAIL"
        if err >= 3.0: all_pass = False
        print(f"{f_num:14.2f}  {f_th:14.2f}  {err:10.2f}  {st}")
    print("-" * 60)
    status = "PASS" if all_pass else "FAIL"
    print(f"Overall: {status}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    t = np.arange(len(probe_history)) * dt
    axes[0].plot(t * 1000, probe_history, 'b-', lw=0.5)
    axes[0].set_xlabel('Time (ms)'); axes[0].set_ylabel('p'); axes[0].set_title('Ring-down')
    axes[0].grid(True, alpha=0.3)
    axes[1].semilogy(freqs, spectrum, 'b-', lw=0.5, alpha=0.7)
    axes[1].plot(peak_freqs, spectrum[peaks][:len(peak_freqs)], 'ro', ms=5)
    for f_th, n, m in modes[:15]: axes[1].axvline(f_th, color='g', ls='--', alpha=0.3)
    axes[1].set_xlabel('Frequency (Hz)'); axes[1].set_ylabel('|P(f)|')
    axes[1].set_title('FFT: cavity modes'); axes[1].set_xlim(0, 4000)
    axes[1].grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / 'test_04_eigenfrequencies.png', dpi=150)
    plt.close(fig)
    return status


def test_energy_conservation():
    """TEST 05: Energy drift in closed lossless cavity."""
    global BC_LEFT, BC_RIGHT, BC_TOP, BC_BOTTOM, p, u, v

    print("\n" + "=" * 60)
    print("TEST 05: Energy Conservation")
    print("=" * 60)

    BC_LEFT = 'hard_wall'; BC_RIGHT = 'hard_wall'
    BC_TOP = 'hard_wall'; BC_BOTTOM = 'hard_wall'

    # Initial Gaussian pulse
    x_idx = np.arange(NX)
    y_idx = np.arange(NY)
    X, Y = np.meshgrid(x_idx, y_idx, indexing='ij')
    p[:, :] = 1.0 * np.exp(-((X - NX // 2)**2 + (Y - NY // 2)**2) / (2 * 5**2))

    n = int(0.020 / dt)
    energies = []
    for step_n in range(n):
        t = step_n * dt
        u[1:NX, :] -= (dt / RHO0) * (p[1:NX, :] - p[0:NX-1, :]) / DX
        v[:, 1:NY] -= (dt / RHO0) * (p[:, 1:NY] - p[:, 0:NY-1]) / DY
        apply_boundary_conditions()
        du_dx = (u[1:NX+1, :] - u[0:NX, :]) / DX
        dv_dy = (v[:, 1:NY+1] - v[:, 0:NY]) / DY
        p -= dt * RHO0 * c2 * (du_dx + dv_dy)

        # Energy computation
        e_pot = np.sum(p**2 / (2.0 * RHO0 * c2)) * DX * DX
        u_avg = 0.5 * (u[0:NX, :]**2 + u[1:NX+1, :]**2)
        v_avg = 0.5 * (v[:, 0:NY]**2 + v[:, 1:NY+1]**2)
        e_kin = 0.5 * RHO0 * np.sum(u_avg + v_avg) * DX * DX
        energies.append(e_kin + e_pot)

    energies = np.array(energies)
    t = np.arange(n) * dt

    e_initial = energies[0]
    e_max = np.max(energies)
    e_min = np.min(energies)
    drift = (e_max - e_min) / e_initial * 100
    status = "PASS" if drift < 8.0 else "FAIL"

    print(f"Initial: {e_initial:.6e} J")
    print(f"Max: {e_max:.6e} J")
    print(f"Min: {e_min:.6e} J")
    print(f"Drift: {drift:.2f}%  ->  {status}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(t * 1000, energies, 'k-', lw=0.8)
    axes[0].set_ylabel('Energy'); axes[0].set_title('Total energy')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t * 1000, (energies / e_initial - 1) * 100, 'k-', lw=0.8)
    axes[1].axhline(0, color='g', ls='--', alpha=0.5)
    axes[1].set_xlabel('Time (ms)'); axes[1].set_ylabel('Drift (%)')
    axes[1].set_title(f'Drift: {drift:.2f}%'); axes[1].grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / 'test_05_energy_conservation.png', dpi=150)
    plt.close(fig)
    return status


def test_phase_velocity():
    """TEST 06: Wavelength measurement at multiple frequencies."""
    global BC_LEFT, BC_RIGHT, BC_TOP, BC_BOTTOM, SOURCE_TYPE, SRC_FREQ, SRC_AMP, SRC_I0, SRC_I1

    print("\n" + "=" * 60)
    print("TEST 06: Phase Velocity")
    print("=" * 60)

    freqs_test = [2000.0, 5000.0, 8000.0, 10000.0]
    results = []

    BC_LEFT = 'absorbing'; BC_RIGHT = 'absorbing'
    BC_TOP = 'hard_wall'; BC_BOTTOM = 'hard_wall'
    SOURCE_TYPE = 'continuous_sine'
    SRC_I0, SRC_I1 = 3, 6

    for freq in freqs_test:
        SRC_FREQ, SRC_AMP = freq, 1.0
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
        status = "PASS" if err < 3.0 else "FAIL" if not np.isnan(err) else "FAIL"
        results.append((freq, lambda_theory, lambda_num, err, status, profile, peaks))
        print(f"f = {freq/1000:.1f} kHz: theory={lambda_theory*1000:.3f} mm, num={lambda_num*1000:.3f} mm, err={err:.2f}%  ->  {status}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, (freq, _, _, _, _, profile, peaks) in zip(axes, results):
        x = np.arange(NX) * DX * 1000
        ax.plot(x, profile, 'b-', lw=0.8)
        ax.plot(peaks * DX * 1000, profile[peaks], 'ro', ms=4)
        ax.set_xlabel('x (mm)'); ax.set_ylabel('p'); ax.set_title(f'{freq/1000:.1f} kHz')
        ax.grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / 'test_06_phase_velocity.png', dpi=150)
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
    results = {}
    for test_name in TESTS_TO_RUN:
        if test_name == 'pulse_arrival':
            results[test_name] = test_pulse_arrival()
        elif test_name == 'hard_wall':
            results[test_name] = test_hard_wall()
        elif test_name == 'pressure_release':
            results[test_name] = test_pressure_release()
        elif test_name == 'eigenfrequencies':
            results[test_name] = test_eigenfrequencies()
        elif test_name == 'energy_conservation':
            results[test_name] = test_energy_conservation()
        elif test_name == 'phase_velocity':
            results[test_name] = test_phase_velocity()
        else:
            print(f"Unknown test: {test_name}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        print(f"  {name:25s}  {status}")
    print("=" * 60)
    total_pass = sum(1 for s in results.values() if s == "PASS")
    total_fail = sum(1 for s in results.values() if s == "FAIL")
    print(f"TOTAL: {total_pass} PASS, {total_fail} FAIL")
