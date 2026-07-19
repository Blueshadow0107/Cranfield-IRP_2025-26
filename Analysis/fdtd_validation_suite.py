"""
================================================================================
FDTD VALIDATION SUITE — All 6 Tests in One File
================================================================================

Run all validation tests sequentially:
    python fdtd_validation_suite.py

Each test:
    1. Configures the solver for that specific case
    2. Runs the simulation
    3. Analyses against theory
    4. Prints PASS/FAIL with error metrics
    5. Auto-saves a figure

Modify TESTS_TO_RUN below to select which tests execute.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks
from pathlib import Path

# =============================================================================
# SECTION 0: SELECT WHICH TESTS TO RUN
# =============================================================================

TESTS_TO_RUN = [
    'pulse_arrival',
    'hard_wall',
    'pressure_release',
    'eigenfrequencies',
    'energy_conservation',
    'phase_velocity',
]

FIG_DIR = Path(__file__).parent / 'figures'
FIG_DIR.mkdir(exist_ok=True)


# =============================================================================
# SECTION 1: CORE SOLVER (reusable for all tests)
# =============================================================================

class FDTDSolver:
    """Minimal FDTD solver — no frills, just physics."""

    def __init__(self, nx, ny, dx, c0=343.0, rho0=1.225):
        self.nx, self.ny = nx, ny
        self.dx = dx
        self.dy = dx
        self.c0 = c0
        self.rho0 = rho0

        self.p = np.zeros((nx, ny))
        self.u = np.zeros((nx + 1, ny))
        self.v = np.zeros((nx, ny + 1))

        self.c_field = None
        self.dt = None
        self.time = 0.0
        self.n_steps = 0

        self.bc = {'left': 'absorbing', 'right': 'absorbing',
                   'top': 'hard_wall', 'bottom': 'hard_wall'}
        self.source = None

    def set_c_field(self, c_field):
        self.c_field = c_field.astype(float)
        self.c2 = c_field ** 2
        c_max = float(c_field.max())
        self.dt = 0.9 * self.dx / (c_max * np.sqrt(2))

    def set_bc(self, left=None, right=None, top=None, bottom=None):
        if left: self.bc['left'] = left
        if right: self.bc['right'] = right
        if top: self.bc['top'] = top
        if bottom: self.bc['bottom'] = bottom

    def set_source(self, stype, i0, i1, **kwargs):
        self.source = {'type': stype, 'i0': i0, 'i1': i1, 'params': kwargs}

    def _src_amp(self, t):
        if self.source is None:
            return 0.0
        typ = self.source['type']
        prm = self.source['params']
        if typ == 'continuous_sine':
            return prm['amp'] * np.sin(2 * np.pi * prm['freq'] * t)
        elif typ in ('gaussian_pulse', 'broadband_pulse'):
            env = np.exp(-((t - prm['t0']) / prm['tau']) ** 2)
            return prm['amp'] * env * np.sin(2 * np.pi * prm['f0'] * (t - prm['t0']))
        return 0.0

    def _apply_bc(self):
        nx, ny = self.nx, self.ny
        dt, rho0, dx, dy = self.dt, self.rho0, self.dx, self.dy
        p, u, v = self.p, self.u, self.v
        c = self.c_field

        # Left
        bc = self.bc['left']
        if bc == 'absorbing':   u[0, :] = -p[0, :] / (rho0 * c[0, :])
        elif bc == 'hard_wall': u[0, :] = 0.0
        elif bc == 'pressure_release': u[0, :] -= (dt / rho0) * 2.0 * p[0, :] / dx

        # Right
        bc = self.bc['right']
        if bc == 'absorbing':   u[nx, :] = p[nx - 1, :] / (rho0 * c[nx - 1, :])
        elif bc == 'hard_wall': u[nx, :] = 0.0
        elif bc == 'pressure_release': u[nx, :] += (dt / rho0) * 2.0 * p[nx - 1, :] / dx

        # Bottom
        bc = self.bc['bottom']
        if bc == 'absorbing':   v[:, 0] = -p[:, 0] / (rho0 * c[:, 0])
        elif bc == 'hard_wall': v[:, 0] = 0.0
        elif bc == 'pressure_release': v[:, 0] -= (dt / rho0) * 2.0 * p[:, 0] / dy

        # Top
        bc = self.bc['top']
        if bc == 'absorbing':   v[:, ny] = p[:, ny - 1] / (rho0 * c[:, ny - 1])
        elif bc == 'hard_wall': v[:, ny] = 0.0
        elif bc == 'pressure_release': v[:, ny] += (dt / rho0) * 2.0 * p[:, ny - 1] / dy

    def step(self):
        nx, ny = self.nx, self.ny
        dt, rho0 = self.dt, self.rho0
        dx, dy = self.dx, self.dy
        p, u, v = self.p, self.u, self.v
        c2 = self.c2

        u[1:nx, :] -= (dt / rho0) * (p[1:nx, :] - p[0:nx - 1, :]) / dx
        v[:, 1:ny] -= (dt / rho0) * (p[:, 1:ny] - p[:, 0:ny - 1]) / dy
        self._apply_bc()
        p -= dt * rho0 * c2 * (
            (u[1:nx + 1, :] - u[0:nx, :]) / dx +
            (v[:, 1:ny + 1] - v[:, 0:ny]) / dy
        )
        if self.source is not None:
            s = self._src_amp(self.time)
            p[self.source['i0']:self.source['i1'], :] += s
        self.time += dt
        self.n_steps += 1

    def reset(self):
        self.p.fill(0.0)
        self.u.fill(0.0)
        self.v.fill(0.0)
        self.time = 0.0
        self.n_steps = 0

    def run(self, duration, probe_coords=None):
        n_steps = int(duration / self.dt)
        probe = [] if probe_coords is not None else None
        for _ in range(n_steps):
            self.step()
            if probe_coords is not None:
                probe.append(float(self.p[probe_coords[0], probe_coords[1]]))
        return np.array(probe) if probe is not None else None

    def compute_energy(self):
        nx, ny = self.nx, self.ny
        rho0, dx = self.rho0, self.dx
        p, u, v = self.p, self.u, self.v
        c2 = self.c2
        e_pot = np.sum(p ** 2 / (2.0 * rho0 * c2)) * dx * dx
        u_avg = 0.5 * (u[0:nx, :] ** 2 + u[1:nx + 1, :] ** 2)
        v_avg = 0.5 * (v[:, 0:ny] ** 2 + v[:, 1:ny + 1] ** 2)
        e_kin = 0.5 * rho0 * np.sum(u_avg + v_avg) * dx * dx
        return e_kin, e_pot, e_kin + e_pot


# =============================================================================
# SECTION 2: TEST FUNCTIONS
# =============================================================================

def test_pulse_arrival():
    """TEST 01: Measure pulse transit time, compare to L/c."""
    print("\n" + "=" * 60)
    print("TEST 01: Pulse Transit Time")
    print("=" * 60)

    NX, NY, DX = 400, 50, 0.002
    C0, RHO0 = 343.0, 1.225
    SRC_CENTRE = (3 + 6) / 2.0
    PROBE_I = NX - 4
    PULSE_T0, PULSE_TAU, PULSE_F0 = 0.0005, 0.0001, 6000.0

    solver = FDTDSolver(NX, NY, DX, c0=C0, rho0=RHO0)
    solver.set_c_field(np.full((NX, NY), C0))
    solver.set_bc(left='absorbing', right='absorbing', top='absorbing', bottom='absorbing')
    solver.set_source('gaussian_pulse', 3, 6, amp=1.0, t0=PULSE_T0, tau=PULSE_TAU, f0=PULSE_F0)

    probe = solver.run(0.020, probe_coords=(PROBE_I, NY // 2))
    t = np.arange(len(probe)) * solver.dt

    peak_idx = np.argmax(np.abs(probe))
    t_meas = t[peak_idx]
    t_theory = (PROBE_I - SRC_CENTRE) * DX / C0 + PULSE_T0
    err = abs(t_meas - t_theory) / t_theory * 100
    status = "PASS" if err < 3.0 else "FAIL"

    print(f"Transit (theory): {t_theory * 1000:.3f} ms")
    print(f"Transit (measured): {t_meas * 1000:.3f} ms")
    print(f"Error: {err:.2f}%  ->  {status}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t * 1000, probe, 'b-', lw=0.8)
    ax.axvline(t_theory * 1000, color='g', ls='--', label=f'theory: {t_theory * 1000:.3f} ms')
    ax.axvline(t_meas * 1000, color='r', ls='--', label=f'measured: {t_meas * 1000:.3f} ms')
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('p'); ax.set_title('Test 01: Pulse arrival')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / 'test_01_pulse_arrival.png', dpi=150)
    plt.close(fig)
    return status


def test_hard_wall():
    """TEST 02: Compare peak pressure at hard wall vs. absorbing."""
    print("\n" + "=" * 60)
    print("TEST 02: Hard Wall Reflection")
    print("=" * 60)

    NX, NY, DX = 400, 50, 0.002
    C0, RHO0 = 343.0, 1.225
    FREQ, AMP = 5000.0, 1.0
    SRC_I0, SRC_I1 = 197, 203
    PROBE_I = 0

    # Run A: hard wall left
    solver_a = FDTDSolver(NX, NY, DX, c0=C0, rho0=RHO0)
    solver_a.set_c_field(np.full((NX, NY), C0))
    solver_a.set_bc(left='hard_wall', right='absorbing', top='hard_wall', bottom='hard_wall')
    solver_a.set_source('continuous_sine', SRC_I0, SRC_I1, amp=AMP, freq=FREQ)
    probe_a = solver_a.run(0.010, probe_coords=(PROBE_I, NY // 2))
    p_max_hard = np.max(np.abs(probe_a[int(0.6 * len(probe_a)):]))

    # Run B: absorbing left
    solver_b = FDTDSolver(NX, NY, DX, c0=C0, rho0=RHO0)
    solver_b.set_c_field(np.full((NX, NY), C0))
    solver_b.set_bc(left='absorbing', right='absorbing', top='hard_wall', bottom='hard_wall')
    solver_b.set_source('continuous_sine', SRC_I0, SRC_I1, amp=AMP, freq=FREQ)
    probe_b = solver_b.run(0.010, probe_coords=(PROBE_I, NY // 2))
    p_max_abs = np.max(np.abs(probe_b[int(0.6 * len(probe_b)):]))

    ratio = p_max_hard / (p_max_abs + 1e-30)
    status = "PASS" if ratio > 1.3 else "FAIL"

    print(f"Peak (hard wall): {p_max_hard:.4f}")
    print(f"Peak (absorbing): {p_max_abs:.4f}")
    print(f"Ratio: {ratio:.3f}  ->  {status}")

    t = np.arange(len(probe_a)) * solver_a.dt
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(t * 1000, probe_a, 'b-', lw=0.5); axes[0].set_title('Hard wall'); axes[0].grid(True, alpha=0.3)
    axes[1].plot(t * 1000, probe_b, 'r-', lw=0.5); axes[1].set_title('Absorbing'); axes[1].set_xlabel('Time (ms)')
    axes[1].grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / 'test_02_hard_wall.png', dpi=150)
    plt.close(fig)
    return status


def test_pressure_release():
    """TEST 03: Quarter-wave resonator with pressure-release boundary."""
    print("\n" + "=" * 60)
    print("TEST 03: Pressure-Release Boundary")
    print("=" * 60)

    NX, NY, DX = 400, 5, 0.002
    C0 = 343.0
    Lx = NX * DX

    solver = FDTDSolver(NX, NY, DX, c0=C0)
    solver.set_c_field(np.full((NX, NY), C0))
    solver.set_bc(left='hard_wall', right='pressure_release', top='hard_wall', bottom='hard_wall')
    solver.set_source('broadband_pulse', 195, 205, amp=1.0, t0=0.0005, tau=0.0002, f0=2000.0)

    # Initial Gaussian pulse instead of continuous source
    x_idx = np.arange(NX)
    y_idx = np.arange(NY)
    X, Y = np.meshgrid(x_idx, y_idx, indexing='ij')
    solver.p = 1.0 * np.exp(-((X - NX // 2) ** 2 + (Y - NY // 2) ** 2) / (2 * 5 ** 2))

    probe = solver.run(0.050, probe_coords=(5, NY // 2))

    window = np.hanning(len(probe))
    freqs = np.fft.rfftfreq(len(probe), solver.dt)
    spectrum = np.abs(np.fft.rfft(probe * window))

    prom = np.max(spectrum) * 0.005
    peaks, _ = find_peaks(spectrum, prominence=prom, distance=5)
    peak_freqs = freqs[peaks]
    peak_freqs = peak_freqs[peak_freqs < 5000][:15]

    f_theory = (2 * np.arange(1, 21) - 1) * C0 / (4 * Lx)

    matches = []
    used = set()
    for f_num in peak_freqs:
        best_err = np.inf
        best_idx = -1
        for idx, f_th in enumerate(f_theory):
            if idx in used: continue
            err = abs(f_num - f_th)
            if err < best_err:
                best_err = err; best_idx = idx
        if best_idx >= 0 and best_err < 50.0:
            used.add(best_idx)
            err_pct = abs(f_num - f_theory[best_idx]) / f_theory[best_idx] * 100
            matches.append((best_idx + 1, f_theory[best_idx], f_num, err_pct))

    all_pass = True
    print(f"{'n':>3}  {'Theory (Hz)':>14}  {'Numeric (Hz)':>14}  {'Error %':>10}  {'Status'}")
    print("-" * 60)
    for n, f_th, f_num, err in matches[:10]:
        tol = 10.0 if f_th < 150.0 else 3.0
        st = "PASS" if err < tol else "FAIL"
        if err >= tol: all_pass = False
        print(f"{n:3d}  {f_th:14.2f}  {f_num:14.2f}  {err:10.2f}  {st}")
    print("-" * 60)
    status = "PASS" if all_pass else "FAIL"
    print(f"Overall: {status}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    t = np.arange(len(probe)) * solver.dt
    axes[0].plot(t * 1000, probe, 'b-', lw=0.5)
    axes[0].set_xlabel('Time (ms)'); axes[0].set_ylabel('p'); axes[0].set_title('Ring-down signal')
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(freqs, spectrum, 'b-', lw=0.5, alpha=0.7)
    matched_freqs = [m[2] for m in matches]
    matched_amps = [spectrum[np.argmin(np.abs(freqs - f))] for f in matched_freqs]
    axes[1].plot(matched_freqs, matched_amps, 'ro', ms=6, label='matched')
    for f_th in f_theory[:10]: axes[1].axvline(f_th, color='g', ls='--', alpha=0.4)
    axes[1].set_xlabel('Frequency (Hz)'); axes[1].set_ylabel('|P(f)|')
    axes[1].set_title('FFT: quarter-wave peaks (green = theory)')
    axes[1].set_xlim(0, 5000); axes[1].grid(True, alpha=0.3); axes[1].legend()
    fig.savefig(FIG_DIR / 'test_03_pressure_release.png', dpi=150)
    plt.close(fig)
    return status


def test_eigenfrequencies():
    """TEST 04: Closed cavity eigenmodes."""
    print("\n" + "=" * 60)
    print("TEST 04: Eigenfrequencies")
    print("=" * 60)

    NX, NY, DX = 400, 50, 0.002
    C0 = 343.0
    Lx, Ly = NX * DX, NY * DX

    solver = FDTDSolver(NX, NY, DX, c0=C0)
    solver.set_c_field(np.full((NX, NY), C0))
    solver.set_bc(left='hard_wall', right='hard_wall', top='hard_wall', bottom='hard_wall')

    X, Y = np.meshgrid(np.arange(NX), np.arange(NY), indexing='ij')
    solver.p = 1.0 * np.exp(-((X - NX // 2) ** 2 + (Y - NY // 2) ** 2) / (2 * 5 ** 2))

    probe = solver.run(0.050, probe_coords=(5, NY // 2))

    window = np.hanning(len(probe))
    freqs = np.fft.rfftfreq(len(probe), solver.dt)
    spectrum = np.abs(np.fft.rfft(probe * window))

    peaks, _ = find_peaks(spectrum, height=np.max(spectrum) * 0.005, distance=30)
    peak_freqs = freqs[peaks]
    peak_freqs = peak_freqs[peak_freqs < 4000][:10]

    modes = []
    for n in range(20):
        for m in range(20):
            if n == 0 and m == 0: continue
            f_th = 0.5 * C0 * np.sqrt((n / Lx) ** 2 + (m / Ly) ** 2)
            if f_th < 4000: modes.append((f_th, n, m))
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
    t = np.arange(len(probe)) * solver.dt
    axes[0].plot(t * 1000, probe, 'b-', lw=0.5)
    axes[0].set_xlabel('Time (ms)'); axes[0].set_ylabel('p'); axes[0].set_title('Ring-down')
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(freqs, spectrum, 'b-', lw=0.5, alpha=0.7)
    axes[1].plot(peak_freqs, spectrum[peaks][:len(peak_freqs)], 'ro', ms=5)
    for f_th, n, m in modes[:15]: axes[1].axvline(f_th, color='g', ls='--', alpha=0.3)
    axes[1].set_xlabel('Frequency (Hz)'); axes[1].set_ylabel('|P(f)|')
    axes[1].set_title('FFT: cavity modes (green = theory)')
    axes[1].set_xlim(0, 4000); axes[1].grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / 'test_04_eigenfrequencies.png', dpi=150)
    plt.close(fig)
    return status


def test_energy_conservation():
    """TEST 05: Energy drift in closed lossless cavity."""
    print("\n" + "=" * 60)
    print("TEST 05: Energy Conservation")
    print("=" * 60)

    NX, NY, DX = 400, 50, 0.002
    C0 = 343.0

    solver = FDTDSolver(NX, NY, DX, c0=C0)
    solver.set_c_field(np.full((NX, NY), C0))
    solver.set_bc(left='hard_wall', right='hard_wall', top='hard_wall', bottom='hard_wall')

    X, Y = np.meshgrid(np.arange(NX), np.arange(NY), indexing='ij')
    solver.p = 1.0 * np.exp(-((X - NX // 2) ** 2 + (Y - NY // 2) ** 2) / (2 * 5 ** 2))

    n_steps = int(0.020 / solver.dt)
    energies = []
    for _ in range(n_steps):
        solver.step()
        energies.append(solver.compute_energy()[2])
    energies = np.array(energies)
    t = np.arange(n_steps) * solver.dt

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
    axes[0].set_ylabel('Energy'); axes[0].set_title('Total energy vs time')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t * 1000, (energies / e_initial - 1) * 100, 'k-', lw=0.8)
    axes[1].axhline(0, color='g', ls='--', alpha=0.5)
    axes[1].set_xlabel('Time (ms)'); axes[1].set_ylabel('Drift (%)')
    axes[1].set_title(f'Drift: {drift:.2f}%')
    axes[1].grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / 'test_05_energy_conservation.png', dpi=150)
    plt.close(fig)
    return status


def test_phase_velocity():
    """TEST 06: Wavelength measurement at multiple frequencies."""
    print("\n" + "=" * 60)
    print("TEST 06: Phase Velocity")
    print("=" * 60)

    NX, NY, DX = 400, 50, 0.002
    C0 = 343.0
    freqs_test = [2000.0, 5000.0, 8000.0, 10000.0]

    results = []
    for freq in freqs_test:
        solver = FDTDSolver(NX, NY, DX, c0=C0)
        solver.set_c_field(np.full((NX, NY), C0))
        solver.set_bc(left='absorbing', right='absorbing', top='hard_wall', bottom='hard_wall')
        solver.set_source('continuous_sine', 3, 6, amp=1.0, freq=freq)

        n_steps = int(0.015 / solver.dt)
        discard = int(0.6 * n_steps)
        profiles = []
        for n in range(n_steps):
            solver.step()
            if n >= discard:
                profiles.append(solver.p[:, NY // 2].copy())

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
        print(f"f = {freq / 1000:.1f} kHz: theory={lambda_theory * 1000:.3f} mm, num={lambda_num * 1000:.3f} mm, err={err:.2f}%  ->  {status}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, (freq, _, _, _, _, profile, peaks) in zip(axes, results):
        x = np.arange(NX) * DX * 1000
        ax.plot(x, profile, 'b-', lw=0.8)
        ax.plot(peaks * DX * 1000, profile[peaks], 'ro', ms=4)
        ax.set_xlabel('x (mm)'); ax.set_ylabel('p')
        ax.set_title(f'{freq / 1000:.1f} kHz')
        ax.grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / 'test_06_phase_velocity.png', dpi=150)
    plt.close(fig)

    all_pass = all(r[4] == "PASS" for r in results)
    return "PASS" if all_pass else "FAIL"


# =============================================================================
# SECTION 3: MAIN — Run Selected Tests
# =============================================================================

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
