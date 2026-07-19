#!/usr/bin/env python3
"""
porous_fdtd_validation_v1.py
============================
Validation suite for the rigid-frame Zwikker-Kosten porous FDTD solver
in volume-velocity form.

Tests are porosity-focused and run quickly. Each test prints PASS/FAIL
and saves a diagnostic figure.

Tests
-----
1. Free-air pulse transit           : phi=1,  alpha=1,  sigma=0
2. Uniform porous slab slowdown     : phi=0.5, alpha=1,  sigma=0
3. Porosity-step reflection         : phi=1 -> phi=0.5, sigma=0
4. Flow-resistivity damping         : phi=1,  alpha=1,  sigma>0
5. Energy conservation (lossless)   : phi=1,  alpha=1,  sigma=0, hard walls
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# Core solver class
# =============================================================================
class PorousFDTDSolver:
    """2D rigid-frame ZK porous acoustic FDTD in volume-velocity form."""

    def __init__(self, nx, ny, dx, c0=343.0, rho0=1.225):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dx
        self.c0 = c0
        self.rho0 = rho0
        self.K0 = rho0 * c0 ** 2

        self.phi = None
        self.alpha_inf = None
        self.sigma = None

        self.c_eff = None
        self.Z_V = None
        self.a_coeff = None
        self.b_coeff = None
        self.dt = None

        self.p = np.zeros((nx, ny))
        self.Vx = np.zeros((nx + 1, ny))
        self.Vy = np.zeros((nx, ny + 1))

        self.bc = {
            'left': 'absorbing',
            'right': 'absorbing',
            'bottom': 'hard_wall',
            'top': 'hard_wall'
        }

        self._src_type = 'none'
        self._src_amp = 1.0
        self._src_freq = 5000.0
        self._src_i0 = 20
        self._src_i1 = 23
        self._pulse_t0 = 0.0005
        self._pulse_tau = 0.0001
        self._pulse_f0 = 6000.0

    def set_params(self, phi, alpha_inf, sigma):
        """Set parameter fields."""
        self.phi = np.asarray(phi, dtype=float)
        self.alpha_inf = np.asarray(alpha_inf, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self._update_derived()

    def set_uniform_params(self, phi_val, alpha_val, sigma_val):
        """Uniform parameters everywhere."""
        phi = np.full((self.nx, self.ny), float(phi_val))
        alpha = np.full((self.nx, self.ny), float(alpha_val))
        sigma = np.full((self.nx, self.ny), float(sigma_val))
        self.set_params(phi, alpha, sigma)

    def set_region(self, mask, phi_val, alpha_val, sigma_val):
        """Override parameters inside a boolean mask."""
        self.phi = self.phi.copy()
        self.alpha_inf = self.alpha_inf.copy()
        self.sigma = self.sigma.copy()
        self.phi[mask] = float(phi_val)
        self.alpha_inf[mask] = float(alpha_val)
        self.sigma[mask] = float(sigma_val)
        self._update_derived()

    def _update_derived(self):
        """Recompute derived quantities and stable time step."""
        self.c_eff = self.c0 / np.sqrt(self.phi * self.alpha_inf)
        self.Z_V = self.rho0 * self.c0 * np.sqrt(self.alpha_inf) / (self.phi ** 1.5)
        c_max = float(self.c_eff.max())
        self.dt = 0.9 * self.dx / (c_max * np.sqrt(2))
        self.a_coeff = self.phi / (self.rho0 * self.alpha_inf)
        self.b_coeff = self.sigma * self.phi * self.dt / (self.rho0 * self.alpha_inf)

    def set_bc(self, left=None, right=None, bottom=None, top=None):
        for side, val in [('left', left), ('right', right),
                          ('bottom', bottom), ('top', top)]:
            if val is not None:
                self.bc[side] = val

    def set_source(self, stype, i0, i1, **kwargs):
        self._src_type = stype
        self._src_i0 = i0
        self._src_i1 = i1
        for k, v in kwargs.items():
            setattr(self, f'_src_{k}', v) if not k.startswith('pulse_') else setattr(self, f'_{k}', v)
        # Handle pulse kwargs explicitly
        if 'pulse_t0' in kwargs:
            self._pulse_t0 = kwargs['pulse_t0']
        if 'pulse_tau' in kwargs:
            self._pulse_tau = kwargs['pulse_tau']
        if 'pulse_f0' in kwargs:
            self._pulse_f0 = kwargs['pulse_f0']

    def _src_value(self, t):
        if self._src_type == 'none':
            return 0.0
        elif self._src_type == 'sine':
            return self._src_amp * np.sin(2.0 * np.pi * self._src_freq * t)
        elif self._src_type == 'pulse':
            env = np.exp(-((t - self._pulse_t0) / self._pulse_tau) ** 2)
            return self._src_amp * env * np.sin(2.0 * np.pi * self._pulse_f0 * (t - self._pulse_t0))
        return 0.0

    def _apply_bc(self):
        nx, ny = self.nx, self.ny
        dt, dx, dy = self.dt, self.dx, self.dy
        p, Vx, Vy = self.p, self.Vx, self.Vy
        Z = self.Z_V
        a = self.a_coeff

        if self.bc['left'] == 'absorbing':
            Vx[0, :] = -p[0, :] / Z[0, :]
        elif self.bc['left'] == 'hard_wall':
            Vx[0, :] = 0.0
        elif self.bc['left'] == 'pressure_release':
            Vx[0, :] -= dt * a[0, :] * 2.0 * p[0, :] / dx

        if self.bc['right'] == 'absorbing':
            Vx[nx, :] = p[nx - 1, :] / Z[nx - 1, :]
        elif self.bc['right'] == 'hard_wall':
            Vx[nx, :] = 0.0
        elif self.bc['right'] == 'pressure_release':
            Vx[nx, :] += dt * a[nx - 1, :] * 2.0 * p[nx - 1, :] / dx

        if self.bc['bottom'] == 'absorbing':
            Vy[:, 0] = -p[:, 0] / Z[:, 0]
        elif self.bc['bottom'] == 'hard_wall':
            Vy[:, 0] = 0.0
        elif self.bc['bottom'] == 'pressure_release':
            Vy[:, 0] -= dt * a[:, 0] * 2.0 * p[:, 0] / dy

        if self.bc['top'] == 'absorbing':
            Vy[:, ny] = p[:, ny - 1] / Z[:, ny - 1]
        elif self.bc['top'] == 'hard_wall':
            Vy[:, ny] = 0.0
        elif self.bc['top'] == 'pressure_release':
            Vy[:, ny] += dt * a[:, ny - 1] * 2.0 * p[:, ny - 1, :] / dy

    def step(self, t):
        nx, ny = self.nx, self.ny
        dt, dx = self.dt, self.dx
        p, Vx, Vy = self.p, self.Vx, self.Vy
        a = self.a_coeff
        b = self.b_coeff

        dpdx = (p[1:nx, :] - p[0:nx - 1, :]) / dx
        dpdy = (p[:, 1:ny] - p[:, 0:ny - 1]) / dx

        Vx[1:nx, :] = (Vx[1:nx, :] - dt * a[1:nx, :] * dpdx) / (1.0 + b[1:nx, :])
        Vy[:, 1:ny] = (Vy[:, 1:ny] - dt * a[:, 1:ny] * dpdy) / (1.0 + b[:, 1:ny])

        self._apply_bc()

        dVx_dx = (Vx[1:nx + 1, :] - Vx[0:nx, :]) / dx
        dVy_dy = (Vy[:, 1:ny + 1] - Vy[:, 0:ny]) / dx
        p -= dt * self.K0 * (dVx_dx + dVy_dy)

        p[self._src_i0:self._src_i1, :] += self._src_value(t)

    def reset_fields(self):
        self.p.fill(0.0)
        self.Vx.fill(0.0)
        self.Vy.fill(0.0)

    def run(self, duration, probe_coords=None):
        n_steps = int(duration / self.dt)
        probe = [] if probe_coords is not None else None
        for n in range(n_steps):
            self.step(n * self.dt)
            if probe is not None:
                probe.append(float(self.p[probe_coords]))
        return np.array(probe) if probe is not None else None

    def compute_energy(self):
        e_pot = np.sum(self.p ** 2 / (2.0 * self.K0)) * self.dx ** 2
        Vx_avg = 0.5 * (self.Vx[0:self.nx, :] ** 2 + self.Vx[1:self.nx + 1, :] ** 2)
        Vy_avg = 0.5 * (self.Vy[:, 0:self.ny] ** 2 + self.Vy[:, 1:self.ny + 1] ** 2)
        rho_macro = self.rho0 * self.alpha_inf / self.phi
        e_kin = 0.5 * np.sum(rho_macro * (Vx_avg + Vy_avg)) * self.dx ** 2
        return e_kin, e_pot, e_kin + e_pot


# =============================================================================
# Test harness
# =============================================================================
FIG_DIR = Path(__file__).parent / 'figures'
FIG_DIR.mkdir(exist_ok=True)


def _plot_probe(t, probe, t_theory, title, filename):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t * 1000, probe, 'b-', lw=0.8)
    ax.axvline(t_theory * 1000, color='g', ls='--', label=f'theory: {t_theory*1000:.3f} ms')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('p [Pa]')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filename}")


def test_01_free_air_pulse():
    print("\nTEST 01: Free-air pulse transit")
    solver = PorousFDTDSolver(400, 50, 0.002)
    solver.set_uniform_params(1.0, 1.0, 0.0)
    solver.set_bc(left='absorbing', right='absorbing', bottom='absorbing', top='absorbing')
    solver.set_source('pulse', 20, 23, amp=1.0, pulse_t0=0.0005, pulse_tau=0.0001, pulse_f0=6000.0)
    solver.reset_fields()

    probe_i = 396
    probe = solver.run(0.020, probe_coords=(probe_i, 25))
    t = np.arange(len(probe)) * solver.dt

    src_centre = 0.5 * (solver._src_i0 + solver._src_i1)
    t_theory = (probe_i - src_centre) * solver.dx / solver.c0 + solver._pulse_t0
    t_peak = t[np.argmax(np.abs(probe))]
    err = abs(t_peak - t_theory) / t_theory * 100
    status = "PASS" if err < 3.0 else "FAIL"

    print(f"  Theoretical transit: {t_theory*1000:.3f} ms")
    print(f"  Measured transit:    {t_peak*1000:.3f} ms")
    print(f"  Error:               {err:.2f}%  ->  {status}")

    _plot_probe(t, probe, t_theory, 'Test 01: Free-air pulse transit',
                FIG_DIR / 'porous_val_01_free_air.png')
    return status


def test_02_porous_slab_slowdown():
    print("\nTEST 02: Uniform porous slab slowdown")
    phi_test = 0.5
    # Volume-velocity ZK gives c_eff = c0 * sqrt(phi / alpha)
    c_theory = 343.0 * np.sqrt(phi_test)

    solver = PorousFDTDSolver(400, 50, 0.002)
    solver.set_uniform_params(phi_test, 1.0, 0.0)
    solver.set_bc(left='absorbing', right='absorbing', bottom='absorbing', top='absorbing')
    solver.set_source('pulse', 20, 23, amp=1.0, pulse_t0=0.0005, pulse_tau=0.0001, pulse_f0=6000.0)
    solver.reset_fields()

    probe_i = 396
    probe = solver.run(0.020, probe_coords=(probe_i, 25))
    t = np.arange(len(probe)) * solver.dt

    src_centre = 0.5 * (solver._src_i0 + solver._src_i1)
    t_theory = (probe_i - src_centre) * solver.dx / c_theory + solver._pulse_t0
    t_peak = t[np.argmax(np.abs(probe))]
    err = abs(t_peak - t_theory) / t_theory * 100
    status = "PASS" if err < 3.0 else "FAIL"

    print(f"  Theoretical c_eff:   {c_theory:.1f} m/s")
    print(f"  Theoretical transit: {t_theory*1000:.3f} ms")
    print(f"  Measured transit:    {t_peak*1000:.3f} ms")
    print(f"  Error:               {err:.2f}%  ->  {status}")

    _plot_probe(t, probe, t_theory, f'Test 02: phi={phi_test} slab slowdown',
                FIG_DIR / 'porous_val_02_slab_slowdown.png')
    return status


def test_03_porosity_step():
    print("\nTEST 03: Porosity-step reflection")
    phi1, phi2 = 1.0, 0.5
    alpha1, alpha2 = 1.0, 1.0
    sigma = 0.0
    freq = 5000.0

    solver = PorousFDTDSolver(400, 50, 0.002)
    solver.set_uniform_params(phi1, alpha1, sigma)
    mask = np.zeros((solver.nx, solver.ny), dtype=bool)
    mask[solver.nx // 2:, :] = True
    solver.set_region(mask, phi2, alpha2, sigma)
    solver.set_bc(left='absorbing', right='absorbing', bottom='hard_wall', top='hard_wall')
    solver.set_source('sine', 20, 23, amp=1.0, freq=freq)
    solver.reset_fields()

    probe = solver.run(0.015, probe_coords=(solver.nx // 2 - 20, 25))
    t = np.arange(len(probe)) * solver.dt

    # Steady-state amplitude of reflected wave
    steady = probe[int(0.6 * len(probe)):]
    amp_reflected = np.max(np.abs(steady))

    # Analytic reflection coefficient
    Z1 = solver.rho0 * solver.c0 * np.sqrt(alpha1) / (phi1 ** 1.5)
    Z2 = solver.rho0 * solver.c0 * np.sqrt(alpha2) / (phi2 ** 1.5)
    R_analytic = abs((Z2 - Z1) / (Z2 + Z1))

    # Incident amplitude estimated from a run with no step
    solver_inc = PorousFDTDSolver(400, 50, 0.002)
    solver_inc.set_uniform_params(phi1, alpha1, sigma)
    solver_inc.set_bc(left='absorbing', right='absorbing', bottom='hard_wall', top='hard_wall')
    solver_inc.set_source('sine', 20, 23, amp=1.0, freq=freq)
    solver_inc.reset_fields()
    probe_inc = solver_inc.run(0.015, probe_coords=(solver_inc.nx // 2 - 20, 25))
    amp_incident = np.max(np.abs(probe_inc[int(0.6 * len(probe_inc)):]))

    R_measured = amp_reflected / amp_incident if amp_incident > 1e-12 else np.nan
    err = abs(R_measured - R_analytic) / (R_analytic + 1e-12) * 100
    status = "PASS" if err < 10.0 else "FAIL"

    print(f"  Z1 = {Z1:.3f}, Z2 = {Z2:.3f}")
    print(f"  Analytic |R| = {R_analytic:.4f}")
    print(f"  Measured |R| = {R_measured:.4f}")
    print(f"  Error:        {err:.2f}%  ->  {status}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t * 1000, probe, 'b-', lw=0.5, label='with step')
    ax.plot(np.arange(len(probe_inc)) * solver_inc.dt * 1000, probe_inc, 'r-', lw=0.5, alpha=0.5, label='no step (incident)')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('p [Pa]')
    ax.set_title('Test 03: Porosity-step reflection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / 'porous_val_03_step_reflection.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'porous_val_03_step_reflection.png'}")
    return status


def test_04_damping():
    print("\nTEST 04: Flow-resistivity damping")
    phi = 1.0
    sigma_val = 5000.0
    freq = 4000.0

    solver = PorousFDTDSolver(200, 40, 0.002)
    solver.set_uniform_params(phi, 1.0, sigma_val)
    solver.set_bc(left='absorbing', right='absorbing', bottom='hard_wall', top='hard_wall')
    solver.set_source('sine', 10, 13, amp=1.0, freq=freq)
    solver.reset_fields()

    probe = solver.run(0.010, probe_coords=(150, 20))
    t = np.arange(len(probe)) * solver.dt

    # Fit exponential envelope to the late-time envelope
    analytic = np.abs(probe)
    # Smooth a bit
    window = max(1, int(0.0005 / solver.dt))
    if window > 1:
        analytic = np.convolve(analytic, np.ones(window) / window, mode='same')

    # Use late portion after source has reached steady state
    idx_start = int(0.003 / solver.dt)
    t_fit = t[idx_start:]
    env_fit = analytic[idx_start:]
    # log-linear fit: log(A) = log(A0) - gamma * t
    coeffs = np.polyfit(t_fit, np.log(env_fit + 1e-12), 1)
    gamma_measured = -coeffs[0]

    # Theoretical damping rate for plane wave in ZK medium
    # From (rho0*alpha/phi) dV/dt + sigma*V = -grad(p) and dp/dt + K0 div(V) = 0
    # For small sigma, gamma ≈ sigma * phi^2 / (2 * rho0 * alpha) * (c_eff / c0^2) ...
    # Simpler: use the complex wavenumber from the frequency-domain ZK model.
    rho_eff = solver.rho0 * solver.alpha_inf[0, 0] / phi
    k0 = 2 * np.pi * freq / solver.c_eff[0, 0]
    gamma_theory = sigma_val * phi ** 2 / (2.0 * rho_eff) * (solver.c_eff[0, 0] / solver.c0 ** 2)

    err = abs(gamma_measured - gamma_theory) / (gamma_theory + 1e-12) * 100
    status = "PASS" if err < 30.0 else "FAIL"

    print(f"  Theoretical damping rate: {gamma_theory:.2f} Np/s")
    print(f"  Measured damping rate:    {gamma_measured:.2f} Np/s")
    print(f"  Error:                    {err:.2f}%  ->  {status}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t * 1000, probe, 'b-', lw=0.5, label='probe')
    ax.plot(t * 1000, analytic, 'r-', lw=1.0, label='envelope')
    ax.plot(t_fit * 1000, np.exp(coeffs[1]) * np.exp(-gamma_measured * t_fit), 'g--', lw=1.5, label='fit')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('p [Pa]')
    ax.set_title('Test 04: Flow-resistivity damping')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / 'porous_val_04_damping.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'porous_val_04_damping.png'}")
    return status


def test_05_energy_conservation():
    print("\nTEST 05: Energy conservation (lossless)")
    solver = PorousFDTDSolver(200, 40, 0.002)
    solver.set_uniform_params(1.0, 1.0, 0.0)
    solver.set_bc(left='hard_wall', right='hard_wall', bottom='hard_wall', top='hard_wall')
    solver.reset_fields()

    # Initial Gaussian pressure pulse
    X, Y = np.meshgrid(np.arange(solver.nx), np.arange(solver.ny), indexing='ij')
    solver.p = 1.0 * np.exp(-((X - solver.nx // 2) ** 2 + (Y - solver.ny // 2) ** 2) / (2 * 5 ** 2))

    n_steps = int(0.020 / solver.dt)
    energies = []
    for _ in range(n_steps):
        solver.step(0.0)
        energies.append(solver.compute_energy()[2])
    energies = np.array(energies)
    t = np.arange(n_steps) * solver.dt

    drift = (energies.max() - energies.min()) / energies[0] * 100
    status = "PASS" if drift < 8.0 else "FAIL"

    print(f"  Initial energy: {energies[0]:.6e}")
    print(f"  Max energy:     {energies.max():.6e}")
    print(f"  Min energy:     {energies.min():.6e}")
    print(f"  Drift:          {drift:.2f}%  ->  {status}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t * 1000, (energies / energies[0] - 1) * 100, 'k-', lw=0.8)
    ax.axhline(0, color='g', ls='--', alpha=0.5)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Energy drift [%]')
    ax.set_title('Test 05: Energy conservation')
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / 'porous_val_05_energy.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'porous_val_05_energy.png'}")
    return status


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("POROUS FDTD VALIDATION SUITE V1")
    print("=" * 60)

    results = {}
    results['01_free_air_pulse'] = test_01_free_air_pulse()
    results['02_porous_slab'] = test_02_porous_slab_slowdown()
    results['03_porosity_step'] = test_03_porosity_step()
    results['04_damping'] = test_04_damping()
    results['05_energy'] = test_05_energy_conservation()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        print(f"  {name:30s}  {status}")
    n_pass = sum(1 for s in results.values() if s == 'PASS')
    n_fail = sum(1 for s in results.values() if s == 'FAIL')
    print("=" * 60)
    print(f"TOTAL: {n_pass} PASS, {n_fail} FAIL")
