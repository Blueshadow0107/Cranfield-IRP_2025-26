"""
Minimal 3D extension of the light-sensitive Oregonator BZ solver.

This is a feasibility proof-of-concept, not production code.  It keeps the
same Tyson-Fife Oregonator kinetics and adaptive reaction subcycling as
``rd_core.py`` but replaces the 2D tensor-diffusion discretisation with a
simple scalar 7-point Laplacian in 3D.

Governing equations (Tyson & Fife scaling)::

    du/dt = (1/eps) * [u - u^2 - (f*v + phi) * (u - q)/(u + q)] + Du * laplacian(u)
    dv/dt = u - v + Dv * laplacian(v)

Numerics
--------
- Operator splitting: explicit diffusion step, then adaptively subcycled
  explicit reaction integration over dt.
- Scalar isotropic diffusion with the 7-point Laplacian.
- No-flux (Neumann) on all six domain faces by construction (no exterior faces).
- Walls are inert cells (u=v=0) and no flux crosses any face touching a wall.
- Stability limit for explicit diffusion in 3D: Du*dt/dx^2 <= 1/6.

Array layout: u[i, j, k] with i = x, j = y, k = z.
"""

import numpy as np


class RDSubstrate3D:
    """3D light-sensitive Oregonator solver with scalar diffusion."""

    def __init__(self, nx=64, ny=64, nz=64, dx=1.0, dt=0.05,
                 eps=0.0501, q=0.002, f=1.4375, Du=1.0, Dv=0.0,
                 phi=0.010, clamp_rest=(0.0, 0.0)):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dt = dx, dt
        self.eps, self.q, self.f = eps, q, f
        self.Du, self.Dv = Du, Dv

        self.u = np.zeros((nx, ny, nz), dtype=float)
        self.v = np.zeros((nx, ny, nz), dtype=float)
        self.wall = np.zeros((nx, ny, nz), dtype=bool)
        self.phi = float(phi)

        self.ports = {}
        self.probes = {}
        self._holds = []
        self._clamped = set()
        self._clamp_u, self._clamp_v = float(clamp_rest[0]), float(clamp_rest[1])
        self.t = 0
        self._n_react_sub = 0

        # warn if explicit diffusion limit is violated
        if Du * dt / dx**2 > 1.0 / 6.0:
            print(f'[rd_core_3d] WARNING: Du*dt/dx^2 = {Du*dt/dx**2:.4f} '
                  f'exceeds 3D explicit stability limit 1/6')

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def set_walls(self, mask):
        """Set wall cells.  Walls are inert and block diffusive flux."""
        mask = np.asarray(mask, dtype=bool)
        assert mask.shape == (self.nx, self.ny, self.nz), 'wall mask shape mismatch'
        self.wall = mask
        self.u[self.wall] = 0.0
        self.v[self.wall] = 0.0

    def set_phi(self, field):
        """Set light-suppression field phi (scalar or 3D array)."""
        field = np.asarray(field, dtype=float)
        if field.ndim == 0:
            self.phi = float(field)
        else:
            assert field.shape == (self.nx, self.ny, self.nz), 'phi shape mismatch'
            self.phi = field

    def set_rest(self, u_rest, v_rest):
        """Set between-pulse port clamp values."""
        self._clamp_u, self._clamp_v = float(u_rest), float(v_rest)

    def add_port(self, name, mask):
        """Register an input injection zone (3D bool mask)."""
        mask = np.asarray(mask, dtype=bool)
        assert mask.shape == (self.nx, self.ny, self.nz), 'port mask shape mismatch'
        self.ports[name] = mask

    def add_probe(self, name, mask):
        """Register a readout zone; mean u is recorded every step."""
        mask = np.asarray(mask, dtype=bool)
        assert mask.shape == (self.nx, self.ny, self.nz), 'probe mask shape mismatch'
        assert mask.any(), f'probe {name!r} is empty'
        self.probes[name] = mask

    def fire(self, port, value=0.8, v_value=0.2, duration=30):
        """Hold the named port at (value, v_value) for `duration` steps."""
        self._holds.append((self.t, self.t + duration,
                            self.ports[port], value, v_value))
        self._clamped.add(port)

    # ------------------------------------------------------------------
    # Numerics
    # ------------------------------------------------------------------
    def _diffuse(self, c, D):
        """Scalar 7-point Laplacian with wall no-flux and domain Neumann BCs."""
        if D == 0.0:
            return np.zeros_like(c)
        dx = self.dx
        div = np.zeros_like(c)

        # x-faces: shape (nx-1, ny, nz)
        fx = D * (c[1:, :, :] - c[:-1, :, :]) / dx
        fx[self.wall[:-1, :, :] | self.wall[1:, :, :]] = 0.0
        div[:-1, :, :] += fx / dx
        div[1:, :, :] -= fx / dx

        # y-faces: shape (nx, ny-1, nz)
        fy = D * (c[:, 1:, :] - c[:, :-1, :]) / dx
        fy[self.wall[:, :-1, :] | self.wall[:, 1:, :]] = 0.0
        div[:, :-1, :] += fy / dx
        div[:, 1:, :] -= fy / dx

        # z-faces: shape (nx, ny, nz-1)
        fz = D * (c[:, :, 1:] - c[:, :, :-1]) / dx
        fz[self.wall[:, :, :-1] | self.wall[:, :, 1:]] = 0.0
        div[:, :, :-1] += fz / dx
        div[:, :, 1:] -= fz / dx

        return div

    def _react_rates(self, u, v):
        """Oregonator reaction rate and its diagonal Jacobian d(du/dt)/du."""
        s = self.f * v + self.phi
        g = (u - self.q) / (u + self.q)
        react = (u - u**2 - s * g) / self.eps
        jac = (1.0 - 2.0 * u - s * (2.0 * self.q / (u + self.q)**2)) / self.eps
        return react, jac

    def _react_adaptive(self, dt):
        """Adaptive reaction subcycling (identical logic to rd_core.py)."""
        STAB, ACC = 0.5, 0.1
        MAXSUB = 200000
        u, v = self.u, self.v
        wall = self.wall if self.wall.any() else None
        u_dom = -0.5 * self.q
        t_acc = 0.0
        h = dt / 8.0
        n_sub = 0
        while t_acc < dt - 1e-15:
            react, jac = self._react_rates(u, v)
            neg_jac = -jac
            if wall is not None:
                neg_jac = np.where(wall, 0.0, neg_jac)
                rate_u = np.where(wall, 0.0, np.abs(react))
                rate_v = np.where(wall, 0.0, np.abs(u - v))
            else:
                rate_u = np.abs(react)
                rate_v = np.abs(u - v)
            lam = float(neg_jac.max())
            if lam > 0.0:
                h = min(h, STAB / lam)
            rate = max(float(rate_u.max()), float(rate_v.max()))
            if rate > 0.0:
                h = min(h, ACC / rate)
            falling = (react < 0.0) if wall is None else (react < 0.0) & ~wall
            if np.any(u[falling] <= u_dom):
                raise FloatingPointError(
                    f'[rd_core_3d] u left kinetic domain at step {self.t}')
            if falling.any():
                denom = np.where(falling, -react, 1.0)
                ratios = (u - u_dom) / denom
                ratios = np.where(falling, ratios, np.inf)
                h_dom = float(ratios.min())
                h = min(h, h_dom)
            h = min(h, dt - t_acc)
            u += h * react
            v += h * (u - v)
            t_acc += h
            n_sub += 1
            if n_sub > MAXSUB:
                raise FloatingPointError(
                    f'[rd_core_3d] reaction subcycling exceeded {MAXSUB} '
                    f'substeps at step {self.t}')
            h *= 4.0
        self._n_react_sub = n_sub

    def _step(self):
        u, v = self.u, self.v
        u += self.dt * self._diffuse(u, self.Du)
        v += self.dt * self._diffuse(v, self.Dv)
        self._react_adaptive(self.dt)
        if self.wall.any():
            u[self.wall] = 0.0
            v[self.wall] = 0.0
        self._guard()

    def _guard(self):
        """Raise on NaN/Inf or absurd magnitude."""
        u, v = self.u, self.v
        bad = not (np.isfinite(u).all() and np.isfinite(v).all())
        um = float(np.abs(u).max()) if not bad else float('nan')
        vm = float(np.abs(v).max()) if not bad else float('nan')
        if bad or um > 10.0 or vm > 10.0:
            raise FloatingPointError(
                f'[rd_core_3d] blow-up at step {self.t}: |u|max={um:.3g}, '
                f'|v|max={vm:.3g}')

    # ------------------------------------------------------------------
    # Time loop
    # ------------------------------------------------------------------
    def run(self, nsteps):
        """Advance `nsteps` steps, applying scheduled port holds.

        Returns dict with keys 't' and one entry per probe name.
        """
        out = {'t': self.dt * (self.t + np.arange(1, nsteps + 1))}
        series = {name: np.empty(nsteps) for name in self.probes}
        for n in range(nsteps):
            step_idx = self.t
            active_masks = set()
            for (t0, t1, mask, uval, vval) in self._holds:
                if t0 <= step_idx < t1:
                    self.u[mask] = uval
                    self.v[mask] = vval
                    active_masks.add(id(mask))
            for pname in self._clamped:
                pmask = self.ports[pname]
                if id(pmask) not in active_masks:
                    self.u[pmask] = self._clamp_u
                    self.v[pmask] = self._clamp_v
            self._step()
            self.t += 1
            for name, pmask in self.probes.items():
                series[name][n] = self.u[pmask].mean()
        out.update(series)
        return out
