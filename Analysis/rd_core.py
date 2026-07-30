"""
Reusable reaction-diffusion (RD) solver core for the light-sensitive
Oregonator model of Belousov-Zhabotinsky (BZ) chemistry.

Equations (Tyson & Fife scaling), identical to the validated reference
scripts ``oregonator_bz_demo.py`` / ``oregonator_timing_sweep.py``::

    du/dt = (1/eps) * [u - u^2 - (f*v + phi) * (u - q)/(u + q)] + div(D . grad u)
    dv/dt = u - v + Dv * laplacian(v)

u  ~ HBrO2 (activator), v ~ catalyst oxidation level (inhibitor),
phi ~ light-suppression field (photosensitive BZ variant).

Numerics
--------
- Operator-split explicit Euler: one explicit diffusion step, then an
  adaptively subcycled explicit reaction integration over dt (see below).
- Diffusion in divergence (flux) form on a cell-centred grid.  Face fluxes
  use arithmetic-mean face tensors; the cross derivative on a face is the
  averaged 4-point centred difference.  Reduces EXACTLY to the plain
  5-point Laplacian when Dxx = Dyy = const and Dxy = 0.
- Walls: bool mask of inert cells.  No reaction in wall cells and NO flux
  crosses any face touching a wall cell (true no-flux, not edge-copy).
- Array layout follows the reference scripts: u[i, j] with i = x index,
  j = y index.

NO CLIPPING (de-hack, 2026-07-20)
---------------------------------
The reference scripts (oregonator_bz_demo.py, barkley_bz_demo.py) applied
np.clip(u, ...) / np.clip(v, ...) after every Euler step.  That clip is
LOAD-BEARING, not cosmetic: point kinetics at eps=0.05, q=0.002, f=1.4
fire ONCE in 4000 t.u. unclipped but 1741 times clipped -- the v <= 1.5
rail truncates the deep-inhibition recovery and turns one-shot
excitability into perpetual oscillation.  The clips are therefore removed
from this core.  In their place:

1. Adaptive reaction subcycling.  The fast kinetics are far too stiff
   for a single explicit Euler step at the working dt, and a FIXED
   substep count scaled on 1/eps alone is not enough: the Oregonator's
   (u-q)/(u+q) switch gives a pointwise Jacobian
       dF/du = (1/eps) * [1 - 2u - (f v + phi) * 2q / (u+q)^2]
   which reaches ~-7000 on the recovery manifold (v ~ 1, u ~ q) at
   eps=0.05 -- explicit Euler then needs dt_eff <~ 3e-4 (the clip used
   to hide this).  Each reaction step is therefore integrated with
   adaptive substeps h chosen PROACTIVELY from the current state:
       h <= 0.5 / max(-dF/du, 0)   (linear stability, per-substep)
       h <= 0.1 / max(|du/dt|)     (accuracy: |du| <= 0.1 per substep)
   with wall cells excluded from both maxima.  The substeps reuse the
   ONE already-computed Laplacian, so the extra cost is pointwise
   reaction evaluations only -- no extra diffusion work.  This is
   first-order (Godunov) operator splitting: same formal order as the
   original combined explicit Euler step, so results stay comparable
   with the pre-de-hack runs (verified: Test 1 free-medium speed and
   Test 4a c_par/c_perp reproduce within 5%).  A fixed substep count
   can be forced with the react_substeps constructor argument.
2. A non-intrusive guard.  After every step the state is checked for
   NaN/Inf or absurd magnitude (|u| or |v| > 10, far outside any
   physically reachable range for these kinetics).  A violation raises
   FloatingPointError with the step number -- a genuine blow-up surfaces
   loudly instead of being silently reshaped into a plausible-looking
   trajectory.

Stability: explicit diffusion requires dt <= dx^2 / (4 * lambda_max(D));
the class warns when this (with a margin for cross terms) is violated.
At Du = 1, dt = 0.05 is safe up to D_max = 4; for D_max = 8 use dt = 0.02.
The reaction side is handled by the adaptive subcycling above.

Example
-------
    import numpy as np
    from rd_core import RDSubstrate

    rd = RDSubstrate()
    Y, X = np.mgrid[0:rd.ny, 0:rd.nx]          # note: index [x, y]
    wall = np.ones((rd.nx, rd.ny), bool)
    wall[:, 120:136] = False                    # horizontal channel, width 16
    rd.set_walls(wall)
    port = np.zeros_like(wall); port[5:20, 120:136] = True
    rd.add_port('in', port)
    probe = np.zeros_like(wall); probe[200, 126:130] = True
    rd.add_probe('out', probe)
    rd.fire('in')                               # hold 0.8/0.2 for 30 steps
    data = rd.run(800)
    # data['t'], data['out'] -> time series of mean u over the probe mask
"""

import numpy as np


class RDSubstrate:
    """2D light-sensitive Oregonator solver with walls, ports and probes.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions (u.shape == (nx, ny); axis 0 is x, axis 1 is y).
    dx : float
        Grid spacing (isotropic).
    dt : float
        Time step (explicit Euler).
    eps, q, f : float
        Oregonator (Tyson-Fife) kinetic parameters.
    Du, Dv : float
        Default activator / inhibitor diffusivities.  The activator
        diffusion can be upgraded to a full tensor with
        :meth:`set_diffusion_tensor`; the inhibitor always keeps the plain
        scalar Laplacian with Dv.
    kinetics : {'oregonator', 'barkley'}
        Reaction kinetics.  'oregonator' is the Tyson-Fife BZ model above
        (NOTE: at the baseline parameters its rest state is an UNSTABLE
        node -- the medium is a relaxation oscillator, see
        Notes/rd-oscillatory-regime finding).  'barkley' is the Barkley
        excitable-medium model (Barkley 1991), already used in this
        project's ``barkley_*.py`` demos:
            du/dt = (1/eps) * u (1 - u) (u - (v + b)/a) + div(D . grad u)
            dv/dt = u - v + Dv * laplacian(v)
        with a stable rest state and a genuine firing threshold.
    a, b : float
        Barkley kinetic parameters (only used with kinetics='barkley').
    react_substeps : int or 'auto'
        Reaction integration per (diffusion) time step.  'auto' (default)
        uses adaptive substeps chosen from the pointwise Jacobian and the
        current rates (see module docstring).  An int forces that many
        fixed substeps of dt/N -- useful for debugging; N=1 recovers the
        old unsplit behaviour (minus the clips) and will generally blow
        up, which the guard will report.
    clamp_rest : (float, float)
        Values (u, v) that fired ports are clamped to between pulses
        (see :meth:`fire`).  Default (0.0, 0.0) -- correct for kinetics
        whose homogeneous rest state is the origin (Barkley).  For the
        light-held Oregonator (phi > 0) the rest state is u* = v* > 0;
        pass clamp_rest=(u*, u*) (or call :meth:`set_rest`) so the port
        returns to the TRUE rest state after release instead of sitting
        at (0, 0) as a small sustained sink.  The initial field should
        then also be (u*, u*).  Can be changed later with
        :meth:`set_rest`.
    """

    def __init__(self, nx=256, ny=256, dx=1.0, dt=0.05,
                 eps=0.05, q=0.002, f=1.4, Du=1.0, Dv=0.6,
                 kinetics='oregonator', a=0.75, b=0.01,
                 react_substeps='auto', clamp_rest=(0.0, 0.0)):
        assert kinetics in ('oregonator', 'barkley')
        self.kinetics = kinetics
        self.a, self.b = a, b
        self.nx, self.ny = nx, ny
        self.dx, self.dt = dx, dt
        self.eps, self.q, self.f = eps, q, f
        self.Dv = Dv
        if react_substeps == 'auto':
            self._M = None          # adaptive
        else:
            self._M = int(react_substeps)
            assert self._M >= 1

        self.u = np.zeros((nx, ny), dtype=float)
        self.v = np.zeros((nx, ny), dtype=float)

        self.wall = np.zeros((nx, ny), dtype=bool)
        self.phi = 0.0  # scalar or (nx, ny) array

        # activator diffusion tensor components (scalars or (nx, ny) arrays)
        self.Dxx = float(Du)
        self.Dyy = float(Du)
        self.Dxy = 0.0

        self.ports = {}    # name -> bool mask (injection zones)
        self.probes = {}   # name -> bool mask (readout zones)
        self._holds = []   # (start_step, end_step, mask, u_value, v_value)
        self._clamped = set()  # port names clamped to clamp_rest between pulses
        self._clamp_u, self._clamp_v = float(clamp_rest[0]), float(clamp_rest[1])
        self._Su = None    # manufactured/forcing source for du/dt
        self._Sv = None    # manufactured/forcing source for dv/dt
        self._XY = None    # cached (X, Y) cell-centre coordinate arrays
        self.t = 0         # global step counter

        self._check_stability()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def set_walls(self, mask):
        """Set wall cells.  Walls are inert (no reaction) and no diffusive
        flux crosses any face touching a wall cell (true no-flux)."""
        mask = np.asarray(mask, dtype=bool)
        assert mask.shape == (self.nx, self.ny), 'wall mask shape mismatch'
        self.wall = mask
        # keep wall cells quiescent
        self.u[self.wall] = 0.0
        self.v[self.wall] = 0.0

    def set_phi(self, field):
        """Set the light-suppression field phi (scalar or (nx, ny) array)."""
        field = np.asarray(field, dtype=float)
        if field.ndim == 0:
            self.phi = float(field)
        else:
            assert field.shape == (self.nx, self.ny), 'phi shape mismatch'
            self.phi = field

    def set_rest(self, u_rest, v_rest):
        """Set the between-pulse clamp values for fired ports.

        Must match the homogeneous rest state the medium is initialised
        at: (0, 0) for Barkley, (u*, u*) for the light-held Oregonator
        with phi > 0.  See the clamp_rest constructor argument."""
        self._clamp_u, self._clamp_v = float(u_rest), float(v_rest)

    def set_source(self, Su=None, Sv=None):
        """Set additive source terms for MMS / forced problems.

        Su, Sv : callables (X, Y, t) -> ndarray, or None to clear.
            X, Y are the (nx, ny) cell-centre coordinate arrays
            (X = (i + 0.5) * dx along axis 0, likewise Y), t is the
            physical time.  The returned arrays are added to du/dt and
            dv/dt respectively.

        TIME INTEGRATION IS FIRST ORDER: each source is evaluated ONCE
        per outer time step at the beginning-of-step time t_n and added
        explicitly (u += dt * Su(X, Y, t_n)) after the diffusion and
        (subcycled) reaction stages -- a Godunov-split source stage,
        formally O(dt) in time, consistent with the first-order
        operator splitting used throughout this solver.  The source is
        NOT subcycled with the reaction.  Wall cells are re-zeroed after
        the source stage as usual (walls stay inert).
        """
        self._Su = Su
        self._Sv = Sv

    def _coords(self):
        """Cell-centre coordinate arrays (cached)."""
        if self._XY is None:
            i = np.arange(self.nx, dtype=float)
            j = np.arange(self.ny, dtype=float)
            self._XY = ((i[:, None] + 0.5) * self.dx,
                        (j[None, :] + 0.5) * self.dx)
        return self._XY

    def set_diffusion_tensor(self, Dxx, Dyy, Dxy):
        """Set the activator diffusion tensor components.

        Each component may be a scalar or an (nx, ny) array.  The flux form
        uses arithmetic-mean face tensors:
            flux_x = Dxx_face * du/dx + Dxy_face * du/dy   (on x-faces)
            flux_y = Dyy_face * du/dy + Dxy_face * du/dx   (on y-faces)
        with the transverse derivative on a face evaluated as the averaged
        4-point centred difference.  For Dxx = Dyy = const, Dxy = 0 this is
        exactly the 5-point Laplacian.
        """
        self.Dxx = self._as_field(Dxx, 'Dxx')
        self.Dyy = self._as_field(Dyy, 'Dyy')
        self.Dxy = self._as_field(Dxy, 'Dxy')
        self._check_stability()

    def _as_field(self, val, name):
        arr = np.asarray(val, dtype=float)
        if arr.ndim == 0:
            return float(arr)
        assert arr.shape == (self.nx, self.ny), f'{name} shape mismatch'
        return arr

    def _check_stability(self):
        lam_max = self._lambda_max()
        # conservative explicit limit including cross-term contribution
        dt_lim = self.dx**2 / (4.0 * lam_max)
        if self.dt > 0.95 * dt_lim:
            print(f'[rd_core] WARNING: dt={self.dt} close to/above explicit '
                  f'diffusion stability limit ~{dt_lim:.4f} '
                  f'(lambda_max(D)={lam_max:.2f})')

    def _lambda_max(self):
        Dxx, Dyy, Dxy = self.Dxx, self.Dyy, self.Dxy
        tr2 = 0.5 * (Dxx + Dyy)
        disc = np.sqrt((0.5 * (Dxx - Dyy))**2 + Dxy**2)
        lam = tr2 + disc
        # include |Dxy| margin for the cross-derivative stencil
        lam = lam + np.abs(Dxy)
        return float(np.max(lam)) if isinstance(lam, np.ndarray) else float(lam)

    # ------------------------------------------------------------------
    # Ports, probes, stimuli
    # ------------------------------------------------------------------
    def add_port(self, name, mask):
        """Register an input injection zone (bool mask)."""
        mask = np.asarray(mask, dtype=bool)
        assert mask.shape == (self.nx, self.ny), 'port mask shape mismatch'
        self.ports[name] = mask

    def add_probe(self, name, mask):
        """Register a readout zone; mean u over the mask is recorded every
        step during :meth:`run`."""
        mask = np.asarray(mask, dtype=bool)
        assert mask.shape == (self.nx, self.ny), 'probe mask shape mismatch'
        assert mask.any(), f'probe {name!r} is empty'
        self.probes[name] = mask

    def fire(self, port, value=0.8, v_value=0.2, duration=30):
        """Hold the named port's cells at (value, v_value) for `duration`
        steps, starting at the current step.  This is the pulse-injection
        mechanism of the reference scripts.

        The port is CLAMPED to `clamp_rest` (default (0, 0), see the
        constructor and :meth:`set_rest`) outside the firing window: at
        the baseline parameters the homogeneous rest state is weakly
        unstable (relaxation-oscillator-like), so a released port would
        otherwise re-fire spontaneously and act as a pacemaker.  The
        clamp makes the port an ideal signal generator: exactly one
        pulse per fire().  For kinetics with a rest state away from the
        origin (light-held Oregonator, phi > 0, rest u* = v* > 0),
        clamp_rest MUST be set to (u*, u*): with the (0, 0) default the
        port would sit below rest as a small sustained sink and distort
        the passing wave."""
        self._holds.append((self.t, self.t + duration,
                            self.ports[port], value, v_value))
        self._clamped.add(port)

    def pulse_train(self, port, times, value=0.8, v_value=0.2, duration=30):
        """Schedule pulses of the named port at the given (absolute) step
        indices.  As with :meth:`fire`, the port is clamped to
        `clamp_rest` (default (0, 0)) between pulses to suppress
        spontaneous re-firing."""
        mask = self.ports[port]
        for t0 in times:
            self._holds.append((int(t0), int(t0) + duration, mask,
                                value, v_value))
        self._clamped.add(port)

    # ------------------------------------------------------------------
    # Numerics
    # ------------------------------------------------------------------
    def _face_means(self, D, axis):
        """Arithmetic mean of a scalar/array tensor component on faces."""
        if not isinstance(D, np.ndarray):
            return D
        if axis == 0:
            return 0.5 * (D[:-1, :] + D[1:, :])
        return 0.5 * (D[:, :-1] + D[:, 1:])

    def _diffuse(self, c, Dxx, Dyy, Dxy):
        """Divergence of tensor flux; returns div(D . grad c) on the grid.

        Faces touching a wall cell carry zero flux (true no-flux walls).
        Domain boundaries are no-flux by construction (no exterior faces).
        """
        dx = self.dx
        div = np.zeros_like(c)
        has_cross = (isinstance(Dxy, np.ndarray) and np.any(Dxy != 0.0)) or \
                    (not isinstance(Dxy, np.ndarray) and Dxy != 0.0)

        # --- x-faces: shape (nx-1, ny); face (i, j) sits between i and i+1
        fx = self._face_means(Dxx, 0) * (c[1:, :] - c[:-1, :]) / dx
        if has_cross:
            Dxy_f = self._face_means(Dxy, 0)
            dcdy = np.zeros_like(fx)
            # averaged 4-point centred cross derivative on the x-face
            dcdy[:, 1:-1] = 0.25 * (c[:-1, 2:] - c[:-1, :-2]
                                    + c[1:, 2:] - c[1:, :-2]) / dx
            fx = fx + Dxy_f * dcdy
        fx[self.wall[:-1, :] | self.wall[1:, :]] = 0.0
        div[:-1, :] += fx / dx
        div[1:, :] -= fx / dx

        # --- y-faces: shape (nx, ny-1); face (i, j) sits between j and j+1
        fy = self._face_means(Dyy, 1) * (c[:, 1:] - c[:, :-1]) / dx
        if has_cross:
            Dxy_f = self._face_means(Dxy, 1)
            dcdx = np.zeros_like(fy)
            dcdx[1:-1, :] = 0.25 * (c[2:, :-1] - c[:-2, :-1]
                                    + c[2:, 1:] - c[:-2, 1:]) / dx
            fy = fy + Dxy_f * dcdx
        fy[self.wall[:, :-1] | self.wall[:, 1:]] = 0.0
        div[:, :-1] += fy / dx
        div[:, 1:] -= fy / dx

        return div

    def _react_rates(self, u, v):
        """Pointwise reaction rate du/dt and its diagonal Jacobian
        d(du/dt)/du (used for the adaptive substep stability limit)."""
        if self.kinetics == 'oregonator':
            s = self.f * v + self.phi
            g = (u - self.q) / (u + self.q)
            react = (u - u**2 - s * g) / self.eps
            # s is independent of u; dg/du = 2q/(u+q)^2
            jac = (1.0 - 2.0 * u - s * (2.0 * self.q / (u + self.q)**2)) / self.eps
        else:  # barkley: threshold (v + b)/a, phi acts as extra threshold
            u_thr = (v + self.b + self.phi) / self.a
            react = (u * (1.0 - u) * (u - u_thr)) / self.eps
            jac = ((1.0 - 2.0 * u) * (u - u_thr) + u * (1.0 - u)) / self.eps
        return react, jac

    def _react_fixed(self, dt):
        """Reaction integration with a fixed number of substeps."""
        u, v = self.u, self.v
        h = dt / self._M
        for _ in range(self._M):
            react, _ = self._react_rates(u, v)
            u += h * react
            v += h * (u - v)

    def _react_adaptive(self, dt):
        """Reaction integration over dt with adaptive substeps.

        h is chosen proactively at every substep from the current state:
          - linear stability:  h <= STAB / max(-dF/du, 0)  (the stiff
            Oregonator switch reaches dF/du ~ -7000 on the recovery
            manifold at eps=0.05; fixed coarse substeps explode there)
          - accuracy:          h <= ACC / max(|du/dt|, |dv/dt|)
          - domain (Oregonator): the kinetics have a pole at u = -q, so h
            is further limited such that no substep can carry u below
            -q/2.  This is STEP-SIZE control, not clipping -- the state
            itself is never altered (the continuous flow keeps u > 0;
            only a discrete overshoot can reach the pole).
        Wall cells are excluded from the maxima (they are inert and are
        re-zeroed after the reaction anyway).  Only pointwise operations
        are involved -- no extra Laplacian work.
        """
        STAB, ACC = 0.5, 0.1
        MAXSUB = 200000
        u, v = self.u, self.v
        wall = self.wall if self.wall.any() else None
        # lower domain bound for u (pole of the switch term sits at -q)
        u_dom = -0.5 * self.q if self.kinetics == 'oregonator' else None
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
            if u_dom is not None:
                falling = (react < 0.0) if wall is None else \
                    (react < 0.0) & ~wall
                if np.any(u[falling] <= u_dom):
                    raise FloatingPointError(
                        f'[rd_core] u left the kinetic domain (u <= '
                        f'{u_dom}) at step {self.t} -- reaction substep '
                        f'control failed; reduce dt.')
                if falling.any():
                    h_dom = float(((u - u_dom) / (-react))[falling].min())
                    h = min(h, h_dom)
            h = min(h, dt - t_acc)
            u += h * react
            v += h * (u - v)
            t_acc += h
            n_sub += 1
            if n_sub > MAXSUB:
                raise FloatingPointError(
                    f'[rd_core] reaction subcycling exceeded {MAXSUB} '
                    f'substeps at step {self.t} (h={h:.3g}) -- kinetics too '
                    f'stiff for explicit integration at dt={self.dt}.')
            h *= 4.0  # relax back up; re-clamped from the new state
        self._n_react_sub = n_sub  # diagnostics

    def _step(self):
        u, v = self.u, self.v
        # 1) diffusion: one explicit step on the already-computed fluxes
        u += self.dt * self._diffuse(u, self.Dxx, self.Dyy, self.Dxy)
        v += self.dt * self._diffuse(v, self.Dv, self.Dv, 0.0)
        # 2) reaction: adaptive subcycling (the de-hacked replacement for
        #    the old np.clip crutch -- see module docstring).  The v
        #    substep uses the just-updated u (Gauss-Seidel flavour,
        #    slightly more stable on the slow manifold).
        if self._M is None:
            self._react_adaptive(self.dt)
        else:
            self._react_fixed(self.dt)
        # 3) external source (MMS / forcing): explicit, first order in
        #    time, evaluated once per outer step at t_n (see set_source)
        if self._Su is not None or self._Sv is not None:
            X, Y = self._coords()
            t_n = self.t * self.dt
            if self._Su is not None:
                u += self.dt * self._Su(X, Y, t_n)
            if self._Sv is not None:
                v += self.dt * self._Sv(X, Y, t_n)
        if self.wall.any():
            u[self.wall] = 0.0
            v[self.wall] = 0.0
        self._guard()

    def _guard(self):
        """Non-intrusive blow-up detector: raise on NaN/Inf or absurd
        magnitude, never silently reshape the trajectory (the old clips
        did exactly that -- see module docstring)."""
        u, v = self.u, self.v
        bad = not (np.isfinite(u).all() and np.isfinite(v).all())
        um = float(np.abs(u).max()) if not bad else float('nan')
        vm = float(np.abs(v).max()) if not bad else float('nan')
        if bad or um > 10.0 or vm > 10.0:
            raise FloatingPointError(
                f'[rd_core] numerical blow-up at step {self.t} '
                f'(kinetics={self.kinetics}, eps={self.eps}, dt={self.dt}, '
                f'M={self._M}): |u|max={um:.3g}, |v|max={vm:.3g}, '
                f'finite={not bad}. Reduce dt or raise react_substeps.')

    # ------------------------------------------------------------------
    # Time loop
    # ------------------------------------------------------------------
    def run(self, nsteps):
        """Advance `nsteps` steps, applying scheduled port holds.

        Returns
        -------
        dict
            't' : time array (physical time, (nsteps,)); one entry per
            registered probe name with the mean-u time series over the
            probe mask.
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
            # clamp fired ports to clamp_rest outside their pulse windows
            # so a released port cannot turn into a pacemaker; clamp_rest
            # is the medium's homogeneous rest state ((0,0) for Barkley,
            # (u*,u*) for the light-held Oregonator) so the released port
            # sits AT rest, not as a sink
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


# ---------------------------------------------------------------------------
# Self-diagnostics
# ---------------------------------------------------------------------------
def _selftest():
    """Check the tensor flux form reduces to the 5-point Laplacian and that
    walls block all flux."""
    rng = np.random.default_rng(0)
    rd = RDSubstrate(nx=32, ny=32, dt=0.05, Du=1.0)
    c = rng.random((32, 32))

    # 1) Dxx = Dyy = const, Dxy = 0  ==  D * 5-point Laplacian (interior)
    div = rd._diffuse(c, 1.0, 1.0, 0.0)
    lap = np.zeros_like(c)
    lap[1:-1, 1:-1] = (c[:-2, 1:-1] + c[2:, 1:-1]
                       + c[1:-1, :-2] + c[1:-1, 2:]
                       - 4 * c[1:-1, 1:-1])
    err = np.abs(div[1:-1, 1:-1] - lap[1:-1, 1:-1]).max()
    assert err < 1e-12, f'tensor form does not reduce to Laplacian: {err}'
    print(f'[selftest] tensor->Laplacian reduction OK (max err {err:.2e})')

    # 2) wall flux blocking: constant field must stay constant; a gradient
    #    across a full wall barrier must produce zero flux through it.
    wall = np.zeros((32, 32), bool)
    wall[16, :] = True  # complete vertical barrier
    rd.set_walls(wall)
    c2 = np.zeros((32, 32))
    c2[:16, :] = 1.0
    div2 = rd._diffuse(c2, 1.0, 1.0, 0.0)
    leak = np.abs(div2[15, :]).max()  # cell just left of the barrier
    # cell 15 feels no pull from the walled cell 16 -> div there must be 0
    assert leak < 1e-12, f'flux leaks through wall: {leak}'
    print('[selftest] wall no-flux blocking OK')


if __name__ == '__main__':
    _selftest()
