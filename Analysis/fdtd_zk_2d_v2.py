#!/usr/bin/env python3
"""
fdtd_zk_2d_v2.py
================
Two-dimensional FDTD solver for the Zwikker-Kosten (ZK) equivalent-fluid model
in a rigid-frame porous medium.

Variables (staggered grid):
    p : pressure at cell centres         (nx, ny)
    u : x-velocity at x-faces            (nx+1, ny)
    v : y-velocity at y-faces            (nx, ny+1)

Governing equations (time domain):
    rho_eff * du/dt + sigma * u = - dp/dx
    rho_eff * dv/dt + sigma * v = - dp/dy
    (1/K_eff) * dp/dt           = - (du/dx + dv/dy)

where in the porous medium:
    rho_eff = k_s * rho0 / phi
    K_eff   = p0 / phi

and in air (phi = 1, sigma = 0, k_s = 1):
    rho_eff = rho0
    K_eff   = rho0 * c0^2

Boundary conditions:
    - All boundaries: first-order Mur ABC with a damping sponge layer
    - Sources: soft (additive) interior sources
"""

import numpy as np
from scipy.ndimage import zoom


class ZKFDTD2D:
    def __init__(self, nx, ny, dx, c0=343.0, rho0=1.225, p0=101325.0):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dx
        self.c0 = c0
        self.rho0 = rho0
        self.p0 = p0
        self.Z_air = rho0 * c0

        # fields
        self.p = np.zeros((nx, ny))
        self.u = np.zeros((nx + 1, ny))
        self.v = np.zeros((nx, ny + 1))
        self.time = 0.0
        self.n_step = 0

        # material fields: default to air
        self.phi = np.ones((nx, ny))
        self.sigma = np.zeros((nx, ny))
        self.ks = np.ones((nx, ny))
        self._update_derived()

        # time step: safety factor slightly below the 2-D CFL limit
        self.dt = 0.85 * dx / (c0 * np.sqrt(2.0))

        # sponge damping (zero until configured)
        self.sigma_sponge = np.zeros((nx, ny))

    def _update_derived(self):
        """Compute rho_eff and K_eff from phi, sigma, ks."""
        self.rho_eff = self.ks * self.rho0 / self.phi
        self.K_eff = np.where(
            (self.phi == 1.0) & (self.ks == 1.0),
            self.rho0 * self.c0**2,
            self.p0 / self.phi
        )

    def set_uniform_material(self, phi, sigma, ks):
        """Set uniform porous material throughout the domain."""
        self.phi[:] = phi
        self.sigma[:] = sigma
        self.ks[:] = ks
        self._update_derived()

    def set_sponge_layer(self, width_cells, sigma_max=1e4):
        """
        Add a quadratic sponge damping profile near all four boundaries.

        Parameters
        ----------
        width_cells : int
            Number of cells over which the sponge ramps up.
        sigma_max : float
            Maximum artificial damping (1/s) at the boundary.
        """
        nx, ny = self.nx, self.ny
        w = int(width_cells)
        x = np.arange(nx)
        y = np.arange(ny)

        def ramp(idx, n):
            left = np.minimum(idx, w)
            right = np.minimum(n - 1 - idx, w)
            d = np.minimum(left, right)
            profile = np.zeros(n)
            mask = d < w
            profile[mask] = ((w - d[mask]) / w) ** 2
            return profile

        rx = ramp(x, nx)[:, None]
        ry = ramp(y, ny)[None, :]
        self.sigma_sponge = sigma_max * np.maximum(rx, ry)

    def set_material_from_design(self, phi_design, sigma_design, ks_design,
                                  design_i0, design_i1, design_j0, design_j1,
                                  order=1):
        """Map coarse design grids onto the simulation grid."""
        def interp(design):
            fine = zoom(design,
                        ((design_i1 - design_i0) / design.shape[0],
                         (design_j1 - design_j0) / design.shape[1]),
                        order=order)
            return np.clip(fine, 0.0, 1.0)

        phi_fine = interp(phi_design)
        sigma_fine = interp(sigma_design)
        ks_fine = interp(ks_design)

        phi_fine = 0.30 + 0.65 * phi_fine
        sigma_fine = 1e5 * sigma_fine
        ks_fine = 1.0 + 2.0 * ks_fine

        self.phi[design_i0:design_i1, design_j0:design_j1] = phi_fine
        self.sigma[design_i0:design_i1, design_j0:design_j1] = sigma_fine
        self.ks[design_i0:design_i1, design_j0:design_j1] = ks_fine
        self._update_derived()

    def step(self):
        """Advance one time step."""
        dt = self.dt
        dx = self.dx

        # Store old pressure boundary rows/cols for Mur ABC
        p_old_left = self.p[0, :].copy()
        p_old_right = self.p[-1, :].copy()
        p_old_top = self.p[:, -1].copy()
        p_old_bottom = self.p[:, 0].copy()

        u_new = self.u.copy()
        v_new = self.v.copy()

        # u update on interior x-faces
        dp_dx = (self.p[1:, :] - self.p[:-1, :]) / dx
        sigma_u = 0.5 * (self.sigma[:-1, :] + self.sigma[1:, :])
        rho_u = 0.5 * (self.rho_eff[:-1, :] + self.rho_eff[1:, :])
        u_new[1:-1, :] = self.u[1:-1, :] + dt * (
            -dp_dx - sigma_u * self.u[1:-1, :]
        ) / rho_u

        # v update on interior y-faces
        dp_dy = (self.p[:, 1:] - self.p[:, :-1]) / dx
        sigma_v = 0.5 * (self.sigma[:, :-1] + self.sigma[:, 1:])
        rho_v = 0.5 * (self.rho_eff[:, :-1] + self.rho_eff[:, 1:])
        v_new[:, 1:-1] = self.v[:, 1:-1] + dt * (
            -dp_dy - sigma_v * self.v[:, 1:-1]
        ) / rho_v

        # pressure update
        du_dx = (u_new[1:, :] - u_new[:-1, :]) / dx
        dv_dy = (v_new[:, 1:] - v_new[:, :-1]) / dx
        p_new = self.p - dt * self.K_eff * (du_dx + dv_dy)

        # Apply sponge damping to pressure (acts as artificial absorption)
        p_new -= dt * self.sigma_sponge * p_new

        # First-order Mur ABC on all four sides (Taflove form)
        c = self.c0
        r = c * dt / dx
        if r == 1.0:
            # exact transparent condition
            p_new[0, :] = p_old_left[1, :]  # left
            p_new[-1, :] = p_old_right[-2, :]  # right
            p_new[:, 0] = p_old_bottom[:, 1]  # bottom
            p_new[:, -1] = p_old_top[-2]  # top
        else:
            coef = (r - 1.0) / (r + 1.0)
            # Left boundary (outward -x)
            p_new[0, :] = p_new[1, :] + coef * (p_old_left - p_new[1, :])
            # Bottom boundary (outward -y)
            p_new[:, 0] = p_new[:, 1] + coef * (p_old_bottom - p_new[:, 1])
            # Right boundary (outward +x)
            p_new[-1, :] = p_old_right[-2] + coef * (p_new[-2, :] - p_old_right)
            # Top boundary (outward +y)
            p_new[:, -1] = p_old_top[-2] + coef * (p_new[:, -2] - p_old_top)

        self.p = p_new
        self.u = u_new
        self.v = v_new
        self.time += dt
        self.n_step += 1

    def apply_soft_source(self, src_value, src_mask):
        """
        Add a soft source to pressure cells selected by src_mask.

        Parameters
        ----------
        src_value : float
            Source amplitude at this time step.
        src_mask : 2-D bool array, shape (nx, ny)
            True for cells that receive the additive source.
        """
        self.p[src_mask] += src_value

    def run(self, n_steps, source_func, source_masks, record_interval=None,
            probe_coords=None):
        """
        Run the simulation.

        Parameters
        ----------
        n_steps : int
        source_func : callable(n) -> float
            Returns the source pressure at step n.
        source_masks : list of 2-D bool arrays
            One mask per source port.  All active masks receive the same
            source time series; use this to define port geometry.
        record_interval : int or None
            If given, record full pressure field every `record_interval` steps.
        probe_coords : list of (i, j) tuples or None
            Cell indices where probe pressure is recorded.

        Returns
        -------
        probe_history : ndarray, shape (n_probes, n_steps)
        field_history : list of p snapshots (if record_interval is set)
        """
        if probe_coords is None:
            probe_coords = []
        n_probes = len(probe_coords)
        probe_history = np.zeros((n_probes, n_steps))
        field_history = []

        for n in range(n_steps):
            self.step()
            src = source_func(n)
            for mask in source_masks:
                self.apply_soft_source(src, mask)

            for k, (i, j) in enumerate(probe_coords):
                probe_history[k, n] = self.p[i, j]

            if record_interval and n % record_interval == 0:
                field_history.append(self.p.copy())

        return probe_history, field_history
