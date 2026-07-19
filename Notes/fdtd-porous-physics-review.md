# Physics Review: Zwikker-Kosten Porous FDTD Solver

**Date:** 2026-06-13  
**Scope:** `Analysis/fdtd_porous.py` and `Analysis/optimise_porous_router.py`  
**Status:** Solver is a sound starting point, but blockwise porosity optimisation has a physically incorrect interface treatment that must be fixed before results are trusted.

---

## 1. Physical system

2D first-order acoustic FDTD for sound propagation in a rigid-framed porous medium, using the Zwikker-Kosten (ZK) equivalent-fluid model.

Discretised equations:

    rho0 * alpha_inf * du/dt + sigma * phi * u = -grad(p)
    dp/dt + (rho0 * c0^2 / phi) * div(u) = 0

where:
- phi = porosity
- alpha_inf = tortuosity
- sigma = static flow resistivity
- u = pore-fluid particle velocity

Effective sound speed:

    c_eff = c0 / sqrt(phi * alpha_inf)

---

## 2. What is correct

- Semi-implicit damping update is stable and first-order accurate.
- Effective bulk modulus K_eff = rho0 * c0^2 / phi is consistent with rho_eff = rho0 * alpha_inf, giving c_eff^2 = K_eff / rho_eff.
- Hard-wall and pressure-release boundary conditions are implemented correctly for the staggered grid.
- Absorbing boundary condition is a reasonable first-order characteristic approximation for uniform media.
- Energy calculation and decay test are sensible diagnostics.

---

## 3. Issues to fix

### 3.1 Porosity discontinuities are handled incorrectly (serious)

`optimise_porous_router.py` uses blockwise constant phi that jumps between 0.4 and 1.0. The correct jump conditions across a porosity step are:

- pressure p is continuous
- normal volume velocity phi * u_n is continuous

The code stores u at cell faces and updates pressure using div(u), not div(phi*u). It therefore enforces continuity of pore velocity u_n, not volume velocity phi*u_n. This gives wrong reflection and transmission coefficients at every block edge.

Equivalently, the continuity equation for discontinuous phi should contain div(phi*u) = phi*div(u) + u.grad(phi). The code keeps only the phi*div(u) term and drops u.grad(phi), so mass balance is violated at block boundaries.

**Status:** definitely wrong for blockwise constant phi. Acceptable only for smooth phi transitions.

### 3.2 Damping term convention is ambiguous

The code uses sigma * phi * u. Some ZK references define sigma per unit bulk volume and write sigma * phi^2 * u; others define sigma per unit pore area and write sigma * phi * u. The code is internally consistent, but the physical meaning of the numerical sigma value depends on the chosen convention. This must be tied to a specific reference.

### 3.3 Source is very close to absorbing boundary

Source injection at cells 3-6 is only three cells from the left absorbing boundary. A hard source this close can couple to the ABC and create small spurious reflections. Move the source farther in for high-contrast optimisation work.

### 3.4 Missing interface validation test

The existing tests check bulk behaviour (transit time, damping, energy decay) but do not verify behaviour at a porosity step. Add a 1D reflection/transmission test against the analytic ZK result.

---

## 4. Hidden assumptions and regime of validity

- **Rigid frame:** skeleton does not move. Valid for stiff materials, wrong for soft foams.
- **Low-frequency limit:** thermal and viscous boundary-layer effects are ignored. rho_eff is real and sigma is constant. At 5 kHz in air the viscous skin depth is about 0.1 mm, so this is only valid for pores much larger than that.
- **2D geometry:** no out-of-plane spreading. Energy and amplitude scaling differ from 3D.
- **Lossless channel walls:** hard walls do not include thermoviscous boundary layers.
- **Smooth phi required:** the current formulation is only valid when grad(phi) is negligible.

---

## 5. Suggested fixes

### Option A: switch to macroscopic volume velocity (recommended for blockwise designs)

Define:

    v = phi * u

Then the system becomes:

    (rho0 * alpha_inf / phi) * dv/dt + sigma * v = -grad(p)
    dp/dt + rho0 * c0^2 * div(v) = 0

v is naturally continuous at interfaces on a staggered grid, and the divergence term is clean. This is the physically correct way to handle sharp porosity jumps.

### Option B: keep u but smooth phi

If you keep u as the primary variable, apply a Gaussian or sigmoid filter to the blockwise phi field before passing it to the solver. This makes grad(phi) small and the current formulation becomes acceptable.

### Option C: add the missing term

Keep u and add the u.grad(phi) contribution to the pressure update. This is messier on a staggered grid and Option A is preferred.

---

## 6. Concrete next steps

1. Add an analytic 1D interface test: plane wave hitting a single porosity step. Compare reflected/transmitted amplitudes against the ZK analytic formula.
2. Add a volume-velocity continuity diagnostic: check that phi * u_n is continuous across block edges in post-processing.
3. Decide on primary variable (v = phi*u recommended) and refactor `fdtd_porous.py` accordingly, or smooth phi in the design script.
4. Move the source away from the left absorbing boundary.
5. Document the low-frequency, rigid-frame, 2D assumptions in the thesis methodology.
6. Re-run the blockwise optimisation after the interface fix and compare the new optimum with the current FOM = 0.668 result.

---

## 7. Files referenced

- `Analysis/fdtd_porous.py`
- `Analysis/optimise_porous_router.py`
- `Analysis/figures/porous_router_best_design.png`
- `Analysis/figures/porous_router_best_pressure.png`
- `Analysis/figures/porous_router_random_history.png`
