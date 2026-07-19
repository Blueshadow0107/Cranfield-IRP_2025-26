# Solver Specifications — Cranfield IRP 2025-26

*All solvers documented with domain, boundary conditions, initial conditions, and numerical method.*

---

## 1. Thermoviscous Channel Solver

**File:** `Analysis/thermoviscous_channel.py` (and `thermoviscous_cylinder.py` variant)

### Domain
| Parameter | Value |
|-----------|-------|
| Length $L$ | 10 mm |
| Height $H$ | 2.5 mm |
| Grid $N_x \times N_y$ | $201 \times 51$ |
| Spacing $\Delta x = \Delta y$ | $5 \times 10^{-5}$ m |

### Boundary Conditions
| Location | Condition |
|----------|-----------|
| Inlet (left, $x=0$) | Parabolic velocity: $u = U_{\max} \cdot 4(y/H)(1-y/H)$, $v=0$ |
| Outlet (right, $x=L$) | Zero-gradient: $\partial u/\partial x = 0$, $\partial v/\partial x = 0$ |
| Walls (top/bottom, $y=0,H$) | No-slip: $u=v=0$ |
| Temperature walls | Insulated: $\partial T/\partial n = 0$ |
| Heat strips | Neumann flux: $q = \text{STRIP\_HEAT\_SOURCE}$ [K/s] |

### Initial Conditions
- Velocity: $u = v = 0$ everywhere
- Temperature: $T = T_{\text{cold}} = 300$ K everywhere

### Numerical Method
- **Formulation:** Streamfunction–vorticity ($\psi$–$\omega$)
- **Time stepping:** Explicit Euler, $\Delta t = 5 \times 10^{-5}$ s
- **Vorticity advection:** Upwind differencing
- **Vorticity diffusion:** 5-point Laplacian
- **Streamfunction Poisson:** 5-point Laplacian with Gauss–Seidel relaxation
- **Temperature:** Advection–diffusion with Arrhenius viscosity $\nu(T)$
- **Viscosity model:** $\nu(T) = \nu_0 \exp(-\beta/T)$ with $\beta = 1900$ K$^{-1}$

### Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| $U_{\max}$ | 4–40 mm/s | Peak inlet velocity |
| Re | 10–100 | Reynolds number |
| $\nu_0$ | $1 \times 10^{-6}$ m²/s | Cold kinematic viscosity (water) |
| $\beta$ | 1900 K$^{-1}$ | Arrhenius coefficient |
| $\alpha$ | $1.4 \times 10^{-7}$ | Thermal diffusivity |

---

## 2. Discrete Cylinder Array Solver

**File:** `Analysis/cylinder_array_solver.py`

### Domain
| Parameter | Value |
|-----------|-------|
| Length $L$ | 10 mm |
| Height $H$ | 2.5 mm |
| Grid $N_x \times N_y$ | $201 \times 51$ |
| Spacing $\Delta x = \Delta y$ | $5 \times 10^{-5}$ m |

### Boundary Conditions
| Location | Condition |
|----------|-----------|
| Inlet (left) | Parabolic velocity: $u = U_{\max} \cdot 4(y/H)(1-y/H)$, $v=0$ |
| Outlet (right) | Zero-gradient: $\partial u/\partial x = 0$, $\partial v/\partial x = 0$ |
| Walls (top/bottom) | No-slip: $u=v=0$ |
| Obstacles (square) | No-slip hard walls inside obstacle cells |

### Initial Conditions
- Velocity: $u = v = 0$ everywhere

### Numerical Method
- **Formulation:** Streamfunction–vorticity ($\psi$–$\omega$)
- **Time stepping:** Explicit Euler, $\Delta t = 5 \times 10^{-5}$ s
- **Vorticity advection:** Upwind differencing
- **Vorticity diffusion:** 5-point Laplacian
- **Streamfunction Poisson:** 5-point Laplacian with Gauss–Seidel relaxation
- **Obstacle treatment:** Mask array — zero velocity inside obstacle cells

### Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| $U_{\max}$ | 40 mm/s | Peak inlet velocity |
| Re | 40 | Reynolds number (based on $U_{\max}$ and channel height) |
| $\Delta t$ | $5 \times 10^{-5}$ s | Time step |
| $T_{\text{total}}$ | 1.0 s | Simulation time |
| Obstacle size | 10 grid cells | Square obstacle half-width |
| Probe location | $(150, 25)$ | Downstream velocity probe |

### Configurations Tested
| Config | Cylinders (ix, iy, half-width) | Notes |
|--------|-------------------------------|-------|
| none | — | Baseline laminar |
| single | (100, 25, 10) | Standard wake |
| tandem_close | (100, 25, 10), (115, 25, 10) | Inline, close |
| tandem_far | (100, 25, 10), (130, 25, 10) | Inline, far — suppressed |
| side_close | (100, 22, 10), (100, 28, 10) | Side-by-side, close |
| side_far | (100, 20, 10), (100, 30, 10) | Side-by-side, far |

---

## 3. Brinkman Projection Solver

**File:** `Analysis/projection_solver.py` (evolved from `collocated_ns_solver.py`)

### Domain
| Parameter | Value |
|-----------|-------|
| Length $L$ | 10 mm |
| Height $H$ | 2.5 mm |
| Grid $N_x \times N_y$ | $201 \times 51$ |
| Spacing $\Delta x = \Delta y$ | $5 \times 10^{-5}$ m |

### Boundary Conditions
| Location | Condition |
|----------|-----------|
| Inlet (left, ghost cell $i=0$) | Dirichlet: $u_{0,j} = U_{\text{inlet}}(y_j)$, $v_{0,j} = 0$ |
| Outlet (right, ghost cell $i=N_x+1$) | Zero-gradient: $u_{N_x+1,j} = u_{N_x,j}$, $v_{N_x+1,j} = v_{N_x,j}$ |
| | Pressure Dirichlet: $p_{N_x+1,j} = -p_{N_x,j}$ (so face pressure $p_{\text{face}} = 0$) |
| Walls (bottom/top, ghost cells $j=0,N_y+1$) | No-slip: $u_{i,0} = -u_{i,1}$, $u_{i,N_y+1} = -u_{i,N_y}$ |
| | $v_{i,0} = 0$, $v_{i,N_y+1} = 0$ |
| | Pressure Neumann: $\partial p/\partial n = 0$ |

### Initial Conditions
- Velocity: $u = v = 0$ everywhere (including ghost cells)
- Pressure: $p = 0$ everywhere (including ghost cells)

### Numerical Method
**Face-consistent collocated projection method** (the key fix)

1. **Predictor step** (explicit):
   $$\mathbf{u}^* = \mathbf{u}^n + \Delta t \left( -\mathbf{u}^n \cdot \nabla \mathbf{u}^n + \nu \nabla^2 \mathbf{u}^n - \frac{1}{\rho}\nabla p^n - \alpha \mathbf{u}^n \right)$$

2. **Pressure Poisson equation**:
   $$\nabla^2 p^{n+1} = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^*$$
   - Discretized with 5-point Laplacian
   - Sparse LU factorization (`scipy.sparse.linalg.splu`) pre-computed once
   - Solved each step in ~1 ms

3. **Face corrector** (the critical fix):
   - Compute pressure gradient at **cell faces**: $\partial p/\partial x|_{i+1/2} = (p_{i+1} - p_i)/\Delta x$
   - Subtract from face velocities: $u_{i+1/2}^{n+1} = u_{i+1/2}^* - (\Delta t/\rho) \cdot \partial p/\partial x|_{i+1/2}$
   - Average corrected faces back to cell centers

4. **Semi-implicit Brinkman drag**:
   $$\mathbf{u}^{n+1} = \frac{\mathbf{u}^{n+1}}{1 + \Delta t \cdot \alpha}$$
   - $\alpha = 0$ in fluid, $\alpha = \alpha_{\max} = 10^6$ inside obstacles

### Why This Works
On a collocated grid, the pressure Laplacian is naturally face-based:
$$\nabla \cdot (\nabla p)|_{\text{face}} \approx \frac{p_{i+1} - 2p_i + p_{i-1}}{\Delta x^2}$$
If you use cell-center gradients for the corrector, the divergence of the corrected velocity does **not** equal the Laplacian of pressure → instability (NaN in 15 steps). By keeping divergence and gradient both at faces, $\text{div}_{\text{face}}(\text{grad}_{\text{face}}(p)) \equiv \nabla^2 p$ exactly.

### Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| $U_{\max}$ | 40 mm/s | Peak inlet velocity |
| Re | 100 | Base Reynolds number |
| $\nu$ | $1 \times 10^{-6}$ m²/s | Kinematic viscosity |
| $\alpha_{\max}$ | $10^6$ | Brinkman penalty inside obstacles |
| $\Delta t$ | Auto (CFL-limited) | Typically $5 \times 10^{-5}$ s |
| CFL target | 0.15 | Stability criterion |

### Stability
- CFL condition: $\Delta t < \text{CFL} \cdot \Delta x / U_{\max}$
- Viscous condition: $\Delta t < \Delta x^2 / (4\nu)$
- Stable for 10,000+ steps with face-consistent stencil
- NaN in 15 steps with cell-center gradient corrector

---

## 4. Womersley / Oscillatory Inlet Solver

**File:** `Analysis/womersley_cylinder.py`

### Domain & Grid
Same as Brinkman projection solver above.

### Boundary Conditions
| Location | Condition |
|----------|-----------|
| Inlet | Oscillatory parabolic: $U(t) = U_{\max}(1 + \varepsilon \sin(2\pi f t)) \cdot 4(y/H)(1-y/H)$ |
| Outlet | Zero-gradient velocity, $p=0$ |
| Walls | No-slip |

### Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| $f$ | 0.5–20 Hz | Oscillation frequency |
| $\varepsilon$ | 0.1, 0.3 | Relative amplitude |
| $U_{\max}$ | 40 mm/s | Mean peak velocity |
| Re | 100 | Based on mean flow |

---

## 5. Reynolds Number Sweep

**File:** `Analysis/re_sweep.py`

### Method
Same Brinkman projection solver. Re varied from 10 to 1000 (log-spaced 15 points).
- $U_{\max}$ fixed at 40 mm/s
- $\nu = U_{\max} H / \text{Re}$ varies
- $\Delta t$ auto-scaled with CFL

---

## Summary Table: All Solvers

| Solver | Equations | Grid | Time Step | Obstacles |
|--------|-----------|------|-----------|-----------|
| Thermoviscous channel | $\psi$–$\omega$ + energy | $201 \times 51$ | $5\times10^{-5}$ s | None |
| Cylinder array | $\psi$–$\omega$ | $201 \times 51$ | $5\times10^{-5}$ s | Discrete squares |
| Brinkman projection | Primitive NS + Brinkman | $201 \times 51$ | Auto CFL | Smooth porous |
| Womersley | Primitive NS + Brinkman | $201 \times 51$ | Auto CFL | Smooth porous |
| Re sweep | Primitive NS + Brinkman | $201 \times 51$ | Auto CFL | Single cylinder |
