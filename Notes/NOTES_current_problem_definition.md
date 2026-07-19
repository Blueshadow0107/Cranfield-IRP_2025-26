# Current Problem Definition — Acoustic Diffractive Neural Network
# Saved: 2026-06-09
# Status: Physics agreed, awaiting discretization + implementation

## What We Are Solving

A 2D acoustic wave propagation problem where a spatially varying temperature field
creates a programmable diffractive medium. The temperature field is optimized
("trained") to act as a frequency filter.

## Key Insight (The Physics Chain)

T_k (heater block temperature) -> c(x,y) = c0 * sqrt(T/T0) -> local sound speed
Local sound speed -> local wavelength lambda = c/f -> phase accumulation
Phase accumulation -> interference pattern at probe -> output pressure

By optimizing T_k, we sculpt the interference to pass some frequencies and block others.

## Governing Equations (First-Order Acoustic System)

    dp/dt = -rho0 * c(x,y)^2 * (du/dx + dv/dy) + s(x,y,t)
    du/dt = -(1/rho0) * dp/dx
    dv/dt = -(1/rho0) * dp/dy

where c(x,y) = c0 * sqrt(T(x,y)/T0)

## Domain & Parameters

- Fluid: Air
- rho0 = 1.2 kg/m^3
- T0 = 300 K (reference/baseline temperature)
- c0 = 343 m/s (speed of sound at T0)
- Domain: [0, 0.5] m x [0, 0.1] m
- Grid: 500 x 100 cells, dx = dy = 1 mm
- Number of heater blocks: TBD (suggest 4-8 rectangular blocks)

## Temperature Field

T(x,y) = T0 + sum_{k=1}^{N_blocks} (Tk - T0) * indicator_function(block_k)

Trainable parameters: T1, T2, ..., T_Nblocks

## Source

Soft source (adds to field, doesn't overwrite):
    s(x,y,t) = A * sin(2*pi*f*t) * window(x) * 1_strip(y)

Located in first 3-5 grid cells at left boundary.
Window tapers smoothly (Gaussian or cosine bell).

## Boundary Conditions

- Left: Absorbing (characteristic boundary condition)
- Right: Absorbing (characteristic boundary condition)
- Top/Bottom: Rigid wall (v=0, Neumann for p)

## Probe

Single point at right boundary center: (Lx, Ly/2)
Measures: p(t), u(t), v(t)

## Training Task

Frequency filter: Pass f1 = 2 kHz, block f2 = 5 kHz

Loss: L = (|p_probe(f1)| - 1)^2 + (|p_probe(f2)| - 0)^2

Training method: in-silico optimization (SciPy or JAX)

## Why First-Order Form (Not Second-Order Wave Equation)

1. Absorbing boundary conditions are much easier
2. Velocity naturally available at probes (p-u lag analysis possible)
3. More natural for staggered-grid FDTD

## Remaining Steps Before Code

1. Spatial discretization (stencil for dp/dx, du/dx)
2. CFL stability condition for this system
3. Absorbing boundary condition formula
4. Pseudo-code structure
5. Actual implementation

## What Was Archived

All incompressible Navier-Stokes solvers moved to:
    Analysis/archive/2026-06-09_incompressible_ns/

Kept acoustic FDTD code:
    Analysis/y_junction_fdtd/
    Analysis/diffractive_slice/

## Literature to Cite

- Lin et al. (2018) Science — Diffractive Deep Neural Network (foundational)
- Hughes et al. (2019) Sci. Adv. — Wave physics as RNN
- Wang et al. (2024) IEEE TCI — All-Acoustic Diffractive Network
- He et al. (2025) DAC — Diffractive Acoustic Neural Network (DANN)
- Momeini et al. (2023) — Acoustic Physical Neural Network experimental
