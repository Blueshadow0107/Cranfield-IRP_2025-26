# Solver options for the porous acoustic vowel classifier

Date: 2026-06-14
Status: comparison / decision aid

This note lists the main ways to solve acoustic propagation in a rigid-frame porous medium, with trade-offs. The goal is to choose one approach to present to supervisors and implement.

---

## Option 1: Frequency-domain ZK Helmholtz (simplified isothermal)

### Equations

For harmonic time dependence `exp(i omega t)`:

```
div( (1 / rho_eff(omega)) * grad(p) ) + (omega^2 / K_eff(omega)) * p = 0
```

with

```
rho_eff(omega) = (rho0 * alpha_inf / phi) * (1 + sigma*phi / (i*omega*rho0*alpha_inf))
K_eff(omega) = K0 / phi
```

where `K0 = rho0 * c0^2`.

### Workflow for broadband input

1. FFT the vowel waveform to get `S(omega)`.
2. Solve Helmholtz at each frequency of interest.
3. Multiply solution by `S(omega)`.
4. IFFT to get time-domain probe signals.
5. Integrate squared pressure to get features.

### Pros

- Physically standard and defensible.
- Frequency-dependent attenuation is built in.
- Fast per frequency (sparse linear solve).
- Optimisation is straightforward: design `phi(x,y)` and `sigma(x,y)` to shape the frequency response.
- No ambiguity about where `phi` sits: `rho_eff` and `K_eff` are defined.

### Cons

- No direct time-domain dynamics; time signal is reconstructed.
- Need enough frequency samples to resolve the vowel spectrum.
- Reflection/transmission at interfaces is handled by the spatial variation of `rho_eff` and `K_eff`.

### Best for

- A physically rigorous first demonstration.
- Comparing with impedance-tube or absorption data.

---

## Option 2: Frequency-domain ZK Helmholtz (full thermal + viscous)

### Equations

Same Helmholtz structure, but `rho_eff(omega)` and `K_eff(omega)` include both viscous and thermal boundary-layer corrections:

```
rho_eff(omega) = (rho0 * alpha_inf / phi) * G_rho(omega, sigma, alpha_inf, phi, pore_radius)
K_eff(omega) = (K0 / phi) * G_K(omega, pore_radius, Pr, gamma)
```

`G_rho` and `G_K` involve Bessel functions of the viscous and thermal Womersley numbers.

### Pros

- Most accurate equivalent-fluid model for real porous materials.
- Captures both viscous and thermal losses.

### Cons

- More parameters needed: pore radius, Prandtl number, gamma.
- More expensive to evaluate.
- Thermal parameters may be hard to justify for a designed structure.

### Best for

- Final validation against experiments or high-fidelity simulations.

---

## Option 3: Time-domain ZK with relaxation kernels

### Idea

Approximate the frequency-dependent `rho_eff(omega)` by a rational function or sum of relaxation terms:

```
rho_eff(omega) = rho_infinity + sum_j a_j / (1 + i*omega*tau_j)
```

Transform back to time domain to get a differential equation with memory variables.

### Equations (schematic)

```
dV/dt = f(p, V, memory variables)
d(memory_j)/dt = g_j(V, memory_j)
dp/dt + K0 * div(V) = 0
```

### Pros

- Captures time dynamics and frequency-dependent losses simultaneously.
- Direct time-domain readout (matches Hughes experiment).
- No need for FFT/IFFT per sample.

### Cons

- Complex to implement and validate.
- Adds memory variables per grid cell.
- Optimisation is harder because each forward run is time-stepping.

### Best for

- A final, physically faithful time-domain solver after the frequency-domain model is validated.

---

## Option 4: Simplified time-domain ZK (constant sigma)

### Equations

```
(rho0 * alpha_inf / phi) * dV/dt + sigma * V = -grad(p)
dp/dt + rho0 * c0^2 * div(V) = 0
```

with `sigma` treated as a constant (frequency-independent) design variable.

### Pros

- Simple and fast.
- Direct time-domain readout.
- Easy to prototype and debug.

### Cons

- `sigma` is not really constant with frequency; valid only in low-frequency limit.
- Pore-size validity constraint: pores must be smaller than viscous skin depth.
- Ambiguity about exact form and `phi` placement, as discussed.

### Best for

- A first proof-of-concept if supervisors accept the simplification.

---

## Option 5: Direct pore-resolving simulation

### Idea

Solve the linearized Navier-Stokes equations inside the actual pore geometry.

### Pros

- Most accurate.
- No equivalent-fluid assumptions.
- Captures all viscous and thermal effects automatically.

### Cons

- Extremely expensive.
- Requires detailed 3D geometry.
- Not feasible inside an optimisation loop.

### Best for

- Validation of the equivalent-fluid model on a small test geometry.

---

## Option 6: Hybrid — frequency-domain response + time-domain synthesis

### Idea

Use Option 1 or 2 to compute the frequency-domain transfer function from source to probes. Then multiply by the input spectrum and IFFT to get the time-domain probe signals used for classification.

### Pros

- Combines physical accuracy of frequency-domain ZK with time-domain readout.
- Easier to implement than full relaxation time-domain.
- Still allows optimisation over `phi` and `sigma`.

### Cons

- Requires a discrete set of frequencies and FFT/IFFT overhead.
- Transient effects at the start of the vowel window are not captured as naturally as in pure time-domain.

### Best for

- A practical middle ground: rigorous physics + Hughes-style time-integrated readout.

---

## How the options relate to Hughes et al. (2019)

Hughes used a **3D printed acoustic cavity** and injected time-domain vowel waveforms. Their readout was time-integrated pressure at output probes.

To reproduce that style:

- **Time-domain readout is required.**
- **Time-domain solver options:** Option 3 (relaxation), Option 4 (simplified), or Option 6 (hybrid with IFFT).
- **Frequency-domain only** (Options 1–2) gives frequency-response information but needs synthesis for time-domain features.

---

## Recommendation

For supervisor discussion, present the choice as:

1. **Target model:** frequency-domain simplified ZK Helmholtz (Option 1) or full ZK (Option 2) for physical rigour.
2. **Practical classifier:** Option 6 hybrid, using the frequency-domain transfer function to synthesize time-domain probe outputs.
3. **Future extension:** Option 3 (time-domain relaxation) once the frequency-domain model is validated.

This gives a clear path from rigorous physics to a working classifier.

---

## Open questions for supervisors

1. Which level of physical fidelity do we need: simplified isothermal ZK, full ZK, or direct pore simulation?
2. Is frequency-domain + IFFT acceptable for the time-domain readout, or do we need a pure time-domain solver?
3. Should `phi` and `sigma` be independent design variables, or linked through a pore-radius model?
4. What pore size / manufacturing method should we target for physical validity?
