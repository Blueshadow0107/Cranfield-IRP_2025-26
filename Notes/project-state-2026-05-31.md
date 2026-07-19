# Project State: Continuous Analog Wave Computation in Porous Media

*Date: 2026-05-31*  
*Status: Architecture exploration and physics foundations phase*  
*Related docs: `architecture-decision-framework.md`, `thermal-and-nonlinearity-summary.md`, `0f839564.md`*

---

## Executive Summary

This document captures the complete state of the Cranfield IRP 2025-26 project following a deep architecture review session on 2026-05-31. The project has evolved from a narrow acoustic-wave-in-porous-media concept into a **multi-physics trainable computer** that exploits:

1. **Wave propagation** (acoustic + elastic) through a porous scaffold
2. **Optothermal reconfiguration** (laser heating patterns rewrite acoustic properties)
3. **Inherent nonlinearity** (Forchheimer, Biot coupling, bubbles, thermal gradients)

The core insight from today's work: **porous media are not linear passive substrates — they are nonlinear active systems whose computational behavior can be spatially programmed by temperature fields.**

---

## 1. The Three Candidate Architectures

| Architecture | Structure | Fluid | Reconfiguration | Nonlinearity | Risk |
|-------------|-----------|-------|-----------------|--------------|------|
| **A** — Fixed scaffold + thermal fluid | 3D porous solid (generic) | Saturates pores | Laser changes c(T), rho(T), mu(T) | Forchheimer + Biot coupling + thermal | Low |
| **B** — Bubble cloud = porosity | None (fluid chamber only) | Pure fluid | Laser nucleates bubbles | **Bubble resonance (extreme)** + thermal | Medium |
| **C** — Hybrid | 3D porous solid | Saturates pores + bubbles | Laser changes fluid + nucleates bubbles | **All mechanisms combined** | High |

**User preference:** Architecture A as core, with Architecture C elements as stretch goal. Full Biot poroelasticity required (fast P, slow P, shear). External lasers for heating.

See `architecture-decision-framework.md` for full comparison.

---

## 2. Thermal Reconfiguration: The Control Mechanism

### 2.1 Physical Effects Available

| Effect | Mechanism | Strength | Speed | Reversibility |
|--------|-----------|----------|-------|---------------|
| Sound-speed change | c(T) for water decreases ~3 m/s per degC | Moderate | ms | Perfect |
| Density change | Thermal expansion rho(T) | Weak | ms | Perfect |
| Viscosity change | nu(T) drops sharply with T | Moderate | ms | Perfect |
| **Bubble nucleation** | Superheat > T_sat(P) | **Extreme** | us | Imperfect |
| Thermal stress | Expansion mismatch -> prestress | Moderate | ms | Perfect |
| Convection | grad T -> buoyancy flow | Weak-Moderate | 100 ms-s | Perfect |

### 2.2 Spatial Resolution Limit

Thermal diffusion in water: alpha_th ~ 0.14e-6 m^2/s.
Diffusion length in 10 ms: L_d = sqrt(alpha_th * t) ~ 37 um.

**Implication:** Sharp thermal "weights" need short heating pulses or acceptance of thermal blur. At 1 MHz acoustic period (1 us), thermal diffusion is negligible — the acoustic wave sees a frozen temperature landscape.

### 2.3 Timescale Hierarchy

| Process | Timescale | Implication |
|---------|-----------|-------------|
| Acoustic period @ 1 MHz | 1 us | Wave sees static T(x) |
| Bubble nucleation | 1-100 us | Comparable to acoustic period |
| Bubble dissolution | 10 ms-1 s | Limits dynamic reconfiguration rate |
| Thermal diffusion across 1 mm | ~7 s | Static programming is clean |

**Static programming** (heat, equilibrate, compute) is the near-term target. **Dynamic reconfiguration** is a stretch goal limited by bubble lifetime.

---

## 3. Nonlinearity in Porous Media: Six Mechanisms

### 3.1 Mechanism Hierarchy

| # | Mechanism | Origin | Strength | Trigger | Scale |
|---|-----------|--------|----------|---------|-------|
| 1 | **Grain contact (Hertzian)** | F ~ delta^(3/2) at grain contacts | 10^2-10^4 x solid | Finite strain | Microscopic |
| 2 | **Forchheimer inertial** | Inertial pressure drop ~ v^2 | Moderate | High velocity | Pore network |
| 3 | **Biot solid-fluid coupling** | Relative motion inertia (w-u) | Moderate | Finite relative displacement | Mesoscopic |
| 4 | **Bubble resonance** | p_g ~ R^(-3*gamma) (nonlinear spring) | **Extreme** | Bubbles present | Microscopic |
| 5 | **Thermal gradient** | c(T(x)) spatial variation | Tunable | grad T != 0 | Macroscopic |
| 6 | **Acoustoelastic prestress** | Wave speed depends on sigma(T) | Moderate | Static stress | Macroscopic |

### 3.2 Mechanism 1: Grain Contact Nonlinearity

In granular/unconsolidated porous media, grains touch at points. Hertzian contact mechanics:

F = (4/3) E* R^(1/2) delta^(3/2)

Stiffness dF/ddelta **increases** with load. During compression: stiffer, faster wave. During rarefaction: softer, slower wave. Wave speed becomes strain-dependent:

c(epsilon) = c0(1 + beta*epsilon + ...)

**Result:** Wave steepening (shock formation). Harmonic generation. The nonlinear parameter beta in granular media is 10^2-10^4 x larger than in homogeneous solids.

### 3.3 Mechanism 2: Forchheimer Inertial Nonlinearity

At low velocity: viscous drag dominates (Darcy's law, linear).
At high velocity: fluid must accelerate around pore obstacles:

grad p = -(mu/k) v - F_xi |v| v

The quadratic |v|v term causes:
- Amplitude-dependent attenuation
- Harmonic generation
- Waveform distortion (sine -> sawtooth)

**Experimental:** Porous absorbers at >140 dB show SPL-dependent absorption (NASA 1985, Umnova 2003).

### 3.4 Mechanism 3: Biot Solid-Fluid Coupling Nonlinearity

Tong et al. (2017) showed that even linear solid + linear fluid = **nonlinear coupled system** when relative displacement is finite. The Lagrangian formulation introduces three coupled nonlinear parameters (gamma1, gamma2, gamma3):

d^2u/dt^2 = c_L^2 d^2u/dx^2 + gamma1*(du/dx)^2 + gamma2*(du/dx)(dw/dx) + gamma3*(dw/dx)^2

**Unique to porous media:** The cross-coupling term gamma2 does not exist in pure solid or pure fluid. Fast + slow wave coupling generates **double-frequency waves** (2*omega from input omega).

### 3.5 Mechanism 4: Bubble Resonance Nonlinearity

**The strongest effect.** A gas bubble in liquid is a nonlinear oscillator (Rayleigh-Plesset):

R * R_ddot + (3/2)*(R_dot)^2 = (1/rho)*(p_g - p_inf - 2*sigma/R - 4*mu*R_dot/R)

Key: p_g ~ p0*(R0/R)^(3*gamma) — **nonlinear restoring force**.

Near Minnaert resonance omega0 = (1/R0)*sqrt(3*gamma*p0/rho):
- Effective sound speed drops dramatically (c_eff ~ 10 m/s possible)
- Nonlinearity parameter B/A explodes (orders of magnitude above water)
- Harmonic generation is extreme
- Parametric mixing (omega1 + omega2) is efficient

**In porous media:** Bubbles are confined in pores (containment shifts resonance). Bubble-bubble interaction creates collective nonlinear modes.

### 3.6 Mechanism 5: Thermal Gradient Nonlinearity

Temperature-dependent sound speed:

c(T) = c0 + alpha_c*(T - T0)

If T(x) is spatially varying, the wave equation becomes:

d^2p/dt^2 - div[c(T(x))^2 grad p] = 0

This is **nonlinear in the sense of nonlinear optics** — the medium has a spatially varying "refractive index." Thermal lensing focuses/defocuses sound.

Add viscous heating (DeltaT ~ p^2) and you get the Westervelt equation — **intrinsic amplitude-dependent nonlinearity**.

### 3.7 Mechanism 6: Acoustoelastic Prestress Nonlinearity

Wave speeds in a prestressed medium depend on stress through third-order elastic constants:

c(sigma) = c0*(1 + (1/2)*(d^2c/dsigma^2)*sigma^2 + ...)

Thermal expansion from laser heating creates internal stress sigma_th(x). This changes wave speeds — a **second-order thermal effect** that is inherently nonlinear.

---

## 4. Convergence: Thermal + Nonlinear + Porous = Trainable Computer

### 4.1 The Neural Network Analogy

| Neural Network Component | Physical Realization in Porous Medium | Control Knob |
|--------------------------|--------------------------------------|--------------|
| **Input layer** | Acoustic transducers / laser-induced sound | Source amplitude, phase, frequency |
| **Hidden layers** | Porous scaffold regions with different T(x) | Laser heating pattern |
| **Weights** | Local sound speed c(T), impedance Z(T) | Temperature field |
| **Activation function** | Forchheimer saturation + bubble clipping + harmonic generation | Laser intensity + source amplitude |
| **Bias** | Static pressure, background temperature | Chamber conditions |
| **Output layer** | Microphone array / pressure sensors at output face | Sensor placement |

### 4.2 Why This Is More Than a Metamaterial

| Feature | Metasurface / Optical DNN | Porous Thermal Computer |
|---------|--------------------------|------------------------|
| Dimensionality | 2D surface | 3D volume |
| Tunability | Fixed at fabrication | Continuously rewritable via T(x,t) |
| Nonlinearity | Weak (Kerr effect) | Strong (Forchheimer, bubbles, thermal) |
| Wave modes | Single (EM) | Multiple (fast P, slow P, shear) |
| Fabrication | Lithography / 3D print | Casting / sintering + no post-fab rebuild |
| Energy storage | Minimal | Fluid compressibility + solid elasticity + thermal |

### 4.3 The Unique Selling Point

> **No existing system combines:** 3D porous substrate + acoustic/elastic waves + optothermal reconfiguration + trainable nonlinearity.

Zangeneh-Nejad et al. (2021) explicitly identified porous media as **underexplored** for metamaterial computing. This project fills that gap.

---

## 5. Open Questions (Ranked by Priority)

### Critical (Must Answer Before Simulation)

1. **Architecture commitment:** Architecture A, B, or C? (User currently evaluating)
2. **Scaffold material:** What porous solid? (3D printed lattice, sintered beads, reticulated foam, or keep generic?)
3. **Operating fluid:** Water? Oil? Something with stronger c(T) variation?
4. **Frequency regime:** 1 MHz (per Y-junction design doc) or different?
5. **Laser parameters:** What wavelength, power, spot size? Does fluid absorb at that wavelength?

### High Priority (Needed for FDTD/OpenFOAM Setup)

6. **Biot parameter values:** What are realistic K, mu, phi, k, alpha_inf for the chosen scaffold?
7. **Temperature range:** What DeltaT is achievable? Is c(T) variation large enough to create meaningful wave-path changes?
8. **Thermal diffusion limit:** How localized can T(x) be before blur destroys the "weight" pattern?
9. **Bubble nucleation threshold:** At what T/P does nucleation occur? Is it repeatable?
10. **Forchheimer threshold:** What source amplitude activates inertial nonlinearity in the chosen pore geometry?

### Medium Priority (Needed for Results Interpretation)

11. **Slow wave role:** Does the strongly attenuated slow Biot wave serve as a "memory" channel?
12. **Shear wave utility:** Can shear polarization be used as an extra information channel?
13. **Thermal lensing quantification:** What focal lengths are achievable with realistic DeltaT gradients?
14. **Logic contrast:** What amplitude ratio between "high" and "low" output states is achievable?

### Low Priority / Future Work

15. **Dynamic reconfiguration:** Can T(x,t) be varied during wave propagation for time-varying weights?
16. **3D scaling:** Does 2D diffractive logic extrude cleanly to 3D porous volume?
17. **Experimental feasibility:** What is the minimal demonstrable experiment (Y-junction + laser + hydrophone)?

---

## 6. What We Learned Today (2026-05-31)

| Topic | Key Insight |
|-------|-------------|
| **Architecture** | Three viable options; A (fixed scaffold + thermal) is recommended core with C as stretch |
| **Thermal control** | Laser heating of fluid is feasible; spatial resolution limited by thermal diffusion (~37 um in 10 ms) |
| **Timescales** | Acoustic (us) << thermal (ms) << bubble dissolution (100 ms-s). Static programming is clean. |
| **Nonlinearity** | Six distinct mechanisms; porous media are NOT inherently linear |
| **Bubbles** | Strongest nonlinearity source; B/A can increase by 10^2-10^4x with tiny bubble fraction |
| **Biot coupling** | Even individually linear solid + fluid = nonlinear coupled system (Tong et al. 2017) |
| **Trainability** | Temperature field T(x) controls both linear weights (c(T)) and nonlinear activation strength |

---

## 7. Next Steps

### Immediate (This Week)
- [ ] **User commits to architecture** (A, B, or C)
- [ ] **Choose scaffold material** (or commit to generic)
- [ ] **Define operating fluid and temperature range**
- [ ] **Specify laser parameters** (or keep abstract)

### Short-Term (Next 2 Weeks)
- [ ] **Download and read Tong et al. 2017** (nonlinear Biot theory)
- [ ] **Download and read Moufid et al. 2022** (Forchheimer in time domain)
- [ ] **Compile c(T), rho(T), mu(T) data** for chosen fluid across operating range
- [ ] **Estimate thermal diffusion length** for realistic heating pulse durations
- [ ] **Sketch 2D Y-junction geometry** with thermal zones annotated

### Medium-Term (Next Month)
- [ ] **Implement linear Biot FDTD** (baseline, no thermal)
- [ ] **Add thermal modulation** (c(T) variation, static T(x))
- [ ] **Validate against analytical Biot solutions**
- [ ] **Quantify logic contrast** (high vs. low output states)
- [ ] **Set up OpenFOAM poroelastic case** (or rhoPimpleFoam with effective fluid)

### Long-Term (July-August)
- [ ] **Diffractive slice with thermal tuning**
- [ ] **Minimal optimisation study** (2-10 thermal parameters)
- [ ] **Bubble nucleation simulation** (stretch)
- [ ] **Thesis chapter updates** (introduction + literature review rewrite)
- [ ] **Physical concept / experimental sketch**

---

## 8. File Inventory

| File | Purpose | Status |
|------|---------|--------|
| `Notes/0f839564.md` | Simulation roadmap | Complete, may need update post-architecture choice |
| `Notes/architecture-decision-framework.md` | Three architectures compared | Complete |
| `Notes/thermal-and-nonlinearity-summary.md` | Thermal + nonlinear physics | Complete |
| `Notes/project-state-2026-05-31.md` | **This file — master state** | Complete |
| `Report/references.bib` | Bibliography | Updated (+11 entries today) |
| `Report/chapters/introduction.tex` | Thesis intro | **Outdated** — still describes chemical direction |
| `Report/chapters/literature_review.tex` | Thesis lit review | **Outdated** — still describes chemical direction |
| `Report/chapters/methodology.tex` | Thesis methods | Skeleton only |

---

*End of state document. Next update expected after architecture commitment.*
