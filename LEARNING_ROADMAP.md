# Learning Roadmap: Physics of Fluid Neural Networks

*Compiled for Cranfield IRP 2025-26*

---

## 1. Fundamentals of Acoustics

### What to Learn
- Linear acoustic wave equation (first-order and second-order forms)
- Sound speed in gases: `c = sqrt(gamma * R * T)`
- Acoustic impedance: `Z = rho * c`
- Reflection/transmission at interfaces
- Boundary conditions: Dirichlet, Neumann, Robin, absorbing

### Key Resources

**Textbooks:**
- *Fundamentals of Acoustics* — Kinsler & Frey (Chapters 1-6)
- *Theoretical Acoustics* — Morse & Ingard (Chapter 6: sound in ducts)
- *Acoustics* — Beranek & Mellow (Chapter 3: reflection and transmission)

**Papers in your references.bib:**
- `biot1956theory` — Biot's classic on poroelastic wave propagation
- `tong2017nonlinear` — Nonlinear acoustics in porous materials

**Online:**
- Sengpiel Audio: Temperature dependence of sound speed
  - https://sengpielaudio.com/TemperatureSound.htm
  - Practical formulas: `c ≈ 331 + 0.6*T(°C)` m/s

**Focus for thesis:**
- How does temperature change sound speed? (Section 2.1 of any acoustics text)
- What happens at an interface between hot and cold air? (Impedance mismatch)
- Why do we need 2nd-order accuracy in space and time?

---

## 2. FDTD Numerical Methods

### What to Learn
- Yee staggered grid: why p at centers, u/v at faces
- Leapfrog time stepping: stability, accuracy
- CFL condition: `dt < dx / (c * sqrt(2))`
- Numerical dispersion: grid resolution requirements
- Absorbing boundary conditions: impedance matching vs PML

### Key Resources

**Textbooks:**
- *Computational Electrodynamics: The Finite-Difference Time-Domain Method* — Taflove & Hagness
  - Chapter 1: 1D FDTD (the acoustic analog is identical)
  - Chapter 3: Yee cell and leapfrog
  - Chapter 7: Absorbing boundary conditions

**Papers in your references.bib:**
- `hughes2019wave` — Uses FDTD as the forward model for wave-based RNN

**Online:**
- Purdue ECE 604 Lecture 37: FDTD and Yee Algorithm
  - https://engineering.purdue.edu/wcchew/ece604s20/Lecture%20Notes/Lect37.pdf
  - Covers CFL, numerical dispersion, grid resolution rules

- arXiv: "Acoustic Wave Modeling Using 2D FDTD" (2025)
  - https://arxiv.org/html/2507.09376v1
  - Modern FDTD for acoustics with code examples

**Focus for thesis:**
- Derive the CFL condition from the dispersion relation
- Understand why staggered grid eliminates need for interpolation
- Compare impedance BC vs Mur ABC vs PML

---

## 3. Periodic Structures & Bragg Gratings

### What to Learn
- Bragg condition: `d = m * lambda / 2`
- Band gaps in periodic media
- Coupled-mode theory: `kappa` (coupling coefficient)
- Reflection from finite gratings: `R = tanh²(kappa * L)`
- Frequency selectivity: bandwidth vs number of periods

### Key Resources

**Textbooks:**
- *Fundamentals of Photonic Crystals* — Joannopoulos et al. (Chapters 4-5)
  - Same math applies to acoustic crystals

**Papers:**
- "Bragg waveguide grating as a 1D photonic band gap structure"
  - https://www.photonics.intec.ugent.be/download/pub_1251.pdf
  - Excellent tutorial on Bragg gratings, band diagrams, FB modes

- "Narrow-bandwidth Bragg grating filter based on Ge-Sb-Se"
  - https://flip.lab.westlake.edu.cn/202204.pdf
  - Practical design equations: `lambda_B = 2 * Lambda * n_eff`

**Online:**
- Ansys Lumerical: Bragg grating initial design with FDTD
  - https://optics.ansys.com/hc/en-us/articles/360042304394
  - Step-by-step FDTD simulation of a Bragg grating

**Focus for thesis:**
- Derive Bragg condition from constructive interference of partial reflections
- Understand `kappa` and how it depends on impedance contrast
- Calculate bandwidth: `Delta_lambda = lambda² * kappa / (pi * n_g)`
- Our result: 20 periods at d=24mm gave ratio 1776 — analyze why

---

## 4. Physical Neural Networks / Wave Computing

### What to Learn
- How wave propagation maps to neural network computation
- Forward pass = wave propagation, weights = material properties
- Training = optimizing geometry/material to minimize loss
- Types: diffractive networks, reservoir computing, mechanical networks

### Key Resources (ALL in your references.bib)

**Core papers:**
- `hughes2019wave` — **THE foundational paper for your thesis**
  - "Wave physics as an analog recurrent neural network"
  - Shows mapping between Maxwell's equations and RNN dynamics
  - Read Sections 1-3 carefully

- `wright2022deep` — **Training physical neural networks with backprop**
  - "Deep physical neural networks trained with backpropagation"
  - Demonstrates physical realization and training
  - Nature 2022 — highly cited

- `stern2020supervised` — **Learning in mechanical systems**
  - "Supervised learning through physical changes in a mechanical system"
  - PNAS 2020 — shows training of physical networks

- `stern2021physical` — **Theory of physical networks**
  - "Supervised learning in physical networks: from machine learning to learning machines"
  - Phys. Rev. X 2021 — theoretical framework

**Review papers:**
- `zangenehnejad2021analogue` — "Analogue computing with metamaterials"
  - Nature Reviews Materials 2021
  - Broad overview of metamaterial computing

- `silva2014mathematical` — "Performing mathematical operations with metamaterials"
  - Science 2014 — early work on metamaterial computing

**Application-specific:**
- `lin2018alloptical` — "All-optical machine learning using diffractive deep neural networks"
  - Science 2018 — 3D-printed diffractive optical network

- `zuo2018acoustic` — "Acoustic analog computing system based on labyrinthine metasurfaces"
  - Sci. Rep. 2018 — acoustic computing demonstration

**Online:**
- Science Robotics: "Mechanical neural networks: Architected materials that learn behaviors"
  - https://www.science.org/doi/10.1126/scirobotics.abq7278
  - Mechanical implementation of trainable networks

**Focus for thesis:**
- Map your FDTD equations to Hughes' RNN formulation
- Explain how temperature field T(x,y) = weights
- Cite Wright et al. for physical training methodology
- Cite Stern et al. for theoretical justification

---

## 5. Porous Media Acoustics

### What to Learn
- Biot theory: two-phase wave propagation (fast and slow waves)
- Equivalent fluid models: Zwikker-Kosten, Johnson-Champoux-Allard
- Darcy drag: viscous attenuation in pores
- Key parameters: porosity phi, tortuosity, permeability K

### Key Resources

**Textbooks:**
- *Propagation of Sound in Porous Media* — Allard & Atalla
  - THE standard reference on porous acoustics
  - Covers JCA, JCAL, Zwikker-Kosten models

**Papers in your references.bib:**
- `biot1956theory` — Biot's original 1956 paper
  - Start with the abstract and introduction
  - Skip the full derivation on first read

**Additional papers:**
- "An equivalent fluid model based finite-difference time-domain algorithm for sound propagation in porous material with rigid frame"
  - Zhao et al., JASA 2018
  - Directly relevant to your FDTD work

- "Energy analysis and discretization of the time-domain equivalent fluid model for wave propagation in rigid porous media"
  - Moufid et al., 2022
  - Stability proofs and time-domain formulations

- "Time-domain simulations of sound propagation in a flow duct with extended-reacting liners"
  - Alomar et al., J. Sound Vib. 2021
  - FDTD with ADE method for porous liners

**Online:**
- COMSOL Poroacoustics Models documentation
  - https://manuals.plus/m/83c5324975253a7b163e29a70417d544c7a5c513393d371e2aa73a88c2a682f7.pdf
  - Overview of all porous fluid models

**Focus for thesis:**
- Understand when equivalent fluid model is valid (rigid frame)
- How does Darcy drag create frequency-dependent attenuation?
- Compare to your temperature-phase-shift mechanism

---

## 6. Thermoacoustics

### What to Learn
- How temperature affects sound speed: `c = c0 * sqrt(T/T0)` for ideal gas
- Thermal diffusion: timescales and steady-state solutions
- Heat transfer in ducts: convection vs conduction
- Thermoacoustic engines (optional — interesting but not directly relevant)

### Key Resources

**Textbooks:**
- *Thermoacoustics* — Swift (for thermoacoustic engines/heat pumps)
  - Chapter 2: acoustic approximation and governing equations

**Online:**
- Sengpiel Audio (cited above)
  - Practical formulas and calculator for c(T)

**Focus for thesis:**
- Derive `c = c0 * sqrt(T/T0)` from ideal gas law
- Calculate thermal diffusion time: `tau = L² / D`
- Justify quasi-static assumption: `tau_thermal >> tau_acoustic`

---

## 7. Optimization in Physical Systems

### What to Learn
- Gradient-free optimization: random search, Nelder-Mead, CMA-ES
- Gradient-based: backpropagation through physics (adjoint method)
- Constraints: bounds, physical feasibility
- Challenges: non-convexity, local minima, computational cost

### Key Resources

**Papers in your references.bib:**
- `wright2022deep` — Backpropagation through physical systems
- `stern2021physical` — Theory of learning in physical networks

**Additional:**
- "Training of Physical Neural Networks" — Momeni et al., 2024
  - https://hal.science/hal-04775753v1/file/2406.03372v1%20%281%29.pdf
  - Comprehensive review of training methods

**Focus for thesis:**
- Why L-BFGS-B? (smooth, bounded, medium-dimensional)
- Why not gradient descent? (no automatic differentiation through FDTD yet)
- Random search + local refinement = standard approach

---

## Suggested Reading Order

### Week 1 (This week)
1. **Hughes et al. 2019** (`hughes2019wave`) — Read Sections 1-3
2. **Purdue FDTD lecture** — Understand Yee grid and CFL
3. **Beranek Chapter 3** — Reflection/transmission at interfaces

### Week 2
4. **Wright et al. 2022** (`wright2022deep`) — How to train physical networks
5. **Bragg grating tutorial** (Ansys or photonics.intec.ugent.be)
6. **Taflove Chapter 7** — Absorbing boundary conditions

### Week 3
7. **Stern et al. 2021** (`stern2021physical`) — Theoretical framework
8. **Allard & Atalla** — Porous media models (Chapters 1-3)
9. **Zhao et al. 2018** — FDTD for porous media

### Week 4
10. **Zangeneh-Nejad et al. 2021** (`zangenehnejad2021analogue`) — Metamaterial computing review
11. **Lin et al. 2018** (`lin2018alloptical`) — Diffractive deep neural networks
12. **Momeni et al. 2024** — Training methods review

---

## How to Read Papers

**For Hughes 2019:**
- Print it. Read with pen in hand.
- Write down the mapping between their equations and yours.
- Figure 1: understand the RNN diagram.
- Ask: "How does my temperature field correspond to their epsilon(x,y)?"

**For Wright 2022:**
- Focus on Methods section: how did they fabricate and train?
- Look at Figure 2: training curves.
- Ask: "What loss function did they use? How is it similar to my ratio objective?"

**For Bragg grating papers:**
- Draw the geometry yourself.
- Calculate the Bragg wavelength for your parameters.
- Verify: does your simulation match the theory?

---

## Questions to Answer After Reading

1. What is the mathematical mapping between acoustic wave propagation and RNN computation? (Hughes)
2. Why does the Yee grid eliminate numerical dispersion in the simplest case? (Taflove)
3. How many grid points per wavelength do we need for 1% phase accuracy? (Purdue lecture)
4. What is the coupling coefficient `kappa` for our Bragg grating? (Bragg tutorial)
5. How does Darcy drag scale with frequency in the Zwikker-Kosten model? (Allard)
6. What is the theoretical maximum reflection from a finite Bragg grating? (coupled-mode theory)
7. How does backpropagation work in a physical system? (Wright, Stern)

---

*Last updated: 2026-06-09*
