# Conceptual Comparison: Continuous Acoustic-Wave vs. Discrete Vortex-Street Neural Computation

**Date:** 2026-06-01  
**Purpose:** Side-by-side conceptual analysis of two fluid-based physical neural network paradigms. This document informs which paradigm to pursue for full simulation in the thesis.

---

## The Two Paradigms

### Paradigm A: Continuous Analog Wave Computer (Acoustic)
**Core concept:** Acoustic pressure waves propagating through a structured fluid domain perform computation via interference, scattering, and dispersion. The medium itself is the neural network.

**Mechanism:**
- **Input:** Phase/amplitude-modulated pressure waves at inlets
- **Weights:** Spatial distribution of acoustic impedance (via pore geometry or thermal tuning)
- **Propagation:** Wave equation acts as continuous-depth recurrent neural network (Hughes et al. 2019)
- **Output:** Pressure field at readout transducers
- **Nonlinearity (activation):** Forchheimer damping, bubble resonance, or thermal stress (Tong 2017, Moufid 2022, Pham 2020)

**Key advantage:** Fully continuous — no discretisation into "neurons." Information flows as a field, not as node activations.

---

### Paradigm B: Discrete Analog Vortex Computer (Kármán Street)
**Core concept:** Arrays of cylinders in fluid flow shed vortices at Strouhal-frequency rates. Cylinder diameters act as weights; shedding frequencies act as outputs.

**Mechanism:**
- **Input:** Flow velocity / pressure perturbations at upstream boundary
- **Weights:** Cylinder diameters (St–Re relationship is sigmoid-like; Kim 2017)
- **Propagation:** Navier–Stokes vortex shedding acts as reservoir (Goto et al. 2021)
- **Output:** Vortex-shedding frequency or wake-velocity pattern (read via photoelasticity or hot-wire)
- **Nonlinearity (activation):** Hopf bifurcation at Re ≈ 47 — the St–Re sigmoid itself is the activation

**Key advantage:** Discrete, trainable weights (cylinder diameters) with a clear input→output mapping. Photoelastic stress readout is real-time and visual.

---

## Comparison Matrix

| Criterion | Acoustic Wave (Continuous) | Vortex Street (Discrete) | Winner |
|---|---|---|---|
| **Mathematical foundation** | Wave equation ↔ RNN (proven mapping, Hughes 2019) | NS equation ↔ reservoir (empirical, Goto 2021) | Acoustic |
| **Trainability** | Thermal/optical tuning of c(T), ρ(T) — smooth gradients | Mechanical adjustment of cylinder diameters — discrete steps | Vortex |
| **Weight precision** | Continuous thermal field → high resolution | Discrete diameter changes → lower resolution | Acoustic |
| **Readout ease** | Pressure transducers or microphone arrays | Photoelasticity (visual, real-time, no electronics) | Vortex |
| **Energy efficiency** | Passive propagation, no moving parts | Requires continuous flow pump | Acoustic |
| **Speed** | μs–ms (acoustic transit time) | ms–s (vortex shedding period) | Acoustic |
| **Nonlinearity strength** | Weak (Forchheimer, bubble) — needs engineering | Strong (Hopf bifurcation) — emergent | Vortex |
| **Scalability** | 2D/3D field propagation — complex | Cylinder array — modular and extensible | Vortex |
| **Fabrication** | Porous scaffold + fluid chamber | Transparent channel + cylinder array | Comparable |
| **Experimental validation** | Optical NNs proven (Wright 2022); acoustic less so | Soap-film + CFD validated (Goto 2021, Morast 2019) | Vortex |
| **Thesis novelty** | Thermal tuning of porous Biot medium is unexplored | Variable-diameter cylinder array as NN is unexplored | Both high |

---

## Computational Task Comparison

### Task 1: Logic Gate (XOR / AND)

| Aspect | Acoustic | Vortex |
|---|---|---|
| **Implementation** | Y-junction with phase-controlled inlets | Twin-vortex state with two input perturbations |
| **Mechanism** | Wave interference (constructive/destructive) | Vortex-shedding phase locking |
| **Contrast ratio** | Phase-sweep gives amplitude modulation | Re-sweep gives frequency modulation |
| **Readout** | Probe pressure amplitude | Count vortices per unit time |
| **Feasibility** | ✅ FDTD already prototyped | ⚠️ Needs CFD validation |

### Task 2: Classification (N-bit input)

| Aspect | Acoustic | Vortex |
|---|---|---|
| **Architecture** | Diffractive slice with multiple inlets | Cylinder array with spatial input encoding |
| **Training** | Adjoint method on refractive index | Reservoir computing on wake features |
| **Depth** | Continuous — no discrete layers | Single reservoir layer + linear readout |
| **Reference** | Lin et al. 2018 (optical diffractive NN) | Goto et al. 2021 (twin-vortex computer) |
| **Feasibility** | ✅ Theory mature | ⚠️ Limited to reservoir computing frame |

### Task 3: Time-Series Prediction (Memory)

| Aspect | Acoustic | Vortex |
|---|---|---|
| **Memory mechanism** | Wave reverberation in cavity / slow Biot wave | Vortex street advection delay |
| **Echo-state property** | Depends on boundary absorption | Breaks down after Hopf bifurcation |
| **Optimal regime** | Moderate damping (critical coupling) | Re ≈ 40 (twin-vortex state) |
| **Reference** | Hughes 2019 (RNN mapping) | Goto 2021 (peak at critical Re) |
| **Feasibility** | ✅ Natural RNN recurrence | ⚠️ Narrow operating window |

---

## Conceptual Synthesis: Can They Be Combined?

**Hypothesis:** A hybrid architecture could use:
- **Acoustic waves** for fast, continuous, local computation (hidden layers)
- **Vortex streets** for slow, discrete, global readout (output layer)

**Physical realisation:** A fluid channel where:
1. Acoustic signals propagate through a thermally-tuned porous pre-processing stage
2. The processed acoustic field drives flow perturbations at a cylinder array
3. The cylinder array's vortex-shedding pattern is read out photoelastically

**Problem:** The timescales don't match (acoustic = μs, vortex = ms). The acoustic computation finishes before the vortex responds. This creates a **timescale bottleneck**.

**Verdict:** Probably not worth pursuing as a combined system. Choose one paradigm.

---

## Decision Framework: Which Paradigm to Simulate?

### Choose Acoustic Wave if:
- You want a **theoretically rigorous** thesis (wave equation ↔ RNN is proven)
- You want **thermal reconfigurability** (laser-tuned weights are novel)
- You have access to **optics/thermal equipment** for experiments
- You prefer **continuous dynamics** over discrete elements
- You want to build on the **existing MATLAB FDTD** work

### Choose Vortex Street if:
- You want **visual, intuitive** demonstrations (photoelasticity is compelling)
- You want **discrete, mechanical trainability** (cylinder diameters = weights)
- You have access to a **flow loop / water tunnel**
- Your prof is **actively pushing this direction**
- You prefer **experimental validation** over theoretical depth

### The "Equal Depth" Compromise:
For the thesis, simulate **both at reduced scope**:
- **Acoustic:** 2D Y-junction logic gate + diffractive slice classification (FDTD)
- **Vortex:** 2D cylinder-array reservoir computing (OpenFOAM / Python CFD)
- **Comparison:** Same task (e.g., 2-bit XOR) on both paradigms

This gives a **genuine comparative result** without requiring full depth in either.

---

## Open Questions to Resolve Before Simulating

### For Acoustic:
1. What pore geometry gives sufficient nonlinearity for logic contrast?
2. Can thermal tuning achieve the required refractive-index contrast?
3. What's the minimum feature size for thermal weights?

### For Vortex:
1. What's the optimal cylinder spacing for wake coupling?
2. How many cylinders are needed for reservoir-computing memory?
3. Can photoelasticity resolve the vortex-shedding signal?

### For Both:
1. What metric defines "better" — energy, speed, accuracy, or trainability?
2. What baseline digital NN should they be compared against?
3. Is the comparison fair if one is continuous and one is discrete?

---

## Recommended Thesis Structure (if comparing both)

| Chapter | Content |
|---|---|
| 1. Introduction | Motivation: von Neumann bottleneck, physical computing. Two fluid paradigms introduced. |
| 2. Literature Review | Wave-based NNs (Hughes, Wright, Stern) + Vortex-based computing (Goto, Kim, Roshko) + Porous media physics (Biot, Tong) |
| 3. Methodology | FDTD framework for acoustic; OpenFOAM/CFD framework for vortex; comparison metrics |
| 4. Results | Acoustic Y-junction + diffractive slice; Vortex cylinder array; side-by-side performance table |
| 5. Conclusions | Which paradigm wins on which criteria; future experimental work |

---

## Next Actions

1. **Decide comparison metric** — what task will both paradigms attempt?
2. **Scope the simulations** — how complex can each get within thesis timeframe?
3. **Update methodology chapter** — write the dual-framework structure
4. **Run first simulations** — acoustic Y-junction parametric sweep; vortex single-cylinder St–Re verification

---

*This document is living — update as the comparison evolves.*
