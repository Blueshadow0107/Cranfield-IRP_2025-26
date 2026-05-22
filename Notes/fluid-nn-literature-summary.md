# Fluid Neural Networks over Continuous Computations — Working Literature Summary

*Compiled: 2026-05-12*  
*Project: Physical fluid-based neural computation (Project 11)*  
*Supervisors: Prof. Weisi Guo, Prof. Takis Tsoutsanis*

---

## Project Summary

This project investigates a **physical fluid-based neural computer** — a device where controlled mixing of reactive fluids performs neural network inference. Unlike digital "Liquid Neural Networks" (which simulate continuous dynamics on silicon), this project builds actual continuous computers from fluids and chemicals.

**Two key advantages sought:**
- **A. Continuity:** Processing occurs across the whole architecture space, not at discrete nodes
- **B. Parallelism:** Non-reactive fluid streams allow multiple independent inference processes simultaneously

**Three phases:**
1. Model a controllable physical space (reactor/microfluidic geometry) with input fluids and output detection
2. Train control units (flow rates, valve timing, mixing topology) to encode neural functions for parallel tasks
3. Benchmark against silicon neural networks

---

## 1. Reaction-Diffusion Computing

The foundational literature for chemical computation using excitable media.

### Tóth & Showalter (1995) — Logic Gates in Excitable Media
**Journal:** *J. Chem. Phys.* 103, 2058  
**Key contribution:** First empirical realisation of chemical logic gates (AND, OR) using the Belousov-Zhabotinsky (BZ) reaction. Waves guided through geometric channels interact at junctions to perform Boolean operations.

### Steinbock, Kettunen & Showalter (1996) — Chemical Wave Logic Gates
**Journal:** *J. Phys. Chem.* 100, 18970  
**Key contribution:** "Printed" catalyst patterns on gel substrates create reconfigurable logic gates. Gate function determined by spatial catalyst arrangement, not fixed hardware.

### Adamatzky & De Lacy Costello (2002) — Chemical XOR Gate
**Journal:** *Phys. Rev. E* 66, 046112  
**Key contribution:** Experimental XOR gate in palladium chloride/potassium iodide reaction in structured gel. Wave collision and annihilation at junctions implements the logic.

### Adamatzky (2002, 2005) — Monographs
- *Collision-Based Computing* (Springer, 2002)
- *Reaction-Diffusion Computers* (Elsevier, 2005, with De Lacy Costello & Asai)
**Key contribution:** Comprehensive theoretical and experimental treatment of chemical computation. Establishes two paradigms: geometrically-constrained (fixed channels) vs. free-space/collision-based (architectureless).

### Magnasco (1997) — Chemical Kinetics is Turing Universal
**Journal:** *Phys. Rev. Lett.* 78, 1190  
**Key contribution:** Proved that logic gates can be constructed in homogeneous chemical kinetics with computational power equivalent to a Turing machine.

### Egbert, Gagnon & Pérez-Mercader (2018, 2019) — Chemical Soup to Computing Circuit
**Journal:** *J. R. Soc. Interface* 15:20180169; 16:20190190  
**Key contribution:** Showed a contiguous chemical medium (no pre-wired channels) can be transformed into logic gates by dynamically modulating external conditions (feed rates, temperature) in space and time. Demonstrated full binary adder from cascaded NAND gates. **Directly relevant to Phase 2** — the "control units" modulate external conditions to program the medium.

---

## 2. Neural Architectures in Chemical Media

Papers that implement neural-network primitives (neurons, associative memory) directly in chemistry.

### Hjelmfelt, Weinberger & Ross (1991) — Chemical Implementation of Neural Networks
**Journal:** *PNAS* 88, 10983  
**Key contribution:** Designed neural networks and finite state machines using chemical diodes. Showed universal Turing machine execution is possible with connected chemical diodes.

### Rambidi, Maximychev & Usatov (1994) — Molecular Neural Network Devices
**Journal:** *Biosystems* 33, 125  
**Key contribution:** Proposed "molecular neural network devices" based on nonlinear dynamic media. Framed physical realisation of NNs in molecular systems as a design problem.

### Gentili et al. (2012) — BZ "Chemical Neuron"
**Journal:** *Int. J. Unconventional Computing* 8, 177  
**Key contribution:** A single BZ reaction site functions as a binary/fuzzy logic processor with thresholding, saturation, and refractory behaviour — direct analogue of biological neuron.

### Górecki & Górecka (2009) — Information Processing with Structured Excitable Medium
**Journal:** *Natural Computing* 8, 473  
**Key contribution:** Chemical diodes, counters, cross-junction logic. Building blocks for chemical neural networks.

### Sielewiesiuk & Górecki (2001) — Logical Functions at Cross Junctions
**Journal:** *J. Phys. Chem. A* 105, 8189  
**Key contribution:** Complete truth table for wave interactions at four-way junctions of excitable media.

### Gruenert et al. (2015) — Networks of Computing Chemical Droplet Neurons
**Journal:** *Int. J. Neural Systems* 25, 1450032  
**Key contribution:** Interconnected BZ droplets communicating through lipid membranes form functional networks with collective computational properties. Information-theoretic analysis of droplet-droplet communication.

### Stovold & O'Keefe (2012, 2016, 2017) — Associative Memory in Reaction-Diffusion Chemistry
**Publications:**
- *Simulating Neurons in Reaction-Diffusion Chemistry* (IPCAT 2012)
- *Reaction-Diffusion Chemistry Implementation of Associative Memory Neural Network* (Int. J. Parallel Emergent Distrib. Syst. 2016)
- *Associative Memory in Reaction-Diffusion Chemistry* (Springer 2017)

**Key contribution:** The most direct antecedent. Showed BZ reaction implements an associative memory (recurrent NN with attractor dynamics). Binary patterns encoded as initial concentration distributions; system evolves toward nearest stored pattern. **Methodology directly applicable to Phase 2** — mapping NN weights onto chemical reaction rates and diffusion coefficients.

---

## 3. Microfluidic Logic and Programmable Fluidics

Papers on using hydrodynamics (rather than chemistry) for computation — directly relevant to parallelism and control.

### Prakash & Gershenfeld (2007) — Microfluidic Bubble Logic
**Journal:** *Science* 315, 832  
**Key contribution:** Gas bubbles in liquid carrier serve as information bits; channel geometries implement AND, OR, NOT, universal logic. Bubbles carry chemical payload. **Directly relevant to Point B (parallelism)** — bubbles in separate channels don't interact unless merged, enabling spatially parallel computation.

### Weaver et al. (2010) — Fluidic Logic Gates
**Journal:** *Lab on a Chip* 10, 2612  
**Key contribution:** Pneumatic microvalves as fluidic resistor-transistor circuits. Parallel valves = AND gate; series valves = OR gate. Finite-state machines and feedback controllers in fluidic domain.

### Kim et al. (2023) — MIMIC Microfluidic Platform
**Journal:** *Nature Communications* 14, 6530  
**Key contribution:** Signal processing (amplification, thresholding, shaping, detection) through designed chemical reactions in flowing fluid. ASCII message transmission via chemical bit sequences. **Relevant to Phase 1 I/O design** — input injection, reaction, optical/electrochemical readout.

### Recent Pneumatic Computers for Microfluidics
**Key contribution:** Reprogrammable logic arrays in layered microfluidic membranes. Clock rates up to 5 Hz. Demonstrates integrated fluidic control without external electronics.

---

## 4. Trainable Chemical Networks and Reservoir Computing

### Baltussen et al. (2024) — Chemical Reservoir Computing
**Journal:** *Nature* 633, 87  
**Key contribution:** Formose reaction in CSTR as reservoir. ~10⁶ intermediate compounds serve as high-dimensional reservoir. Linear readout trained for classification, prediction, chaotic forecasting. **Relevant to Phase 1** — CSTR is inherently fluid and continuous. But reservoir dynamics are fixed; only readout is trained.

### Nagipogu & Reif (2025) — Neural CRNs
**Journal:** *ACS Synthetic Biology* 14, 2744  
**Key contribution:** Trainable chemical reaction networks implementing Neural ODEs. Only unimolecular/bimolecular reactions required. End-to-end supervised learning in two phases. **Most credible bridge to trainable fluid NNs** — directly maps to Phase 2 objective of training control parameters.

### Wetzels (2025) — Thinking in Molecules (PhD thesis, Radboud)
**Key contribution:** Extended Formose reservoir analysis. Temporal memory properties, heterogeneous memory buffers with distinct forgetting rates. "Fading memory" from continuous inflow/outflow.

---

## 5. Mathematical / Control Background

### Chen et al. (2018) — Neural Ordinary Differential Equations
**Venue:** NeurIPS 2018 (Best Paper)  
**arXiv:** 1806.07366  
**Key contribution:** Hidden state evolves continuously: `dh/dt = f(h(t), t, θ)`. While digital in implementation, provides mathematical language for continuous computation in physical media. **Relevant to Phase 2** — the vector field `f` corresponds to reaction kinetics + advection; parameters `θ` correspond to controllable physical variables (flow rates, concentrations, temperature).

### Neural Fluidic System Design and Control (2024)
**arXiv:** 2405.14903  
**Key contribution:** Differentiable fluid simulation for optimising control of fluidic systems with dynamic boundaries. Backpropagates gradients through Navier-Stokes solver to optimise geometry and control. **Relevant to Phases 1–2** — CFD-informed design + gradient-based control optimisation.

---

## 6. Synthesis: What This Project Adds

### Current state of the art
| Approach | Fixed/Trainable | Parallel | Neural capability |
|----------|-----------------|----------|-------------------|
| Reaction-diffusion logic | Fixed geometry | No | Boolean only |
| Chemical associative memory | Fixed kinetics | No | Attractor NN |
| Bubble logic | Fixed geometry | Yes | Boolean only |
| Chemical reservoir | Readout only | No | Linear readout of RNN |
| Neural CRN | Trainable rates | No | Full NN (in silico) |

### Research gaps this project addresses
1. **Trainable fluid architecture** — No existing system trains hydrodynamic control parameters (flow rates, mixing topology, inlet scheduling) to optimise neural inference.
2. **Parallel neural inference** — Bubble logic is parallel but Boolean; chemical NNs are serial. This project seeks parallel + neural.
3. **CFD-informed design** — Most chemical computers use simple geometries. This project uses CFD to design optimised reactor spaces.
4. **Silicon benchmarking** — Few studies systematically compare chemical/fluidic computers against silicon baselines.

### Positioning
This project sits at the intersection of:
- Reaction-diffusion computing (chemical dynamics)
- Microfluidic logic (hydrodynamic control + parallelism)
- Reservoir computing (high-dimensional physical reservoir)
- Optimal control (training control policies)

It extends each: from Boolean to neural, from fixed to trainable, from serial to parallel, from heuristic design to CFD optimisation.

---

## Bibliography (Quick Reference)

| Citation | Year | Topic | Key URL/DOI |
|----------|------|-------|-------------|
| Tóth & Showalter | 1995 | First chemical logic gates | J. Chem. Phys. 103, 2058 |
| Steinbock et al. | 1996 | Printed catalyst logic gates | J. Phys. Chem. 100, 18970 |
| Magnasco | 1997 | Turing-universal chemical kinetics | PRL 78, 1190 |
| Adamatzky | 2002 | Collision-based computing | Springer |
| Adamatzky et al. | 2005 | Reaction-diffusion computers | Elsevier |
| Adamatzky & De Lacy Costello | 2002 | Chemical XOR gate | PRE 66, 046112 |
| Hjelmfelt et al. | 1991 | Chemical neural networks / Turing machines | PNAS 88, 10983 |
| Rambidi et al. | 1994 | Molecular neural network devices | Biosystems 33, 125 |
| Prakash & Gershenfeld | 2007 | Bubble logic | Science 315, 832 |
| Górecki & Górecka | 2009 | Structured excitable media | Nat. Comput. 8, 473 |
| Sielewiesiuk & Górecki | 2001 | Cross-junction logic | J. Phys. Chem. A 105, 8189 |
| Gentili et al. | 2012 | BZ "chemical neuron" | IJUC 8, 177 |
| Stovold & O'Keefe | 2012–2017 | Associative memory in RD chemistry | IPCAT 2012; IJPEDS 2016 |
| Gruenert et al. | 2015 | Droplet neuron networks | Int. J. Neural Syst. 25, 1450032 |
| Weaver et al. | 2010 | Fluidic control-logic devices | Lab Chip 10, 2612 |
| Kim et al. | 2023 | MIMIC microfluidic platform | Nat. Commun. 14, 6530 |
| Egbert et al. | 2018, 2019 | Modulated chemical soup → logic | J. R. Soc. Interface 15, 16 |
| Baltussen et al. | 2024 | Formose chemical reservoir | Nature 633, 87 |
| Wetzels | 2025 | Formose reservoir extended | Radboud PhD thesis |
| Nagipogu & Reif | 2025 | Neural CRNs (trainable) | ACS Synth. Biol. 14, 2744 |
| Chen et al. | 2018 | Neural ODEs (math framework) | NeurIPS; arXiv:1806.07366 |
| Neural Fluidic Control | 2024 | Differentiable fluid simulation | arXiv:2405.14903 |
