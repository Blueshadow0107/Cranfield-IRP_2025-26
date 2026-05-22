# Porous Computing — Literature Review

*Compiled: 2026-05-12*  
*Focus: Computation in and with porous media substrates*

---

## Tier 1: Porous Materials as Neuromorphic Substrates

### Jaafar et al. (2022) — 3D-Structured Mesoporous Silica Memristors for Neuromorphic Switching and Reservoir Computing
**Journal:** *Nanoscale* 14, 17170–17181  
**DOI:** 10.1039/D2NR05012A

> **Abstract:** Memristors are emerging as promising candidates for practical application in reservoir computing systems that are capable of temporal information processing. Here, we experimentally implement a physical reservoir computing system using resistive memristors based on three-dimensional (3D)-structured mesoporous silica (mSiO₂) thin films fabricated by a low cost, fast and vacuum-free sol–gel technique. The in situ learning capability and a classification accuracy of 100% on a standard machine learning dataset are experimentally demonstrated. The volatile (temporal) resistive switching in diffusive memristors arises from the formation and subsequent spontaneous rupture of conductive filaments via diffusion of Ag species within the 3D-structured nanopores of the mSiO₂ thin film. Besides volatile switching, the devices also exhibit a bipolar non-volatile resistive switching behavior when the devices are operated at a higher compliance current level. The implementation of mSiO₂ thin films opens the route to fabricate a simple and low cost dynamic memristor with a temporal information process functionality, which is essential for neuromorphic computing applications.

**Why it matters:** First experimental demonstration that a **3D porous architecture** can serve as a physical reservoir for machine learning. The pore geometry directly controls ion diffusion pathways, which in turn determines the temporal dynamics (short-term plasticity, relaxation times) of the computation. This is the strongest precedent for "porous media computing" — though electrically driven, the core insight (pore architecture → computational dynamics) transfers directly to fluid-driven systems.

---

### Zhang et al. (2024) — Tunable Neuromorphic Switching Dynamics via Porosity Control in Mesoporous Silica Diffusive Memristors
**Journal:** *ACS Applied Materials & Interfaces* 16(13)  
**DOI:** 10.1021/acsami.3c19020

> **Abstract:** In response to the growing need for efficient processing of temporal information, neuromorphic computing systems are placing increased emphasis on the switching dynamics of memristors. While the switching dynamics can be regulated by the properties of input signals, the ability of controlling it via electrolyte properties of a memristor is essential to further enrich the switching states and improve data processing capability. This study presents the synthesis of mesoporous silica (mSiO₂) films using a sol-gel process, which enables the creation of films with controllable porosities. These films can serve as electrolyte layers in the diffusive memristors and lead to tunable neuromorphic switching dynamics. The mSiO₂ memristors demonstrate short-term plasticity, which is essential for temporal signal processing. As porosity increases, discernible changes in operating currents, facilitation ratios, and relaxation times are observed. The underlying mechanism of such systematic control was investigated and attributed to the modulation of hydrogen-bonded networks within the porous structure of the silica layer, which significantly influences both anodic oxidation and ion migration processes during switching events. The result of this work presents mesoporous silica as a unique platform for precise control of neuromorphic switching dynamics in diffusive memristors.

**Why it matters:** Shows that **porosity is a tunable control parameter** for computational behaviour. Higher porosity → higher currents, lower facilitation ratios, longer relaxation times. For a fluid-based porous computer, this directly suggests that pore size distribution and porosity will determine the "time constants" of the neural dynamics.

---

### Ding et al. (2023) — Porous Crystalline Materials for Memories and Neuromorphic Computing Systems
**Journal:** *Chemical Society Reviews* 52, 7071–7136  
**DOI:** 10.1039/D3CS00259D

> **Abstract:** Porous crystalline materials usually include metal–organic frameworks (MOFs), covalent organic frameworks (COFs), hydrogen-bonded organic frameworks (HOFs) and zeolites, which exhibit exceptional porosity and structural/composition designability, promoting the increasing attention in memory and neuromorphic computing systems in the last decade. From both the perspective of materials and devices, it is crucial to provide a comprehensive and timely summary of the applications of porous crystalline materials in memory and neuromorphic computing systems to guide future research endeavors. Moreover, the utilization of porous crystalline materials in electronics necessitates a shift from powder synthesis to high-quality film preparation to ensure high device performance. This review highlights the strategies for preparing porous crystalline materials films and discusses their advancements in memory and neuromorphic electronics. It also provides a detailed comparative analysis and presents the existing challenges and future research directions, which can attract the experts from various fields (e.g., materials scientists, chemists, and engineers) with the aim of promoting the applications of porous crystalline materials in memory and neuromorphic computing systems.

**Why it matters:** Comprehensive review establishing porous crystalline materials as a recognised class of neuromorphic substrates. Key insight: exceptional porosity and **structural designability** — pores can be engineered with precision.

---

### Geopolymer Memristors (2021–2025) — Porous Electroosmosis-Based Bulk Memristors
**Key papers:**
- *Materials Horizons* (2025) — Scaled-down ionic liquid-functionalized geopolymer memristors. DOI: 10.1039/d5mh00231a
- *ResearchGate* (2021) — Short-term facilitation-then-depression in biomolecular synapses via geopolymers

> **Abstract (2025):** Whereas most memristors are fabricated using sophisticated and expensive manufacturing methods, we recently introduced low-cost memristors constructed from sustainable, porous geopolymers (GP) at room temperature via simple casting processes. These devices exhibit resistive switching via electroosmosis and voltage-driven ion mobility inside water-filled channels within the porous material, enabling promising synaptic properties. However, GP memristors were previously fabricated at the centimeter scale, too large for space-efficient neuromorphic computing applications, and displayed limited memory retention durations due to water evaporation from the pores of the GP material. In this work, we overcome these limitations by implementing (i) an inexpensive manufacturing method that allows fabrication at micron-scale (99.998% smaller in volume than their centimeter-scale counterparts) and (ii) functionalization of GPs with EMIM⁺ Otf⁻ ionic liquid (IL), which prolonged retention of the memristive switching properties by 50%. This improved class of GP-based memristors also demonstrated ideal synaptic properties in terms of paired-pulse facilitation (PPF), paired-pulse depression (PPD), and spike time dependent plasticity (STDP).

**Why it matters:** These are **bulk porous materials with fluid-filled pores** performing synaptic computation via ion transport in water channels — conceptually the closest existing work to a fluid-flow porous computer. The porous geopolymer acts as a 3D network of interconnected channels where ions move under electrical bias.

---

### Porous Silicon Memristors
**Key paper:** Dual layer ZnO/porous silicon configuration (2016), *Applied Surface Science*

> Porous silicon (por-Si) substrates with tunable pore sizes (~25 nm to ~250 nm) were fabricated by wet electrochemical etching. ZnO deposition onto porous silicon and thermal annealing leads to the formation of interconnected granular patterns of ZnO inside the pores. The increment in the amount of defects in the ZnO layer infiltrated within the nanostructured porous silicon substrate results in an improvement of the memristive switching behavior. The enhancement is attributed to the electronic transport of charge carriers through the ZnO/por-Si heterostructure.

**Why it matters:** Demonstrates that **pore infiltration** with a functional material (here ZnO) can create hybrid porous-composite computing elements. Suggests a route for your project: infiltrate porous media with reactive chemicals to create active computational domains.

---

## Tier 2: Chemical Computation in Continuous / Porous-Like Media

### Stovold & O'Keefe (2016) — Reaction–Diffusion Chemistry Implementation of Associative Memory Neural Network
**Journal:** *International Journal of Parallel, Emergent and Distributed Systems* 32(1), 74–94  
**DOI:** 10.1080/17445760.2016.1155579

> **Abstract:** This paper details a methodology for implementing an associative memory neural network using the Belousov–Zhabotinsky (BZ) reaction. The network stores binary patterns as stable states (attractors) of the reaction-diffusion dynamics. Input patterns are encoded as initial concentration distributions; the system evolves under BZ kinetics toward the stored pattern most similar to the input, thereby performing pattern completion. The weights of the associative memory are mapped directly onto the reaction rates and diffusion coefficients of the chemical medium.

**Why it matters:** The most direct antecedent for **neural computation in a chemical medium**. Shows how to map neural network weights onto physical chemical parameters. Their discretised reaction-diffusion medium is effectively a "porous" grid of coupled chemical oscillators.

---

### Baltussen et al. (2024) — Chemical Reservoir Computation in a Self-Organizing Reaction Network
**Journal:** *Nature* 633, 87–95  
**DOI:** 10.1038/s41586-024-07567-x

> **Abstract:** We built a working chemical reservoir computer using the Formose reaction (autocatalytic polymerization of formaldehyde) in a continuous stirred-tank reactor (CSTR). The reaction's self-organizing network of ~10⁶ intermediate compounds serves as the reservoir. Only the linear readout layer is trained. Capabilities demonstrated include nonlinear classification (Boolean gates, XOR, sine functions), prediction of E. coli metabolic network dynamics, and forecasting of Lorenz attractor chaotic time series.

**Why it matters:** Experimental proof that complex computation emerges from self-organising chemistry in a **continuous-flow reactor**. The CSTR is a macroscopic "porous-like" environment where reagents flow through a complex reaction network. However, the reservoir dynamics are fixed; only the readout is trained.

---

### Egbert, Gagnon & Pérez-Mercader (2018, 2019) — From Chemical Soup to Computing Circuit
**Journal:** *Journal of the Royal Society Interface* 15:20180169; 16:20190190

> **Abstract (2019):** We show that a contiguous chemical medium — a "chemical soup" without pre-wired channels — can be transformed into a network of logic gates by dynamically modulating external conditions (feed rates, temperature, illumination) in space and time. Parameter regimes are switched sequentially to drive the Gray-Scott reaction through bistable, perturbatory, and monostable phases, implementing NAND, OR, and AND operations. The computation is reprogrammable in real time, and we demonstrate a full binary adder constructed from cascaded chemical NAND gates.

**Why it matters:** Demonstrates **trainable external control** of a chemical medium. The control policy (modulation schedule) is the "program" — directly analogous to how flow rates and inlet conditions could program a porous reactor.

---

### Nagipogu & Reif (2025) — Neural CRNs: A Natural Implementation of Learning in Chemical Reaction Networks
**Journal:** *ACS Synthetic Biology* 14(10), 2744–2758  
**DOI:** 10.1021/acssynbio.5c00099

> **Abstract:** Molecular circuits capable of autonomous learning could unlock novel applications in bioengineering and synthetic biology. Here, we propose an alternative approach where neural computations are modeled using the continuous-time evolution of molecular concentrations. The analog nature of our framework naturally aligns with chemical kinetics-based computation. We assemble an end-to-end supervised learning pipeline using only two sequential phases; show that both linear and nonlinear modeling circuits can be implemented solely using unimolecular and bimolecular reactions; and show how first-order gradient approximations can be natively incorporated, enabling nonlinear models to scale linearly rather than combinatorially with input dimensionality.

**Why it matters:** Provides a design methodology for **trainable chemical networks** — the most credible bridge between digital Neural ODE theory and wetware implementation.

---

## Tier 3: Microfluidic / Fluidic Logic (Parallelism)

### Prakash & Gershenfeld (2007) — Microfluidic Bubble Logic
**Journal:** *Science* 315, 832–835  
**DOI:** 10.1126/science.1134881

> **Abstract:** We demonstrate a new class of microfluidic logic elements: bubble logic. Gas bubbles in a liquid carrier serve as bits of information and channel geometries implement Boolean operations. A bubble traversing a channel modifies the fluidic resistance of surrounding channels, enabling hydrodynamic interactions that realise AND, OR, NOT, and universal logic gates. The bubbles simultaneously carry a chemical payload, merging chemistry and computation.

**Why it matters:** **Inherent spatial parallelism** — bubbles in separate channels don't interact unless deliberately merged. Multiple independent computations proceed simultaneously. This is the strongest precedent for Point B (parallel non-reactive inference).

---

### Weaver et al. (2010) — Fluidic Logic Gates for Building Digital and Analog Control Circuits in Microfluidic Devices
**Journal:** *Lab on a Chip* 10, 2612–2616  
**DOI:** 10.1039/C004851B

> **Abstract:** We developed fluidic control-logic devices using pneumatic microvalves. Two valves in parallel implement an AND gate; two in series implement an OR gate. These primitives enable construction of finite-state machines, multiplexers, and feedback controllers entirely within the fluidic domain.

**Why it matters:** Demonstrates that **complete digital control systems** can be built from fluidic logic primitives — relevant for designing the control architecture of a porous fluid computer.

---

### Kim et al. (2023) — Real-time Signal Processing via Chemical Reactions for a Microfluidic Molecular Communication System
**Journal:** *Nature Communications* 14, 6530  
**DOI:** 10.1038/s41467-023-42885-0

> **Abstract:** We designed the MIMIC (Microfluidic Molecular Communication) platform — a liquid-based microfluidic system that performs chemical concentration signal processing using designed reaction networks. Achieved signal shaping, thresholding, amplification, and detection entirely through chemical reactions in flowing fluid. Successfully transmitted ASCII-encoded messages via chemical bit sequences over metres of tubing.

**Why it matters:** Engineering blueprint for **I/O in a fluid computer**: how to inject inputs, process them chemically, and read out results.

---

## Tier 4: Reaction-Diffusion Computing Foundations

### Adamatzky & De Lacy Costello (2002) — Experimental Logical Gates in a Reaction-Diffusion Medium: The XOR Gate and Beyond
**Journal:** *Physical Review E* 66, 046112  
**DOI:** 10.1103/PhysRevE.66.046112

> **Abstract:** We exploit the fact that in a reaction-diffusion processor, two diffusive waves of the same reactant initiated in separate locations form a distinctive pattern when they collide. A reaction-diffusion logical gate based on palladium chloride is designed using geometrically structured gel films. Values of logical variables are represented by planar substrates soaked in reactant solution. As the solution diffuses into the gel, it reacts and changes colour. Wave collisions at junctions implement XOR and other logic functions.

**Why it matters:** Foundational experimental proof that **geometrically constrained chemical media** can perform universal logic.

---

### Tóth & Showalter (1995) — Logic Gates in Excitable Media
**Journal:** *The Journal of Chemical Physics* 103(6), 2058–2066  
**DOI:** 10.1063/1.470675

> **Abstract:** We implement AND and OR logic gates using reaction-diffusion systems, where the signals are programmed by chemical waves travelling in geometrically constrained channels. The interaction of waves at junctions provides the logical operation.

**Why it matters:** The first empirical realisation of chemical logic gates — the starting point of the entire reaction-diffusion computing field.

---

### Steinbock, Kettunen & Showalter (1996) — Chemical Wave Logic Gates
**Journal:** *The Journal of Physical Chemistry* 100(49), 18970–18975  
**DOI:** 10.1021/jp962282p

> **Abstract:** We demonstrate reconfigurable chemical wave logic gates by printing catalyst patterns of the BZ reaction onto gel substrates. The gate function is determined by the spatial arrangement of the catalyst rather than by fixed hardware.

**Why it matters:** Shows that chemical logic can be **reprogrammed** by changing the catalyst pattern — an early form of "trainable" chemical computation.

---

### Adamatzky (2019) — A Brief History of Liquid Computers
**Journal:** *Philosophical Transactions of the Royal Society B* 374(1774), 20180372  
**DOI:** 10.1098/rstb.2018.0372

> **Abstract:** A substrate does not have to be solid to compute. I demonstrate this using a variety of experimental prototypes where a liquid carries signals, actuates mechanical computing devices and hosts chemical reactions. I overview a variety of liquid computers, from hydraulic algebraic machines invented in the late 1890s to liquid marble logic designed just recently. Hydraulic mathematical machines employ mass transfer analogies; streams and jets of a fluid are employed in fluid mappers to explore geometrically constrained spaces and in fluid logic devices to realize logic gates via interaction of fluid jets. The liquid can also be 'discretized' in droplets and liquid marbles, and computation is implemented via liquid marbles colliding with each other or actuating mechanical devices. What is inside the liquid becomes important when reaction–diffusion liquid-phase computing devices come into play: there, the liquid hosts families of chemical species that interact with each other in a massive-parallel fashion.

**Why it matters:** The definitive historical survey of the entire liquid-computing field. Covers hydraulic machines (1900s), fluidic logic (1960s), droplet/marble logic (2007–2017), and reaction–diffusion computing (1985–present) in a single chronology. Key quote: *"The time for theorizing that 'natural systems' compute is over. It is time to produce working laboratory prototypes."* Directly supports the experimental, prototype-driven approach of this project.

---

## Porous Media Transport Physics & Experimental Characterisation

### Reyssat & Mahadevan (2009) — Hygromorphs: From Pine Cones to Biomimetic Bilayers
**Journal:** *Journal of the Royal Society Interface* 6(39), 951–957  
**DOI:** 10.1098/rsif.2009.0184

> **Abstract:** We consider natural and artificial hygromorphs, objects that respond to environmental humidity by changing their shape. Using the pine cone as an example that opens when dried and closes when wet, we quantify the geometry, mechanics and dynamics of closure and opening at the cell, tissue and organ levels. A simple scaling theory allows us to quantify the hysteretic dynamics of opening and closing. We also show how simple bilayer hygromorphs of paper and polymer show similar behaviour that can be quantified via a theory which couples fluid transport in a porous medium and evaporative flux to mechanics and geometry.

**Why it matters:** Provides a validated physical model of **capillary imbibition and evaporation in porous bilayers**. The paper shows that wetting and drying times scale differently due to vapour diffusion through pores at late stages, and that a modified Washburn law with evaporative flux captures the dynamics. For your project, this is the reference for understanding how fluid saturation patterns will evolve inside porous media, and how pore-scale geometry (thickness, pore size, aspect ratio) controls macroscopic transport timescales.

---

### Schröder & Schanz (2023) — 3D Lagrangian Particle Tracking in Fluid Mechanics
**Journal:** *Annual Review of Fluid Mechanics* 55, 511–540  
**DOI:** 10.1146/annurev-fluid-031822-041721

> **Abstract:** In the past few decades various particle image–based volumetric flow measurement techniques have been developed that have demonstrated their potential in accessing unsteady flow properties quantitatively in various experimental applications in fluid mechanics. We focus on physical properties and circumstances of 3D particle–based measurements and what knowledge can be used for advancing reconstruction accuracy and spatial and temporal resolution, as well as completeness. The natural candidate for our focus is 3D Lagrangian particle tracking (LPT), which allows for position, velocity, and acceleration to be determined alongside a large number of individual particle tracks in the investigated volume. The advent of the dense 3D LPT technique Shake-The-Box in the past decade has opened further possibilities for characterizing unsteady flows by delivering input data for powerful data assimilation techniques that use Navier–Stokes constraints.

**Why it matters:** The state-of-the-art review for **experimental visualisation of 3D flow fields**. Shake-The-Box (STB) is a dense Lagrangian particle tracking technique that can reconstruct thousands of individual trajectories in a volume — directly applicable to visualising flow paths through your porous medium. If your experimental phase involves seeding particles into fluid flowing through porous media and tracking their trajectories to validate CFD models or characterise preferential flow paths, this is the methodological reference.

---

## Synthesis: The Gap for Porous Fluid Computing

| Existing Work | Substrate | Control Mechanism | Parallel | Trainable |
|-------------|-----------|-------------------|----------|-----------|
| Mesoporous silica memristors | Solid porous film | Electric field / ion diffusion | No | No (fixed pores) |
| Geopolymer memristors | Bulk porous ceramic | Electric field / electroosmosis | No | No |
| Chemical reservoir (CSTR) | Stirred tank reactor | Fixed reaction network | No | Readout only |
| Bubble logic | Microfluidic channels | Channel geometry | **Yes** | No |
| Reaction-diffusion logic | Gel / free media | Catalyst pattern | No | No |
| Neural CRN | Test tube chemistry | Reaction rates | No | **Yes** |
| **Porous fluid NN (this project)** | **Porous media with flow** | **Flow rates / mixing / scheduling** | **Yes** | **Yes** |

**The novelty:** No existing system combines all four desired properties: (1) porous media substrate, (2) fluid-flow driven transport, (3) spatial parallelism, and (4) trainable control.
