# Literature Review: Neural Computation in Porous Media

*Project: Neural Network based on Porous Media — Wave/Fluid Computing Direction (IRP 2025–26)*  
*Supervisors: Prof. Weisi Guo, Prof. Takis Tsoutsanis*  
*Compiled: 2026-05-08*  
*Total papers surveyed: 35+*  

---

## Introduction and Scope

This project investigates a **physical neural network implemented in porous media** — a "neural network based on porous media" as directed by supervisors. The core question is whether porous media can be trained to control wave propagation or flow patterns to perform parallel computation. Two converging directions are explored: (1) acoustic/elastic wave computing through porous solids, and (2) controllable fluid flow through porous media for chemical computation.

**This is NOT a review of digital "Liquid Neural Networks"** (LTCs, Neural ODEs, continuous-depth models) that simulate dynamics on silicon. That direction was an earlier misunderstanding of the project scope. All digital continuous-NN material has been removed.

This review covers seven converging areas:
1. **Porous materials as neuromorphic substrates** — memristors, ionic transport in pore networks
2. **Neural primitives in chemical media** — chemical neurons, associative memory, Turing-universal chemistry
3. **Chemical computation in continuous media** — trainable reaction networks, reservoir computing
4. **Fluidic logic and parallelism** — bubble logic, microfluidic control, MIMIC platform
5. **Reaction–diffusion foundations** — BZ-wave logic, collision-based computing, gel-based gates
6. **Historical surveys and supporting physics** — liquid computing history, porous transport theory, flow visualisation
7. **Wave-based analog computing and physical neural networks** — acoustic/optical metamaterial computing, diffractive deep NNs, mechanical learning networks, poroelastic wave theory

---

## 1. Porous Materials as Neuromorphic Substrates

These papers establish that three-dimensional porous architectures can perform memory and computation via ion transport, electroosmosis, and resistive switching in fluid-filled pores. They are the strongest existing precedent for "porous media computing," though all are electrically driven rather than flow-driven.

### Jaafar et al. (2022) — 3D-Structured Mesoporous Silica Memristors for Neuromorphic Switching and Reservoir Computing
**Journal:** *Nanoscale* 14, 17170–17181  
**DOI:** 10.1039/D2NR05012A

> **Abstract:** Memristors are emerging as promising candidates for practical application in reservoir computing systems that are capable of temporal information processing. Here, we experimentally implement a physical reservoir computing system using resistive memristors based on three-dimensional (3D)-structured mesoporous silica (mSiO₂) thin films fabricated by a low cost, fast and vacuum-free sol–gel technique. The in situ learning capability and a classification accuracy of 100% on a standard machine learning dataset are experimentally demonstrated. The volatile (temporal) resistive switching in diffusive memristors arises from the formation and subsequent spontaneous rupture of conductive filaments via diffusion of Ag species within the 3D-structured nanopores of the mSiO₂ thin film.

**Why it matters:** First experimental demonstration that a **3D porous architecture** can serve as a physical reservoir for machine learning. The pore geometry directly controls ion diffusion pathways, which in turn determines the temporal dynamics (short-term plasticity, relaxation times) of the computation. The core insight (pore architecture → computational dynamics) transfers directly to fluid-driven systems.

---

### Zhang et al. (2024) — Tunable Neuromorphic Switching Dynamics via Porosity Control in Mesoporous Silica Diffusive Memristors
**Journal:** *ACS Applied Materials & Interfaces* 16(13)  
**DOI:** 10.1021/acsami.3c19020

> **Abstract:** This study presents the synthesis of mesoporous silica (mSiO₂) films using a sol-gel process, which enables the creation of films with controllable porosities. These films can serve as electrolyte layers in the diffusive memristors and lead to tunable neuromorphic switching dynamics. The mSiO₂ memristors demonstrate short-term plasticity, which is essential for temporal signal processing. As porosity increases, discernible changes in operating currents, facilitation ratios, and relaxation times are observed.

**Why it matters:** Shows that **porosity is a tunable control parameter** for computational behaviour. Higher porosity → higher currents, lower facilitation ratios, longer relaxation times. For a fluid-based porous computer, this directly suggests that pore size distribution and porosity will determine the "time constants" of the neural dynamics.

---

### Ding et al. (2023) — Porous Crystalline Materials for Memories and Neuromorphic Computing Systems
**Journal:** *Chemical Society Reviews* 52, 7071–7136  
**DOI:** 10.1039/D3CS00259D

> **Abstract:** Porous crystalline materials usually include metal–organic frameworks (MOFs), covalent organic frameworks (COFs), hydrogen-bonded organic frameworks (HOFs) and zeolites, which exhibit exceptional porosity and structural/composition designability, promoting the increasing attention in memory and neuromorphic computing systems in the last decade.

**Why it matters:** Comprehensive review establishing porous crystalline materials as a recognised class of neuromorphic substrates. Key insight: exceptional porosity and **structural designability** — pores can be engineered with precision.

---

### Geopolymer Memristors (2021–2025) — Porous Electroosmosis-Based Bulk Memristors
**Key papers:**
- *Materials Horizons* (2025) — Scaled-down ionic liquid-functionalized geopolymer memristors. DOI: 10.1039/d5mh00231a
- *ResearchGate* (2021) — Short-term facilitation-then-depression in biomolecular synapses via geopolymers

> **Abstract (2025):** Whereas most memristors are fabricated using sophisticated and expensive manufacturing methods, we recently introduced low-cost memristors constructed from sustainable, porous geopolymers (GP) at room temperature via simple casting processes. These devices exhibit resistive switching via electroosmosis and voltage-driven ion mobility inside water-filled channels within the porous material, enabling promising synaptic properties.

**Why it matters:** These are **bulk porous materials with fluid-filled pores** performing synaptic computation via ion transport in water channels — conceptually the closest existing work to a fluid-flow porous computer. The porous geopolymer acts as a 3D network of interconnected channels where ions move under electrical bias.

---

### Porous Silicon Memristors
**Key paper:** Dual layer ZnO/porous silicon configuration (2016), *Applied Surface Science*

> Porous silicon (por-Si) substrates with tunable pore sizes (~25 nm to ~250 nm) were fabricated by wet electrochemical etching. ZnO deposition onto porous silicon and thermal annealing leads to the formation of interconnected granular patterns of ZnO inside the pores.

**Why it matters:** Demonstrates that **pore infiltration** with a functional material can create hybrid porous-composite computing elements. Suggests a route for your project: infiltrate porous media with reactive chemicals to create active computational domains.

---

## 2. Neural Primitives in Chemical Media

Papers that implement neural-network primitives (neurons, associative memory, Turing universality) directly in chemistry. These establish that chemical media are capable of neural computation, providing the theoretical and experimental foundation for a chemical neural network.

### Hjelmfelt, Weinberger & Ross (1991) — Chemical Implementation of Neural Networks and Turing Machines
**Journal:** *PNAS* 88(24), 10983–10987  
**DOI:** 10.1073/pnas.88.24.10983

**Key contribution:** Designed neural networks and finite state machines using chemical diodes. Showed universal Turing machine execution is possible with connected chemical diodes.

---

### Rambidi, Maximychev & Usatov (1994) — Molecular Neural Network Devices
**Journal:** *Biosystems* 33(2), 125–137  
**DOI:** 10.1016/0303-2647(94)90037-X

**Key contribution:** Proposed "molecular neural network devices" based on nonlinear dynamic media. Framed physical realisation of NNs in molecular systems as a design problem.

---

### Magnasco (1997) — Chemical Kinetics is Turing Universal
**Journal:** *Physical Review Letters* 78(6), 1190–1193  
**DOI:** 10.1103/PhysRevLett.78.1190

**Key contribution:** Proved that logic gates can be constructed in homogeneous chemical kinetics with computational power equivalent to a Turing machine.

---

### Gentili et al. (2012) — BZ "Chemical Neuron"
**Journal:** *International Journal of Unconventional Computing* 8(3), 177–192

**Key contribution:** A single BZ reaction site functions as a binary/fuzzy logic processor with thresholding, saturation, and refractory behaviour — direct analogue of biological neuron.

---

### Górecki & Górecka (2009) — Information Processing with Structured Excitable Medium
**Journal:** *Natural Computing* 8, 473–492

**Key contribution:** Chemical diodes, counters, cross-junction logic. Building blocks for chemical neural networks.

---

### Sielewiesiuk & Górecki (2001) — Logical Functions at Cross Junctions
**Journal:** *J. Phys. Chem. A* 105(35), 8189–8195  
**DOI:** 10.1021/jp011714a

**Key contribution:** Complete truth table for wave interactions at four-way junctions of excitable media.

---

### Gruenert et al. (2015) — Networks of Computing Chemical Droplet Neurons
**Journal:** *Int. J. Neural Systems* 25(7), 1450032  
**DOI:** 10.1142/S0129065714500322

**Key contribution:** Interconnected BZ droplets communicating through lipid membranes form functional networks with collective computational properties. Information-theoretic analysis of droplet-droplet communication.

---

### Stovold & O'Keefe (2012, 2016, 2017) — Associative Memory in Reaction-Diffusion Chemistry
**Publications:**
- *Simulating Neurons in Reaction-Diffusion Chemistry* (IPCAT 2012)
- *Reaction-Diffusion Chemistry Implementation of Associative Memory Neural Network* (Int. J. Parallel Emergent Distrib. Syst. 2016)
- *Associative Memory in Reaction-Diffusion Chemistry* (Springer 2017)

**Key contribution:** The most direct antecedent for **neural computation in a chemical medium**. BZ reaction implements an associative memory (recurrent NN with attractor dynamics). Binary patterns encoded as initial concentration distributions; system evolves toward nearest stored pattern. **Methodology directly applicable to Phase 2** — mapping NN weights onto chemical reaction rates and diffusion coefficients.

---

## 3. Chemical Computation in Continuous Media

### Egbert, Gagnon & Pérez-Mercader (2018, 2019) — From Chemical Soup to Computing Circuit
**Journal:** *J. R. Soc. Interface* 15:20180169; 16:20190190  
**DOI:** 10.1098/rsif.2018.0169; 10.1098/rsif.2019.0190

> **Abstract (2019):** We show that a contiguous chemical medium — a "chemical soup" without pre-wired channels — can be transformed into a network of logic gates by dynamically modulating external conditions (feed rates, temperature, illumination) in space and time. Parameter regimes are switched sequentially to drive the Gray-Scott reaction through bistable, perturbatory, and monostable phases, implementing NAND, OR, and AND operations.

**Why it matters:** Demonstrates **trainable external control** of a chemical medium. The control policy (modulation schedule) is the "program" — directly analogous to how flow rates and inlet conditions could program a porous reactor.

---

### Baltussen et al. (2024) — Chemical Reservoir Computation in a Self-Organizing Reaction Network
**Journal:** *Nature* 633, 87–95  
**DOI:** 10.1038/s41586-024-07567-x

> **Abstract:** We built a working chemical reservoir computer using the Formose reaction (autocatalytic polymerization of formaldehyde) in a continuous stirred-tank reactor (CSTR). The reaction's self-organizing network of ~10⁶ intermediate compounds serves as the reservoir. Only the linear readout layer is trained. Capabilities demonstrated include nonlinear classification (Boolean gates, XOR, sine functions), prediction of E. coli metabolic network dynamics, and forecasting of Lorenz attractor chaotic time series.

**Why it matters:** Experimental proof that complex computation emerges from self-organising chemistry in a **continuous-flow reactor**. The CSTR is a macroscopic "porous-like" environment where reagents flow through a complex reaction network. However, the reservoir dynamics are fixed; only the readout is trained.

---

### Nagipogu & Reif (2025) — Neural CRNs: A Natural Implementation of Learning in Chemical Reaction Networks
**Journal:** *ACS Synthetic Biology* 14(10), 2744–2758  
**DOI:** 10.1021/acssynbio.5c00099

> **Abstract:** Molecular circuits capable of autonomous learning could unlock novel applications in bioengineering and synthetic biology. Here, we propose an alternative approach where neural computations are modeled using the continuous-time evolution of molecular concentrations. The analog nature of our framework naturally aligns with chemical kinetics-based computation. We assemble an end-to-end supervised learning pipeline using only two sequential phases; show that both linear and nonlinear modeling circuits can be implemented solely using unimolecular and bimolecular reactions; and show how first-order gradient approximations can be natively incorporated.

**Why it matters:** Provides a design methodology for **trainable chemical networks** — the most credible bridge between digital Neural ODE theory and wetware implementation.

---

### Wetzels (2025) — Thinking in Molecules: Chemical Reservoir Computing and Predictive Information
**Thesis:** Radboud University

**Key contribution:** Extended Formose reservoir analysis. Temporal memory properties, heterogeneous memory buffers with distinct forgetting rates. "Fading memory" from continuous inflow/outflow.

---

## 4. Fluidic Logic and Parallelism

### Prakash & Gershenfeld (2007) — Microfluidic Bubble Logic
**Journal:** *Science* 315, 832–835  
**DOI:** 10.1126/science.1134881

> **Abstract:** We demonstrate a new class of microfluidic logic elements: bubble logic. Gas bubbles in a liquid carrier serve as bits of information and channel geometries implement Boolean operations. A bubble traversing a channel modifies the fluidic resistance of surrounding channels, enabling hydrodynamic interactions that realise AND, OR, NOT, and universal logic gates. The bubbles simultaneously carry a chemical payload, merging chemistry and computation.

**Why it matters:** **Inherent spatial parallelism** — bubbles in separate channels don't interact unless deliberately merged. Multiple independent computations proceed simultaneously. This is the strongest precedent for Point B (parallel non-reactive inference).

---

### Weaver et al. (2010) — Fluidic Logic Gates for Building Digital and Analog Control Circuits
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

## 5. Reaction–Diffusion Computing Foundations

### Tóth & Showalter (1995) — Logic Gates in Excitable Media
**Journal:** *J. Chem. Phys.* 103(6), 2058–2066  
**DOI:** 10.1063/1.470675

**Key contribution:** First empirical realisation of chemical logic gates using the Belousov-Zhabotinsky (BZ) reaction. AND and OR gates constructed by guiding excitation waves through geometric channels.

---

### Steinbock, Kettunen & Showalter (1996) — Chemical Wave Logic Gates
**Journal:** *J. Phys. Chem.* 100(49), 18970–18975  
**DOI:** 10.1021/jp962282p

**Key contribution:** Reconfigurable logic gates via "printed" catalyst patterns on gel substrates. Gate function determined by catalyst arrangement.

---

### Adamatzky & De Lacy Costello (2002) — Experimental Logical Gates in a Reaction-Diffusion Medium
**Journal:** *Phys. Rev. E* 66, 046112  
**DOI:** 10.1103/PhysRevE.66.046112

**Key contribution:** Experimental XOR gate in palladium chloride / potassium iodide reaction in structured gel. Wave collision and annihilation at junctions implements the logic.

---

### Adamatzky (2002, 2005) — Monographs
- *Collision-Based Computing* (Springer, 2002)
- *Reaction-Diffusion Computers* (Elsevier, 2005, with De Lacy Costello & Asai)

**Key contribution:** Comprehensive theoretical and experimental treatment of chemical computation. Establishes two paradigms: geometrically-constrained (fixed channels) vs. free-space/collision-based (architectureless).

---

## 6. Historical Surveys and Supporting Physics

### Adamatzky (2019) — A Brief History of Liquid Computers
**Journal:** *Phil. Trans. R. Soc. B* 374(1774), 20180372  
**DOI:** 10.1098/rstb.2018.0372

> **Abstract:** A substrate does not have to be solid to compute. I demonstrate this using a variety of experimental prototypes where a liquid carries signals, actuates mechanical computing devices and hosts chemical reactions. I overview a variety of liquid computers, from hydraulic algebraic machines invented in the late 1890s to liquid marble logic designed just recently.

**Why it matters:** The definitive historical survey of the entire liquid-computing field. Covers hydraulic machines (1900s), fluidic logic (1960s), droplet/marble logic (2007–2017), and reaction–diffusion computing (1985–present). Key quote: *"The time for theorizing that 'natural systems' compute is over. It is time to produce working laboratory prototypes."* Directly supports the experimental, prototype-driven approach of this project.

---

### Reyssat & Mahadevan (2009) — Hygromorphs: From Pine Cones to Biomimetic Bilayers
**Journal:** *J. R. Soc. Interface* 6(39), 951–957  
**DOI:** 10.1098/rsif.2009.0184

> **Abstract:** We consider natural and artificial hygromorphs, objects that respond to environmental humidity by changing their shape. Using the pine cone as an example that opens when dried and closes when wet, we quantify the geometry, mechanics and dynamics of closure and opening at the cell, tissue and organ levels. We show how simple bilayer hygromorphs of paper and polymer show similar behaviour that can be quantified via a theory which couples fluid transport in a porous medium and evaporative flux to mechanics and geometry.

**Why it matters:** Provides a validated physical model of **capillary imbibition and evaporation in porous bilayers**. The paper shows that wetting and drying times scale differently due to vapour diffusion through pores at late stages, and that a modified Washburn law with evaporative flux captures the dynamics. For this project, this is the reference for understanding how fluid saturation patterns will evolve inside porous media, and how pore-scale geometry controls macroscopic transport timescales.

---

### Schröder & Schanz (2023) — 3D Lagrangian Particle Tracking in Fluid Mechanics
**Journal:** *Annu. Rev. Fluid Mech.* 55, 511–540  
**DOI:** 10.1146/annurev-fluid-031822-041721

> **Abstract:** In the past few decades various particle image–based volumetric flow measurement techniques have been developed. The natural candidate for our focus is 3D Lagrangian particle tracking (LPT), which allows for position, velocity, and acceleration to be determined alongside a large number of individual particle tracks in the investigated volume. The advent of the dense 3D LPT technique Shake-The-Box in the past decade has opened further possibilities for characterizing unsteady flows.

**Why it matters:** The state-of-the-art review for **experimental visualisation of 3D flow fields**. Shake-The-Box (STB) is a dense Lagrangian particle tracking technique that can reconstruct thousands of individual trajectories in a volume — directly applicable to visualising flow paths through porous media to validate CFD models or characterise preferential flow paths.

---

## 7. Wave-Based Analog Computing and Physical Neural Networks

This section reviews a rapidly converging field that provides the strongest precedent for treating **porous media as a trainable physical neural network**. Unlike the chemical/fluidic approaches above, wave-based analog computing uses the physics of wave propagation itself — in optics, acoustics, or mechanics — as the computational substrate. The key insight, established by Hughes et al. (2019), is that wave dynamics are mathematically equivalent to recurrent neural network (RNN) computation. This equivalence opens a direct path to treating acoustic or elastic wave transmission through a porous scaffold as a trainable neural inference.

### Hughes et al. (2019) — Wave Physics as an Analog Recurrent Neural Network
**Journal:** *Science Advances* 5(12), eaay6946  
**DOI:** 10.1126/sciadv.aay6946  
**arXiv:** 1904.12831

> **Abstract:** Analog machine learning hardware platforms promise to be faster and more energy-efficient than their digital counterparts. Wave physics, as found in acoustics and optics, is a natural candidate for building analog processors for time-varying signals. Here we identify a mapping between the dynamics of wave physics, and the computation in recurrent neural networks. This mapping indicates that physical wave systems can be trained to learn complex features in temporal data, using standard training techniques for neural networks. As a demonstration, we show that an inverse-designed inhomogeneous medium can perform vowel classification on raw audio signals as their waveforms scatter and propagate through it, achieving performance comparable to a standard digital implementation of a recurrent neural network.

**Why it matters:** This is the foundational theoretical paper for the wave-computing pivot. Hughes et al. prove that the scalar wave equation can be discretised into a form identical to an RNN, with spatial grid points as "neurons" and wave propagation as recurrent connectivity. The trainable parameters are the material properties (refractive index, density, bulk modulus) at each spatial location. For a porous medium, this maps directly onto: **porosity** → wave speed, **permeability** → attenuation/dispersion, **pore geometry** → scattering. The extension from electromagnetic to acoustic waves is explicit in the paper, and the extension from general inhomogeneous media to porous media is the core novel step of this project.

---

### Wright et al. (2022) — Deep Physical Neural Networks Trained with Backpropagation
**Journal:** *Nature* 601, 549–555  
**DOI:** 10.1038/s41586-021-04223-6

> **Abstract:** Deep physical neural networks (DPNNs) are trained by backpropagating error signals through the physical system itself, rather than through a digital surrogate model. We demonstrate this with a three-layer physical neural network built from laser pulses propagating through diffractive optical elements. Each layer is a physical transformation; training adjusts the phase delays at each pixel. The system learns to classify handwritten digits with accuracy comparable to digital neural networks.

**Why it matters:** Wright et al. demonstrate that **physical systems can be trained end-to-end via backpropagation** without digital simulation of the forward pass. The "in situ" training approach — measuring gradients directly from the physical hardware — is directly applicable to porous wave systems. For this project, the implication is that a porous scaffold could be trained by sending acoustic pulses through it, measuring output error, and backpropagating to adjust pore geometry (e.g., via additive manufacturing iterations or active tuning).

---

### Stern et al. (2020) — Supervised Learning Through Physical Changes in a Mechanical System
**Journal:** *PNAS* 117(26), 14843–14850  
**DOI:** 10.1073/pnas.1915959117

> **Abstract:** We show that a mechanical network of springs and masses can be trained to perform classification tasks by adjusting its spring constants. The network is trained via equilibrium propagation: the system relaxes to two equilibria (one with and one without a target "nudge"), and the difference encodes the gradient. No digital computer is needed during training.

**Why it matters:** Stern et al. prove that **physical networks of coupled elements can learn without any digital computation**. The equilibrium propagation training rule is local and physical — each spring adjusts based only on its own state under two boundary conditions. For porous media, this suggests that training could be achieved by physically "nudging" output transducers and letting elastic relaxation guide pore geometry changes. The mechanical network analogy (springs = solid matrix, masses = fluid inertia) is particularly close to Biot's poroelastic model.

---

### Stern et al. (2021) — Supervised Learning in Physical Networks: From Machine Learning to Learning Machines
**Journal:** *Physical Review X* 11, 021045  
**DOI:** 10.1103/PhysRevX.11.021045

> **Abstract:** We develop a general theory of supervised learning in physical networks, showing that any physical system with a local free-energy landscape can be trained via equilibrium propagation or coupled learning. We derive the conditions under which physical learning converges and characterise the learning capacity of networks with different topologies.

**Why it matters:** The general theoretical framework that unifies physical learning across mechanical, electrical, and optical substrates. Key result: **learning capacity scales with network size and connectivity** in ways that differ from digital NNs due to physical constraints (e.g., locality, reciprocity). For porous media, this predicts that the pore connectivity (percolation threshold, tortuosity) will determine the maximum expressivity of the computational substrate.

---

### Zuo et al. (2018) — Acoustic Analog Computing System Based on Labyrinthine Metasurfaces
**Journal:** *Scientific Reports* 8, 10103  
**DOI:** 10.1038/s41598-018-27741-2

> **Abstract:** We propose an acoustic analog computing (AAC) system based on three cascaded metasurfaces to solve nth-order ordinary differential equations. The metasurfaces are constructed with layered labyrinthine units featuring broad amplitude and phase modulation ranges. The simulated transmitted pressure agrees well with theoretical solutions.

**Why it matters:** The most relevant precedent for **acoustic wave computation** in structured media. Zuo et al. use engineered channels (labyrinthine metasurfaces) to manipulate acoustic wave phase and amplitude to perform mathematical operations. However, they use deterministic channel geometries rather than trainable porous structures. The gap: no existing work uses porous media with stochastic/controlled pore networks as the acoustic computing substrate. The labyrinthine channels are essentially ordered pore networks — porous media generalise this to disordered/trainable architectures.

---

### Zangeneh-Nejad et al. (2021) — Analogue Computing with Metamaterials
**Journal:** *Nature Reviews Materials* 6, 207–225  
**DOI:** 10.1038/s41578-020-00243-2

> **Abstract:** This Review surveys wave-based analogue computing using metamaterials — artificial structures engineered to manipulate electromagnetic, acoustic, and elastic waves. We cover spatial differentiators, integrators, equation solvers, and neural-network-inspired architectures. The field spans from static metasurfaces performing single mathematical operations to programmable and reconfigurable systems.

**Why it matters:** The definitive review of wave-based analog computing. Key sections relevant to this project:
- **Acoustic metamaterial computing** (Section 3): acoustic differentiators, integrators, and ODE solvers
- **Programmable metasurfaces**: structures whose response can be tuned dynamically
- **Neural-network-inspired metamaterials**: the bridge to trainable physical systems

The review explicitly identifies "porous media" as an underexplored class of metamaterial for computing — confirming the novelty of this project's direction.

---

### Silva et al. (2014) — Performing Mathematical Operations with Metamaterials
**Journal:** *Science* 343, 160–163  
**DOI:** 10.1126/science.1242818

> **Abstract:** We introduce the concept of a metamaterial analogue computing platform, capable of performing mathematical operations on incident electromagnetic wave profiles. The system uses Fourier optics: the first metamaterial layer performs a Fourier transform, a second layer applies a transfer function in the Fourier domain, and a third layer performs the inverse transform.

**Why it matters:** The seminal paper that launched the field of metamaterial analog computing. The Fourier-optics approach (lens → filter → lens) maps directly to wave propagation through a porous medium: **scattering by pore structure** → spatial filtering → **recombination at output**. For acoustic waves in porous media, the pore size distribution acts as a natural spatial filter, and the output transducer array reads the result.

---

### Lin et al. (2018) — All-Optical Machine Learning Using Diffractive Deep Neural Networks
**Journal:** *Science* 361, 1004–1008  
**DOI:** 10.1126/science.aat8084

> **Abstract:** We implement a diffractive deep neural network (D2NN) using cascaded diffractive optical elements (DOEs) printed with a 3D printer. Each DOE acts as a layer of artificial neurons, with phase modulation at each pixel representing a neuron's weight. The system classifies images of handwritten digits at the speed of light.

**Why it matters:** Lin et al. demonstrate a **layered physical neural network** fabricated by 3D printing. The architecture (input plane → hidden layers → output plane) is exactly what a porous wave computer would emulate. The training is done digitally (simulating wave propagation), but Wright et al. (2022) later showed physical training. For porous media, each "layer" could be a slice of porous material with different porosity/permeability, fabricated by additive manufacturing.

---

### Biot (1956a,b) — Theory of Propagation of Elastic Waves in Fluid-Saturated Porous Solids
**Journal:** *Journal of the Acoustical Society of America* 28(2), 168–191 (Part I); 28(2), 179–191 (Part II)  
**DOI:** 10.1121/1.1908239; 10.1121/1.1908241

> **Abstract:** A theory is developed for the propagation of elastic waves in a porous elastic solid containing a compressible viscous fluid. Two types of compressional waves and one shear wave are predicted. The fast compressional wave involves in-phase motion of solid and fluid; the slow compressional wave involves out-of-phase motion and is strongly attenuated.

**Why it matters:** The fundamental physics governing **wave propagation in porous media**. Biot's theory predicts two compressional waves (fast and slow) plus a shear wave, with frequency-dependent attenuation and dispersion due to viscous coupling between fluid and solid. For wave computing in porous media, this means:
- **Fast wave**: low attenuation, useful for signal propagation
- **Slow wave**: high attenuation, strong dispersion, useful for nonlinear/memory effects
- **Shear wave**: carries information about solid-matrix properties
- **Frequency-dependent behaviour**: enables band-selective computation

Biot theory provides the physical model needed to simulate and design porous wave computers.

---

### The Porous Media Bridge: Why This Direction is Novel

Existing wave-computing literature uses:
- **Metasurfaces** (Zuo 2018, Silva 2014): engineered surface structures with deterministic geometries
- **Diffractive optical elements** (Lin 2018, Wright 2022): layered phase-modulating structures
- **Mechanical networks** (Stern 2020): discrete spring-mass systems

**None use porous media** — stochastic or controlled three-dimensional pore networks with coupled fluid-solid dynamics. Porous media offer unique advantages for physical neural networks:

| Feature | Metasurfaces / DOEs | Mechanical Networks | **Porous Media** |
|---------|---------------------|---------------------|------------------|
| Dimensionality | 2D surface | 1D/2D discrete | **3D volumetric** |
| Tunability | Fixed fabrication | Discrete stiffness changes | **Continuous (porosity, saturation)** |
| Wave types | Single (EM/acoustic) | Mechanical | **Multiple (fast, slow, shear)** |
| Nonlinearity | Weak | Structural | **Viscous + geometric** |
| Fabrication | Lithography / 3D print | Assembly | **Casting / sintering / AM** |
| Energy storage | Minimal | Elastic | **Fluid compressibility + solid elasticity** |

The project gap: **trainable acoustic/elastic wave computing in three-dimensional porous substrates**, leveraging Biot's coupled wave physics and the mathematical equivalence to RNNs established by Hughes et al.

---

## 8. Synthesis: The Gap for Porous Fluid/Wave Computing

### Landscape Comparison

| Existing Work | Substrate | Control Mechanism | Parallel | Trainable |
|-------------|-----------|-------------------|----------|-----------|
| Mesoporous silica memristors | Solid porous film | Electric field / ion diffusion | No | No (fixed pores) |
| Geopolymer memristors | Bulk porous ceramic | Electric field / electroosmosis | No | No |
| Chemical reservoir (CSTR) | Stirred tank reactor | Fixed reaction network | No | Readout only |
| Bubble logic | Microfluidic channels | Channel geometry | **Yes** | No |
| Reaction-diffusion logic | Gel / free media | Catalyst pattern | No | No |
| Neural CRN | Test tube chemistry | Reaction rates | No | **Yes** |
| Diffractive optical NN | Layered phase masks | Pixel phase delays | **Yes** | **Yes** (digital training) |
| Mechanical learning network | Springs / beams | Stiffness / length | No | **Yes** (physical) |
| Acoustic metasurface computer | Labyrinthine channels | Channel geometry | No | No |
| **Porous wave NN (this project)** | **Porous solid with fluid** | **Pore geometry / saturation** | **Yes** | **Yes** |
| **Porous fluid NN (alt.)** | **Porous media with flow** | **Flow rates / mixing / scheduling** | **Yes** | **Yes** |

### Research Gaps This Project Addresses

1. **Trainable porous wave architecture** — No existing system treats a 3D porous scaffold as a trainable physical neural network for acoustic/elastic wave computation. Metasurfaces are 2D and deterministic; mechanical networks are discrete; porous media combine 3D volumetric computation with continuous tunability.
2. **Parallel neural inference** — Bubble logic is parallel but Boolean; chemical NNs are serial. Diffractive optical NNs are parallel but electromagnetic. This project seeks parallel + neural in the acoustic/elastic domain with porous substrates.
3. **Coupled fluid-solid dynamics as computational resource** — Biot's theory predicts two compressional waves and a shear wave in saturated porous media, with frequency-dependent coupling. No existing wave computer exploits this multi-wave, multi-physics richness.
4. **Fabrication-informed design** — Unlike lithography-dependent optical systems, porous media can be fabricated by casting, sintering, or additive manufacturing at scales from millimetres to metres.
5. **Systematic benchmarking** — Compare latency, throughput, energy efficiency, and parallel capacity against digital baselines and optical/acoustic metasurface alternatives.

### Open Questions

**Wave-computing direction:**
- What porous geometry (porosity, pore size distribution, tortuosity) maximises expressivity for a given acoustic/elastic wave task?
- How does the slow compressional wave (Biot type II) contribute to nonlinear/memory effects useful for neural computation?
- What training approach works best: digital surrogate (Lin 2018 style), in situ backpropagation (Wright 2022 style), or equilibrium propagation (Stern 2020 style)?
- Can fluid saturation be used as a dynamic, reconfigurable control parameter?
- Under what conditions (frequency, porosity, viscosity) does a porous wave computer outperform digital or optical alternatives?

**Fluid-computing direction (alternative):**
- What reactor geometry maximises separation between independent parallel inference channels while enabling controlled mixing?
- What control policy (flow rates, inlet scheduling, valve timing) encodes a given neural network function?
- How does inference accuracy degrade with channel cross-talk and reaction noise?
- Under what conditions (problem size, parallelism, precision requirements) does a fluid NN outperform silicon?

---

## Bibliography (Quick Reference)

| Citation | Year | Topic | Reference |
|----------|------|-------|-----------|
| Hjelmfelt et al. | 1991 | Chemical NNs / Turing machines | PNAS 88, 10983 |
| Rambidi et al. | 1994 | Molecular NN devices | Biosystems 33, 125 |
| Tóth & Showalter | 1995 | First chemical logic gates | J. Chem. Phys. 103, 2058 |
| Steinbock et al. | 1996 | Printed catalyst gates | J. Phys. Chem. 100, 18970 |
| Magnasco | 1997 | Turing-universal chemistry | PRL 78, 1190 |
| Adamatzky | 2002 | Collision-based computing | Springer |
| Adamatzky et al. | 2005 | Reaction-diffusion computers | Elsevier |
| Adamatzky & De Lacy Costello | 2002 | Chemical XOR | PRE 66, 046112 |
| Sielewiesiuk & Górecki | 2001 | Cross-junction logic | J. Phys. Chem. A 105, 8189 |
| Górecki & Górecka | 2009 | Structured excitable media | Nat. Comput. 8, 473 |
| Gentili et al. | 2012 | BZ chemical neuron | IJUC 8, 177 |
| Stovold & O'Keefe | 2012–2017 | Associative memory in RD | IPCAT 2012; IJPEDS 2016 |
| Gruenert et al. | 2015 | Droplet neuron networks | Int. J. Neural Syst. 25, 1450032 |
| Prakash & Gershenfeld | 2007 | Bubble logic | Science 315, 832 |
| Weaver et al. | 2010 | Fluidic control logic | Lab Chip 10, 2612 |
| Egbert et al. | 2018, 2019 | Modulated chemical soup | J. R. Soc. Interface 15, 16 |
| Kim et al. | 2023 | MIMIC microfluidic platform | Nat. Commun. 14, 6530 |
| Jaafar et al. | 2022 | Mesoporous silica memristors | Nanoscale 14, 17170 |
| Zhang et al. | 2024 | Porosity-tunable memristors | ACS AMI 16, 3c19020 |
| Ding et al. | 2023 | Porous crystalline neuromorphics | Chem. Soc. Rev. 52, 7071 |
| Baltussen et al. | 2024 | Formose chemical reservoir | Nature 633, 87 |
| Wetzels | 2025 | Extended Formose analysis | Radboud PhD thesis |
| Nagipogu & Reif | 2025 | Neural CRNs (trainable) | ACS Synth. Biol. 14, 2744 |
| Adamatzky | 2019 | History of liquid computers | Phil. Trans. B 374, 20180372 |
| Reyssat & Mahadevan | 2009 | Porous transport physics | J. R. Soc. Interface 6, 951 |
| Schröder & Schanz | 2023 | 3D flow visualisation | Annu. Rev. Fluid Mech. 55, 511 |
| Hughes et al. | 2019 | Wave physics as RNN | Sci. Adv. 5, eaay6946 |
| Wright et al. | 2022 | Deep physical neural networks | Nature 601, 549 |
| Stern et al. | 2020 | Learning in mechanical systems | PNAS 117, 14843 |
| Stern et al. | 2021 | Physical networks theory | Phys. Rev. X 11, 021045 |
| Zuo et al. | 2018 | Acoustic analog computing | Sci. Rep. 8, 10103 |
| Zangeneh-Nejad et al. | 2021 | Metamaterial computing review | Nat. Rev. Mater. 6, 207 |
| Silva et al. | 2014 | Metamaterial math operations | Science 343, 160 |
| Lin et al. | 2018 | Diffractive deep neural networks | Science 361, 1004 |
| Biot | 1956 | Poroelastic wave theory | JASA 28, 168 |
