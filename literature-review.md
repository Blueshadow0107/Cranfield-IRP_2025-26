# Literature Review: Fluid Neural Networks — Computing over Continuous Reactions

*Compiled: 2026-04-27*  
*Total papers surveyed: 15*

---

## 1. Liquid Neural Networks & Continuous-Time Architectures

These papers form the theoretical backbone of "fluid" computation in the digital realm — neural networks whose state evolves continuously via ODEs rather than discrete layer-wise updates.

---

### Hasani et al. — *Liquid Time-Constant Recurrent Neural Networks as Universal Approximators* (2018/2021)
**arXiv:** [1811.00321](https://arxiv.org/abs/1811.00321)  
**Key contribution:** Introduced Liquid Time-Constant (LTC) neurons, where each neuron's dynamics are governed by a nonlinear ODE with an input-dependent time constant. The synapses are modeled as nonlinear sigmoidal functions inspired by the biophysics of C. elegans and leech nervous systems.

**Why it matters for this project:** LTCs are the closest digital analogue to "fluid" neural computation. The time constant τ is not fixed but varies dynamically based on the input, creating a "liquid" state where information flows at different rates through different parts of the network. Hasani et al. proved universal approximation capability for this class and established boundedness/stability guarantees.

**Core equation:**
```
dx(t)/dt = -(1/τ + f(x,I,t,θ))·x(t) + f(x,I,t,θ)·A
```
where τ_sys = τ / (1 + τ·f(...)) is the input-dependent effective time constant.

---

### Hasani et al. — *Closed-form Continuous-Depth Models* (2021)
**arXiv:** [2106.13898](https://arxiv.org/abs/2106.13898)  
**Key contribution:** Derived a closed-form (solver-free) approximation to LTC dynamics, enabling efficient training without numerical ODE integration. The Closed-form Continuous-time (CfC) network replaces the implicit Euler solver with an analytical update.

**Why it matters:** Eliminates the computational overhead of ODE solvers while preserving the continuous-time expressiveness. This bridges the gap between theoretical continuous models and practical deployment on edge hardware.

---

### Lechner et al. — *Neural Circuit Policies Enabling Auditable Autonomy* (2020)
**Venue:** Nature Machine Intelligence  
**Key contribution:** Applied LTC networks within Neural Circuit Policies (NCPs) — sparse, interpretable recurrent architectures inspired by C. elegans wiring diagrams — to autonomous vehicle steering. Achieved robust lane-keeping with ~19 parameters (orders of magnitude fewer than deep RL alternatives).

**Why it matters:** Demonstrates that continuous-time, biologically inspired architectures can outperform traditional RNNs/LSTMs in real-world control tasks while remaining interpretable. The sparsity and stability properties are directly relevant to physical implementation.

---

### Anonymous — *Accuracy, Memory Efficiency and Generalization: A Comparative Study on Liquid Neural Networks and Recurrent Neural Networks* (2025)
**arXiv:** [2510.07578](https://arxiv.org/abs/2510.07578)  
**Key contribution:** Comprehensive benchmarking of LNNs (LTC, CfC) against RNNs, LSTMs, and GRUs across chaotic time-series prediction, noisy sequence modeling, and high-dimensional forecasting (ICU patient health). Found that LNNs achieve competitive accuracy with 10–20× fewer parameters but train 2–7× slower per epoch.

**Key finding:** LNNs exhibit superior robustness to input noise and long-horizon stability, attributed to their bounded-state guarantees and adaptive time constants.

---

## 2. Physical Reservoir Computing — Chemical & Fluid Substrates

These works move computation out of silicon and into physical media — chemical reactions, fluids, and wave systems.

---

### Baltussen et al. — *Chemical reservoir computation in a self-organizing reaction network* (2024)
**Venue:** Nature  
**DOI:** [10.1038/s41586-024-07567-x](https://doi.org/10.1038/s41586-024-07567-x)  
**Key contribution:** Built a working chemical reservoir computer using the Formose reaction (autocatalytic polymerization of formaldehyde) in a continuous stirred-tank reactor (CSTR). The reaction's self-organizing network of ~106 intermediate compounds serves as the reservoir. Only the linear readout layer is trained.

**Capabilities demonstrated:**
- Nonlinear classification (Boolean gates, XOR, sine functions)
- Prediction of E. coli metabolic network dynamics
- Forecasting of Lorenz attractor chaotic time series

**Why it matters:** This is the most direct experimental realization of "computing over continuous reactions." The Formose reaction is inherently fluid — reagents flow in, products flow out, and the internal state is a high-dimensional concentration field. The paper proves that complex computation emerges from the intrinsic nonlinear dynamics without explicit circuit design.

---

### Parrilla et al. / Cronin Group — *A programmable chemical computer with memory and pattern recognition*
**Venue:** Chem. Sci. / ChemRxiv  
**Key contribution:** Engineered a programmable platform based on the Belousov-Zhabotinsky (BZ) reaction for pattern recognition and memory. The BZ medium stores temporal information about past inputs, enabling sequence recall and associative memory.

**Why it matters:** Demonstrates that chemical systems can do more than reservoir computing — they can exhibit genuine memory and programmable logic. The BZ reaction's oscillatory dynamics create a natural substrate for temporal information processing.

---

### Huck Group (Radboud) — *Thinking in Molecules* (PhD thesis, 2025)
**URL:** [Radboud Repository](https://repository.ubn.ru.nl/bitstream/handle/2066/322901/322901.pdf)  
**Key contribution:** Extended the Formose reservoir work with detailed analysis of temporal memory effects, predictive information, and chemical readouts (colorimetric detection without electronic sensors). Demonstrated that different chemical species in the network act as heterogeneous memory buffers with distinct timescales.

**Key insight:** The continuous inflow/outflow of the CSTR creates a natural "fading memory" property — the reservoir's state depends on a window of past inputs, with different chemical species forgetting at different rates.

---

### Anonymous — *Beyond Silicon: Materials, Mechanisms, and Methods for Physical Neural Computing* (2026)
**arXiv:** [2604.09833](https://arxiv.org/abs/2604.09833)  
**Key contribution:** Comprehensive review of Physical Neural Networks (PNNs). Proposes a taxonomy based on signal encoding (analog vs. digital vs. event-driven), memory architecture (fixed, programmable, adaptive), and execution model. Includes extensive coverage of chemical/reaction-diffusion systems.

**Key insight for this project:** Chemical neural systems occupy a distinct niche — they operate at low bandwidth but with immense internal parallelism (~10^20 molecular interactions simultaneously). Best suited for spatial pattern processing, morphological control, and chemically active environments rather than high-speed inference.

---

## 3. Neural ODEs & Learning in Chemical Kinetics

These papers connect the mathematical framework of Neural ODEs to actual chemical reaction networks.

---

### Anonymous — *Neural CRNs: A Natural Implementation of Learning in Chemical Reaction Networks* (2024)
**arXiv:** [2409.00034](https://arxiv.org/abs/2409.00034)  
**Key contribution:** Introduced Neural CRNs — a framework for implementing Neural ODEs using deterministic mass-action chemical reaction networks. Showed that CRN concentration dynamics can mirror Neural ODE state evolution, with training performed via analog concentration readouts at arbitrary time points.

**Technical advance:** Reduced required reactions from trimolecular (prior art) to bimolecular by exploiting the analog, pre-convergence regime of the CRN. This makes experimental realization far more practical.

**Why it matters:** This is the bridge between the "digital" theory of Neural ODEs and the "wetware" of chemical computing. It provides a design methodology for implementing fluid neural networks in actual test tubes.

---

### Anonymous — *Learning Adaptive Hydrodynamic Models Using Neural ODEs in Complex Conditions* (2024)
**arXiv:** [2410.00490](https://arxiv.org/abs/2410.00490)  
**Key contribution:** Used attention-based Neural ODEs to learn hydrodynamic force models directly from data. The Neural ODE learns the kernel function f representing the fluid dynamics, integrated via ODE solvers (Runge-Kutta).

**Why it matters:** Shows that Neural ODEs are not just theoretical toys — they can capture real fluid dynamics. Directly relevant to the "fluid" aspect of fluid neural networks: the learned ODE is a surrogate for the Navier-Stokes equations.

---

### Anonymous — *Continuous-Time Homeostatic Dynamics for Reentrant Inference Models* (2025)
**arXiv:** [2512.05158](https://arxiv.org/abs/2512.05158)  
**Key contribution:** Proposed the Fast-Weights Homeostatic Reentry Network (FHRN), a continuous-time model where recurrence is not just temporal buffering but self-referential computation. The reentry operator repeatedly encounters its own evolving state, performing iterative refinement.

**Key insight:** Introduces a computational regime where "internal representations are not merely propagated but continuously revisited and reinterpreted." This is a higher-order form of fluid computation where the flow itself becomes the object of computation.

---

## 4. Wave & Fluid-Based Analog Computation

These works explore using physical wave propagation (acoustic, optical, hydrodynamic) as a computational substrate.

---

### Hughes et al. — *Wave physics as an analog recurrent neural network* (2019)
**Venue:** Science Advances  
**Key contribution:** Showed that wave propagation through a disordered medium can implement reservoir computing. The multiple scattering paths create a high-dimensional, nonlinear transformation of the input field.

**Why it matters:** Waves in fluids obey the same mathematical structure as recurrent neural networks. This provides a theoretical justification for treating fluid dynamics itself as a neural network.

---

### Marcucci et al. — *Theory of neuromorphic computing by waves* (2020)
**Venue:** Physical Review Letters  
**Key contribution:** Theoretical framework for computing with rogue waves, dispersive shocks, and solitons in nonlinear media. Derived conditions under which wave systems can implement universal computation.

**Why it matters:** Extends the wave-computing paradigm to strongly nonlinear regimes where fluid-like behaviors (shocks, turbulence) become computational resources rather than nuisances.

---

### Pathak et al. — *Model-Free Prediction of Large Spatiotemporally Chaotic Systems from Data* (2018)
**Venue:** Physical Review Letters  
**arXiv:** [1805.09917](https://arxiv.org/abs/1805.09917)  
**Key contribution:** Applied reservoir computing to predict the evolution of chaotic fluid flows (Kuramoto-Sivashinsky equation) without any knowledge of the underlying PDEs. The reservoir model accurately reproduced the Lyapunov spectrum and long-term statistical properties.

**Why it matters:** Proves that a simple recurrent dynamical system (the reservoir) can learn to emulate complex fluid dynamics. This motivates the reverse question: can a fluid itself serve as the reservoir?

---

## 5. Molecular Communication & Microfluidic Chemical Computing

This section covers works that use chemical reactions in fluid environments for signal processing and computation — directly aligned with Prof. Weisi Guo's research in molecular communication and physics-informed machine learning.

---

### Kim et al. — *Real-time signal processing via chemical reactions for a microfluidic molecular communication system* (2023)
**Venue:** Nature Communications  
**DOI:** [10.1038/s41467-023-42885-0](https://doi.org/10.1038/s41467-023-42885-0)  
**Key contribution:** Designed and built the **MIMIC** (Microfluidic Molecular Communication) platform — a liquid-based microfluidic system that performs chemical concentration signal processing using designed reaction networks. Achieved signal shaping, thresholding, amplification, and detection entirely through chemical reactions in flowing fluid.

**Capabilities demonstrated:**
- **Signal shaping:** Controlled pulse width by tuning reaction timing between signal (Y) and suppressor (P) chemicals
- **Signal thresholding:** Used a thresholder chemical (ThL) to consume all signal below a user-defined concentration, mitigating inter-symbol interference (ISI)
- **Signal amplification:** Used an amplifier chemical (Amp) to produce a constant detectable output regardless of input concentration (above threshold)
- **Text message transmission:** Successfully transmitted "Hi" and other messages via ASCII-encoded chemical bit sequences over metres of tubing

**Why it matters:** This is the most direct experimental precedent for "fluid neural networks." It demonstrates that digital signal processing primitives (amplification, thresholding, shaping) can be implemented as chemical reactions in a continuous-flow fluid medium. The MIMIC platform provides an engineering blueprint for building programmable fluid computers.

---

### Deng et al. / Nallanathan Group — *Digital Signal Processing for Molecular Communication via Chemical Reactions-based Microfluidic Circuits* (2020)
**arXiv:** [2009.05009](https://arxiv.org/abs/2009.05009)  
**Key contribution:** Surveyed and designed microfluidic circuits for molecular communication signal processing using chemical reaction networks. Covered modulation, demodulation, and detection schemes implemented as continuous-flow chemical circuits.

**Why it matters:** Establishes the communication-theoretic framework for treating chemical reactions in fluids as signal processing elements. Directly connects to Prof. Guo's molecular communication research.

---

### Egan et al. — *Stochastic reaction and diffusion systems in molecular communications* (2022)
**Venue:** Digital Signal Processing  
**DOI:** [10.1016/j.dsp.2021.103109](https://doi.org/10.1016/j.dsp.2021.103109)  
**Key contribution:** Comprehensive survey of stochastic models (Langevin diffusion, chemical master equation, reaction-diffusion master equation) for molecular communication. Covers modulation/detection, statistical inference via molecular circuits, and modeling of biochemical environments.

**Key insight:** The review frames molecular circuits — where computation is performed via chemical reactions — as a desirable means of implementing low-complexity inference algorithms in synthetic devices. This positions chemical reactions not just as communication channels, but as computational substrates.

---

### Lotter et al. — *Molecular Noise In Synaptic Communication* (2022)
**arXiv:** [2201.05431](https://arxiv.org/abs/2201.05431)  
**Key contribution:** Statistical model of synaptic molecular communication using the chemical master equation (CME). Characterizes how the stochastic reaction-diffusion process of neurotransmitter binding shapes the postsynaptic membrane potential.

**Why it matters:** Biological synapses are the original "fluid neural network" — computation occurs through stochastic reaction-diffusion of neurotransmitters across a fluid gap. This paper provides the mathematical tools (CME, particle-based simulations) for analyzing computation in noisy chemical environments.

---

## 6. Synthesis & Positioning

### What connects these fields?

All of the above share a common move: **replacing discrete, layer-wise computation with continuous dynamical evolution.** The differences lie in the substrate:

| Substrate | Dynamics | Training | Key Reference |
|-----------|----------|----------|---------------|
| Digital ODE solver | Neural ODE / LTC | End-to-end gradient descent | Chen et al. 2018; Hasani et al. 2021 |
| Chemical reactor | Mass-action kinetics | Readout-only (reservoir) | Baltussen et al. 2024 |
| Chemical test tube | Mass-action kinetics | Embedded in reaction rates | Neural CRNs 2024 |
| Microfluidic channel | Designed reaction networks | Fixed reactions, engineered topology | Kim et al. 2023 (MIMIC) |
| Wave medium | Wave equation / NLSE | Readout-only | Hughes et al. 2019 |
| Chaotic flow | Navier-Stokes | Readout-only | Pathak et al. 2018 |

### Open Questions for This Project

1. **Can we design a fluid neural network that trains its *internal* dynamics, not just a readout layer?** Current chemical reservoirs fix the reaction network and only train weights. Neural CRNs suggest a path to trainable reaction parameters.

2. **What is the right mathematical framework for PDE-based neural networks?** Neural ODEs handle temporal continuity. For spatial-temporal fluids, we need Neural PDEs — architectures where both space and time are continuous.

3. **How do we interface with a fluid computer?** The "transduction tax" (reading chemical states) is a major bottleneck. The MIMIC platform addresses this with UV-Vis spectroscopy integrated directly into microfluidic channels. Lab-on-chip electrical sensors and optical readouts are promising directions.

4. **What problems are fluid computers *uniquely* good at?** Not high-speed inference. But perhaps: real-time CFD surrogates, chemical process control, morphological computing, electromagnetic-denied environments (Prof. Guo's DSTL MEDE project), and biodegradable sensors.

5. **How do fluid neural networks connect to molecular communication?** Prof. Guo's work on molecular communication treats chemical diffusion as a communication channel. Fluid neural networks treat the same medium as a computer. The two perspectives — communication vs. computation — are dual. A fluid that can compute can also communicate; a molecular channel that carries information can also process it.

---

## Bibliography (Quick Reference)

| Citation | Year | Topic | URL |
|----------|------|-------|-----|
| Hasani et al., LTC RNNs | 2018/21 | Liquid time-constant neurons | arXiv:1811.00321 |
| Hasani et al., CfC | 2021 | Closed-form continuous models | arXiv:2106.13898 |
| Lechner et al., NCPs | 2020 | Sparse LTC for autonomy | Nature MI |
| LNN Comparative Study | 2025 | Benchmarking LNNs vs RNNs | arXiv:2510.07578 |
| Baltussen et al. | 2024 | Formose chemical reservoir | Nature, DOI:10.1038/s41586-024-07567-x |
| Cronin Group, BZ computer | ~2021 | Programmable chemical computing | Chem. Sci. |
| Huck Group thesis | 2025 | Formose reservoir extended | Radboud Repository |
| Beyond Silicon review | 2026 | PNN taxonomy & review | arXiv:2604.09833 |
| Neural CRNs | 2024 | CRN implementation of Neural ODEs | arXiv:2409.00034 |
| Neural ODEs + Hydrodynamics | 2024 | Learning fluid forces | arXiv:2410.00490 |
| FHRN | 2025 | Reentrant continuous inference | arXiv:2512.05158 |
| Hughes et al., Wave RNN | 2019 | Wave physics as RNN | Science Advances |
| Marcucci et al. | 2020 | Wave neuromorphic theory | PRL |
| Kim et al., MIMIC | 2023 | Microfluidic chemical signal processing | Nature Comm. |
| Deng et al. | 2020 | Chemical reaction DSP for mol-com | arXiv:2009.05009 |
| Egan et al. | 2022 | Stochastic RD systems in mol-com | Digital Signal Processing |
| Lotter et al. | 2022 | Synaptic molecular noise | arXiv:2201.05431 |
| Pathak et al. | 2018 | Reservoir computing for chaos | PRL, arXiv:1805.09917 |
