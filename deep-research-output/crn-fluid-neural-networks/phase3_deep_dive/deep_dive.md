# Phase 3 Deep Dive: Trainable Chemical Reaction Networks and Reaction-Diffusion Computing

**Scope.** This document contains focused notes on 12 papers that are most relevant to the thesis pivot toward *continuous analog computation in chemical reaction networks (CRNs) and reaction-diffusion (RD) media*. The selection spans (i) trainable well-mixed CRNs, (ii) DNA/enzymatic/metabolic neural implementations, (iii) reservoir and physical-learning frameworks, (iv) spatial RD computing, and (v) the acoustic-wave analog that was the original thesis direction and now serves as a methodological contrast.

**Selection criteria.** Each paper was chosen because it either (a) gives a rigorous theoretical foundation for universal computation/approximation with chemistry, (b) reports a recent experimental demonstration of trainable chemical computing, (c) provides a directly transferable numerical modeling approach, or (d) defines a benchmark substrate that the thesis can compare against.

---

## 1. Dack, Qureshi, Ouldridge & Plesa — Recurrent Neural Chemical Reaction Networks (RNCRNs)

**Reference.** Dack, A., Qureshi, B., Ouldridge, T. E. & Plesa, T. *Recurrent neural chemical reaction networks that approximate arbitrary dynamics.* arXiv:2406.03456 (2024/2025).

**Why it matters.** This is the closest theoretical foundation for the new thesis direction. It proves that a purely chemical network can approximate arbitrary ODE dynamics, including multistability, oscillations and chaos, and gives an explicit training algorithm.

### Core architecture

The RNCRN separates species into two classes:

- **Executive species** `X_1 ... X_N` — these are the variables whose dynamics we want to program.
- **Chemical perceptrons** `Y_1 ... Y_M` — fast auxiliary species that shape the executive dynamics.

The mass-action ODEs are:

```
dx_i/dt = beta_i + x_i * sum_{j=1..M} alpha_{i,j} y_j,       x_i(0) = a_i >= 0
dy_j/dt = (gamma/mu) + (theta_j/mu) y_j
          + (1/mu) sum_{i=1..N} omega_{j,i} x_i y_j
          - (1/mu) y_j^2,                                  y_j(0) = b_j >= 0
```

The parameter `mu > 0` controls how much faster the perceptrons equilibrate than the executive species. In the quasi-static limit `mu -> 0`, each perceptron reaches its steady-state concentration:

```
y_j* = sigma_gamma( sum_{i=1..N} omega_{j,i} x_i + theta_j )
```

where the *chemical activation function* is:

```
sigma_gamma(z) = (1/2) * [ z + sqrt(z^2 + 4 gamma) ]
```

As `gamma -> 0`, `sigma_gamma(z)` approaches the ReLU `max(0, z)`.

Substituting `y_j*` back gives the reduced executive dynamics:

```
dx_i/dt = beta_i + x_i * sum_{j=1..M} alpha_{i,j}
          sigma_gamma( sum_{k=1..N} omega_{j,k} x_k + theta_j )
```

### Universal-approximation result

The authors prove a two-step approximation theorem:

1. **Quasi-static approximation.** Use classical neural-network universal approximation to choose `alpha`, `omega`, `theta` so that the reduced vector field is close to a target vector field `f(x)` over a compact set.
2. **Dynamical approximation.** With the same coefficients, make `mu` small enough that the full system tracks the target ODE arbitrarily closely on a finite time horizon.

The theorem requires `M` large enough and `mu` small enough, but the examples use only 3-4 perceptrons and `mu = 0.01`.

### Training algorithm (Algorithm 1)

1. Fix target region `K_i` for each executive species.
2. Train a small feed-forward network (via backpropagation) to approximate `(f_i(x) - beta_i) / x_i` by `sum_j alpha_{i,j} sigma_gamma(...)`.
3. Insert the trained weights into the full RNCRN and simulate.
4. If the trajectory mismatch is too large, reduce `mu` and repeat.

### Demonstrated behaviors

| Target | Perceptrons | Result |
|--------|-------------|--------|
| `dx/dt = sin(x)` bistability | M = 3 | two stable fixed points reproduced |
| `dx/dt = sin(x)` tristability | M = 4 | three stable fixed points reproduced |
| Limit cycle | M = 4 | stable oscillation |
| Lorenz chaos | M = 8 | chaotic attractor reproduced |

### DNA-strand-displacement implementation

The abstract quadratic reactions are compiled into DNA strand-displacement cascades. For example, the catalytic production reaction `X + Y -> X + 2Y` is implemented with auxiliary fuel complexes `L, H, T`:

```
X + L  <=>  H + B    (fast reversible)
Y + H  ->  O
O + T  ->  X + 2Y
```

The fuel species `L, T` must be held constant (chemostat) because they are consumed.

### Limitations / thesis-relevant caveats

- The theorem is for deterministic mass-action ODEs; stochastic effects at low copy number are not guaranteed to preserve the approximation.
- Negative rate coefficients in the abstract CRN require dual-rail or fuel-mediated implementations.
- Fuel replenishment is assumed; a closed batch system would degrade over time.
- The examples are low-dimensional; scaling to tens of species is not demonstrated experimentally.

### Connection to thesis

RNCRNs provide the formal justification that a well-mixed CRN can act as a trainable recurrent neural network. The thesis can use the same architecture as a simulation target: choose a small target ODE, train the weights in silico, and verify that the CRN ODE reproduces it. The chemical perceptron ODE is a natural starting point for numerical experiments.

---

## 2. Baltussen, de Jong, Duez, Robinson & Huck — Formose Reservoir Computer

**Reference.** Baltussen, M. G. et al. *Chemical reservoir computation in a self-organizing reaction network.* Nature 631, 549-555 (2024).

**Why it matters.** This is the strongest recent *experimental* anchor: a real chemical reaction network (the prebiotic formose reaction) performs multiple nonlinear tasks in parallel without any molecule-by-molecule engineering of the network.

### Experimental setup

- **Reservoir:** formose reaction in a continuous stirred-tank reactor (CSTR).
- **Inputs:** flow rates of formaldehyde, dihydroxyacetone (DHA), NaOH and CaCl2.
- **Readout:** ion-mobility mass spectrometer extracts up to 106 ion signals at 500 ms resolution.
- **Training:** only a single linear readout layer is trained on the reservoir state.

### Tasks demonstrated

1. **Nonlinear classification.** 132 input points in a 2D formaldehyde/NaOH space. The reservoir + linear SVC emulated Boolean gates (AND, OR, XOR), sine, triangle, circle, concentric circles and checkerboard decision boundaries.
2. **Dynamical-system emulation.** The reservoir was trained to reproduce the dynamics of another complex chemical system (the copolymerization model) from the same input drive.
3. **Time-series forecasting.** The Lorenz `x(t)` chaotic time series was forecast one step ahead using the reservoir's past ion signals.

### Key quantitative points

- The readout is a linear support-vector classifier (LSVC) trained on the 106-dimensional steady-state ion vector.
- Cross-validation: 520 leave-five-out splits; Matthews correlation coefficient (`Phi` accuracy) is reported.
- The formose reaction is *deterministic* in this operating window and the linear readout avoids overfitting.

### Conceptual significance

The paper argues for two shifts:

- Move beyond digital molecular computation.
- Move beyond bottom-up molecule-by-molecule design; exploit the emergent dynamics of a self-organizing network.

### Limitations / caveats

- The training is *offline*: the readout weights are computed in silico; the chemistry itself does not learn.
- Each task requires a separate set of readout weights, but the same raw reservoir data can be reused.
- Timescales are slow (minutes to tens of minutes per steady-state point).
- The formose reaction is chemically specific and hard to generalize; it is not a programmable substrate in the same sense as DNA strand displacement.

### Connection to thesis

Reservoir computing is a lower-risk experimental paradigm than full in-chemico training. The thesis can position its own CRN training work as a step *beyond* reservoir computing: rather than training only a readout, train the chemical weights themselves. Baltussen sets the experimental benchmark for what chemistry can already do without training.

---

## 3. Poole, Ouldridge, Gopalkrishnan & Winfree — Detailed-Balanced CRNs as Boltzmann Machines

**Reference.** Poole, W., Ouldridge, T. E., Gopalkrishnan, M. & Winfree, E. *Detailed balanced chemical reaction networks as generalized Boltzmann machines.* arXiv:2205.06313 (2022).

**Why it matters.** It connects stochastic CRNs to probabilistic inference and shows how a broad class of chemical equilibria can represent complex distributions.

### Core idea

A **detailed-balanced CRN (dbCRN)** has reversible reactions with forward/backward rates satisfying:

```
k_+ / k_- = exp(-DeltaG)
```

where `DeltaG` is the Gibbs free-energy change. The stationary distribution is an equilibrium distribution:

```
pi(s) = (1/Z) exp(-G(s))
G(s) = sum_i G_i s_i + log(s_i!)
```

If the reachability class `Gamma_s0` is the entire non-negative integer lattice, this factorizes into independent Poisson distributions. **Complex distributions arise only when conservation laws restrict the reachability class.**

### Key result: reachability entanglement

Hidden species are useful only if their reachability class is *entangled* with the visible species. If the hidden and visible states are independent, marginalizing over hidden species does not change the visible distribution.

The authors construct dbCRNs that produce non-Poisson distributions (e.g., a smiley/frowny face over pixel species) by enforcing tight conservation laws:

```
c_s + c_h = 1
sum_{x,y,alpha} p^alpha_{x,y} = 1
sum_{x,y} p^alpha_{x,y} = c_alpha
s_1 = sum_{x,y,alpha} x p^alpha_{x,y}
s_2 = sum_{x,y,alpha} y p^alpha_{x,y}
```

### Clamping and inference

Clamping a subset of species means holding their concentrations (or means) fixed. The conditional equilibrium distribution of the remaining species then performs probabilistic inference. The paper derives thermodynamic costs for such inference.

### Limitations

- Detailed balance is a strong constraint; living cells and many engineered CRNs operate out of equilibrium.
- The constructions are proof-of-principle; no experimental implementation of a trained dbCRN Boltzmann machine is shown.
- Learning the energies `G_i` from data is discussed only in companion work.

### Connection to thesis

This paper gives the stochastic-chemistry perspective: a CRN is not just a deterministic ODE but a sampler. For the thesis, it motivates why *stochastic* simulations (Gillespie) may show richer behavior than deterministic ODEs, and it provides the formal link between CRN equilibrium and probabilistic computation.

---

## 4. Okumura, Gines, Lobato-Dauzier et al. — Enzymatic Neural Networks

**Reference.** Okumura, S. et al. *Nonlinear decision-making with enzymatic neural networks.* Nature 610, 496-501 (2022).

**Why it matters.** First experimental demonstration of a *multilayer* molecular neural network with a hidden layer, using DNA-encoded enzymatic neurons. It solves nonlinearly separable classification problems in cell-sized droplets.

### Chemical neuron

The neuron is built from three enzymes:

- **Polymerase** produces output strand `alpha`.
- **Nickase** cuts templates to activate production.
- **Exonuclease** degrades single-stranded DNA, providing a threshold/bias.

The neuron computes a step function:

```
alpha_ON  if  sum_i w_i X_i > threshold
alpha_OFF otherwise
```

where `w_i` are programmed by the concentrations of converter templates and fake templates, and the threshold is set by the drain-template concentration.

### Achievements

1. **Linear classifier** in ~100 pL droplets with a Hill coefficient of ~16 (very sharp boundary).
2. **Majority voting on 10 bits** with 97% accuracy.
3. **Two-layer network** synthesizing a family of rectangular functions on a microRNA input (let7a).
4. **Hybrid neural/logical circuit** implementing a decision tree that partitions a 2D concentration plane into three nonlinearly separable regions.

### Sensitivity analysis

- A 3% pipetting error on drain concentration translates to ~10% bias error.
- A 0.1 degC temperature error translates to ~2% bias error.

### Limitations

- The activation is a sharp step, not a smooth sigmoid; training is not gradient-based — weights are set by template stoichiometry.
- The network is hand-designed for each task; no in-chemico learning rule is demonstrated.
- Droplet microfluidics is required for parallel screening.

### Connection to thesis

Okumura shows that *layered* chemical computation is experimentally feasible. The thesis can contrast its own smooth-activation, gradient-trainable CRN approach with this enzymatic, threshold-based approach. The droplet format also points to a possible experimental platform for testing trained CRNs.

---

## 5. Qian, Winfree & Bruck — DNA Strand-Displacement Neural Network

**Reference.** Qian, L., Winfree, E. & Bruck, J. *Neural network computation with DNA strand displacement cascades.* Nature 475, 368-372 (2011).

**Why it matters.** Landmark paper: the first autonomous molecular system that implements a small neural network and a Hopfield associative memory using DNA strand displacement.

### Mechanism

The network uses a "seesaw" gate motif:

- Each gate is a double-stranded DNA complex with an exposed toehold.
- An input strand binds the toehold and displaces an output strand.
- Threshold complexes absorb a small number of input molecules, implementing a sharp threshold.
- The released output strands act as inputs to downstream gates.

A neuron computes a linear threshold function:

```
output = 1  if  sum_i w_i x_i >= theta
output = 0  otherwise
```

where `w_i` and `theta` are encoded by the concentrations of gate and threshold complexes.

### Demonstrations

- A four-neuron Hopfield associative memory trained in silico to remember four single-stranded DNA patterns.
- When presented with an incomplete pattern, the network converges to the closest stored memory.

### Limitations

- Digital/binary outputs rather than analog concentrations.
- Leak reactions limit scalability and signal restoration.
- Training is performed on a computer; the DNA circuit is a compiled fixed network.

### Connection to thesis

This is the historical starting point for DNA neural networks. The thesis can cite it to show how the field moved from compiled threshold circuits to analog, trainable CRNs (Dack, Kang, Poole) and to enzymatic networks (Okumura).

---

## 6. Kang & Kim — Noise-Robust CRN Training with Smooth Activations

**Reference.** Kang, S.-H. & Kim, J. *Noise-robust chemical reaction networks training artificial neural networks.* arXiv:2410.12548 (2024).

**Why it matters.** It addresses a practical concern for chemical neural networks: noise robustness. It proposes a single-pot CRN that performs both forward inference *and* backpropagation training using a smooth activation function.

### Smooth ReLU

The activation is:

```
sigma(z) = ( z + sqrt(z^2 + 4h) ) / 2
```

with derivative:

```
sigma'(z) = sigma(z)^2 / (sigma(z)^2 + h)
```

The parameter `h > 0` controls smoothing around `z = 0`. When `h = 0`, this is the standard ReLU.

### Why smoothness matters

- ReLU and leaky ReLU have discontinuous derivatives at zero.
- Small perturbations in rate constants or concentrations can flip the sign of `z`, causing large errors during backpropagation.
- The smooth ReLU derivative is continuous and bounded, making the CRN training network more robust.

### Three CRN sub-networks

1. **Forward CRN** implements the feed-forward network.
2. **Backward CRN** computes gradients, including `sigma'` via the reaction set:
   ```
   2X -> 2X + D
   D + A -> A
   A -> empty
   2X -> 2X + A
   H -> H + A
   ```
   with rate constraints `kappa5 kappa2 / kappa3 = kappa4 kappa2 / kappa3 = kappa1`, so that `[D] -> X^2 / (X^2 + h)`.
3. **Update CRN** performs `w <- w - eta * grad(L)`.

The full system is one-pot but time-scale separated: forward/backward reactions are fast, update reactions are slow.

### Tasks

- XOR
- Iris classification
- MNIST
- Nonlinear regression

The authors report that the smooth-activation CRN maintains accuracy under noise in rate constants and species concentrations, and is less sensitive to running time than leaky-ReLU CRNs.

### Limitations

- The analysis is deterministic ODE-based; stochastic training is not proved.
- Dual-rail encoding for negative weights doubles the species count.
- No experimental validation is shown.

### Connection to thesis

This paper gives the thesis a concrete design for a trainable CRN and a robustness argument for using smooth activations. The smooth-ReLU derivative reaction can be reused in the thesis's own simulations.

---

## 7. Pandi, Koch, Voyvodic et al. — Metabolic Perceptrons

**Reference.** Pandi, A. et al. *Metabolic perceptrons for neural computing in biological systems.* Nature Communications 10, 3880 (2019).

**Why it matters.** Shows that analog neural computation can be performed by *metabolic* enzymes in living cells and cell-free extracts, expanding the substrate beyond DNA.

### Architecture

- **Transducer layer:** enzymes convert input metabolites into a common intermediate (benzoate).
- **Actuator layer:** benzoate activates a transcription factor (BenR) driving GFP expression.
- **Weighted adder:** multiple transducers feed the same actuator; enzyme expression levels set the weights.

The perceptron computes:

```
GFP_output = threshold( sum_i w_i [input_i] + bias )
```

where `w_i` are determined by the relative expression/activity of the transducer enzymes.

### Demonstrations

- Whole-cell analog adder combining hippurate and benzaldehyde inputs.
- Cell-free system with multiple weighted transducers.
- Two four-input metabolic perceptrons trained (in silico) for binary classification of metabolite combinations.

### Modeling

The authors use Hill-function models plus resource-competition terms to fit the cell-free circuits, then use the fitted model to predict perceptron weights.

### Limitations

- Training is model-based and done in silico; the cells do not learn autonomously.
- The output is gene-expression based, so response times are slow (~hours).
- Resource competition among plasmids reduces dynamic range.

### Connection to thesis

Metabolic perceptrons illustrate that enzyme kinetics already implement weighted summation. The thesis can treat them as a biological existence proof for analog chemical neural computation, while focusing on simpler in-vitro CRNs for tractable modeling.

---

## 8. Kamsma, Kim, Kim, Boon, Spitoni, Park & van Roij — Fluidic Iontronic Nanochannels

**Reference.** Kamsma, T. M. et al. *Brain-inspired reservoir computing with fluidic iontronic nanochannels.* arXiv:2309.11438 (2024).

**Why it matters.** Represents a non-molecular but *fluid-based* neuromorphic substrate. Useful as a comparison point for the thesis and as a bridge from chemical computing to the original fluid/acoustic-wave interest.

### Device

- Tapered microfluidic channel (length `L = 150 um`, width 200 um -> 10 um) filled with a close-packed colloidal crystal of charged silica spheres.
- Pore network between colloids forms nanochannels.
- 10 mM KCl electrolyte.

### Physics

The device is described by Poisson-Nernst-Planck (PNP) transport with slab-averaging. Salt concentration polarization at the tip gives:

- **Ion-current rectification (ICR)** — diode-like I-V curve.
- **Volatile memristive behavior** — conductance depends on previous voltage pulses.
- **Short-term plasticity** — facilitation and depression depending on pulse spacing relative to a memory time `tau`.

The memory retention time scales as `tau ~ L^2`, i.e. quadratically with channel length, allowing design for a specific timescale.

### Reservoir-computing task

- Handwritten digits encoded as time-series voltage pulses.
- A single channel processes the time series; the resulting current response is classified by a simple linear readout.
- Performance comparable to solid-state memristor reservoirs.

### Connection to thesis

Iontronics show that *fluidic* media can do neuromorphic computing, but they rely on electron/ion transport rather than chemical reaction networks. The thesis can compare the energy/space/time scaling of iontronic reservoirs with CRN-based computation.

---

## 9. Stern & Murugan — Learning Without Neurons in Physical Systems

**Reference.** Stern, M. & Murugan, A. *Learning without neurons in physical systems.* Annual Review of Condensed Matter Physics 14, 417-441 (2023).

**Why it matters.** Provides the broader theoretical framing: physical systems can learn via *local learning rules* without mimicking a digital computer.

### Key concepts

A physical learning machine has:

1. **Physical degrees of freedom** `s` that respond to stimuli `f`.
2. **Learning degrees of freedom** `w` that modify how `s` responds.
3. **Local learning rule:** `dw[x,t]/dt = h(s[x,t])` — the change at point `x` depends only on the local state.

### Contrast with gradient descent

Conventional ML uses non-local gradient descent:

```
dw_i/dt ~ -dC/dw_i
```

where `C` is a global cost. Physical learning instead exploits local dynamics; the global cost is minimized implicitly because the physical state `s` encodes information from the whole system through collective dynamics.

### Examples reviewed

- Hebbian growth in elastic networks.
- Contrastive learning in flow networks.
- Spike-timing-dependent plasticity in memristor networks.
- Molecular self-assembly with trainable interactions.

### Categories

- **Physical unsupervised learning:** system adapts to stimuli without an external error signal.
- **Physical supervised learning:** a supervisor provides an error signal, but the parameter update is still local.

### Connection to thesis

The thesis can frame its CRN training work as an instance of physical supervised learning: the desired output is provided externally, but the chemical weight updates could in principle be implemented by local reaction rules (as in Kang & Kim). Stern & Murugan justify why this is scientifically interesting even if it does not beat digital ML on raw performance.

---

## 10. Hughes, Williamson, Minkov & Fan — Wave Physics as an Analog RNN

**Reference.** Hughes, T. W. et al. *Wave physics as an analog recurrent neural network.* Science Advances 5, eaay6946 (2019).

**Why it matters.** This is the canonical reference for the *original* thesis direction (acoustic-wave analog computing). It is included here to make the contrast with chemical computing explicit.

### Mapping

A standard RNN is:

```
h_t = sigma_h( W_h h_{t-1} + W_x x_t )
y_t = sigma_y( W_y h_t )
```

A discretized scalar wave equation is:

```
(u_{t+1} - 2 u_t + u_{t-1}) / Delta_t^2 - c(x)^2 nabla^2 u_t = f_t
```

Defining the hidden state as `h_t = [u_t, u_{t-1}]^T`, the wave update can be written as:

```
h_t = A(h_{t-1}) h_{t-1} + P^(i) x_t
y_t = (P^(o) h_t)^2
```

The trainable parameter is the spatial distribution `c(x)` (material layout). Nonlinearity enters through intensity-dependent wave speed `c = c_lin + u^2 c_nl` or through quadratic intensity detection.

### Demonstration

- 2D vowel-classification domain.
- Input vowel waveform injected at one point.
- Three probes on the right side measure time-integrated power.
- Material distribution optimized by backpropagation through the wave solver (adjoint method).
- Achieved ~92.6% train / ~86.3% test accuracy on three vowel classes, comparable to a digital RNN but with fewer parameters.

### Limitations

- Requires fabrication of an inhomogeneous material.
- Training is done in simulation; the learned structure is then manufactured.
- The thesis found this route experimentally difficult (long fabrication times, limited lab access).

### Connection to thesis

The wave-RNN paper motivates the *analog continuous computation* goal but also shows its experimental challenges. The thesis pivot to CRNs keeps the same conceptual framework — physical dynamics as a trainable network — but switches to a substrate (chemistry) that is cheaper, faster to prototype in simulation, and closer to the "fluid neural network" metaphor.

---

## 11. Adamatzky & De Lacy Costello — Reaction-Diffusion XOR Gate

**Reference.** Adamatzky, A. & De Lacy Costello, B. P. J. *Experimental logical gates in a reaction-diffusion medium: the XOR gate and beyond.* Physical Review E 66, 046112 (2002).

**Why it matters.** Classic experimental demonstration that spatial chemical dynamics can perform Boolean logic via wave-front interactions.

### Setup

- Gel containing palladium chloride (yellow) is cut into T-shaped chambers.
- Inputs are drops of potassium iodide solution at the two horizontal chambers.
- A reaction front produces a dark-brown precipitate (PdI2) as KI diffuses inward.

### Logic encoding

- **TRUE** = colored (precipitate present).
- **FALSE** = uncolored.

### XOR behavior

- If only one input receives KI, the front reaches the output chamber and colors it (output = TRUE).
- If both inputs receive KI, the two fronts collide at the junction and leave an uncolored bisector extending to the output (output = FALSE).

The authors also discuss an AND gate using an unstable copper chloride / potassium hydroxide system, where two inputs together stabilize precipitate at the output.

### Limitations

- The processor is disposable; the precipitate cannot be reset.
- Computation speed is diffusion-limited (minutes).
- Scaling to many gates requires careful geometric design and suffers from reactant depletion.

### Connection to thesis

This paper establishes the earliest spatial-chemical computing paradigm. The thesis can use it to contrast *hard-wired geometry* (RD gates) with *well-mixed trainable CRNs* (Dack/Kang) and *reservoir* approaches (Baltussen).

---

## 12. Gorecki, Gizynski, Guzowski et al. — Chemical Computing with Reaction-Diffusion Processes

**Reference.** Gorecki, J. et al. *Chemical computing with reaction-diffusion processes.* Philosophical Transactions of the Royal Society A 373, 20140219 (2015).

**Why it matters.** Comprehensive review of RD computing, especially using excitable Belousov-Zhabotinsky (BZ) droplets and structured media.

### Themes

- **Excitable media:** BZ reaction produces propagating excitation waves. Information is encoded by wave presence/absence or pulse timing.
- **Structured media:** channels, junctions and compartments impose logic by controlling wave collisions.
- **Droplet networks:** aqueous BZ droplets in oil communicate through chemical diffusion across interfaces, enabling flexible network topology.

### Operations demonstrated in the RD literature

- Logic gates (AND, OR, XOR) via wave collisions.
- Memory units using feedback loops.
- Counters and coincidence detectors.
- Path planning by wave expansion.
- Image processing using light-sensitive BZ media.

### Key insight

RD computing naturally performs *parallel* computation: a single wave front explores space simultaneously. The output is a spatial or temporal pattern rather than a single scalar.

### Limitations

- Most RD computers are designed for specific tasks; general programmability remains limited.
- Excitable media require continuous energy supply (open systems).
- Wave collisions can be sensitive to initial conditions and geometry.

### Connection to thesis

Gorecki et al. provide the spatial-RD context. The thesis can position its planned 1D/2D RD simulations as a bridge between well-mixed CRNs and spatial wave-based computation. The BZ droplet paradigm is also an experimental platform that could eventually host trained CRNs.

---

## Cross-cutting themes

### Well-mixed vs. spatial computation

| Paradigm | Examples | Training | Strength | Weakness |
|----------|----------|----------|----------|----------|
| Well-mixed CRN | Dack, Kang, Poole | In-chemico or in-silico | Easy to simulate; universal approximation | Hard to read out; noise at low copy number |
| Reservoir chemistry | Baltussen | Readout only | Experimental; minimal design | Chemistry does not learn; task-specific readout |
| Spatial RD | Adamatzky, Gorecki | Hand-designed geometry | Natural parallelism; visual output | Difficult to scale; geometry-specific |
| Iontronic | Kamsma | Readout only | Fast electrical I/O; memristive | Requires microfabrication; not chemically programmable |

### Trainability vs. programmability

- **Compiled networks** (Qian 2011, Okumura 2022): weights set by molecular stoichiometry; no in-chemico learning.
- **Trainable deterministic CRNs** (Dack 2024, Kang 2024): weights are rate constants or initial concentrations that can be optimized by gradient descent.
- **Stochastic/trainable CRNs** (Poole 2022): energies of detailed-balanced reactions can in principle be learned from data.
- **Physical learning** (Stern 2023): local update rules may eventually allow the material itself to learn.

### Noise and robustness

- Kang & Kim identify smooth activation functions as a key to noise-robust chemical backpropagation.
- Okumura shows that enzymatic neurons can tolerate ~10-20% concentration variation.
- Baltussen shows that a complex self-organizing reservoir is reproducible enough for linear readout.

### Timescales

- DNA strand displacement: minutes to hours.
- Enzymatic neurons: minutes to tens of minutes.
- Metabolic perceptrons: hours (gene expression).
- Formose reservoir: minutes to tens of minutes per steady state.
- Iontronic channels: milliseconds to seconds.
- Acoustic waves: microseconds to milliseconds.

This timescale hierarchy matters for the thesis: chemical computation is slow but massively parallel and cheap; acoustic computation is fast but fabrication-heavy.

---

## Implications for the thesis

1. **Theoretical justification.** Dack et al. proves that well-mixed CRNs are universal approximators of ODE dynamics. This is the strongest argument for the new thesis direction.
2. **Simulation target.** The thesis can implement the RNCRN equations numerically and train them to reproduce simple dynamical tasks (XOR, oscillation, bistability), providing original results without an immediate wet-lab requirement.
3. **Smooth activations.** Kang & Kim's smooth ReLU gives a concrete, robust building block for trainable CRNs.
4. **Spatial extension.** Adamatzky and Gorecki show how to move from well-mixed CRNs to RD media, which is the natural next step after the first simulation milestone.
5. **Comparison benchmark.** Baltussen's formose reservoir and Kamsma's iontronic channel provide experimental benchmarks against which the thesis's trained-CRN approach can be compared conceptually.
6. **Risk mitigation.** By keeping the wave-RNN paper (Hughes et al.) in the literature review, the thesis can explain the pivot and retain the original motivation for analog physical computation.

---

## Open gaps identified

- **Stochastic trainability.** Universal approximation is proved for deterministic ODEs. Does the result hold for stochastic CRNs at finite copy number? Poole's dbCRN work is a first step for equilibrium distributions, but nonequilibrium stochastic trainability is open.
- **In-chemico learning.** No published experiment has shown a CRN that updates its own rate constants in response to error signals. Kang & Kim is purely in silico.
- **Readout.** How do you read the state of a well-mixed CRN continuously without disturbing it? Mass spectrometry (Baltussen) and fluorescence (Okumura) are partial answers but add complexity.
- **Scalability.** Demonstrations are at <10 species or <10 neurons. Scaling to useful networks (tens to hundreds of species) is unproven.
- **Energy and fuel.** CRNs that consume fuel species require replenishment. A closed-batch "battery-powered" chemical computer has not been demonstrated.
- **Programmable RD.** Moving from hand-designed RD gates to trainable RD patterns (e.g., optimizing initial conditions or diffusion coefficients) is an underexplored middle ground that the thesis could target.
