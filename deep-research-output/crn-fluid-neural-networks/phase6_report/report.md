# Research Report: Crn Fluid Neural Networks

*Generated from 63 papers*

## Paper Statistics

**Total papers**: 63

**Papers by year**:
- 2026: 2
- 2025: 13
- 2024: 11
- 2023: 4
- 2022: 6
- 2021: 1
- 2020: 1
- 2019: 4
- 2018: 2
- 2017: 1
- 2016: 1
- 2015: 3
- 2013: 2
- 2012: 2
- 2011: 2
- 2010: 1
- 2009: 1
- 2005: 1
- 2003: 1
- 2002: 1
- 1996: 1
- 1977: 1
- 1943: 1

**Top venues**:
- arXiv: 9
- Nature: 9
- Science: 3
- Nature Chemistry: 2
- Journal of the Royal Society Interface: 2
- Nature Communications: 2
- DNA Computing and Molecular Programming: 2
- Artificial Life Conference Proceedings: 2
- Radboud University MSc thesis: 2
- Cell: 1

**Most cited papers**:
- [11] Neural network computation with DNA strand displacement cascades (1323 citations)
- [21] Experimental logical gates in a reaction-diffusion medium: the XOR gate and beyo (151 citations)
- [16] Metabolic perceptrons for neural computing in biological systems (97 citations)
- [20] Chemical computing with reaction–diffusion processes (78 citations)
- [2] Chemical reservoir computation in a self-organizing reaction network (76 citations)
- [28] Parallel molecular computation on digital data stored in DNA (47 citations)
- [3] Supervised learning in DNA neural networks (27 citations)
- [1] Recurrent neural chemical reaction networks that approximate arbitrary dynamics (9 citations)

---

# Phase 1: Frontier — Latest Directions in Chemical and Reaction-Diffusion Computing

**Date:** 2026-07-02  
**Topic:** Trainable chemical reaction networks, DNA/enzymatic neural networks, reaction-diffusion computing, and physical neural networks for continuous analog computation.

---

## 1. Current research frontier (2023–2026)

The field has shifted from hand-designed molecular circuits toward **trainable, autonomous chemical systems** that can learn from examples. Four active fronts dominate recent work:

1. **Recurrent neural CRNs** — ODE/CRN models that approximate arbitrary dynamics and can be trained like RNNs.
2. **DNA neural networks** — strand-displacement cascades that perform pattern recognition and supervised learning.
3. **Enzymatic/protein neural networks** — metabolic or enzymatic reaction networks that make nonlinear decisions.
4. **Chemical reservoir computing** — complex self-organizing reaction networks used as reservoirs for ML tasks.
5. **Reaction-diffusion computing** — spatial chemical dynamics for logic, pattern recognition, and PDE-based world models.

---

## 2. Key recent papers

### 2.1 Recurrent neural chemical reaction networks (RNCRNs)

**Dack, te Vrugt, Huck, Löwen (2024)** — *Recurrent neural chemical reaction networks that approximate arbitrary dynamics* (arXiv:2406.03456, updated 2025).

- Proposes a molecular RNN built from coupled CRNs.
- Proves universal approximation of arbitrary dynamics with enough neurons and fast reactions.
- Demonstrates multistability, oscillations, chaos.
- Experimentally implementable with DNA strand displacement.
- **Relevance:** Direct theoretical foundation for trainable CRN Fluid NNs.

### 2.2 Chemical reservoir computing

**Baltussen, de Jong, Duez, Robinson, Huck (2024)** — *Chemical reservoir computation in a self-organizing reaction network* (Nature 631, 549–555).

- Formose reaction network in a CSTR acts as a reservoir.
- Performs Boolean logic gates, XOR, nonlinear classification, and Lorenz attractor forecasting.
- Uses mass-spectrometry readout; up to 106 tracked molecules.
- Proof-of-concept for fully chemical readout via colorimetric reactions.
- **Relevance:** Shows that unengineered complex chemistry can compute; bridges origin-of-life and neuromorphic computing.

### 2.3 DNA neural networks

**Cherry, Qian (2025)** — *Supervised learning in DNA neural networks* (Nature, 2025).

- Extends earlier winner-take-all DNA networks to supervised learning.
- Trains weights by optimizing DNA gate concentrations.
- Demonstrates molecular pattern recognition.
- Builds directly on Qian, Winfree & Bruck (2011) Nature 475, 368–372.
- **Relevance:** Strong experimental anchor for DNA-based trainable classifiers.

**Song & Qian (2025)** — *Heat-rechargeable computation in DNA logic circuits and neural networks* (Nature 646, 315–322).

- Introduces reusable DNA circuits that can be thermally reset.
- Addresses energy supply and reusability in DNA computing.
- **Relevance:** Practical concern for any continuous chemical computer — refresh and recharge.

### 2.4 Enzymatic and protein neural networks

**Okumura et al. (2022)** — *Nonlinear decision-making with enzymatic neural networks* (Nature 610, 496–501).

- Enzymatic reactions implement a winner-take-all neural network.
- Classifies molecular inputs nonlinearly.
- **Relevance:** Demonstrates that enzymes, not just DNA, can form trainable networks.

**Ghosh et al. (2025)** — *A recursive enzymatic competition network capable of multitask molecular information processing* (Nature Chemistry).

- Recursive competition network performs multiple information-processing tasks.
- Uses enzymatic competition rather than strand displacement.
- **Relevance:** Expands the toolkit beyond DNA to enzyme-based computing.

**Parres-Gold et al. (2025)** — *Contextual computation by competitive protein dimerization networks* (Cell 188, 1984–2002).

- Protein dimerization networks perform contextual computation in cells.
- Shows how competition motifs enable nonlinear mixing.
- **Relevance:** Biological inspiration for competitive gating in CRNs.

**Chen et al. (2024)** — *A synthetic protein-level neural network in mammalian cells* (Science 386, 1243–1250).

- Implements a trainable protein neural network inside living cells.
- **Relevance:** Pushes chemical computing into in vivo settings.

### 2.5 Machine learning for CRN discovery

**Kang & Kim (2024)** — *Noise-robust chemical reaction networks training artificial neural networks* (arXiv:2410.11919).

- Focuses on training CRNs to implement ANNs robust to molecular noise.
- **Relevance:** Critical for small-copy-number regimes.

**Poole, Ouldridge, Gopalkrishnan (2025)** — *Autonomous learning of generative models with chemical reaction network ensembles* (J. R. Soc. Interface 22, 20240373).

- CRN ensembles learn generative models autonomously.
- Builds on detailed-balanced CRN Boltzmann machines.
- **Relevance:** Connection between CRNs and probabilistic generative models.

### 2.6 Reaction-diffusion and spatial computing

**Gorecki & Gorecka (2015)** — *Chemical computing with reaction–diffusion processes* (Phil. Trans. R. Soc. A 373, 20140219).

- Review of logic gates, pattern recognition, and information processing in RD media.
- **Relevance:** Foundational review for spatial extension of CRN computing.

**Adamatzky et al. (2002)** — *Experimental logical gates in a reaction-diffusion medium: the XOR gate and beyond* (Phys. Rev. E 66, 046112).

- Laboratory prototype of XOR gate using precipitate wave interactions.
- **Relevance:** Early experimental demonstration that RD media can compute.

**Xiong et al. (2022)** — *Molecular convolutional neural networks with DNA regulatory circuits* (Nature Machine Intelligence 4, 625–635).

- DNA circuits implement convolutional pattern recognition.
- **Relevance:** Spatial pattern recognition with molecular circuits.

### 2.7 PDE-based world models and neural operators

**FluidWorld / FluidLM / FluidVLA (2024–2026)** — GitHub projects replacing attention/recurrence with reaction-diffusion PDEs.

- Latent states evolve via Laplacian diffusion + learned reaction terms.
- Claims O(N) complexity and spatial coherence.
- **Relevance:** Emerging ML paradigm that aligns with reaction-diffusion computing.

---

## 3. Trends and breakthroughs

| Trend | Implication |
|-------|-------------|
| From hand design to learning | CRN rate constants / gate concentrations are now optimized rather than engineered manually. |
| From digital to analog | Recent work emphasizes molecular perceptrons, winner-take-all, and continuous-valued computation. |
| From DNA to enzymes/proteins | The substrate is broadening; enzymes and proteins offer faster kinetics and in vivo compatibility. |
| From static to recurrent | RNCRNs and reservoir computing add temporal/stateful computation. |
| From well-mixed to spatial | DNA convolutional networks and RD logic gates show how space enables pattern-based computation. |
| From electronic to chemical readout | Colorimetric/chemical readouts reduce dependence on external instruments. |

---

## 4. Open questions at the frontier

1. How do we train CRNs robustly under molecular noise and finite copy numbers?
2. Can we design spatial RD systems that learn patterns rather than hand-wiring gates?
3. What is the energy-delay-computation trade-off relative to digital and memristive hardware?
4. How can chemical systems be recharged, reset, or gated between tasks?
5. Can orthogonal chemical species provide true parallel computation in a shared vessel?

---

## 5. Implications for this thesis

- The RNCRN framework and chemical reservoir computing papers give a strong theoretical and experimental basis.
- A trainable well-mixed CRN classifier is a well-defined, achievable contribution.
- Reaction-diffusion and gating are natural extensions that link to active research.
- Energy/noise analysis can draw on recent enzymatic and DNA network papers.

---

## 6. Files generated

- `phase1_frontier/frontier.md` (this file)
- `phase1_frontier/search_results/` (raw search outputs, if any)


---

# Phase 2: Survey — Landscape of Chemical and Reaction-Diffusion Computing

**Date:** 2026-07-02  
**Database:** `paper_db.jsonl` (63 papers)

---

## 1. Scope

This survey covers the literature relevant to building a **trainable continuous analog computer made of reactive fluids**. The boundary is deliberately broad, spanning:

- Chemical reaction network (CRN) theory and learning.
- DNA and enzymatic molecular neural networks.
- Reaction-diffusion computing.
- Chemical reservoir computing.
- Physical / neuromorphic computing (photonic, iontronic, memristive).

The goal is to identify what has been done, what is missing, and where a thesis contribution can sit.

---

## 2. Paper clusters

### 2.1 CRN theory and foundations

Key references:

| Paper | Contribution |
|-------|--------------|
| Feinberg (2019) *Foundations of CRN Theory* | Definitive mathematical treatment; deficiency zero/one theorems. |
| Gunawardena (2003) lecture notes | Accessible introduction to mass-action CRNs. |
| Horn & Jackson (1972) | General mass-action kinetics. |
| Gillespie (1977) | Stochastic simulation algorithm for CRNs. |

These provide the ODE/stochastic formalism and stability guarantees needed to formulate a CRN Fluid NN rigorously.

### 2.2 Trainable CRNs and molecular neural networks

| Paper | Contribution |
|-------|--------------|
| Qian, Winfree, Bruck (2011) | First DNA strand-displacement neural network (Hopfield memory). |
| Cherry & Qian (2018) | Scaled DNA winner-take-all networks. |
| Cherry & Qian (2025) | Supervised learning in DNA neural networks. |
| Song & Qian (2025) | Heat-rechargeable DNA logic and neural nets. |
| Okumura et al. (2022) | Enzymatic winner-take-all neural network. |
| Genot, Fujii, Rondelez (2012) | Competition motifs for biochemical WTA. |
| Cai et al. (2024) | Efficient computation by molecular competition. |
| Pandi et al. (2019) | Metabolic perceptrons in cell-free systems. |
| Banda, Teuscher, Stefanovic (2015) | Chemical perceptron. |
| Daniel et al. (2013) | Synthetic analog computation in living cells. |

This cluster shows that **both DNA and enzymatic chemistry can implement perceptrons, WTA networks, and small classifiers**. The move from hand-design to optimization/training is recent and still limited to small networks.

### 2.3 Recurrent and trainable CRN architectures

| Paper | Contribution |
|-------|--------------|
| Dack et al. (2024) RNCRN | Recurrent neural CRNs; universal approximation proof. |
| Dack et al. (2025) SVM-CRN | CRN implementation of support vector machines. |
| Kang & Kim (2024) | Noise-robust CRN training of ANNs. |
| Poole et al. (2017, 2022, 2025) | CRN Boltzmann machines and autonomous generative learning. |
| Gopalkrishnan (2016) | CRN computation of maximum-likelihood estimators. |

This is the **closest theoretical cluster** to the proposed thesis. It establishes that CRNs can be trained to approximate dynamics and perform classification.

### 2.4 Chemical reservoir computing

| Paper | Contribution |
|-------|--------------|
| Baltussen et al. (2024) | Formose reaction reservoir; logic, XOR, chaos forecasting. |
| Nguyen et al. (2020) | Reservoir computing with random CRNs. |
| Yahiro et al. (2018) | Reservoir computing approach for molecular computing. |
| Csizi & Lörtscher (2024) | Perspective on complex CRNs for information processing. |
| Yirik (2025) ChemReservoir | Open-source framework for chemical reservoir computing. |

Reservoir computing sidesteps the need to design every reaction by using a **rich fixed dynamical system** and training only a linear readout. The Baltussen paper is the strongest experimental demonstration in this direction.

### 2.5 Reaction-diffusion computing

| Paper | Contribution |
|-------|--------------|
| Adamatzky & De Lacy Costello (2002) | Experimental XOR gate in RD medium. |
| Adamatzky, De Lacy Costello, Asai (2005) | Book: Reaction-Diffusion Computers. |
| Gorecki & Gorecka (2015) | Review of RD logic and pattern recognition. |
| Steinbock, Toth, Showalter (1996) | Logic gates by chemical oscillators. |
| Motoike et al. (2022) | Chemical wave computing from BZ to electrical systems. |

RD computing shows that **spatial chemical dynamics can implement logic**. Most work is hand-designed; learning/trainable RD classifiers are rare.

### 2.6 Physical and neuromorphic computing

| Paper | Contribution |
|-------|--------------|
| Hughes et al. (2019) | Wave physics as analog RNN. |
| Wright et al. (2022) | Deep physical neural networks trained with backpropagation. |
| Stern & Murugan (2023) | Learning without neurons in physical systems. |
| Stern et al. (2021) | Supervised learning in physical networks. |
| Momeni et al. (2023, 2025) | Backpropagation-free and in-situ training of physical NNs. |
| Kamsma et al. (2024) | Fluidic iontronic nanochannels for reservoir computing. |
| Xia et al. (2024, 2025) | Iontronic memristors and fluidic neuromorphic computing reviews. |

This cluster provides **comparative context** and shows the broader trend toward learning in physical substrates. The wave-physics paper is directly relevant to the abandoned acoustic direction.

### 2.7 DNA kinetics and design tools

| Paper | Contribution |
|-------|--------------|
| Zhang & Winfree (2009) | Toehold kinetics control. |
| Zhang & Seelig (2011) | Dynamic DNA nanotechnology review. |
| Simmel, Yurke, Singh (2019) | Principles and applications of strand displacement. |
| Kwiatkowska et al. (2012) | Probabilistic model checking of DNA devices. |
| Wang et al. (2023) | Parallel molecular computation on DNA registers. |

These are important for **experimental realizability** of DNA-based CRN networks.

---

## 3. Comparative landscape

| Approach | Trainable? | Spatial? | Experimental? | Energy scaling | Scalability |
|----------|------------|----------|---------------|----------------|-------------|
| DNA strand-displacement NN | Partially (gate concs) | No | Yes (test tube) | ~fJ/event | Limited by sequence design |
| Enzymatic WTA | Yes (enzyme amounts) | No | Yes | ~fJ–pJ/event | Limited by cross-talk |
| RNCRN (Dack et al.) | Yes (rates) | No | DNA possible | ~fJ/event | Small networks proven |
| Chemical reservoir (Formose) | Readout only | No (CSTR) | Yes | nJ/update | Large molecule count |
| Reaction-diffusion gates | Hand-designed | Yes | Yes (BZ) | nJ/update | Geometry-limited |
| Iontronic memristor | Yes (conductance) | Yes (channels) | Yes | µJ–nJ/update | Device-limited |
| Photonic/wave PNN | Yes (phase) | Yes | Yes | pJ–nJ/op | Fabrication-limited |

A clear gap: **trainable, spatially extended CRN classifiers that learn from examples rather than being hand-wired.**

---

## 4. Key open problems

1. **Training algorithms for CRNs.** Most work uses hand tuning or small-scale optimization. Gradient-based training through stiff ODEs is underexplored.
2. **Noise and robustness.** Molecular copy-number noise can destroy computation; only a few papers address this.
3. **Spatial learning.** RD systems that learn classification boundaries from data are rare.
4. **Physical realizability.** Optimized rate constants may not map to available DNA sequences or enzymes.
5. **Gating and programmability.** Switching between functions within the same chemical network is largely theoretical.
6. **Energy and delay benchmarking.** Few papers compare chemical computing against digital baselines rigorously.

---

## 5. Implications for this thesis

The proposed contribution — a **trainable well-mixed CRN classifier, extended to gated and reaction-diffusion variants** — sits in a fertile gap:

- RNCRNs show that trainable CRNs are theoretically possible.
- DNA/enzyme experiments show that elementary molecular classifiers work.
- RD computing shows that space adds computational power.
- Chemical reservoir computing shows that complex chemistry can compute.

What is missing is a **unified simulation study** that goes from a trainable CRN to a spatial RD classifier, with gating and parallelism, within a consistent framework.

---

## 6. Selected papers for deep dive

For Phase 3, the following papers will be read in full:

1. Dack et al. (2024) — RNCRN universal approximation.
2. Baltussen et al. (2024) — Formose reservoir computing.
3. Cherry & Qian (2025) — Supervised DNA neural networks.
4. Qian, Winfree, Bruck (2011) — Foundational DNA NN.
5. Okumura et al. (2022) — Enzymatic WTA.
6. Pandi et al. (2019) — Metabolic perceptron.
7. Feinberg (2019) — CRN theory foundations.
8. Adamatzky & De Lacy Costello (2002) — RD XOR gate.
9. Gorecki & Gorecka (2015) — RD computing review.
10. Stern & Murugan (2023) — Learning without neurons.
11. Hughes et al. (2019) — Wave physics as RNN.
12. Kamsma et al. (2024) — Fluidic iontronic reservoir.

---

## 7. Files generated

- `phase2_survey/survey.md` (this file)
- `paper_db.jsonl` (master database, 63 papers)
- `phase2_survey/search_results/` (raw WebSearch outputs)


---

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


---

# Phase 5 Synthesis: Taxonomy, Debates, and Opportunities

This document synthesizes the Phase 3 deep-dive notes and Phase 4 tool survey into a structured overview of trainable chemical reaction networks (CRNs) and reaction–diffusion (RD) computing. The goal is to identify recurring design patterns, unresolved debates, and concrete research opportunities for the thesis.

---

## 1. Taxonomy of chemical/physical computing paradigms

We divide the surveyed work into four overlapping paradigms based on where the "programmability" lives.

| Paradigm | What is trained / designed | Representative work | Output modality | Timescale |
|----------|---------------------------|---------------------|-----------------|-----------|
| **Well-mixed trainable CRNs** | Rate constants / initial concentrations of a homogeneous CRN | Dack et al., Kang & Kim | Species concentrations | Seconds to hours |
| **Reservoir chemistry** | Only a linear readout on a fixed chemical reservoir | Baltussen et al. | Ion-spectrum or fluorescence time series | Minutes to tens of minutes |
| **Compiled molecular circuits** | Stoichiometry of DNA or enzyme gates | Qian et al., Okumura et al. | Binary or analog concentration | Minutes to hours |
| **Spatial reaction–diffusion computing** | Geometry / initial conditions / diffusion coefficients | Adamatzky & De Lacy Costello, Gorecki et al. | Spatial patterns / wave fronts | Seconds to minutes |
| **Fluidic / iontronic physical learning** | Channel geometry or readout weights | Kamsma et al. | Electrical current | Milliseconds to seconds |

These paradigms are not mutually exclusive. A long-term goal is to combine them: a trainable CRN running inside a structured RD medium, read out by simple sensors.

---

## 2. Cross-paradigm design patterns

### 2.1 Neural-network-to-chemistry mapping

Almost every trainable CRN paper maps the same ML primitives onto chemistry:

| ML concept | Chemical realization | Example |
|------------|----------------------|---------|
| Weighted sum | Bimolecular catalysis / annihilation | Dack perceptrons, Kang & Kim forward CRN |
| Activation function | Nonlinear steady-state of fast reactions | `sigma_gamma(z) = (z + sqrt(z^2 + 4 gamma))/2` |
| Bias | Constant production (chemostatted fuel) | Dack `beta_i`, Kang & Kim threshold species |
| Negative weights | Dual-rail encoding or fuel-mediated reverse reactions | Qian seesaw gates, Kang & Kim |
| Recurrent state | Executive species that are not consumed | Dack RNCRN |
| Loss gradient | Additional species that encode error/backprop | Kang & Kim backward CRN |

The activation `sigma_gamma` appears independently in Dack (as the quasi-static perceptron) and in Kang & Kim (as the smooth ReLU). This is strong convergent evidence that it is a natural chemical nonlinearity.

### 2.2 Separation of timescales

Trainable CRNs universally rely on timescale separation:

- **Fast perceptrons / hidden reactions** implement nonlinear functions quasi-statically.
- **Slow executive / update reactions** carry the network state and learnable parameters.

This mirrors the quasi-static approximation used in Dack's proof: if the hidden layer equilibrates quickly, the executive dynamics reduce to a recurrent neural network.

### 2.3 Energy supply

Every practical chemical computer is an **open system**. Fuels, precursors, or chemostatted species must be replenished:

- Dack: fuel complexes `L`, `T` held constant.
- Kang & Kim: input species and fuel species supplied continuously.
- Baltussen: CSTR with continuous inflow of formose reagents.
- Okumura: enzyme and template species present in droplets, but not regenerated within a run.

A closed-batch chemical computer with autonomous learning remains an open challenge.

### 2.4 Readout bottleneck

Reading a CRN state without perturbing it is hard. Solutions in the literature:

- **Mass spectrometry** (Baltussen): rich, high-dimensional, but destructive sampling.
- **Fluorescence** (Okumura, Pandi): non-destructive but typically one or few channels.
- **Electrochemical** (Kamsma): fast, but only for ionic species.
- **Optical / colorimetric** (Adamatzky): qualitative spatial patterns.

For the thesis, the readout is a modeling choice rather than an experimental constraint: simulation provides full species concentrations at no cost.

---

## 3. Comparative table of trainable CRN approaches

| Feature | Dack RNCRN | Kang & Kim | Poole dbCRN | Qian DNA NN | Okumura enzymatic |
|---------|------------|------------|-------------|-------------|-------------------|
| Trainable parameters | Rate constants / concentrations | Rate constants / concentrations | Free energies / rate ratios | Gate stoichiometry | Template stoichiometry |
| Training method | Gradient descent on abstract CRN | Single-pot forward+backward+update CRN | Equilibrium learning (companion work) | In-silico design | Model-based fitting |
| Activation | Smooth ReLU-like `sigma_gamma` | Smooth ReLU | Boltzmann / energy-based | Sharp threshold | Sharp Hill / step |
| Dynamics | Recurrent ODEs | Feed-forward + backprop CRN | Equilibrium sampling | Feed-forward threshold cascade | Feed-forward threshold cascade |
| Experimental validation | DNA simulation only | None | None | Yes (DNA) | Yes (enzymes) |
| Noise robustness | Not analyzed | Smooth activation improves robustness | Stochastic by construction | Limited by leak | ~10-20% tolerance |
| Scalability shown | <10 species | MNIST (hundreds of species in principle) | Toy distributions | 4-neuron Hopfield | 2-layer, ~10 neurons |

The thesis can position itself in the gap between Dack (theory) and Kang & Kim (practical robust training): implement and compare smooth-activation CRNs numerically, then extend to stochastic simulations.

---

## 4. Debates and trade-offs

### 4.1 Deterministic vs. stochastic models

- **Deterministic ODEs** (Dack, Kang & Kim) are mathematically tractable and enable gradient-based training.
- **Stochastic CRNs** (Poole, Gillespie simulators) are physically realistic at low copy number and can represent distributions, but training is harder.

**Implication for thesis:** Start with deterministic mass-action ODEs for trainability; use stochastic simulations to test robustness.

### 4.2 In-chemico learning vs. compile-and-run

- **Compile-and-run** (Qian, Okumura) works experimentally today but requires hand design and cannot adapt after fabrication.
- **In-chemico learning** (Dack, Kang & Kim) is more flexible but has not been demonstrated experimentally.

**Implication for thesis:** The simulation phase can explore in-chemico learning without waiting for experimental validation.

### 4.3 Well-mixed vs. spatially structured

- **Well-mixed CRNs** are easier to train and prove theorems about.
- **RD media** offer natural parallelism and visual output but are harder to program.

**Implication for thesis:** Use well-mixed CRNs for the core trainable-computation result; use RD simulations as a spatial extension.

### 4.4 Digital molecular logic vs. analog physical computation

- **Digital logic** (Qian-style DNA circuits) gives noise margins and composability but sacrifices energy and speed.
- **Analog computation** (CRN RNNs, reservoirs) uses fewer molecules and continuous states but is more sensitive to parameter variation.

**Implication for thesis:** The analog approach aligns with the "continuous analog wave computation" framing and can cite Stern & Murugan for justification.

---

## 5. Opportunities directly relevant to the thesis

1. **Original numerical experiments.** Implement Dack's RNCRN architecture and Kang & Kim's smooth-activation CRN in Python/SciPy. Reproduce XOR, bistability, and limit-cycle tasks. This is original work that does not require a wet lab.

2. **Stochastic robustness study.** Take the trained deterministic CRN and run Gillespie simulations at finite copy number. Quantify how task accuracy degrades with volume and molecule count. This connects Dack's deterministic theorem to Poole's stochastic perspective.

3. **Smooth vs. non-smooth activations.** Systematically compare ReLU, leaky ReLU, and smooth ReLU (`sigma_gamma`) under parameter noise. Kang & Kim claim robustness; a simulation study can verify and quantify this.

4. **Trainable reaction–diffusion patterns.** Move from well-mixed CRNs to a 1D or 2D RD solver. Optimize initial conditions, diffusion coefficients, or local reaction rates so that the medium implements a simple classifier (e.g., XOR in space). This occupies the underexplored middle ground between hand-designed RD gates and trainable well-mixed CRNs.

5. **Energy / fuel budgeting.** Model fuel consumption explicitly. Ask: how long can a trained CRN operate in a closed batch before performance degrades? This is a practical concern absent from most theoretical papers.

6. **Comparison with iontronic / acoustic benchmarks.** Use Kamsma's iontronic reservoir and Hughes's wave-RNN as benchmarks for energy, speed, and programmability. Argue where CRN computing wins (cheap reagents, parallel chemistry) and loses (slow, readout difficulty).

7. **CRN compiler path.** Use Peppercorn / Nuskell to show how an abstract trained CRN could be compiled to DNA strand displacement, even if only in simulation. This strengthens the experimental-feasibility argument.

---

## 6. Thesis-relevant narrative

The surveyed literature supports a clear story:

- **Existence proof:** Dack et al. proves that CRNs can approximate arbitrary ODE dynamics, i.e., CRNs are trainable analog recurrent networks.
- **Robustness mechanism:** Kang & Kim shows that smooth activations make chemical backpropagation robust to noise.
- **Experimental precedent:** Baltussen and Okumura show that real chemistry can support neural-network-like computation, even if not fully trainable today.
- **Spatial extension:** Adamatzky and Gorecki show that RD media can compute via wave interference, analogous to the original acoustic-wave idea.
- **Physical framing:** Stern & Murugan justifies why local physical learning is interesting independent of digital performance.

The thesis can therefore claim: *trainable CRNs offer a chemically realizable, continuous analog computing substrate that avoids the fabrication barriers of acoustic metamaterials while preserving the conceptual benefits of physical neural networks.*


---

# Phase 5: Gap Analysis

This document distills the unresolved problems and open research gaps identified during the deep dive. Each gap is stated, justified with references, and linked to a concrete thesis opportunity.

---

## Gap 1: Stochastic trainability at finite copy number

**Statement.** Universal approximation is proved for deterministic mass-action ODEs (Dack et al.), but it is not known whether the same trainable CRNs preserve their function under stochastic chemical kinetics at low molecule counts.

**Evidence.**
- Dack's proof assumes the thermodynamic limit; finite-copy-number effects are not bounded.
- Poole et al. handles stochastic CRNs, but only for detailed-balanced equilibrium distributions, not for the nonequilibrium recurrent dynamics needed for RNNs.
- Kang & Kim's noise analysis is deterministic (rate perturbations), not Gillespie-stochastic.

**Why it matters.** Any experimental implementation will operate at finite copy number. A trained CRN that works in ODE simulation may fail when reactions become discrete and noisy.

**Thesis opportunity.** Take the trained deterministic CRN from Dack/Kang and run stochastic simulations (Gillespie or chemical Langevin). Quantify the probability of correct classification / target dynamics as a function of reactor volume and initial copy number.

---

## Gap 2: No experimental demonstration of in-chemico learning

**Statement.** No published wet-lab experiment has shown a CRN that updates its own rate constants or weights in response to an error signal.

**Evidence.**
- Qian et al. and Okumura et al. compile fixed networks designed in silico.
- Baltussen et al. trains only the readout; the chemistry is fixed.
- Kang & Kim proposes a single-pot learning CRN but remains a simulation.

**Why it matters.** In-chemico learning is the defining feature of a "trainable" chemical computer versus a merely programmable one.

**Thesis opportunity.** Simulate the full Kang & Kim forward+backward+update CRN in a well-mixed reactor. Show that the weights converge to values that solve XOR or a regression task, and study how the learning rate and noise affect convergence.

---

## Gap 3: Readout without perturbation

**Statement.** Continuous, non-destructive readout of many chemical species in a well-mixed CRN remains unsolved.

**Evidence.**
- Baltussen uses mass spectrometry on sampled aliquots — destructive and slow.
- Okumura and Pandi use fluorescence reporters — limited to one or a few channels and require reporter species that may sequester products.
- Electrochemical readouts (Kamsma) work only for ionic outputs.

**Why it matters.** A trainable CRN needs an error signal, which implies measuring the output. If measurement consumes the output, the feedback loop is compromised.

**Thesis opportunity.** In simulation this gap disappears because concentrations are fully observable. The thesis can therefore focus on trainability first, then discuss readout engineering as a future-work concern.

---

## Gap 4: Scalability beyond toy systems

**Statement.** Demonstrations of trainable CRNs use fewer than ~10 species/neurons. Scaling to tens or hundreds of species is unproven.

**Evidence.**
- Dack's examples: 3–8 perceptrons.
- Okumura: ~10-neuron two-layer network.
- Kang & Kim: MNIST in principle, but no detailed network-size analysis.

**Why it matters.** Useful computation (e.g., even simple pattern classification) may require dozens of species, and the training landscape may become harder as the network grows.

**Thesis opportunity.** Perform scaling experiments in simulation: train RNCRNs with 2, 4, 8, 16 perceptrons on increasingly complex tasks. Measure training success rate and convergence time. This provides an original empirical scalability curve.

---

## Gap 5: Fuel and energy budget

**Statement.** Trainable CRNs are open systems that consume fuel species. The lifetime and energy efficiency of a chemical computer are rarely analyzed.

**Evidence.**
- Dack assumes chemostatted fuel complexes.
- Kang & Kim assumes continuous supply of inputs and fuels.
- Baltussen's CSTR is continuously fed.

**Why it matters.** A closed-batch "battery-powered" chemical computer would be more practical, but fuel depletion will degrade performance over time.

**Thesis opportunity.** Introduce finite fuel reservoirs into the simulation. Track task accuracy versus cumulative fuel consumption. Compare the energy per operation with digital and iontronic benchmarks.

---

## Gap 6: Trainable reaction–diffusion media

**Statement.** RD computers are typically hand-designed for specific logic gates. There is little work on optimizing RD parameters (diffusion coefficients, initial conditions, reaction rates) to perform arbitrary tasks.

**Evidence.**
- Adamatzky & De Lacy Costello design XOR gates by cutting gel geometry.
- Gorecki et al. review pre-designed BZ droplet networks.
- No gradient-based training of RD patterns for classification was found in the survey.

**Why it matters.** RD media naturally implement parallel wave computation. If their parameters can be learned, they become trainable spatial analog computers.

**Thesis opportunity.** Build a 1D or 2D finite-difference RD solver. Define a simple spatial task (e.g., classify two input patterns by final concentration profile). Optimize diffusion coefficients or initial conditions via gradient descent or evolutionary search. This is a genuinely novel contribution at the intersection of CRNs and RD computing.

---

## Gap 7: Benchmarking against fluidic / acoustic alternatives

**Statement.** The relative advantages of CRN computing versus iontronic, memristive, or acoustic-wave physical neural networks are not quantitatively compared in the literature.

**Evidence.**
- Kamsma et al. compares iontronic reservoirs to solid-state memristors but not to chemical systems.
- Hughes et al. compares wave-RNN to digital RNNs but not to chemistry.
- Baltussen does not compare the formose reservoir to other physical computers.

**Why it matters.** The thesis must justify why CRNs are a worthwhile substrate compared to faster or more mature alternatives.

**Thesis opportunity.** Construct a qualitative comparison table (energy, speed, programmability, readout, fabrication) using published numbers. Where simulation provides data (e.g., operations per joule), make quantitative estimates.

---

## Gap 8: Compiler from trained CRN to experimental chemistry

**Statement.** While tools exist to compile abstract CRNs to DNA strand displacement (Peppercorn, Nuskell), no pipeline compiles a *trained* CRN with specific rate constants to a realizable experimental protocol.

**Evidence.**
- Dack argues DNA implementability but does not provide a full compiler pipeline.
- Nuskell maps CRNs to DSD systems, but the mapping assumes idealized rates.

**Why it matters.** Without a compiler, a trained CRN remains an abstract model.

**Thesis opportunity.** As a conceptual future-work section, outline how a trained CRN could be passed through Nuskell + NUPACK/Nuad. Identify which rate constants map to which DNA toehold lengths/concentrations, and discuss the mismatch between desired and achievable rates.

---

## Summary of thesis-relevant gaps

| Priority | Gap | Feasible in simulation? | Original contribution potential |
|----------|-----|------------------------|---------------------------------|
| High | Stochastic trainability | Yes | High |
| High | In-chemico learning simulation | Yes | High |
| High | Scalability curves | Yes | High |
| Medium | Fuel / energy budget | Yes | Medium-High |
| Medium | Trainable RD media | Yes | High |
| Medium | Benchmarking vs. other substrates | Partially (qualitative + some numbers) | Medium |
| Lower | Non-destructive readout | No (engineering problem) | Conceptual discussion |
| Lower | CRN-to-DNA compiler | Partially (conceptual) | Low-Medium |

The highest-impact simulation work for the thesis is: **stochastic trainability, in-chemico learning, and scalability of smooth-activation CRNs**, followed by **trainable RD media** as a spatial extension.


---

## Code & Tools

# Phase 4: Code and Tools Survey

Open-source repositories and software tools relevant to trainable chemical reaction networks (CRNs) and reaction–diffusion computing.

## 1. End-to-end trainable CRN implementations

### 1.1 Dack et al. — Recurrent neural CRNs (RNCRNs)

- **Repository:** `alexdack/recurrent_neural_chemical_reaction_networks`
- **URL:** https://github.com/alexdack/recurrent_neural_chemical_reaction_networks
- **Language / stack:** Python 3.10 + TensorFlow 2.13 / Keras + Jupyter; MATLAB 2023a for ODE simulation
- **License:** Check repository for details
- **Description:** Code and data for *Recurrent neural chemical reaction networks that approximate arbitrary dynamics* (Cell Systems 2026 / arXiv:2406.03456). Trains a fast neural subsystem in Python/Keras, exports parameters, and simulates the mass-action ODE in MATLAB. Includes examples for bistable, tristable, oscillatory, chaotic, DNA-implementation, robustness, and scaling studies.
- **Relevance:** Directly implements the RNCRN universal-approximation framework; a template for train-then-verify workflows.
- **Caveats:** Mixed Python/MATLAB dependency; TensorFlow-based training does not enforce physical realizability.

### 1.2 Baltussen et al. — Formose reservoir computer

- **Repository:** `huckgroup/Formose_reservoir_computation`
- **URL:** https://github.com/huckgroup/Formose_reservoir_computation
- **Language / stack:** Python 3.11, Jupyter, Conda, scikit-learn, pandas, numpy; AMICI for metabolic reference model
- **License:** BSD-3-Clause
- **Description:** All data and analysis notebooks for *Chemical reservoir computation in a self-organizing reaction network* (Nature 2024). Preprocesses ion-mobility mass-spectrometry data from a formose-reaction CSTR, trains linear readouts, and reproduces all data figures. Covers nonlinear classification, metabolic-model emulation, and chaotic time-series forecasting.
- **Relevance:** Complete experimental + computational pipeline for chemical reservoir computing; benchmark for converting chemical time series into trained readouts.
- **Caveats:** Chemistry-specific; readout is linear, so the reservoir itself is not trained.

### 1.3 Lakin et al. — ProBioSim

- **Reference:** *Design and Simulation of a Multilayer Chemical Neural Network That Learns via Backpropagation*, Artificial Life 29(3), 2023.
- **Language / stack:** Python with a domain-specific language for molecular circuits
- **Description:** Python library and DSL for specifying reactions, rate constants, and scheduled perturbations; supports deterministic (ODE) and stochastic CRN simulation. Used to train a multilayer chemical neural network via backpropagation.
- **Relevance:** CRN-first training environment, as opposed to generic ML training followed by translation.
- **Caveats:** Smaller community than general-purpose simulators; documentation may be limited.

## 2. General-purpose CRN and reaction–diffusion simulators

### 2.1 Catalyst.jl

- **Repository:** `SciML/Catalyst.jl`
- **URL:** https://github.com/SciML/Catalyst.jl
- **Language / stack:** Julia; wraps ModelingToolkit.jl, Symbolics.jl, DifferentialEquations.jl, JumpProcesses.jl
- **Description:** Symbolic DSL for CRNs; generates and compiles ODE, chemical Langevin SDE, and Gillespie jump-process representations. Supports spatial reaction–diffusion master equations and method-of-lines interfaces.
- **Relevance:** Best-in-class performance and symbolic analysis for CRNs and RD systems; ideal for gradient-based training of rates or diffusion coefficients.
- **Caveats:** Requires Julia; not in the current Python-centric stack.

### 2.2 BioCRNPyler + Bioscrape / libRoadRunner

- **Repositories:**
  - https://github.com/BuildACell/BioCRNPyler
  - https://github.com/biocircuits/bioscrape
  - https://github.com/sys-bio/roadrunner
- **Language / stack:** Python; SBML
- **Description:** BioCRNPyler composes CRNs and exports to SBML; Bioscrape and libRoadRunner provide fast stochastic and deterministic simulation.
- **Relevance:** Keeps workflow in Python; SBML export enables interoperability and verification.
- **Caveats:** BioCRNPyler is pre-1.0; spatial RD is not a primary focus.

### 2.3 COEL

- **Repository:** https://github.com/coel-sim/coel (Zenodo doi:10.5281/zenodo.46544)
- **Web instance:** https://coel-sim.org
- **Language / stack:** Web/cloud backend
- **Description:** Web-based ODE simulator for CRNs and multicompartment networks; includes plotting, DNA-strand-displacement transformation, SBML/Octave/MATLAB export, and a genetic-algorithm rate-constant optimizer.
- **Relevance:** Quick prototyping; GA optimization is a lightweight alternative to gradient-based training.
- **Caveats:** Cloud or self-hosted; not designed for spatial RD or HPC sweeps.

### 2.4 COPASI

- **URL:** https://copasi.org
- **Language / stack:** C++ with GUI and Python bindings (Basico)
- **Description:** Industrial-strength biochemical network simulator with ODE, SDE, and Gillespie solvers; parameter estimation, sensitivities, and SBML support.
- **Relevance:** Gold-standard verification tool; can be scripted from Python via `basico`.
- **Caveats:** Standalone application; spatial RD requires external PDE coupling.

### 2.5 python-crn

- **Repository:** `enricozb/python-crn`
- **URL:** https://github.com/enricozb/python-crn
- **Language / stack:** Python 3.6+, NumPy, SciPy, SymPy, StochPy
- **Description:** Compact Python CRN simulator with simple reaction-literal syntax; supports deterministic and stochastic simulation.
- **Relevance:** Lightweight starting point for a custom Python CRN simulator.
- **Caveats:** Repository is no longer actively maintained.

## 3. DNA / chemistry compilers and sequence designers

### 3.1 Peppercorn

- **Repository:** `DNA-and-Natural-Algorithms-Group/peppercorn`
- **URL:** https://github.com/DNA-and-Natural-Algorithms-Group/peppercorn
- **Language / stack:** Python
- **Description:** Enumerates and condenses domain-level DNA strand-displacement reaction networks.
- **Relevance:** Bridge from abstract CRN to experimental DNA chemistry.

### 3.2 Nuskell

- **Repository:** `DNA-and-Natural-Algorithms-Group/Nuskell`
- **URL:** https://github.com/DNA-and-Natural-Algorithms-Group/Nuskell
- **Language / stack:** Python
- **Description:** Compiler from abstract CRNs to domain-level DNA strand-displacement systems; uses Peppercorn for enumeration.
- **Relevance:** Automated path from a trained CRN to candidate DNA molecules.

### 3.3 NUPACK / Nuad

- **NUPACK:** https://www.nupack.org
- **Nuad repository:** https://github.com/UC-Davis/nuad
- **Description:** NUPACK performs thermodynamic analysis and sequence design for nucleic acids; Nuad supports custom DNA sequence designers.
- **Relevance:** Needed only if moving from abstract CRNs to experimental DNA sequence design.

## 4. Recommendations

| Stage | Suggested tool(s) |
|-------|-------------------|
| Rapid CRN prototyping | `python-crn`, BioCRNPyler + Bioscrape, or Catalyst.jl |
| Training CRN weights | Dack RNCRN repo or Catalyst.jl + SciMLSensitivity |
| Reservoir-computing benchmark | `huckgroup/Formose_reservoir_computation` |
| Spatial RD extension | Catalyst.jl (spatial SSA / MOL) or custom Python finite-difference solver |
| DNA implementation | Peppercorn + Nuskell + NUPACK/Nuad |
| Gold-standard verification | COPASI (via `basico`) |

For the immediate project roadmap, the practical path is:

1. Implement a small well-mixed CRN XOR gate in Python/SciPy.
2. Verify the same CRN in COPASI or BioCRNPyler/Bioscrape.
3. Extend to 1D/2D reaction–diffusion via a custom NumPy solver or Catalyst.jl.
4. Use the Formose reservoir code as a benchmark for data formatting and readout training.


---

## References

[1] Alexander Dack, Michael te Vrugt, Wilhelm T. S. Huck et al.. "Recurrent neural chemical reaction networks that approximate arbitrary dynamics". 2024. arXiv.

[2] Mathieu G. Baltussen, Thijs J. de Jong, Quentin Duez et al.. "Chemical reservoir computation in a self-organizing reaction network". 2024. Nature.

[3] Kevin M. Cherry, Lulu Qian. "Supervised learning in DNA neural networks". 2025. Nature.

[4] Tianqi Song, Lulu Qian. "Heat-rechargeable computation in DNA logic circuits and neural networks". 2025. Nature.

[5] S. Ghosh, et al.. "A recursive enzymatic competition network capable of multitask molecular information processing". 2025. Nature Chemistry.

[6] J. Parres-Gold, M. Levine, B. Emert et al.. "Contextual computation by competitive protein dimerization networks". 2025. Cell.

[7] Z. Chen, et al.. "A synthetic protein-level neural network in mammalian cells". 2024. Science.

[8] S. Kang, J. Kim. "Noise-robust chemical reaction networks training artificial neural networks". 2024. arXiv.

[9] William Poole, Thomas E. Ouldridge, Manoj Gopalkrishnan. "Autonomous learning of generative models with chemical reaction network ensembles". 2025. Journal of the Royal Society Interface.

[10] X. Xiong, T. Zhu, Y. Zhu et al.. "Molecular convolutional neural networks with DNA regulatory circuits". 2022. Nature Machine Intelligence.

[11] Lulu Qian, Erik Winfree, Jehoshua Bruck. "Neural network computation with DNA strand displacement cascades". 2011. Nature.

[12] Kevin M. Cherry, Lulu Qian. "Scaling up molecular pattern recognition with DNA-based winner-take-all neural networks". 2018. Nature.

[13] S. Okumura, et al.. "Nonlinear decision-making with enzymatic neural networks". 2022. Nature.

[14] A. J. Genot, T. Fujii, Y. Rondelez. "Computing with competition in biochemical networks". 2012. Physical Review Letters.

[15] H. Cai, X. Zhang, R. Qiao et al.. "Efficient computation by molecular competition networks". 2024. Physical Review Research.

[16] A. Pandi, et al.. "Metabolic perceptrons for neural computing in biological systems". 2019. Nature Communications.

[17] R. Daniel, J. R. Rubens, R. Sarpeshkar et al.. "Synthetic analog computation in living cells". 2013. Nature.

[18] Martin Feinberg. "Foundations of Chemical Reaction Network Theory". 2019. Springer.

[19] Jeremy Gunawardena. "Chemical reaction network theory for in-silico biologists". 2003. Harvard lecture notes.

[20] J. Gorecki, J. N. Gorecka. "Chemical computing with reaction–diffusion processes". 2015. Philosophical Transactions of the Royal Society A.

[21] Andrew Adamatzky, Ben De Lacy Costello. "Experimental logical gates in a reaction-diffusion medium: the XOR gate and beyond". 2002. Physical Review E.

[22] Andrew Adamatzky, Ben De Lacy Costello, Tetsuya Asai. "Reaction-Diffusion Computers". 2005. Elsevier.

[23] O. Steinbock, A. Toth, K. Showalter. "Logic gates by chemical oscillators". 1996. Science.

[24] Klaus-Peter Zauner. "Molecular computing: paths to chemical Turing machines". 2015. Chemical Science.

[25] Friedrich C. Simmel, Bernard Yurke, Harish R. Singh. "Principles and applications of nucleic acid strand displacement reactions". 2019. Chemical Reviews.

[26] David Y. Zhang, Georg Seelig. "Dynamic DNA nanotechnology using strand-displacement reactions". 2011. Nature Chemistry.

[27] David Y. Zhang, Erik Winfree. "Control of DNA strand displacement kinetics using toehold exchange". 2009. Journal of the American Chemical Society.

[28] B. Wang, et al.. "Parallel molecular computation on digital data stored in DNA". 2023. PNAS.

[29] M. Kwiatkowska, et al.. "Design and analysis of DNA strand displacement devices using probabilistic model checking". 2012. Journal of the Royal Society Interface.

[30] William Poole, Thomas E. Ouldridge, Manoj Gopalkrishnan. "Chemical boltzmann machines". 2017. DNA Computing and Molecular Programming.

[31] William Poole, Thomas E. Ouldridge, Manoj Gopalkrishnan et al.. "Detailed balanced chemical reaction networks as generalized boltzmann machines". 2022. arXiv.

[32] Manoj Gopalkrishnan. "A scheme for molecular computation of maximum likelihood estimators for log-linear models". 2016. DNA Computing and Molecular Programming.

[33] R. Pei, E. Matamoros, M. Liu et al.. "Training a molecular automaton to play a game". 2010. Nature Nanotechnology.

[34] A. Dack, M. te Vrugt, W. T. S. Huck et al.. "Implementation of Support Vector Machines using Reaction Networks". 2025. arXiv.

[35] Hoang Nguyen, Peter Banda, Darko Stefanovic et al.. "Reservoir computing with random chemical systems". 2020. Artificial Life Conference Proceedings.

[36] Wataru Yahiro, Nathanael Aubert-Kato, Masami Hagiya. "A reservoir computing approach for molecular computing". 2018. Artificial Life Conference Proceedings.

[37] Katja-Sophia Csizi, Emanuel Lörtscher. "Complex chemical reaction networks for future information processing". 2024. Frontiers in Neuroscience.

[38] Matthew Stern, Arvind Murugan. "Learning Without Neurons in Physical Systems". 2023. Annual Review of Condensed Matter Physics.

[39] Matthew Stern, David Hexner, J. W. Rocks et al.. "Supervised Learning in Physical Networks: From Machine Learning to Learning Machines". 2021. Physical Review X.

[40] A. Momeni, B. Rahmani, M. Mallejac et al.. "Backpropagation-free training of deep physical neural networks". 2023. Science.

[41] A. Momeni, et al.. "Training of physical neural networks". 2025. Nature.

[42] Logan G. Wright, et al.. "Deep physical neural networks trained with backpropagation". 2022. Nature.

[43] Tyler W. Hughes, Ian A. D. Williamson, Momchil Minkov et al.. "Wave physics as an analog recurrent neural network". 2019. Science Advances.

[44] W. Kamsma, et al.. "Brain-inspired reservoir computing with fluidic iontronic nanochannels". 2024. Nature Communications.

[45] Y. Xia, et al.. "Recent advances in fluidic neuromorphic computing". 2025. APL Review.

[46] Y. Xia, C. Zhang, Z. Xu et al.. "Organic iontronic memristors for artificial synapses and bionic neuromorphic computing". 2024. Nanoscale.

[47] A. R. Podolsky, et al.. "DNA Input Classification by a Riboregulator-Based Cell-Free Perceptron". 2022. ACS Synthetic Biology.

[48] R. Datta, Z. Guan, B. Espinoza et al.. "Realizing Reduced and Sparse Biochemical Reaction Networks from Dynamics". 2025. arXiv.

[49] Z. Xiang, et al.. "Tracking large chemical reaction networks and rare events by neural networks". 2025. arXiv.

[50] A. Dack, M. te Vrugt, W. T. S. Huck et al.. "Implementation of Support Vector Machines using Reaction Networks". 2025. arXiv.

[51] A. Dack, et al.. "Machine Learning-Driven Chemical Reactor Network Modeling of the Sandia-D Flame". 2026. arXiv.

[52] H. Zhang, D. V. Vargas. "A survey on reservoir computing and its interdisciplinary applications beyond traditional machine learning". 2023. IEEE Access.

[53] C. Wringe, M. Trefzer, S. Stepney. "Reservoir Computing: A Tutorial Review and Critique". 2025. International Journal of Parallel, Emergent and Distributed Systems.

[54] J. D. A. K.. "The electrofluidic brain as a basement layer for neural computation". 2026. Nature Communications Biology.

[55] P. Banda, C. Teuscher, D. Stefanovic. "A chemical perceptron". 2015. Natural Computing.

[56] I. N. Motoike, et al.. "Chemical Wave Computing from Labware to Electrical Systems". 2022. Electronics.

[57] S. M. Dillavou, et al.. "A universal approximation theorem for nonlinear resistive networks". 2024. arXiv.

[58] Leon Chua. "Memristor, Hodgkin-Huxley, and edge of chaos". 2013. Nanotechnology.

[59] Daniel T. Gillespie. "Stochastic simulation of chemical kinetics". 1977. Journal of Computational Physics.

[60] Warren S. McCulloch, Walter Pitts. "A logical calculus of the ideas immanent in nervous activity". 1943. Bulletin of Mathematical Biophysics.

[61] C. Thöni, et al.. "An analogue approach to computational function in chemical reaction networks". 2024. Radboud University MSc thesis.

[62] Chiara Thöni. "Modelling chemical reaction networks using neural differential equations". 2024. Radboud University MSc thesis.

[63] M. A. Yirik. "ChemReservoir: An open-source framework for chemical reservoir computing". 2025. GitHub / arXiv.
