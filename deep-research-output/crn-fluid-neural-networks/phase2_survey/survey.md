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
