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
