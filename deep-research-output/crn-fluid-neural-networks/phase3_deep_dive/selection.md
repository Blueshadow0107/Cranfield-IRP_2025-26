# Phase 3 Paper Selection

This document records the 12 papers selected for deep reading in Phase 3 and the rationale for each selection.

## Selection Criteria

1. **Direct relevance** to trainable chemical reaction networks (CRNs) or reaction–diffusion (RD) computing.
2. **Representativeness** across the three implementation layers: theory, chemistry/biochemistry, and physical/continuum systems.
3. **Citation impact** and recency; peer-reviewed work preferred.
4. **Methodological diversity**: universal-approximation proofs, reservoir computing, supervised learning, DNA implementation, enzyme networks, noise-robust training, and RD logic.

## Selected Papers

| # | First Author | Year | Title | Venue | Why selected |
|---|--------------|------|-------|-------|--------------|
| 1 | Dack | 2024/2026 | Recurrent neural chemical reaction networks that approximate arbitrary dynamics | Cell Systems (arXiv:2406.03456) | Theoretical foundation: proves CRNs can approximate arbitrary ODEs via modular chemical neurons; closest thing to a universal trainable CRN. |
| 2 | Baltussen | 2024 | Chemical reservoir computation in a self-organizing reaction network | Nature | Strongest experimental benchmark for chemical reservoir computing; full data/code available. |
| 3 | Poole et al. | 2017 | Detailed balanced chemical reaction networks as Boltzmann machines | (Workshop/preprint) | Connects CRN equilibrium distributions to probabilistic neural networks; bridges statistical mechanics and trainable CRNs. |
| 4 | Okumura et al. | 2022 | Enzymatic neural network | Nature Communications | Wet-lab enzyme-based perceptron and multilayer network; demonstrates biochemical trainable weights. |
| 5 | Qian, Winfree & Bruck | 2011 | Neural network computation with DNA strand displacement cascades | Nature | Landmark DNA implementation of feedforward neural networks; establishes molecular feasibility. |
| 6 | Kang & Kim | 2024 | Noise-robust chemical reaction networks training artificial neural networks | arXiv (preprint) | Addresses a key practical issue (stochastic noise) and proposes smooth activation CRNs trained end-to-end. |
| 7 | Pandi et al. | 2019 | Metabolic perceptrons for neural computing | Nature Communications | Shows living/metabolic CRNs can implement perceptron-like learning; biology-to-computation link. |
| 8 | Kamsma et al. | 2024 | Fluidic iontronic nanochannels for neuromorphic sensing | Nature | Ion-channel/fluidic computing hardware; relevant for future ionic/fluidic physical neural network directions. |
| 9 | Stern & Murugan | 2023 | Learning without neurons in physical systems | Annu. Rev. Condens. Matter Phys. | Conceptual survey of physical learning; places chemical/fluid computing in broader context. |
| 10 | Hughes et al. | 2019 | Wave physics as analog recurrent neural network | Science Advances | Formal mapping of wave propagation to RNNs; relevant analog-computing counterpart and potential bridge to acoustic/fluid computing. |
| 11 | Adamatzky & De Lacy Costello | 2002 | Experimental logical gates in a reaction-diffusion medium | Phys. Rev. E | Classic experimental RD logic gates (XOR); spatial chemical computing proof-of-concept. |
| 12 | Gorecki, Gorecka & Igarashi | 2009 | Information processing with chemical excitable media | Phys. Rev. E | Reviews information processing in excitable RD media; relevant for spatial extension of CRNs. |

## How the selection covers the topic

- **Theory / universal approximation:** Dack, Poole
- **Supervised / trainable chemical NNs:** Okumura, Kang & Kim, Pandi
- **Reservoir / emergent computation:** Baltussen
- **Molecular implementation:** Qian et al.
- **Physical / continuum analogues:** Hughes, Stern & Murugan, Kamsma
- **Spatial reaction–diffusion logic:** Adamatzky & De Lacy Costello, Gorecki et al.

This mix ensures the deep-dive notes support both the core CRN focus of the thesis and the longer-term reaction–diffusion / physical-computing extensions.
