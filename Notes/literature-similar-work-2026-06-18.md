# Literature Search: Similar Work and Implementations

Date: 2026-06-18
Scope: acoustic/flow physical neural networks, diffractive computing, acoustic logic gates, inverse design of scatterer arrays

---

## 1. Executive summary

The idea of using a structured fluid/porous domain as a trainable analog wave computer sits at the intersection of three well-documented fields:

1. **Wave-based physical neural networks** — treating wave propagation as a recurrent or layered matrix transformation (Hughes et al. 2019, Wright et al. 2022, Lin et al. 2018).
2. **Acoustic metamaterial analog computing** — designing static structures that perform mathematical operations or logic gates (Silva et al. 2014, Zuo et al. 2018, Zangeneh-Nejad & Fleury 2021).
3. **Physical reservoir computing with fluids** — using flow vortices, water waves, or bubbles as a nonlinear dynamical reservoir (Goto et al. 2021, Maksymov & Pototsky 2023, Notsu & Nakajima 2021).

For the current Cranfield IRP, the most directly comparable recent work is **Kalthoff et al. (2025)** "Acoustic neural networks: Identifying design principles and exploring physical feasibility", which explicitly maps simple RNNs onto passive acoustic systems and identifies the same two constraints we face:

- weights are transmission coefficients limited to `[0, 1]` because the medium is passive;
- nonlinearity must be introduced by the medium, the readout, or a downstream activation.

That paper uses a **digital-twin / physics-constrained RNN** rather than a full-wave solver, and focuses on temporal audio classification (AudioMNIST). It does not report a 2-D FDTD inverse-design loop for static scatterer arrays. This leaves a clear gap for the present project: a finite-difference acoustic model of a trainable 2-D obstacle/porous domain, validated against CFD/OpenFOAM, with a readout layer providing the required nonlinearity.

---

## 2. Directly comparable work: acoustic neural networks

### Kalthoff et al. (2025) — arXiv:2511.21313
- **What they do:** Map neural-network operations onto acoustic processes. Connection weights = acoustic transmission coefficients; summation = wave superposition; activation = intensity-dependent attenuation (offset ReLU on intensity).
- **Key result:** Constrained RNNs with non-negative weights and offset-ReLU/absolute-value activations can reach ~95% accuracy on AudioMNIST, approaching unconstrained digital networks for large hidden sizes.
- **Implication for us:** Confirms that a passive linear acoustic layer + constrained readout can perform useful computation. Also confirms the need for nonlinearity (their activation is a digital proxy for nonlinear acoustic attenuation).
- **Gap:** No spatial FDTD or trainable geometry; the acoustic mapping is conceptual/model-level.

### Hughes et al. (2019) — *Science Advances*
- **What they do:** Show that a discretized linear wave equation is mathematically equivalent to an RNN; the medium's refractive-index distribution plays the role of recurrent weights.
- **Implication for us:** Provides the theoretical justification for treating our obstacle/porous domain as a recurrent or single-layer physical network.

### Wright et al. (2022) — *Nature*
- **What they do:** Deep physical neural networks (PNNs) trained via a hybrid in situ / in silico algorithm across mechanical, electronic, and optical systems.
- **Implication for us:** Demonstrates that real physical systems can be trained as layers in a deep network, but the experimental setup is complex. Our project can follow the same conceptual split: physical wave layer + digital readout/training.

---

## 3. Closest optical analog: diffractive deep neural networks

### Lin et al. (2018) — *Science*
- **What they do:** 3-D printed diffractive layers act as a physical neural network for image classification; each pixel modulates phase, and free-space diffraction performs the matrix multiplication.
- **Implication for us:** The canonical proof that a static wave-propagating medium can be optimized to perform inference. The acoustic analog would replace optical phase modulation with sound-speed/impedance modulation from obstacles or porous regions.
- **Limitation:** Requires many precisely fabricated layers; our project can explore whether a single fluid layer with obstacles is expressive enough for simple benchmarks.

---

## 4. Acoustic metamaterial computing and logic gates

### Zuo et al. (2018) — *Scientific Reports*
- **What they do:** Acoustic analog computing using metamaterials (differentiation, integration, equation solving).
- **Implication for us:** Shows that engineered acoustic structures can act as analog math kernels. Our obstacle-array transmission-loss maps are essentially tunable spatial filters of the same kind.

### Zangeneh-Nejad & Fleury (2021) — *Nature Reviews Materials*
- **What they do:** Review of wave-based-metamaterial analog computing, including spatial differentiators/integrators, equation solvers, and logic gates.
- **Implication for us:** Good source for the broader intellectual framing; confirms that passive linear metamaterials can implement matrix-like operations but need external nonlinearity for full neural-network universality.

### Banerjee et al. (2023) — tuneable phononic-crystal logic gates
- **What they do:** 35 x 35 phononic-crystal matrix with rotatable square columns; AND, OR, NAND, NOR operations realized using Dirac-cone/deaf-band physics.
- **Implication for us:** Demonstrates that a single static acoustic structure can perform multiple Boolean operations, but tuning is mechanical rotation rather than optimization of fixed geometry.

### Liu et al. (2026) — valley-locked interface-state logic gates
- **What they do:** Experimental AND/OR/XOR in a 2-D acoustic metamaterial using valley-locked waveguides; robust to defects.
- **Implication for us:** Proof that acoustic logic gates can be robust and experimentally realizable. The topology is specialized, not trainable.

### Parsa et al. (2022) — granular metamaterial logic gates
- **What they do:** Evolutionary search finds placements of stiff/soft grains in a granular metamaterial to implement AND and XOR gates.
- **Implication for us:** Direct precedent for using optimization/search to discover computational structure in a physical medium. Our cylinder-array geometry search is conceptually similar but uses continuum acoustics instead of discrete grains.

---

## 5. Inverse design of acoustic scatterer arrays

This is the methodology most relevant to our planned FDTD + optimization loop.

### Donda et al. (2025) — review: "Machine learning for inverse design of acoustic and elastic metamaterials"
- **What they do:** Survey of CNNs, GANs, VAEs, reinforcement learning, TNNs, and PINNs for metamaterial inverse design.
- **Relevant examples:**
  - Shah et al. (2021) used deep Q-learning to adjust cylindrical scatterer positions to minimize total scattering cross-section (TSCS).
  - Zhou et al. combined cGAN + ResNet for 2-D tunable phononic crystals.
- **Implication for us:** Machine-learning surrogate models can dramatically speed up inverse design, but a simple parameter sweep / gradient-free optimizer on an FDTD solver is a valid first step and matches our current tooling.

### Acoustic cloak design via VAE + supervised learning
- **What they do:** Train a VAE to encode cylinder configurations and a CNN to predict TSCS; Gaussian-process optimization in latent space finds low-TSCS designs.
- **Implication for us:** A generative-model approach is feasible for obstacle-array design if the forward model is fast enough. Not necessary for a first demonstrator but useful for future work.

### Sun et al. (2021) — DNN inverse design of acoustic structures
- **What they do:** Fully connected network maps target sound-transmission-loss spectrum to equivalent geometric parameters of a Helmholtz resonator array.
- **Implication for us:** Similar output (STL spectrum) to what our FDTD sweep already produces; provides a benchmark for how inverse design can be formulated.

---

## 6. Fluid-flow-based physical reservoir computing (Option B comparison)

If the project later returns to a flow-driven substrate, the following literature is directly relevant.

### Goto et al. (2021) / Notsu & Nakajima (2021)
- **What they do:** Numerical simulation of flow past a cylinder used as a physical reservoir; NARMA benchmark.
- **Key finding:** Information-processing capability peaks just before transition to a Kármán vortex street; once vortex shedding is fully established, the echo-state property is lost.
- **Implication for us:** Explains why our OpenFOAM cylinder flow at Re=100 may not be a stable reservoir — the periodic vortex shedding violates the fading-memory property required for reservoir computing.

### Maksymov & Pototsky (2023, 2024)
- **What they do:** Reviews and develops fluid-flow-based reservoir computing, including pump-modulated input.
- **Implication for us:** Practical implementation recipes for flow-based RC, but confirms the need for careful Reynolds-number selection.

### Analogue and Physical Reservoir Computing Using Water Waves (arXiv:2306.09095)
- **What they do:** Comprehensive review of water-wave, bubble, and soliton reservoirs.
- **Relevant examples:**
  - Bubble-based RC: nonlinear bubble oscillations provide memory and nonlinearity.
  - KdV soliton / Aqua-PACMANN: shallow-water solitons implement XNOR.
  - Solitary-like surface waves (SLRC): good memory capacity due to merging solitons.
- **Implication for us:** Confirms that fluids can be reservoirs, but experimental control and stable dynamics are harder than for acoustic scattering.

---

## 7. Key physical mechanisms and what is still missing

| Required NN ingredient | Physical acoustic analog | Precedent | Status for this project |
|---|---|---|---|
| Weighted connections | Transmission/reflection/attenuation through structured paths | Hughes 2019, Kalthoff 2025 | FDTD sweep already shows tunable transmission loss |
| Linear superposition | Wave interference | Zuo 2018, Lin 2018 | Built into FDTD solver |
| Nonlinear activation | (a) Nonlinear medium, (b) intensity-dependent attenuation, (c) nonlinear readout | Kalthoff 2025, Ning 2025 (RF ReLU), bubble/soliton RC | Not yet implemented; simplest path is nonlinear readout |
| Trainable parameters | Geometry (cylinder positions/radii), porosity, flow speed | Shah 2021, Parsa 2022, Donda 2025 | Positions/radii are natural design variables |
| Readout layer | Linear classifier on output-field samples | All reservoir-computing work | Straightforward to add |

**What is missing in the literature:** A complete, open-source pipeline that combines
1. a validated 2-D acoustic FDTD solver with arbitrary obstacles/porous regions,
2. an OpenFOAM-validated flow/acoustics coupling for the same geometry,
3. a gradient-free or differentiable optimizer that trains the geometry for a simple classification/regression task,
4. a nonlinear readout layer that compensates for the linearity of the physical propagation.

The present Cranfield IRP is positioned to close this gap at the simulation/proof-of-concept level.

---

## 8. Suggested additions to `Report/references.bib`

The following entries should be added to the master bibliography (Vancouver/NLM style). Many are not yet in the current `.bib` file.

```bibtex
@article{kalthoff2025acoustic,
  title={Acoustic neural networks: Identifying design principles and exploring physical feasibility},
  author={Kalthoff, I and others},
  journal={arXiv preprint arXiv:2511.21313},
  year={2025}
}

@article{lin2018alloptical,
  title={All-optical machine learning using diffractive deep neural networks},
  author={Lin, X and Rivenson, Y and Yardimci, NT and others},
  journal={Science},
  volume={361},
  pages={1004--1008},
  year={2018}
}

@article{zangenehnejad2021analogue,
  title={Analogue computing with metamaterials},
  author={Zangeneh-Nejad, F and Sounas, DL and Al\\`u, A and Fleury, R},
  journal={Nature Reviews Materials},
  volume={6},
  pages={207--225},
  year={2021}
}

@article{goto2021vortex,
  title={Vortex, the key to information-processing capability in physical reservoir computing},
  author={Goto, H and Tsubaki, T and Notstu, H and Nakajima, K},
  journal={Neural Networks},
  year={2021}
}

@article{maksymov2023fluid,
  title={Fluid flow for physical reservoir computing: a review},
  author={Maksymov, I and Pototsky, A},
  journal={...},
  year={2023}
}

@article{shah2021deepq,
  title={Deep Q-learning for suppression of acoustic scattering by cylinders},
  author={Shah, AA and others},
  journal={Journal of the Acoustical Society of America},
  year={2021}
}

@article{donda2025mlreview,
  title={Machine learning for inverse design of acoustic and elastic metamaterials},
  author={Donda, K and others},
  journal={...},
  year={2025}
}

@article{ning2025multilayer,
  title={Multilayer nonlinear diffraction neural networks with programmable metasurfaces},
  author={Ning, YM and others},
  journal={...},
  year={2025}
}

@article{parsa2022evolution,
  title={Evolution of acoustic logic gates in granular metamaterials},
  author={Parsa, A and Wang, D and O'Hern, CS and Shattuck, MD and Kramer-Bottiglio, R and Bongard, J},
  journal={...},
  year={2022}
}

@article{liu2026valley,
  title={Experimental realization of acoustic logic gates based on valley-locked interface states in two-dimensional metamaterials},
  author={Liu, YD and others},
  journal={...},
  year={2026}
}
```

> Note: full Vancouver/NLM formatting should be filled in from the actual paper metadata before adding to `Report/references.bib`.

---

## 9. Strategic implications for the project

1. **Choose Option A (static acoustic scatterers) as the primary substrate.** The literature strongly supports this: faster simulation, easier optimization, and direct analogies to diffractive optical networks and acoustic metamaterial computing.
2. **Keep Option B (flow-driven reservoir) only as a comparative validation.** Literature shows it is harder to stabilize and less suitable for trainable wave computing at the MSc-project scale.
3. **Add a nonlinear readout layer immediately.** Kalthoff 2025 and Ning 2025 both show that linear propagation alone is insufficient; a readout nonlinearity (thresholded intensity, ReLU on power, or small digital layer) is the pragmatic choice.
4. **Frame the contribution as a simulation pipeline.** The novelty is not the existence of acoustic computation, but an integrated FDTD/OpenFOAM/optimization workflow for trainable 2-D acoustic scatterer arrays with a nonlinear readout.

---

Tags: #literature-review #acoustic-neural-network #metamaterial #reservoir-computing #inverse-design
