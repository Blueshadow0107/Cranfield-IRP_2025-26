# Phase 4: Code and Tools Survey

This survey inventories open-source repositories and software tools that are directly relevant to building, simulating, and training trainable chemical reaction networks (CRNs) and reaction–diffusion (RD) computers. The goal is to identify reusable code assets for the MSc project’s numerical roadmap, rather than to install or run them now.

Three categories are covered:

1. **End-to-end trainable CRN implementations** — code released with papers that design CRNs as neural networks.
2. **General-purpose CRN / RD simulators** — languages, libraries, and frameworks for specifying and integrating CRNs.
3. **DNA / chemistry compilers and sequence designers** — tools for lowering abstract CRNs into experimental chemistry.

---

## 1. End-to-end trainable CRN implementations

### 1.1 Dack et al. — Recurrent neural CRNs (RNCRNs)

- **Repository:** `alexdack/recurrent_neural_chemical_reaction_networks`
- **URL:** https://github.com/alexdack/recurrent_neural_chemical_reaction_networks
- **Language / stack:** Python 3.10 + TensorFlow 2.13 / Keras + Jupyter; MATLAB 2023a for ODE simulation
- **License:** Not explicitly stated in README; check repository for details
- **What it does:**
  - Implements the RNCRN construction from Dack, Qureshi, Ouldridge & Plesa, *Cell Systems* 2026 / arXiv:2406.03456.
  - Splits the workflow into:
    1. Train a fast “neural subsystem” in Python using gradient descent.
    2. Export parameters and simulate the mass-action ODE in MATLAB.
    3. Plot results with Jupyter notebooks.
  - Covers bistable, tristable, oscillatory, and chaotic target dynamics; includes a DNA-strand-displacement example and scaling/robustness analyses.
- **Relevance to this project:**
  - The RNCRN is the strongest theoretical foundation identified: executive species + fast chemical perceptrons give universal ODE approximation.
  - The two-step train-then-simulate pattern is a useful template: train weights in a high-level framework, then verify with a chemistry-aware ODE simulator.
  - However, it is well-mixed only; spatial RD extensions are not included.
- **Caveats:**
  - Mixed Python/MATLAB dependency chain complicates reproducibility without a MATLAB license.
  - Training relies on TensorFlow/Keras, not a chemistry simulator, so the learned parameters must still be checked for physical realizability.

### 1.2 Baltussen et al. — Formose reservoir computer

- **Repository:** `huckgroup/Formose_reservoir_computation`
- **URL:** https://github.com/huckgroup/Formose_reservoir_computation
- **Language / stack:** Python 3.11, Jupyter notebooks, Conda, scikit-learn, pandas, numpy; AMICI for in-silico carbon-metabolism reference model
- **License:** BSD-3-Clause
- **What it does:**
  - Provides all data, preprocessing, and analysis notebooks for *Chemical reservoir computation in a self-organizing reaction network*, *Nature* 2024.
  - Reads ion-mobility mass-spectrometry time series from a continuous-flow stirred-tank reactor (CSTR) running the formose reaction.
  - Trains linear readouts (scikit-learn classifiers / regressors) on the reservoir state to solve nonlinear classification, metabolic-model emulation, and chaotic time-series forecasting tasks.
  - Includes scripts that reproduce every data figure in the paper.
- **Relevance to this project:**
  - Demonstrates a complete experimental + computational pipeline for chemical reservoir computing, which is the closest experimental analogue to a “trainable fluid computer.”
  - The preprocessing and reservoir-training notebooks are a concrete benchmark for how to convert raw chemical sensor data into a trained readout.
  - Shows that a *fixed, unengineered* reaction network can be computationally useful, supporting the project’s broader claim that structure/geometry can encode computation.
- **Caveats:**
  - The chemistry is specific to the formose reaction; the code is not a generic CRN trainer.
  - The readout is linear, so the network itself is not trained; this is reservoir computing rather than full weight training.

### 1.3 Lakin et al. — ProBioSim (multilayer chemical neural network with backpropagation)

- **Repository / URL:** ProBioSim is described in *Artificial Life* 2023 (doi:10.1162/artl_a_00355). The implementation is based on the molecular-circuits DSL by Lakin & Phillips and is available from the authors’ group page / repository.
- **Language / stack:** Python with a domain-specific language for molecular circuits
- **What it does:**
  - Provides a Python library and DSL for specifying reactions, rate constants, and scheduled perturbations.
  - Supports both deterministic (ODE) and stochastic simulation of user-defined CRNs.
  - Was used to design and train a multilayer chemical neural network that learns via backpropagation.
- **Relevance to this project:**
  - Offers a CRN-first training environment, as opposed to training in a generic ML framework and translating to chemistry afterward.
  - The perturbation-action / perturbation-function mechanism is useful for encoding time-varying inputs and boundary conditions in batch simulations.
- **Caveats:**
  - Less widely adopted than general-purpose tools such as COPASI or Tellurium; community support and documentation may be limited.

---

## 2. General-purpose CRN and reaction–diffusion simulators

### 2.1 Catalyst.jl (Julia)

- **Repository:** `SciML/Catalyst.jl`
- **URL:** https://github.com/SciML/Catalyst.jl
- **Language / stack:** Julia; wraps ModelingToolkit.jl, Symbolics.jl, DifferentialEquations.jl, JumpProcesses.jl
- **What it does:**
  - Symbolic DSL for chemical reaction networks.
  - Automatically generates and compiles ODE (mass action), chemical Langevin SDE, and Gillespie jump-process representations.
  - Supports spatial reaction–diffusion master equations and spatial SSA methods via `JumpProcesses.jl` / `MOL` (method-of-lines) interfaces.
- **Relevance to this project:**
  - The most performant and mathematically clean option for simulating CRNs and RD systems.
  - Symbolic Jacobians, parameter sensitivities, and automatic differentiation make it ideal for gradient-based training of reaction rates or diffusion coefficients.
  - A future implementation could specify a CRN in Catalyst, convert to an ODEProblem or JumpProblem, and optimize parameters with DiffEqFlux / SciMLSensitivity.
- **Caveats:**
  - Requires Julia, which is not currently in the project stack (Python + NumPy/SciPy is the planned FDTD stack).
  - Steep learning curve if the user is not already familiar with Julia.

### 2.2 BioCRNPyler + Bioscrape / libRoadRunner (Python)

- **Repository:** `BuildACell/BioCRNPyler` + `biocircuits/bioscrape` + `sys-bio/roadrunner`
- **URLs:**
  - https://github.com/BuildACell/BioCRNPyler
  - https://github.com/biocircuits/bioscrape
  - https://github.com/sys-bio/roadrunner
- **Language / stack:** Python; SBML export/import
- **What it does:**
  - BioCRNPyler lets the user compose CRNs from parts and export to SBML.
  - Bioscrape is a fast stochastic and deterministic simulator for biological circuits.
  - libRoadRunner is a high-performance deterministic SBML simulator.
- **Relevance to this project:**
  - Keeps the workflow entirely in Python, matching the planned stack.
  - SBML export provides an interoperable intermediate representation.
  - Useful for verifying that a designed CRN behaves correctly under both deterministic and stochastic kinetics.
- **Caveats:**
  - BioCRNPyler is still pre-1.0; API stability should be checked.
  - Spatial RD is not the primary focus of these tools.

### 2.3 COEL — Cloud-based CRN simulator

- **Repository / URL:** https://github.com/coel-sim/coel (Zenodo doi:10.5281/zenodo.46544); web instance at https://coel-sim.org
- **Language / stack:** Web / cloud; open-source backend
- **What it does:**
  - Web-based framework for ODE simulation of CRNs and multicompartment reaction networks.
  - Includes plotting, automatic DNA-strand-displacement transformation, SBML/Octave/MATLAB export, and a genetic-algorithm toolbox for rate-constant optimization.
- **Relevance to this project:**
  - Useful for quick prototyping and teaching; the GA optimization is a lightweight alternative to gradient-based training.
  - Could sanity-check small CRN designs before committing to a custom Python solver.
- **Caveats:**
  - Cloud dependency or self-hosted server setup; less suitable for batch HPC sweeps.
  - Does not target spatial RD.

### 2.4 COPASI

- **URL:** https://copasi.org
- **Language / stack:** C++ with GUI and Python bindings (Basico)
- **What it does:**
  - Industrial-strength biochemical network simulator with ODE, SDE, and Gillespie solvers.
  - Parameter estimation, sensitivities, metabolic control analysis, and SBML import/export.
- **Relevance to this project:**
  - Gold-standard verification tool for CRN dynamics.
  - The Python `basico` package allows scripting within the project’s Python stack.
- **Caveats:**
  - Not a code repository in the same sense as the others; primarily a standalone application.
  - Spatial RD requires coupling with external PDE solvers.

### 2.5 StochPy / python-crn

- **Repository:** `enricozb/python-crn`
- **URL:** https://github.com/enricozb/python-crn
- **Language / stack:** Python 3.6+, NumPy, SciPy, SymPy, StochPy
- **What it does:**
  - A compact Python CRN simulator inspired by David Soloveichik’s Mathematica CRN simulator.
  - Provides a simple reaction-literal syntax and deterministic/stochastic simulation.
- **Relevance to this project:**
  - Could serve as a lightweight starting point for a custom Python CRN simulator.
  - Note: repository is no longer actively maintained.

---

## 3. DNA / chemistry compilers and sequence designers

### 3.1 Peppercorn

- **Repository:** `DNA-and-Natural-Algorithms-Group/peppercorn`
- **URL:** https://github.com/DNA-and-Natural-Algorithms-Group/peppercorn
- **Language / stack:** Python
- **What it does:**
  - Enumerates domain-level DNA strand-displacement (DSD) reaction networks.
  - Can condense fast/slow reactions to produce a compact CRN equivalent to the original DSD system.
- **Relevance to this project:**
  - The bridge from an abstract CRN to an experimental DNA implementation.
  - Dack et al. argue that their RNCRNs are implementable via DNA strand displacement; Peppercorn is the natural next step after designing the CRN.

### 3.2 NUPACK / Nuad

- **NUPACK URL:** https://www.nupack.org
- **Nuad repository:** https://github.com/UC-Davis/nuad
- **What it does:**
  - NUPACK: thermodynamic analysis and sequence design for nucleic acid systems.
  - Nuad: a Python library for writing custom DNA sequence designers when NUPACK’s built-in constraints are insufficient.
- **Relevance to this project:**
  - Needed only if the project eventually moves from abstract CRNs to an experimental DNA implementation.

### 3.3 Nuskell

- **Repository:** `DNA-and-Natural-Algorithms-Group/Nuskell`
- **URL:** https://github.com/DNA-and-Natural-Algorithms-Group/Nuskell
- **Language / stack:** Python
- **What it does:**
  - A compiler from abstract CRNs to domain-level DNA strand-displacement systems.
  - Uses Peppercorn for network enumeration.
- **Relevance to this project:**
  - Provides an automated path from a trained CRN to candidate DNA molecules.

---

## 4. Recommendations for the project

| Stage | Suggested tool(s) | Rationale |
|-------|-------------------|-----------|
| Rapid CRN prototyping | `python-crn`, BioCRNPyler + Bioscrape, or Catalyst.jl | Fast iteration; deterministic and stochastic verification |
| Training CRN weights | Dack RNCRN repo (TensorFlow → MATLAB) or Catalyst.jl + SciMLSensitivity | Gradient-based optimization of reaction-rate parameters |
| Reservoir-computing benchmark | `huckgroup/Formose_reservoir_computation` | End-to-end experimental-data pipeline for chemical RC |
| Spatial RD extension | Catalyst.jl (spatial SSA / MOL) or custom Python finite-difference solver | RD requires PDE or spatial stochastic methods not in the CRN-only tools |
| DNA implementation | Peppercorn + Nuskell + NUPACK/Nuad | Lower abstract CRN to domain-level and then sequence-level DNA |
| Gold-standard verification | COPASI (via `basico`) | Cross-check mass-action ODE behavior and parameter estimation |

For the immediate 8-week roadmap, the most practical path is:

1. Implement a small well-mixed CRN XOR gate in Python/SciPy, using the RNCRN construction or Kang–Kim smooth-activation CRN as the design target.
2. Verify the same CRN in COPASI or BioCRNPyler/Bioscrape.
3. Later, port to a 1D/2D reaction–diffusion finite-difference solver; Catalyst.jl is the strongest candidate if Julia is acceptable, otherwise a custom NumPy solver.
4. Use the Formose reservoir code as a benchmark for how to format, preprocess, and read out chemical time-series data.

---

## 5. References

- Dack et al., *Recurrent neural chemical reaction networks that approximate arbitrary dynamics*, Cell Systems 2026 / arXiv:2406.03456.
- Baltussen et al., *Chemical reservoir computation in a self-organizing reaction network*, Nature 631, 2024.
- Lakin et al., *Design and Simulation of a Multilayer Chemical Neural Network That Learns via Backpropagation*, Artificial Life 29(3), 2023.
- Catalyst.jl documentation: https://docs.sciml.ai/Catalyst/stable/
- BioCRNPyler: https://github.com/BuildACell/BioCRNPyler
- COEL: https://coel-sim.org and Zenodo doi:10.5281/zenodo.46544
- Peppercorn: https://github.com/DNA-and-Natural-Algorithms-Group/peppercorn
- Nuskell: https://github.com/DNA-and-Natural-Algorithms-Group/Nuskell
