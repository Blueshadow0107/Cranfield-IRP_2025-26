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
