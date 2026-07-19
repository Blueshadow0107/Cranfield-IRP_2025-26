# Project Status: Continuous Analog Wave Computation in Porous Media

Date: 2026-06-21
Scope: Cranfield MSc Individual Research Project 2025–26

---

## 1. One-sentence summary

The project is building a simulation pipeline that treats structured acoustic media as trainable physical neural-network layers, where geometry or a continuous sound-speed map encodes the weights and a nonlinear readout supplies the activation function.

---

## 2. Architectural decisions (locked)

| Decision | Chosen option | Rationale |
|---|---|---|
| Physical substrate | Static acoustic scatterers / porous sound-speed map | Faster, cheaper, and more controllable than fluid-flow reservoirs |
| Numerical method | 2-D FDTD (Python/NumPy) | Validated, fast enough for optimisation loops |
| Validation | OpenFOAM cylinder-flow case | Cross-checks the physical modelling; confirms flow is not the right substrate |
| Weight encoding | Continuous porous-medium `c_field(x,y)` map | Outperforms discrete hard-wall cylinders |
| Nonlinearity | Digital ReLU readout on probe intensities | Simplest physically honest choice; physical nonlinearities left as future work |
| Optimisation | Differential evolution on a coarse control grid | Gradient-free, works with the black-box FDTD solver |
| Scaling | Add input/output ports, refine control grid | Parallelism is in the physical inference, not the training |

---

## 3. Code inventory and results

### Core solver

| File | Status | Purpose |
|---|---|---|
| `Analysis/fdtd_core.py` | Validated (6/6 tests pass) | 2-D staggered leapfrog acoustic FDTD with thermal solver |

### Scattering / parameter sweeps

| File | Status | Key result |
|---|---|---|
| `Analysis/fdtd_cylinder_array_scatter_v1.py` | Working | Transmission loss tunable from ~0 dB to ~−25 dB by changing cylinder diameter and spacing |

### MxV benchmarks

| File | Size | Encoding | Optimiser | Best loss | Notes |
|---|---|---|---|---|---|
| `Analysis/fdtd_mxv_v1.py` | 2×2 | 2 hard-wall cylinders | Random search (40 trials) | 0.347 | First proof of concept; cross-router rather than diagonal |
| `Analysis/fdtd_mxv_v2.py` | 2×2 | 4 hard-wall cylinders + radii | Differential evolution | 0.046 | Much better; diagonal routing emerges |
| `Analysis/fdtd_mxv_porous_v1.py` | 2×2 | Continuous `c_field` (4×3 control grid) | Differential evolution | **0.020** | Best result so far; smooth porous map outperforms cylinders |
| `Analysis/fdtd_mxv_readout_v1.py` | 2×2 | Continuous `c_field` + ReLU readout | Differential evolution | **0.0005** | **100% classification accuracy** on 2-class task |
| `Analysis/fdtd_mxv_readout_v3.py` | 3×3 | Continuous `c_field` + ReLU readout | Differential evolution (5 generations) | **0.025** | **100% classification accuracy** on 3-class task; diagonal routing preserved |
| `Analysis/fdtd_mxv_readout_v2.py` | 3×3 | Continuous `c_field` + ReLU readout | Differential evolution (10 generations) | Timed out | Original run killed; superseded by v3 |

### OpenFOAM validation

| File / directory | Status | Purpose |
|---|---|---|
| `Validation/cylinder_array_2d/` | Working | `pimpleFoam` laminar cylinder flow at Re=100; Strouhal number in range 0.13–0.16 |

### Literature

| File | Status |
|---|---|
| `Notes/literature-similar-work-2026-06-18.md` | Complete literature search summary |
| `Notes/papers/kalthoff2025_acoustic_neural_networks.pdf` | Key reference downloaded |

---

## 4. Key findings

1. **Passive acoustic media can implement trainable linear transformations.**
   - A continuous sound-speed map encodes an effective weight matrix `W`.
   - The matrix is measured by firing each source alone and recording steady-state probe intensities.

2. **Continuous porous maps outperform discrete cylinders.**
   - Cylinder v2 loss: 0.046
   - Porous v1 loss: 0.020
   - Reason: smooth gradients guide waves more controllably than sharp scattering.

3. **A ReLU readout turns the device into a neural-network layer.**
   - Physical layer: linear MxV.
   - Readout: `y = max(W @ x − threshold, 0)`.
   - Result: 100% accuracy on a 2-class classification task.

4. **Physical constraints are real.**
   - All weights are non-negative because the medium is passive.
   - The raw (unnormalised) transmission matrix columns sum to ≤ 1 (energy conservation).
   - Normalisation during training hides the energy budget; the raw matrix must be reported for physical correctness.

5. **Flow-based computation was rejected.**
   - OpenFOAM cylinder flow shows laminar vortex shedding.
   - The Kármán vortex street loses the echo-state property needed for reservoir computing.
   - Acoustic scattering is the better substrate for this project.

6. **Scaling is conceptually straightforward but training-cost limited.**
   - Inference is parallel: all sources can fire simultaneously.
   - Training is sequential: each source fires alone to calibrate one column of `W`.
   - Larger matrices need finer control grids and more optimisation iterations.

---

## 5. Physical constraints in detail

### Linear wave equation

The FDTD solver integrates:

```
∂p/∂t = −rho0 · c(x,y)² · (∂u/∂x + ∂v/∂y)
∂u/∂t = −(1/rho0) · ∂p/∂x
∂v/∂t = −(1/rho0) · ∂p/∂y
```

`c(x,y)` is the design variable in the porous-medium version.

### Passive weight constraint

For a source amplitude `A_j` and measured probe intensity `I_ij`:

```
W_raw[i,j] = I_ij / max_k(I_kj)
```

The physically correct check is:

```
sum_i W_raw[i,j] ≤ 1   for each input j
```

because the medium cannot create energy.

### Readout nonlinearity

The ReLU readout is applied after the physical propagation:

```
y_linear = W @ x
y_neural = max(y_linear − threshold, 0)
```

The nonlinearity is in the detection, not in the wave propagation.

---

## 6. Scaling approach

| Component | How it scales | Bottleneck |
|---|---|---|
| Physical inference | Add ports; all fire simultaneously | None — wave physics is parallel |
| Control grid | Finer grid for more complex routing | Number of optimisation parameters |
| Training | Fire each source alone to measure one matrix column | Number of ports `N` |
| Optimisation | Population size × parameters × iterations | Compute time |

Current demonstrated sizes:

- 2×2: perfect classifier (loss 0.0005).
- 3×3: perfect classifier (loss 0.025), diagonal routing preserved.

Stretch goal:

- 4×4 identity router or averaging filter.

---

## 7. Future extensions (not core)

These are documented but not required for the MSc core:

1. **Phase-encoded inputs + coherent/differential readout**
   - Would enable signed weights and full linear algebra.
   - Requires measuring complex amplitude, not just intensity.
   - Could implement XOR and edge-detection kernels.

2. **Physical nonlinearity in the medium**
   - Options: Forchheimer drag, nonlinear membranes, bubbles, granular media.
   - Harder than readout nonlinearity; left for future work.

3. **Full JCA time-domain model**
   - Captures frequency-dependent viscous/thermal relaxation.
   - Requires memory kernels or auxiliary differential equations.
   - Overkill for the current demonstrator.

4. **3-D porous extension**
   - Extrude the 2-D design into a slab.
   - Demonstrates true spatial parallelism.
   - Computationally expensive.

---

## 8. Thesis chapter mapping

| Thesis section | What goes there | Status |
|---|---|---|
| Introduction | Motivation, aim, objectives, scope | Needs rewrite from chemical direction |
| Literature Review | Wave-based computing, acoustic metamaterials, physical NNs, inverse design | Draft exists; add Kalthoff 2025 |
| Methodology | FDTD theory, porous-medium encoding, optimisation loop, readout layer | Skeleton; needs writing |
| Results | Cylinder sweep, MxV comparison, porous MxV, classifier, scaling | Figures ready; needs prose |
| Conclusions | Contributions, limitations, future work | Skeleton |

---

## 9. Figures ready for the report

| Figure | File | Caption idea |
|---|---|---|
| Cylinder transmission sweep | `figures/cylinder_array_sweep_v1.png` | Transmission loss vs. diameter and spacing |
| Pressure field snapshot | `figures/cylinder_array_snapshot_v1.png` | Wave scattering from cylinder array |
| Cylinder MxV matrix | `figures/mxv_matrix_comparison_v2.png` | Target vs. measured 2×2 matrix |
| Porous MxV matrix | `figures/mxv_porous_matrix_v1.png` | Target vs. measured 2×2 porous matrix |
| Porous field map | `figures/mxv_porous_field_v1.png` | Optimised sound-speed map and pressure fields |
| Classifier matrix | `figures/mxv_readout_matrix_2x2_v1.png` | Physical matrix for 2-class classifier |
| Classifier fields | `figures/mxv_readout_field_2x2_v1.png` | Sound-speed map and fields for classifier |
| 3×3 classifier matrix | `figures/mxv_readout_matrix_3x3_v2.png` | Physical matrix for 3-class classifier (pending) |

---

## 10. Next documentation tasks

1. Rewrite `Report/chapters/introduction.tex` with acoustic wave-computing aim and objectives.
2. Update `Report/chapters/methodology.tex` with FDTD, porous encoding, optimisation, and readout.
3. Update `Report/chapters/results.tex` with the figures and comparison tables.
4. Add Kalthoff 2025 and related references to `Report/references.bib`.
5. Create a note on phase-encoding extension if time permits.

---

Tags: #project-status #acoustic-neural-network #mxv #porous-medium #fdtd #thesis
