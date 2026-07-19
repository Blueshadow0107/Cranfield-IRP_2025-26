# Master Project Timeline — Continuous Analog Wave Computation in Porous Media

*Cranfield MSc Individual Research Project 2025–26*  
*Compiled: 2026-06-21*  
*Sources: dated notes in `Notes/`, slide decks in `Presentations/`, and `Analysis/` file timestamps*

---

## 1. High-level phases

| Phase | Dates | What happened | Key output |
|-------|-------|---------------|------------|
| **1. Direction setting** | early May – 19 May 2026 | Pivoted from chemical reaction–diffusion to acoustic wave computing | `Notes/wave-computing-pivot.md`, `Presentations/wave-computing-pivot.tex` |
| **2. Roadmap & first FDTD** | 19 May – 31 May 2026 | Wrote simulation roadmap; built first Y-junction and diffractive-slice FDTD scripts | `Notes/0f839564.md`, `Analysis/y_junction_fdtd/`, `Analysis/diffractive_slice/` |
| **3. Architecture & nonlinearity review** | 31 May – 7 June 2026 | Compared three porous-medium architectures; mapped thermal/nonlinearity mechanisms | `Notes/project-state-2026-05-31.md`, `Presentations/thermal-nonlinearity-review-2026-05-31.tex` |
| **4. Flow-computer dead end** | 7 June – 9 June 2026 | Tried thermoviscous channel and cylinder wake computing; got null / too-simple results | `Notes/thermoviscous-null-findings-2026-06-07.md`, `Notes/cylinder-poc-results-2026-06-07.md`, `Presentations/progress-review-2026-06-09.tex` |
| **5. Acoustic MxV pipeline** | 13 June – 21 June 2026 | Built validated FDTD core and trained 2×2 / 3×3 acoustic matrix-vector benchmarks | `Analysis/fdtd_core.py`, `Analysis/fdtd_mxv_*.py`, `Notes/project-status-2026-06-21.md` |

---

## 2. Detailed chronology

### Early May 2026 — Project start and chemical direction

* Initial repo scaffolding (`f2d4f44`, 2026-05-01).
* Early notes (`Notes/89aa5df5.md`, `Notes/fluid-nn-literature-summary.md`, `Notes/porous-computing-lit-review.md`) still framed the work around chemical / reaction-diffusion computing.
* Supervisor feedback pointed toward a non-reactive solid substrate.

### 8 May 2026 — Wave-computing pivot

* Wrote `Notes/wave-computing-pivot.md`.
* Core argument: Hughes et al. (2019) shows wave physics ≡ RNN; porous geometry can act as trainable weights.
* Identified novelty gap: no existing work combines 3D porous substrate + acoustic/elastic waves + trainable pore geometry + Biot coupling.

### 19 May 2026 — Supervisor discussion deck

* `Presentations/wave-computing-pivot.tex` compiled.
* Added wave-computing section to `Notes/literature-review.md` and `Report/references.bib`.

### 22 May 2026 — Simulation roadmap and first FDTD codes

* Wrote canonical roadmap `Notes/0f839564.md`.
* Established five research questions (Y-junction logic, diffractive slice, optimisation, 3D porous, physical concept).
* First FDTD prototypes:
  * `Analysis/y_junction_fdtd/y_junction_fdtd.py` — 2-D Y-junction phase sweep.
  * `Analysis/diffractive_slice/diffractive_fdtd.py` and `diffractive_fdtd_jax.py` — obstacle-array diffraction.
  * `Analysis/diffractive_slice/train_jax.py` — first JAX optimisation of scatterer phases.

### 25 May 2026 — Y-junction theory presentation

* `Presentations/y-junction-theory-and-simulation.tex`.
* Documented constructive/destructive interference logic for AND / XOR mappings.

### 31 May 2026 — Architecture decision and nonlinearity review

* `Notes/architecture-decision-framework.md` compared Architectures A (fixed scaffold + thermal fluid), B (bubble cloud), C (hybrid).
* User locked Architecture A as core, C as stretch.
* `Notes/thermal-and-nonlinearity-summary.md` catalogued six nonlinear mechanisms (Hertzian contact, Forchheimer, Biot coupling, bubble resonance, thermal gradient, acoustoelastic prestress).
* `Notes/project-state-2026-05-31.md` summarised state: thermal reconfiguration, nonlinearity hierarchy, open questions.
* `Presentations/thermal-nonlinearity-review-2026-05-31.tex` presented the same material.

### 1 June 2026 — Obstacle-array exploration

* `Presentations/obstacle-array-exploration-2026-06-01.tex`.
* MATLAB Y-junction steps added in `Analysis/y_junction_fdtd/matlab/`.

### 7 June 2026 — Flow-computer experiments

* **Thermoviscous channel — null result**
  * `thermoviscous_channel.py` + `sweep_heat_patterns.py`: 12 heat patterns across 8 strips, Re = 10–100.
  * Frequency ratio readout invariant to machine precision → viscosity modulation in straight channel is not trainable.
  * Documented in `Notes/thermoviscous-null-findings-2026-06-07.md`.
* **Cylinder array — limited PoC**
  * `cylinder_array_solver.py`: discrete square cylinders in 2-D incompressible NS.
  * Binary separable outputs (ratio ~1,700× between no-cylinder and side-by-side cases).
  * Documented in `Notes/cylinder-poc-results-2026-06-07.md`.

### 8 June 2026 — Weekly recap

* `Notes/weekly-recap-2026-06-08.md` summarised the week:
  * Face-consistent collocated projection solver (`projection_solver.py`) became stable.
  * Brinkman vs discrete ψ–ω comparison.
  * Frequency/Reynolds sweeps showed flow is essentially linear at tested Re.
  * Porous grating invisible to flow.
  * Key lesson: single smooth obstacles / Brinkman media are too simple for analog NN computation.

### 9 June 2026 — Progress review presentation

* `Presentations/progress-review-2026-06-09.tex`.
* Presented the pivot away from flow and toward acoustic FDTD as the main numerical path.

### 13 June 2026 — Modular porous FDTD

* `Analysis/fdtd_modular.py` — refactored FDTD.
* `Analysis/fdtd_porous.py` — first porous-medium time-domain solver.
* `Analysis/optimise_porous_router.py` — random-search porous router.
* `Notes/fdtd-porous-physics-review.md` — theory review.

### 14 June 2026 — Supervisor meeting and validation ladder

* `Presentations/supervisor-meeting-2026-06-14.tex`.
* `Notes/problem-definition-analog-wave-computer.md` and `Notes/problem-definition-vowel-classifier.md`.
* `Notes/porous-solver-validation-ladder.md` defined five-step validation plan.
* `Analysis/porous_fdtd_validation_v1.py` validated free-air, slab slowdown, step reflection, damping, and energy conservation.
* `Notes/model-comparison-detailed.md` compared Biot, JCA, and equivalent-fluid models.

### 15 June 2026 — JCA Helmholtz exploration

* `Analysis/jca_helmholtz_2d_v1.py` and `Analysis/validate_jca_1d.py`.
* Investigated Johnson-Champoux-Allard frequency-domain attenuation and phase shifts.

### 16 June 2026 — Thermal blockwise optimisation

* `Analysis/thermal_blockwise_optimise_v1.py` — tried optimising a block-wise temperature / sound-speed pattern.

### 18 June 2026 — Similar-work literature search

* `Notes/literature-similar-work-2026-06-18.md`.
* Downloaded and read Kalthoff et al. 2025 (*Acoustic neural networks*) — closest experimental precedent.

### 21 June 2026 — Validated FDTD core and MxV benchmarks

* `Analysis/fdtd_core.py` validated (6/6 tests pass); replaced earlier monolithic scripts.
* `Analysis/fdtd_cylinder_array_scatter_v1.py` — acoustic scattering sweep over cylinder diameter/spacing.
* `Analysis/fdtd_mxv_v1.py` — first trainable 2×2 MxV with 2 hard-wall cylinders (loss 0.347).
* `Analysis/fdtd_mxv_v2.py` — 2×2 MxV with 4 cylinders + radii, differential evolution (loss 0.046).
* `Analysis/fdtd_mxv_porous_v1.py` — 2×2 MxV with continuous `c_field` control grid (loss **0.020**).
* `Analysis/fdtd_mxv_readout_v1.py` — 2×2 classifier with continuous porous layer + ReLU readout (loss 0.0005, **100% accuracy**).
* `Analysis/fdtd_mxv_readout_v3.py` — 3×3 classifier scaling test; **100% accuracy** with reduced generation budget (loss 0.025).
* Wrote `Notes/project-status-2026-06-21.md`.

---

## 3. Major pivots and decisions

| Date | Decision | Why it mattered |
|------|----------|-----------------|
| 8 May 2026 | Move from chemical computing to acoustic wave computing | Aligned with supervisor guidance and exploitability of wave-interference physics |
| 31 May 2026 | Architecture A (fixed scaffold + thermal fluid) as core | Faster/cheaper than bubble-cloud option; kept Architecture C as stretch |
| 7 June 2026 | Reject thermoviscous channel flow as weight mechanism | Null result: frequency ratio invariant to heat pattern |
| 8 June 2026 | Conclude flow-based reservoirs are too simple at tested Re | Single obstacles and Brinkman media give linear pass-through behaviour |
| 21 June 2026 | Adopt continuous porous `c_field` as weight encoding | Outperforms discrete cylinders; directly trainable via differential evolution |
| 21 June 2026 | Use digital ReLU readout for nonlinearity | Simplest physically honest choice; defers physical nonlinearity to future work |

---

## 4. Current state (21 June 2026)

* **Working substrate:** 2-D acoustic FDTD with spatially varying real sound speed.
* **Validated primitive:** 2×2 matrix-vector multiplication via structured porous medium.
* **Classifier:** 2-class, 100% accuracy with ReLU readout.
* **Scaling test:** 3×3 classifier running but timed out; needs shorter runtime or reduced design space.
* **Rejected path:** Navier-Stokes flow computing (linear response, no echo-state property).
* **OpenFOAM role:** cross-physical validation only (`Validation/cylinder_array_2d/`).

---

## 5. Open items and next steps

1. ~~**Complete 3×3 scaling test**~~ — done; `fdtd_mxv_readout_v3.py` achieves 100% accuracy with 5 generations.
2. **Report raw (unnormalised) transmission matrices** — energy-budget check for physical correctness.
3. **Rewrite thesis chapters**:
   * `Report/chapters/introduction.tex` — acoustic wave-computing aim and objectives.
   * `Report/chapters/methodology.tex` — FDTD, porous encoding, optimisation loop, readout.
   * `Report/chapters/results.tex` — cylinder sweep, MxV comparison, classifier results.
4. **Update bibliography** — add Kalthoff 2025 and related acoustic-NN references.
5. **Physical concept section** — dimensions, frequencies, materials for a realistic demonstrator.

---

*Tags: #timeline #project-history #acoustic-neural-network #fdtd #porous-medium #mxv #thesis*
