# Wave Computing Pivot: Research Summary and Recommendation

*Date: 2026-05-08*  
*Context: Cranfield IRP 2025-26 — "Neural Network based on Porous Media"*

---

## 1. The Core Insight

**Hughes et al. (2019, *Science Advances*)** proved that wave physics is mathematically equivalent to a recurrent neural network (RNN). The wave equation, when discretised in space and time, takes the same form as an RNN update rule. This means:

> **Any physical system that supports wave propagation can, in principle, be trained to perform neural computation.**

The trainable parameters are the material properties at each spatial location (refractive index, density, bulk modulus, porosity, etc.).

---

## 2. Why This Fits Your Project Better Than Chemistry

| Criterion | Chemical/Fluidic Direction | Wave Computing Direction |
|-----------|---------------------------|--------------------------|
| **Supervisor guidance** | Chemistry is messy; "start with non-reactive solid" | ✅ Non-reactive solid is the substrate |
| **Core variable** | Flow rates, mixing, reactions | ✅ Pore geometry (trainable parameter) |
| **Readout mechanism** | Chemical sensors, fluorescence | ✅ Acoustic transducers, accelerometers |
| **Parallelism** | Limited by diffusion times | ✅ Wave superposition is naturally parallel |
| **Theoretical grounding** | Ad hoc reaction networks | ✅ Biot theory + RNN equivalence |
| **Fabrication** | Microfluidic lithography | ✅ Casting, sintering, 3D printing |
| **Scalability** | Channel-limited | ✅ Volumetric 3D computation |

**Prof. Guo's original question:** *"Can we train porous media to control patterns/materials?"*

**Wave computing answer:** Yes — by training the pore geometry to control acoustic/elastic wave patterns that encode and process information.

---

## 3. The Precedent Landscape

### 3.1 Wave = RNN (Theoretical Foundation)
- **Hughes et al. 2019** (*Sci. Adv.*): Wave physics ≡ RNN. Demonstrated vowel classification with inverse-designed acoustic medium.
- **Wright et al. 2022** (*Nature*): Deep physical neural networks trained with backpropagation through the physical system itself.

### 3.2 Physical Learning (Training Methods)
- **Stern et al. 2020** (*PNAS*): Mechanical networks learn by adjusting spring constants via equilibrium propagation.
- **Stern et al. 2021** (*Phys. Rev. X*): General theory of supervised learning in physical networks with local free-energy landscapes.
- **Altman et al. 2024** (*Phys. Rev. Appl.*): Experimental demonstration of coupled learning in elastic networks.

### 3.3 Acoustic Analog Computing (Applications)
- **Silva et al. 2014** (*Science*): Metamaterials perform mathematical operations (Fourier optics approach).
- **Zuo et al. 2018** (*Sci. Rep.*): Acoustic ODE solver using labyrinthine metasurfaces.
- **Zangeneh-Nejad et al. 2021** (*Nat. Rev. Mater.*): Comprehensive review of metamaterial analog computing.

### 3.4 Porous Media Physics (Your Unique Angle)
- **Biot 1956** (*JASA*): Theory of elastic wave propagation in fluid-saturated porous solids. Predicts **two compressional waves + one shear wave** with coupled fluid-solid dynamics.

---

## 4. The Novelty Gap (Why This is Research)

**Existing work uses:**
- 2D metasurfaces (deterministic channels)
- Layered diffractive elements (optical)
- Discrete spring-mass networks (mechanical)

**No existing work combines:**
1. ✅ Three-dimensional porous substrate
2. ✅ Acoustic/elastic wave propagation (not EM/optical)
3. ✅ Trainable pore geometry as the computational parameter
4. ✅ Coupled fluid-solid dynamics (Biot waves) as computational resource

The Zangeneh-Nejad 2021 review **explicitly identifies porous media as underexplored** for metamaterial computing.

---

## 5. Proposed Project Structure (Wave Direction)

### Phase 1: Physics Modelling (Weeks 1–6)
- Implement Biot's poroelastic wave equations in COMSOL/ANSYS or Python
- Simulate wave propagation through idealised porous geometries
- Characterise fast wave, slow wave, and shear wave contributions

### Phase 2: Computational Primitives (Weeks 7–12)
- Design porous "neurons": localised regions where pore geometry creates desired phase/amplitude modulation
- Demonstrate basic operations: spatial filtering, interference, frequency selection
- Build a single-layer porous wave processor

### Phase 3: Trainable Architecture (Weeks 13–18)
- Implement digital surrogate model (Hughes 2019 style) for gradient computation
- Optimise pore geometry for a simple task (e.g., frequency classification, spatial pattern recognition)
- Compare against acoustic metasurface baselines

### Phase 4: Experimental Validation (Weeks 19–24, if feasible)
- Fabricate optimised porous scaffold (3D printed or sintered)
- Measure acoustic transmission with speaker/microphone array
- Validate training predictions against physical measurements

---

## 6. Key Design Decisions Needed

| Decision | Options | Recommendation |
|----------|---------|----------------|
| **Wave type** | Acoustic (fluid-borne) / Elastic (solid-borne) / Coupled (Biot) | Start with **air-saturated porous solid** (simpler boundary conditions) |
| **Frequency range** | Audible / Ultrasonic / Hypersonic | **Ultrasonic** (10 kHz – 1 MHz) balances wavelength with attenuation |
| **Pore scale** | Microporous (<2 nm) / Mesoporous (2–50 nm) / Macroporous (>50 nm) | **Macroporous** (100 μm – 1 mm) for ultrasonic wavelengths |
| **Fabrication** | 3D printed lattice / Sintered beads / Reticulated foam | **3D printed lattice** (designable geometry) or **sintered glass beads** (tunable porosity) |
| **Training method** | Digital surrogate / In situ backpropagation / Evolutionary | Start with **digital surrogate** (COMSOL + PyTorch/TensorFlow) |
| **Output readout** | Microphone array / Laser vibrometry / Embedded sensors | **Microphone array** at output face (simplest) |

---

## 7. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| High attenuation in porous media | Medium | Use low-frequency ultrasonic; model viscous losses upfront |
| Slow wave too attenuated to be useful | Low | Fast wave carries signal; slow wave provides nonlinearity/memory |
| Fabrication resolution insufficient | Medium | Start with mm-scale pores; use commercial 3D printing |
| Training does not converge | Low | Digital surrogate eliminates physical instability; start simple |
| Supervisor prefers fluid direction | Medium | Present both; wave direction better aligns with "non-reactive solid" guidance |

---

## 8. Immediate Next Steps

1. **Discuss with supervisors** — Present this summary; gauge appetite for wave-computing pivot
2. **Download and read Hughes 2019** (now in `Notes/papers/hughes2019.pdf`) — Understand the RNN-wave mapping in detail
3. **Run a simple COMSOL/Py simulation** — 2D acoustic wave through a circular pore to verify scattering behaviour
4. **Review Biot theory** — Understand fast/slow wave conditions; identify frequency-porosity regime of interest
5. **Update thesis introduction** — Rewrite to frame project as "trainable physical neural network in porous media" with acoustic/elastic waves as the computational modality

---

## 9. Files Updated

- `Notes/literature-review.md` — Added Section 7: Wave-Based Analog Computing (8 new paper summaries)
- `Report/references.bib` — Added 10 new BibTeX entries (Hughes, Wright, Stern×2, Zuo, Zangeneh-Nejad, Silva, Lin, Biot)
- `Notes/papers/hughes2019.pdf` — Downloaded and verified
- `Notes/papers/altman2023_coupled_learning.pdf` — Downloaded and verified (coupled learning in elastic networks)

---

## 10. Recommended Reading Order

1. **Hughes et al. 2019** — The theoretical foundation (most important)
2. **Zangeneh-Nejad et al. 2021** — The field overview (context)
3. **Stern et al. 2021** — Physical learning theory (training approach)
4. **Zuo et al. 2018** — Acoustic computing precedent (closest application)
5. **Wright et al. 2022** — Physical training demonstration (ambitious target)
