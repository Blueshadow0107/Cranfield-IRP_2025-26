# Problem Definition: Continuous Analog Wave Computation in Porous Media

> **DEPRECATED — exploratory scratchpad.** This file preserves the historical discussion and open questions from the initial scoping phase. The authoritative, clean specification is now in `problem-definition-vowel-classifier.md`.

**Date:** 2026-06-13  
**Status:** Draft — open decisions marked below  
**Goal:** Define the full scientific and engineering problem before committing to a solver or implementation.

---

## 1. High-level aim

Build a trainable physical system in which acoustic waves propagate through a structured porous medium and perform a classification task. The medium itself is the neural network; its geometry or material properties encode the weights.

This is conceptually similar to Hughes et al. (2019), who used a 3D-printed acoustic cavity with spatially varying wave speed to classify vowels.

---

## 2. Physical substrate

**Chosen substrate:** rigid skeleton with air-filled pores.

- The solid frame does not move.
- Sound propagates only in the fluid inside the pores.
- The relevant material properties are porosity, tortuosity, and flow resistivity.

**Why this substrate:**
- It avoids the complexity of Biot theory (moving frame).
- It maps naturally onto additive manufacturing: a solid part with a designed void network.
- It gives a continuous spatial variation of acoustic properties, which is attractive for analog computation.

**Open question:** is the medium a bulk porous foam, a perforated plate, an array of small channels, or something else? This affects how porosity is interpreted and manufactured.

---

## 3. Computation task and intended contribution

**Chosen task:** vowel classification (Hughes-style).

- Input: an acoustic waveform containing a vowel sound.
- Medium: a spatial map of acoustic properties.
- Output: time-integrated or steady-state pressure readings at several probe locations.
- Goal: the output vector should distinguish vowel classes.

**Why vowels:**
- Vowels are defined mainly by their spectral envelope (formants).
- The task is well-known from Hughes et al., providing a benchmark.
- It is nontrivial but not as complex as general speech recognition.

**Intended contribution:** somewhere between reproducing Hughes in a new substrate and exploiting the extra physics of porous media.

- The core demonstration is a porous-medium analog wave computer for vowel classification (reproduces Hughes in a different physical substrate).
- The added contribution is that porous media provide coupled control of sound speed, impedance, and damping through porosity and flow resistivity. We will show that the extra damping knob improves the design or enables something that a pure sound-speed map cannot do.
- A likely comparison: phi-only designs versus phi-plus-sigma designs.

**Open question:** do we need true time-domain dynamics, or is a frequency-domain spectral classifier sufficient? Vowels can be classified from spectrum alone, but a time-domain solver is closer to the Hughes formulation.

---

## 4. Operating frequency regime

### Option A: speech band (300 Hz – 4 kHz)

This is where vowel information actually lives.

- Free-air wavelength at 1 kHz: ~343 mm.
- Free-air wavelength at 4 kHz: ~86 mm.
- Viscous skin depth at 1 kHz: ~70 µm.
- Viscous skin depth at 4 kHz: ~35 µm.

**Pros:** physically meaningful for vowel classification; larger viscous skin depth makes the simple ZK model more valid for a given pore size.

**Cons:** wavelengths are large, so a 2D device would be tens of centimetres long unless the wave is slowed significantly.

### Option B: higher carrier frequency (e.g. 15–20 kHz)

Use a high-frequency carrier and modulate the speech envelope onto it.

- Free-air wavelength at 20 kHz: ~17 mm.
- Viscous skin depth at 20 kHz: ~15 µm.

**Pros:** compact device; short wavelengths allow rich interference patterns in a small domain.

**Cons:** not a direct vowel classifier — the vowel information must be encoded in the modulation; the porous model needs full ZK/JCA because pore size is likely larger than skin depth.

### Current leaning

Speech band is more natural for a vowel classifier, but device size is a concern. A folded 3D geometry (like Hughes) could solve the size issue, but the current simulation is 2D.

**Open decision:** speech band versus carrier frequency. This is the single biggest fork in the project.

---

## 5. Design variables

### Primary candidate: porosity phi(x,y)

- Controls effective sound speed: `c_eff = c0 / sqrt(phi * alpha_inf)`.
- Controls characteristic impedance.
- Easy to visualise as a spatial map.
- Natural for additive manufacturing: more or less open volume.

### Secondary candidates

- **Tortuosity alpha_inf(x,y):** harder to control independently; probably fixed initially.
- **Flow resistivity sigma(x,y):** adds damping; can be controlled by pore size or surface roughness; useful for richer dynamics but not essential for first demonstration.

### Current leaning

Start with `phi(x,y)` only, `alpha_inf = 1`, `sigma = 0`. This is the minimal design space and gives a pure wave-speed map analogous to Hughes.

**Open decision:** include damping/resistivity from the start, or add it later?

---

## 6. Pore size and manufacturability

Pore size determines both physical validity and fabrication feasibility.

| Pore size | Physics | Fabrication | Model validity |
|---|---|---|---|
| ~10–50 µm | Strong viscous damping; ZK low-freq limit valid | Hard (fine 3D printing, sintering) | Good for simplified time-domain ZK |
| ~100 µm | Moderate damping; borderline low-freq limit | Achievable (high-res SLA/DLP, laser sintering) | Simplified ZK approximate; full ZK/JCA better |
| ~500 µm | Weak damping; strong frequency dependence | Easy (standard 3D printing, perforated plates) | Full ZK/JCA required |
| ~1 mm | Geometric scattering regime | Very easy (drilled/milled holes) | Equivalent-fluid model may fail |

**Key constraint:** for any equivalent-fluid model, pore size must be much smaller than the acoustic wavelength.

- At 4 kHz: wavelength ~86 mm, so pores up to ~1 mm are acceptable.
- At 20 kHz: wavelength ~17 mm, so pores should be well below ~1 mm.

**Open decision:** what pore size range are we targeting? This depends on the chosen fabrication method.

---

## 7. Candidate governing equations

### 7.1 Simplified time-domain ZK

Equations in pore-velocity form:

    rho0 * alpha_inf * du/dt + sigma * phi * u = -grad(p)
    dp/dt + (rho0 * c0^2 / phi) * div(u) = 0

Or in volume-velocity form (correct at interfaces):

    (rho0 * alpha_inf / phi) * dV/dt + sigma * V = -grad(p)
    dp/dt + rho0 * c0^2 * div(V) = 0

**Pros:** intuitive; fast FDTD; captures time dynamics.

**Cons:** strictly valid only in the low-frequency limit (pore size << viscous skin depth).

### 7.2 Full ZK Helmholtz (frequency-domain)

    div( (1 / rho_eff(omega)) * grad(p) ) + (omega^2 / K_eff(omega)) * p = 0

with complex, frequency-dependent `rho_eff(omega)` and `K_eff(omega)` derived from ZK.

**Pros:** physically accurate for any pore size << wavelength; fast frequency-by-frequency solution; easy optimisation.

**Cons:** no direct time dynamics; requires solving at many frequencies for broadband inputs.

### 7.3 Johnson-Champoux-Allard (JCA) Helmholtz

Same structure as ZK Helmholtz, but `rho_eff` and `K_eff` use empirical/semi-empirical functions with viscous and thermal characteristic lengths.

**Pros:** more accurate for real porous materials.

**Cons:** needs more material parameters; still frequency-domain.

### 7.4 Time-domain ZK with relaxation

Approximate the frequency-dependent `rho_eff(omega)` and `K_eff(omega)` by relaxation kernels in time.

**Pros:** captures time dynamics with full ZK physics.

**Cons:** significantly more complex; adds memory variables per cell; harder to implement and optimise.

### 7.5 Direct pore-resolving simulation

Solve linearized Navier-Stokes in the actual pore geometry.

**Pros:** exact; no homogenisation.

**Cons:** computationally expensive; infeasible for design optimisation; useful only for validation.

### Current leaning

For a first vowel-classification demonstration, **ZK Helmholtz** is probably the best balance of physical accuracy and tractability. Time-domain simplified ZK is acceptable only if we commit to small pores and treat it as a design proxy.

**Open decision:** which model is the main solver? Is another model used for validation?

---

## 8. Input and readout

### Input options

- Broadband pulse (contains all frequencies).
- Swept sine (covers the frequency band sequentially).
- Single vowel waveform.
- Pure tone at a specific frequency.

### Readout options

- Time-integrated pressure at fixed probe points.
- Steady-state amplitude at fixed probe points per frequency.
- Spatial pressure map at a fixed time.
- Energy in designated output regions.

### Current leaning

For Helmholtz: solve at many frequencies, read steady-state probe amplitudes, build a spectral feature vector.

For time-domain FDTD: inject broadband pulse, time-integrate probe traces.

**Open decision:** exact input/readout strategy depends on the chosen model.

---

## 9. Key trade-offs summary

| Decision | Options | Implications |
|---|---|---|
| Frequency band | Speech band vs carrier | Speech band is natural but large; carrier is compact but indirect |
| Design variable | phi only vs phi+sigma | phi only is simpler; sigma adds damping and complexity |
| Pore size | 10–100 µm vs 100–500 µm | Smaller validates simple ZK; larger needs full ZK/JCA |
| Model | Time-domain ZK vs Helmholtz vs relaxation | Time-domain for dynamics; Helmholtz for accuracy and manufacturable pores |
| Dimensionality | 2D simulation vs 3D simulation/device | 2D is easier to simulate; 3D is more realistic and allows folded geometries for compactness |

---

## 10. Open decisions to resolve

1. **Frequency regime:** speech band (300 Hz – 4 kHz) or carrier-based (15–20 kHz)?
2. **Pore size / fabrication target:** what is realistic to manufacture?
3. **Governing equations:** simplified time-domain ZK, full ZK Helmholtz, JCA Helmholtz, or time-domain relaxation?
4. **Design variable:** porosity only, or include damping/tortuosity?
5. **Input/readout:** pulse + time integration, or multi-frequency Helmholtz + spectral vector?
6. **Dimensionality:** stay in 2D for simulation, or plan for a 3D-printed prototype?
7. **Readout:** time-integrated pressure, frequency spectrum, or spatial map?
8. **Objective function:** cross-entropy, MSE, contrast ratio, or classification accuracy?
9. **Readout:** raw pressure, squared pressure, RMS, or spectrum?
10. **Classifier after probes:** fixed or trainable linear readout?
11. **Optimisation method:** gradient-free or gradient-based?

## 12. Current working decision

As of this discussion, the working plan is:

- **Substrate:** rigid porous skeleton, air-filled pores (rigid-frame equivalent-fluid model; frame motion neglected).
- **Task:** Hughes-style vowel classification.
- **Frequency band:** speech band (300 Hz – 4 kHz).
- **Design variables:** porosity `phi(x,y)` as primary, flow resistivity `sigma(x,y)` as secondary. Porosity changes sound speed and impedance; flow resistivity adds local damping.
- **Porosity bounds:** realistic range `0.3 ≤ phi ≤ 1.0`. phi = 1 is free air; phi below ~0.3 is hard to manufacture and pushes the equivalent-fluid assumption.
- **Flow resistivity bounds:** `0 ≤ sigma ≤ 10^5 Pa·s/m²` as a realistic starting range for acoustic porous materials.
- **Tortuosity:** alpha_inf kept fixed at 1.0 for the simplified time-domain model. A frequency-dependent dynamic tortuosity would require the full ZK/JCA model.
- **Pore size target:** small enough that the simplified constant-sigma ZK model remains valid. Target pore radius ~10–20 µm (diameter ~20–40 µm), so the pore size is smaller than or comparable to the viscous skin depth at 4 kHz (~35 µm). Larger pores would require validation with the full ZK Helmholtz model.
- **Open question for supervisors:** should `phi` and `sigma` be treated as independent design variables, or linked through a pore-radius model such as `sigma ≈ 8 η / (R² φ)`?
- **Domain layout:** rectangular domain with first-order absorbing boundaries on all four sides. The middle third in x is the trainable design region, spanning the full height in y (0 to Ly), and split into a 4×4 block grid. Source sits just to the left of the trainable region; probes sit inside or just to the right of it.
- **Input injection:** pressure source injected over a 3×3 cell patch just to the left of the trainable region.
- **Readout:** time-integrated squared pressure at three fixed probe locations near the right edge of the trainable region, placed at y = Ly/4, Ly/2, and 3Ly/4 (i.e. one quarter-height, mid-height, and three-quarter-height).
- **Absorbing boundary padding:** none; first-order absorbing BC applied directly at the outer domain boundary.
- **Classifier:** trainable linear readout mapping the three probe energies to vowel-class logits.
- **Loss:** cross-entropy over vowel classes.
- **Input normalisation:** each vowel sample normalised to the same amplitude before injection.
- **Validation:** 5-fold cross-validation (see below).
- **Model:** time-domain FDTD solving the **rigid-frame Zwikker-Kosten equivalent-fluid equations** in volume-velocity form. Also called the first-order acoustic wave equations with ZK effective properties.
- **Pore-size validity:** simplified constant-sigma ZK is valid when pore size is comparable to or smaller than the viscous skin depth. At 4 kHz this is ~35 µm, so the target pore radius is ~10–20 µm. If final designs use larger pores, they must be validated with full ZK Helmholtz or direct pore simulation.
- **Dimensionality:** **2D for now**, with a clear path to extend to 3D for a realisable device concept.
- **Time step:** fixed `dt` chosen for the fastest possible effective wave speed in the design space (simpler and stable for all designs).
- **Integration window:** fixed time window for probe energy; exact length depends on wave travel time through the medium and will be chosen after a baseline transit test.
- **Domain physical size:** to be chosen based on desired number of wavelengths across the trainable region; a starting point is ~0.5–1.0 m in x and ~0.1–0.2 m in y.
- **Boundary conditions:** first-order absorbing boundary conditions on all four outer sides, surrounding the central design region. To be reviewed with supervisors before final implementation.
- **Grid resolution:** at least 30 grid points per minimum wavelength.
- **Contribution:** reproduce Hughes-style vowel classification in a porous medium, then demonstrate that adding damping through `sigma` provides extra design freedom beyond a pure sound-speed map.

### 5-fold cross-validation

The dataset is split into five groups of roughly equal size. Five independent training runs are performed:

1. Train on groups 1–4, test on group 5.
2. Train on groups 1, 2, 3, 5, test on group 4.
3. Train on groups 1, 2, 4, 5, test on group 3.
4. Train on groups 1, 3, 4, 5, test on group 2.
5. Train on groups 2–5, test on group 1.

The final reported accuracy is the average over the five test folds. This gives a more reliable estimate than a single train/test split, especially with only ~280 samples.

---

## 11. Suggested path forward

Before writing any more solver code, resolve decisions 1 and 3. They determine everything else.

A defensible starting position:
- **Speech band** for vowel classification.
- **Porosity phi(x,y)** as the primary design variable, with **flow resistivity sigma(x,y)** added as a secondary variable to demonstrate the extra design freedom of porous media.
- **Time-domain FDTD** to stay close to the Hughes formulation, using the corrected volume-velocity ZK formulation for stability at porosity jumps.
- Accept that this is the simplified ZK model and therefore most valid for smaller pores; use it as a design proxy, with the option to validate final designs using ZK Helmholtz or direct pore simulation.
- **2D or 3D simulation** for proof of concept. 3D is more realistic and allows folded geometries that shrink device size, but it is computationally more expensive.

This gives a thesis narrative: reproduce the Hughes vowel classifier in a porous medium, then show that adding damping through flow resistivity improves classification performance or enables designs that pure sound-speed maps cannot achieve.
