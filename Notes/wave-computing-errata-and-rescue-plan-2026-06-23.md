# Wave-Computing Errata and Rescue Plan

**Date:** 2026-06-23  
**Status:** The acoustic porous wave-computing direction has fundamental physics and implementation issues. A pivot is strongly recommended. This note documents the problems and proposes a realistic rescue path given the July simulation / August report deadline.

---

## Part 1: Errata in the wave-computing work

### 1.1 Device geometry does not match the wavelength

At the chosen operating frequency:

```
f = 2000 Hz
c0 = 343 m/s
λ = c0 / f = 0.1715 m = 17.15 cm
```

Simulation domain (`Analysis/fdtd_zk_mxv_v2.py`):

| Feature | Physical size | Size in wavelengths |
|---|---:|---:|
| Full domain | 0.50 m | 2.9 λ |
| Design region width | 0.20 m | 1.2 λ |
| Each 3x3 design block in x | 0.067 m | 0.39 λ |
| Source-to-probe distance | 0.34 m | 2.0 λ |

**Issue:** A useful diffractive or scattering computer usually needs many wavelengths to accumulate phase shifts and form distinct interference patterns. A device only 2–3 wavelengths long, with blocks nearly half a wavelength wide, is operating in a weak-scattering near-field regime. It cannot behave like a compact trainable diffractive network.

### 1.2 Sound-speed discontinuity at the air-porous interface

In `Analysis/fdtd_zk_2d_v2.py`, lines 66–72:

```python
self.rho_eff = self.ks * self.rho0 / self.phi
self.K_eff = np.where(
    (self.phi == 1.0) & (self.ks == 1.0),
    self.rho0 * self.c0**2,
    self.p0 / self.phi
)
```

In the porous region this gives:

```
c_eff = sqrt(K_eff / rho_eff) = sqrt(p0 / (ks * rho0)) = sqrt(101325 / (1 * 1.225)) ≈ 288 m/s
```

In air the code gives 343 m/s. The effective sound speed therefore drops discontinuously at every air-porous interface, producing unphysical reflections.

For adiabatic pore air the correct bulk modulus would be `K_eff = gamma * p0 / phi`, giving `c_eff ≈ 340 / sqrt(ks) m/s`. The current code uses `p0 / phi`, which corresponds to an isothermal assumption, and then mixes it with adiabatic air. This is a solver bug.

### 1.3 Presentation shows Helmholtz, code does FDTD

The presentation `Presentations/progress-2026-06-23.tex` now displays the frequency-domain JCA Helmholtz equation:

```
∇ · (1/ρ̃(ω,x) ∇p) + ω²/K̃(ω,x) p = S(ω) δ(x - x_s)
```

But the methodology pipeline and actual solver are time-domain Zwikker-Kosten FDTD. These are different formulations:

| Aspect | Slide | Code |
|---|---|---|
| Domain | Frequency domain | Time domain |
| Loss model | JCA, frequency-dependent | ZK, frequency-independent |
| ρ, K | Complex | Real |

This mismatch makes the methodology hard to defend.

### 1.4 ZK model is not valid at 2 kHz

For typical foam pores `Λ ~ 100 µm`:

```
Wo = Λ * sqrt(rho0 * omega / eta)
   ≈ 100e-6 * sqrt(1.225 * 12566 / 1.81e-5)
   ≈ 2.9
```

A Womersley number of order unity or larger means the viscous boundary layer is thin compared with the pore size. The ZK assumption (parabolic velocity profile, frequency-independent drag) is not satisfied. The physically correct model at 2 kHz is JCA, not ZK.

### 1.5 Design variables are ambiguous

The "Domain Description" slide lists both `c(x,y)` and `(φ, σ, α∞)` as design variables. In a ZK/JCA model, `c(x,y)` is not independent; it is derived from `φ` and `α∞`. The presentation should commit to one parameterisation.

### 1.6 Absorbing sponge is too thin

```python
SPONGE_WIDTH = 10
DX = 0.004
```

Sponge width = `10 × 0.004 = 0.04 m = 4 cm`, or about `0.23 λ`. A good absorbing boundary layer should be roughly one wavelength thick at the lowest frequency of interest. The current sponge will reflect energy back into the domain, corrupting the probe signals.

### 1.7 Energy and length scale are physically unattractive

Even if the above issues were fixed, the energy estimate is poor:

```
I = p_rms² / (rho0 c0) = 1² / (1.225 × 343) ≈ 2.4 mW/m²
E = I × A × t = 2.4 mW/m² × 10^-4 m² × 10^-3 s ≈ 0.24 nJ
```

A 0.24 nJ per-operation cost is 100 to 100,000 times worse than a digital MAC. The device is also half a metre across because the wavelength is 17 cm. These are fundamental drawbacks of audio-frequency acoustic computing in air.

---

## Part 2: What is still salvageable

The following components are reusable in another direction or as comparison material:

- Staggered-grid FDTD infrastructure in `fdtd_zk_2d_v2.py`.
- Design-grid-to-simulation-grid mapping (`set_material_from_design`).
- Differential-evolution optimisation loop.
- Complex-amplitude lock-in readout (`measure_complex_amplitudes`).
- The envelope-study comparison framework.

The wave-computing results themselves should be treated as a **negative feasibility result**, not as a working design.

---

## Part 3: Rescue plan

### Constraints

- Simulations must run in **July**.
- Report must be finished in **August**.
- Therefore the final model and task must be locked by **end of June**.

### Recommended direction

Pivot to a **chemical or ionic continuous-medium model** with a tractable forward solver. The supervisor explicitly wants concentrations, reactions, and ion-channel-like dynamics. These models also have better fundamental energy scaling than audio acoustic waves.

### Three candidate models ranked by speed of implementation

| Rank | Model | Forward solver | Time to first result | Novelty |
|---:|---|---|---|---|
| 1 | Trainable Chemical Reaction Networks (CRNs) | Mass-action ODEs in SciPy | Days | Medium |
| 2 | Reaction-diffusion computing (Gray-Scott / FitzHugh-Nagumo) | 1-D/2-D FDTD in NumPy | 1 week | Medium |
| 3 | Gated ion transport in a porous membrane | Nernst-Planck + Poisson + gating in 1-D | 1–2 weeks | High |

### Proposed choice: start with CRNs, add reaction-diffusion if time permits

**Why CRNs first:**
- The forward model is a system of ODEs. It can be written and debugged in one day.
- Nagipogu & Reif 2025 provides a direct anchor: ReLU neural networks can be translated into CRNs.
- It is exactly "concentrations + reactions", matching the supervisor's request.
- It gives a trainable benchmark quickly (XOR, 2x2 classifier, Iris subset).

**Why reaction-diffusion second:**
- Adds the continuous spatial medium that the supervisor may want.
- Demonstrates logic gates or pattern completion.
- Can be built by adding diffusion terms to the CRN ODEs.

**Why gated ion transport third:**
- Most novel and closest to ion channels / porous memristors.
- Only attempt if CRN and reaction-diffusion benchmarks are running ahead of schedule.

### Proposed first benchmark

A trainable CRN that implements a 2-input XOR or a 2x2 matrix-vector product. Inputs are initial concentrations of two species; outputs are steady-state concentrations of two other species. Train the rate constants with SciPy `differential_evolution` or `minimize` to minimise mean-squared error.

### Timeline

| Week | Task |
|---|---|
| 23–29 June | Lock direction, implement CRN forward solver, run first XOR benchmark. |
| 30 June – 6 July | Add a 2x2/3x3 classifier or Iris subset benchmark. Document results. |
| 7–13 July | Add reaction-diffusion logic gate or pattern-formation benchmark if feasible. |
| 14–20 July | Compare against acoustic baseline; finalise envelope-study table. |
| 21–27 July | Fix any remaining simulation issues; generate figures. |
| 28 July – 31 August | Write results, methodology, and updated introduction. |

### Report structure after pivot

1. **Introduction:** frame the project as continuous analog computation, compare acoustic and chemical substrates, justify the chemical/ionic focus.
2. **Literature Review:** physical neural networks, CRN computing, reaction-diffusion computing, ionic/porous neuromorphic devices.
3. **Methodology:** CRN mass-action formulation; optional reaction-diffusion extension; optimisation procedure.
4. **Results:** CRN benchmarks; comparison with the acoustic baseline from the envelope study.
5. **Conclusions:** contributions, limitations, future work toward experimental realisation.

### Immediate next actions

1. Create a minimal CRN forward solver (`Analysis/crn_forward_v1.py`).
2. Implement a trainable XOR benchmark (`Analysis/crn_xor_v1.py`).
3. Write a short note comparing CRN results to the acoustic baseline.
4. Update the supervisor with the pivot and the first CRN result within one week.

---

## Part 4: Open questions to resolve quickly

1. Does the supervisor want a spatial continuous medium, or is a well-mixed CRN acceptable?
2. Did he have a specific chemical system in mind (BZ, Formose, ion channels, OECTs)?
3. Should the acoustic work be removed entirely, or kept as a short envelope-study comparison?
4. Is an experimental component expected, or is simulation-only acceptable?

---

## Files referenced

- `Analysis/fdtd_zk_2d_v2.py` — ZK FDTD solver with sound-speed inconsistency.
- `Analysis/fdtd_zk_mxv_v2.py` — MxV benchmark parameters.
- `Presentations/progress-2026-06-23.tex` — current wave-computing deck.
- `Notes/89aa5df5.md` — CRN literature review.
- `Notes/fluid-nn-literature-summary.md` — comparison of fluid/chemical computing approaches.
- `Notes/porous-computing-lit-review.md` — porous memristor / ion-transport papers.
- `Notes/supervisor-meeting-pivot-2026-06-23.md` — supervisor feedback and candidate directions.
