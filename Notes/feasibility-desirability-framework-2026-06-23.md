# Feasibility / Desirability Framework for Pivot Decision

**Date:** 2026-06-23  
**Purpose:** Justify whether to continue with acoustic porous wave computing or pivot to a chemical / ionic continuous-computing direction. This framework compares candidates on first-principles physical metrics, implementation feasibility, and research desirability.

---

## 1. Metrics

### 1.1 Physical performance metrics

| Metric | Symbol | Why it matters |
|---|---|---|
| Energy per operation | `E_op` (J) | Must compete with or beat digital silicon to be desirable. |
| Operation delay | `t_op` (s) | Latency limits throughput and real-time applicability. |
| Energy-delay product | `E_op × t_op` (J·s) | Combined figure of merit; lower is better. |
| Device volume | `V` (m³) | Determines footprint and integration density. |
| Information / computation density | `ops / (s·m³)` | Measures spatial parallelism. |
| Fundamental energy floor | `kT ln(2) ≈ 3 × 10^-21 J` | Thermodynamic minimum for irreversible bit operations. |

### 1.2 Implementation feasibility metrics

| Metric | Why it matters |
|---|---|
| Forward model maturity | Do we have a solver we can implement and trust within days? |
| Parameter availability | Do we know the material constants, or are they speculative? |
| Numerical tractability | Can we optimise the design space in a reasonable time? |
| Experimental feasibility | Can the device actually be fabricated with available methods? |
| Readout / I/O feasibility | Can we inject inputs and read outputs reliably? |

### 1.3 Research desirability metrics

| Metric | Why it matters |
|---|---|
| Novelty / differentiation | Is the approach already saturated in the literature? |
| Supervisor alignment | Does it match the supervisor's stated interests? |
| Thesis coherence | Can it support a clear narrative and concrete results? |
| Risk within MSc timeline | Can a first demonstration be completed by August? |

---

## 2. Reference baselines

| System | Energy per operation | Delay | Notes |
|---|---:|---|---|
| Thermodynamic minimum | `3 × 10^-21 J` | — | Landauer limit for irreversible bit erasure. |
| Biological synapse | `10^-16` to `10^-15 J` | ~1 ms | Nature's reference for analog neural computation. |
| Digital MAC (7 nm AI accelerator) | `10^-15` to `10^-12 J` | ~1 ns | Current silicon baseline. |
| Digital SRAM access | `10^-15 J` per bit | ~1 ns | Memory reference. |
| Human brain (synaptic) | `~10^-14 J` | ~10 ms | Whole-system estimate including support metabolism. |

---

## 3. Candidate evaluation

### 3.1 Acoustic porous wave computing (current direction)

#### Physical estimates

At 2 kHz in air:

```
λ = c0 / f = 343 / 2000 = 0.1715 m
I = p_rms² / (rho0 c0) = 1² / (1.225 × 343) ≈ 2.4 mW/m²
E_op ≈ I × A × t = 2.4 mW/m² × 10^-4 m² × 10^-3 s ≈ 2.4 × 10^-10 J = 0.24 nJ
```

For a 0.5 m × 0.5 m × 0.01 m device:

```
V ≈ 2.5 × 10^-3 m³
```

#### Feasibility scores

| Criterion | Score (1–5) | Justification |
|---|---:|---|
| Forward model maturity | 3 | ZK solver exists but is invalid at 2 kHz; JCA Helmholtz solver not yet working. |
| Parameter availability | 4 | Acoustic properties of porous materials are tabulated. |
| Numerical tractability | 3 | FDTD works, but optimisation is slow and device is wavelength-limited. |
| Experimental feasibility | 2 | 0.5 m audio device is large; precise porous microstructure is hard. |
| Readout / I/O | 4 | Microphones and speakers are mature. |

#### Desirability scores

| Criterion | Score (1–5) | Justification |
|---|---:|---|
| Energy efficiency | 1 | 0.24 nJ is 2 to 5 orders of magnitude worse than digital. |
| Speed | 2 | ms-scale latency; kHz bandwidth. |
| Size / density | 1 | Device is metres-scale if multiple wavelengths are needed. |
| Novelty | 2 | Wave-physics PNNs are now well-covered (Hughes, Kalthoff, diffractive networks). |
| Supervisor alignment | 1 | Supervisor explicitly does not want this direction. |
| Thesis coherence | 2 | Current model has internal inconsistencies (ZK vs JCA, Helmholtz vs FDTD, sound-speed bug). |
| Timeline risk | 4 | Could fix and run quickly, but the result would still be physically unattractive. |

#### Summary

Acoustic porous computing at audio frequencies is **feasible to simulate but physically undesirable**. The energy, size, and novelty cases are weak. Continuing would require either much higher frequencies or a different medium, which changes the project scope.

---

### 3.2 Chemical Reaction Networks (CRNs)

#### Physical estimates

Single molecular transformation costs roughly `kT` to `10 kT`:

```
kT at 300 K ≈ 4 × 10^-21 J
```

For `N` molecules per operation:

| N | E_op |
|---:|---:|
| 1 | `4 × 10^-21 J` |
| 10^6 | `4 fJ` |
| 10^9 | `4 pJ` |
| 10^12 | `4 nJ` |

A 1 µL well-mixed reactor is small and can hold `10^12` to `10^15` molecules depending on concentration.

#### Feasibility scores

| Criterion | Score (1–5) | Justification |
|---|---:|---|
| Forward model maturity | 5 | Mass-action ODEs are standard; solvable in minutes with SciPy. |
| Parameter availability | 3 | Rate constants can be chosen or fitted; some real data exists. |
| Numerical tractability | 5 | Gradient-free optimisation over rate constants is fast. |
| Experimental feasibility | 3 | Test-tube chemistry is mature, but precise rate control is hard. |
| Readout / I/O | 3 | Fluorescence, absorbance, or electrochemical readout; encoding inputs as concentrations needs care. |

#### Desirability scores

| Criterion | Score (1–5) | Justification |
|---|---:|---|
| Energy efficiency | 4 | Can approach fJ per operation with small molecule counts. |
| Speed | 2 | Reaction times from µs to seconds; often diffusion/mixing limited. |
| Size / density | 4 | µL reactors are compact; massive parallelism possible in droplet arrays. |
| Novelty | 3 | CRN neural networks exist (Nagipogu & Reif 2025) but are not saturated. |
| Supervisor alignment | 5 | Directly matches "concentrations and reactions". |
| Thesis coherence | 4 | Clean mathematical model, clear trainable parameters, reproducible benchmarks. |
| Timeline risk | 5 | First benchmark can run within days. |

#### Summary

CRNs are the **fastest path to a defensible thesis contribution**. They are energetically attractive, mathematically clean, and directly aligned with the supervisor's direction. The main weakness is speed and the abstraction from a spatial device.

---

### 3.3 Reaction-diffusion computing

#### Physical estimates

Energy per pattern update for a 1 mm³ active region with 1 mM reactants:

```
N = 10^-3 mol/L × 10^-9 L × 6 × 10^23 ≈ 6 × 10^11 molecules
E_op ≈ 6 × 10^11 × 4 × 10^-21 J ≈ 2 nJ
```

Pattern wavelength:

```
λ ≈ sqrt(D τ)
```

For `D = 10^-9 m²/s` and `τ = 0.1 s`, `λ ≈ 10 µm`. Device size is mm to cm.

#### Feasibility scores

| Criterion | Score (1–5) | Justification |
|---|---:|---|
| Forward model maturity | 4 | Reaction-diffusion PDEs are standard; 1-D/2-D solvers are straightforward. |
| Parameter availability | 3 | Reaction and diffusion parameters depend on chosen chemical system. |
| Numerical tractability | 3 | PDE optimisation is slower and more fragile than ODE optimisation. |
| Experimental feasibility | 3 | Gel reactors, microfluidic channels, or thin films can host RD dynamics. |
| Readout / I/O | 3 | Optical readout of colour/intensity changes; input patterning by light or masks. |

#### Desirability scores

| Criterion | Score (1–5) | Justification |
|---|---:|---|
| Energy efficiency | 2 | nJ per update is worse than CRNs and competitive with acoustic. |
| Speed | 1 | Pattern evolution takes 0.1 to 10 seconds. |
| Size / density | 3 | µm-scale features but cm-scale devices for useful patterns. |
| Novelty | 3 | Demonstrated for logic gates and associative memory, but still niche. |
| Supervisor alignment | 4 | Spatial continuous medium with concentrations and reactions. |
| Thesis coherence | 4 | Visual, intuitive results; connects to well-known BZ / Gray-Scott literature. |
| Timeline risk | 3 | More code and tuning than CRNs, but feasible in weeks. |

#### Summary

Reaction-diffusion is **desirable as a spatial demonstration** but slower and less efficient than CRNs. Best used as an extension, not the primary model.

---

### 3.4 Ion channels / excitable membranes

#### Physical estimates

Energy to move one ion across 100 mV, including gradient maintenance:

```
E_ion ≈ q V ≈ 1.6 × 10^-20 J
E_pump ≈ 10 kT ≈ 4 × 10^-20 J per ion
E_op for 10^6 ions ≈ 4 × 10^-14 J = 40 fJ
```

#### Feasibility scores

| Criterion | Score (1–5) | Justification |
|---|---:|---|
| Forward model maturity | 4 | Hodgkin-Huxley and Markov models are well-established. |
| Parameter availability | 4 | Channel kinetics and conductances are tabulated for common channels. |
| Numerical tractability | 4 | ODE networks simulate quickly. |
| Experimental feasibility | 2 | Reconstituting channel networks in artificial membranes is hard. |
| Readout / I/O | 3 | Voltage / current readout; input via ligands or voltage. |

#### Desirability scores

| Criterion | Score (1–5) | Justification |
|---|---:|---|
| Energy efficiency | 5 | Best among the candidates; competitive with biology. |
| Speed | 3 | Channel gating is µs to ms; network propagation is ms. |
| Size / density | 4 | Membranes are nm-thin; devices can be µm to mm. |
| Novelty | 3 | Ion-channel computing is known but not mainstream. |
| Supervisor alignment | 4 | Explicitly includes ion channels. |
| Thesis coherence | 3 | Biophysics framing may drift from "fluid neural network" theme. |
| Timeline risk | 3 | Model is tractable, but connecting it to a physical device is speculative. |

#### Summary

Ion channels are **energetically the most attractive** but harder to map onto a manufacturable device. Strong as a conceptual anchor, weaker as an MSc implementation.

---

### 3.5 Organic / iontronic devices (OECTs, porous memristors)

#### Physical estimates

Ideal switching energy:

```
E = 1/2 C V² = 0.5 × 1 fF × (0.1 V)² = 5 × 10^-18 J = 5 aJ
```

Real devices are larger and leakier, so practical energies are fJ to pJ.

#### Feasibility scores

| Criterion | Score (1–5) | Justification |
|---|---:|---|
| Forward model maturity | 2 | Coupled electrochemistry, ion transport, and polymer physics; harder to model cleanly. |
| Parameter availability | 2 | Doping, ion mobility, and switching thresholds are device-specific. |
| Numerical tractability | 2 | Drift-diffusion-Poisson plus hysteresis is complex. |
| Experimental feasibility | 3 | OECT fabrication is established in labs. |
| Readout / I/O | 4 | Electrical readout is straightforward. |

#### Desirability scores

| Criterion | Score (1–5) | Justification |
|---|---:|---|
| Energy efficiency | 4 | Potentially very low switching energy. |
| Speed | 3 | µs to ms, limited by ion diffusion. |
| Size / density | 4 | µm-scale channels. |
| Novelty | 5 | Highly novel for this project; connects porous media to neuromorphic electronics. |
| Supervisor alignment | 3 | Less explicitly chemical; more electronic materials. |
| Thesis coherence | 2 | Hardest to fit into a clean thesis narrative in the time available. |
| Timeline risk | 1 | High risk of getting stuck on model formulation. |

#### Summary

Organic / iontronic devices are **the most novel but the least feasible within an MSc timeline**. Save for future work.

---

## 4. Summary comparison

| Candidate | Energy | Speed | Size | Feasibility | Novelty | Supervisor fit | Timeline risk | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Acoustic porous (current) | 1 | 2 | 1 | 3 | 2 | 1 | 4 | **1.7** |
| CRNs | 4 | 2 | 4 | 5 | 3 | 5 | 5 | **4.0** |
| Reaction-diffusion | 2 | 1 | 3 | 3 | 3 | 4 | 3 | **2.7** |
| Ion channels | 5 | 3 | 4 | 4 | 3 | 4 | 3 | **3.7** |
| Organic / iontronic | 4 | 3 | 4 | 2 | 5 | 3 | 1 | **3.2** |

*Scores are 1 (poor) to 5 (excellent). Overall is the rounded mean.*

---

## 5. Recommended decision

**Pivot to CRNs as the primary model, with reaction-diffusion as a spatial extension if time permits.**

Justification:
- CRNs score highest on feasibility, supervisor alignment, and timeline risk.
- They have the best energy scaling if small molecule counts are used.
- They provide a clean mathematical foundation for the thesis.
- Reaction-diffusion can be added later by spatially extending the CRN model.
- The acoustic work is retained only as a comparison point in the envelope study.

---

## 6. How to present this to the supervisor

Frame the pivot as an evidence-based decision, not a retreat:

1. **We evaluated five physical-computing substrates** on energy, speed, size, feasibility, novelty, and alignment.
2. **Acoustic porous computing scored lowest** because audio wavelengths force a large device and high energy per operation, and the ZK model is invalid at speech frequencies.
3. **CRNs scored highest** because they offer a tractable ODE forward model, good energy scaling, and direct concentration/reaction dynamics.
4. **We propose CRNs as the main contribution**, with the acoustic work becoming a comparison baseline.

This turns the meeting outcome into a rigorous envelope study rather than an admission of failure.

---

## 7. Next step

Use this framework to update `Notes/supervisor-meeting-pivot-2026-06-23.md` and prepare a short summary slide or paragraph for the supervisor. Once direction is confirmed, implement the CRN forward solver.
