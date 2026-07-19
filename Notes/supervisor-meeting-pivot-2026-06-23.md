# Supervisor Meeting Outcome and Possible Pivot

**Date:** 2026-06-23  
**Context:** Post-progress-presentation supervisor meeting.

## Key outcome

The supervisor is not convinced by the acoustic wave-computing / porous-media direction. His view is that wave-physics physical neural networks are already well-covered in the literature (Hughes 2019, Kalthoff 2025, diffractive networks, etc.). He wants the project to move back toward **concentrations, reactions, and ion-channel-like dynamics**.

He asked for an **envelope study**: map the feasible operating region of the chemical / ion-channel / reaction-based approach before committing to a specific implementation.

## What an envelope study means here

Map the outer limits of what a chemical / ionic computing approach can do, including:

- What concentrations, reaction rates, and diffusion lengths are physically achievable?
- What input/output encoding schemes work?
- How fast can it compute, how much information can it process, and when does it fail?
- What has been done in the literature, and where is the hard ceiling?

The wave-computing work already completed can serve as a baseline for comparison in this envelope study.

## Candidate directions under consideration

These are not final. The purpose is to compare them and pick the most viable for the remaining MSc timeline.

### 1. Trainable Chemical Reaction Networks (CRNs)
- **Model:** Mass-action kinetics ODEs.
- **Trainable:** Reaction rate constants and network topology.
- **Anchor literature:** Nagipogu & Reif 2025 (Neural CRNs).
- **Pros:** Fast to simulate; compact; direct mapping from trained NN to reactions.
- **Cons:** Well-mixed (no spatial structure); encoding inputs/outputs as concentrations needs care.

### 2. Reaction-Diffusion Computing
- **Model:** PDEs such as Gray-Scott or FitzHugh-Nagumo.
- **Trainable:** Reaction parameters, diffusion coefficients, initial/boundary conditions.
- **Anchor literature:** Adamatzky, Stovold, BZ logic gates.
- **Pros:** Spatial and visual; demonstrated logic gates and pattern formation.
- **Cons:** Stiff PDEs; optimisation harder; dynamics can be chaotic.

### 3. Ion Transport in Porous / Membrane Media
- **Model:** Nernst-Planck + Poisson + gating kinetics in a porous host.
- **Trainable:** Pore geometry, channel density, ionic diffusivity.
- **Anchor literature:** Mesoporous silica memristors, geopolymer memristors.
- **Pros:** Keeps porous-media angle; memristive memory; continuous.
- **Cons:** Electrochemistry parameters hard to obtain; drift-diffusion-Poisson more code.

### 4. Ion-Channel / Excitable Membrane Networks
- **Model:** Hodgkin-Huxley or Markov-state gating ODEs.
- **Trainable:** Channel densities, conductances, gating parameters.
- **Pros:** Biologically grounded; spiking logic.
- **Cons:** More neuroscience than fluid computing; network design non-trivial.

### 5. Organic Electrochemical / Iontronic Systems
- **Model:** Doping-level dynamics in conducting polymers (e.g., PEDOT:PSS), coupled to ion diffusion.
- **Trainable:** Polymer geometry, ionic properties, switching thresholds.
- **Pros:** Highly novel; real experimental precedents (OECTs).
- **Cons:** Complex physics; parameters scarce; highest risk for MSc timeline.

### 6. Acoustic Wave Computing (existing work)
- **Model:** Zwikker-Kosten / JCA Helmholtz in porous media.
- **Status:** Already implemented and validated.
- **Role:** Use as a comparison baseline for the envelope study, not necessarily the final thesis direction.

## Proposed MSc-scoped unifying idea

A **continuous porous excitable medium** that combines reaction-diffusion with voltage-gated ion transport:

```
∂c_i/∂t = ∇ · [ D_i(g) ∇c_i + (z_i F D_i / RT) c_i ∇φ ] + R_i(c)
∇ · (ε ∇φ) = -F Σ z_i c_i
∂g/∂t = α(φ,c)(1 - g) - β(φ,c)g
```

where `g` is a pore/channel gating variable and `D_i(g)` is the gated diffusivity. This merges options 2, 3, and 4 into one framework. A simplified first step could be a reaction-diffusion system with concentration-dependent diffusivity `D(u)`.

## Immediate next steps (to revisit later)

1. Build the candidate comparison table in a dedicated note.
2. Quickly implement one or two simple benchmarks:
   - CRN 2-input XOR or 2x2 classifier (mass-action ODEs).
   - 1D/2D reaction-diffusion logic gate (Gray-Scott or FitzHugh-Nagumo).
3. Compare against the existing acoustic baseline.
4. Take the comparison back to the supervisor and ask him to pick a concrete direction.

## Open questions

- Does the supervisor want a spatial continuous medium, or is a well-mixed CRN acceptable?
- Did he have a specific system in mind (neurons, OECTs, BZ reaction, ion channels)?
- How much of the wave-computing material can be retained as context / comparison?
- What is the hard deadline by which the final direction must be locked?

## Files to revisit when resuming

- `Notes/89aa5df5.md` — fluid neural network literature review (chemical CRN focus)
- `Notes/fluid-nn-literature-summary.md` — comparison table of fluid/chemical computing approaches
- `Notes/porous-computing-lit-review.md` — porous memristor / ion-transport papers
- `Notes/papers/methodology_review.md` — detailed chemical reaction-diffusion notes
- `Notes/zk-jca-regime-decision-2026-06-18.md` — wave-computing regime note (for comparison)
- `Presentations/progress-2026-06-23.tex` — current wave-computing deck (may need updating or reframing)
