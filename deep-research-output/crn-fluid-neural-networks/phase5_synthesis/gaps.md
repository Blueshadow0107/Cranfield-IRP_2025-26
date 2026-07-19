# Phase 5: Gap Analysis

This document distills the unresolved problems and open research gaps identified during the deep dive. Each gap is stated, justified with references, and linked to a concrete thesis opportunity.

---

## Gap 1: Stochastic trainability at finite copy number

**Statement.** Universal approximation is proved for deterministic mass-action ODEs (Dack et al.), but it is not known whether the same trainable CRNs preserve their function under stochastic chemical kinetics at low molecule counts.

**Evidence.**
- Dack's proof assumes the thermodynamic limit; finite-copy-number effects are not bounded.
- Poole et al. handles stochastic CRNs, but only for detailed-balanced equilibrium distributions, not for the nonequilibrium recurrent dynamics needed for RNNs.
- Kang & Kim's noise analysis is deterministic (rate perturbations), not Gillespie-stochastic.

**Why it matters.** Any experimental implementation will operate at finite copy number. A trained CRN that works in ODE simulation may fail when reactions become discrete and noisy.

**Thesis opportunity.** Take the trained deterministic CRN from Dack/Kang and run stochastic simulations (Gillespie or chemical Langevin). Quantify the probability of correct classification / target dynamics as a function of reactor volume and initial copy number.

---

## Gap 2: No experimental demonstration of in-chemico learning

**Statement.** No published wet-lab experiment has shown a CRN that updates its own rate constants or weights in response to an error signal.

**Evidence.**
- Qian et al. and Okumura et al. compile fixed networks designed in silico.
- Baltussen et al. trains only the readout; the chemistry is fixed.
- Kang & Kim proposes a single-pot learning CRN but remains a simulation.

**Why it matters.** In-chemico learning is the defining feature of a "trainable" chemical computer versus a merely programmable one.

**Thesis opportunity.** Simulate the full Kang & Kim forward+backward+update CRN in a well-mixed reactor. Show that the weights converge to values that solve XOR or a regression task, and study how the learning rate and noise affect convergence.

---

## Gap 3: Readout without perturbation

**Statement.** Continuous, non-destructive readout of many chemical species in a well-mixed CRN remains unsolved.

**Evidence.**
- Baltussen uses mass spectrometry on sampled aliquots — destructive and slow.
- Okumura and Pandi use fluorescence reporters — limited to one or a few channels and require reporter species that may sequester products.
- Electrochemical readouts (Kamsma) work only for ionic outputs.

**Why it matters.** A trainable CRN needs an error signal, which implies measuring the output. If measurement consumes the output, the feedback loop is compromised.

**Thesis opportunity.** In simulation this gap disappears because concentrations are fully observable. The thesis can therefore focus on trainability first, then discuss readout engineering as a future-work concern.

---

## Gap 4: Scalability beyond toy systems

**Statement.** Demonstrations of trainable CRNs use fewer than ~10 species/neurons. Scaling to tens or hundreds of species is unproven.

**Evidence.**
- Dack's examples: 3–8 perceptrons.
- Okumura: ~10-neuron two-layer network.
- Kang & Kim: MNIST in principle, but no detailed network-size analysis.

**Why it matters.** Useful computation (e.g., even simple pattern classification) may require dozens of species, and the training landscape may become harder as the network grows.

**Thesis opportunity.** Perform scaling experiments in simulation: train RNCRNs with 2, 4, 8, 16 perceptrons on increasingly complex tasks. Measure training success rate and convergence time. This provides an original empirical scalability curve.

---

## Gap 5: Fuel and energy budget

**Statement.** Trainable CRNs are open systems that consume fuel species. The lifetime and energy efficiency of a chemical computer are rarely analyzed.

**Evidence.**
- Dack assumes chemostatted fuel complexes.
- Kang & Kim assumes continuous supply of inputs and fuels.
- Baltussen's CSTR is continuously fed.

**Why it matters.** A closed-batch "battery-powered" chemical computer would be more practical, but fuel depletion will degrade performance over time.

**Thesis opportunity.** Introduce finite fuel reservoirs into the simulation. Track task accuracy versus cumulative fuel consumption. Compare the energy per operation with digital and iontronic benchmarks.

---

## Gap 6: Trainable reaction–diffusion media

**Statement.** RD computers are typically hand-designed for specific logic gates. There is little work on optimizing RD parameters (diffusion coefficients, initial conditions, reaction rates) to perform arbitrary tasks.

**Evidence.**
- Adamatzky & De Lacy Costello design XOR gates by cutting gel geometry.
- Gorecki et al. review pre-designed BZ droplet networks.
- No gradient-based training of RD patterns for classification was found in the survey.

**Why it matters.** RD media naturally implement parallel wave computation. If their parameters can be learned, they become trainable spatial analog computers.

**Thesis opportunity.** Build a 1D or 2D finite-difference RD solver. Define a simple spatial task (e.g., classify two input patterns by final concentration profile). Optimize diffusion coefficients or initial conditions via gradient descent or evolutionary search. This is a genuinely novel contribution at the intersection of CRNs and RD computing.

---

## Gap 7: Benchmarking against fluidic / acoustic alternatives

**Statement.** The relative advantages of CRN computing versus iontronic, memristive, or acoustic-wave physical neural networks are not quantitatively compared in the literature.

**Evidence.**
- Kamsma et al. compares iontronic reservoirs to solid-state memristors but not to chemical systems.
- Hughes et al. compares wave-RNN to digital RNNs but not to chemistry.
- Baltussen does not compare the formose reservoir to other physical computers.

**Why it matters.** The thesis must justify why CRNs are a worthwhile substrate compared to faster or more mature alternatives.

**Thesis opportunity.** Construct a qualitative comparison table (energy, speed, programmability, readout, fabrication) using published numbers. Where simulation provides data (e.g., operations per joule), make quantitative estimates.

---

## Gap 8: Compiler from trained CRN to experimental chemistry

**Statement.** While tools exist to compile abstract CRNs to DNA strand displacement (Peppercorn, Nuskell), no pipeline compiles a *trained* CRN with specific rate constants to a realizable experimental protocol.

**Evidence.**
- Dack argues DNA implementability but does not provide a full compiler pipeline.
- Nuskell maps CRNs to DSD systems, but the mapping assumes idealized rates.

**Why it matters.** Without a compiler, a trained CRN remains an abstract model.

**Thesis opportunity.** As a conceptual future-work section, outline how a trained CRN could be passed through Nuskell + NUPACK/Nuad. Identify which rate constants map to which DNA toehold lengths/concentrations, and discuss the mismatch between desired and achievable rates.

---

## Summary of thesis-relevant gaps

| Priority | Gap | Feasible in simulation? | Original contribution potential |
|----------|-----|------------------------|---------------------------------|
| High | Stochastic trainability | Yes | High |
| High | In-chemico learning simulation | Yes | High |
| High | Scalability curves | Yes | High |
| Medium | Fuel / energy budget | Yes | Medium-High |
| Medium | Trainable RD media | Yes | High |
| Medium | Benchmarking vs. other substrates | Partially (qualitative + some numbers) | Medium |
| Lower | Non-destructive readout | No (engineering problem) | Conceptual discussion |
| Lower | CRN-to-DNA compiler | Partially (conceptual) | Low-Medium |

The highest-impact simulation work for the thesis is: **stochastic trainability, in-chemico learning, and scalability of smooth-activation CRNs**, followed by **trainable RD media** as a spatial extension.
