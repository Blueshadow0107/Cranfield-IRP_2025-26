# Substrate limitations and how to use what works

## 1. Acoustic wave computing (legacy)

| Limitation | Why it arises | Workaround / use |
|------------|---------------|------------------|
| Linear at low Re | Acoustic wave equation is essentially linear for small perturbations | Need nonlinear media or high amplitude; not pursued |
| Weak trainable nonlinearity | Cylinder wakes give amplitude readout but not rich Boolean logic | Brinkman penalization considered, but literature is crowded |
| Weak CFD link | Supervisor challenged it as well-covered | Dropped; pivoted to chemistry |

## 2. Well-mixed CRNs

| Limitation | Why it arises | Workaround / use |
|------------|---------------|------------------|
| No spatial continuity | ODEs, no spatial transport | Move to PDE / reaction-diffusion |
| No trainable geometry | State is a vector, not a field | Use spatial RD with geometry or phi field |
| Weak CFD link | No flow, no diffusion numerics | RD gives direct PDE/CFD connection |

## 3. DNA strand displacement / winner-take-all

| Limitation | Why it arises | Workaround / use |
|------------|---------------|------------------|
| No spatial continuity | Molecular computation in well-mixed soup | RD gives spatial dynamics |
| No trainable geometry | Reactions happen in a compartment | Geometry + phi field become weights |
| Complex biochemistry | Strand displacement rates hard to control in silico | Oregonator is simpler and well-parameterised |

## 4. Oregonator reaction-diffusion (current)

| Limitation | Why it arises | Workaround / use |
|------------|---------------|------------------|
| Static phi cannot classify dynamic inputs (duration, frequency) | phi is a spatial field; refractory memory is erased as waves propagate | Use trainable geometry (walls), sequential stages, or accept phi as a programmable router |
| Static phi cannot distinguish symmetric inputs (A vs B in same stem) | Field sees both inputs identically | Different input geometries, trainable walls, time-sequenced inputs |
| Narrow phi dynamic range (~0.010-0.040) | Oregonator excitability has a sharp threshold | Use phi as a sharp switch, not a graded weight |
| Each training eval is expensive | Forward PDE solve takes seconds to minutes | Small design spaces, CMA-ES, surrogate models, HPC |
| Readout needs windowed spatial max | Wavefront is compact; mean is diluted by wake | Design probes as spatial-max detectors |
| Two-input logic needs timing | Collision computing requires coincident pulses | Use delay lines / anisotropic patches to synchronise pulses |

## What works and how to build networks from it

### Primitive building blocks
1. **Channel** = wire (near-ideal propagation, speed set by kinetics).
2. **T-junction** = splitter / collision gate (OR natively; A AND NOT B with windowed readout).
3. **Anisotropic patch** = tunable delay / directional coupler.
4. **Trainable phi router** = programmable single-input switch (left vs right).

### Possible small circuits
- **Two routers + collision gate**: router A and router B each steer their pulses; a downstream T-junction performs collision logic.
- **Delay line + T-junction**: anisotropic patch creates a controlled delay so pulses meet at the junction for coincidence logic.
- **Frequency discriminator**: use the medium's native refractory filtering; a high-frequency train is attenuated, a low-frequency train passes. ( phi alone cannot route by frequency, but walls + frequency could.)
- **Sequential state**: a first pulse puts a region in refractory state; a second pulse arriving shortly after is blocked or rerouted. This gives one bit of memory.

### Honest thesis scope
For the remaining time, the safest result is:
- Characterise the primitives (channel, refractory, junction logic, anisotropy, trainable phi router).
- Show that composition is conceptually possible with delays/junctions.
- Report the negative results (phi cannot classify dynamic features) as constraints.

A full multi-gate trained network is likely out of scope for the time left.
