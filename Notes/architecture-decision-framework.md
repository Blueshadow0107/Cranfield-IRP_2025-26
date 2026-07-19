# Architecture Decision Framework: Porous Wave Computer

*Date: 2026-05-31*  
*Context: Cranfield IRP 2025-26 — Continuous Analog Wave Computation in Porous Media*  
*Status: Decision pending — user reviewing trade-offs*

---

## The Core Question

> Is "porous media" a **fixed material property** of the device, or a **dynamically emergent property** of the fluid?

This document captures the three candidate architectures, their physics, simulation feasibility, experimental realizability, and novelty claims. The user is currently evaluating these options before committing to a direction.

---

## Architecture A: Fixed Solid Scaffold + Saturated Fluid (Thermal Tuning)

### Concept
A permanent 3D porous solid (sintered beads, printed lattice, reticulated foam) is immersed in fluid. The pore geometry is fixed at fabrication time. External lasers project heating patterns onto the scaffold, locally altering fluid properties (sound speed, density, viscosity) within the pores. Optionally, heating may nucleate bubbles that add temporary scatterers inside the fixed pore network.

```
[Laser] → heats fluid in selected pores
                ↓
    [Fixed solid scaffold] → [Fluid-filled pores] → [Fluid properties c(T), ρ(T) changed]
                ↓
    [Acoustic/elastic waves propagate through Biot-coupled medium]
```

### Physics
- **Governing equations:** Biot poroelasticity (solid displacement **u**, fluid displacement **w**, pressure **p** coupled)
- **Wave modes:** 3 modes — fast compressional, slow compressional, shear
- **Thermal coupling:** Linear parameter variation — c(T), ρ(T), viscosity(T)
- **Nonlinearity:** Weak (unless large ΔT)
- **Timescales:** Acoustic (~μs) >> Thermal (~ms) → quasi-static programming

### Simulation
- **FDTD difficulty:** Hard but documented. Biot FDTD exists in literature.
- **OpenFOAM suitability:** Good — `poroelasticFoam` or Biot extensions exist.
- **CFD validation:** Straightforward — compare to Biot analytical solutions.
- **Time to first result:** ~4–6 weeks for 2D Y-junction
- **Grid requirements:** Resolve pore scale + wavelength

### Experimental
- **Build complexity:** Medium — porous scaffold + fluid chamber + lasers
- **Repeatability:** High — fixed scaffold, controllable thermal pattern
- **Laser requirements:** CW or pulsed, moderate power
- **Sensing:** Microphone array / hydrophone
- **Safety:** Low risk (heated fluid, controlled ΔT)
- **Materials cost:** ~£50–200

### Novelty
- **Claim:** "Trainable poroelastic wave computer via optothermal programming"
- **Gap:** No existing work trains Biot media with spatial thermal patterns.
- **Defensibility:** High — physics is well-established, results reproducible.

---

## Architecture B: Fluid Domain + Dynamically Generated Porosity (Bubble Cloud)

### Concept
No fixed solid scaffold. Just a fluid chamber. Laser heating nucleates microbubbles at specific locations. The bubble cloud *is* the porous medium — its geometry, porosity, and scattering properties are entirely determined by the heating pattern.

```
[Laser] → heats fluid → nucleates bubbles at nucleation sites
                ↓
    [Bubble cloud] = [Temporary porous structure with φ(x,y,z,t)]
                ↓
    [Acoustic waves propagate through bubbly fluid]
```

### Physics
- **Governing equations:** Compressible Navier-Stokes + Rayleigh-Plesset / Keller-Miksis for each bubble
- **Wave modes:** 1 mode — compression in bubbly fluid (no shear without solid)
- **Thermal coupling:** Nonlinear — nucleation threshold, latent heat, bubble growth/collapse
- **Nonlinearity:** Strong (bubble resonance, collapse, Bjerknes forces)
- **Timescales:** Acoustic (~μs) ≈ Bubble oscillation (~μs) → resonant coupling

### Simulation
- **FDTD difficulty:** Medium for single bubble; extremely hard for cloud. Need multiple scattering theory (Foldy-Lax).
- **OpenFOAM suitability:** Poor — bubbly flow requires specialized solvers (interFoam, compressibleInterFoam).
- **CFD validation:** Hard — bubble clouds are stochastic.
- **Time to first result:** ~2–3 weeks for single bubble; ~3+ months for cloud
- **Grid requirements:** Resolve bubble interface + wavelength

### Experimental
- **Build complexity:** Low — just a fluid chamber + lasers
- **Repeatability:** Low — nucleation is stochastic, bubbles rise/dissolve
- **Laser requirements:** High peak power for nucleation, precise focusing
- **Sensing:** Hydrophone + high-speed imaging
- **Safety:** Medium risk — cavitation damage, vapor pressure
- **Materials cost:** ~£20–50

### Novelty
- **Claim:** "Fluid computer with dynamically generated porosity via laser nucleation"
- **Gap:** Bubble computing exists (Prakash & Gershenfeld 2007) but not as trainable porosity.
- **Defensibility:** Medium — bubble stochasticity makes results hard to reproduce.

---

## Architecture C: Hybrid — Fixed Scaffold + Dynamic Bubble Injection

### Concept
Fixed porous scaffold provides the base structure. Additional bubbles are nucleated within the pores via laser heating, creating a two-phase flow inside an already-porous medium.

```
[Laser] → heats fluid in pores → nucleates bubbles inside fixed pore network
                ↓
    [Fixed scaffold] + [Fluid] + [Dynamic bubbles] = [Two-phase porous medium]
                ↓
    [Full Biot + two-phase effects]
```

### Physics
- **Governing equations:** Biot poroelasticity + two-phase flow + bubble dynamics
- **Wave modes:** 3 Biot modes + bubble resonance modes
- **Thermal coupling:** Both linear (c(T)) and nonlinear (nucleation)
- **Nonlinearity:** Strong + coupled across all physics
- **Timescales:** Acoustic, thermal, and bubble dynamics all coupled

### Simulation
- **FDTD difficulty:** Very hard. No off-the-shelf code.
- **OpenFOAM suitability:** Very poor. Would require custom solver.
- **CFD validation:** Very hard — no analytical solutions.
- **Time to first result:** ~4+ months before demonstrable results
- **Grid requirements:** Resolve pore scale + bubble interface + wavelength

### Experimental
- **Build complexity:** High — both scaffold and bubble control needed
- **Repeatability:** Medium-low — bubble behavior inside pores is unpredictable
- **Laser requirements:** High power + good focusing into pore network
- **Sensing:** Both microphone array and high-speed imaging
- **Safety:** Medium-high risk — heat + pressure + cavitation
- **Materials cost:** ~£100–300

### Novelty
- **Claim:** "Two-phase poroelastic computer with thermal and bubble reconfiguration"
- **Gap:** Completely unexplored. Very strong novelty.
- **Defensibility:** High risk — if simulations fail, thesis has no core contribution.

---

## Side-by-Side Comparison

| Criterion | A (Fixed + Thermal) | B (Bubble Cloud) | C (Hybrid) |
|-----------|---------------------|------------------|------------|
| **Physics maturity** | ⭐⭐⭐ Well-established | ⭐⭐ Moderate | ⭐ Poor — custom needed |
| **Simulation feasibility** | ⭐⭐⭐ Hard but doable | ⭐⭐ Medium | ⭐ Very hard |
| **Experimental ease** | ⭐⭐ Medium | ⭐⭐⭐ Easy | ⭐ Hard |
| **Result repeatability** | ⭐⭐⭐ High | ⭐ Low | ⭐⭐ Medium-low |
| **Novelty strength** | ⭐⭐⭐ Strong gap | ⭐⭐ Moderate gap | ⭐⭐⭐⭐ Unexplored |
| **Thesis defensibility** | ⭐⭐⭐ Low risk | ⭐⭐ Medium risk | ⭐ High risk |
| **Time to first result** | ~4–6 weeks | ~2–3 weeks (1 bubble) | ~4+ months |
| **Shear waves?** | ✅ Yes | ❌ No | ✅ Yes |
| **Biot theory applies?** | ✅ Yes | ❌ No | ✅ Yes |
| **Optothermal reconfig?** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Dynamic bubble effects?** | ⚠️ Optional | ✅ Core | ✅ Core |

---

## User Preferences (from Q&A, 2026-05-31)

| Question | User Response |
|----------|--------------|
| Thermal scope | **Trainable resource** — actively exploit thermal effects |
| Thermal waves as carriers | **Unsure** — needs more investigation |
| Thermal reconfigurability | **Yes** — wants laser heating of fluid regions, NOT metamaterials. Considering bubbles or heated fluid. |
| Cross-domain interest | **Elastic/shear waves** — wants full poroelasticity |
| Architecture preference | **Undecided** — requested trade-off elaboration |
| Scaffold type (if A/C) | **Generic** — keep material abstract for now |
| Nucleation mechanism (if B/C) | **Laser heating past boiling point** |
| Poroelastic model | **Full Biot** — all 3 wave types |
| Timescale | **Both** — start static, explore dynamic |

---

## Recommendation

**Primary recommendation: Architecture A as core thesis, with Architecture C features as stretch goal/future work.**

This gives:
- A solid, defensible core contribution (trainable Biot medium)
- A clear path to greater ambition if time permits
- A thesis that won't collapse if bubble dynamics prove uncontrollable
- Full use of shear waves and Biot theory (user requirement)
- Optothermal reconfigurability via c(T), ρ(T) modulation

**If time allows in August:** Add bubble nucleation experiments as a "dynamic reconfiguration" extension, moving toward Architecture C.

---

## Open Questions to Resolve

1. **Temperature range:** What ΔT is achievable with the chosen laser + fluid? Does c(T) vary enough to create meaningful wave-path changes?
2. **Thermal diffusion:** How localized can the heating be? Does heat spread too fast to maintain sharp "weight" boundaries?
3. **Bubble threshold:** At what temperature/pressure does nucleation occur in the chosen fluid? Is it repeatable?
4. **Scaffold material:** Does the user want to commit to a specific porous material (e.g., 3D printed resin, sintered glass, metal foam)?
5. **Shear wave I/O:** How are shear waves excited and detected? This is harder than pressure-wave transduction.

---

## Related Literature to Survey

Once architecture is chosen, search these domains:

### For Architecture A
- Optoacoustic / photothermal generation in fluids
- Temperature-dependent acoustic properties of water/fluids (c(T), ρ(T), viscosity(T))
- Full Biot theory FDTD/CFD implementation
- Thermal lensing / gradient-index acoustics
- Dynamically tunable acoustic metasurfaces

### For Architecture B
- Bubble nucleation & cavitation in confined geometries
- Rayleigh-Plesset / Keller-Miksis bubble dynamics
- Foldy-Lax multiple scattering theory for bubble clouds
- Porous media boiling / two-phase flow in pores

### For Architecture C
- Two-phase flow in porous media (relative permeability, capillary pressure)
- Biot theory with fluid phase change
- Laser-induced cavitation in porous media
- Nonlinear acoustics in bubbly porous media

---

*Next action: User to confirm architecture choice, then proceed to targeted literature search.*
