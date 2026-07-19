# Meeting synthesis: CRN hardware / reaction-diffusion PDE roadmap

Date: 2026-06-18
Source: handwritten supervisor meeting notes + follow-up discussion

This note answers each question raised in the meeting notes and turns them into a single executable roadmap.

---

## 1. Is Patankar the simplest source-term treatment for stiff systems?

**Yes, for production-destruction chemistry.**

Patankar-Euler is the simplest *positivity-preserving* point-implicit treatment for a source term that can be split into production and destruction parts. It replaces a stiff sink `-d(y)` with an implicit factor, giving

```
y^{n+1} = y^n * (1 + dt * p(y^n)/y^n) / (1 + dt * d(y^n)/y^n)
```

It is not a general stiff solver — it assumes `p(y) >= 0`, `d(y) >= 0`, and that you only need pointwise stability/positivity rather than global conservation. If mass conservation matters, use **Modified Patankar-Euler (MPE)**; if higher order is needed, use **MPRK** schemes.

For an arbitrary nonlinear vector field (e.g. our full RCN), Patankar does not apply directly unless the reactions are written as production-destruction networks.

---

## 2. How should we handle stiff systems overall?

The meeting notes sketch four sub-problems. Here is the ordered plan:

| Sub-problem | Chosen approach | Rationale |
|-------------|-----------------|-----------|
| Convert ODE CRN to spatial PDE | Add diffusion: `dC/dt = R(C) + D ∇²C` | Natural next step; turns well-mixed RCN into a spatial analog computer |
| Find real reactions | DNA strand displacement as reference platform; map abstract weights to toehold-mediated rates | Most programmable chemistry for independent rate tuning |
| Decouple equations | **Operator splitting**: diffusion step, then reaction step | Lets us use a fast/implicit Riccati solve for perceptrons and a cheap Patankar/MPE step for executives |
| Multi-step point-implicit source update | Use **MPE or BDF/Radau** for the stiff reaction block; optional subcycling | Keeps positivity while allowing larger global timestep |

---

## 3. Which ODE/PDE stiff solver should we use?

For the reaction block after splitting:

| Solver | When to use it | Pros | Cons |
|--------|----------------|------|------|
| **Radau** (`scipy.integrate.Radau`) | Very stiff, implicit, high accuracy | A-stable, adaptive order | Expensive per step, needs Jacobian |
| **BDF** (`solve_ivp` method='BDF' or CVODE) | Moderately stiff, large systems | Good for PDE semidiscretisations | Can be slow for highly nonlinear systems |
| **LSODA** (`solve_ivp` method='LSODA') | Mixed stiff/non-stiff, prototyping | Auto-switches | Less robust for strongly stiff reaction networks |
| **MPE / MPRK** | Production-destruction networks | Positivity + conservation guaranteed | First/second order only; must reformulate reactions |
| **Patankar-Euler** | Simplest positivity fix | Trivial to implement | Not conservative, first order |

**Recommendation:**
- Prototype the reaction step with **Radau** or **BDF** in SciPy for correctness.
- Once the reaction network is fixed as production-destruction, implement **MPE** for the PDE solver so we can take larger diffusion-limited timesteps without worrying about negative concentrations.
- For the PDE diffusion step, use **implicit Euler** (Crank-Nicolson for second-order) or FFT-based spectral solve in 1D.

---

## 4. Do we need specific reactions, and how do we control parameters?

**Yes — the abstract RCN must be grounded in a real chemistry.**

Current mapping (DNA strand displacement):

| Learned parameter | Abstract role | DNA-level control |
|-------------------|---------------|-------------------|
| `ω_{ji}` | Weight executive → perceptron | Toehold length / sequence on `X_i + Y_j` gate |
| `θ_j` | Perceptron bias | Constant fuel / template concentration |
| `α_{ij}` | Weight perceptron → executive | Toehold design on `Y_j + X_i` catalysis |
| `β_out` | Output basal production | Fuel / precursor concentration |
| `δ_i` | Degradation/dilution | Degradase or sink-strand concentration |
| `μ` | Timescale ratio | Relative catalyst concentrations |

Toehold-mediated strand displacement gives roughly **10⁶-fold independent rate tuning**, which is why DNA is the natural target. Simpler chemistries (BZ, small molecules) are easier to run but do not let you tune individual weights independently.

**Control in a real device:** concentrations and sequences are set at fabrication/mixing time; the "program" is the sequence design, not an in-situ knob. See `Notes/experimental_parameter_control_crn.md` for full details.

---

## 5. Should we build a fluorescence chamber?

**For a liquid-phase demonstration, yes.**

A microfluidic Hele-Shaw cell or droplet array with fluorescent reporters is the easiest way to image 1D concentration fields. This matches the "1D space with diffusion" target at the bottom of the notes.

Alternative readouts:
- **Fluorescence** — easiest, high sensitivity, direct concentration image.
- **Colorimetric** — cheaper, lower sensitivity.
- **Interferometric / schlieren** — non-invasive, harder to quantify.

**Recommendation:** design for fluorescence imaging in a thin (< 1 mm) liquid layer or microfluidic channel. This is the standard platform for DNA and enzymatic reaction-diffusion experiments.

---

## 6. What representative realistic test case should we target?

Two candidates, ordered by complexity:

### Option A (recommended first): 1D reaction-diffusion bistability / front propagation
- Two stable states (high/low output concentration).
- Inputs `X1`, `X2` injected at left boundary.
- Output `X3` forms a front whose position or steady-state level depends on `X1 XOR X2`.
- **Why:** simpler than training a full RCN; directly demonstrates spatial analog computation; easy to image; maps cleanly to a 1D PDE.

### Option B: XOR stripe pattern in 1D
- Train/set parameters so that `X3` is high only when exactly one input is high, producing a high-concentration stripe in the centre of the domain.
- **Why:** keeps the XOR task from the ODE work but adds spatial structure.

**Avoid for now:** training the full two-timescale RCN in ODEs (`dack_rncrn_xor_riccati_free_inputs.py`) is still numerically unstable (overflow during optimisation). Fix the stability problem or deprioritise it in favour of the spatial test case.

---

## 7. Phase choice: gas, liquid, or powder?

| Phase | Programmability | Imaging | Diffusion control | Recommendation |
|-------|-----------------|---------|-------------------|----------------|
| **Liquid (aqueous)** | Excellent (DNA, enzymes) | Easy fluorescence | Tunable via viscosity/gels | **Best overall** |
| Gas | Moderate (plasma, combustion) | Harder | Fast, hard to confine | Avoid for first demo |
| Powder / solid | Poor | Hard | Very slow | Avoid |

**Recommendation:** liquid phase, specifically aqueous DNA strand displacement in a microfluidic or Hele-Shaw geometry.

---

## 8. Target geometry: input/output in 1D space with diffusion

This is the right simplification. The governing PDE set is:

```
∂X_i/∂t = D_i ∇²X_i + β_i - δ_i X_i + X_i Σ_j α_{ij} Y_j
μ ∂Y_j/∂t = D_j ∇²Y_j + γ + a_j Y_j - Y_j²
a_j = θ_j + Σ_i ω_{ji} X_i
```

In 1D with operator splitting:

1. **Diffusion step** (implicit or spectral): `∂C/∂t = D ∂²C/∂x²` for each species over `Δt`.
2. **Reaction step** (pointwise at each grid point):
   - Solve Riccati for `Y_j` with `X` frozen (fast perceptron).
   - Update `X_i` with Patankar/MPE/BDF using the new `Y_j`.

Boundary conditions: no-flux (Neumann) for a closed chamber, or Dirichlet inputs at the left boundary for continuous driving.

---

## 9. What is still broken and what do we do about it?

The full RCN script (`Analysis/dack_rncrn_xor_riccati_free_inputs.py`) overflows during training even after adding saturating degradation and parameter clipping. Likely causes:

- Free `X1`, `X2` inputs decay too slowly, allowing runaway catalytic growth.
- The Riccati solve can still blow up if `a_j` becomes large and positive.
- Parameter clipping is applied after gradients, not inside the dynamics, so the optimiser explores unstable regions.

**Options to fix:**
1. **Hard ceiling:** clamp all concentrations to a maximum physical value (e.g. 10 µM) inside the simulation.
2. **Log-transform:** optimise `log(rate)` and `log(concentration)` so positivity and scale are built in.
3. **Deprioritise:** move to the 1D reaction-diffusion test case, which is more physically meaningful and easier to stabilise.

**Recommendation:** proceed with Option A (1D RD bistability front) while keeping the ODE RCN as a secondary debugging task.

---

## 10. Summary roadmap

| Step | Action | Deliverable |
|------|--------|-------------|
| 1 | Write 1D reaction-diffusion solver with operator splitting | `Analysis/rd_rcn_1d.py` |
| 2 | Choose concrete DNA reactions and map parameters to toehold rates | Note: `Notes/dna_reaction_scheme.md` |
| 3 | Implement MPE or BDF reaction step and verify positivity/conservation | Unit test in `Analysis/tests/` |
| 4 | Demonstrate bistability front / XOR-like spatial response | Figure + short note |
| 5 | Design microfluidic/Hele-Shaw geometry and fluorescence readout | Sketch + OpenFOAM or CAD file in `Design/` |
| 6 | (Parallel) stabilise ODE RCN training with log transform or hard ceiling | `Analysis/dack_rncrn_xor_stable.py` |

---

## Open questions for next meeting

1. Do we commit to DNA strand displacement, or should we also keep a small-molecule RD fallback for faster experiments?
2. Is the 1D bistability front an acceptable thesis demonstration, or do we need a full XOR stripe?
3. Should we build the fluorescence chamber in-house, or rely on existing microfluidics facilities?
