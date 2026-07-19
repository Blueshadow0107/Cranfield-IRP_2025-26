:
# Experimental parameter control for CRN hardware

## Learned parameters and their physical meaning

Our RCN has four kinds of learned parameters. Each is a reaction rate or an
offset in the mass-action equations.

| Parameter | Equation | Physical meaning | Typical implementation |
|-----------|----------|------------------|------------------------|
| `ω_{ji}` | `a_j = θ_j + Σ_i ω_{ji} X_i` | Weight from executive `X_i` to perceptron `Y_j` | Bimolecular rate constant for `X_i + Y_j` catalysis |
| `θ_j` | `a_j = θ_j + Σ_i ω_{ji} X_i` | Bias of perceptron `Y_j` | Offset rate (equivalent to a constant catalytic species) |
| `α_{ij}` | `dX_i/dt = ... + X_i Σ_j α_{ij} Y_j` | Weight from perceptron `Y_j` to executive `X_i` | Bimolecular rate constant for `Y_j + X_i` catalysis |
| `β_out` | `dX_3/dt = β_out - δ_3 X_3 ...` | Basal production of output `X_3` | Inflow rate or precursor concentration |

Fixed parameters:

| Parameter | Value | Physical meaning | Implementation |
|-----------|-------|------------------|----------------|
| `γ` | 0.05 | Perceptron basal production | Fuel/leak rate |
| `μ` | 0.02 | Perceptron/executive timescale ratio | Relative catalyst concentrations |
| `δ` | `[0.05, 0.05, 1.0]` | Executive degradation rates | Degradase concentration or dilution |
| `κ` | 1.0 | Output saturating degradation | Annihilation/sink reaction rate |

## DNA strand displacement (most mature)

### How each parameter is tuned

| Parameter | DNA-level knob | Typical range |
|-----------|----------------|---------------|
| `ω_{ji}`, `α_{ij}` | Toehold length | ~1 nt → ~10⁶-fold rate change |
| | Toehold GC content | weaker/stronger binding |
| | Mismatch position | ~10³-fold slowdown |
| | Remote toehold / spacer | 10–10³-fold slowdown |
| `θ_j` | Constant-template concentration or biased fuel | offset rate |
| `β_out` | Fuel / precursor concentration | continuous |
| `δ_i` | Degradase / sink-strand concentration | continuous |
| `μ` | Relative catalyst or toehold strengths for perceptron vs executive reactions | ratio tuning |
| `γ` | Leak / fuel background | continuous |
| `κ` | Sink-strand concentration for `X_3` | continuous |

### Timescales

- Toehold-mediated strand displacement rate saturates at ~10⁶–10⁷ M⁻¹s⁻¹ for 7–8 nt toeholds.
- Typical concentrations: 10 nM – 1 μM.
- Effective reaction timescales: seconds to hours.
- Perceptron equilibration can be made fast (seconds) by strong catalysts.

### Key references

- Zhang & Winfree, JACS 2009: toehold length vs rate
- Genot et al., J. R. Soc. Interface 2013: remote toeholds
- Qian, Winfree & Bruck, Nature 2011: DNA neural network
- Cherry & Qian, Nature 2018: winner-take-all DNA networks

## Enzymatic circuits

### How each parameter is tuned

| Parameter | Enzymatic knob |
|-----------|----------------|
| `ω_{ji}`, `α_{ij}` | DNA template concentration, gate design |
| `θ_j` | Drain/template ratio (Okumura et al. 2022) |
| `β_out` | Enzyme/substrate concentration |
| `δ_i` | Degradase or antitarget concentration |
| `μ` | Relative enzyme/template concentrations |

### Timescales

- Enzymatic turnover: ~1–10³ s⁻¹ per enzyme.
- With typical concentrations, network dynamics: minutes to hours.
- Faster than enzyme-free DNA for amplification, but requires protein handling.

### Key reference

- Okumura et al., Nature 2022: enzymatic neural networks in microdroplets

## Simpler chemical alternatives

### 1. Small-molecule reaction-diffusion systems

Examples: Belousov-Zhabotinsky (BZ), Oregonator, Gray-Scott, chlorine dioxide-iodine-malonic acid (CDIMA).

| Pros | Cons |
|------|------|
| No macromolecular synthesis | Fewer orthogonal reactions |
| Fast visual readout (color change) | Hard to tune individual rate constants independently |
| Well-studied pattern formation | Reactions are autocatalytic/oscillatory, not easily trainable |
| Cheap reagents | Parameter space is narrow before oscillations/chaos |

### Timescales

- BZ reaction: oscillation period ~1 s to minutes depending on concentrations.
- Diffusion-limited pattern formation: seconds to minutes in thin films.

### Mapping to our parameters

| Our parameter | Small-molecule equivalent | Tunability |
|---------------|---------------------------|------------|
| `ω_{ji}`, `α_{ij}` | Rate constants of bimolecular steps | Moderate (temperature, pH, catalysts) |
| `θ_j`, `β_out`, `γ` | Precursor/inflow concentrations | Good |
| `δ_i` | Degradation/dilution | Moderate |
| `μ` | Relative catalyst concentrations | Limited |

### Why it is harder

Small-molecule networks usually have a small number of reactions with strongly coupled parameters. You cannot easily assign one independent rate constant to each learned weight as you can with DNA toeholds.

### 2. Porous-media flow chemistry

A more CFD-relevant platform: pump reagents through a porous matrix or microfluidic chip.

| Pros | Cons |
|------|------|
| Direct spatial parallelism | Hard to make reactions fast enough |
| Matches your CFD background | Mixing limits reaction rates |
| Easy visualization | Individual rate tuning is coarse |

### Timescales

- Flow residence time: seconds to minutes.
- Reaction rates: set by reagent concentrations and temperature.

### 3. Belousov-Zhabotinsky / excitable media computing

Used historically for collision-based and reaction-diffusion computing.

| Pros | Cons |
|------|------|
| True spatial analog computing | Not trainable in the Dack sense |
| Beautiful wave patterns | Parameters are global (temperature, concentrations) |
| Demonstrated logic gates | No per-weight control |

## Can we use simpler chemistry for the Dack RCN?

Short answer: **yes in principle, but with much less independent control over each learned parameter.**

The Dack framework assumes you can set each `α_{ij}` and `ω_{ji}` independently. DNA strand displacement is the natural choice because each weight can be a separate strand-displacement reaction with its own toehold. In small-molecule chemistry, reactions are not so modular — changing one rate usually changes others.

### Best simpler compromise

A **porous-media flow reactor** with a few coupled reactions could demonstrate spatial analog computation, but the "training" would be coarse (tuning global concentrations and flow rates) rather than optimizing hundreds of independent weights.

For your thesis, a realistic framing is:

1. **Simulation/theory**: Dack-style trainable RCN with many independent parameters.
2. **Physical realization**: DNA strand displacement as the most plausible near-term platform.
3. **Simpler demonstration**: BZ or flow chemistry as a spatial analog-computing benchmark, not a trainable RCN.

## Recommended note for thesis

When discussing experimental realization, emphasize that:
- DNA strand displacement provides a direct mapping from trained rate constants to physical sequences.
- Each learned weight `α_{ij}` or `ω_{ji}` corresponds to a designed toehold or spacer.
- Simpler chemistries can show spatial analog computing, but they do not offer the same parametric programmability.
