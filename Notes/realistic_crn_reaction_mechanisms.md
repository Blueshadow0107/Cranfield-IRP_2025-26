# Realistic reaction mechanisms for CRN neural networks

Date: 2026-06-18
Purpose: Ground the abstract RCN model in real DNA and enzymatic chemistry.

---

## 1. DNA strand displacement: toehold exchange (Zhang & Winfree 2009)

### Elementary three-step model

For an invading strand `X(m,n)` displacing incumbent `Y` from substrate `S`:

```
X(m,n) + S  --kf1-->  I(m,n)        (toehold binding)
I(m,n)      --kr-->   X(m,n) + S    (toehold dissociation)
I(m,n)      --kb-->   J(m,n)        (branch migration)
J(m,n)      --kr(δm)--> Y + L(m,n)  (incumbent toehold release)
```

The reverse reaction is also possible via the incumbent toehold `δm`.

### Effective bimolecular rate

At low concentration, the whole process can be approximated as an effective bimolecular reaction:

```
X + S --keff--> Y + L
```

The effective rate `keff` depends exponentially on toehold binding free energy:

```
keff ≈ k_max / (1 + exp(ΔG_toehold / RT) * C)
```

or, in the irreversible limit:

```
keff ∝ exp(-ΔG_toehold / RT)
```

Key facts:
- `k_max` for toehold-mediated strand displacement is ~10^6–10^7 M^-1 s^-1.
- Each additional base pair in the toehold changes the rate by roughly a factor of 10.
- Sequence composition (A/T vs G/C) changes `ΔG_toehold` and therefore `keff`.
- The reaction is reversible unless the incumbent toehold is very short or thermodynamically weak.

Reference: [Zhang & Winfree, JACS 2009](https://www.dna.caltech.edu/Papers/Toehold_Exchange2009.pdf)

---

## 2. DNA seesaw neural networks (Qian & Winfree 2011)

### Three basic reactions

**Seesawing** (reversible strand displacement, implements weighted sum):
```
Input_i + Gate_i:j  <-->  Input_i:Gate_i + Output_j
```

**Thresholding** (irreversible, fast, implements bias/threshold):
```
Input_i + Threshold_i  -->  waste
```

**Reporting** (readout):
```
Output_j + Reporter_j  -->  Fluor_j
```

### How weights and thresholds map to concentrations

| Neural-network concept | DNA implementation |
|------------------------|--------------------|
| Weight w_ij | Initial concentration of signal strand connecting gate i to gate j |
| Bias / threshold | Initial concentration of threshold complex that consumes input |
| Neuron activation | Free output signal concentration after thresholding |
| Fan-in / fan-out | Multiple signal strands competing at a gate |

### Simulation model

In Qian & Winfree's own simulations, the seesaw system was modelled as a mass-action CRN with:
- Forward/reverse seesaw rate constants.
- Faster thresholding rate (longer toehold).
- Reporter reaction treated as fast and irreversible.

All reactions are at the domain level; sequences are chosen to minimise secondary structure and crosstalk.

Reference: [Qian, Winfree & Bruck, Nature 2011](http://www.qianlab.caltech.edu/nature10262-s1.pdf)

---

## 3. DNA winner-take-all networks (Cherry & Qian 2018)

### Indexed CRN (from Lee, Buse & Winfree 2025)

```
X_i + W_ij + XF_i  --α1_ij-->  X_i + P_ij        (weighted production)
P_ij + SG_j        --α2_ij-->  S_j                (signal integration)
S_j + S_k + Anh_jk --α3_jk-->  ∅                  (pairwise annihilation)
S_j + RG_j + YF_j  --α4_j -->  S_j + Y_j         (output production)
Y_j + Rep_j        --α5_j -->  Fluor_j           (fluorescent readout)
```

Key features:
- Weights `α1_ij` are set by concentrations of weight strands `W_ij`.
- Pairwise annihilation `S_j + S_k -> ∅` implements winner-take-all competition.
- The output `Y_j` is high only for the "winning" class.

Reference: [Lee, Buse & Winfree, DNA 31](https://www.dna.caltech.edu/Papers/Lee-Buse-Winfree-2025-ICRNs.pdf)

---

## 4. Enzymatic neural networks (Okumura et al. 2022)

### PEN DNA toolbox

Uses three enzymes:
- **Polymerase** extends a primer bound to a template.
- **Nickase** nicks the extended strand.
- **Exonuclease** degrades single-stranded DNA.

### Core chemical neuron

For one input `X1` producing signal `α`:

```
X1 + cT_α  --polymerase-->  α + cT_α      (converter template)
α + aT_α   --polymerase/nickase-->  2α + aT_α   (autocatalytic amplification)
α + dT_α   --polymerase-->  dT_α:α_waste    (drain / deactivation)
α          --exonuclease-->  ∅              (degradation)
```

where:
- `cT_α` = converter template (sets input weight).
- `aT_α` = autocatalytic template (creates threshold nonlinearity).
- `dT_α` = drain template (sets bias / threshold).

### Effective dynamics

The signal `α` follows approximately:

```
dα/dt = w1 * X1 + k_amp * α^2 / (K_M + α) - k_drain * α / (K_M' + α) - k_deg * α
```

Key features:
- **Weighted sum**: production rate of `α` is proportional to input concentration and converter-template concentration.
- **Threshold / bistability**: the interplay of autocatalytic production (saturable) and drain removal (saturable) plus linear exonuclease degradation creates an S-shaped nullcline with two stable fixed points.
- **Bias**: controlled by drain-template concentration.

### Timescales

- Polymerase/nickase turnover: minutes.
- Exonuclease degradation: sets overall lifetime of signals.
- Typical droplet experiments: 1–6 hours.

Reference: [Okumura et al., Nature 2022](https://www.nature.com/articles/s41586-022-05218-7)

---

## 5. Implications for our abstract RCN model

### What our abstract equations miss

| Abstract term | Real chemistry | Correction needed |
|---------------|----------------|-------------------|
| `X_i * Y_j` catalysis | Toehold binding + branch migration | Effective bimolecular rate, possibly reversible |
| `Y_j^2` saturation | Dimerization / annihilation / deactivation | Explicit annihilation or Michaelis-Menten sink |
| `beta - delta X` production/degradation | Exonuclease degradation + fuel/template saturation | Saturation terms, enzyme conservation |
| Free weight parameters | Concentrations of DNA strands / templates | Parameters are constrained by total strand budget |

### Recommended effective-rate upgrade (Option B)

Keep the executive/perceptron structure but replace the mass-action terms with:

**For DNA strand displacement:**
```
rate = k_eff * [signal] * [gate]
k_eff = k_max / (1 + exp(ΔG_toehold / RT) * f(branch_migration))
```

**For enzymatic PEN toolbox:**
```
production = k_cat * [E_total] * [substrate] / (K_M + [substrate])
removal    = k_drain * [E_total] * [α] / (K_drain + [α]) + k_deg * [α]
```

This keeps the model computationally tractable while making the parameters physically interpretable.

---

## 6. Next steps

1. Pick a platform: DNA seesaw (Qian/Winfree) or enzymatic PEN (Okumura/Rondelez).
2. Write an effective-rate version of the RCN ODEs using the formulas above.
3. Constrain learned parameters by physical limits (total strand concentration, enzyme concentration, toehold energy range).
4. Train the constrained model and compare stability to the current abstract model.
5. Add 1D diffusion once the well-mixed effective-rate model is stable.
