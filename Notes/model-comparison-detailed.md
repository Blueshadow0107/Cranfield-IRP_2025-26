# Detailed model comparison for porous acoustic classifier

Date: 2026-06-14
Status: decision aid

This note compares the three candidate models in enough detail to pick one.
No derivations are included — just equations, variables, validity, and what you need to implement each.

---

## Model 1: Simplified time-domain Zwikker-Kosten

### Governing equations

Primary variables: pressure `p(x,y,t)` and bulk velocity `V(x,y,t)`.

```
(rho0 * alpha_inf / phi) * dV/dt + sigma * V = -grad(p)
dp/dt + rho0 * c0^2 * div(V) = 0
```

### Parameters

| Symbol | Meaning | How chosen |
|--------|---------|------------|
| `rho0` | air density | fixed, ~1.225 kg/m³ |
| `c0` | sound speed in air | fixed, ~343 m/s |
| `alpha_inf` | tortuosity | fixed at 1.0 for straight pores |
| `phi(x,y)` | porosity | design variable, [0.3, 1.0] |
| `sigma(x,y)` | flow resistivity | design variable, [0, 1e5] Pa·s/m² |

### What you need

- A 2D staggered-grid FDTD solver.
- First-order absorbing boundary conditions.
- A pressure source that injects the vowel waveform.
- Time-integrated squared pressure at probes.
- An optimiser that updates `phi` and `sigma` blocks.

### Validity

- Rigid frame.
- Pore radius much smaller than viscous skin depth.
- At 4 kHz, skin depth ~35 µm. Target pore radius 10–20 µm is borderline.
- `sigma` treated as constant with frequency (major simplification).

### Output

Direct time-domain probe signals. Easy to match Hughes readout style.

### Pros

- Fast.
- Intuitive time dynamics.
- Direct Hughes-style readout.

### Cons

- Constant `sigma` is physically wrong for 10–20 µm pores.
- Ambiguity about exact form and `phi` placement, as already discussed.
- Harder to defend as "physical".

---

## Model 2: Frequency-domain simplified ZK Helmholtz

### Governing equation

Primary variable: pressure `p(x,y,omega)`.

```
div( (1 / rho_eff(omega)) * grad(p) ) + (omega^2 / K_eff(omega)) * p = 0
```

with

```
rho_eff(omega) = (rho0 * alpha_inf / phi) * (1 + sigma*phi / (i*omega*rho0*alpha_inf))
K_eff(omega) = K0 / phi
K0 = rho0 * c0^2
```

Velocity, if needed:

```
V = (1 / (i * omega * rho_eff(omega))) * grad(p)
```

### Parameters

| Symbol | Meaning | How chosen |
|--------|---------|------------|
| `rho0` | air density | fixed |
| `c0` | sound speed in air | fixed |
| `alpha_inf` | tortuosity | fixed at 1.0 |
| `phi(x,y)` | porosity | design variable |
| `sigma(x,y)` | flow resistivity | design variable |
| `omega` | angular frequency | solve per frequency |

### What you need

- A 2D Helmholtz solver for each frequency.
- Sparse linear system solver (e.g. scipy.sparse.linalg.spsolve).
- FFT of vowel input to get frequency components.
- IFFT to reconstruct time-domain probe signals.
- Time integration of squared pressure.
- Optimiser over `phi` and `sigma`.

### Validity

- Rigid frame.
- Pores much smaller than acoustic wavelength (~86 mm at 4 kHz, easily satisfied).
- Pore radius comparable to or smaller than viscous skin depth is OK because `rho_eff(omega)` captures the frequency-dependent damping.
- Isothermal approximation (ignores thermal losses).

### Output

Frequency response per frequency, reconstructed time-domain signal via IFFT.

### Pros

- Physically standard and defensible.
- Frequency-dependent damping built in.
- No ambiguity about `phi` placement.
- Fast per frequency; easy to parallelise on HPC.

### Cons

- Time signal is reconstructed, not directly simulated.
- Need enough frequency samples for clean IFFT.
- Less intuitive than time-domain FDTD.

---

## Model 3: Full ZK or Johnson-Champoux-Allard (JCA)

### Governing equation

Same Helmholtz structure:

```
div( (1 / rho_eff(omega)) * grad(p) ) + (omega^2 / K_eff(omega)) * p = 0
```

but `rho_eff` and `K_eff` include both viscous and thermal boundary-layer effects, usually via Bessel functions.

### Full ZK form

```
rho_eff(omega) = (rho0 * alpha_inf / phi) * G_rho(omega)
K_eff(omega) = (K0 / phi) * G_K(omega)
```

where `G_rho` and `G_K` involve the Womersley number and Bessel functions of the viscous and thermal penetration depths.

### JCA form

```
rho_eff(omega) = (rho0 * alpha_inf / phi) * (1 + sigma*phi/(i*omega*rho0*alpha_inf) * sqrt(1 + i*4*alpha_inf^2*eta*rho0*omega/(sigma^2*Lambda^2*phi^2)) )
K_eff(omega) = (K0 / phi) / (1 + (gamma-1) / sqrt(1 + i*... thermal terms ...))
```

### Parameters

| Symbol | Meaning | How chosen |
|--------|---------|------------|
| `rho0`, `c0` | air properties | fixed |
| `alpha_inf` | tortuosity | fixed or linked to phi |
| `phi(x,y)` | porosity | design variable |
| `sigma(x,y)` | flow resistivity | design variable |
| `Lambda` | viscous characteristic length | derived from pore radius |
| `Lambda'` | thermal characteristic length | derived from pore radius |
| `gamma`, `Pr`, `eta` | gas properties | fixed |

### What you need

- Same Helmholtz solver as Model 2.
- Evaluation of complex `rho_eff(omega)` and `K_eff(omega)` at each frequency.
- A pore-radius model that links `phi`, `sigma`, `Lambda`, and `Lambda'`.

### Validity

- Rigid frame.
- Pores much smaller than wavelength.
- More accurate than Model 2 because it includes thermal losses.
- Requires realistic pore geometry to estimate `Lambda` and `Lambda'`.

### Output

Same as Model 2: frequency response, reconstructed time-domain signal.

### Pros

- Most accurate equivalent-fluid model.
- Can validate against impedance-tube measurements.

### Cons

- More parameters.
- Need to justify pore-radius / characteristic-length estimates.
- More implementation time.

---

## Quick comparison

| Aspect | Model 1 (time-domain ZK) | Model 2 (freq-domain ZK) | Model 3 (full ZK/JCA) |
|---|---|---|---|
| Physical fidelity | Low | Medium | High |
| Implementation time | Days | Days to 1 week | 1–2 weeks |
| Compute cost | Low | Medium (parallelisable) | Medium |
| Frequency-dependent damping | No | Yes | Yes |
| Thermal losses | No | No | Yes |
| Pore-size validity | Borderline for 10–20 µm | Good | Good |
| Design variables | phi, sigma | phi, sigma | phi, sigma (plus linked pore radius) |
| Supervisor defensibility | Weak | Strong | Strongest |

---

## Recommendation

For a 2.5-month MSc with HPC available:

- **Primary model:** Model 2, frequency-domain simplified ZK Helmholtz.
- **Optional sanity check:** Model 1, simplified time-domain ZK, run on a small test case.
- **Future work / validation:** Model 3, full JCA, if time permits.

This gives physical defensibility without the extra parameter burden of JCA.
